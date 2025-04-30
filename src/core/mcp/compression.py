import zlib
import json
import time
from typing import Dict, Any
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager

logger = get_logger(__name__)

def optimize_context_data(context: ContextProtocol) -> ContextProtocol:
    try:
        optimized_context = context.optimize()
        if optimized_context is not context:
            logger.debug(f'Applied context-specific optimization for {type(context).__name__}')
            context = optimized_context
        elif hasattr(context, '_optimization_applied_inplace'):
            logger.debug(f'Applied in-place context optimization for {type(context).__name__}')
        else:
            logger.debug(f'Context-specific optimization for {type(context).__name__} returned self or did not change the object.')
    except NotImplementedError:
        logger.debug(f'No specific optimize method implemented for {type(context).__name__}')
    except Exception as e:
        logger.warning(f'Error during context-specific optimization for {type(context).__name__}: {e}')
    return context

def compress_context(context_data: Dict[str, Any]) -> bytes:
    try:
        if context_data is None:
            raise TypeError("Cannot compress None data")
        
        start_time = time.time()
        json_data = json.dumps(context_data, sort_keys=True, default=str).encode('utf-8')
        original_size = len(json_data)
        
        # Choose compression level based on data size
        if original_size < 1024:  # Small context
            compression_level = 1  # Fastest, less compression
        elif original_size < 1024 * 10:  # Medium context
            compression_level = 6  # Default
        else:  # Large context
            compression_level = 9  # Maximum compression
            
        compressed = zlib.compress(json_data, level=compression_level)
        compressed_size = len(compressed)
        
        # Record compression metrics
        get_metrics_manager().track_memory(
            'operations',
            operation_type='compress_context'
        )
        
        # Record compression time
        compression_time = time.time() - start_time
        get_metrics_manager().track_memory(
            'duration', 
            operation_type='compress_context', 
            value=compression_time
        )
        
        # Record compression ratio if enabled
        if hasattr(get_metrics_manager(), 'track_mcp'):
            compression_ratio = compressed_size / original_size
            get_metrics_manager().track_mcp(
                'compression_ratio',
                value=compression_ratio
            )
        
        logger.debug(f'Compressed context from {original_size} to {compressed_size} bytes (ratio: {compressed_size/original_size:.2f})')
        
        # Only use compression if it actually reduces size
        if compressed_size < original_size:
            return compressed
        else:
            logger.debug(f'Compression ineffective, using original data')
            return json_data
        
    except TypeError as e:
        # Log but re-raise TypeError for tests to catch
        logger.error(f'Failed to compress context data: {e}')
        raise  # Re-raise TypeError
            
    except Exception as e:
        logger.error(f'Failed to compress context data: {e}', exc_info=True)
        logger.warning('Compression failed. Returning original JSON bytes as fallback.')
        return json.dumps(context_data, sort_keys=True, default=str).encode('utf-8')

def decompress_context(compressed_data: bytes) -> Dict[str, Any]:
    try:
        decompressed_json: bytes = zlib.decompress(compressed_data)
        context_data: Dict[str, Any] = json.loads(decompressed_json.decode('utf-8'))
        return context_data
    except zlib.error as e:
        logger.error(f'Failed to decompress context data (zlib error): {e}', exc_info=True)
        raise ValueError('Failed to decompress context data (invalid zlib format?)') from e
    except json.JSONDecodeError as e:
        logger.error(f'Failed to decode JSON after decompression: {e}', exc_info=True)
        raise ValueError('Decompressed data is not valid JSON') from e
    except Exception as e:
        logger.error(f'Unexpected error during context decompression: {e}', exc_info=True)
        raise ValueError('Failed to decompress context') from e
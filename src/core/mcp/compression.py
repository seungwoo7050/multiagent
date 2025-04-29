import zlib
import json
from typing import Dict, Any
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.config.logger import get_logger
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
        json_data = json.dumps(context_data, sort_keys=True, default=str).encode('utf-8')
        original_size = len(json_data)
        compressed = zlib.compress(json_data)
        compressed_size = len(compressed)
        logger.debug(f'Compressed context data from {original_size} to {compressed_size} bytes (zlib level {zlib.Z_DEFAULT_COMPRESSION})')
        return compressed
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
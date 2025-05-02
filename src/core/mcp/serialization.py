
import importlib
import inspect
from typing import Any, Dict, Optional, Type

from src.config.logger import get_logger
from src.core.exceptions import SerializationError
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.utils.serialization import SerializationFormat
from src.utils.serialization import deserialize as general_deserialize
from src.utils.serialization import serialize as general_serialize

logger = get_logger(__name__)

def serialize_context(context: ContextProtocol, format: SerializationFormat=SerializationFormat.MSGPACK) -> bytes:
    try:
        if hasattr(context, 'serialize') and callable(context.serialize):
            data_dict: Dict[str, Any] = context.serialize()
            logger.debug(f"Using context's serialize() method for {type(context).__name__}")
        else:
            logger.debug(f'Using general serializer for {type(context).__name__}')
            data_dict = context
        return general_serialize(data_dict, format=format)
    except Exception as e:
        if isinstance(e, SerializationError):
            raise e
        logger.error(f'Failed to serialize context object (type: {type(context).__name__}): {e}', exc_info=True)
        raise SerializationError(f'Failed to serialize context: {e}', original_error=e)

def deserialize_context(data: bytes, target_class: Optional[Type[ContextProtocol]]=None, format: SerializationFormat=SerializationFormat.MSGPACK) -> ContextProtocol:
    """
    Deserialize binary data into a Context object.
    
    Args:
        data: The binary data to deserialize
        target_class: Optional target class for deserialization
        format: Serialization format (MSGPACK or JSON)
        
    Returns:
        Deserialized context object
        
    Raises:
        SerializationError: If deserialization fails
    """
    try:
        # First, deserialize the binary data using the general deserializer
        deserialized_data = general_deserialize(data, format=format)
        
        # Validate the deserialized data is a dictionary
        if not isinstance(deserialized_data, dict):
            raise SerializationError(f'Deserialized data is not a dictionary (got {type(deserialized_data).__name__})')
        
        # If we know the target class, use it directly
        if target_class is not None:
            logger.debug(f'Deserializing into specified target class: {target_class.__name__}')
            return _deserialize_with_target_class(deserialized_data, target_class)
        else:
            # Otherwise try to infer the class from the data
            logger.debug('Target class not provided. Attempting to infer type from data...')
            return _infer_and_deserialize(deserialized_data)
            
    except Exception as e:
        if isinstance(e, SerializationError):
            raise e
        logger.error(f'Failed to deserialize context data: {e}', exc_info=True)
        raise SerializationError(f'Failed to deserialize context: {e}', original_error=e)

def _deserialize_with_target_class(data: Dict[str, Any], target_class: Type[ContextProtocol]) -> ContextProtocol:
    """
    Deserialize data using the provided target class.
    
    Args:
        data: The dictionary data to deserialize
        target_class: The target class for deserialization
        
    Returns:
        Instantiated context object
        
    Raises:
        SerializationError: If deserialization fails
    """
    # Check if class has a proper deserialize classmethod
    if hasattr(target_class, 'deserialize') and callable(target_class.deserialize):
        if isinstance(getattr(target_class, 'deserialize'), classmethod) or inspect.isfunction(getattr(target_class, 'deserialize')):
            logger.debug(f'Using {target_class.__name__}.deserialize classmethod')
            return target_class.deserialize(data)
    
    # Fall back to Pydantic validation for BaseContextSchema subclasses
    if issubclass(target_class, BaseContextSchema):
        logger.debug(f'Using Pydantic validation for {target_class.__name__}')
        return target_class.model_validate(data)
    
    # Last resort: direct instantiation
    logger.warning(f'Target class {target_class.__name__} has no standard deserialize method. Attempting direct instantiation.')
    try:
        return target_class(**data)
    except Exception as init_e:
        raise SerializationError(
            f'Failed to initialize {target_class.__name__} from deserialized data', 
            original_error=init_e
        )

def _infer_and_deserialize(data: Dict[str, Any]) -> ContextProtocol:
    """
    Infer the target class from the data and deserialize.
    
    Args:
        data: The dictionary data to deserialize
        
    Returns:
        Instantiated context object
        
    Raises:
        SerializationError: If deserialization fails
    """
    # Look for type hints in the data
    context_type_hint = data.get('__type__')
    class_path_hint = data.get('class')
    
    if context_type_hint == 'model' and class_path_hint:
        logger.debug(f'Found model hint: {class_path_hint}')
        try:
            # Import the class dynamically
            module_path, class_name = class_path_hint.rsplit('.', 1)
            module = importlib.import_module(module_path)
            inferred_class = getattr(module, class_name)
            
            # Verify it's a ContextProtocol
            if issubclass(inferred_class, ContextProtocol):
                logger.debug(f'Successfully inferred class: {inferred_class.__name__}')
                return _deserialize_with_target_class(data, inferred_class)
            else:
                raise SerializationError(f'Inferred class {class_path_hint} is not a ContextProtocol subclass')
        except (ValueError, ImportError, AttributeError) as e:
            raise SerializationError(f"Failed to import class from hint '{class_path_hint}'", original_error=e)
        except Exception as e:
            raise SerializationError(f"Unexpected error inferring class from hint '{class_path_hint}'", original_error=e)
    
    # Check for version field to detect contexts without type hints
    if 'version' in data and isinstance(data.get('version'), str):
        logger.debug('Found version field, attempting to deserialize as BaseContextSchema')
        try:
            return BaseContextSchema.model_validate(data)
        except Exception as e:
            logger.warning(f'Failed to deserialize as BaseContextSchema: {e}')
            # Continue to generic fallback
    
    # Fall back to BaseContextSchema as last resort
    logger.warning('Could not infer target class from data. Using BaseContextSchema as fallback.')
    try:
        return BaseContextSchema.model_validate(data)
    except Exception as base_e:
        raise SerializationError(
            'Failed to deserialize as BaseContextSchema (target class unknown)', 
            original_error=base_e
        )
    


from typing import Any, Dict, Type, Optional
import json
import msgpack
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.utils.serialization import serialize as general_serialize, deserialize as general_deserialize, SerializationFormat, SerializationError
from src.config.logger import get_logger
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
    try:
        deserialized_data: Any = general_deserialize(data, format=format)
        if not isinstance(deserialized_data, dict):
            raise SerializationError(f'Deserialized data is not a dictionary (got {type(deserialized_data).__name__})')
        if target_class:
            logger.debug(f'Attempting to deserialize into specified target class: {target_class.__name__}')
            if hasattr(target_class, 'deserialize') and callable(target_class.deserialize):
                if isinstance(getattr(target_class, 'deserialize'), classmethod) or inspect.isfunction(getattr(target_class, 'deserialize')):
                    return target_class.deserialize(deserialized_data)
                else:
                    logger.warning(f'Deserialize method on {target_class.__name__} is not a classmethod. Cannot call directly.')
                    if issubclass(target_class, BaseContextSchema):
                        logger.debug(f'Falling back to Pydantic validation for {target_class.__name__}.')
                        return target_class.model_validate(deserialized_data)
                    else:
                        raise SerializationError(f'Cannot call instance method deserialize without an instance for {target_class.__name__}.')
            elif issubclass(target_class, BaseContextSchema):
                return target_class.model_validate(deserialized_data)
            else:
                logger.warning(f'Target class {target_class.__name__} has no standard deserialize method. Attempting direct instantiation.')
                try:
                    return target_class(**deserialized_data)
                except Exception as init_e:
                    raise SerializationError(f'Failed to initialize {target_class.__name__} from deserialized data', original_error=init_e)
        else:
            logger.debug('Target class not provided. Attempting to infer type from data...')
            context_type_hint = deserialized_data.get('__type__')
            class_path_hint = deserialized_data.get('class')
            if context_type_hint == 'model' and class_path_hint:
                logger.debug(f'Found model hint: {class_path_hint}')
                try:
                    module_path, class_name = class_path_hint.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    inferred_class = getattr(module, class_name)
                    if issubclass(inferred_class, ContextProtocol):
                        logger.debug(f'Successfully inferred and imported class: {inferred_class.__name__}')
                        if hasattr(inferred_class, 'deserialize') and callable(inferred_class.deserialize):
                            return inferred_class.deserialize(deserialized_data)
                        elif issubclass(inferred_class, BaseContextSchema):
                            return inferred_class.model_validate(deserialized_data)
                        else:
                            raise SerializationError('Inferred class does not have a suitable deserialization method.')
                    else:
                        raise SerializationError(f'Inferred class {class_path_hint} is not a ContextProtocol subclass.')
                except Exception as import_e:
                    raise SerializationError(f"Failed to infer or import target class from hint '{class_path_hint}'", original_error=import_e)
            else:
                logger.warning('Could not infer target class from data. Attempting BaseContextSchema.')
                try:
                    return BaseContextSchema.model_validate(deserialized_data)
                except Exception as base_e:
                    raise SerializationError('Failed to deserialize data as BaseContextSchema (target class unknown)', original_error=base_e)
    except Exception as e:
        if isinstance(e, SerializationError):
            raise e
        logger.error(f'Failed to deserialize context data: {e}', exc_info=True)
        raise SerializationError(f'Failed to deserialize context: {e}', original_error=e)
import inspect
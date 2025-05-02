import dataclasses
import datetime
import importlib
import json
import uuid
from enum import Enum
from typing import Any, Dict, Optional, Type

import msgpack
from pydantic import BaseModel

from src.config.logger import get_logger
from src.core.exceptions import SerializationError

logger = get_logger(__name__)
_ENUM_REGISTRY: Dict[str, Type[Enum]] = {}

# Determine Pydantic version once at import time
_PYDANTIC_V2 = hasattr(BaseModel, "model_dump")

class SerializationFormat(str, Enum):
    JSON = 'json'
    MSGPACK = 'msgpack'

def _default_encoder(obj: Any) -> Any:
    """Enhanced encoder with support for more types"""
    if isinstance(obj, datetime.datetime):
        return {'__type__': 'datetime', 'value': obj.isoformat()}
    elif isinstance(obj, datetime.date):
        return {'__type__': 'date', 'value': obj.isoformat()}
    elif isinstance(obj, datetime.time):
        return {'__type__': 'time', 'value': obj.isoformat()}
    elif isinstance(obj, uuid.UUID):
        return {'__type__': 'uuid', 'value': str(obj)}
    elif isinstance(obj, set):
        return {'__type__': 'set', 'value': list(obj)}
    elif isinstance(obj, bytes):
        return {'__type__': 'bytes', 'value': obj.hex()}
    elif isinstance(obj, Enum):
        class_path = f'{obj.__class__.__module__}.{obj.__class__.__name__}'
        _ENUM_REGISTRY[class_path] = obj.__class__
        return {'__type__': 'enum', 'class': class_path, 'value': obj.value}
    elif isinstance(obj, BaseModel):
        return {'__type__': 'model', 
                'class': f'{obj.__class__.__module__}.{obj.__class__.__name__}', 
                'value': model_to_dict(obj)}
    elif dataclasses.is_dataclass(obj):
        return {'__type__': 'dataclass',
                'class': f'{obj.__class__.__module__}.{obj.__class__.__name__}',
                'value': dataclasses.asdict(obj)}
    # Support for numpy types
    elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
        return {'__type__': 'ndarray', 'value': obj.tolist()}
    elif hasattr(obj, '__dict__'):
        return {'__type__': 'object', 
                'class': f'{obj.__class__.__module__}.{obj.__class__.__name__}', 
                'value': obj.__dict__}
    raise TypeError(f'Object of type {type(obj)} is not serializable')

def _object_hook(obj: Dict[str, Any]) -> Any:
    if not isinstance(obj, dict) or '__type__' not in obj:
        return obj
    obj_type = obj['__type__']
    value = obj.get('value')
    if obj_type == 'datetime':
        if value is None:
            raise SerializationError('Missing value for datetime deserialization')
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError as e:
            raise SerializationError(f'Invalid datetime format: {value}', original_error=e)
    elif obj_type == 'date':
        if value is None:
            raise SerializationError('Missing value for date deserialization')
        try:
            return datetime.date.fromisoformat(value)
        except ValueError as e:
            raise SerializationError(f'Invalid date format: {value}', original_error=e)
    elif obj_type == 'time':
        if value is None:
            raise SerializationError('Missing value for time deserialization')
        try:
            return datetime.time.fromisoformat(value)
        except ValueError as e:
            raise SerializationError(f'Invalid time format: {value}', original_error=e)
    elif obj_type == 'uuid':
        if value is None:
            raise SerializationError('Missing value for UUID deserialization')
        try:
            return uuid.UUID(value)
        except ValueError as e:
            raise SerializationError(f'Invalid UUID format: {value}', original_error=e)
    elif obj_type == 'set':
        if not isinstance(value, list):
            raise SerializationError(f'Expected list for set deserialization, got: {type(value)}')
        return set(value)
    elif obj_type == 'bytes':
        if value is None:
            raise SerializationError('Missing value for bytes deserialization')
        try:
            return bytes.fromhex(value)
        except ValueError as e:
            raise SerializationError(f'Invalid hex format for bytes: {value}', original_error=e)
    elif obj_type == 'enum':
        class_path = obj.get('class')
        enum_value = value
        if not class_path:
            raise SerializationError('Missing class path for enum deserialization')
        if enum_value is None:
            raise SerializationError('Missing value for enum deserialization')
        if class_path in _ENUM_REGISTRY:
            enum_class = _ENUM_REGISTRY[class_path]
            try:
                return enum_class(enum_value)
            except ValueError:
                logger.warning(f"Invalid value '{enum_value}' for enum {class_path}. Returning raw value.", exc_info=True)
                return enum_value
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            enum_class = getattr(module, class_name)
            if not issubclass(enum_class, Enum):
                raise TypeError(f'{class_path} is not an Enum')
            _ENUM_REGISTRY[class_path] = enum_class
            return enum_class(enum_value)
        except Exception:
            logger.error(f'Failed to deserialize enum: {class_path}, value: {enum_value}', exc_info=True)
            logger.warning(f"Returning raw value '{enum_value}' instead of enum instance for {class_path}")
            return enum_value
    elif obj_type == 'model':
        class_path_str = obj.get('class')
        model_data = value
        if not class_path_str:
            raise SerializationError('Missing class path for model deserialization')
        if not isinstance(model_data, dict):
            raise SerializationError(f'Expected dict for model data, got: {type(model_data)}')
        try:
            module_path, class_name = class_path_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            if issubclass(model_class, BaseModel):
                try:
                    return model_class.model_validate(model_data)
                except AttributeError:
                    return model_class.parse_obj(model_data)
            else:
                logger.warning(f'Class {class_path_str} is not a Pydantic model. Attempting generic instantiation.')
                return model_class(**model_data)
        except Exception as e:
            raise SerializationError(f'Failed to deserialize model: {class_path_str}', original_error=e)
    elif obj_type == 'object':
        class_path_str = obj.get('class')
        obj_data = value
        if not class_path_str:
            raise SerializationError('Missing class path for object deserialization')
        if not isinstance(obj_data, dict):
            raise SerializationError(f'Expected dict for object data, got: {type(obj_data)}')
        try:
            module_path, class_name = class_path_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls.__new__(cls)
            instance.__dict__.update(obj_data)
            return instance
        except Exception as e:
            raise SerializationError(f'Failed to deserialize object: {class_path_str}', original_error=e)
    logger.warning(f'Unknown serialized type encountered in object_hook: {obj_type}')
    return obj

def serialize(data: Any, format: SerializationFormat=SerializationFormat.MSGPACK, pretty: bool=False) -> bytes:
    try:
        if format == SerializationFormat.JSON:
            indent = 2 if pretty else None
            result = json.dumps(data, default=_default_encoder, ensure_ascii=False, indent=indent).encode('utf-8')
        elif format == SerializationFormat.MSGPACK:
            result = msgpack.packb(data, default=_default_encoder, use_bin_type=True)
        else:
            raise ValueError(f'Unsupported serialization format: {format}')
        return result
    except Exception as e:
        # Remove redundant logging - the exception will be logged where it's caught
        raise SerializationError(message=f'Failed to serialize data: {str(e)}', original_error=e)

def deserialize(data: bytes, format: SerializationFormat=SerializationFormat.MSGPACK, cls: Optional[Type]=None) -> Any:
    try:
        if format == SerializationFormat.JSON:
            result = json.loads(data.decode('utf-8'), object_hook=_object_hook)
        elif format == SerializationFormat.MSGPACK:
            result = msgpack.unpackb(data, object_hook=_object_hook, raw=False)
        else:
            raise ValueError(f'Unsupported serialization format: {format}')
        if cls and result is not None:
            if issubclass(cls, BaseModel):
                if _PYDANTIC_V2:
                    return cls.model_validate(result)
                else:
                    return cls.parse_obj(result)
            elif hasattr(cls, 'from_dict') and callable(getattr(cls, 'from_dict')):
                return cls.from_dict(result)
            else:
                try:
                    return cls(**result)
                except TypeError as te:
                    raise SerializationError(f'Failed to instantiate {cls.__name__} with deserialized data: {te}', original_error=te)
        return result
    except Exception as e:
        # Remove redundant logging
        raise SerializationError(message=f'Failed to deserialize data: {str(e)}', original_error=e)

def serialize_to_json(data: Any, pretty: bool=False) -> str:
    try:
        indent = 2 if pretty else None
        return json.dumps(data, default=_default_encoder, ensure_ascii=False, indent=indent)
    except Exception as e:
        # Remove redundant logging
        raise SerializationError(message=f'Failed to serialize to JSON: {str(e)}', original_error=e)

def deserialize_from_json(data: str, cls: Optional[Type]=None) -> Any:
    try:
        result = json.loads(data, object_hook=_object_hook)
        if cls and result is not None:
            if issubclass(cls, BaseModel):
                if _PYDANTIC_V2:
                    return cls.model_validate(result)
                else:
                    return cls.parse_obj(result)
            elif hasattr(cls, 'from_dict') and callable(getattr(cls, 'from_dict')):
                return cls.from_dict(result)
            else:
                try:
                    return cls(**result)
                except TypeError as te:
                    raise SerializationError(f'Failed to instantiate {cls.__name__} from JSON: {te}', original_error=te)
        return result
    except Exception as e:
        # Remove redundant logging
        raise SerializationError(message=f'Failed to deserialize from JSON: {str(e)}', original_error=e)

def model_to_dict(model: BaseModel, exclude_none: bool=False) -> Dict[str, Any]:
    """Convert Pydantic model to dict with version compatibility"""
    if _PYDANTIC_V2:
        return model.model_dump(exclude_none=exclude_none)
    else:
        return model.dict(exclude_none=exclude_none)

def model_to_json(model: BaseModel, pretty: bool=False, exclude_none: bool=False) -> str:
    """Convert Pydantic model to JSON with version compatibility"""
    indent = 2 if pretty else None
    if _PYDANTIC_V2:
        return model.model_dump_json(indent=indent, exclude_none=exclude_none)
    else:
        model_dict = model.dict(exclude_none=exclude_none)
        return json.dumps(model_dict, default=_default_encoder, ensure_ascii=False, indent=indent)
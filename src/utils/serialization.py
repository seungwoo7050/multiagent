import importlib
import datetime
import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import msgpack
from pydantic import BaseModel

from src.config.logger import get_logger
from src.core.exceptions import SerializationError

# Module logger
logger = get_logger(__name__)

# 추가: 열거형 레지스트리
_ENUM_REGISTRY = {}


class SerializationFormat(str, Enum):
    """Supported serialization formats."""
    JSON = "json"
    MSGPACK = "msgpack"


def _default_encoder(obj: Any) -> Any:
    """Default encoder for types that need special handling.
    
    Args:
        obj: The object to encode.
        
    Returns:
        Serializable version of the object.
        
    Raises:
        TypeError: If the object cannot be serialized.
    """
    if isinstance(obj, datetime.datetime):
        return {
            "__type__": "datetime",
            "value": obj.isoformat()
        }
    elif isinstance(obj, datetime.date):
        return {
            "__type__": "date",
            "value": obj.isoformat()
        }
    elif isinstance(obj, datetime.time):
        return {
            "__type__": "time",
            "value": obj.isoformat()
        }
    elif isinstance(obj, uuid.UUID):
        return {
            "__type__": "uuid",
            "value": str(obj)
        }
    elif isinstance(obj, set):
        return {
            "__type__": "set",
            "value": list(obj)
        }
    elif isinstance(obj, bytes):
        return {
            "__type__": "bytes",
            "value": obj.hex()
        }
    elif isinstance(obj, Enum):
        class_path = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
        # 등록: 열거형 클래스 레지스트리에 추가
        _ENUM_REGISTRY[class_path] = obj.__class__
        return {
            "__type__": "enum",
            "class": class_path,
            "value": obj.value
        }
    elif isinstance(obj, BaseModel):
        return {
            "__type__": "model",
            "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            "value": obj.dict()
        }
    elif hasattr(obj, "__dict__"):
        # Generic object serialization for simple classes
        return {
            "__type__": "object",
            "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            "value": obj.__dict__
        }
    # Default for unsupported types
    raise TypeError(f"Object of type {type(obj)} is not serializable")


def _object_hook(obj: Dict[str, Any]) -> Any:
    """Hook for deserializing custom types."""
    # Handle non-dict objects or regular dicts without __type__
    if not isinstance(obj, dict) or "__type__" not in obj:
        return obj

    obj_type = obj["__type__"]
    value = obj.get("value")

    # --- Type-specific deserialization ---
    if obj_type == "datetime":
        if value is None:
            raise SerializationError("Missing value for datetime deserialization")
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError as e:
            raise SerializationError(f"Invalid datetime format: {value}", original_error=e)

    elif obj_type == "date":
        if value is None:
            raise SerializationError("Missing value for date deserialization")
        try:
            return datetime.date.fromisoformat(value)
        except ValueError as e:
            raise SerializationError(f"Invalid date format: {value}", original_error=e)

    elif obj_type == "time":
        if value is None:
            raise SerializationError("Missing value for time deserialization")
        try:
            return datetime.time.fromisoformat(value)
        except ValueError as e:
            raise SerializationError(f"Invalid time format: {value}", original_error=e)

    elif obj_type == "uuid":
        if value is None:
            raise SerializationError("Missing value for UUID deserialization")
        try:
            return uuid.UUID(value)
        except ValueError as e:
            raise SerializationError(f"Invalid UUID format: {value}", original_error=e)

    elif obj_type == "set":
        if not isinstance(value, list):
            raise SerializationError(f"Expected list for set deserialization, got: {type(value)}")
        return set(value)

    elif obj_type == "bytes":
        if value is None:
            raise SerializationError("Missing value for bytes deserialization")
        try:
            return bytes.fromhex(value)
        except ValueError as e:
            raise SerializationError(f"Invalid hex format for bytes: {value}", original_error=e)

    elif obj_type == "enum":
        class_path = obj.get("class")
        enum_value = value
        
        if not class_path:
            raise SerializationError("Missing class path for enum deserialization")
        if enum_value is None:
            raise SerializationError("Missing value for enum deserialization")
        
        # 변경: 레지스트리에서 먼저 확인
        if class_path in _ENUM_REGISTRY:
            enum_class = _ENUM_REGISTRY[class_path]
            return enum_class(enum_value)
            
        # 동적 임포트 시도
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            enum_class = getattr(module, class_name)
            return enum_class(enum_value)
        except Exception as e:
            logger.error(f"Failed to deserialize enum: {class_path}, value: {enum_value}", exc_info=True)
            # 실패 시 값만 반환
            logger.warning(f"Returning string value instead of enum instance for {class_path}")
            return enum_value

    elif obj_type == "model":
        class_path_str = obj.get("class")
        model_data = value

        if not class_path_str:
            raise SerializationError("Missing class path for model deserialization")
        if not isinstance(model_data, dict):
            raise SerializationError(f"Expected dict for model data, got: {type(model_data)}")

        try:
            module_path, class_name = class_path_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Pydantic V2 compatibility
            if issubclass(model_class, BaseModel):
                try:
                    return model_class.model_validate(model_data)
                except AttributeError:
                    return model_class.parse_obj(model_data)
            else:
                return model_class(**model_data)
        except Exception as e:
            raise SerializationError(f"Failed to deserialize model: {class_path_str}", original_error=e)

    elif obj_type == "object":
        class_path_str = obj.get("class")
        obj_data = value

        if not class_path_str:
            raise SerializationError("Missing class path for object deserialization")
        if not isinstance(obj_data, dict):
            raise SerializationError(f"Expected dict for object data, got: {type(obj_data)}")

        try:
            module_path, class_name = class_path_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls.__new__(cls)
            instance.__dict__.update(obj_data)
            return instance
        except Exception as e:
            raise SerializationError(f"Failed to deserialize object: {class_path_str}", original_error=e)

    # Warning for unknown types
    logger.warning(f"Unknown serialized type: {obj_type}")
    return obj


def serialize(
    data: Any,
    format: SerializationFormat = SerializationFormat.MSGPACK,
    pretty: bool = False
) -> bytes:
    """Serialize data into the specified format.
    
    Args:
        data: Data to serialize.
        format: Serialization format to use.
        pretty: Whether to format JSON for readability (ignored for MessagePack).
        
    Returns:
        bytes: Serialized data.
        
    Raises:
        SerializationError: If serialization fails.
    """
    try:
        if format == SerializationFormat.JSON:
            indent = 2 if pretty else None
            result = json.dumps(
                data,
                default=_default_encoder,
                ensure_ascii=False,
                indent=indent
            ).encode('utf-8')
        elif format == SerializationFormat.MSGPACK:
            result = msgpack.packb(
                data,
                default=_default_encoder,
                use_bin_type=True
            )
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
        
        return result
    except Exception as e:
        logger.exception(f"Serialization error: {e}")
        raise SerializationError(
            message=f"Failed to serialize data: {str(e)}",
            original_error=e
        )


def deserialize(
    data: bytes,
    format: SerializationFormat = SerializationFormat.MSGPACK,
    cls: Optional[Type] = None
) -> Any:
    """Deserialize data from the specified format.
    
    Args:
        data: Serialized data.
        format: Serialization format to use.
        cls: Optional class to deserialize into.
        
    Returns:
        Any: Deserialized data.
        
    Raises:
        SerializationError: If deserialization fails.
    """
    try:
        if format == SerializationFormat.JSON:
            result = json.loads(
                data.decode('utf-8'),
                object_hook=_object_hook
            )
        elif format == SerializationFormat.MSGPACK:
            result = msgpack.unpackb(
                data,
                object_hook=_object_hook,
                raw=False
            )
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
        
        # Convert to specified class if provided
        if cls and result is not None:
            if issubclass(cls, BaseModel):
                return cls.parse_obj(result)
            elif hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
                return cls.from_dict(result)
            else:
                return cls(**result)
        
        return result
    except Exception as e:
        logger.exception(f"Deserialization error: {e}")
        raise SerializationError(
            message=f"Failed to deserialize data: {str(e)}",
            original_error=e
        )


def serialize_to_json(data: Any, pretty: bool = False) -> str:
    """Serialize data to a JSON string.
    
    Args:
        data: Data to serialize.
        pretty: Whether to format JSON for readability.
        
    Returns:
        str: JSON string.
        
    Raises:
        SerializationError: If serialization fails.
    """
    try:
        indent = 2 if pretty else None
        return json.dumps(
            data,
            default=_default_encoder,
            ensure_ascii=False,
            indent=indent
        )
    except Exception as e:
        logger.exception(f"JSON serialization error: {e}")
        raise SerializationError(
            message=f"Failed to serialize to JSON: {str(e)}",
            original_error=e
        )


def deserialize_from_json(data: str, cls: Optional[Type] = None) -> Any:
    """Deserialize data from a JSON string.
    
    Args:
        data: JSON string.
        cls: Optional class to deserialize into.
        
    Returns:
        Any: Deserialized data.
        
    Raises:
        SerializationError: If deserialization fails.
    """
    try:
        result = json.loads(data, object_hook=_object_hook)
        
        # Convert to specified class if provided
        if cls and result is not None:
            if issubclass(cls, BaseModel):
                return cls.parse_obj(result)
            elif hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
                return cls.from_dict(result)
            else:
                return cls(**result)
        
        return result
    except Exception as e:
        logger.exception(f"JSON deserialization error: {e}")
        raise SerializationError(
            message=f"Failed to deserialize from JSON: {str(e)}",
            original_error=e
        )


def model_to_dict(model: BaseModel, exclude_none: bool = False) -> Dict[str, Any]:
    """Convert a Pydantic model to a dictionary.
    
    Args:
        model: Pydantic model to convert.
        exclude_none: Whether to exclude None values.
        
    Returns:
        Dict[str, Any]: Dictionary representation of the model.
    """
    return model.dict(exclude_none=exclude_none)


def model_to_json(model: BaseModel, pretty: bool = False, exclude_none: bool = False) -> str:
    """Convert a Pydantic model to a JSON string.
    
    Args:
        model: Pydantic model to convert.
        pretty: Whether to format JSON for readability.
        exclude_none: Whether to exclude None values.
        
    Returns:
        str: JSON string.
    """
    indent = 2 if pretty else None
    # Pydantic V2 호환성을 위해 model_dump_json 사용
    return model.model_dump_json(indent=indent, exclude_none=exclude_none)
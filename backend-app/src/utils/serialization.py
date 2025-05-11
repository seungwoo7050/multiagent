                            
import dataclasses
import datetime
import json
import pickle
import uuid
import base64
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar
import msgspec
from src.utils.logger import get_logger
from src.config.errors import SystemError, ErrorCode

logger = get_logger(__name__)

try:
    from pydantic import BaseModel
    _HAS_PYDANTIC = True
except ImportError:
    BaseModel = type(None)
    _HAS_PYDANTIC = False

T = TypeVar('T')

class SerializationFormat(str, Enum):
    JSON = 'json'
    MSGPACK = 'msgpack'
    PICKLE = 'pickle'

def _json_encoder_hook(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('ascii')
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.time):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Enum):
        return obj.value
    if _HAS_PYDANTIC and isinstance(obj, BaseModel):
        dump_method = getattr(obj, 'model_dump', getattr(obj, 'dict', None))
        if dump_method:
            return dump_method(mode='json')
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
                                                       
    raise TypeError(f"Object of type {type(obj)} is not directly serializable to JSON by this hook.")

                            
def _json_decoder_hook(type_: Type, obj: Any) -> Any:
                                              
    if type_ is bytes and isinstance(obj, str):
        try:
            return base64.b64decode(obj)
        except Exception as e:
            logger.debug(f"Base64 decoding failed: {e}")
    return None
                      
def _msgpack_encoder_hook(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.time):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, Enum):
        return obj.value
    if _HAS_PYDANTIC and isinstance(obj, BaseModel):
        dump_method = getattr(obj, 'model_dump', getattr(obj, 'dict', None))
        if dump_method:
            return dump_method()
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj)} is not directly serializable to msgpack by this hook.")

def serialize(data: Any, format: SerializationFormat = SerializationFormat.MSGPACK, pretty: bool = False) -> bytes:
    """Serialize data to bytes using the specified format."""
    try:
        if format == SerializationFormat.MSGPACK:
            encoder = msgspec.msgpack.Encoder(enc_hook=_msgpack_encoder_hook)
            return encoder.encode(data)
        elif format == SerializationFormat.JSON:
            if pretty:
                return json.dumps(
                    data,
                    default=_json_encoder_hook,
                    ensure_ascii=False,
                    indent=2
                ).encode('utf-8')
            else:
                encoder = msgspec.json.Encoder(enc_hook=_json_encoder_hook)
                return encoder.encode(data)
        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(data)
        else:
            raise ValueError(f'Unsupported serialization format: {format}')
    except Exception as e:
        raise SystemError(
            code=ErrorCode.SYSTEM_ERROR,                     
            message=f'Failed to serialize data using {format.value}: {str(e)}',
            details={'format': format.value, 'data_type': str(type(data))},              
            original_error=e
        ) from e

def deserialize(data: bytes, format: SerializationFormat, cls: Optional[Type[T]] = None) -> T:
    """Deserialize bytes to the specified type using the specified format."""
    if not data:
        raise SystemError("Cannot deserialize empty data.")
    target_type = cls if cls else Any
    try:
        if format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif format == SerializationFormat.MSGPACK:
            decoder = msgspec.msgpack.Decoder(target_type)
            return decoder.decode(data)
        elif format == SerializationFormat.JSON:
            decoder = msgspec.json.Decoder(target_type, dec_hook=_json_decoder_hook)
            result = decoder.decode(data)
            if cls is dict and isinstance(result, dict) and "bytes" in result and isinstance(result["bytes"], str):
                try:
                    result["bytes"] = base64.b64decode(result["bytes"])
                except Exception:
                    pass
            
            return result
        else:
            raise ValueError(f'Unsupported deserialization format specified: {format}')
    except Exception as e:
        raise SystemError(
            code=ErrorCode.SYSTEM_ERROR,                     
            message=f'Failed to deserialize data using {format.value} (target_cls: {cls}): {str(e)}',
            details={'format': format.value, 'target_cls': str(cls)},              
            original_error=e
        ) from e

def serialize_to_json(data: Any, pretty: bool = False) -> str:
    """Serialize data to a JSON string."""
    try:
        serialized_bytes = serialize(data, format=SerializationFormat.JSON, pretty=pretty)
        return serialized_bytes.decode('utf-8')
    except SystemError as e:
        raise SystemError(
            code=ErrorCode.SYSTEM_ERROR,
            message=f'Failed to serialize to JSON: {e.message}',
            original_error=e.original_error
        ) from e

def deserialize_from_json(data: str, cls: Optional[Type[T]] = None) -> T:
    """Deserialize a JSON string to the specified type."""
    try:
        result = deserialize(data.encode('utf-8'), format=SerializationFormat.JSON, cls=cls)
        
                                                                      
        if cls is dict and isinstance(result, dict) and "bytes" in result and isinstance(result["bytes"], str):
            try:
                result["bytes"] = base64.b64decode(result["bytes"])
            except Exception:
                pass
        
        return result
    except SystemError as e:
        raise SystemError(
            code=ErrorCode.SYSTEM_ERROR,
            message=f'Failed to deserialize from JSON: {e.message}',
            original_error=e.original_error
        ) from e
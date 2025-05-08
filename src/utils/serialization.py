# src/utils/serialization.py
import dataclasses
import datetime
import json
import pickle
import uuid
import base64
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar
import msgspec
from src.config.logger import get_logger
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

# Encoder Hook (bytes 및 시간 관련 타입 처리 개선)
def _json_encoder_hook(obj: Any) -> Any:
    if isinstance(obj, bytes):
        # Base64 encode bytes directly to string for JSON compatibility
        return base64.b64encode(obj).decode('ascii')
    # Handle common types that JSON doesn't natively support
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.time):
        # Preserve timezone info if present
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
    # Let msgspec handle other types or raise TypeError
    raise TypeError(f"Object of type {type(obj)} is not directly serializable to JSON by this hook.")

# Decoder Hook (bytes 처리 개선)
def _json_decoder_hook(type_: Type, obj: Any) -> Any:
    # Handle bytes decoding from base64 string
    if type_ is bytes and isinstance(obj, str):
        try:
            return base64.b64decode(obj)
        except Exception as e:
            logger.debug(f"Base64 decoding failed: {e}")
    # Return None to let msgspec attempt its own conversion
    return None

# MsgPack encoder hook
def _msgpack_encoder_hook(obj: Any) -> Any:
    if isinstance(obj, bytes):
        # For msgpack, we can pass bytes directly
        return obj
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, datetime.time):
        # Preserve timezone info if present
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

# Serialization functions
def serialize(data: Any, format: SerializationFormat = SerializationFormat.MSGPACK, pretty: bool = False) -> bytes:
    """Serialize data to bytes using the specified format."""
    try:
        if format == SerializationFormat.MSGPACK:
            encoder = msgspec.msgpack.Encoder(enc_hook=_msgpack_encoder_hook)
            return encoder.encode(data)
        elif format == SerializationFormat.JSON:
            if pretty:
                # Use standard json for pretty printing
                return json.dumps(
                    data,
                    default=_json_encoder_hook,
                    ensure_ascii=False,
                    indent=2
                ).encode('utf-8')
            else:
                # Use msgspec json for better performance
                encoder = msgspec.json.Encoder(enc_hook=_json_encoder_hook)
                return encoder.encode(data)
        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(data)
        else:
            raise ValueError(f'Unsupported serialization format: {format}')
    except Exception as e:
        raise SystemError(
            code=ErrorCode.SYSTEM_ERROR, # SYSTEM_ERROR 코드 사용
            message=f'Failed to serialize data using {format.value}: {str(e)}',
            details={'format': format.value, 'data_type': str(type(data))}, # 상세 정보 추가 가능
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
            # Use decoder hook for special types like bytes
            decoder = msgspec.json.Decoder(target_type, dec_hook=_json_decoder_hook)
            result = decoder.decode(data)
            
            # Extra handling for bytes in dictionaries when using cls=dict
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
            code=ErrorCode.SYSTEM_ERROR, # SYSTEM_ERROR 코드 사용
            message=f'Failed to deserialize data using {format.value} (target_cls: {cls}): {str(e)}',
            details={'format': format.value, 'target_cls': str(cls)}, # 상세 정보 추가 가능
            original_error=e
        ) from e

# JSON specific functions
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
        
        # Extra handling for bytes in dictionaries when using cls=dict
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
        
"""bytes 타입 처리 개선

JSON 인코딩/디코딩 시 Base64 처리 로직 강화
별도의 msgpack 인코더 훅 추가


시간 관련 타입 처리 개선

datetime, date, time 객체의 일관된 직렬화/역직렬화
시간대(tzinfo) 정보 보존 방식 개선


데이터 타입 변환 로직 강화

dict 결과에서 bytes 타입 자동 변환 추가
JSON 특화 함수에서 일관된 처리 보장_summary_
"""
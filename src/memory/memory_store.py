# src/memory/memory_store.py
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable

# 필요한 설정 및 유틸리티 임포트
from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.config.errors import ErrorCode, MemoryError, convert_exception
# Redis 연결 관리 모듈 임포트
from src.config.connections import get_redis_async_connection
# 직렬화 유틸리티 (msgspec 기반) 임포트
from src.utils.serialization import serialize, deserialize, SerializationFormat

logger = get_logger(__name__)
settings = get_settings()

# --- Redis 백엔드 함수 ---

async def _redis_save_state(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Redis에 상태를 저장합니다 (msgspec으로 직렬화)."""
    redis = None # redis 변수 초기화
    try:
        redis = await get_redis_async_connection()
        # 직렬화 (기본적으로 msgpack 사용)
        serialized_value = serialize(value, format=SerializationFormat.MSGPACK)
        data_size_bytes = len(serialized_value)
        logger.debug(f"[RedisStore] Saving key '{key}'. Size: {data_size_bytes} bytes, TTL: {ttl}s")

        start_time = time.monotonic()
        success: bool = False
        if ttl is not None and ttl > 0:
            # setex는 TTL을 초 단위 정수로 받음
            success = await redis.setex(key, ttl, serialized_value)
        else:
            success = await redis.set(key, serialized_value)
        duration = time.monotonic() - start_time
        # 너무 빈번한 로그일 수 있으므로 DEBUG 레벨 유지 또는 샘플링 고려
        # logger.debug(f"[RedisStore] SET operation for key '{key}' took {duration:.4f}s. Success: {success}")

        # TODO: Metrics 추적 추가 (선택 사항)
        # metrics.track_memory('duration', operation_type='redis_set', value=duration)
        # if success: metrics.track_memory('size', memory_type='redis', value=data_size_bytes)

        if not success:
             logger.warning(f"[RedisStore] Failed to save key '{key}' (operation returned false).")
        return bool(success)
    except Exception as e:
        # 연결 오류 포함 모든 예외 처리
        # 모든 Redis write-path 오류는 REDIS_OPERATION_ERROR 로 래핑
        raise MemoryError(
            code=ErrorCode.REDIS_OPERATION_ERROR,
            message=f"Failed to save key '{key}' to Redis",
            original_error=e,
        )
        
async def _redis_load_state(key: str, default: Any = None) -> Any:
    """
    Redis 에서 상태를 로드하고 msgpack → Python 객체로 역직렬화.
    역직렬화 실패 시 MEMORY_RETRIEVAL_ERROR, Redis I/O 실패 시 REDIS_OPERATION_ERROR.
    """
    redis = None
    try:
        redis = await get_redis_async_connection()

        start = time.monotonic()
        data: Optional[bytes] = await redis.get(key)
        duration = time.monotonic() - start
        # metrics.sample('redis_get_latency', duration)

        if data is None:
            logger.debug(f"[RedisStore] Key '{key}' not found.")
            return default

        try:
            return deserialize(data, format=SerializationFormat.MSGPACK, cls=Any)
        except Exception as deser_error:
            logger.error(
                f"[RedisStore] Failed to deserialize data for key '{key}' (msgpack)",
                exc_info=True,
            )
            # 그대로 전파해야 외부 except 에서 다시 래핑하지 않음
            raise MemoryError(
                code=ErrorCode.MEMORY_RETRIEVAL_ERROR,
                message=f"Failed to deserialize data for key '{key}'",
                original_error=deser_error,
            )
    except MemoryError:
        # 내부에서 이미 의미 있는 ErrorCode를 붙여 던진 경우 그대로 전달
        raise
    except Exception as e:
        raise MemoryError(
            code=ErrorCode.REDIS_OPERATION_ERROR,
            message=f"Failed to load key '{key}' from Redis",
            original_error=e,
        )


async def _redis_delete_state(key: str) -> bool:
    """Redis에서 상태를 삭제합니다."""
    redis = None
    try:
        redis = await get_redis_async_connection()
        # logger.debug(f"[RedisStore] Deleting key '{key}'") # 너무 빈번할 수 있음

        start_time = time.monotonic()
        result: int = await redis.delete(key)
        duration = time.monotonic() - start_time
        # logger.debug(f"[RedisStore] DEL operation for key '{key}' took {duration:.4f}s")
        # TODO: Metrics 추적 추가
        # metrics.track_memory('duration', operation_type='redis_delete', value=duration)

        deleted = result > 0
        logger.debug(f"[RedisStore] Key '{key}' deletion status: {deleted}")
        return deleted
    except Exception as e:
        error = convert_exception(
            e, ErrorCode.REDIS_OPERATION_ERROR, f"Failed to delete key '{key}' from Redis"
        )
        raise error

async def _redis_exists(key: str) -> bool:
    """Redis에서 키 존재 여부를 확인합니다."""
    redis = None
    try:
        redis = await get_redis_async_connection()
        # logger.debug(f"[RedisStore] Checking existence for key '{key}'") # 너무 빈번할 수 있음

        start_time = time.monotonic()
        result: int = await redis.exists(key)
        duration = time.monotonic() - start_time
        # logger.debug(f"[RedisStore] EXISTS operation for key '{key}' took {duration:.4f}s")
        # TODO: Metrics 추적 추가
        # metrics.track_memory('duration', operation_type='redis_exists', value=duration)

        key_exists = result > 0
        # logger.debug(f"[RedisStore] Key '{key}' exists: {key_exists}")
        return key_exists
    except Exception as e:
        error = convert_exception(
            e, ErrorCode.REDIS_OPERATION_ERROR, f"Failed to check existence for key '{key}' from Redis"
        )
        raise error

async def _redis_get_history(key_prefix: str, limit: Optional[int] = None) -> List[Any]:
    """
    Redis에서 특정 접두사를 가진 키들의 값을 (역순으로 가정된 키 정렬 후) 가져옵니다.
    주의: 이 구현은 단순 키-값 구조를 가정하며, 실제 'history' 저장 방식에 따라 변경되어야 합니다.
          대규모 데이터셋에서는 SCAN 방식이 필수적입니다.
    """
    redis = None
    try:
        redis = await get_redis_async_connection()
        logger.debug(f"[RedisStore] Getting history for prefix '{key_prefix}', limit={limit}")

        # --- SCAN 방식 사용 ---
        keys_bytes = []
        cursor = b'0'
        scan_pattern = f"{key_prefix}*"
        scan_start_time = time.monotonic()

        async for key_bytes in redis.scan_iter(match=scan_pattern, count=1000):
            keys_bytes.append(key_bytes)
            # SCAN 중에도 limit을 초과하면 중단하는 로직 추가 가능 (성능 개선)
            # if limit is not None and len(keys_bytes) >= limit * 2: # 여유롭게 가져옴
            #     logger.debug(f"[RedisStore] SCAN found enough keys ({len(keys_bytes)}), stopping early.")
            #     break # 비효율적일 수 있음. 전체 스캔 후 정렬하는 것이 나을 수 있음

        scan_duration = time.monotonic() - scan_start_time
        logger.debug(f"[RedisStore] SCAN operation took {scan_duration:.4f}s, found {len(keys_bytes)} keys for prefix '{key_prefix}'.")

        if not keys_bytes:
            return []

        # --- 키 정렬 (키 형식에 따라 수정 필요) ---
        # 예: 키가 'prefix:timestamp' 형식이라고 가정하고 시간 역순 정렬
        def sort_key_func(key_b: bytes) -> float:
            try:
                # 키 형식에 맞게 파싱 로직 수정
                # 예: 'memory:task123:history_1678886400.123' -> 1678886400.123
                timestamp_str = key_b.decode().split(':')[-1].split('_')[-1]
                return float(timestamp_str)
            except (IndexError, ValueError, UnicodeDecodeError):
                logger.warning(f"Could not parse timestamp from key: {key_b.decode(errors='ignore')}")
                return 0.0 # 파싱 실패 시 맨 뒤로

        sorted_keys = sorted(keys_bytes, key=sort_key_func, reverse=True)

        if limit is not None:
            keys_to_fetch = sorted_keys[:limit]
        else:
            keys_to_fetch = sorted_keys

        if not keys_to_fetch:
            return []

        # --- 값 가져오기 (MGET) ---
        mget_start_time = time.monotonic()
        values_bytes: List[Optional[bytes]] = await redis.mget(*keys_to_fetch)
        mget_duration = time.monotonic() - mget_start_time
        logger.debug(f"[RedisStore] MGET operation took {mget_duration:.4f}s for {len(keys_to_fetch)} keys.")

        # --- 역직렬화 ---
        history = []
        for i, data in enumerate(values_bytes):
            if data:
                try:
                    deserialized_item = deserialize(data, format=SerializationFormat.MSGPACK, cls=Any)
                    history.append(deserialized_item)
                except Exception as deser_err:
                    key_str = keys_to_fetch[i].decode(errors='ignore')
                    logger.warning(f"[RedisStore] Failed to deserialize history item for key '{key_str}': {deser_err}")
            else:
                # MGET은 키가 없거나 만료되면 None 반환
                key_str = keys_to_fetch[i].decode(errors='ignore')
                logger.warning(f"[RedisStore] MGET returned None for key '{key_str}', possibly expired or deleted.")

        logger.info(f"[RedisStore] Retrieved {len(history)} history items for prefix '{key_prefix}'.")
        return history

    except Exception as e:
        error = convert_exception(
            e, ErrorCode.REDIS_OPERATION_ERROR, f"Failed to get history for prefix '{key_prefix}' from Redis"
        )
        raise error


# --- 파일 시스템 백엔드 함수 (구현 시 추가) ---
# async def _file_save_state(key: str, value: Any, ttl: Optional[int] = None) -> bool: ...
# async def _file_load_state(key: str, default: Any = None) -> Any: ...
# async def _file_delete_state(key: str) -> bool: ...
# async def _file_exists(key: str) -> bool: ...
# async def _file_get_history(key_prefix: str, limit: Optional[int] = None) -> List[Any]: ...


# --- 백엔드 함수 타입 정의 ---
SaveStateFunc = Callable[[str, Any, Optional[int]], Awaitable[bool]]
LoadStateFunc = Callable[[str, Optional[Any]], Awaitable[Any]]
DeleteStateFunc = Callable[[str], Awaitable[bool]]
ExistsFunc = Callable[[str], Awaitable[bool]]
GetHistoryFunc = Callable[[str, Optional[int]], Awaitable[List[Any]]]

class StoreBackendFunctions(Dict[str, Optional[Callable]]):
    save_state: Optional[SaveStateFunc]
    load_state: Optional[LoadStateFunc]
    delete_state: Optional[DeleteStateFunc]
    exists: Optional[ExistsFunc]
    get_history: Optional[GetHistoryFunc]

# --- 백엔드 선택 로직 ---
_backend_functions: StoreBackendFunctions = {
    'save_state': None,
    'load_state': None,
    'delete_state': None,
    'exists': None,
    'get_history': None,
}
_backend_initialized = False

def _initialize_store_backend():
    """설정에 따라 백엔드 함수를 설정합니다."""
    global _backend_functions, _backend_initialized
    if _backend_initialized:
        return

    memory_type = getattr(settings, 'MEMORY_TYPE', 'redis').lower() # 설정에서 메모리 타입 읽기
    logger.info(f"Initializing memory store backend with type: '{memory_type}'")

    if memory_type == 'redis':
        _backend_functions['save_state'] = _redis_save_state
        _backend_functions['load_state'] = _redis_load_state
        _backend_functions['delete_state'] = _redis_delete_state
        _backend_functions['exists'] = _redis_exists
        _backend_functions['get_history'] = _redis_get_history
        logger.info("Redis memory store backend selected.")
    # elif memory_type == 'file':
        # 파일 시스템 백엔드 함수 연결
        # _backend_functions['save_state'] = _file_save_state
        # _backend_functions['load_state'] = _file_load_state
        # ...
        # logger.info("File system memory store backend selected.")
    else:
        raise ValueError(f"Unsupported MEMORY_TYPE configured: '{memory_type}'. Supported types: 'redis'.") # 'file' 추가 가능

    _backend_initialized = True
    logger.info("Memory store backend initialization complete.")

# --- 공개 인터페이스 ---
async def save_state(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """선택된 백엔드를 사용하여 상태를 저장합니다."""
    if not _backend_initialized: _initialize_store_backend()
    func = _backend_functions['save_state']
    if not func: raise RuntimeError("Memory store 'save_state' function not initialized.")
    # 여기서 MemoryError를 직접 처리하거나 Manager에서 처리하도록 위임 가능
    try:
        return await func(key, value, ttl)
    except MemoryError: # 이미 MemoryError인 경우 그대로 발생
        raise
    except Exception as e: # 다른 예외는 MemoryError로 변환
        raise convert_exception(e, ErrorCode.MEMORY_STORAGE_ERROR, f"Failed to save state for key '{key}'")

async def load_state(key: str, default: Any = None) -> Any:
    """선택된 백엔드를 사용하여 상태를 로드합니다."""
    if not _backend_initialized: _initialize_store_backend()
    func = _backend_functions['load_state']
    if not func: raise RuntimeError("Memory store 'load_state' function not initialized.")
    try:
        return await func(key, default)
    except MemoryError:
        raise
    except Exception as e:
        raise convert_exception(e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to load state for key '{key}'")

async def delete_state(key: str) -> bool:
    """선택된 백엔드를 사용하여 상태를 삭제합니다."""
    if not _backend_initialized: _initialize_store_backend()
    func = _backend_functions['delete_state']
    if not func: raise RuntimeError("Memory store 'delete_state' function not initialized.")
    try:
        return await func(key)
    except MemoryError:
        raise
    except Exception as e:
        raise convert_exception(e, ErrorCode.MEMORY_STORAGE_ERROR, f"Failed to delete state for key '{key}'")

async def exists(key: str) -> bool:
    """선택된 백엔드를 사용하여 키 존재 여부를 확인합니다."""
    if not _backend_initialized: _initialize_store_backend()
    func = _backend_functions['exists']
    if not func: raise RuntimeError("Memory store 'exists' function not initialized.")
    try:
        return await func(key)
    except MemoryError:
        raise
    except Exception as e:
        raise convert_exception(e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to check existence for key '{key}'")

async def get_history(key_prefix: str, limit: Optional[int] = None) -> List[Any]:
    """
    선택된 백엔드를 사용하여 특정 접두사를 가진 키들의 값을 가져옵니다.
    주의: 이 구현은 백엔드 및 키 저장 방식에 따라 성능 특성이 다릅니다.
    """
    if not _backend_initialized: _initialize_store_backend()
    func = _backend_functions['get_history']
    if not func: raise RuntimeError("Memory store 'get_history' function not initialized.")
    try:
        return await func(key_prefix, limit)
    except MemoryError:
        raise
    except Exception as e:
        raise convert_exception(e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to get history for prefix '{key_prefix}'")
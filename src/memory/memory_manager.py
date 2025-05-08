# src/memory/memory_manager.py
import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, TypeVar
from cachetools import TTLCache, keys

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.config.errors import ErrorCode, MemoryError, convert_exception
# memory_store 모듈의 함수들을 임포트합니다.
from src.memory import memory_store
# 키 생성을 위한 유틸리티 (기존 memory/utils.py 에 있었다면 해당 경로 사용)
# 여기서는 로드맵에 따라 src/utils/ids.py 를 사용한다고 가정합니다.
# 만약 generate_memory_key가 다른 곳에 있다면 경로 수정 필요
from src.utils.ids import generate_uuid # 예시 ID 생성기 (키 생성은 직접 구현)

logger = get_logger(__name__)
settings = get_settings()

# 캐싱된 결과를 위한 타입 변수
R = TypeVar('R')

# 기본 메모리 키 접두사
DEFAULT_KEY_PREFIX = "memory"

class MemoryManager:
    """
    메모리 저장을 위한 고수준 관리자 인터페이스.
    키 관리, TTL, L1 캐싱 기능을 제공합니다.
    """
    def __init__(
        self,
        default_ttl: Optional[int] = None,
        cache_size: int = 10000, # 기본 캐시 크기 설정
        cache_ttl: Optional[int] = 3600 # 기본 캐시 TTL 설정 (초)
    ):
        """
        MemoryManager를 초기화합니다.

        Args:
            default_ttl: 저장소에 저장 시 기본 TTL (초). None이면 설정값(MEMORY_TTL) 사용.
            cache_size: L1 캐시의 최대 항목 수.
            cache_ttl: L1 캐시의 기본 TTL (초). None이면 캐시 TTL 없음.
        """
        self.default_ttl = default_ttl if default_ttl is not None else settings.MEMORY_TTL
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size

        # L1 캐시 초기화 (TTLCache 사용)
        if self.cache_ttl is not None and self.cache_ttl > 0:
            self._cache: TTLCache[str, Any] = TTLCache(maxsize=self.cache_size, ttl=self.cache_ttl)
            logger.info(f"MemoryManager L1 Cache enabled. Max size: {self.cache_size}, TTL: {self.cache_ttl}s")
        else:
            self._cache = None # 캐시 비활성화
            logger.info("MemoryManager L1 Cache is disabled.")

        # 캐시 접근 동기화를 위한 Lock (key별 Lock)
        self._cache_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock() # _cache_locks 딕셔너리 자체 접근 보호

        logger.info(f"MemoryManager initialized. Default storage TTL: {self.default_ttl}s")

    def _get_full_key(self, context_id: str, key: str) -> str:
        """Context ID와 Key를 조합하여 저장소에서 사용할 전체 키를 생성합니다."""
        # src/memory/utils.py의 generate_memory_key 와 동일한 로직
        if not context_id:
            raise ValueError("context_id cannot be empty.")
        if not key:
            raise ValueError("key cannot be empty.")
        return f"{DEFAULT_KEY_PREFIX}:{context_id}:{key}"

    def _get_history_key_prefix(self, context_id: str, history_key_prefix: str) -> str:
        """기록 조회를 위한 키 접두사를 생성합니다."""
        if not context_id:
             raise ValueError("context_id cannot be empty.")
        if not history_key_prefix:
             raise ValueError("history_key_prefix cannot be empty.")
        # 접두사 형식을 일관되게 관리 (예: memory:context_id:history_prefix)
        return f"{DEFAULT_KEY_PREFIX}:{context_id}:{history_key_prefix}"


    def _get_effective_ttl(self, ttl: Optional[int]) -> Optional[int]:
        """입력된 TTL과 기본 TTL을 고려하여 실제 적용될 TTL을 계산합니다."""
        if ttl is not None:
            # 명시적 TTL이 0 또는 음수이면 영구 저장 (None 반환)
            return ttl if ttl > 0 else None
        else:
            # 기본 TTL이 0 또는 음수이면 영구 저장 (None 반환)
            return self.default_ttl if self.default_ttl > 0 else None

    async def _get_cache_lock(self, key: str) -> asyncio.Lock:
        """캐시 키에 대한 비동기 Lock을 가져옵니다 (없으면 생성)."""
        async with self._locks_lock:
            if key not in self._cache_locks:
                self._cache_locks[key] = asyncio.Lock()
            return self._cache_locks[key]

    async def save_state(self, context_id: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        상태를 저장합니다. L1 캐시도 업데이트합니다.

        Args:
            context_id: 컨텍스트 식별자.
            key: 저장할 상태의 키.
            value: 저장할 상태 값.
            ttl: 저장소 TTL (초). None이면 기본 TTL 사용.

        Returns:
            성공 여부 (bool).
        """
        full_key = self._get_full_key(context_id, key)
        effective_ttl = self._get_effective_ttl(ttl)
        # logger.debug(f"Saving state for key: '{full_key}', TTL: {effective_ttl}s") # 너무 빈번할 수 있음

        try:
            success = await memory_store.save_state(full_key, value, effective_ttl)
            if success and self._cache is not None:
                # 캐시 업데이트 (쓰기 시 캐시 업데이트)
                try:
                    # logger.debug(f"Updating L1 cache for key '{full_key}' after save.")
                    self._cache[full_key] = value
                except Exception as cache_err:
                    # 캐시 업데이트 실패는 로깅만 하고 무시 (저장은 성공했으므로)
                    logger.warning(f"Failed to update L1 cache for key '{full_key}' after save: {cache_err}")
            elif not success:
                 logger.warning(f"Failed to save state for key '{full_key}' to the store.")

            return success
        except Exception as e:
            # memory_store에서 MemoryError가 발생했거나, 여기서 다른 예외 발생 시
            error = convert_exception(
                e, ErrorCode.MEMORY_STORAGE_ERROR, f"Failed to save state for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger) # 여기서 에러 로깅
            return False # 실패 시 False 반환

    async def load_state(self, context_id: str, key: str, default: Any = None) -> Any:
        """
        상태를 로드합니다. L1 캐시를 먼저 확인합니다.

        Args:
            context_id: 컨텍스트 식별자.
            key: 로드할 상태의 키.
            default: 키가 없을 때 반환할 기본값.

        Returns:
            로드된 상태 값 또는 기본값.
        """
        full_key = self._get_full_key(context_id, key)
        # logger.debug(f"Loading state for key: '{full_key}'") # 너무 빈번할 수 있음

        # 1. L1 캐시 확인
        if self._cache is not None:
            try:
                cached_value = self._cache[full_key]
                # TTLCache는 만료된 항목 접근 시 KeyError 발생
                logger.debug(f"L1 Cache HIT for key: '{full_key}'")
                # TODO: Metrics 추적 추가
                # metrics.track_cache('hits', cache_type='memory_manager_l1')
                return cached_value
            except KeyError:
                logger.debug(f"L1 Cache MISS for key: '{full_key}'")
                # TODO: Metrics 추적 추가
                # metrics.track_cache('misses', cache_type='memory_manager_l1')
            except Exception as cache_err:
                logger.warning(f"Error accessing L1 cache for key '{full_key}': {cache_err}")

        # 2. 캐시 미스 시 저장소에서 로드 (캐시 경쟁 방지 Lock 사용)
        lock = await self._get_cache_lock(full_key)
        async with lock:
            # Lock 획득 후 캐시 재확인 (다른 요청이 먼저 로드했을 수 있음)
            if self._cache is not None:
                 try:
                     cached_value = self._cache[full_key]
                     logger.debug(f"L1 Cache HIT for key '{full_key}' after acquiring lock.")
                     return cached_value
                 except KeyError:
                     pass # 캐시 미스 확인
                 except Exception:
                     pass # 오류 무시하고 저장소 조회

            # 저장소에서 로드
            try:
                value = await memory_store.load_state(full_key, default)
                # 성공적으로 로드했고, 기본값이 아니며, 캐시가 활성화 상태면 캐시에 저장
                if value is None:
                    return default
                # 3) 캐시 활성 시 결과 저장
                if self._cache is not None:
                    try:
                        self._cache[full_key] = value
                    except Exception as cache_err:
                        logger.warning(f"Failed to store loaded value into L1 cache for key '{full_key}': {cache_err}")
                return value
            except Exception as e:
                error = convert_exception(
                    e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to load state for key '{key}' in context '{context_id}'"
                )
                error.log_error(logger)
                return default # 로드 실패 시 default 반환

    async def delete_state(self, context_id: str, key: str) -> bool:
        """
        상태를 삭제합니다. L1 캐시에서도 제거합니다.

        Args:
            context_id: 컨텍스트 식별자.
            key: 삭제할 상태의 키.

        Returns:
            성공 여부 (bool).
        """
        full_key = self._get_full_key(context_id, key)
        # logger.debug(f"Deleting state for key: '{full_key}'") # 너무 빈번할 수 있음

        # 1. 캐시에서 제거 (먼저 또는 나중에 해도 됨)
        if self._cache is not None:
            if full_key in self._cache:
                try:
                    del self._cache[full_key]
                    logger.debug(f"Removed key '{full_key}' from L1 cache.")
                except Exception as cache_err:
                    logger.warning(f"Failed to remove key '{full_key}' from L1 cache during delete: {cache_err}")

        # 2. 저장소에서 삭제
        try:
            success = await memory_store.delete_state(full_key)
            if not success:
                logger.debug(f"Delete operation for key '{full_key}' returned false (key might not have existed).")
            return success
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to delete state for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger)
            return False

    async def exists(self, context_id: str, key: str) -> bool:
        full_key = self._get_full_key(context_id, key)
        # 1. 캐시 확인 (기존 로직 유지)
        if self._cache is not None and full_key in self._cache:
            logger.debug(f"Key '{full_key}' found in L1 cache during existence check.")
            return True

        # 2. 저장소 호출
        try:
            return await memory_store.exists(full_key)
        except MemoryError:
            # 이미 분류된 MemoryError 재전파
            raise
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to check existence for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger)
            return False


    async def get_history(self, context_id: str, history_key_prefix: str, limit: Optional[int] = None) -> List[Any]:
        full_prefix = self._get_history_key_prefix(context_id, history_key_prefix)
        logger.debug(f"Getting history for prefix: '{full_prefix}', limit: {limit}")
        try:
            # 캐시 없이 직접 memory_store 호출
            return await memory_store.get_history(full_prefix, limit)
        except MemoryError:
            # 이미 MemoryError 라면 상위로
            raise
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to get history for prefix '{history_key_prefix}' in context '{context_id}'"
            )
            error.log_error(logger)
            return []


    async def clear_context(self, context_id: str) -> bool:
        """
        특정 컨텍스트에 해당하는 모든 상태를 저장소와 캐시에서 삭제합니다.
        주의: 저장소의 clear 기능이 필요합니다 (현재 memory_store에는 없음).
             구현되지 않았다면 False를 반환하거나 예외를 발생시켜야 합니다.

        Args:
            context_id: 삭제할 컨텍스트 식별자.

        Returns:
            성공 여부 (bool).
        """
        logger.warning(f"Clearing context '{context_id}' - Store-level clear not implemented in this version.")
        # TODO: memory_store에 clear_context 기능 구현 필요 (예: SCAN + DELETE 사용)
        #      구현 전까지는 이 기능을 사용하지 않거나, 개별 delete를 반복해야 함.

        # 임시 방편: 캐시만 클리어
        cleared_cache_count = 0
        if self._cache is not None:
             prefix = f"{DEFAULT_KEY_PREFIX}:{context_id}:"
             keys_to_delete = [k for k in list(self._cache.keys()) if k.startswith(prefix)]
             for k in keys_to_delete:
                 if k in self._cache:
                     del self._cache[k]
                     cleared_cache_count += 1
             logger.info(f"Cleared {cleared_cache_count} items from L1 cache for context '{context_id}'.")

        # 저장소 클리어 기능이 없으므로 False 반환
        logger.error("Persistent storage clear functionality for context is not available.")
        return False

# --- 싱글톤 관리 ---
_memory_manager_instance: Optional[MemoryManager] = None
_memory_manager_lock = threading.Lock()

def get_memory_manager() -> MemoryManager:
    """
    MemoryManager 싱글톤 인스턴스를 가져옵니다.
    첫 호출 시 설정을 기반으로 인스턴스를 생성합니다.
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        with _memory_manager_lock:
            # Double-check locking
            if _memory_manager_instance is None:
                logger.info("Initializing MemoryManager singleton instance...")
                try:
                    # MemoryManager 생성 (기본 설정값 사용)
                    _memory_manager_instance = MemoryManager(
                        # default_ttl=settings.MEMORY_TTL, # 생성자에서 이미 처리
                        # cache_size=settings.MEMORY_MANAGER_CACHE_SIZE,
                        # cache_ttl=settings.CACHE_TTL
                    )
                    logger.info("MemoryManager singleton instance created successfully.")
                except Exception as e:
                    logger.critical(f"Failed to initialize MemoryManager instance: {e}", exc_info=True)
                    # 실패 시 앱 실행이 어려울 수 있으므로 에러 발생
                    raise RuntimeError("Failed to initialize MemoryManager") from e

    return _memory_manager_instance
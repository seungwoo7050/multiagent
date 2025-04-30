import abc
import asyncio
import json
import time
from typing import Any, Dict, Optional, Union, TypeVar, Generic, List, Tuple, cast
import hashlib
from functools import lru_cache
from src.config.settings import get_settings
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager, CACHE_METRICS, MEMORY_METRICS
from src.config.errors import MemoryError, ErrorCode
from src.config.connections import get_connection_manager

settings = get_settings()
logger = get_logger(__name__)
metrics = get_metrics_manager()
connection_manager = get_connection_manager()

T = TypeVar('T')
_CACHE_INSTANCE = None
_CACHE_LOCK = asyncio.Lock()

class LLMCache(abc.ABC, Generic[T]):

    @abc.abstractmethod
    async def get(self, key: str) -> Optional[T]:
        pass

    @abc.abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int]=None) -> bool:
        pass

    @abc.abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abc.abstractmethod
    async def clear(self) -> bool:
        pass

    @abc.abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass

class TwoLevelCache(LLMCache[T]):

    def __init__(self, namespace: str='llm', local_maxsize: int=1000, ttl: int=3600, serializer: Optional[callable]=None, deserializer: Optional[callable]=None):
        self.namespace = namespace
        self.ttl = ttl
        self.local_maxsize = local_maxsize
        self.serializer = serializer or self._default_serializer
        self.deserializer = deserializer or self._default_deserializer
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.local_cache_order: list[str] = []
        logger.debug(f"Initialized two-level cache with namespace '{namespace}', L1 maxsize={local_maxsize}, default TTL={ttl}s")

    def _default_serializer(self, value: T) -> str:
        try:
            return json.dumps(value, default=str)
        except TypeError as e:
            logger.error(f'Default serialization failed: {e}. Value type: {type(value)}')
            raise

    def _default_deserializer(self, serialized: Union[str, bytes]) -> T:
        try:
            if isinstance(serialized, bytes):
                serialized = serialized.decode('utf-8')
            return json.loads(serialized)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f'Default deserialization failed: {e}. Data: {serialized[:100]}...')
            raise

    def _get_redis_key(self, key: str) -> str:
        return f'{self.namespace}:{key}'

    def _update_lru_order(self, key: str) -> None:
        if key in self.local_cache_order:
            self.local_cache_order.remove(key)
        self.local_cache_order.append(key)
        while len(self.local_cache_order) > self.local_maxsize:
            oldest_key = self.local_cache_order.pop(0)
            self.local_cache.pop(oldest_key, None)
            logger.debug(f'L1 cache evicted key: {oldest_key} (size exceeded {self.local_maxsize})')

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'cache_get'})
    async def get(self, key: str) -> Optional[T]:
        metrics.track_cache('operations', operation_type='get')
        if key in self.local_cache:
            entry = self.local_cache[key]
            expires_at = entry.get('expires_at')
            if expires_at is None or expires_at > time.time():
                self._update_lru_order(key)
                self.hit_count += 1
                metrics.track_cache('hits', cache_type='local')
                logger.debug(f'L1 cache hit for key: {key}')
                return cast(T, entry['value'])
            else:
                logger.debug(f'L1 cache expired for key: {key}')
                self.local_cache.pop(key, None)
                if key in self.local_cache_order:
                    self.local_cache_order.remove(key)
        try:
            async with connection_manager.redis_async_connection() as redis:
                redis_key = self._get_redis_key(key)
                start_time = time.time()
                serialized = await redis.get(redis_key)
                if serialized is not None:
                    value: T = self.deserializer(serialized)
                    redis_ttl = await redis.ttl(redis_key)
                    if redis_ttl == -1:
                        l1_expires_at = None
                    elif redis_ttl > 0:
                        l1_expires_at = time.time() + redis_ttl
                    self.local_cache[key] = {'value': value, 'expires_at': l1_expires_at}
                    self._update_lru_order(key)
                    metrics.track_cache('size', cache_type='local', value=len(self.local_cache))
                    
                    self.hit_count += 1
                    metrics.track_cache('hits', cache_type='redis')
                    logger.debug(f'L2 cache hit for key: {key}. Stored in L1.')
                    return value
                else:
                    self.miss_count += 1
                    metrics.track_cache('misses', cache_type='redis')
                    logger.debug(f'Cache miss for key: {key} (not found in L1 or L2).')
                    return None
        except Exception as e:
            logger.warning(f"Error retrieving key '{key}' from Redis cache: {str(e)}", exc_info=True)
            self.miss_count += 1
            metrics.track_cache('misses', cache_type='redis_error')
            return None

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'cache_set'})
    async def set(self, key: str, value: T, ttl: Optional[int]=None) -> bool:
        metrics.track_cache('operations', operation_type='set')
        effective_ttl = ttl if ttl is not None else self.ttl
        l1_expires_at: Optional[float] = None
        if effective_ttl > 0:
            l1_expires_at = time.time() + effective_ttl
        self.local_cache[key] = {'value': value, 'expires_at': l1_expires_at}
        self._update_lru_order(key)
        metrics.track_cache('size', cache_type='local', value=len(self.local_cache))
        try:
            async with connection_manager.redis_async_connection() as redis:
                redis_key = self._get_redis_key(key)
                serialized = self.serializer(value)
                if effective_ttl > 0:
                    await redis.setex(redis_key, effective_ttl, serialized)
                else:
                    await redis.set(redis_key, serialized)
                self.set_count += 1
                logger.debug(f"Set key '{key}' in L1 and L2 cache. TTL: {effective_ttl}s.")
                return True
        except Exception as e:
            logger.warning(f"Error storing key '{key}' in L2 Redis cache: {str(e)}", exc_info=True)
        return True
    
    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'cache_delete'})
    async def delete(self, key: str) -> bool:
        metrics.track_cache('operations', operation_type='delete')
        deleted_from_l1 = False
        if key in self.local_cache:
            self.local_cache.pop(key, None)
            if key in self.local_cache_order:
                self.local_cache_order.remove(key)
            deleted_from_l1 = True
            metrics.track_cache('size', cache_type='local', value=len(self.local_cache))
            logger.debug(f"Deleted key '{key}' from L1 cache.")
        deleted_from_l2 = False
        try:
            async with connection_manager.redis_async_connection() as redis:
                redis_key = self._get_redis_key(key)
                deleted_count = await redis.delete(redis_key)
                deleted_from_l2 = deleted_count > 0
                if deleted_from_l2:
                    self.delete_count += 1
                    logger.debug(f"Deleted key '{redis_key}' from L2 cache.")
                elif not deleted_from_l1:
                    logger.debug(f"Key '{key}' not found in L1 or L2 cache for deletion.")
        except Exception as e:
            logger.warning(f"Error deleting key '{key}' from L2 Redis cache: {str(e)}", exc_info=True)

        return deleted_from_l1 or deleted_from_l2

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'cache_clear'})
    async def clear(self) -> bool:
        metrics.track_cache('operations', operation_type='clear')
        l1_cleared_count = len(self.local_cache)
        self.local_cache.clear()
        self.local_cache_order.clear()
        metrics.track_cache('size', cache_type='local', value=0)
        logger.info(f'Cleared L1 cache ({l1_cleared_count} items).')
        l2_cleared = False
        try:
            async with connection_manager.redis_async_connection() as redis:
                pattern = f'{self.namespace}:*'
                logger.info(f'Clearing L2 cache keys matching pattern: {pattern}')
                cursor = '0'
                deleted_count = 0
                while True:
                    cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=1000)
                    if keys:
                        num_deleted = await redis.delete(*keys)
                        deleted_count += num_deleted
                        logger.debug(f'Deleted {num_deleted} keys from Redis in current scan iteration.')
                    if cursor == b'0':
                        break
                    try:
                        if int(cursor) == 0:
                            break
                    except ValueError:
                        logger.error(f'Unexpected cursor value from Redis SCAN: {cursor}')
                        break
                logger.info(f"Cleared {deleted_count} keys from Redis cache namespace '{self.namespace}'.")
                l2_cleared = True
        except Exception as e:
            logger.error(f"Error clearing Redis cache namespace '{self.namespace}': {str(e)}", exc_info=True)
            l2_cleared = False
        return l2_cleared

    async def get_stats(self) -> Dict[str, Any]:
        stats = {'cache_type': 'TwoLevelCache', 'namespace': self.namespace, 'local_cache_size': len(self.local_cache), 'local_cache_maxsize': self.local_maxsize, 'hit_count': self.hit_count, 'miss_count': self.miss_count, 'set_count': self.set_count, 'delete_count': self.delete_count, 'hit_ratio': self.hit_count / (self.hit_count + self.miss_count) if self.hit_count + self.miss_count > 0 else 0.0}
        try:
            async with connection_manager.redis_async_connection() as redis:
                redis_key_count = 0
                pattern = f'{self.namespace}:*'
                cursor = '0'
                while True:
                    cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=5000)
                    redis_key_count += len(keys)
                    if cursor == b'0' or int(cursor) == 0:
                        break
                stats['redis_key_count_in_namespace'] = redis_key_count
                stats['redis_connected'] = True
        except Exception as e:
            logger.warning(f'Could not get Redis stats: {str(e)}')
            stats['redis_connected'] = False
            stats['redis_error'] = str(e)
        return stats

async def get_cache() -> LLMCache:
    global _CACHE_INSTANCE, _CACHE_LOCK
    if _CACHE_INSTANCE is not None:
        return _CACHE_INSTANCE
    async with _CACHE_LOCK:
        if _CACHE_INSTANCE is not None:
            return _CACHE_INSTANCE
        logger.info('Creating the global LLM cache instance...')
        try:
            _CACHE_INSTANCE = TwoLevelCache(
                namespace=settings.APP_NAME.lower() + '_llm_cache',
                local_maxsize=5000,
                ttl=settings.CACHE_TTL
            )
            logger.info('Global LLM cache instance created.')
            return _CACHE_INSTANCE
        except Exception as e:
            logger.exception(f'Critical error creating global LLM cache instance: {e}')
            _CACHE_INSTANCE = None
            raise MemoryError(
                code=ErrorCode.INITIALIZATION_ERROR,
                message='Failed to create LLM Cache',
                original_error=e
            )

async def clear_cache() -> bool:
    try:
        cache = await get_cache()
        return await cache.clear()
    except Exception as e:
        logger.error(f'Failed to clear global cache: {e}', exc_info=True)
        return False

async def cache_result(key: str, value: Any, ttl: Optional[int]=None) -> bool:
    try:
        cache = await get_cache()
        return await cache.set(key, value, ttl)
    except Exception as e:
        logger.error(f"Failed to cache result for key '{key}': {e}", exc_info=True)
        return False

async def get_cache_stats() -> Dict[str, Any]:
    try:
        cache = await get_cache()
        return await cache.get_stats()
    except Exception as e:
        logger.error(f'Failed to get cache stats: {e}', exc_info=True)
        return {'error': f'Failed to get cache stats: {str(e)}'}
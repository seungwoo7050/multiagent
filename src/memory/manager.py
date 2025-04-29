import asyncio
import functools
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Coroutine
from cachetools import LRUCache, TTLCache
from src.config.errors import ErrorCode, MemoryError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import MEMORY_OPERATION_DURATION, timed_metric, track_cache_hit, track_cache_miss, track_cache_operation, track_cache_size, track_memory_operation, track_memory_operation_completed
from src.config.settings import get_settings
from src.memory.base import BaseMemory, BaseVectorStore
from src.memory.utils import compute_fingerprint, generate_memory_key
from src.utils.timing import async_timed
logger = get_logger(__name__)
settings = get_settings()

class MemoryManager:

    def __init__(self, primary_memory: BaseMemory, vector_store: Optional[BaseVectorStore]=None, cache_size: int=10000, cache_ttl: int=3600, memory_ttl: int=86400):
        self.primary: BaseMemory = primary_memory
        self.vector_store: Optional[BaseVectorStore] = vector_store
        self.memory_ttl: int = memory_ttl
        self.cache: TTLCache[str, Any] = TTLCache(maxsize=cache_size, ttl=cache_ttl, timer=time.time)
        self._cache_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock: asyncio.Lock = asyncio.Lock()
        logger.info(f'MemoryManager initialized. Primary: {type(primary_memory).__name__}, Vector Store: {(type(vector_store).__name__ if vector_store else 'None')}, Cache Size: {cache_size}, Cache TTL: {cache_ttl}s, Memory TTL: {memory_ttl}s')

    async def _get_cache_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_lock:
            if key not in self._cache_locks:
                self._cache_locks[key] = asyncio.Lock()
                logger.debug(f'Created new cache lock for key: {key}')
            current_lock_count = len(self._cache_locks)
            if current_lock_count > 100 and random.random() < 0.01:
                active_locks = {k: lock for k, lock in self._cache_locks.items() if lock.locked()}
                if key in self._cache_locks:
                    active_locks[key] = self._cache_locks[key]
                if len(active_locks) < current_lock_count * 0.7:
                    removed_count = current_lock_count - len(active_locks)
                    logger.debug(f'Cleaning up unused cache locks. Removed: {removed_count}, Remaining: {len(active_locks)}')
                    self._cache_locks = active_locks
            return self._cache_locks[key]

    def _get_cache_key(self, key: str, context_id: str) -> str:
        return generate_memory_key(key, context_id)

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'load'})
    async def load(self, key: str, context_id: str, default: Any=None, use_cache: bool=True) -> Any:
        track_memory_operation('load')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Loading data for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (use_cache={use_cache})")
        if use_cache:
            track_cache_operation('check')
            try:
                cached_value = self.cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f'L1 Cache hit for key: {cache_key}')
                    track_cache_hit('memory_manager_l1')
                    return cached_value
                else:
                    logger.debug(f'L1 Cache miss or expired for key: {cache_key}')
                    track_cache_miss('memory_manager_l1')
            except Exception as e:
                logger.warning(f"Error accessing L1 cache for key '{cache_key}': {e}")
                track_cache_miss('memory_manager_l1_error')
        lock = await self._get_cache_lock(cache_key)
        async with lock:
            if use_cache:
                try:
                    cached_value = self.cache.get(cache_key)
                    if cached_value is not None:
                        logger.debug(f"L1 Cache hit for key '{cache_key}' after acquiring lock.")
                        return cached_value
                except Exception:
                    pass
            logger.debug(f"Loading key '{key}' from primary storage for context '{context_id}'...")
            try:
                value = await self.primary.load_context(key, context_id, default)
                if value is not default and use_cache:
                    try:
                        logger.debug(f"Updating L1 cache for key '{cache_key}' from primary storage.")
                        self.cache[cache_key] = value
                        track_cache_size('memory_manager_l1', len(self.cache))
                    except Exception as e:
                        logger.warning(f"Failed to update L1 cache for key '{cache_key}' after load: {e}")
                return value
            except Exception as e:
                error = convert_exception(e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to load key '{key}' from primary storage (context: '{context_id}')")
                error.log_error(logger)
                return default

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'save'})
    async def save(self, key: str, context_id: str, data: Any, ttl: Optional[int]=None, update_cache: bool=True) -> bool:
        track_memory_operation('save')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Saving data for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (update_cache={update_cache})")
        try:
            effective_ttl = ttl if ttl is not None else self.memory_ttl
            success = await self.primary.save_context(key, context_id, data, effective_ttl)
            if success and update_cache:
                try:
                    logger.debug(f"Updating L1 cache for key '{cache_key}' after successful save.")
                    track_cache_operation('update')
                    self.cache[cache_key] = data
                    track_cache_size('memory_manager_l1', len(self.cache))
                except Exception as e:
                    logger.warning(f"Failed to update L1 cache for key '{cache_key}' after save: {e}")
            if not success:
                logger.warning(f"Failed to save key '{key}' to primary storage for context '{context_id}'.")
            return success
        except Exception as e:
            error = convert_exception(e, ErrorCode.MEMORY_STORAGE_ERROR, f"Failed to save key '{key}' to context '{context_id}'")
            error.log_error(logger)
            return False

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'delete'})
    async def delete(self, key: str, context_id: str, clear_cache: bool=True) -> bool:
        track_memory_operation('delete')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Deleting data for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (clear_cache={clear_cache})")
        try:
            success = await self.primary.delete_context(key, context_id)
            if success and clear_cache:
                try:
                    if cache_key in self.cache:
                        del self.cache[cache_key]
                        logger.debug(f"Removed key '{cache_key}' from L1 cache after successful delete.")
                        track_cache_operation('delete')
                        track_cache_size('memory_manager_l1', len(self.cache))
                    else:
                        logger.debug(f"Key '{cache_key}' not found in L1 cache during delete.")
                except Exception as e:
                    logger.warning(f"Failed to remove key '{cache_key}' from L1 cache after delete: {e}")
            if not success:
                logger.warning(f"Failed to delete key '{key}' from primary storage for context '{context_id}'.")
            return success
        except Exception as e:
            error = convert_exception(e, ErrorCode.MEMORY_STORAGE_ERROR, f"Failed to delete key '{key}' from context '{context_id}'")
            error.log_error(logger)
            return False

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'clear'})
    async def clear(self, context_id: Optional[str]=None, clear_cache: bool=True, clear_vectors: bool=True) -> bool:
        track_memory_operation('clear')
        operation_desc = f"context '{context_id}'" if context_id else 'all contexts'
        logger.info(f'Clearing memory for {operation_desc} (clear_cache={clear_cache}, clear_vectors={clear_vectors})')
        overall_success = True
        try:
            l2_success = await self.primary.clear(context_id)
            if not l2_success:
                logger.warning(f'Primary storage clear operation reported failure for {operation_desc}.')
        except Exception as e:
            error = convert_exception(e, ErrorCode.MEMORY_STORAGE_ERROR, f'Failed to clear primary storage for {operation_desc}')
            error.log_error(logger)
            overall_success = False
        if clear_cache:
            try:
                track_cache_operation('clear')
                if context_id:
                    prefix = self._get_cache_key('', context_id)[:-1]
                    keys_to_delete = [k for k in list(self.cache.keys()) if k.startswith(prefix)]
                    count = 0
                    for k in keys_to_delete:
                        if k in self.cache:
                            del self.cache[k]
                            count += 1
                    logger.info(f"Cleared {count} items from L1 cache for context '{context_id}'.")
                else:
                    count = len(self.cache)
                    self.cache.clear()
                    logger.info(f'Cleared all {count} items from L1 cache.')
                track_cache_size('memory_manager_l1', len(self.cache))
            except Exception as e:
                logger.error(f'Failed to clear L1 cache: {str(e)}')
                overall_success = False
        if clear_vectors and self.vector_store:
            if context_id:
                logger.debug(f"Clearing vector store for context '{context_id}'...")
                try:
                    vector_success = await self.vector_store.delete_vectors(ids=None, context_id=context_id)
                    if not vector_success:
                        logger.warning(f"Vector store clear operation reported failure for context '{context_id}'.")
                except Exception as e:
                    error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, f"Failed to clear vector store for context '{context_id}'")
                    error.log_error(logger)
                    overall_success = False
            else:
                logger.warning('Skipping vector store clear because context_id is None. Clearing all vectors requires explicit intent.')
        logger.info(f'Memory clearing process finished for {operation_desc} with overall success: {overall_success}')
        return overall_success

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'bulk_load'})
    async def bulk_load(self, keys: List[str], context_id: str, default: Any=None, use_cache: bool=True) -> Dict[str, Any]:
        track_memory_operation('bulk_load')
        logger.debug(f"Bulk loading {len(keys)} keys for context '{context_id}' (use_cache={use_cache})")
        if not keys:
            return {}
        result: Dict[str, Any] = {}
        keys_to_fetch_from_l2: List[str] = []
        cache_hits = 0
        if use_cache:
            track_cache_operation('bulk_check')
            for key in keys:
                cache_key = self._get_cache_key(key, context_id)
                try:
                    cached_value = self.cache.get(cache_key)
                    if cached_value is not None:
                        result[key] = cached_value
                        cache_hits += 1
                    else:
                        keys_to_fetch_from_l2.append(key)
                        track_cache_miss('memory_manager_l1_bulk')
                except Exception as e:
                    logger.warning(f"Error accessing L1 cache for key '{cache_key}' during bulk load: {e}")
                    keys_to_fetch_from_l2.append(key)
                    track_cache_miss('memory_manager_l1_error_bulk')
            if cache_hits > 0:
                for _ in range(cache_hits):
                    track_cache_hit('memory_manager_l1_bulk')
                logger.debug(f'Bulk load: L1 cache hits = {cache_hits}/{len(keys)}')
        else:
            keys_to_fetch_from_l2 = keys
        if keys_to_fetch_from_l2:
            logger.debug(f'Bulk loading {len(keys_to_fetch_from_l2)} keys from primary storage...')
            try:
                fetched_values: Dict[str, Any] = await self.primary.bulk_load(keys_to_fetch_from_l2, context_id, default)
                for key, value in fetched_values.items():
                    result[key] = value
                    if value is not default and use_cache:
                        try:
                            cache_key = self._get_cache_key(key, context_id)
                            self.cache[cache_key] = value
                        except Exception as e:
                            logger.warning(f"Failed to update L1 cache for key '{cache_key}' during bulk load: {e}")
            except Exception as e:
                error = convert_exception(e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to bulk load keys from context '{context_id}'")
                error.log_error(logger)
                for key in keys_to_fetch_from_l2:
                    if key not in result:
                        result[key] = default
        if use_cache:
            track_cache_size('memory_manager_l1', len(self.cache))
        logger.debug(f"Bulk load completed for context '{context_id}'. Returning {len(result)} results.")
        return result

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'bulk_save'})
    async def bulk_save(self, data: Dict[str, Any], context_id: str, ttl: Optional[int]=None, update_cache: bool=True) -> bool:
        track_memory_operation('bulk_save')
        logger.debug(f"Bulk saving {len(data)} keys for context '{context_id}' (update_cache={update_cache})")
        if not data:
            return True
        try:
            effective_ttl = ttl if ttl is not None else self.memory_ttl
            success = await self.primary.bulk_save(data, context_id, effective_ttl)
            if success and update_cache:
                try:
                    logger.debug(f'Updating L1 cache for {len(data)} keys after successful bulk save.')
                    track_cache_operation('bulk_update')
                    for key, value in data.items():
                        cache_key = self._get_cache_key(key, context_id)
                        self.cache[cache_key] = value
                    track_cache_size('memory_manager_l1', len(self.cache))
                except Exception as e:
                    logger.warning(f'Failed to update L1 cache after bulk save: {e}')
            if not success:
                logger.warning(f"Bulk save to primary storage failed for context '{context_id}'.")
            return success
        except Exception as e:
            error = convert_exception(e, ErrorCode.MEMORY_STORAGE_ERROR, f"Failed to bulk save keys to context '{context_id}'")
            error.log_error(logger)
            return False

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'exists'})
    async def exists(self, key: str, context_id: str, check_cache: bool=True) -> bool:
        track_memory_operation('exists')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Checking existence for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (check_cache={check_cache})")
        if check_cache:
            try:
                if cache_key in self.cache:
                    logger.debug(f"Key '{cache_key}' found in L1 cache during existence check.")
                    track_cache_hit('memory_manager_l1_exists')
                    return True
                else:
                    track_cache_miss('memory_manager_l1_exists')
            except Exception as e:
                logger.warning(f"Error checking L1 cache for key '{cache_key}' during existence check: {e}")
        logger.debug(f"Checking existence for key '{key}' in primary storage...")
        try:
            key_exists_in_l2 = await self.primary.exists(key, context_id)
            logger.debug(f"Existence check in primary storage for key '{key}' returned: {key_exists_in_l2}")
            return key_exists_in_l2
        except Exception as e:
            error = convert_exception(e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to check existence for key '{key}' in context '{context_id}'")
            error.log_error(logger)
            return False

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'list_keys'})
    async def list_keys(self, context_id: Optional[str]=None, pattern: Optional[str]=None) -> List[str]:
        track_memory_operation('list_keys')
        logger.debug(f'Listing keys from primary storage (context: {context_id or 'all'}, pattern: {pattern or 'none'})')
        try:
            return await self.primary.list_keys(context_id, pattern)
        except Exception as e:
            error = convert_exception(e, ErrorCode.MEMORY_RETRIEVAL_ERROR, f"Failed to list keys for context '{context_id or 'all'}'")
            error.log_error(logger)
            return []

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'invalidate_cache'})
    async def invalidate_cache(self, key: Optional[str]=None, context_id: Optional[str]=None, pattern: Optional[str]=None) -> int:
        track_memory_operation('invalidate_cache')
        track_cache_operation('invalidate')
        logger.debug(f'Invalidating L1 cache (key: {key}, context: {context_id}, pattern: {pattern})')
        removed_count = 0
        try:
            if key and context_id:
                cache_key = self._get_cache_key(key, context_id)
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    removed_count = 1
                    logger.debug(f'Invalidated single cache key: {cache_key}')
            else:
                keys_to_delete = []
                current_cache_keys = list(self.cache.keys())
                for cache_key in current_cache_keys:
                    should_delete = False
                    parts = cache_key.split(':', 2)
                    if len(parts) == 3:
                        key_context = parts[1]
                        base_key = parts[2]
                        context_match = context_id is None or key_context == context_id
                        pattern_match = pattern is None or self._matches_pattern(base_key, pattern)
                        if context_match and pattern_match:
                            should_delete = True
                    else:
                        logger.warning(f'Unexpected cache key format found during invalidation: {cache_key}')
                    if should_delete:
                        keys_to_delete.append(cache_key)
                if keys_to_delete:
                    logger.debug(f'Invalidating {len(keys_to_delete)} cache keys matching criteria...')
                    for k in keys_to_delete:
                        if k in self.cache:
                            del self.cache[k]
                            removed_count += 1
                    logger.debug(f'Finished invalidating keys.')
            track_cache_size('memory_manager_l1', len(self.cache))
            logger.info(f'Invalidated {removed_count} items from L1 cache.')
            return removed_count
        except Exception as e:
            logger.error(f'Failed to invalidate L1 cache: {str(e)}', exc_info=True)
            return 0

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        if pattern == '*':
            return True
        if '*' not in pattern:
            return text == pattern
        try:
            regex_pattern = pattern.replace('.', '\\.').replace('+', '\\+').replace('?', '\\?').replace('*', '.*')
            return bool(re.fullmatch(regex_pattern, text))
        except re.error as e:
            logger.warning(f"Invalid pattern for matching: '{pattern}'. Error: {e}")
            return False

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'with_cache'})
    async def with_cache(self, func: Callable[..., Coroutine[Any, Any, R]], key: str, context_id: str, ttl: Optional[int]=None, force_refresh: bool=False) -> R:
        track_memory_operation('with_cache')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f'Executing function with cache (key: {cache_key}, force_refresh: {force_refresh})')
        if not force_refresh:
            try:
                cached_value = await asyncio.wait_for(self.load(key, context_id, use_cache=True), timeout=settings.CACHE_LOOKUP_TIMEOUT or 1.0)
                if cached_value is not None:
                    logger.debug(f'Returning cached result for key: {cache_key}')
                    return cast(R, cached_value)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout loading cached value for key '{cache_key}'. Computing fresh result.")
            except Exception as e:
                logger.error(f"Error loading cached value for key '{cache_key}': {str(e)}. Computing fresh result.", exc_info=True)
        lock = await self._get_cache_lock(cache_key)
        execution_timeout = settings.FUNCTION_EXECUTION_TIMEOUT or 30.0
        async with lock:
            if not force_refresh:
                try:
                    cached_value = await asyncio.wait_for(self.load(key, context_id, use_cache=True), timeout=settings.CACHE_LOOKUP_TIMEOUT or 1.0)
                    if cached_value is not None:
                        logger.debug(f"Returning cached result for key '{cache_key}' after acquiring lock.")
                        return cast(R, cached_value)
                except (asyncio.TimeoutError, Exception):
                    pass
            logger.debug(f'Executing function to populate cache for key: {cache_key}')
            try:
                start_time = time.monotonic()
                result: R = await asyncio.wait_for(func(), timeout=execution_timeout)
                exec_duration = time.monotonic() - start_time
                logger.debug(f"Function executed successfully for key '{cache_key}' in {exec_duration:.4f}s")
                if result is not None:
                    try:
                        await asyncio.wait_for(self.save(key, context_id, result, ttl=ttl, update_cache=True), timeout=settings.CACHE_SAVE_TIMEOUT or 2.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout saving result to cache for key '{cache_key}'.")
                    except Exception as save_err:
                        logger.error(f"Error saving result to cache for key '{cache_key}': {save_err}")
                return result
            except asyncio.TimeoutError:
                logger.error(f"Timeout ({execution_timeout}s) executing function for cache key '{cache_key}'")
                return None
            except Exception as func_err:
                logger.error(f"Error executing function for cache key '{cache_key}': {func_err}", exc_info=True)
                return None

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'with_bulk_cache'})
    async def with_bulk_cache(self, func: Callable[[List[str]], Coroutine[Any, Any, Dict[str, R]]], keys: List[str], context_id: str, ttl: Optional[int]=None, force_refresh: bool=False) -> Dict[str, R]:
        track_memory_operation('with_bulk_cache')
        logger.debug(f'Executing function with bulk cache for {len(keys)} keys (context: {context_id}, force_refresh: {force_refresh})')
        if not keys:
            return {}
        results: Dict[str, R] = {}
        keys_to_compute: List[str] = []
        if not force_refresh:
            try:
                cached_values: Dict[str, Any] = await asyncio.wait_for(self.bulk_load(keys, context_id, use_cache=True), timeout=settings.CACHE_LOOKUP_TIMEOUT_BULK or 5.0)
                for key in keys:
                    if key in cached_values and cached_values[key] is not None:
                        results[key] = cast(R, cached_values[key])
                    else:
                        keys_to_compute.append(key)
                logger.debug(f'Bulk cache check: {len(results)} found in cache, {len(keys_to_compute)} need computation.')
            except asyncio.TimeoutError:
                logger.warning(f"Timeout loading bulk cache for context '{context_id}'. Computing all {len(keys)} keys.")
                keys_to_compute = keys
            except Exception as e:
                logger.error(f"Error loading bulk cache for context '{context_id}': {e}. Computing all {len(keys)} keys.", exc_info=True)
                keys_to_compute = keys
        else:
            keys_to_compute = keys
        if not keys_to_compute:
            logger.debug('All requested keys found in cache.')
            return results
        logger.debug(f'Executing function for {len(keys_to_compute)} uncached keys...')
        execution_timeout = settings.FUNCTION_EXECUTION_TIMEOUT_BULK or 60.0
        try:
            computed_values: Dict[str, R] = await asyncio.wait_for(func(keys_to_compute), timeout=execution_timeout)
            logger.debug(f'Function executed, received {len(computed_values)} results.')
            if computed_values:
                to_cache: Dict[str, Any] = {k: v for k, v in computed_values.items() if v is not None}
                if to_cache:
                    try:
                        await asyncio.wait_for(self.bulk_save(to_cache, context_id, ttl=ttl, update_cache=True), timeout=settings.CACHE_SAVE_TIMEOUT_BULK or 5.0)
                    except asyncio.TimeoutError:
                        logger.warning(f'Timeout saving {len(to_cache)} computed values to cache.')
                    except Exception as save_err:
                        logger.error(f'Error saving computed values to cache: {save_err}')
                results.update(computed_values)
        except asyncio.TimeoutError:
            logger.error(f'Timeout ({execution_timeout}s) executing function for {len(keys_to_compute)} keys.')
            for key in keys_to_compute:
                if key not in results:
                    results[key] = None
        except Exception as func_err:
            logger.error(f'Error executing function for keys {keys_to_compute}: {func_err}', exc_info=True)
            for key in keys_to_compute:
                if key not in results:
                    results[key] = None
        return results

    async def store_vector(self, text: str, metadata: Dict[str, Any], context_id: Optional[str]=None) -> Optional[str]:
        if not self.vector_store:
            logger.warning('Vector store is not configured. Cannot store vector.')
            return None
        logger.debug(f"Storing vector in '{type(self.vector_store).__name__}' for context '{context_id or 'global'}'")
        try:
            return await self.vector_store.store_vector(text=text, metadata=metadata, context_id=context_id)
        except Exception as e:
            if not isinstance(e, MemoryError):
                error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, 'Failed to store vector via MemoryManager')
                error.log_error(logger)
            else:
                logger.error(f'Error storing vector via MemoryManager: {e.message}', extra=e.to_dict())
            return None

    async def search_vectors(self, query: str, k: int=5, context_id: Optional[str]=None, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        if not self.vector_store:
            logger.warning('Vector store is not configured. Cannot search vectors.')
            return []
        logger.debug(f"Searching vectors in '{type(self.vector_store).__name__}' for query (length: {len(query)}) in context '{context_id or 'global'}' (k={k})")
        try:
            return await self.vector_store.search_vectors(query=query, k=k, context_id=context_id, filter_metadata=filter_metadata)
        except Exception as e:
            if not isinstance(e, MemoryError):
                error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, 'Failed to search vectors via MemoryManager')
                error.log_error(logger)
            else:
                logger.error(f'Error searching vectors via MemoryManager: {e.message}', extra=e.to_dict())
            return []

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'get_stats'})
    async def get_stats(self) -> Dict[str, Any]:
        logger.debug('Getting memory manager stats...')
        stats = {'manager': {'cache_size': len(self.cache), 'cache_max_size': self.cache.maxsize, 'cache_ttl_seconds': self.cache.ttl, 'cache_current_items': self.cache.currsize}}
        try:
            primary_stats = await self.primary.get_stats()
            stats['primary_storage'] = primary_stats
        except Exception as e:
            logger.error(f'Failed to get primary storage stats: {str(e)}', exc_info=True)
            stats['primary_storage'] = {'status': 'error', 'message': str(e)}
        if self.vector_store:
            try:
                vector_stats = await self.vector_store.get_stats()
                stats['vector_store'] = vector_stats
            except Exception as e:
                logger.error(f'Failed to get vector store stats: {str(e)}', exc_info=True)
                stats['vector_store'] = {'status': 'error', 'message': str(e)}
        else:
            stats['vector_store'] = {'status': 'not_configured'}
        logger.debug(f'Memory manager stats retrieved: {list(stats.keys())}')
        return stats
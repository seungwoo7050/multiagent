"""
Memory manager with caching and orchestration.
Provides a high-level interface to memory and vector storage systems.
"""
import asyncio
import threading
import time
from typing import (Any, Callable, Coroutine, Dict, List, Optional, TypeVar,
                    cast)

from cachetools import TTLCache

from src.config.errors import ErrorCode, MemoryError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import MEMORY_METRICS, get_metrics_manager
from src.config.settings import get_settings
from src.memory.base import BaseMemory, BaseVectorStore
from src.memory.redis_memory import \
    RedisMemory  # backends 폴더가 아니라 src.memory 에서 바로 가져옴
from src.memory.utils import generate_memory_key, matches_pattern

logger = get_logger(__name__)
metrics = get_metrics_manager()
settings = get_settings()

# Type variable for generic return type
R = TypeVar('R')

class MemoryManager:
    """
    Manager for memory storage with caching.
    
    Coordinates between primary storage and optional vector store with an in-memory cache layer.
    """
    def __init__(
        self, 
        primary_memory: BaseMemory, 
        vector_store: Optional[BaseVectorStore]=None, 
        cache_size: int=10000, 
        cache_ttl: int=3600, 
        memory_ttl: int=86400
    ):
        """
        Initialize the memory manager.
        
        Args:
            primary_memory: Primary storage backend
            vector_store: Optional vector storage backend
            cache_size: Maximum number of cache entries
            cache_ttl: Cache TTL in seconds
            memory_ttl: Default memory storage TTL in seconds
        """
        self.primary: BaseMemory = primary_memory
        self.vector_store: Optional[BaseVectorStore] = vector_store
        self.memory_ttl: int = memory_ttl
        self.cache: TTLCache[str, Any] = TTLCache(maxsize=cache_size, ttl=cache_ttl, timer=time.time)
        self._cache_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock: asyncio.Lock = asyncio.Lock()
        self._last_locks_cleanup: float = time.monotonic()
        
        logger.info(
            f'MemoryManager initialized. '
            f'Primary: {type(primary_memory).__name__}, '
            f'Vector Store: {(type(vector_store).__name__ if vector_store else "None")}, '
            f'Cache Size: {cache_size}, '
            f'Cache TTL: {cache_ttl}s, '
            f'Memory TTL: {memory_ttl}s'
        )

    async def _get_cache_lock(self, key: str) -> asyncio.Lock:
        """
        Get a lock for a cache key, creating if necessary.
        
        Args:
            key: Cache key
            
        Returns:
            asyncio.Lock: Lock for the cache key
        """
        async with self._locks_lock:
            if key not in self._cache_locks:
                self._cache_locks[key] = asyncio.Lock()
                logger.debug(f'Created new cache lock for key: {key}')
                
            # Time-based lock cleanup (every 60 seconds)
            current_time = time.monotonic()
            current_lock_count = len(self._cache_locks)
            
            if current_lock_count > 100 and (current_time - self._last_locks_cleanup) > 60:
                self._last_locks_cleanup = current_time
                active_locks = {k: lock for k, lock in self._cache_locks.items() if lock.locked()}
                
                if key in self._cache_locks:
                    active_locks[key] = self._cache_locks[key]
                    
                if len(active_locks) < current_lock_count * 0.7:
                    removed_count = current_lock_count - len(active_locks)
                    logger.debug(f'Cleaning up unused cache locks. Removed: {removed_count}, Remaining: {len(active_locks)}')
                    self._cache_locks = active_locks
                    
            return self._cache_locks[key]

    def _get_cache_key(self, key: str, context_id: str) -> str:
        """
        Generate a cache key.
        
        Args:
            key: Base key
            context_id: Context identifier
            
        Returns:
            str: Formatted cache key
        """
        return generate_memory_key(key, context_id)

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'load'})
    async def load(self, key: str, context_id: str, default: Any=None, use_cache: bool=True) -> Any:
        """
        Load data with caching support.
        
        Args:
            key: The key to load
            context_id: The context identifier
            default: Default value to return if key not found
            use_cache: Whether to use the cache
            
        Returns:
            The stored data or default if not found
        """
        metrics.track_memory('operations', operation_type='load')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Loading data for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (use_cache={use_cache})")
        
        if use_cache:
            metrics.track_cache('operations', operation_type='check')
            try:
                cached_value = self.cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f'L1 Cache hit for key: {cache_key}')
                    metrics.track_cache('hits', cache_type='memory_manager_l1')
                    return cached_value
                else:
                    logger.debug(f'L1 Cache miss or expired for key: {cache_key}')
                    metrics.track_cache('misses', cache_type='memory_manager_l1')
            except Exception as e:
                logger.warning(f"Error accessing L1 cache for key '{cache_key}': {e}")
                metrics.track_cache('misses', cache_type='memory_manager_l1_error')
        
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
                        metrics.track_cache('size', cache_type='memory_manager_l1', value=len(self.cache))
                    except Exception as e:
                        logger.warning(f"Failed to update L1 cache for key '{cache_key}' after load: {e}")
                        
                return value
                
            except Exception as e:
                error = convert_exception(
                    e, 
                    ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                    f"Failed to load key '{key}' from primary storage (context: '{context_id}')"
                )
                error.log_error(logger)
                return default

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'save'})
    async def save(self, key: str, context_id: str, data: Any, ttl: Optional[int]=None, update_cache: bool=True) -> bool:
        """
        Save data with cache update.
        
        Args:
            key: The key to save
            context_id: The context identifier
            data: The data to save
            ttl: Time-to-live in seconds, or None for no expiration
            update_cache: Whether to update the cache
            
        Returns:
            bool: True if save was successful
        """
        metrics.track_memory('operations', operation_type='save')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Saving data for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (update_cache={update_cache})")
        
        try:
            effective_ttl = ttl if ttl is not None else self.memory_ttl
            success = await self.primary.save_context(key, context_id, data, effective_ttl)
            
            if success and update_cache:
                try:
                    logger.debug(f"Updating L1 cache for key '{cache_key}' after successful save.")
                    metrics.track_cache('operations', operation_type='update')
                    self.cache[cache_key] = data
                    metrics.track_cache('size', cache_type='memory_manager_l1', value=len(self.cache))
                except Exception as e:
                    logger.warning(f"Failed to update L1 cache for key '{cache_key}' after save: {e}")
                    
            if not success:
                logger.warning(f"Failed to save key '{key}' to primary storage for context '{context_id}'.")
                
            return success
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f"Failed to save key '{key}' to context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'delete'})
    async def delete(self, key: str, context_id: str, clear_cache: bool=True) -> bool:
        """
        Delete data with cache invalidation.
        
        Args:
            key: The key to delete
            context_id: The context identifier
            clear_cache: Whether to clear from cache
            
        Returns:
            bool: True if deletion was successful
        """
        metrics.track_memory('operations', operation_type='delete')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Deleting data for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (clear_cache={clear_cache})")
        
        try:
            success = await self.primary.delete_context(key, context_id)
            
            if success and clear_cache:
                try:
                    if cache_key in self.cache:
                        del self.cache[cache_key]
                        logger.debug(f"Removed key '{cache_key}' from L1 cache after successful delete.")
                        metrics.track_cache('operations', operation_type='delete')
                        metrics.track_cache('size', cache_type='memory_manager_l1', value=len(self.cache))
                    else:
                        logger.debug(f"Key '{cache_key}' not found in L1 cache during delete.")
                except Exception as e:
                    logger.warning(f"Failed to remove key '{cache_key}' from L1 cache after delete: {e}")
                    
            if not success:
                logger.warning(f"Failed to delete key '{key}' from primary storage for context '{context_id}'.")
                
            return success
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f"Failed to delete key '{key}' from context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'clear'})
    async def clear(self, context_id: Optional[str]=None, clear_cache: bool=True, clear_vectors: bool=True) -> bool:
        """
        Clear all data for a context, or all contexts if none specified.
        
        Args:
            context_id: The context identifier, or None for all contexts
            clear_cache: Whether to clear from cache
            clear_vectors: Whether to clear from vector store
            
        Returns:
            bool: True if clearing was successful
        """
        metrics.track_memory('operations', operation_type='clear')
        operation_desc = f"context '{context_id}'" if context_id else 'all contexts'
        logger.info(f'Clearing memory for {operation_desc} (clear_cache={clear_cache}, clear_vectors={clear_vectors})')
        
        overall_success = True
        
        # Clear primary storage
        try:
            l2_success = await self.primary.clear(context_id)
            if not l2_success:
                logger.warning(f'Primary storage clear operation reported failure for {operation_desc}.')
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f'Failed to clear primary storage for {operation_desc}'
            )
            error.log_error(logger)
            overall_success = False
        
        # Clear cache if requested
        if clear_cache:
            try:
                metrics.track_cache('operations', operation_type='clear')
                
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
                    
                metrics.track_cache('size', cache_type='memory_manager_l1', value=len(self.cache))
                
            except Exception as e:
                logger.error(f'Failed to clear L1 cache: {str(e)}')
                overall_success = False
        
        # Clear vector store if requested
        if clear_vectors and self.vector_store:
            if context_id:
                logger.debug(f"Clearing vector store for context '{context_id}'...")
                try:
                    vector_success = await self.vector_store.delete_vectors(ids=None, context_id=context_id)
                    if not vector_success:
                        logger.warning(f"Vector store clear operation reported failure for context '{context_id}'.")
                except Exception as e:
                    error = convert_exception(
                        e, 
                        ErrorCode.VECTOR_DB_ERROR, 
                        f"Failed to clear vector store for context '{context_id}'"
                    )
                    error.log_error(logger)
                    overall_success = False
            else:
                logger.warning('Skipping vector store clear because context_id is None. Clearing all vectors requires explicit intent.')
                
        logger.info(f'Memory clearing process finished for {operation_desc} with overall success: {overall_success}')
        return overall_success

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'bulk_load'})
    async def bulk_load(self, keys: List[str], context_id: str, default: Any=None, use_cache: bool=True) -> Dict[str, Any]:
        """
        Load multiple keys at once with caching.
        
        Args:
            keys: List of keys to load
            context_id: The context identifier
            default: Default value for keys not found
            use_cache: Whether to use the cache
            
        Returns:
            Dict[str, Any]: Dictionary of key-value pairs
        """
        metrics.track_memory('operations', operation_type='bulk_load')
        logger.debug(f"Bulk loading {len(keys)} keys for context '{context_id}' (use_cache={use_cache})")
        
        if not keys:
            return {}
            
        result: Dict[str, Any] = {}
        keys_to_fetch_from_l2: List[str] = []
        cache_hits = 0
        
        # Check cache first if requested
        if use_cache:
            metrics.track_cache('operations', operation_type='bulk_check')
            
            for key in keys:
                cache_key = self._get_cache_key(key, context_id)
                try:
                    cached_value = self.cache.get(cache_key)
                    if cached_value is not None:
                        result[key] = cached_value
                        cache_hits += 1
                    else:
                        keys_to_fetch_from_l2.append(key)
                        metrics.track_cache('misses', cache_type='memory_manager_l1_bulk')
                except Exception as e:
                    logger.warning(f"Error accessing L1 cache for key '{cache_key}' during bulk load: {e}")
                    keys_to_fetch_from_l2.append(key)
                    metrics.track_cache('misses', cache_type='memory_manager_l1_error_bulk')
                    
            if cache_hits > 0:
                for _ in range(cache_hits):
                    metrics.track_cache('hits', cache_type='memory_manager_l1_bulk')
                logger.debug(f'Bulk load: L1 cache hits = {cache_hits}/{len(keys)}')
        else:
            keys_to_fetch_from_l2 = keys
        
        # Fetch remaining keys from primary storage
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
                error = convert_exception(
                    e, 
                    ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                    f"Failed to bulk load keys from context '{context_id}'"
                )
                error.log_error(logger)
                
                for key in keys_to_fetch_from_l2:
                    if key not in result:
                        result[key] = default
        
        if use_cache:
            metrics.track_cache('size', cache_type='memory_manager_l1', value=len(self.cache))
            
        logger.debug(f"Bulk load completed for context '{context_id}'. Returning {len(result)} results.")
        return result

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'bulk_save'})
    async def bulk_save(self, data: Dict[str, Any], context_id: str, ttl: Optional[int]=None, update_cache: bool=True) -> bool:
        """
        Save multiple key-value pairs at once with cache update.
        
        Args:
            data: Dictionary of key-value pairs to save
            context_id: The context identifier
            ttl: Time-to-live in seconds, or None for no expiration
            update_cache: Whether to update the cache
            
        Returns:
            bool: True if all saves were successful
        """
        metrics.track_memory('operations', operation_type='bulk_save')
        logger.debug(f"Bulk saving {len(data)} keys for context '{context_id}' (update_cache={update_cache})")
        
        if not data:
            return True
            
        try:
            effective_ttl = ttl if ttl is not None else self.memory_ttl
            success = await self.primary.bulk_save(data, context_id, effective_ttl)
            
            if success and update_cache:
                try:
                    logger.debug(f'Updating L1 cache for {len(data)} keys after successful bulk save.')
                    metrics.track_cache('operations', operation_type='bulk_update')
                    
                    for key, value in data.items():
                        cache_key = self._get_cache_key(key, context_id)
                        self.cache[cache_key] = value
                        
                    metrics.track_cache('size', cache_type='memory_manager_l1', value=len(self.cache))
                    
                except Exception as e:
                    logger.warning(f'Failed to update L1 cache after bulk save: {e}')
                    
            if not success:
                logger.warning(f"Bulk save to primary storage failed for context '{context_id}'.")
                
            return success
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f"Failed to bulk save keys to context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'exists'})
    async def exists(self, key: str, context_id: str, check_cache: bool=True) -> bool:
        """
        Check if a key exists in memory.
        
        Args:
            key: The key to check
            context_id: The context identifier
            check_cache: Whether to check the cache
            
        Returns:
            bool: True if key exists
        """
        metrics.track_memory('operations', operation_type='exists')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f"Checking existence for key: '{key}', context: '{context_id}', cache_key: '{cache_key}' (check_cache={check_cache})")
        
        if check_cache:
            try:
                if cache_key in self.cache:
                    logger.debug(f"Key '{cache_key}' found in L1 cache during existence check.")
                    metrics.track_cache('hits', cache_type='memory_manager_l1_exists')
                    return True
                else:
                    metrics.track_cache('misses', cache_type='memory_manager_l1_exists')
            except Exception as e:
                logger.warning(f"Error checking L1 cache for key '{cache_key}' during existence check: {e}")
        
        logger.debug(f"Checking existence for key '{key}' in primary storage...")
        try:
            key_exists_in_l2 = await self.primary.exists(key, context_id)
            logger.debug(f"Existence check in primary storage for key '{key}' returned: {key_exists_in_l2}")
            return key_exists_in_l2
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                f"Failed to check existence for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'list_keys'})
    async def list_keys(self, context_id: Optional[str]=None, pattern: Optional[str]=None) -> List[str]:
        """
        List keys matching pattern within a context.
        
        Args:
            context_id: The context identifier, or None for all contexts
            pattern: Optional pattern for filtering keys
            
        Returns:
            List[str]: List of matching keys
        """
        metrics.track_memory('operations', operation_type='list_keys')
        logger.debug(f'Listing keys from primary storage (context: {context_id or "all"}, pattern: {pattern or "none"})')
        
        try:
            return await self.primary.list_keys(context_id, pattern)
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                f"Failed to list keys for context '{context_id or 'all'}'"
            )
            error.log_error(logger)
            return []

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'invalidate_cache'})
    async def invalidate_cache(self, key: Optional[str]=None, context_id: Optional[str]=None, pattern: Optional[str]=None) -> int:
        """
        Invalidate cache entries matching criteria.
        
        Args:
            key: Optional specific key to invalidate
            context_id: Optional context identifier
            pattern: Optional pattern for filtering keys
            
        Returns:
            int: Number of invalidated entries
        """
        metrics.track_memory('operations', operation_type='invalidate_cache')
        metrics.track_cache('operations', operation_type='invalidate')
        logger.debug(f'Invalidating L1 cache (key: {key}, context: {context_id}, pattern: {pattern})')
        
        removed_count = 0
        try:
            if key and context_id:
                # Invalidate specific key
                cache_key = self._get_cache_key(key, context_id)
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    removed_count = 1
                    logger.debug(f'Invalidated single cache key: {cache_key}')
            else:
                # Invalidate by pattern
                keys_to_delete = []
                current_cache_keys = list(self.cache.keys())
                
                for cache_key in current_cache_keys:
                    should_delete = False
                    parts = cache_key.split(':', 2)
                    
                    if len(parts) == 3:
                        key_context = parts[1]
                        base_key = parts[2]
                        context_match = context_id is None or key_context == context_id
                        pattern_match = pattern is None or matches_pattern(base_key, pattern)
                        
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
                    logger.debug('Finished invalidating keys.')
                    
            metrics.track_cache('size', cache_type='memory_manager_l1', value=len(self.cache))
            logger.info(f'Invalidated {removed_count} items from L1 cache.')
            return removed_count
            
        except Exception as e:
            logger.error(f'Failed to invalidate L1 cache: {str(e)}', exc_info=True)
            return 0

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'with_cache'})
    async def with_cache(self, func: Callable[..., Coroutine[Any, Any, R]], key: str, context_id: str, ttl: Optional[int]=None, force_refresh: bool=False) -> R:
        """
        Execute a function with caching of the result.
        
        Args:
            func: Async function to execute
            key: Cache key
            context_id: Context identifier
            ttl: Time-to-live in seconds, or None for default
            force_refresh: Whether to force a cache refresh
            
        Returns:
            R: Function result
        """
        metrics.track_memory('operations', operation_type='with_cache')
        cache_key = self._get_cache_key(key, context_id)
        logger.debug(f'Executing function with cache (key: {cache_key}, force_refresh: {force_refresh})')
        
        if not force_refresh:
            try:
                cached_value = await asyncio.wait_for(
                    self.load(key, context_id, use_cache=True), 
                    timeout=settings.CACHE_LOOKUP_TIMEOUT or 1.0
                )
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
                    cached_value = await asyncio.wait_for(
                        self.load(key, context_id, use_cache=True), 
                        timeout=settings.CACHE_LOOKUP_TIMEOUT or 1.0
                    )
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
                        await asyncio.wait_for(
                            self.save(key, context_id, result, ttl=ttl, update_cache=True), 
                            timeout=settings.CACHE_SAVE_TIMEOUT or 2.0
                        )
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

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'with_bulk_cache'})
    async def with_bulk_cache(self, func: Callable[[List[str]], Coroutine[Any, Any, Dict[str, R]]], keys: List[str], context_id: str, ttl: Optional[int]=None, force_refresh: bool=False) -> Dict[str, R]:
        """
        Execute a function with bulk caching of the results.
        
        Args:
            func: Async function taking list of keys, returning dict of results
            keys: List of keys to process
            context_id: Context identifier
            ttl: Time-to-live in seconds, or None for default
            force_refresh: Whether to force a cache refresh
            
        Returns:
            Dict[str, R]: Dictionary of function results by key
        """
        metrics.track_memory('operations', operation_type='with_bulk_cache')
        logger.debug(f'Executing function with bulk cache for {len(keys)} keys (context: {context_id}, force_refresh: {force_refresh})')
        
        if not keys:
            return {}
            
        results: Dict[str, R] = {}
        keys_to_compute: List[str] = []
        
        # Load from cache if not forcing refresh
        if not force_refresh:
            try:
                cached_values: Dict[str, Any] = await asyncio.wait_for(
                    self.bulk_load(keys, context_id, use_cache=True), 
                    timeout=settings.CACHE_LOOKUP_TIMEOUT_BULK or 5.0
                )
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
        
        # Return early if all keys found in cache
        if not keys_to_compute:
            logger.debug('All requested keys found in cache.')
            return results
        
        # Compute uncached values
        logger.debug(f'Executing function for {len(keys_to_compute)} uncached keys...')
        execution_timeout = settings.FUNCTION_EXECUTION_TIMEOUT_BULK or 60.0
        
        try:
            computed_values: Dict[str, R] = await asyncio.wait_for(func(keys_to_compute), timeout=execution_timeout)
            logger.debug(f'Function executed, received {len(computed_values)} results.')
            
            if computed_values:
                to_cache: Dict[str, Any] = {k: v for k, v in computed_values.items() if v is not None}
                if to_cache:
                    try:
                        await asyncio.wait_for(
                            self.bulk_save(to_cache, context_id, ttl=ttl, update_cache=True), 
                            timeout=settings.CACHE_SAVE_TIMEOUT_BULK or 5.0
                        )
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
        """
        Store vector embedding with text and metadata.
        
        Args:
            text: Text to convert to vector embedding
            metadata: Associated metadata
            context_id: Optional context identifier
            
        Returns:
            Optional[str]: Vector ID if successful, None otherwise
        """
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
        """
        Search for vector embeddings similar to query.
        
        Args:
            query: Query text
            k: Number of results to return
            context_id: Optional context identifier
            filter_metadata: Optional filter to apply to metadata
            
        Returns:
            List[Dict[str, Any]]: Search results with similarity scores
        """
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

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'get_stats'})
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        logger.debug('Getting memory manager stats...')
        
        stats = {
            'manager': {
                'cache_size': len(self.cache),
                'cache_max_size': self.cache.maxsize,
                'cache_ttl_seconds': self.cache.ttl,
                'cache_current_items': self.cache.currsize
            }
        }
        
        # Get primary storage stats
        try:
            primary_stats = await self.primary.get_stats()
            stats['primary_storage'] = primary_stats
        except Exception as e:
            logger.error(f'Failed to get primary storage stats: {str(e)}', exc_info=True)
            stats['primary_storage'] = {'status': 'error', 'message': str(e)}
        
        # Get vector store stats if available
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
    
# manager.py 파일의 get_memory_manager 함수 수정



# --- 싱글톤 인스턴스 관리 (이전과 동일) ---
_memory_manager_instance: Optional['MemoryManager'] = None
_memory_manager_lock = threading.Lock()

async def get_memory_manager() -> 'MemoryManager':
    """
    MemoryManager 싱글톤 인스턴스를 가져옵니다.
    첫 호출 시 설정을 기반으로 Redis 및 Vector Store 백엔드를 사용하여 인스턴스를 생성합니다.
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        with _memory_manager_lock:
            if _memory_manager_instance is None:
                logger.info("Initializing MemoryManager instance...")

                # 1. Redis 주 메모리 백엔드 초기화 (찾은 파일 사용!)
                try:
                    logger.info("Initializing Redis primary memory backend using RedisMemory class...")
                    # RedisMemory 클래스를 직접 사용합니다.
                    # __init__ 메서드가 default_ttl만 받으므로, Redis 연결은 클래스 내부에서 처리하는 방식입니다.
                    primary_backend = RedisMemory(default_ttl=settings.MEMORY_TTL)
                    # 내부적으로 settings.REDIS_HOST 등을 사용해서 연결할 것으로 예상됩니다.
                    # 만약 RedisMemory 초기화 시 추가 인자가 필요하다면 여기에 전달해야 합니다.
                    logger.info("Redis primary memory backend initialized.")

                except Exception as e:
                    logger.error(f"Failed to initialize Redis primary memory backend: {e}", exc_info=True)
                    raise RuntimeError("Failed to initialize Redis backend") from e

                # 2. 벡터 저장소 백엔드 초기화 (수정됨)
                vector_backend: Optional[BaseVectorStore] = None
                # settings.py의 VECTOR_DB_TYPE이 'none'이 아닐 때만 초기화 시도
                if settings.VECTOR_DB_TYPE != 'none':
                    try:
                        vector_store_type = settings.VECTOR_DB_TYPE.lower() # 설정 이름 수정
                        logger.info(f"Initializing vector store backend: {vector_store_type}")

                        # --- BaseVectorStore를 사용하여 초기화 ---
                        # BaseVectorStore 클래스가 백엔드 타입과 설정을 받아
                        # 내부적으로 적절한 함수(backends 폴더의 함수들)를 연결한다고 가정합니다.

                        # BaseVectorStore 초기화 시 필요한 설정값들을 settings에서 가져옵니다.
                        # BaseVectorStore의 __init__이 어떻게 구현되었는지 확인 필요!
                        # 여기서는 backend_type과 api_url을 받는다고 가정합니다.
                        init_kwargs = {
                            "backend_type": vector_store_type,
                            "api_url": settings.VECTOR_DB_URL, # 공통 URL 설정 사용
                            # 필요한 경우 다른 설정 추가 (예: settings.FAISS_DIRECTORY 등)
                            # "faiss_directory": settings.FAISS_DIRECTORY # FAISS 사용 시 예시
                        }
                        # api_key가 필요한 경우, 환경 변수나 settings에서 가져와 전달
                        # if vector_store_type == 'qdrant' and settings.QDRANT_API_KEY:
                        #    init_kwargs['api_key'] = settings.QDRANT_API_KEY

                        # **주의:** BaseVectorStore가 실제로 어떤 인자를 받는지 확인하고 맞춰야 합니다.
                        vector_backend = BaseVectorStore(**init_kwargs)

                        # BaseVectorStore 초기화 후, 내부적으로 backends/__init__.py의
                        # register_backends와 유사한 로직이 실행될 것으로 기대합니다.
                        # 또는 명시적으로 호출해야 할 수도 있습니다:
                        # from src.memory.backends import register_backends
                        # register_backends(BaseVectorStore) # 클래스 자체에 등록하는 방식일 경우

                        logger.info(f"{vector_store_type.capitalize()} vector store backend initialized using BaseVectorStore.")

                    except ImportError as e:
                         logger.error(f"Failed to import BaseVectorStore or its dependencies: {e}", exc_info=True)
                         vector_backend = None
                    except AttributeError as e:
                        logger.error(f"Vector store configuration missing or invalid in settings: {e}. Vector store disabled.")
                        vector_backend = None
                    except Exception as e:
                        logger.error(f"Failed to initialize vector store backend: {e}", exc_info=True)
                        vector_backend = None # 실패 시 비활성화
                else:
                    logger.info("Vector store is disabled (VECTOR_DB_TYPE is 'none').")

                # 3. MemoryManager 인스턴스 생성 (이전과 거의 동일)
                _memory_manager_instance = MemoryManager(
                    primary_memory=primary_backend,
                    vector_store=vector_backend, # <- 여기에 BaseVectorStore 인스턴스 전달
                    cache_size=settings.MEMORY_MANAGER_CACHE_SIZE,
                    cache_ttl=settings.CACHE_TTL,
                    memory_ttl=settings.MEMORY_TTL
                )
                logger.info(f"MemoryManager instance created. Primary: {type(primary_backend).__name__}, Vector Store: {type(vector_backend).__name__ if vector_backend else 'None'}")

    if _memory_manager_instance is None:
         raise RuntimeError("MemoryManager instance could not be created.")
    return _memory_manager_instance
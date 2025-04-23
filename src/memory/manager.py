"""Memory manager with two-level caching for optimized access."""

import asyncio
import functools
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from cachetools import LRUCache, TTLCache

from src.config.errors import ErrorCode, MemoryError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import (
    MEMORY_OPERATION_DURATION,
    timed_metric,
    track_cache_hit,
    track_cache_miss,
    track_cache_operation,
    track_cache_size,
    track_memory_operation,
    track_memory_operation_completed,
)
from src.config.settings import get_settings
from src.memory.base import BaseMemory, BaseVectorStore
from src.memory.utils import compute_fingerprint, generate_memory_key
from src.utils.timing import async_timed

logger = get_logger(__name__)
settings = get_settings()


class MemoryManager:
    """High-performance memory manager with two-level caching.
    
    Features:
    - In-memory LRU cache for fastest access
    - Redis-based distributed cache
    - Optional vector store integration
    - Efficient bulk operations
    """
    
    def __init__(
        self,
        primary_memory: BaseMemory,
        vector_store: Optional[BaseVectorStore] = None,
        cache_size: int = 10000,
        cache_ttl: int = 3600,
        memory_ttl: int = 86400,
    ):
        """Initialize memory manager.
        
        Args:
            primary_memory: Primary storage backend
            vector_store: Optional vector store
            cache_size: Maximum number of items in local cache
            cache_ttl: Cache time-to-live in seconds
            memory_ttl: Default memory TTL in seconds
        """
        self.primary = primary_memory
        self.vector_store = vector_store
        self.memory_ttl = memory_ttl
        
        # Initialize in-memory LRU cache with TTL
        self.cache = TTLCache(
            maxsize=cache_size,
            ttl=cache_ttl,
            timer=time.time
        )
        
        # Cache lock to prevent thundering herd problem
        self._cache_locks = {}
        self._locks_lock = asyncio.Lock()
    
    async def _get_cache_lock(self, key: str) -> asyncio.Lock:
        """Get a lock for a specific cache key."""
        # Clean up expired locks to prevent memory leak
        try:
            async with self._locks_lock:
                # Create new lock if needed
                if key not in self._cache_locks:
                    self._cache_locks[key] = asyncio.Lock()
                
                # Clean up locks dictionary periodically (every ~100 calls)
                if len(self._cache_locks) > 100 and random.random() < 0.01:
                    # Only keep locks that are in use
                    active_locks = {k: lock for k, lock in list(self._cache_locks.items()) 
                                   if lock.locked()}
                    # Keep the current key's lock
                    if key in active_locks:
                        active_locks[key] = self._cache_locks[key]
                    # Replace with filtered dictionary if it reduces size significantly
                    if len(active_locks) < len(self._cache_locks) * 0.7:
                        self._cache_locks = active_locks
                
                return self._cache_locks[key]
        except Exception as e:
            # Fallback to a new lock if there's any error
            logger.error(f"Error getting cache lock: {str(e)}")
            return asyncio.Lock()
    
    def _get_cache_key(self, key: str, context_id: str) -> str:
        """Generate a cache key."""
        return generate_memory_key(key, context_id)
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "load"})
    async def load(
        self,
        key: str,
        context_id: str,
        default: Any = None,
        use_cache: bool = True,
    ) -> Any:
        """Load data from memory with caching.
        
        Args:
            key: The key to load
            context_id: The conversation or session ID
            default: Default value if not found
            use_cache: Whether to use in-memory cache
            
        Returns:
            The loaded data or default
        """
        track_memory_operation("load")
        cache_key = self._get_cache_key(key, context_id)
        
        # Check in-memory cache first if enabled
        if use_cache:
            track_cache_operation("check")
            
            try:
                cached_value = self.cache.get(cache_key)
                if cached_value is not None:
                    track_cache_hit("memory_manager")
                    return cached_value
            except Exception as e:
                logger.debug(f"Cache access error (non-critical): {str(e)}")
                
            track_cache_miss("memory_manager")
        
        # Get a lock for this key to prevent redundant loads
        lock = await self._get_cache_lock(cache_key)
        
        # Double-checked locking pattern to prevent redundant loads
        async with lock:
            # Check cache again inside the lock
            if use_cache:
                try:
                    cached_value = self.cache.get(cache_key)
                    if cached_value is not None:
                        track_cache_hit("memory_manager")
                        return cached_value
                except Exception:
                    pass
            
            # Load from primary storage
            try:
                value = await self.primary.load_context(key, context_id, default)
                
                # Update cache if value was found
                if value is not default and use_cache:
                    try:
                        self.cache[cache_key] = value
                        track_cache_size("memory_manager", len(self.cache))
                    except Exception as e:
                        logger.debug(f"Cache update error (non-critical): {str(e)}")
                
                return value
                
            except Exception as e:
                error = convert_exception(
                    e,
                    ErrorCode.MEMORY_RETRIEVAL_ERROR,
                    f"Failed to load key {key} from context {context_id}"
                )
                error.log_error(logger)
                return default
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "save"})
    async def save(
        self,
        key: str,
        context_id: str,
        data: Any,
        ttl: Optional[int] = None,
        update_cache: bool = True,
    ) -> bool:
        """Save data to memory with caching.
        
        Args:
            key: The key to save under
            context_id: The conversation or session ID
            data: The data to save
            ttl: Optional TTL override
            update_cache: Whether to update in-memory cache
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("save")
        cache_key = self._get_cache_key(key, context_id)
        
        try:
            # Save to primary storage
            success = await self.primary.save_context(
                key, context_id, data, ttl or self.memory_ttl
            )
            
            # Update cache if successful
            if success and update_cache:
                try:
                    track_cache_operation("update")
                    self.cache[cache_key] = data
                    track_cache_size("memory_manager", len(self.cache))
                except Exception as e:
                    logger.debug(f"Cache update error (non-critical): {str(e)}")
            
            return success
            
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to save key {key} to context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "delete"})
    async def delete(
        self,
        key: str,
        context_id: str,
        clear_cache: bool = True,
    ) -> bool:
        """Delete data from memory.
        
        Args:
            key: The key to delete
            context_id: The conversation or session ID
            clear_cache: Whether to clear cache entry
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("delete")
        cache_key = self._get_cache_key(key, context_id)
        
        try:
            # Delete from primary storage
            success = await self.primary.delete_context(key, context_id)
            
            # Remove from cache if successful
            if success and clear_cache:
                try:
                    track_cache_operation("delete")
                    if cache_key in self.cache:
                        del self.cache[cache_key]
                        track_cache_size("memory_manager", len(self.cache))
                except Exception as e:
                    logger.debug(f"Cache delete error (non-critical): {str(e)}")
            
            return success
            
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to delete key {key} from context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "clear"})
    async def clear(
        self,
        context_id: Optional[str] = None,
        clear_cache: bool = True,
        clear_vectors: bool = True,
    ) -> bool:
        """Clear memory contexts.
        
        Args:
            context_id: Optional context to clear (None for all)
            clear_cache: Whether to clear in-memory cache
            clear_vectors: Whether to clear vector store
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("clear")
        success = True
        
        # Clear from primary storage
        try:
            success = await self.primary.clear(context_id)
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to clear context {context_id or 'all'}"
            )
            error.log_error(logger)
            success = False
        
        # Clear cache if requested
        if clear_cache:
            try:
                track_cache_operation("clear")
                
                if context_id:
                    # Clear only entries for this context
                    prefix = f"memory:{context_id}:"
                    keys_to_delete = [k for k in self.cache if k.startswith(prefix)]
                    
                    for k in keys_to_delete:
                        del self.cache[k]
                else:
                    # Clear entire cache
                    self.cache.clear()
                    
                track_cache_size("memory_manager", len(self.cache))
                
            except Exception as e:
                logger.error(f"Failed to clear cache: {str(e)}")
                success = False
        
        # Clear vector store if requested and available
        if clear_vectors and self.vector_store and context_id:
            try:
                await self.vector_store.delete_vectors(
                    ids=None, context_id=context_id
                )
            except Exception as e:
                logger.error(f"Failed to clear vector store: {str(e)}")
                success = False
        
        return success
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "bulk_load"})
    async def bulk_load(
        self,
        keys: List[str],
        context_id: str,
        default: Any = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Load multiple values efficiently.
        
        Args:
            keys: List of keys to load
            context_id: The conversation or session ID
            default: Default value for missing keys
            use_cache: Whether to use in-memory cache
            
        Returns:
            Dictionary of {key: value} pairs
        """
        track_memory_operation("bulk_load")
        
        if not keys:
            return {}
        
        # Prepare results and track which keys to fetch from storage
        result = {}
        keys_to_fetch = []
        cache_hits = 0
        
        # Check cache first for each key
        if use_cache:
            for key in keys:
                cache_key = self._get_cache_key(key, context_id)
                try:
                    cached_value = self.cache.get(cache_key)
                    if cached_value is not None:
                        result[key] = cached_value
                        cache_hits += 1
                    else:
                        keys_to_fetch.append(key)
                except Exception:
                    keys_to_fetch.append(key)
        else:
            keys_to_fetch = keys
        
        # Track cache hit/miss metrics
        if use_cache and keys:
            track_cache_operation("bulk_check")
            if cache_hits > 0:
                # Call track_cache_hit once for each hit
                for _ in range(cache_hits):
                    track_cache_hit("memory_manager")
            if keys_to_fetch:
                # Call track_cache_miss once for each miss
                for _ in range(len(keys_to_fetch)):
                    track_cache_miss("memory_manager")
        
        # Fetch remaining keys from storage
        if keys_to_fetch:
            try:
                fetched_values = await self.primary.bulk_load(
                    keys_to_fetch, context_id, default
                )
                
                # Update result and cache
                for key, value in fetched_values.items():
                    result[key] = value
                    
                    # Update cache if value was found
                    if value is not default and use_cache:
                        try:
                            cache_key = self._get_cache_key(key, context_id)
                            self.cache[cache_key] = value
                        except Exception as e:
                            logger.debug(f"Cache update error (non-critical): {str(e)}")
                
            except Exception as e:
                error = convert_exception(
                    e,
                    ErrorCode.MEMORY_RETRIEVAL_ERROR,
                    f"Failed to bulk load keys from context {context_id}"
                )
                error.log_error(logger)
                
                # Return defaults for any keys not already in result
                for key in keys_to_fetch:
                    if key not in result:
                        result[key] = default
        
        # Update cache size metric
        if use_cache:
            track_cache_size("memory_manager", len(self.cache))
        
        return result
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "bulk_save"})
    async def bulk_save(
        self,
        data: Dict[str, Any],
        context_id: str,
        ttl: Optional[int] = None,
        update_cache: bool = True,
    ) -> bool:
        """Save multiple values efficiently.
        
        Args:
            data: Dictionary of {key: value} pairs to save
            context_id: The conversation or session ID
            ttl: Optional TTL override
            update_cache: Whether to update in-memory cache
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("bulk_save")
        
        if not data:
            return True
        
        try:
            # Save to primary storage
            success = await self.primary.bulk_save(
                data, context_id, ttl or self.memory_ttl
            )
            
            # Update cache if successful
            if success and update_cache:
                try:
                    track_cache_operation("bulk_update")
                    for key, value in data.items():
                        cache_key = self._get_cache_key(key, context_id)
                        self.cache[cache_key] = value
                    track_cache_size("memory_manager", len(self.cache))
                except Exception as e:
                    logger.debug(f"Cache update error (non-critical): {str(e)}")
            
            return success
            
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to bulk save keys to context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "exists"})
    async def exists(
        self,
        key: str,
        context_id: str,
        check_cache: bool = True,
    ) -> bool:
        """Check if key exists in memory.
        
        Args:
            key: The key to check
            context_id: The conversation or session ID
            check_cache: Whether to check in-memory cache
            
        Returns:
            True if key exists, False otherwise
        """
        track_memory_operation("exists")
        
        # Check cache first if enabled
        if check_cache:
            cache_key = self._get_cache_key(key, context_id)
            try:
                if cache_key in self.cache:
                    return True
            except Exception:
                pass
        
        # Check primary storage
        try:
            return await self.primary.exists(key, context_id)
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to check existence for key {key} in context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "list_keys"})
    async def list_keys(
        self,
        context_id: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> List[str]:
        """List keys in memory.
        
        Args:
            context_id: Optional context to filter by
            pattern: Optional pattern to filter by
            
        Returns:
            List of keys
        """
        track_memory_operation("list_keys")
        
        try:
            return await self.primary.list_keys(context_id, pattern)
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to list keys for context {context_id or 'all'}"
            )
            error.log_error(logger)
            return []
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "invalidate_cache"})
    async def invalidate_cache(
        self,
        key: Optional[str] = None,
        context_id: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries.
        
        Args:
            key: Specific key to invalidate
            context_id: Optional context to filter by
            pattern: Optional pattern to filter by
            
        Returns:
            Number of invalidated entries
        """
        track_memory_operation("invalidate_cache")
        track_cache_operation("invalidate")
        
        try:
            # Handle specific key
            if key and context_id:
                cache_key = self._get_cache_key(key, context_id)
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    track_cache_size("memory_manager", len(self.cache))
                    return 1
                return 0
            
            # Handle pattern-based invalidation
            keys_to_delete = []
            
            if context_id:
                # Filter by context ID
                prefix = f"memory:{context_id}:"
                if pattern:
                    # Both context and pattern
                    full_pattern = f"{prefix}{pattern}"
                    keys_to_delete = [
                        k for k in self.cache 
                        if k.startswith(prefix) and self._matches_pattern(k, full_pattern)
                    ]
                else:
                    # Just context
                    keys_to_delete = [k for k in self.cache if k.startswith(prefix)]
            elif pattern:
                # Just pattern (across all contexts)
                keys_to_delete = [
                    k for k in self.cache 
                    if self._matches_pattern(k.split(":", 2)[2] if ":" in k else k, pattern)
                ]
            else:
                # Neither - invalidate everything
                count = len(self.cache)
                self.cache.clear()
                track_cache_size("memory_manager", 0)
                return count
            
            # Delete selected keys
            for k in keys_to_delete:
                del self.cache[k]
                
            track_cache_size("memory_manager", len(self.cache))
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {str(e)}")
            return 0
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if a key matches a pattern.
        
        Simple pattern matching with * wildcard support.
        """
        if pattern == "*":
            return True
            
        if "*" not in pattern:
            return key == pattern
            
        parts = pattern.split("*")
        if not parts:
            return True
            
        if pattern.startswith("*") and pattern.endswith("*"):
            # *middle*
            return parts[1] in key
        elif pattern.startswith("*"):
            # *end
            return key.endswith(parts[1])
        elif pattern.endswith("*"):
            # start*
            return key.startswith(parts[0])
        else:
            # start*middle*end
            if not key.startswith(parts[0]):
                return False
                
            key = key[len(parts[0]):]
            for part in parts[1:-1]:
                if part not in key:
                    return False
                key = key[key.index(part) + len(part):]
                
            return key.endswith(parts[-1])
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "with_cache"})
    async def with_cache(
        self,
        func,
        key: str,
        context_id: str,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Any:
        """Execute function with automatic caching.
        
        Args:
            func: Async function to execute and cache result
            key: Cache key
            context_id: The conversation or session ID
            ttl: Optional TTL override
            force_refresh: Force execution even if cached
            
        Returns:
            Function result (from cache or fresh execution)
        """
        track_memory_operation("with_cache")
        
        # Try to load from cache first unless force_refresh
        if not force_refresh:
            try:
                cached_value = await asyncio.wait_for(
                    self.load(key, context_id),
                    timeout=3.0  # 3-second timeout
                )
                if cached_value is not None:
                    return cached_value
            except (asyncio.TimeoutError, Exception) as e:
                if isinstance(e, asyncio.TimeoutError):
                    logger.warning(f"Timeout loading cached value for {key}, computing fresh")
                else:
                    logger.error(f"Error loading cached value: {str(e)}, computing fresh")
        
        # Get lock to prevent duplicate computation
        cache_key = self._get_cache_key(key, context_id)
        
        try:
            # Get lock with timeout to prevent deadlock
            lock = await asyncio.wait_for(
                self._get_cache_lock(cache_key),
                timeout=2.0  # 2-second timeout for getting lock
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting lock for {key}, executing without lock")
            # Execute without lock as a fallback
            result = await func()
            if result is not None:
                try:
                    await self.save(key, context_id, result, ttl)
                except Exception as e:
                    logger.error(f"Error saving result to cache: {str(e)}")
            return result
        
        try:
            async with lock:
                # Double check cache inside lock (prevent race conditions)
                if not force_refresh:
                    try:
                        cached_value = await asyncio.wait_for(
                            self.load(key, context_id),
                            timeout=3.0
                        )
                        if cached_value is not None:
                            return cached_value
                    except (asyncio.TimeoutError, Exception):
                        # Continue with computation if cache check fails
                        pass
                
                # Execute function with timeout and cache result
                try:
                    start_time = time.time()
                    result = await asyncio.wait_for(
                        func(),
                        timeout=30.0  # 30-second timeout for computation
                    )
                    execution_time = time.time() - start_time
                    
                    # Only cache non-None results
                    if result is not None:
                        try:
                            await asyncio.wait_for(
                                self.save(key, context_id, result, ttl),
                                timeout=3.0  # 3-second timeout for saving
                            )
                        except (asyncio.TimeoutError, Exception) as e:
                            if isinstance(e, asyncio.TimeoutError):
                                logger.warning(f"Timeout saving result for {key}")
                            else:
                                logger.error(f"Error saving result: {str(e)}")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    logger.error(f"Timeout executing function for {key}")
                    return None
                except Exception as e:
                    logger.error(f"Error executing function: {str(e)}")
                    return None
                
        except Exception as e:
            logger.error(f"Error in with_cache operation: {str(e)}")
            # Last resort - try to execute the function directly
            try:
                return await func()
            except Exception:
                return None
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "with_bulk_cache"})
    async def with_bulk_cache(
        self,
        func,
        keys: List[str],
        context_id: str,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Execute function with automatic bulk caching.
        
        Args:
            func: Async function that takes list of uncached keys
            keys: List of keys to process
            context_id: The conversation or session ID
            ttl: Optional TTL override
            force_refresh: Force execution even if cached
            
        Returns:
            Dictionary of all results (from cache and fresh execution)
        """
        track_memory_operation("with_bulk_cache")
        
        if not keys:
            return {}
        
        # First try loading all from cache unless force_refresh
        result = {}
        keys_to_compute = []
        
        if not force_refresh:
            # Use a timeout to prevent potential deadlocks
            try:
                cached_values = await asyncio.wait_for(
                    self.bulk_load(keys, context_id),
                    timeout=5.0  # 5-second timeout as a safety mechanism
                )
                
                # Determine which keys need computation
                for key in keys:
                    if key in cached_values and cached_values[key] is not None:
                        result[key] = cached_values[key]
                    else:
                        keys_to_compute.append(key)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout loading cached values for {len(keys)} keys, computing all")
                keys_to_compute = keys
            except Exception as e:
                logger.error(f"Error loading cached values: {str(e)}, computing all")
                keys_to_compute = keys
        else:
            keys_to_compute = keys
        
        # If all values were cached, return immediately
        if not keys_to_compute:
            return result
        
        # Compute missing values
        start_time = time.time()
        try:
            # Add timeout to prevent potential hangs in the function
            computed_values = await asyncio.wait_for(
                func(keys_to_compute),
                timeout=30.0  # Longer timeout for computation
            )
            execution_time = time.time() - start_time
            
            # Save computed values to cache
            if computed_values:
                # Filter out None values
                to_cache = {k: v for k, v in computed_values.items() if v is not None}
                if to_cache:
                    try:
                        await asyncio.wait_for(
                            self.bulk_save(to_cache, context_id, ttl),
                            timeout=5.0  # 5-second timeout for saving
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout saving {len(to_cache)} computed values to cache")
                    except Exception as e:
                        logger.error(f"Error saving computed values to cache: {str(e)}")
            
            # Merge cached and computed results
            result.update(computed_values)
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout computing values for {len(keys_to_compute)} keys")
            # Return partial results with defaults for timed-out keys
            for key in keys_to_compute:
                if key not in result:
                    result[key] = None
        except Exception as e:
            logger.error(f"Error computing values: {str(e)}")
            # Return partial results with defaults for failed keys
            for key in keys_to_compute:
                if key not in result:
                    result[key] = None
                    
        return result
    
    # Vector store methods
    async def store_vector(
        self,
        text: str,
        metadata: Dict[str, Any],
        context_id: Optional[str] = None,
    ) -> Optional[str]:
        """Store text in vector store.
        
        Args:
            text: Text to store
            metadata: Associated metadata
            context_id: Optional context ID
            
        Returns:
            Vector ID if successful, None otherwise
        """
        if not self.vector_store:
            logger.warning("Vector store not configured, cannot store vector")
            return None
            
        try:
            return await self.vector_store.store_vector(
                text=text,
                metadata=metadata,
                context_id=context_id
            )
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.VECTOR_DB_ERROR,
                f"Failed to store vector in context {context_id or 'global'}"
            )
            error.log_error(logger)
            return None
    
    async def search_vectors(
        self,
        query: str,
        k: int = 5,
        context_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query: Search query
            k: Number of results
            context_id: Optional context ID
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching documents with similarity scores
        """
        if not self.vector_store:
            logger.warning("Vector store not configured, cannot search vectors")
            return []
            
        try:
            return await self.vector_store.search_vectors(
                query=query,
                k=k,
                context_id=context_id,
                filter_metadata=filter_metadata
            )
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.VECTOR_DB_ERROR,
                f"Failed to search vectors in context {context_id or 'global'}"
            )
            error.log_error(logger)
            return []
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "get_stats"})
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        stats = {
            "cache_size": len(self.cache),
            "cache_max_size": self.cache.maxsize,
            "cache_ttl": self.cache.ttl,
        }
        
        # Get primary storage stats
        try:
            primary_stats = await self.primary.get_stats()
            stats["primary_storage"] = primary_stats
        except Exception as e:
            logger.error(f"Failed to get primary storage stats: {str(e)}")
            stats["primary_storage"] = {"error": str(e)}
        
        # Get vector store stats if available
        if self.vector_store:
            try:
                vector_stats = await self.vector_store.get_stats()
                stats["vector_store"] = vector_stats
            except Exception as e:
                logger.error(f"Failed to get vector store stats: {str(e)}")
                stats["vector_store"] = {"error": str(e)}
        
        return stats
"""Redis-based memory implementation with optimized access patterns."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.config.connections import get_redis_async_connection
from src.config.errors import ErrorCode, MemoryError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import (
    MEMORY_OPERATION_DURATION,
    timed_metric,
    track_cache_miss,
    track_memory_operation,
    track_memory_operation_completed,
    track_memory_size,
)
from src.config.settings import get_settings
from src.memory.base import BaseMemory
from src.memory.utils import (
    AsyncLock,
    ExpirationPolicy,
    deserialize_data,
    generate_memory_key,
    serialize_data,
)
from src.utils.timing import async_timed

logger = get_logger(__name__)
settings = get_settings()


class RedisMemory(BaseMemory):
    """High-performance Redis-based memory implementation.
    
    Features:
    - Connection pooling
    - Async operations
    - Batched reads/writes
    - Efficient serialization
    - Memory segmentation by context ID
    """
    
    def __init__(self, default_ttl: Optional[int] = None):
        """Initialize Redis memory.
        
        Args:
            default_ttl: Default time-to-live for keys in seconds
        """
        self.default_ttl = default_ttl or settings.MEMORY_TTL
        self._redis = None  # Lazy initialization
    
    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = await get_redis_async_connection()
        return self._redis
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "load_context"})
    async def load_context(
        self, 
        key: str, 
        context_id: str, 
        default: Any = None
    ) -> Any:
        """Load data from Redis.
        
        Args:
            key: The key to load
            context_id: The conversation or session ID
            default: Default value if key not found
            
        Returns:
            The stored data or default if not found
        """
        track_memory_operation("load")
        
        try:
            redis = await self._get_redis()
            full_key = generate_memory_key(key, context_id)
            
            # Get data from Redis
            start_time = time.time()
            data = await redis.get(full_key)
            track_memory_operation_completed("redis_get", time.time() - start_time)
            
            if data is None:
                track_cache_miss("redis_memory")
                return default
            
            # Deserialize the data
            return await deserialize_data(data, default)
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to load context for key {key} in context {context_id}"
            )
            error.log_error(logger)
            return default
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "save_context"})
    async def save_context(
        self, 
        key: str, 
        context_id: str, 
        data: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Save data to Redis.
        
        Args:
            key: The key to save under
            context_id: The conversation or session ID
            data: The data to save
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("save")
        
        try:
            redis = await self._get_redis()
            full_key = generate_memory_key(key, context_id)
            
            # Serialize the data
            serialized_data = await serialize_data(data)
            
            # Determine TTL
            effective_ttl = ExpirationPolicy.get_ttl(self.default_ttl, ttl)
            
            # Save to Redis
            start_time = time.time()
            if effective_ttl is not None:
                success = await redis.setex(full_key, effective_ttl, serialized_data)
            else:
                success = await redis.set(full_key, serialized_data)
            track_memory_operation_completed("redis_set", time.time() - start_time)
            
            # Track memory size
            if success:
                size = len(serialized_data)
                track_memory_size("redis", size)
            
            return bool(success)
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to save context for key {key} in context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "delete_context"})
    async def delete_context(
        self, 
        key: str, 
        context_id: str
    ) -> bool:
        """Delete data from Redis.
        
        Args:
            key: The key to delete
            context_id: The conversation or session ID
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("delete")
        
        try:
            redis = await self._get_redis()
            full_key = generate_memory_key(key, context_id)
            
            # Delete from Redis
            start_time = time.time()
            result = await redis.delete(full_key)
            track_memory_operation_completed("redis_delete", time.time() - start_time)
            
            return result > 0
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to delete context for key {key} in context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "clear"})
    async def clear(
        self, 
        context_id: Optional[str] = None
    ) -> bool:
        """Clear all data for a context or all contexts.
        
        Args:
            context_id: The context to clear, or None for all
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("clear")
        
        try:
            redis = await self._get_redis()
            
            # Create pattern to match keys to delete
            if context_id:
                pattern = f"memory:{context_id}:*"
            else:
                pattern = "memory:*"
            
            # Use scan to find keys to delete (safer than KEYS)
            cursor = 0
            deleted_count = 0
            
            start_time = time.time()
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                
                if keys:
                    # Delete keys in batches
                    result = await redis.delete(*keys)
                    deleted_count += result
                
                # Exit when scan is complete
                if cursor == 0:
                    break
            
            track_memory_operation_completed("redis_clear", time.time() - start_time)
            return True
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to clear context {context_id or 'all'}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "list_keys"})
    async def list_keys(
        self, 
        context_id: Optional[str] = None, 
        pattern: Optional[str] = None
    ) -> List[str]:
        """List all keys in memory.
        
        Args:
            context_id: Filter by this context
            pattern: Additional pattern filter
            
        Returns:
            List of keys
        """
        track_memory_operation("list_keys")
        
        try:
            redis = await self._get_redis()
            
            # Create pattern to match keys
            if context_id:
                base_pattern = f"memory:{context_id}:"
            else:
                base_pattern = "memory:*:"
                
            if pattern:
                search_pattern = f"{base_pattern}{pattern}"
            else:
                search_pattern = f"{base_pattern}*"
            
            # Use scan to find keys
            cursor = 0
            all_keys = []
            
            start_time = time.time()
            while True:
                cursor, keys = await redis.scan(cursor, match=search_pattern, count=100)
                
                # Extract the actual key names (remove prefix)
                for full_key in keys:
                    parts = full_key.split(":", 2)
                    if len(parts) == 3:  # memory:context_id:key
                        all_keys.append(parts[2])
                
                # Exit when scan is complete
                if cursor == 0:
                    break
            
            track_memory_operation_completed("redis_list_keys", time.time() - start_time)
            return all_keys
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to list keys for context {context_id or 'all'}"
            )
            error.log_error(logger)
            return []
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "exists"})
    async def exists(
        self, 
        key: str, 
        context_id: str
    ) -> bool:
        """Check if a key exists.
        
        Args:
            key: The key to check
            context_id: The conversation or session ID
            
        Returns:
            True if key exists, False otherwise
        """
        track_memory_operation("exists")
        
        try:
            redis = await self._get_redis()
            full_key = generate_memory_key(key, context_id)
            
            # Check if key exists
            start_time = time.time()
            result = await redis.exists(full_key)
            track_memory_operation_completed("redis_exists", time.time() - start_time)
            
            return result > 0
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to check existence for key {key} in context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "bulk_load"})
    async def bulk_load(
        self, 
        keys: List[str], 
        context_id: str, 
        default: Any = None
    ) -> Dict[str, Any]:
        """Load multiple values in a single operation.
        
        Args:
            keys: List of keys to retrieve
            context_id: The conversation or session ID
            default: Default value for missing keys
            
        Returns:
            Dictionary of {key: value} for all found keys
        """
        track_memory_operation("bulk_load")
        
        if not keys:
            return {}
            
        try:
            redis = await self._get_redis()
            
            # Generate full keys
            full_keys = [generate_memory_key(k, context_id) for k in keys]
            
            # Fetch all values in a single operation
            start_time = time.time()
            values = await redis.mget(*full_keys)
            track_memory_operation_completed("redis_mget", time.time() - start_time)
            
            # Deserialize all values
            result = {}
            deserialize_tasks = []
            
            # Create tasks for deserializing all values in parallel
            for i, key in enumerate(keys):
                if values[i] is not None:
                    task = asyncio.create_task(deserialize_data(values[i], default))
                    deserialize_tasks.append((key, task))
                else:
                    result[key] = default
            
            # Wait for all deserialization tasks to complete
            for key, task in deserialize_tasks:
                result[key] = await task
            
            return result
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR,
                f"Failed to bulk load keys in context {context_id}"
            )
            error.log_error(logger)
            return {k: default for k in keys}
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "bulk_save"})
    async def bulk_save(
        self, 
        data: Dict[str, Any], 
        context_id: str, 
        ttl: Optional[int] = None
    ) -> bool:
        """Save multiple values in a single operation.
        
        Args:
            data: Dictionary of {key: value} pairs to store
            context_id: The conversation or session ID
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("bulk_save")
        
        if not data:
            return True
            
        try:
            redis = await self._get_redis()
            pipe = redis.pipeline()
            
            # Determine TTL
            effective_ttl = ExpirationPolicy.get_ttl(self.default_ttl, ttl)
            
            # Serialize all values in parallel
            serialize_tasks = {}
            for key, value in data.items():
                task = asyncio.create_task(serialize_data(value))
                serialize_tasks[key] = task
            
            # Wait for all serialization tasks to complete
            serialized_data = {}
            for key, task in serialize_tasks.items():
                serialized_data[key] = await task
            
            # Add all set commands to pipeline
            start_time = time.time()
            for key, value in serialized_data.items():
                full_key = generate_memory_key(key, context_id)
                
                if effective_ttl is not None:
                    pipe.setex(full_key, effective_ttl, value)
                else:
                    pipe.set(full_key, value)
            
            # Execute pipeline
            results = await pipe.execute()
            track_memory_operation_completed("redis_pipeline_set", time.time() - start_time)
            
            # Track memory size
            total_size = sum(len(value) for value in serialized_data.values())
            track_memory_size("redis", total_size)
            
            # All commands should return True/OK
            return all(bool(result) for result in results)
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR,
                f"Failed to bulk save keys in context {context_id}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "get_stats"})
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        try:
            redis = await self._get_redis()
            
            # Get Redis info
            start_time = time.time()
            info = await redis.info(section="memory")
            dbsize = await redis.dbsize()
            track_memory_operation_completed("redis_info", time.time() - start_time)
            
            # Count memory keys
            memory_keys_count = 0
            cursor = 0
            
            while True:
                cursor, keys = await redis.scan(cursor, match="memory:*", count=100)
                memory_keys_count += len(keys)
                if cursor == 0:
                    break
            
            # Compile stats
            stats = {
                "type": self.__class__.__name__,
                "total_keys": dbsize,
                "memory_keys": memory_keys_count,
                "memory_used_bytes": info.get("used_memory", 0),
                "memory_peak_bytes": info.get("used_memory_peak", 0),
                "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {str(e)}")
            return {
                "type": self.__class__.__name__,
                "error": str(e)
            }
            
    async def get_lock(self, name: str, expire_time: int = 10) -> AsyncLock:
        """Get a distributed lock for coordinating access.
        
        Args:
            name: Lock name
            expire_time: Lock expiration time in seconds
            
        Returns:
            AsyncLock instance
        """
        redis = await self._get_redis()
        return AsyncLock(redis, name, expire_time)
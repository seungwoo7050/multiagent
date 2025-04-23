"""
Caching system for LLM responses with Redis and in-memory cache.
"""

import abc
import asyncio
import json
import time
from typing import Any, Dict, Optional, Union, TypeVar, Generic
import hashlib
from functools import lru_cache

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.config.metrics import (
    CACHE_OPERATIONS_TOTAL, 
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    CACHE_SIZE,
    track_cache_operation,
    track_cache_hit,
    track_cache_miss,
    track_cache_size,
    timed_metric,
    MEMORY_OPERATION_DURATION
)
from src.config.errors import MemoryError, ErrorCode

# Import functions from config module
from src.config.connections import get_redis_async_connection

settings = get_settings()
logger = get_logger(__name__)

T = TypeVar('T')

# Singleton instance of the cache
_CACHE_INSTANCE = None
_CACHE_LOCK = asyncio.Lock()


class LLMCache(abc.ABC, Generic[T]):
    """Abstract base class for LLM response caching."""
    
    @abc.abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[T]: Cached item or None if not found
        """
        pass
    
    @abc.abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abc.abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abc.abstractmethod
    async def clear(self) -> bool:
        """Clear the entire cache.
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abc.abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache stats
        """
        pass


class TwoLevelCache(LLMCache[T]):
    """Two-level cache with in-memory LRU and Redis backend."""
    
    def __init__(
        self, 
        namespace: str = "llm",
        local_maxsize: int = 1000,
        ttl: int = 3600,
        serializer: Optional[callable] = None,
        deserializer: Optional[callable] = None
    ):
        """Initialize the two-level cache.
        
        Args:
            namespace: Cache namespace prefix for Redis keys
            local_maxsize: Maximum size of the local LRU cache
            ttl: Default time-to-live in seconds
            serializer: Function to serialize values for Redis
            deserializer: Function to deserialize values from Redis
        """
        self.namespace = namespace
        self.ttl = ttl
        self.local_maxsize = local_maxsize
        
        # Set default serializer/deserializer if not provided
        self.serializer = serializer or self._default_serializer
        self.deserializer = deserializer or self._default_deserializer
        
        # Initialize metrics
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0
        self.delete_count = 0
        
        # Create in-memory LRU cache
        # Note: We're using a simple dict here with manual LRU management
        # for better async compatibility
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self.local_cache_order: list = []  # For LRU tracking
        
        # Initialize Redis connection later (lazy initialization)
        self._redis = None
        self._initialized = False
        
        logger.debug(f"Initialized two-level cache with namespace '{namespace}'")
    
    async def _ensure_initialized(self) -> bool:
        """Ensure the cache is initialized.
        
        Returns:
            bool: True if initialization was successful
        """
        if self._initialized:
            return True
        
        try:
            # Import here to avoid circular imports
            from src.config.connections import get_redis_async_connection
            
            # Get Redis connection from config# Get Redis connection
            self._redis = await get_redis_async_connection()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize cache: {str(e)}")
            return False
    
    def _default_serializer(self, value: T) -> str:
        """Default serializer for cache values.
        
        Args:
            value: Value to serialize
            
        Returns:
            str: Serialized value
        """
        return json.dumps(value)
    
    def _default_deserializer(self, serialized: str) -> T:
        """Default deserializer for cache values.
        
        Args:
            serialized: Serialized value
            
        Returns:
            T: Deserialized value
        """
        return json.loads(serialized)
    
    def _get_redis_key(self, key: str) -> str:
        """Get the full Redis key with namespace.
        
        Args:
            key: Cache key
            
        Returns:
            str: Full Redis key
        """
        return f"{self.namespace}:{key}"
    
    def _update_lru_order(self, key: str) -> None:
        """Update the LRU order after a key access.
        
        Args:
            key: Cache key
        """
        # Remove from current position (if exists)
        if key in self.local_cache_order:
            self.local_cache_order.remove(key)
        
        # Add to the end (most recently used)
        self.local_cache_order.append(key)
        
        # Trim cache if needed
        while len(self.local_cache_order) > self.local_maxsize:
            oldest_key = self.local_cache_order.pop(0)  # Remove least recently used
            self.local_cache.pop(oldest_key, None)
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "cache_get"})
    async def get(self, key: str) -> Optional[T]:
        """Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[T]: Cached item or None if not found
        """
        track_cache_operation("get")
        
        # Check local cache first (fastest)
        if key in self.local_cache:
            entry = self.local_cache[key]
            # Check if expired
            if "expires_at" not in entry or entry["expires_at"] > time.time():
                self._update_lru_order(key)
                self.hit_count += 1
                track_cache_hit("local")
                return entry["value"]
            else:
                # Expired - remove from local cache
                self.local_cache.pop(key, None)
                if key in self.local_cache_order:
                    self.local_cache_order.remove(key)
        
        # Ensure initialized
        initialized = await self._ensure_initialized()
        if not initialized:
            self.miss_count += 1
            track_cache_miss("local")
            return None
        
        # Check Redis
        try:
            redis_key = self._get_redis_key(key)
            serialized = await self._redis.get(redis_key)
            
            if serialized:
                # Cache hit in Redis
                value = self.deserializer(serialized)
                
                # Get TTL to pass to local cache
                ttl = await self._redis.ttl(redis_key)
                expires_at = time.time() + ttl if ttl > 0 else None
                
                # Store in local cache
                self.local_cache[key] = {
                    "value": value,
                    "expires_at": expires_at
                }
                self._update_lru_order(key)
                
                self.hit_count += 1
                track_cache_hit("redis")
                return value
            else:
                # Not found in Redis either
                self.miss_count += 1
                track_cache_miss("redis")
                return None
                
        except Exception as e:
            logger.warning(f"Error retrieving from Redis cache: {str(e)}")
            self.miss_count += 1
            track_cache_miss("redis_error")
            return None
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "cache_set"})
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            bool: True if successful
        """
        track_cache_operation("set")
        
        # Use default TTL if not specified
        ttl = ttl if ttl is not None else self.ttl
        
        # Store in local cache
        expires_at = time.time() + ttl if ttl > 0 else None
        self.local_cache[key] = {
            "value": value,
            "expires_at": expires_at
        }
        self._update_lru_order(key)
        
        # Ensure initialized
        initialized = await self._ensure_initialized()
        if not initialized:
            return False
        
        # Store in Redis
        try:
            redis_key = self._get_redis_key(key)
            serialized = self.serializer(value)
            
            if ttl > 0:
                await self._redis.setex(redis_key, ttl, serialized)
            else:
                await self._redis.set(redis_key, serialized)
            
            self.set_count += 1
            track_cache_size("local", len(self.local_cache))
            
            return True
        except Exception as e:
            logger.warning(f"Error storing in Redis cache: {str(e)}")
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "cache_delete"})
    async def delete(self, key: str) -> bool:
        """Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        track_cache_operation("delete")
        
        # Remove from local cache
        self.local_cache.pop(key, None)
        if key in self.local_cache_order:
            self.local_cache_order.remove(key)
        
        # Ensure initialized
        initialized = await self._ensure_initialized()
        if not initialized:
            return False
        
        # Remove from Redis
        try:
            redis_key = self._get_redis_key(key)
            await self._redis.delete(redis_key)
            
            self.delete_count += 1
            return True
        except Exception as e:
            logger.warning(f"Error deleting from Redis cache: {str(e)}")
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "cache_clear"})
    async def clear(self) -> bool:
        """Clear the entire cache.
        
        Returns:
            bool: True if successful
        """
        track_cache_operation("clear")
        
        # Clear local cache
        self.local_cache.clear()
        self.local_cache_order.clear()
        
        # Ensure initialized
        initialized = await self._ensure_initialized()
        if not initialized:
            return False
        
        # Clear Redis keys in this namespace
        try:
            # Get all keys in this namespace
            pattern = f"{self.namespace}:*"
            cursor = "0"
            deleted_count = 0
            
            while cursor != 0:
                cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                
                if keys:
                    await self._redis.delete(*keys)
                    deleted_count += len(keys)
                
                # Convert cursor to int and check if we're done
                cursor = int(cursor)
                if cursor == 0:
                    break
            
            logger.debug(f"Cleared {deleted_count} keys from Redis cache namespace '{self.namespace}'")
            return True
        except Exception as e:
            logger.warning(f"Error clearing Redis cache: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache stats
        """
        stats = {
            "namespace": self.namespace,
            "local_cache_size": len(self.local_cache),
            "local_cache_maxsize": self.local_maxsize,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "set_count": self.set_count,
            "delete_count": self.delete_count,
            "hit_ratio": self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
        }
        
        # Add Redis stats if available
        if self._initialized:
            try:
                # Get Redis memory usage
                redis_key_count = 0
                pattern = f"{self.namespace}:*"
                cursor = "0"
                
                while cursor != 0:
                    cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                    redis_key_count += len(keys)
                    
                    # Convert cursor to int and check if we're done
                    cursor = int(cursor)
                    if cursor == 0:
                        break
                
                stats["redis_key_count"] = redis_key_count
            except Exception as e:
                logger.warning(f"Error getting Redis stats: {str(e)}")
        
        return stats


async def get_cache() -> LLMCache:
    """Get the global cache instance.
    
    Returns:
        LLMCache: Cache instance
    """
    global _CACHE_INSTANCE, _CACHE_LOCK
    
    if _CACHE_INSTANCE is not None:
        return _CACHE_INSTANCE
    
    # Use lock to prevent race conditions in initialization
    async with _CACHE_LOCK:
        # Double-check pattern
        if _CACHE_INSTANCE is not None:
            return _CACHE_INSTANCE
        
        # Create new cache
        _CACHE_INSTANCE = TwoLevelCache(
            namespace="llm_cache",
            local_maxsize=5000,  # Large local cache for frequently-used prompts
            ttl=settings.CACHE_TTL
        )
        
        logger.info("Initialized LLM cache")
        return _CACHE_INSTANCE


async def clear_cache() -> bool:
    """Clear the LLM response cache.
    
    Returns:
        bool: True if successful
    """
    cache = await get_cache()
    return await cache.clear()


async def cache_result(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Cache an LLM result.
    
    Args:
        key: Cache key
        value: Result to cache
        ttl: Time-to-live in seconds (optional)
        
    Returns:
        bool: True if successful
    """
    cache = await get_cache()
    return await cache.set(key, value, ttl)


async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics.
    
    Returns:
        Dict[str, Any]: Cache stats
    """
    cache = await get_cache()
    return await cache.get_stats()
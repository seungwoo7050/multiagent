"""Utility functions for the memory module."""

import asyncio
import hashlib
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import msgpack

from src.config.logger import get_logger
from src.config.metrics import track_memory_operation_completed
from src.utils.timing import async_timed

logger = get_logger(__name__)


def generate_memory_key(key: str, context_id: str) -> str:
    """Generate a Redis key with proper namespacing.
    
    Args:
        key: Base key name
        context_id: The context/conversation ID
        
    Returns:
        Full namespaced Redis key
    """
    return f"memory:{context_id}:{key}"


def generate_vector_key(context_id: Optional[str] = None) -> str:
    """Generate a vector store collection key with proper namespacing.
    
    Args:
        context_id: Optional context ID for segmentation
        
    Returns:
        Namespaced vector collection key
    """
    if context_id:
        return f"vectors:{context_id}"
    return "vectors:global"


@async_timed(name="memory_serialize")
async def serialize_data(data: Any) -> bytes:
    """Efficiently serialize data for storage.
    
    Uses MessagePack for speed and compactness.
    
    Args:
        data: Data to serialize
        
    Returns:
        Serialized bytes
    """
    start_time = time.time()
    try:
        # Try MessagePack first for efficiency
        serialized = msgpack.packb(data, use_bin_type=True)
        track_memory_operation_completed("serialize_msgpack", time.time() - start_time)
        return serialized
    except (TypeError, OverflowError):
        # Fall back to pickle for complex Python objects
        # Wrap in another try block to handle pickle-specific errors
        try:
            serialized = pickle.dumps(data)
            # Add a prefix to identify this as pickle data
            result = b"pkl:" + serialized
            track_memory_operation_completed("serialize_pickle", time.time() - start_time)
            return result
        except Exception as e:
            # Last resort: JSON + string encoding
            logger.warning(f"Falling back to JSON serialization: {str(e)}")
            json_str = json.dumps(data, default=str)
            result = b"json:" + json_str.encode('utf-8')
            track_memory_operation_completed("serialize_json", time.time() - start_time)
            return result


@async_timed(name="memory_deserialize")
async def deserialize_data(data: bytes, default: Any = None) -> Any:
    """Deserialize data from storage.
    
    Automatically detects serialization format.
    
    Args:
        data: Serialized bytes to deserialize
        default: Default value to return on error
        
    Returns:
        Deserialized data or default on error
    """
    start_time = time.time()
    
    if data is None:
        return default
    
    try:
        # Check for format prefix
        if data.startswith(b"pkl:"):
            # Pickle data
            result = pickle.loads(data[4:])
            track_memory_operation_completed("deserialize_pickle", time.time() - start_time)
            return result
        elif data.startswith(b"json:"):
            # JSON data
            json_str = data[5:].decode('utf-8')
            result = json.loads(json_str)
            track_memory_operation_completed("deserialize_json", time.time() - start_time)
            return result
        else:
            # Default to MessagePack
            result = msgpack.unpackb(data, raw=False)
            track_memory_operation_completed("deserialize_msgpack", time.time() - start_time)
            return result
    except Exception as e:
        logger.error(f"Error deserializing data: {str(e)}")
        return default


def compute_fingerprint(data: Any) -> str:
    """Compute a stable fingerprint hash for cached data.
    
    Args:
        data: Data to compute fingerprint for
        
    Returns:
        Hex string fingerprint
    """
    if isinstance(data, (str, bytes)):
        # Direct hash for simple types
        payload = data.encode() if isinstance(data, str) else data
    else:
        # Convert to stable JSON for complex types
        try:
            # Sort keys for stable hashing
            payload = json.dumps(data, sort_keys=True, default=str).encode()
        except:
            # Fall back for non-JSON serializable objects
            payload = str(data).encode()
    
    # Compute SHA-256 hash
    return hashlib.sha256(payload).hexdigest()


class AsyncLock:
    """Distributed lock using Redis.
    
    Provides a high-performance async locking mechanism for
    coordinating memory access across processes.
    """
    
    def __init__(self, redis_client, lock_name: str, expire_time: int = 10):
        """Initialize AsyncLock.
        
        Args:
            redis_client: Redis client instance
            lock_name: Unique name for this lock
            expire_time: Lock expiration time in seconds
        """
        self.redis = redis_client
        self.lock_name = f"lock:{lock_name}"
        self.expire_time = expire_time
        self.owner = f"{time.time()}-{id(self)}"
    
    async def __aenter__(self):
        """Acquire the lock."""
        # Try to acquire lock with retries
        retry_count = 5
        retry_delay = 0.05  # 50ms initial delay
        
        for attempt in range(retry_count):
            # Use SET NX to atomically acquire the lock
            acquired = await self.redis.set(
                self.lock_name, 
                self.owner,
                nx=True,
                ex=self.expire_time
            )
            
            if acquired:
                return self  # Lock acquired successfully
            
            # Exponential backoff with jitter
            if attempt < retry_count - 1:
                jitter = retry_delay * 0.1 * (2 * (await self._random()) - 1)
                await asyncio.sleep(retry_delay + jitter)
                retry_delay *= 2  # Exponential backoff
        
        # Could not acquire lock after retries
        logger.warning(f"Failed to acquire lock {self.lock_name} after {retry_count} attempts")
        return self  # Return anyway to avoid raising exception
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the lock."""
        # Only release if we're the owner
        lua_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        await self.redis.eval(lua_script, 1, self.lock_name, self.owner)
    
    async def _random(self) -> float:
        """Generate a random number for jitter calculation."""
        # Simple way to get randomness in async context
        return (time.time() * 1000) % 1.0 / 1.0


class ExpirationPolicy:
    """Manages TTL and expiration for memory entries."""
    
    @staticmethod
    def get_ttl(default_ttl: Optional[int], 
                override_ttl: Optional[int], 
                key_type: str = "default") -> Optional[int]:
        """Calculate appropriate TTL for a memory key.
        
        Args:
            default_ttl: Default TTL from system config
            override_ttl: Specific TTL request for this operation
            key_type: Type of key (affects TTL strategy)
            
        Returns:
            TTL in seconds or None for no expiration
        """
        # Override takes precedence if specified
        if override_ttl is not None:
            return override_ttl
        
        # Otherwise use default, with adjustments based on key type
        if default_ttl is None:
            return None
            
        if key_type == "temporary":
            # Temporary keys get shorter lifetime (25% of default)
            return max(60, int(default_ttl * 0.25))
        elif key_type == "persistent":
            # Persistent keys get longer lifetime (4x default)
            return default_ttl * 4
        else:
            # Default keys use the standard TTL
            return default_ttl
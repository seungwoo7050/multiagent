"""
Utility functions for the memory system.
Provides serialization, key generation, and locking mechanisms.
"""
import asyncio
import hashlib
import json
import pickle
import random
import re
import time
from typing import Any, Optional, cast

import msgpack

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.utils.timing import async_timed

logger = get_logger(__name__)
metrics = get_metrics_manager()

def generate_memory_key(key: str, context_id: str) -> str:
    """
    Generate a standardized key for memory storage.
    
    Args:
        key: Base key
        context_id: Context identifier
        
    Returns:
        str: Formatted memory key
    """
    return f'memory:{context_id}:{key}'

def generate_vector_key(context_id: Optional[str]=None) -> str:
    """
    Generate a standardized key for vector storage.
    
    Args:
        context_id: Optional context identifier
        
    Returns:
        str: Formatted vector key
    """
    if context_id:
        return f'vectors:{context_id}'
    else:
        return 'vectors:global'

@async_timed(name='memory_serialize')
async def serialize_data(data: Any) -> bytes:
    """
    Serialize data for storage, with fallback mechanisms.
    
    Args:
        data: Data to serialize
        
    Returns:
        bytes: Serialized data
        
    Raises:
        Exception: If all serialization methods fail
    """
    start_time = time.monotonic()
    serialized_data: Optional[bytes] = None
    method_used: str = 'unknown'
    
    try:
        # Try MessagePack first (most efficient)
        method_used = 'msgpack'
        serialized_data = msgpack.packb(data, use_bin_type=True)
        logger.debug(f'Data serialized using MessagePack (Size: {len(serialized_data)} bytes)')
    except (TypeError, OverflowError, ValueError) as e_msgpack:
        logger.warning(f'MessagePack serialization failed ({e_msgpack}). Falling back to Pickle.')
        
        try:
            # Fall back to Pickle
            method_used = 'pickle'
            pickled_data = pickle.dumps(data)
            serialized_data = b'pkl:' + pickled_data
            logger.debug(f'Data serialized using Pickle (Size: {len(serialized_data)} bytes)')
        except (pickle.PicklingError, TypeError, AttributeError) as e_pickle:
            logger.warning(f'Pickle serialization failed ({e_pickle}). Falling back to JSON.')
            
            try:
                # Last resort: JSON
                method_used = 'json'
                json_str = json.dumps(data, default=str, ensure_ascii=False)
                serialized_data = b'json:' + json_str.encode('utf-8')
                logger.debug(f'Data serialized using JSON (Size: {len(serialized_data)} bytes)')
            except Exception as e_json:
                logger.error('All serialization methods (MessagePack, Pickle, JSON) failed.', exc_info=True)
                raise Exception(f'Could not serialize data after trying all methods. Last error (JSON): {e_json}') from e_json
    
    # Track metrics
    duration = time.monotonic() - start_time
    metrics.track_memory('duration', operation_type=f'serialize_{method_used}', value=duration)
    
    return cast(bytes, serialized_data)

@async_timed(name='memory_deserialize')
async def deserialize_data(data: bytes, default: Any=None) -> Any:
    """
    Deserialize data from storage.
    
    Args:
        data: Serialized data
        default: Default value to return if deserialization fails
        
    Returns:
        Any: Deserialized data or default on failure
    """
    start_time = time.monotonic()
    method_used = 'unknown'
    
    if data is None:
        logger.debug('deserialize_data received None, returning default.')
        return default
    
    try:
        # Determine serialization method by prefix
        if data.startswith(b'pkl:'):
            method_used = 'pickle'
            result = pickle.loads(data[4:])
            logger.debug('Data deserialized using Pickle.')
        elif data.startswith(b'json:'):
            method_used = 'json'
            json_str = data[5:].decode('utf-8')
            result = json.loads(json_str)
            logger.debug('Data deserialized using JSON.')
        else:
            # Default to MessagePack if no prefix
            method_used = 'msgpack'
            result = msgpack.unpackb(data, raw=False)
            logger.debug('Data deserialized using MessagePack.')
        
        # Track metrics
        duration = time.monotonic() - start_time
        metrics.track_memory('duration', operation_type=f'deserialize_{method_used}', value=duration)
        
        return result
    except Exception as e:
        logger.error(f'Error deserializing data (assumed format: {method_used}, length: {len(data)}): {e}', exc_info=True)
        duration = time.monotonic() - start_time
        metrics.track_memory('duration', operation_type=f'deserialize_failure_{method_used}', value=duration)
        return default

def compute_fingerprint(data: Any) -> str:
    """
    Compute a stable hash fingerprint for data objects.
    
    Args:
        data: Data to fingerprint
        
    Returns:
        str: SHA-256 hash hexdigest
    """
    payload: bytes
    
    if isinstance(data, bytes):
        payload = data
    elif isinstance(data, str):
        payload = data.encode('utf-8')
    else:
        try:
            # Try to serialize to JSON for consistent hashing
            payload = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        except Exception as e:
            logger.warning(f'Could not JSON serialize data for fingerprinting (type: {type(data)}). Using str(). Error: {e}')
            payload = str(data).encode('utf-8')
    
    return hashlib.sha256(payload).hexdigest()

def matches_pattern(text: str, pattern: str) -> bool:
    """
    Check if text matches a pattern (with wildcard support).
    
    Args:
        text: Text to match
        pattern: Pattern to match against (supports * wildcard)
        
    Returns:
        bool: True if text matches pattern
    """
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

class AsyncLock:
    """
    Redis-based distributed locking mechanism.
    """
    def __init__(self, redis_client: Any, lock_name: str, expire_time: int=10):
        """
        Initialize the lock.
        
        Args:
            redis_client: Redis client instance
            lock_name: Name for the lock
            expire_time: Lock expiration time in seconds
        """
        self.redis = redis_client
        self.lock_name: str = f'lock:{lock_name}'
        self.expire_time: int = expire_time
        self._owner_token: str = f'{time.time()}-{id(self)}'

    async def __aenter__(self) -> 'AsyncLock':
        """
        Acquire the lock.
        
        Returns:
            AsyncLock: Self reference
        """
        retry_count = 5
        retry_delay = 0.05
        max_retry_delay = 0.5
        
        for attempt in range(retry_count):
            acquired = await self.redis.set(
                self.lock_name, 
                self._owner_token, 
                nx=True, 
                ex=self.expire_time
            )
            
            if acquired:
                logger.debug(f"Acquired lock '{self.lock_name}' (owner: {self._owner_token})")
                return self
                
            if attempt < retry_count - 1:
                # Add jitter to avoid thundering herd problem
                jitter = retry_delay * 0.2 * (random.random() - 0.5)
                sleep_time = min(max_retry_delay, retry_delay + jitter)
                logger.debug(f"Lock '{self.lock_name}' busy. Retrying attempt {attempt + 1}/{retry_count} after {sleep_time:.3f}s...")
                await asyncio.sleep(max(0, sleep_time))
                retry_delay = min(max_retry_delay, retry_delay * 2)
            else:
                logger.warning(f"Failed to acquire lock '{self.lock_name}' after {retry_count} attempts.")
                break
                
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        """
        Release the lock.
        """
        # Use Lua script to ensure we only release our own lock
        lua_script = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        else
            return 0
        end
        """
        
        try:
            released = await self.redis.eval(
                lua_script, 
                1, 
                self.lock_name, 
                self._owner_token
            )
            
            if released:
                logger.debug(f"Released lock '{self.lock_name}' (owner: {self._owner_token})")
            else:
                current_owner = await self.redis.get(self.lock_name)
                logger.warning(
                    f"Could not release lock '{self.lock_name}': Not the owner or lock expired/missing. "
                    f"Current owner token in Redis: {current_owner}. My token: {self._owner_token}"
                )
        except Exception as e:
            logger.error(f"Error releasing lock '{self.lock_name}': {e}", exc_info=True)

class ExpirationPolicy:
    """
    Utilities for TTL management.
    """
    @staticmethod
    def get_ttl(default_ttl: Optional[int], override_ttl: Optional[int], key_type: str='default') -> Optional[int]:
        """
        Determine effective TTL based on policy.
        
        Args:
            default_ttl: Default TTL value
            override_ttl: TTL override if specified
            key_type: Type of key for TTL policy selection
            
        Returns:
            Optional[int]: Effective TTL in seconds, or None for no expiration
        """
        # Override takes precedence
        if override_ttl is not None:
            logger.debug(f"Using override TTL: {override_ttl}s for key type '{key_type}'")
            return override_ttl if override_ttl > 0 else None
            
        # If no default is provided
        if default_ttl is None:
            logger.debug(f"No default TTL specified, setting no expiration for key type '{key_type}'")
            return None
            
        # Apply TTL policies by key type
        effective_ttl: int
        if key_type == 'temporary':
            # Temporary keys expire faster (25% of default)
            effective_ttl = max(60, int(default_ttl * 0.25))
            logger.debug(f"Using temporary TTL: {effective_ttl}s (25% of default {default_ttl}s) for key type '{key_type}'")
        elif key_type == 'persistent':
            # Persistent keys last longer (4x default)
            effective_ttl = default_ttl * 4
            logger.debug(f"Using persistent TTL: {effective_ttl}s (4x default {default_ttl}s) for key type '{key_type}'")
        else:
            # Default keys use the default TTL
            effective_ttl = default_ttl
            logger.debug(f"Using default TTL: {effective_ttl}s for key type '{key_type}'")
            
        return effective_ttl if effective_ttl > 0 else None
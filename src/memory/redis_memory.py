"""
Redis implementation of the memory storage system.
Provides persistent storage of context data with TTL support.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config.connections import get_connection_manager
from src.config.errors import ConnectionError, ErrorCode, convert_exception
from src.config.logger import get_logger
from src.config.metrics import MEMORY_METRICS, get_metrics_manager
from src.config.settings import get_settings
from src.memory.base import BaseMemory
from src.memory.utils import (AsyncLock, ExpirationPolicy, deserialize_data,
                              generate_memory_key, serialize_data)

logger = get_logger(__name__)
metrics = get_metrics_manager()
settings = get_settings()
conn_manager = get_connection_manager()

class RedisMemory(BaseMemory):
    """
    Redis-based implementation of memory storage system.
    """
    def __init__(self, default_ttl: Optional[int]=None):
        """
        Initialize Redis memory storage.
        
        Args:
            default_ttl: Default TTL in seconds
        """
        self.default_ttl: int = default_ttl if default_ttl is not None else settings.MEMORY_TTL
        self._redis: Optional[Any] = None
        logger.debug(f'RedisMemory initialized with default TTL: {self.default_ttl}s')

    async def _get_redis(self) -> Any:
        """
        Get Redis connection, initializing if needed.
        
        Returns:
            Redis client instance
            
        Raises:
            ConnectionError: If connection fails
        """
        if self._redis is None:
            logger.debug('Initializing Redis connection for RedisMemory...')
            try:
                self._redis = await conn_manager.get_redis_async_connection()
                logger.debug('Redis connection successfully obtained.')
            except Exception as e:
                raise ConnectionError(
                    code=ErrorCode.REDIS_CONNECTION_ERROR, 
                    message=f'Failed to get Redis connection: {str(e)}', 
                    original_error=e,
                    service="redis"
                ) from e
        return self._redis

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'load_context'})
    async def load_context(self, key: str, context_id: str, default: Any=None) -> Any:
        """
        Load context data from Redis.
        
        Args:
            key: The key to load
            context_id: The context identifier
            default: Default value to return if key not found
            
        Returns:
            The stored data or default if not found
        """
        metrics.track_memory('operations', operation_type='load')
        full_key = generate_memory_key(key, context_id)
        logger.debug(f"Loading context for key: '{key}' (Redis key: '{full_key}')")
        
        try:
            redis = await self._get_redis()
            start_time = time.monotonic()
            data: Optional[bytes] = await redis.get(full_key)
            metrics.track_memory('duration', operation_type='redis_get', value=time.monotonic() - start_time)
            
            if data is None:
                logger.debug(f"Key '{full_key}' not found in Redis. Returning default.")
                metrics.track_cache('misses', cache_type='redis_memory')
                return default
                
            logger.debug(f"Key '{full_key}' found in Redis. Deserializing {len(data)} bytes.")
            deserialized_data = await deserialize_data(data, default)
            return deserialized_data
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                f"Failed to load context for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger)
            return default

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'save_context'})
    async def save_context(self, key: str, context_id: str, data: Any, ttl: Optional[int]=None) -> bool:
        """
        Save context data to Redis.
        
        Args:
            key: The key to save
            context_id: The context identifier
            data: The data to save
            ttl: Time-to-live in seconds, or None for no expiration
            
        Returns:
            bool: True if save was successful
        """
        metrics.track_memory('operations', operation_type='save')
        full_key = generate_memory_key(key, context_id)
        logger.debug(f"Saving context for key: '{key}' (Redis key: '{full_key}') with TTL: {ttl}")
        
        try:
            redis = await self._get_redis()
            serialized_data: bytes = await serialize_data(data)
            data_size_bytes = len(serialized_data)
            logger.debug(f"Serialized data size: {data_size_bytes} bytes for key '{full_key}'")
            
            effective_ttl = ExpirationPolicy.get_ttl(self.default_ttl, ttl)
            start_time = time.monotonic()
            success: bool = False
            
            if effective_ttl is not None and effective_ttl > 0:
                success = await redis.setex(full_key, effective_ttl, serialized_data)
                logger.debug(f"Saved key '{full_key}' with TTL {effective_ttl}s")
            else:
                success = await redis.set(full_key, serialized_data)
                logger.debug(f"Saved key '{full_key}' without TTL")
                
            metrics.track_memory('duration', operation_type='redis_set', value=time.monotonic() - start_time)
            
            if success:
                metrics.track_memory('size', memory_type='redis', value=data_size_bytes)
                
            return bool(success)
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f"Failed to save context for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'delete_context'})
    async def delete_context(self, key: str, context_id: str) -> bool:
        """
        Delete context data from Redis.
        
        Args:
            key: The key to delete
            context_id: The context identifier
            
        Returns:
            bool: True if deletion was successful
        """
        metrics.track_memory('operations', operation_type='delete')
        full_key = generate_memory_key(key, context_id)
        logger.debug(f"Deleting context for key: '{key}' (Redis key: '{full_key}')")
        
        try:
            redis = await self._get_redis()
            start_time = time.monotonic()
            result: int = await redis.delete(full_key)
            metrics.track_memory('duration', operation_type='redis_delete', value=time.monotonic() - start_time)
            
            deleted = result > 0
            if deleted:
                logger.debug(f"Successfully deleted key '{full_key}'.")
            else:
                logger.debug(f"Key '{full_key}' not found for deletion.")
                
            return deleted
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f"Failed to delete context for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'clear'})
    async def clear(self, context_id: Optional[str]=None) -> bool:
        """
        Clear all data for a context, or all contexts if none specified.
        
        Args:
            context_id: The context identifier, or None for all contexts
            
        Returns:
            bool: True if clearing was successful
        """
        metrics.track_memory('operations', operation_type='clear')
        operation_desc = f"context '{context_id}'" if context_id else "all contexts"
        logger.info(f'Clearing Redis memory for {operation_desc}...')
        
        try:
            redis = await self._get_redis()
            
            if context_id:
                pattern = f'memory:{context_id}:*'
            else:
                pattern = 'memory:*'
                
            cursor: Union[bytes, int] = b'0'
            deleted_count = 0
            total_scan_time = 0.0
            total_delete_time = 0.0
            
            while True:
                scan_start = time.monotonic()
                next_cursor_bytes: bytes
                keys_bytes: List[bytes]
                current_cursor_int = 0
                
                try:
                    if isinstance(cursor, bytes):
                        current_cursor_int = int(cursor.decode())
                    elif isinstance(cursor, int):
                        current_cursor_int = cursor
                except ValueError:
                    pass
                    
                next_cursor_bytes, keys_bytes = await redis.scan(cursor=current_cursor_int, match=pattern, count=1000)
                total_scan_time += time.monotonic() - scan_start
                
                if keys_bytes:
                    delete_start = time.monotonic()
                    result = await redis.delete(*keys_bytes)
                    total_delete_time += time.monotonic() - delete_start
                    deleted_count += result
                    logger.debug(f'Deleted {result} keys in batch.')
                    
                cursor = next_cursor_bytes
                try:
                    if isinstance(cursor, bytes) and cursor == b'0':
                        break
                    if isinstance(cursor, int) and cursor == 0:
                        break
                except ValueError:
                    logger.error(f'Invalid cursor type from SCAN: {type(cursor)}')
                    break
                    
            metrics.track_memory('duration', operation_type='redis_clear', value=total_scan_time + total_delete_time)
            logger.info(f'Finished clearing Redis memory for {operation_desc}. Deleted {deleted_count} keys. Scan took {total_scan_time:.4f}s, Delete took {total_delete_time:.4f}s.')
            return True
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f"Failed to clear context {context_id or 'all'}"
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
        operation_desc = f"context '{context_id or 'all'}'" if context_id else "all contexts"
        pattern_desc = f" with pattern '{pattern}'" if pattern else ""
        logger.debug(f'Listing keys for {operation_desc}{pattern_desc}')
        
        try:
            redis = await self._get_redis()
            
            if context_id:
                base_pattern = f'memory:{context_id}:'
            else:
                base_pattern = 'memory:*:'
                
            search_pattern = f'{base_pattern}{pattern or "*"}'
            logger.debug(f'Using Redis SCAN pattern: {search_pattern}')
            
            cursor: Union[bytes, int] = b'0'
            all_keys: List[str] = []
            start_time = time.monotonic()
            
            while True:
                current_cursor_int = 0
                try:
                    if isinstance(cursor, bytes):
                        current_cursor_int = int(cursor.decode())
                    elif isinstance(cursor, int):
                        current_cursor_int = cursor
                except ValueError:
                    pass
                    
                next_cursor_bytes, keys_bytes = await redis.scan(cursor=current_cursor_int, match=search_pattern, count=1000)
                
                for full_key_bytes in keys_bytes:
                    full_key_str = full_key_bytes.decode('utf-8')
                    parts = full_key_str.split(':', 2)
                    if len(parts) == 3:
                        all_keys.append(parts[2])
                    else:
                        logger.warning(f'Found key with unexpected format during list_keys: {full_key_str}')
                        
                cursor = next_cursor_bytes
                try:
                    if isinstance(cursor, bytes) and cursor == b'0':
                        break
                    if isinstance(cursor, int) and cursor == 0:
                        break
                except ValueError:
                    break
                    
            metrics.track_memory('duration', operation_type='redis_list_keys', value=time.monotonic() - start_time)
            logger.debug(f'Found {len(all_keys)} matching keys for {operation_desc}{pattern_desc}.')
            return all_keys
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                f"Failed to list keys for context {context_id or 'all'}"
            )
            error.log_error(logger)
            return []

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'exists'})
    async def exists(self, key: str, context_id: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: The key to check
            context_id: The context identifier
            
        Returns:
            bool: True if key exists
        """
        metrics.track_memory('operations', operation_type='exists')
        full_key = generate_memory_key(key, context_id)
        logger.debug(f"Checking existence for key: '{key}' (Redis key: '{full_key}')")
        
        try:
            redis = await self._get_redis()
            start_time = time.monotonic()
            result: int = await redis.exists(full_key)
            metrics.track_memory('duration', operation_type='redis_exists', value=time.monotonic() - start_time)
            
            key_exists = result > 0
            logger.debug(f"Key '{full_key}' exists: {key_exists}")
            return key_exists
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                f"Failed to check existence for key '{key}' in context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'bulk_load'})
    async def bulk_load(self, keys: List[str], context_id: str, default: Any=None) -> Dict[str, Any]:
        """
        Load multiple keys at once.
        
        Args:
            keys: List of keys to load
            context_id: The context identifier
            default: Default value for keys not found
            
        Returns:
            Dict[str, Any]: Dictionary of key-value pairs
        """
        metrics.track_memory('operations', operation_type='bulk_load')
        logger.debug(f"Bulk loading {len(keys)} keys for context '{context_id}'")
        
        if not keys:
            return {}
            
        try:
            redis = await self._get_redis()
            full_keys: List[str] = [generate_memory_key(k, context_id) for k in keys]
            
            start_time = time.monotonic()
            values: List[Optional[bytes]] = await redis.mget(*full_keys)
            metrics.track_memory('duration', operation_type='redis_mget', value=time.monotonic() - start_time)
            
            result: Dict[str, Any] = {}
            deserialize_tasks: List[Tuple[str, asyncio.Task[Any]]] = []
            
            for i, key in enumerate(keys):
                if values[i] is not None:
                    task: asyncio.Task[Any] = asyncio.create_task(
                        deserialize_data(values[i], default), 
                        name=f'deserialize_{key}'
                    )
                    deserialize_tasks.append((key, task))
                else:
                    result[key] = default
                    metrics.track_cache('misses', cache_type='redis_memory_bulk')
            
            if deserialize_tasks:
                gathered_results = await asyncio.gather(*(task for _, task in deserialize_tasks))
                for i, (key, _) in enumerate(deserialize_tasks):
                    result[key] = gathered_results[i]
                    
            logger.debug(f"Bulk load completed for context '{context_id}'. Loaded {len(result) - list(result.values()).count(default)} keys.")
            return result
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_RETRIEVAL_ERROR, 
                f"Failed to bulk load keys in context '{context_id}'"
            )
            error.log_error(logger)
            return {k: default for k in keys}

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'bulk_save'})
    async def bulk_save(self, data: Dict[str, Any], context_id: str, ttl: Optional[int]=None) -> bool:
        """
        Save multiple key-value pairs at once.
        
        Args:
            data: Dictionary of key-value pairs to save
            context_id: The context identifier
            ttl: Time-to-live in seconds, or None for no expiration
            
        Returns:
            bool: True if all saves were successful
        """
        metrics.track_memory('operations', operation_type='bulk_save')
        logger.debug(f"Bulk saving {len(data)} keys for context '{context_id}' with TTL: {ttl}")
        
        if not data:
            return True
            
        try:
            redis = await self._get_redis()
            
            async with redis.pipeline(transaction=False) as pipe:
                effective_ttl = ExpirationPolicy.get_ttl(self.default_ttl, ttl)
                serialize_tasks: Dict[str, asyncio.Task[bytes]] = {}
                
                for key, value in data.items():
                    task: asyncio.Task[bytes] = asyncio.create_task(
                        serialize_data(value), 
                        name=f'serialize_{key}'
                    )
                    serialize_tasks[key] = task
                
                serialized_data: Dict[str, bytes] = {}
                total_size_bytes = 0
                serialized_results = await asyncio.gather(*serialize_tasks.values())
                keys_in_order = list(serialize_tasks.keys())
                
                for i, key in enumerate(keys_in_order):
                    value_bytes = serialized_results[i]
                    serialized_data[key] = value_bytes
                    total_size_bytes += len(value_bytes)
                    
                    full_key = generate_memory_key(key, context_id)
                    if effective_ttl is not None and effective_ttl > 0:
                        pipe.setex(full_key, effective_ttl, value_bytes)
                    else:
                        pipe.set(full_key, value_bytes)
                
                start_time = time.monotonic()
                results: List[Any] = await pipe.execute()
                metrics.track_memory('duration', operation_type='redis_pipeline_set', value=time.monotonic() - start_time)
            
            metrics.track_memory('size', memory_type='redis', value=total_size_bytes)
            success = all((bool(res) for res in results))
            
            if success:
                logger.debug(f"Bulk save successful for {len(data)} keys in context '{context_id}'.")
            else:
                failed_indices = [i for i, res in enumerate(results) if not bool(res)]
                logger.error(f"Bulk save failed for {len(failed_indices)} out of {len(data)} keys in context '{context_id}'. Failed indices: {failed_indices}")
                
            return success
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.MEMORY_STORAGE_ERROR, 
                f"Failed to bulk save keys in context '{context_id}'"
            )
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'get_stats'})
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Redis memory implementation.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        logger.debug('Fetching Redis memory stats...')
        
        try:
            redis = await self._get_redis()
            
            start_time = time.monotonic()
            info: Dict[str, Any] = await redis.info(section='memory')
            dbsize: int = await redis.dbsize()
            metrics.track_memory('duration', operation_type='redis_info_dbsize', value=time.monotonic() - start_time)
            
            memory_keys_count = 0
            scan_start_time = time.monotonic()
            cursor: Union[bytes, int] = b'0'
            
            while True:
                current_cursor_int = 0
                try:
                    if isinstance(cursor, bytes):
                        current_cursor_int = int(cursor.decode())
                    elif isinstance(cursor, int):
                        current_cursor_int = cursor
                except ValueError:
                    pass
                    
                next_cursor_bytes, keys_bytes = await redis.scan(cursor=current_cursor_int, match='memory:*', count=5000)
                memory_keys_count += len(keys_bytes)
                cursor = next_cursor_bytes
                
                try:
                    if isinstance(cursor, bytes) and cursor == b'0':
                        break
                    if isinstance(cursor, int) and cursor == 0:
                        break
                except ValueError:
                    break
                    
            metrics.track_memory('duration', operation_type='redis_scan_memkeys', value=time.monotonic() - scan_start_time)
            
            stats = {
                'implementation_type': self.__class__.__name__,
                'redis_total_keys': dbsize,
                'redis_memory_namespace_keys': memory_keys_count,
                'redis_memory_used_bytes': info.get('used_memory', 0),
                'redis_memory_peak_bytes': info.get('used_memory_peak', 0),
                'redis_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0.0),
                'redis_evicted_keys': info.get('evicted_keys', 0),
                'redis_expired_keys': info.get('expired_keys', 0)
            }
            
            logger.debug(f'Redis memory stats retrieved: {stats}')
            return stats
            
        except Exception as e:
            logger.error(f'Failed to get Redis stats: {str(e)}', exc_info=True)
            return {
                'implementation_type': self.__class__.__name__, 
                'error': f'Failed to retrieve Redis stats: {str(e)}'
            }

    async def get_lock(self, name: str, expire_time: int=10) -> AsyncLock:
        """
        Get a distributed lock using Redis.
        
        Args:
            name: Lock name
            expire_time: Lock expiration time in seconds
            
        Returns:
            AsyncLock: Lock object
        """
        logger.debug(f"Getting async lock named '{name}' with expire time {expire_time}s")
        redis = await self._get_redis()
        lock_key_name = f'memory_lock:{name}'
        return AsyncLock(redis, lock_key_name, expire_time)
import asyncio
import time
import threading
from typing import Optional, Any, Dict, Tuple, Type, Union, Callable

import redis
import redis.asyncio as aioredis
import aiohttp

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.config.metrics import timed_metric, MEMORY_OPERATION_DURATION


settings = get_settings()
logger = get_logger(__name__)

_redis_sync_pool = None
_redis_async_pool = None
_http_session_pool = None

_redis_sync_pool_lock = threading.Lock()
_redis_async_pool_lock = threading.Lock()
_http_session_pool_lock = threading.Lock()

@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "connect_redis"})
def get_redis_connection() -> redis.Redis:
    global _redis_sync_pool
    
    with _redis_sync_pool_lock:
        if _redis_sync_pool is None:
            logger.debug("Creating new Redis connection pool")
            pool_kwargs = {
                "max_connections": settings.REDIS_CONNECTION_POOL_SIZE,
                "decode_responses": True,
            }
            
            if settings.REDIS_PASSWORD:
                pool_kwargs["password"] = settings.REDIS_PASSWORD
                
            _redis_sync_pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                **pool_kwargs
            )
            
        return redis.Redis(connection_pool=_redis_sync_pool)

@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "connect_redis_async"})
async def get_redis_async_connection() -> aioredis.Redis:
    global _redis_async_pool
    
    with _redis_async_pool_lock:
        if _redis_async_pool is None:
            logger.debug("Creating new Redis async connection pool")
            pool_kwargs = {
                "max_connections": settings.REDIS_CONNECTION_POOL_SIZE,
                "decode_responses": True,
            }
            
            if settings.REDIS_PASSWORD:
                pool_kwargs["password"] = settings.REDIS_PASSWORD
                
            _redis_async_pool = aioredis.ConnectionPool.from_url(
                settings.REDIS_URL,
                **pool_kwargs
            )
            
        return aioredis.Redis(connection_pool=_redis_async_pool)

async def get_http_session() -> aiohttp.ClientSession:
    global _http_session_pool
    
    with _http_session_pool_lock:
        if _http_session_pool is None:
            connector = aiohttp.TCPConnector(
                limit=100,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                force_close=True,
                limit_per_host=10,
            )
            
            timeout = aiohttp.ClientTimeout(
                total=settings.REQUEST_TIMEOUT,
                connect=10.0,
                sock_connect=10.0,
                sock_read=settings.REQUEST_TIMEOUT,
            )
            
            _http_session_pool = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=True,
                trust_env=True,
            )
            
            logger.debug("Created new HTTP session pool")

    return _http_session_pool

class ConnectionManager:
    @staticmethod
    def close_all_connections():
        global _redis_sync_pool, _redis_async_pool, _http_session_pool
    
        if _redis_sync_pool is not None:
            try:
                logger.info("Closing Redis sync connection pool")
                _redis_sync_pool.disconnect()
                _redis_sync_pool = None
            except Exception as e:
                logger.error(f"Error closing Redis sync connection pool: {e}")
    
        if _redis_async_pool is not None:
            try:
                logger.info("Closing Redis async connection pool")
                task = asyncio.create_task(_close_redis_async_pool())
                logger.warning(
                    "Redis async connection pool cleanup scheduled. "
                    "For proper cleanup, use close_all_connections_async() in async context."
                )
                _redis_async_pool = None
            except Exception as e:
                logger.error(f"Error scheduling Redis async pool cleanup: {e}")
    
        if _http_session_pool is not None:
            try:
                logger.info("Closing HTTP session pool")
                task = asyncio.create_task(_close_http_session_pool())
                logger.warning(
                    "HTTP session pool cleanup scheduled. "
                    "For proper cleanup, use close_all_connections_async() in async context."
                )
                _http_session_pool = None
            except Exception as e:
                logger.error(f"Error scheduling HTTP session pool cleanup: {e}")
    
    @staticmethod
    async def close_all_connections_async():
        global _redis_sync_pool
        global _redis_async_pool
        global _http_session_pool
        
        if _redis_sync_pool is not None:
            logger.info("Closing Redis sync connection pool")
            _redis_sync_pool.disconnect()
            _redis_sync_pool = None
            
        if _redis_async_pool is not None:
            logger.info("Closing Redis async connection pool")
            await _close_redis_async_pool()
            _redis_async_pool = None
            
        if _http_session_pool is not None:
            logger.info("Closing HTTP session pool")
            await _close_http_session_pool()
            _http_session_pool = None
            
async def _close_redis_async_pool():
    global _redis_async_pool
    if _redis_async_pool is not None:
        await _redis_async_pool.disconnect()
        _redis_async_pool = None
        
async def _close_http_session_pool():
    global _http_session_pool
    if _http_session_pool is not None:
        await _http_session_pool.close()
        _http_session_pool = None
        
def setup_connection_pools():
    _ = get_redis_connection()
    logger.info("Initialized Redis connection pool")
    logger.info("HTTP connection pool will be initialized on first use")
    
def cleanup_connection_pools():
    ConnectionManager.close_all_connections()
    logger.info("Closed all connection pools")
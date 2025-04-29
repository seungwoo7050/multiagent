"""
Connection management for external services including Redis and HTTP.
Provides connection pooling and proper resource management.
"""
import asyncio
import contextlib
import threading
from typing import Optional, Dict, Any, AsyncGenerator, Generator

import redis
import redis.asyncio as aioredis
import aiohttp
from src.config.settings import get_settings
from src.config.logger import get_logger
# metrics.py 변경 사항 반영: get_metrics_manager와 MEMORY_METRICS 임포트
from src.config.metrics import get_metrics_manager, MEMORY_METRICS
from src.config.errors import ConnectionError, ErrorCode

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages connection pools for various services.
    Implements the singleton pattern for global access with proper resource management.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConnectionManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.settings = get_settings()
        # metrics.py 변경 사항 반영: MetricsManager 인스턴스 가져오기
        self.metrics_manager = get_metrics_manager()

        # Initialize connection pools as None
        self._redis_sync_pool = None
        self._redis_sync_pool_lock = threading.Lock()

        self._redis_async_pool = None
        self._redis_async_pool_lock = asyncio.Lock()

        self._http_session_pool = None
        self._http_session_pool_lock = asyncio.Lock()

        self._initialized = True
        logger.debug("ConnectionManager initialized")

    # metrics.py 변경 사항 반영: 데코레이터 호출 방식 및 메트릭 이름 변경
    @get_metrics_manager().timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'connect_redis'})
    def get_redis_connection(self) -> redis.Redis:
        """Get a Redis connection from the pool."""
        with self._redis_sync_pool_lock:
            if self._redis_sync_pool is None:
                logger.debug('Creating new Redis sync connection pool')
                pool_kwargs = {
                    'max_connections': self.settings.REDIS_CONNECTION_POOL_SIZE,
                    'decode_responses': True
                }
                if self.settings.REDIS_PASSWORD:
                    pool_kwargs['password'] = self.settings.REDIS_PASSWORD

                try:
                    self._redis_sync_pool = redis.ConnectionPool.from_url(
                        self.settings.REDIS_URL, **pool_kwargs
                    )
                    logger.info(
                        f'Redis sync connection pool initialized with '
                        f'max_connections={self.settings.REDIS_CONNECTION_POOL_SIZE}'
                    )
                except Exception as e:
                    error_msg = f"Failed to initialize Redis connection pool: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise ConnectionError(
                        code=ErrorCode.REDIS_CONNECTION_ERROR,
                        message=error_msg,
                        original_error=e,
                        service="redis"
                    ) #

        return redis.Redis(connection_pool=self._redis_sync_pool)

    @contextlib.contextmanager
    def redis_connection(self) -> Generator[redis.Redis, None, None]:
        """Context manager for Redis connections."""
        conn = None
        try:
            conn = self.get_redis_connection()
            yield conn
        except Exception as e:
            error_msg = f"Error with Redis connection: {str(e)}"
            logger.error(error_msg, exc_info=True) #
            raise ConnectionError(
                code=ErrorCode.REDIS_OPERATION_ERROR,
                message=error_msg,
                original_error=e,
                service="redis"
            ) #
        finally:
            # Connection goes back to pool automatically
            pass

    # metrics.py 변경 사항 반영: 데코레이터 호출 방식 및 메트릭 이름 변경
    @get_metrics_manager().timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'connect_redis_async'})
    async def get_redis_async_connection(self) -> aioredis.Redis:
        """Get an async Redis connection from the pool."""
        async with self._redis_async_pool_lock:
            if self._redis_async_pool is None:
                logger.debug('Creating new Redis async connection pool') #
                pool_kwargs = {
                    'max_connections': self.settings.REDIS_CONNECTION_POOL_SIZE,
                    'decode_responses': True
                }
                if self.settings.REDIS_PASSWORD:
                    pool_kwargs['password'] = self.settings.REDIS_PASSWORD

                try:
                    self._redis_async_pool = aioredis.ConnectionPool.from_url(
                        self.settings.REDIS_URL, **pool_kwargs
                    )
                    logger.info(
                        f'Redis async connection pool initialized with '
                        f'max_connections={self.settings.REDIS_CONNECTION_POOL_SIZE}'
                    ) #
                except Exception as e:
                    error_msg = f"Failed to initialize Redis async connection pool: {str(e)}"
                    logger.error(error_msg, exc_info=True) #
                    raise ConnectionError(
                        code=ErrorCode.REDIS_CONNECTION_ERROR,
                        message=error_msg,
                        original_error=e,
                        service="redis-async"
                    ) #

        return aioredis.Redis(connection_pool=self._redis_async_pool)

    @contextlib.asynccontextmanager
    async def redis_async_connection(self) -> AsyncGenerator[aioredis.Redis, None]:
        """Async context manager for Redis connections."""
        conn = None
        try:
            conn = await self.get_redis_async_connection()
            yield conn
        except Exception as e:
            error_msg = f"Error with async Redis connection: {str(e)}"
            logger.error(error_msg, exc_info=True) #
            raise ConnectionError(
                code=ErrorCode.REDIS_OPERATION_ERROR,
                message=error_msg,
                original_error=e,
                service="redis-async"
            ) #
        finally:
            # Connection goes back to pool automatically
            pass

    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get an HTTP session from the pool."""
        async with self._http_session_pool_lock:
            if self._http_session_pool is None:
                logger.debug('Creating new aiohttp ClientSession pool') #

                connector = aiohttp.TCPConnector(
                    limit=100,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                    force_close=True,
                    limit_per_host=10
                )

                request_timeout = 30.0
                if self.settings and hasattr(self.settings, 'REQUEST_TIMEOUT'):
                    request_timeout = self.settings.REQUEST_TIMEOUT #

                timeout = aiohttp.ClientTimeout(
                    total=request_timeout,
                    connect=10.0,
                    sock_connect=10.0,
                    sock_read=request_timeout
                )

                try:
                    self._http_session_pool = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        raise_for_status=True
                    )
                    logger.info(f'aiohttp ClientSession initialized with total_timeout={request_timeout}s') #
                except Exception as e:
                    error_msg = f"Failed to initialize HTTP session: {str(e)}"
                    logger.error(error_msg, exc_info=True) #
                    raise ConnectionError(
                        code=ErrorCode.HTTP_ERROR,
                        message=error_msg,
                        original_error=e,
                        service="http"
                    ) #

        return self._http_session_pool

    @contextlib.asynccontextmanager
    async def http_session(self) -> AsyncGenerator[aiohttp.ClientSession, None]:
        """Async context manager for HTTP sessions."""
        try:
            session = await self.get_http_session()
            yield session
        except Exception as e:
            error_msg = f"Error with HTTP session: {str(e)}"
            logger.error(error_msg, exc_info=True) #
            raise ConnectionError(
                code=ErrorCode.HTTP_ERROR,
                message=error_msg,
                original_error=e,
                service="http"
            ) #

    def close_sync_connections(self) -> None:
        """Close all synchronous connections."""
        if self._redis_sync_pool is not None:
            try:
                logger.info('Closing Redis sync connection pool...') #
                self._redis_sync_pool.disconnect()
                self._redis_sync_pool = None
                logger.debug('Redis sync pool disconnected') #
            except Exception as e:
                logger.error(f'Error closing Redis sync connection pool: {e}', exc_info=True) #

    async def close_async_connections(self) -> None:
        """Close all asynchronous connections."""
        tasks = []

        # Close Redis async pool
        if self._redis_async_pool is not None:
            async def close_redis_async():
                try:
                    logger.info('Closing Redis async connection pool...') #
                    await self._redis_async_pool.disconnect()
                    logger.debug('Redis async pool disconnected') #
                except Exception as e:
                    logger.error(f'Error disconnecting Redis async pool: {e}', exc_info=True) #
                finally:
                    self._redis_async_pool = None

            tasks.append(close_redis_async())

        # Close HTTP session pool
        if self._http_session_pool is not None:
            async def close_http_session():
                try:
                    logger.info('Closing HTTP session pool...') #
                    await self._http_session_pool.close()
                    logger.debug('aiohttp ClientSession closed') #
                except Exception as e:
                    logger.error(f'Error closing aiohttp ClientSession: {e}', exc_info=True) #
                finally:
                    self._http_session_pool = None

            tasks.append(close_http_session())

        # Wait for all close operations to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info('All async connection pools closed') #

    async def close_all_connections(self) -> None:
        """Close all connections, both sync and async."""
        self.close_sync_connections()
        await self.close_async_connections()
        logger.info('All connection pools closed') #


# Convenience functions for global access
def get_connection_manager() -> ConnectionManager:
    """Get the singleton connection manager instance."""
    return ConnectionManager()


def setup_connection_pools() -> None:
    """Initialize connection pools."""
    manager = get_connection_manager()
    try:
        # Pre-initialize Redis connection pool
        with manager.redis_connection() as _:
            logger.info('Initialized Redis sync connection pool') #
    except Exception as e:
        logger.error(f'Failed to initialize Redis sync connection pool during setup: {e}', exc_info=True) #

    logger.info('Async Redis and HTTP connection pools will be initialized on first use') #


async def cleanup_connection_pools() -> None:
    """Clean up all connection pools."""
    logger.info('Cleaning up connection pools...') #
    manager = get_connection_manager()
    await manager.close_all_connections()
    logger.info('Connection pool cleanup complete') #
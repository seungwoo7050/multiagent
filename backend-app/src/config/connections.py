import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
from typing import Optional

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.config.errors import ConnectionError, ErrorCode                         

logger = get_logger(__name__)
settings = get_settings()

                      
_redis_pool: Optional[ConnectionPool] = None

async def setup_connection_pools():
    """
    애플리케이션 시작 시 Redis 연결 풀을 설정합니다.
    이 함수는 FastAPI의 lifespan 등 애플리케이션 시작 지점에서 호출되어야 합니다.
    """
    global _redis_pool
    if _redis_pool is not None:
        logger.warning("Redis connection pool already initialized.")
        return

    try:
        logger.info(f"Setting up Redis connection pool for URL: {settings.REDIS_URL} (DB: {settings.REDIS_DB})")
                                                        
                                                                                                  
        _redis_pool = ConnectionPool.from_url(
            settings.REDIS_URL,
                                                           
                                                            
            max_connections=settings.REDIS_CONNECTION_POOL_SIZE,
            decode_responses=False,                             
            health_check_interval=30                   
        )
                    
        conn = aioredis.Redis(connection_pool=_redis_pool)
        await conn.ping()
                                             
                                  
        logger.info("Redis connection pool initialized and ping successful.")
    except Exception as e:
        logger.error(f"Failed to initialize Redis connection pool: {e}", exc_info=True)
        _redis_pool = None                 
                                           
        raise ConnectionError(
            code=ErrorCode.REDIS_CONNECTION_ERROR,
            message=f"Failed to connect to Redis at {settings.REDIS_URL}: {e}",
            original_error=e,
            service="redis"
        ) from e

async def cleanup_connection_pools():
    """
    애플리케이션 종료 시 Redis 연결 풀을 정리합니다.
    이 함수는 FastAPI의 lifespan 등 애플리케이션 종료 지점에서 호출되어야 합니다.
    """
    global _redis_pool
    if _redis_pool:
        logger.info("Closing Redis connection pool...")
        try:
                                          
            await _redis_pool.disconnect()
            _redis_pool = None
            logger.info("Redis connection pool closed.")
        except Exception as e:
            logger.error(f"Error closing Redis connection pool: {e}", exc_info=True)
    else:
        logger.info("Redis connection pool was not initialized or already closed.")

async def get_redis_async_connection() -> aioredis.Redis:
    """
    연결 풀에서 비동기 Redis 클라이언트 인스턴스를 가져옵니다.
    풀이 초기화되지 않았으면 RuntimeError를 발생시킵니다.

    Returns:
        aioredis.Redis: 비동기 Redis 클라이언트 인스턴스.

    Raises:
        RuntimeError: 연결 풀이 초기화되지 않은 경우.
        ConnectionError: 연결 풀에서 클라이언트 생성 중 오류 발생 시.
    """
    global _redis_pool
    if _redis_pool is None:
        logger.critical("Attempted to get Redis connection before the pool was initialized.")
        raise RuntimeError(
            "Redis connection pool is not available. "
            "Ensure setup_connection_pools() is called during application startup."
        )
    try:
                                                
                                                                                            
        return aioredis.Redis(connection_pool=_redis_pool)
    except Exception as e:
                            
        logger.error(f"Failed to create Redis client instance from pool: {e}", exc_info=True)
        raise ConnectionError(
            code=ErrorCode.REDIS_CONNECTION_ERROR,
            message=f"Failed to get Redis connection from pool: {e}",
            original_error=e,
            service="redis"
        ) from e
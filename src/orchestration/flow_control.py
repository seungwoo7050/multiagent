import asyncio
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar

import redis.asyncio as aioredis
from pydantic import BaseModel, Field, field_validator

from src.config.connections import get_connection_manager
from src.config.errors import BaseError, ErrorCode, convert_exception
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings

# Type variables for better type hinting
T = TypeVar('T')
R = TypeVar('R')

# Initialize global dependencies
metrics = get_metrics_manager()
logger = get_logger(__name__)
settings = get_settings()
connection_manager = get_connection_manager()

class RateLimitConfig(BaseModel):
    """Configuration for rate limiting parameters"""
    rate: float = Field(100.0, description='Requests per second allowed')
    burst: int = Field(120, description='Burst capacity (determines delay tolerance)')
    period: float = Field(1.0, description='Period in seconds (usually 1.0 for per-second rate)')

    @field_validator('rate', 'burst', 'period')
    def check_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Rate limit parameters must be positive')
        return v

class BackpressureStrategy(str, Enum):
    """Strategies for handling backpressure"""
    REJECT = 'reject'
    # Future implementation could add: QUEUE, THROTTLE, etc.

class BackpressureConfig(BaseModel):
    """Configuration for backpressure handling"""
    strategy: BackpressureStrategy = Field(
        default=BackpressureStrategy.REJECT, 
        description='Strategy to apply (only REJECT supported by RedisRateLimiter)'
    )
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    @field_validator('strategy')
    def check_strategy(cls, v: BackpressureStrategy) -> BackpressureStrategy:
        if v != BackpressureStrategy.REJECT:
            logger.warning('RedisRateLimiter currently only supports REJECT strategy. Forcing to REJECT.')
            return BackpressureStrategy.REJECT
        return v

class BackpressureMetrics(BaseModel):
    """Metrics for monitoring backpressure controller performance"""
    total_requests: int = 0
    rejected_requests: int = 0
    last_rejection_time: Optional[float] = None

    model_config = {
        "arbitrary_types_allowed": True,
    }

class RedisRateLimiter:
    """
    Rate limiter based on the Generic Cell Rate Algorithm (GCRA) implemented with Redis.
    Uses a Lua script for atomic operations to ensure accurate rate limiting.
    """
    # GCRA implementation in Lua for atomic operations
    GCRA_LUA_SCRIPT: str = """
        local key = KEYS[1]
        local now_ms = tonumber(ARGV[1])
        local emission_interval_ms = tonumber(ARGV[2])
        local delay_tolerance_ms = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])

        -- Get Theoretical Arrival Time (TAT)
        local tat_str = redis.call('get', key)
        local tat = 0
        if tat_str then
            tat = tonumber(tat_str)
        end

        -- TAT cannot be in the past (reset to current time if needed)
        tat = math.max(tat, now_ms)

        -- Calculate new TAT after this request
        local new_tat = tat + (cost * emission_interval_ms)

        -- Calculate earliest time this request could be allowed
        local allow_at = new_tat - delay_tolerance_ms

        -- Determine if request is allowed (allow_at <= now_ms)
        if allow_at <= now_ms then
            -- Allow: Save new TAT and set TTL for memory management
            local ttl_ms = math.max(1, math.ceil(new_tat - now_ms + delay_tolerance_ms))
            redis.call('set', key, new_tat, 'PX', ttl_ms) -- PX sets TTL in milliseconds
            return 1 -- Allowed
        else
            -- Reject: Don't modify TAT
            return 0 -- Rejected
        end
    """
    
    _script_sha: Optional[str] = None
    _script_load_lock = asyncio.Lock()

    def __init__(self, name: str, config: Optional[BackpressureConfig]=None):
        """
        Initialize a Redis-based rate limiter.
        
        Args:
            name: Unique name for this rate limiter instance
            config: Optional configuration for rate limiting parameters
        """
        self.name: str = name
        self.config: BackpressureConfig = config or BackpressureConfig()
        self.metrics: BackpressureMetrics = BackpressureMetrics()
        self._redis: Optional[aioredis.Redis] = None
        self._redis_key: str = f'flow_control:rate_limit:{self.name}'
        
        # Calculate rate limiting parameters
        self.tau_ms: float = self.config.rate_limit.period / self.config.rate_limit.rate * 1000
        self.tv_ms: float = self.tau_ms * self.config.rate_limit.burst
        
        logger.info(
            f'RedisRateLimiter (Flow Control) initialized: {name}', 
            extra={
                'controller_name': name, 
                'strategy': self.config.strategy.value, 
                'rate': self.config.rate_limit.rate, 
                'burst': self.config.rate_limit.burst, 
                'redis_key': self._redis_key, 
                'tau_ms': self.tau_ms, 
                'tv_ms': self.tv_ms
            }
        )

    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection, initializing if needed"""
        if self._redis is None or not self._redis.is_connected:
            try:
                self._redis = await connection_manager.get_redis_async_connection()
                await self._load_lua_script()
            except Exception as e:
                error: BaseError = convert_exception(
                    e, 
                    ErrorCode.REDIS_CONNECTION_ERROR, 
                    f'Failed to get Redis connection for Flow Control {self.name}'
                )
                error.log_error(logger)
                raise error
                
        if self._redis is None:
            raise ConnectionError(f'Redis client failed to initialize for {self.name}')
            
        return self._redis

    async def _load_lua_script(self) -> None:
        """Load the GCRA Lua script into Redis if not already loaded"""
        if RedisRateLimiter._script_sha:
            try:
                redis = await self._get_redis()
                exists = await redis.script_exists(RedisRateLimiter._script_sha)
                if exists and exists[0]:
                    return
                else:
                    logger.warning(f'Cached Lua script SHA {RedisRateLimiter._script_sha} not found in Redis. Reloading.')
                    RedisRateLimiter._script_sha = None
            except Exception as e:
                logger.warning(f'Failed to check script existence, attempting reload: {e}')
                RedisRateLimiter._script_sha = None
                
        async with RedisRateLimiter._script_load_lock:
            if RedisRateLimiter._script_sha:
                try:
                    exists = await (await self._get_redis()).script_exists(RedisRateLimiter._script_sha)
                    if exists and exists[0]:
                        return
                except Exception:
                    pass
                    
            try:
                redis = await self._get_redis()
                sha_bytes: bytes = await redis.script_load(self.GCRA_LUA_SCRIPT)
                RedisRateLimiter._script_sha = sha_bytes.decode()
                logger.info(f'GCRA Lua script loaded into Redis. SHA: {RedisRateLimiter._script_sha}')
            except Exception as e:
                logger.error(f'Failed to load GCRA Lua script into Redis: {e}', exc_info=True)
                RedisRateLimiter._script_sha = None

    async def acquire(self, priority: int=0, cost: int=1) -> bool:
        """
        Attempt to acquire a rate limit permit.
        
        Args:
            priority: Priority level (higher priority may get preferential treatment)
            cost: Cost of this request (defaults to 1)
            
        Returns:
            bool: True if acquired, False if rejected
        """
        self.metrics.total_requests += 1
        
        try:
            redis: aioredis.Redis = await self._get_redis()
            await self._load_lua_script()
            
            if not RedisRateLimiter._script_sha:
                logger.error(f'Lua script SHA not available for rate limiter {self.name}. Denying request.')
                allowed = False
            else:
                now_ms: int = int(time.time() * 1000)
                result = await redis.evalsha(
                    RedisRateLimiter._script_sha, 
                    keys=[self._redis_key], 
                    args=[now_ms, int(self.tau_ms), int(self.tv_ms), cost]
                )
                allowed = int(result) == 1
                
            if not allowed:
                self.metrics.rejected_requests += 1
                self.metrics.last_rejection_time = time.time()
                metrics.track_task('rejections', reason='rate_limit')
                logger.debug(f'Rate limit exceeded for {self.name}. Request rejected.')
                return False
            else:
                logger.debug(f'Rate limit check passed for {self.name}. Request allowed.')
                return True
                
        except Exception as e:
            logger.error(f'Redis error during rate limit check for {self.name}: {e}', exc_info=True)
            self.metrics.rejected_requests += 1
            self.metrics.last_rejection_time = time.time()
            metrics.track_task('rejections', reason='flow_control_error')
            return False

    async def release(self) -> None:
        """Release a permit (no-op for REJECT strategy)"""
        # No action needed for REJECT strategy
        # Would be implemented for other strategies like QUEUE

    async def execute(self, func: Callable[..., Coroutine[Any, Any, R]], *args: Any, priority: int=0, cost: int=1, **kwargs: Any) -> Optional[R]:
        """
        Execute a function with rate limiting.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            priority: Priority level for rate limiting
            cost: Cost of this execution
            **kwargs: Keyword arguments for the function
            
        Returns:
            Optional[R]: Function result if allowed, None if rejected
            
        Raises:
            BackpressureRejectedError: If request is rejected by rate limiter
        """
        acquired: bool = await self.acquire(priority=priority, cost=cost)
        
        if acquired:
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f'Error executing function under flow control {self.name}: {e}', exc_info=True)
                raise
        else:
            logger.debug(f'Execution rejected by flow controller {self.name}')
            raise BackpressureRejectedError(f'Request rejected by flow controller {self.name}')

    async def __aenter__(self) -> 'RedisRateLimiter':
        """Async context manager entry - acquire rate limit permit"""
        acquired: bool = await self.acquire(cost=1)
        if not acquired:
            raise BackpressureRejectedError(f'Request rejected by backpressure controller {self.name}')
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - release permit"""
        await self.release()

class BackpressureRejectedError(Exception):
    """Raised when a request is rejected by the backpressure controller"""

# Global registry of flow controllers
_flow_controllers: Dict[str, RedisRateLimiter] = {}
_controller_lock = asyncio.Lock()

async def get_flow_controller(name: str, config: Optional[BackpressureConfig]=None) -> RedisRateLimiter:
    """
    Get or create a flow controller by name.
    
    Args:
        name: Controller name
        config: Optional configuration to use or update
        
    Returns:
        RedisRateLimiter: Flow controller instance
    """
    global _flow_controllers
    
    async with _controller_lock:
        if name not in _flow_controllers:
            _flow_controllers[name] = RedisRateLimiter(name, config)
            logger.info(f"Created flow controller '{name}'")
        elif config is not None:
            existing_controller = _flow_controllers[name]
            if existing_controller.config != config:
                logger.info(f"Updating configuration for flow controller '{name}'")
                existing_controller.config = config
                existing_controller.tau_ms = config.rate_limit.period / config.rate_limit.rate * 1000
                existing_controller.tv_ms = existing_controller.tau_ms * config.rate_limit.burst
                
        controller_instance = _flow_controllers.get(name)
        if controller_instance is None:
            raise RuntimeError(f"Failed to get or create flow controller '{name}'")
            
        return controller_instance

async def with_flow_control(controller_name: str, func: Callable[..., Coroutine[Any, Any, R]], *args: Any, priority: int=0, cost: int=1, config: Optional[BackpressureConfig]=None, **kwargs: Any) -> Optional[R]:
    """
    Execute a function with flow control.
    
    Args:
        controller_name: Name of the flow controller to use
        func: Async function to execute
        *args: Positional arguments for the function
        priority: Priority level
        cost: Cost of this execution
        config: Optional flow controller configuration
        **kwargs: Keyword arguments for the function
        
    Returns:
        Optional[R]: Function result
    """
    controller: RedisRateLimiter = await get_flow_controller(controller_name, config)
    return await controller.execute(func, *args, priority=priority, cost=cost, **kwargs)
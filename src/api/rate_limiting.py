from fastapi import Request, HTTPException, Depends, status
from typing import Optional, Callable, Any
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.orchestration.flow_control import get_flow_controller, RedisRateLimiter, BackpressureConfig, RateLimitConfig, BackpressureRejectedError
settings = get_settings()
logger = get_logger(__name__)
DEFAULT_RATE_LIMIT_PER_MINUTE = getattr(settings, 'API_RATE_LIMIT_PER_MINUTE', 60)
DEFAULT_RATE_LIMIT_BURST = int(DEFAULT_RATE_LIMIT_PER_MINUTE * 1.5)
default_rate_per_second = DEFAULT_RATE_LIMIT_PER_MINUTE / 60.0
DEFAULT_LIMITER_CONFIG = BackpressureConfig(rate_limit=RateLimitConfig(rate=default_rate_per_second, burst=DEFAULT_RATE_LIMIT_BURST, period=1.0))

async def rate_limiter_dependency(request: Request, identifier: Optional[Callable[[Request], str]]=None, custom_config: Optional[BackpressureConfig]=None) -> None:
    client_id: str
    if identifier:
        try:
            client_id = identifier(request)
        except Exception as e:
            logger.error(f'Failed to get client identifier from custom function: {e}', exc_info=True)
            client_id = request.client.host if request.client else 'unknown_client'
    else:
        client_id = request.client.host if request.client else 'unknown_client'
    config_to_use = custom_config or DEFAULT_LIMITER_CONFIG
    controller_name = f'api:{client_id}'
    try:
        limiter: RedisRateLimiter = await get_flow_controller(controller_name, config=config_to_use)
    except Exception as e:
        logger.exception(f"Failed to get or create flow controller '{controller_name}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Rate limiting service unavailable') from e
    try:
        allowed: bool = await limiter.acquire(cost=1)
        if not allowed:
            logger.warning(f"Rate limit exceeded for client '{client_id}' (Controller: {controller_name})")
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail='Rate limit exceeded. Please try again later.')
        logger.debug(f"Rate limit check passed for client '{client_id}' (Controller: {controller_name})")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking rate limit for client '{client_id}' (Controller: {controller_name}): {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to check rate limit') from e
RateLimiterDep = Depends(rate_limiter_dependency)

def get_rate_limiter(rate: float, burst: int, identifier: Optional[Callable[[Request], str]]=None) -> Callable[..., None]:
    custom_config = BackpressureConfig(rate_limit=RateLimitConfig(rate=rate, burst=burst, period=1.0))

    async def dependency_wrapper(request: Request):
        await rate_limiter_dependency(request=request, identifier=identifier, custom_config=custom_config)
    return Depends(dependency_wrapper)
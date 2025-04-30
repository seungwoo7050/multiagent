import asyncio
import functools
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast, Coroutine
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import get_metrics_manager, MEMORY_METRICS
from src.config.errors import ErrorCode, LLMError, BaseError, RETRYABLE_ERRORS

settings = get_settings()
logger = get_logger(__name__)
metrics = get_metrics_manager()

F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])
R = TypeVar('R')

DEFAULT_RETRYABLE_ERROR_CODES: Set[Union[ErrorCode, str]] = set(RETRYABLE_ERRORS)
LLM_RETRYABLE_ERROR_CODES: Set[Union[ErrorCode, str]] = {ErrorCode.LLM_TIMEOUT, ErrorCode.LLM_RATE_LIMIT, ErrorCode.CONNECTION_ERROR, ErrorCode.HTTP_ERROR, ErrorCode.NETWORK_ERROR, ErrorCode.TIMEOUT_ERROR}
DEFAULT_RETRYABLE_ERROR_CODES.update(LLM_RETRYABLE_ERROR_CODES)

def is_retryable_error(error: Exception, retryable_errors: Optional[Set[Union[ErrorCode, str]]]=None) -> bool:
    codes_to_check = retryable_errors if retryable_errors is not None else DEFAULT_RETRYABLE_ERROR_CODES
    retryable_code_values: Set[str] = {e.value if isinstance(e, ErrorCode) else e for e in codes_to_check}
    if isinstance(error, BaseError):
        error_code = error.code
        code_str = error_code.value if isinstance(error_code, ErrorCode) else error_code
        if code_str in retryable_code_values:
            logger.debug(f"Error code '{code_str}' matched retryable list.")
            return True
    error_name = type(error).__name__.lower()
    retryable_error_types = {'timeout', 'connectionerror', 'connecttimeout', 'readtimeout', 'connectionrefused', 'sockettimeout', 'requestexception', 'serviceunitavailable', 'throttlingerror', 'ratelimiterror', 'temporaryfailure', 'serveroverloaded', 'serviceunavailable'}
    for error_type_keyword in retryable_error_types:
        if error_type_keyword in error_name:
            logger.debug(f"Error type name '{error_name}' contains retryable keyword '{error_type_keyword}'.")
            return True
    error_msg = str(error).lower()
    retryable_phrases = {'status code: 429', 'too many requests', 'rate limit', 'status code: 500', 'status code: 502', 'status code: 503', 'status code: 504', 'internal server error', 'bad gateway', 'service unavailable', 'gateway timeout', 'connection reset', 'connection refused', 'timeout', 'request timed out', 'temporarily unavailable', 'server overloaded', 'server busy'}
    for phrase in retryable_phrases:
        if phrase in error_msg:
            logger.debug(f"Error message contains retryable phrase: '{phrase}'.")
            return True
    logger.debug(f"Error '{error_name}' (msg: '{error_msg[:100]}...') is considered non-retryable.")
    return False

def calculate_backoff(attempt: int, base_delay: float, max_delay: float, jitter: bool=True) -> float:
    delay = base_delay * 2 ** attempt
    delay = min(max_delay, delay)
    if jitter:
        jitter_factor = random.random() * 0.5 - 0.25
        delay = delay * (1 + jitter_factor)
        delay = max(0, delay)
    return delay

def retry_with_exponential_backoff(max_retries: int=3, base_delay: float=0.5, max_delay: float=30.0, jitter: bool=True, retryable_errors: Optional[Set[Union[ErrorCode, str]]]=None) -> Callable[[F], F]:

    def decorator(func: F) -> F:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    current_attempt = attempt
                    attempt += 1
                    if attempt > max_retries or not is_retryable_error(e, retryable_errors):
                        logger.debug(f'Non-retryable error or max retries ({max_retries}) exceeded for {func.__name__}. Raising exception.')
                        raise e
                    delay = calculate_backoff(current_attempt, base_delay, max_delay, jitter)
                    logger.warning(f'Retrying {func.__name__} after error (attempt {attempt}/{max_retries}, delay {delay:.2f}s): {str(e)}')
                    time.sleep(delay)
        return cast(F, wrapper)
    return decorator

def async_retry_with_exponential_backoff(max_retries: int=3, base_delay: float=0.5, max_delay: float=30.0, jitter: bool=True, retryable_errors: Optional[Set[Union[ErrorCode, str]]]=None) -> Callable[[AsyncF], AsyncF]:

    def decorator(func: AsyncF) -> AsyncF:

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    current_attempt = attempt
                    attempt += 1
                    if attempt > max_retries or not is_retryable_error(e, retryable_errors):
                        logger.debug(f'Non-retryable error or max retries ({max_retries}) exceeded for async {func.__name__}. Raising exception.')
                        raise e
                    delay = calculate_backoff(current_attempt, base_delay, max_delay, jitter)
                    logger.warning(f'Retrying async {func.__name__} after error (attempt {attempt}/{max_retries}, delay {delay:.2f}s): {str(e)}')
                    await asyncio.sleep(delay)
        return cast(AsyncF, wrapper)
    return decorator

@metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'retry_operation'})
async def retry_async_operation(operation: Callable[[], Coroutine[Any, Any, R]], max_retries: int=3, base_delay: float=0.5, max_delay: float=30.0, jitter: bool=True, retryable_errors: Optional[Set[Union[ErrorCode, str]]]=None, operation_name: Optional[str]=None) -> R:
    attempt = 0
    op_name = operation_name or getattr(operation, '__name__', 'operation')
    last_error: Optional[Exception] = None
    while attempt <= max_retries:
        try:
            return await operation()
        except Exception as e:
            current_attempt = attempt
            attempt += 1
            last_error = e
            if attempt > max_retries or not is_retryable_error(e, retryable_errors):
                logger.debug(f'Non-retryable error or max retries ({max_retries}) exceeded for {op_name}. Breaking retry loop.')
                break
            delay = calculate_backoff(current_attempt, base_delay, max_delay, jitter)
            logger.warning(f'Retrying {op_name} after error (attempt {attempt}/{max_retries}, delay {delay:.2f}s): {str(e)}')
            await asyncio.sleep(delay)
    if last_error is not None:
        logger.error(f"Operation '{op_name}' failed after {attempt} attempts.")
        raise last_error
    else:
        logger.critical(f"Operation '{op_name}' failed after {max_retries} retries, but no exception was recorded.")
        raise RuntimeError(f'All {max_retries} retry attempts failed for {op_name}, but no specific error is available.')
"""
Retry mechanism for LLM API requests with exponential backoff.
"""

import asyncio
import functools
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import timed_metric, MEMORY_OPERATION_DURATION
from src.config.errors import ErrorCode, LLMError, BaseError, RETRYABLE_ERRORS

settings = get_settings()
logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Default retryable error codes
DEFAULT_RETRYABLE_ERROR_CODES = set(RETRYABLE_ERRORS)

# Additional retryable error codes specific to LLM operations
LLM_RETRYABLE_ERROR_CODES = {
    ErrorCode.LLM_TIMEOUT,
    ErrorCode.LLM_RATE_LIMIT,
    ErrorCode.CONNECTION_ERROR,
    ErrorCode.HTTP_ERROR,
    ErrorCode.NETWORK_ERROR,
    ErrorCode.TIMEOUT_ERROR,
}

# Ensure consistency with global RETRYABLE_ERRORS
DEFAULT_RETRYABLE_ERROR_CODES.update(LLM_RETRYABLE_ERROR_CODES)
# Verify against src.config.errors.RETRYABLE_ERRORS

def is_retryable_error(error: Exception, retryable_errors: Optional[Set[Union[ErrorCode, str]]] = None) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: The exception to check
        retryable_errors: Set of retryable error codes
        
    Returns:
        bool: True if the error is retryable
    """
    if retryable_errors is None:
        retryable_errors = DEFAULT_RETRYABLE_ERROR_CODES
    
    # If it's our own BaseError, check the error code
    if isinstance(error, BaseError):
        error_code = error.code
        
        # Check if the error code (as string or Enum) is in the retryable set
        if error_code in retryable_errors or (
            isinstance(error_code, str) and 
            error_code in [e.value if isinstance(e, ErrorCode) else e for e in retryable_errors]
        ):
            return True
    
    # Check for common HTTP errors in third-party libraries
    error_name = type(error).__name__.lower()
    
    # Common network/HTTP errors from various libraries
    retryable_error_types = {
        'timeout', 'connectionerror', 'connecttimeout', 'readtimeout',
        'connectionrefused', 'sockettimeout', 'requestexception',
        'serviceunitavailable', 'throttlingerror', 'ratelimiterror',
        'temporaryfailure', 'serveroverloaded', 'serviceunavailable'
    }
    
    for error_type in retryable_error_types:
        if error_type in error_name:
            return True
    
    # Look for specific status codes in error message
    error_msg = str(error).lower()
    retryable_phrases = {
        'status code: 429', 'too many requests', 'rate limit',
        'status code: 500', 'status code: 502', 'status code: 503', 'status code: 504',
        'internal server error', 'bad gateway', 'service unavailable', 'gateway timeout',
        'connection reset', 'connection refused', 'timeout', 'request timed out',
        'temporarily unavailable', 'server overloaded', 'server busy'
    }
    
    for phrase in retryable_phrases:
        if phrase in error_msg:
            return True
    
    return False

def calculate_backoff(attempt: int, base_delay: float, max_delay: float, jitter: bool = True) -> float:
    """
    Calculate the backoff delay with exponential increase and optional jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
        
    Returns:
        float: Delay in seconds
    """
    # Calculate exponential backoff
    delay = min(max_delay, base_delay * (2 ** attempt))
    
    # Add jitter if requested (up to 25% in either direction)
    if jitter:
        jitter_factor = (random.random() * 0.5) - 0.25  # -0.25 to 0.25
        delay = delay * (1 + jitter_factor)
    
    return delay

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_errors: Optional[Set[Union[ErrorCode, str]]] = None
) -> Callable[[F], F]:
    """
    Decorator for retrying a synchronous function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
        retryable_errors: Set of retryable error codes
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    
                    # Check if we should retry
                    if attempt > max_retries or not is_retryable_error(e, retryable_errors):
                        raise
                    
                    # Calculate backoff delay
                    delay = calculate_backoff(attempt, base_delay, max_delay, jitter)
                    
                    # Log retry
                    logger.warning(
                        f"Retrying {func.__name__} after error (attempt {attempt}/{max_retries}, "
                        f"delay {delay:.2f}s): {str(e)}"
                    )
                    
                    # Sleep before retry
                    time.sleep(delay)
        
        return cast(F, wrapper)
    
    return decorator

def async_retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_errors: Optional[Set[Union[ErrorCode, str]]] = None
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator for retrying an asynchronous function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
        retryable_errors: Set of retryable error codes
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    
                    # Check if we should retry
                    if attempt > max_retries or not is_retryable_error(e, retryable_errors):
                        raise
                    
                    # Calculate backoff delay
                    delay = calculate_backoff(attempt, base_delay, max_delay, jitter)
                    
                    # Log retry
                    logger.warning(
                        f"Retrying {func.__name__} after error (attempt {attempt}/{max_retries}, "
                        f"delay {delay:.2f}s): {str(e)}"
                    )
                    
                    # Sleep before retry
                    await asyncio.sleep(delay)
        
        return cast(AsyncF, wrapper)
    
    return decorator

# Helper for retry-aware function calls
@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "retry_operation"})
async def retry_async_operation(
    operation: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_errors: Optional[Set[Union[ErrorCode, str]]] = None,
    operation_name: Optional[str] = None
) -> Any:
    """
    Retry an asynchronous operation with exponential backoff.
    
    Args:
        operation: Async function to call
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
        retryable_errors: Set of retryable error codes
        operation_name: Name of the operation for logging
        
    Returns:
        Any: Result of the operation
        
    Raises:
        Exception: The last exception raised by the operation
    """
    attempt = 0
    op_name = operation_name or "operation"
    last_error = None
    
    while attempt <= max_retries:
        try:
            return await operation()
        except Exception as e:
            attempt += 1
            last_error = e
            
            # Check if we should retry
            if attempt > max_retries or not is_retryable_error(e, retryable_errors):
                break
            
            # Calculate backoff delay
            delay = calculate_backoff(attempt - 1, base_delay, max_delay, jitter)
            
            # Log retry
            logger.warning(
                f"Retrying {op_name} after error (attempt {attempt}/{max_retries}, "
                f"delay {delay:.2f}s): {str(e)}"
            )
            
            # Sleep before retry
            await asyncio.sleep(delay)
    
    # If we got here, all retries failed or error is not retryable
    if last_error is not None:
        raise last_error
    
    raise RuntimeError(f"All {max_retries} retry attempts failed for {op_name}")
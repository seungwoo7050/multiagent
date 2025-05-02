import asyncio
import functools
import random
import time
from typing import Any, Callable, Optional, TypeVar, cast

from src.config.logger import get_logger

logger = get_logger(__name__)

# Type variables for function types
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Valid log levels
_VALID_LOG_LEVELS = {'debug', 'info', 'warning', 'error'}

def timed(name: Optional[str]=None) -> Callable[[F], F]:
    """Decorator for timing synchronous function execution"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger.debug(
                    f"Function '{func_name}' executed in {execution_time:.6f} seconds", 
                    extra={
                        'execution_time': execution_time, 
                        'function': func_name,
                        'timing_type': 'function'
                    }
                )
        return cast(F, wrapper)
    return decorator

def async_timed(name: Optional[str]=None) -> Callable[[AsyncF], AsyncF]:
    """Decorator for timing asynchronous function execution"""
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = name or func.__name__
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                logger.debug(
                    f"Async function '{func_name}' executed in {execution_time:.6f} seconds", 
                    extra={
                        'execution_time': execution_time, 
                        'function': func_name,
                        'timing_type': 'async_function'
                    }
                )
        return cast(AsyncF, wrapper)
    return decorator

class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name: str, log_level: str='debug'):
        self.name = name
        self.log_level = self._validate_log_level(log_level)
        self.start_time = 0.0
        self.end_time = 0.0
        self.execution_time = 0.0
        
    def _validate_log_level(self, level: str) -> str:
        """Validate and normalize log level"""
        level_lower = level.lower()
        if level_lower not in _VALID_LOG_LEVELS:
            logger.warning(f"Invalid log level '{level}', defaulting to 'debug'")
            return 'debug'
        return level_lower

    def __enter__(self) -> 'Timer':
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        self._log_execution_time()
        
    def _log_execution_time(self) -> None:
        """Log execution time with consistent format"""
        log_extra = {
            'execution_time': self.execution_time,
            'timer_name': self.name,
            'timing_type': 'timer'
        }
        
        message = f"Timer '{self.name}' completed in {self.execution_time:.6f} seconds"
        
        if self.log_level == 'debug':
            logger.debug(message, extra=log_extra)
        elif self.log_level == 'info':
            logger.info(message, extra=log_extra)
        elif self.log_level == 'warning':
            logger.warning(message, extra=log_extra)
        elif self.log_level == 'error':
            logger.error(message, extra=log_extra)

class AsyncTimer:
    """Asynchronous context manager for timing async code blocks"""
    def __init__(self, name: str, log_level: str='debug'):
        self.name = name
        self.log_level = self._validate_log_level(log_level)
        self.start_time = 0.0
        self.end_time = 0.0
        self.execution_time = 0.0
        
    def _validate_log_level(self, level: str) -> str:
        """Validate and normalize log level"""
        level_lower = level.lower()
        if level_lower not in _VALID_LOG_LEVELS:
            logger.warning(f"Invalid log level '{level}', defaulting to 'debug'")
            return 'debug'
        return level_lower

    async def __aenter__(self) -> 'AsyncTimer':
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        self._log_execution_time()
        
    def _log_execution_time(self) -> None:
        """Log execution time with consistent format"""
        log_extra = {
            'execution_time': self.execution_time,
            'timer_name': self.name,
            'timing_type': 'async_timer'
        }
        
        message = f"AsyncTimer '{self.name}' completed in {self.execution_time:.6f} seconds"
        
        if self.log_level == 'debug':
            logger.debug(message, extra=log_extra)
        elif self.log_level == 'info':
            logger.info(message, extra=log_extra)
        elif self.log_level == 'warning':
            logger.warning(message, extra=log_extra)
        elif self.log_level == 'error':
            logger.error(message, extra=log_extra)

def get_current_time_ms() -> int:
    return int(time.time() * 1000)

async def sleep_with_jitter(base_sleep_time: float, jitter_factor: float=0.2) -> None:
    jitter = base_sleep_time * jitter_factor * (2 * random.random() - 1)
    sleep_time = max(0, base_sleep_time + jitter)
    await asyncio.sleep(sleep_time)
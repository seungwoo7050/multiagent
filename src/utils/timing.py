import time
import functools
import asyncio
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

import numpy

from src.config.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])


def timed(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to measure and log the execution time of a function."""
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
                    extra={"execution_time": execution_time, "function": func_name}
                )
        return cast(F, wrapper)
    return decorator


def async_timed(name: Optional[str] = None) -> Callable[[AsyncF], AsyncF]:
    """Decorator to measure and log the execution time of an async function."""
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
                    extra={"execution_time": execution_time, "function": func_name}
                )
        return cast(AsyncF, wrapper)
    return decorator


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, log_level: str = "debug"):
        self.name = name
        self.log_level = log_level.lower()
        self.start_time = 0.0
        self.end_time = 0.0
        self.execution_time = 0.0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        
        # Log based on specified level
        log_extra = {"execution_time": self.execution_time, "timer_name": self.name}
        
        if self.log_level == "debug":
            logger.debug(
                f"Timer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )
        elif self.log_level == "info":
            logger.info(
                f"Timer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )
        elif self.log_level == "warning":
            logger.warning(
                f"Timer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )
        else:
            # Default to debug
            logger.debug(
                f"Timer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )


class AsyncTimer:
    """Async context manager for timing code blocks."""
    
    def __init__(self, name: str, log_level: str = "debug"):
        self.name = name
        self.log_level = log_level.lower()
        self.start_time = 0.0
        self.end_time = 0.0
        self.execution_time = 0.0
    
    async def __aenter__(self) -> "AsyncTimer":
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        
        # Log based on specified level
        log_extra = {"execution_time": self.execution_time, "timer_name": self.name}
        
        if self.log_level == "debug":
            logger.debug(
                f"AsyncTimer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )
        elif self.log_level == "info":
            logger.info(
                f"AsyncTimer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )
        elif self.log_level == "warning":
            logger.warning(
                f"AsyncTimer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )
        else:
            # Default to debug
            logger.debug(
                f"AsyncTimer '{self.name}' completed in {self.execution_time:.6f} seconds",
                extra=log_extra
            )

def get_current_time_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)

async def sleep_with_jitter(base_sleep_time: float, jitter_factor: float = 0.2) -> None:
    """Sleep with random jitter to prevent thundering herd problem."""
    jitter = base_sleep_time * jitter_factor * (2 * numpy.random.random() - 1)
    sleep_time = max(0, base_sleep_time + jitter)
    await asyncio.sleep(sleep_time)
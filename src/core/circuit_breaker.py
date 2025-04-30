import asyncio
import enum
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast
from pydantic import BaseModel, Field, field_validator
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.utils.timing import get_current_time_ms
logger = get_logger(__name__)
T = TypeVar('T')
R = TypeVar('R')

class CircuitState(str, enum.Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'

class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 5
    success_threshold: int = 3
    reset_timeout_ms: int = 30000
    max_concurrent_requests: int = 10
    failure_window_ms: int = 60000
    count_timeouts_as_failures: bool = True
    request_timeout_ms: Optional[int] = None
    excluded_exceptions: List[str] = Field(default_factory=list)
    
    @field_validator('failure_threshold', 'success_threshold', 'max_concurrent_requests')
    def validate_thresholds(cls, v, values):
        if v <= 0:
            raise ValueError(f"Threshold values must be positive")
        return v
    
    @field_validator('reset_timeout_ms', 'failure_window_ms')
    def validate_timeouts(cls, v, values):
        if v <= 0:
            raise ValueError(f"Timeout values must be positive")
        return v

class CircuitBreakerMetrics(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_elapsed_ms: int = 0
    last_state_change_time: int = Field(default_factory=get_current_time_ms)
    open_count: int = 0
    recent_failures: List[Tuple[int, str]] = Field(default_factory=list)
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    current_concurrent_requests: int = 0
    max_observed_concurrency: int = 0

    model_config = {
        "arbitrary_types_allowed": True,
    }

class CircuitBreaker:

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig]=None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._state_lock = asyncio.Lock()
        self.metrics = CircuitBreakerMetrics()
        logger.info(f'Circuit breaker initialized: {name}', extra={'circuit_name': name, 'initial_state': self._state.value, 'failure_threshold': self.config.failure_threshold, 'reset_timeout_ms': self.config.reset_timeout_ms})

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    @property
    async def failure_count(self) -> int:
        async with self._state_lock:
            current_time = get_current_time_ms()
            window_start = current_time - self.config.failure_window_ms
            # Filter failures and keep only those in the window
            recent_failures_in_window = [f for f in self.metrics.recent_failures if f[0] >= window_start]
            
            # Replace the list with the filtered version to prevent unbounded growth
            self.metrics.recent_failures = recent_failures_in_window
            return len(recent_failures_in_window)
    
    async def _prune_failures(self) -> None:
        """Periodically prune the failures list to prevent memory growth"""
        async with self._state_lock:
            current_time = get_current_time_ms()
            window_start = current_time - self.config.failure_window_ms
            self.metrics.recent_failures = [f for f in self.metrics.recent_failures if f[0] >= window_start]
            
    # This method should be called periodically or after adding a certain number of failures
    
    async def allow_request(self) -> bool:
        # Occasionally prune the failures list to prevent memory issues
        if len(self.metrics.recent_failures) > self.config.failure_threshold * 2:
            await self._prune_failures()
            
        if self._state == CircuitState.CLOSED:
            return self.metrics.current_concurrent_requests < self.config.max_concurrent_requests
        # ... rest of method remains the same ...
        if self._state == CircuitState.OPEN:
            time_in_open_state = get_current_time_ms() - self.metrics.last_state_change_time
            if time_in_open_state >= self.config.reset_timeout_ms:
                async with self._state_lock:
                    if self._state == CircuitState.OPEN:
                        logger.info(f'Circuit {self.name} reset timeout reached. Transitioning to HALF_OPEN.')
                        await self._transition_to(CircuitState.HALF_OPEN)
                        self.metrics.current_concurrent_requests = 1
                        self.metrics.max_observed_concurrency = max(self.metrics.max_observed_concurrency, self.metrics.current_concurrent_requests)
                        return True
            return False
        if self._state == CircuitState.HALF_OPEN:
            async with self._state_lock:
                if self.metrics.current_concurrent_requests == 0:
                    self.metrics.current_concurrent_requests = 1
                    self.metrics.max_observed_concurrency = max(self.metrics.max_observed_concurrency, self.metrics.current_concurrent_requests)
                    logger.debug(f'Circuit {self.name} is HALF_OPEN. Allowing test request.')
                    return True
                logger.debug(f'Circuit {self.name} is HALF_OPEN. Blocking subsequent request.')
                return False
        logger.error(f'Circuit breaker {self.name} in unexpected state: {self._state}')
        return False

    async def _transition_to(self, new_state: CircuitState) -> None:
        if self._state == new_state:
            return
        old_state = self._state
        self._state = new_state
        self.metrics.last_state_change_time = get_current_time_ms()
        if new_state == CircuitState.CLOSED:
            self.metrics.consecutive_failures = 0
            logger.info(f'Circuit {self.name} transitioned to CLOSED. Resetting consecutive failures.')
        elif new_state == CircuitState.OPEN:
            self.metrics.consecutive_successes = 0
            self.metrics.open_count += 1
            logger.warning(f'Circuit {self.name} transitioned to OPEN. Blocking requests. Total opens: {self.metrics.open_count}')
        elif new_state == CircuitState.HALF_OPEN:
            self.metrics.consecutive_successes = 0
            logger.info(f'Circuit {self.name} transitioned to HALF_OPEN. Allowing one test request.')
        logger.info(f'Circuit state changed: {self.name} - {old_state.value} -> {new_state.value}', extra={'circuit_name': self.name, 'old_state': old_state.value, 'new_state': new_state.value, 'failure_count': self.failure_count, 'total_requests': self.metrics.total_requests})

    async def on_success(self) -> None:
        async with self._state_lock:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.current_concurrent_requests = max(0, self.metrics.current_concurrent_requests - 1)
            if self._state == CircuitState.HALF_OPEN:
                logger.debug(f'Success in HALF_OPEN state for {self.name}. Consecutive successes: {self.metrics.consecutive_successes}/{self.config.success_threshold}')
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)

    async def on_failure(self, error_type: str, count_towards_threshold: bool=True) -> None:
        current_time = get_current_time_ms()
        async with self._state_lock:
            self.metrics.failed_requests += 1
            if count_towards_threshold:
                self.metrics.consecutive_failures += 1
                self.metrics.consecutive_successes = 0
                self.metrics.recent_failures.append((current_time, error_type))
                window_start = current_time - self.config.failure_window_ms
                self.metrics.recent_failures = [f for f in self.metrics.recent_failures if f[0] >= window_start]
                if self._state == CircuitState.CLOSED:
                    current_failure_count = len(self.metrics.recent_failures)
                    logger.debug(f'Failure in CLOSED state for {self.name}. Failure count: {current_failure_count}/{self.config.failure_threshold}')
                    if current_failure_count >= self.config.failure_threshold:
                        await self._transition_to(CircuitState.OPEN)
                elif self._state == CircuitState.HALF_OPEN:
                    logger.warning(f'Failure detected in HALF_OPEN state for {self.name}. Transitioning back to OPEN.')
                    await self._transition_to(CircuitState.OPEN)
            self.metrics.current_concurrent_requests = max(0, self.metrics.current_concurrent_requests - 1)

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if not await self.allow_request():
            error_msg = f"Circuit breaker '{self.name}' is {self._state.value} and blocked the request."
            logger.warning(error_msg, extra={'circuit_name': self.name, 'state': self._state.value})
            raise CircuitOpenError(circuit_name=self.name, message=error_msg)
        if self._state == CircuitState.CLOSED:
            async with self._state_lock:
                self.metrics.current_concurrent_requests += 1
                self.metrics.max_observed_concurrency = max(self.metrics.max_observed_concurrency, self.metrics.current_concurrent_requests)
        self.metrics.total_requests += 1
        start_time_ms = get_current_time_ms()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self.on_success()
            return result
        except Exception as e:
            error_type = type(e).__name__
            count_towards_threshold = error_type not in self.config.excluded_exceptions
            await self.on_failure(error_type, count_towards_threshold)
            raise e
        finally:
            execution_time_ms = get_current_time_ms() - start_time_ms
            self.metrics.total_elapsed_ms += execution_time_ms

    def __repr__(self) -> str:
        return f"<CircuitBreaker(name='{self.name}', state={self._state.value}, failures={self.failure_count}/{self.config.failure_threshold})>"

class CircuitOpenError(Exception):

    def __init__(self, circuit_name: str, message: str):
        self.circuit_name = circuit_name
        self.message = message
        super().__init__(message)
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig]=None) -> CircuitBreaker:
    global _circuit_breakers
    if name not in _circuit_breakers:
        logger.info(f'Creating new circuit breaker: {name}')
        _circuit_breakers[name] = CircuitBreaker(name, config)
    elif config is not None:
        cb = _circuit_breakers[name]
        if cb.config != config:
            logger.info(f'Updating configuration for existing circuit breaker: {name}')
            cb.config = config
            cb.metrics.consecutive_failures = 0
    return _circuit_breakers[name]

async def with_circuit_breaker(circuit_name: str, func: Callable[..., R], *args: Any, config: Optional[CircuitBreakerConfig]=None, **kwargs: Any) -> R:
    circuit = get_circuit_breaker(circuit_name, config)
    return await circuit.execute(func, *args, **kwargs)
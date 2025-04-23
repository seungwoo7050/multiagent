import asyncio
import enum
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

from pydantic import BaseModel, Field

from src.config.logger import get_logger
from src.config.metrics import track_agent_error
from src.utils.timing import get_current_time_ms

# Module logger
logger = get_logger(__name__)

# Type variable for generic circuit breaker
T = TypeVar('T')
R = TypeVar('R')


class CircuitState(str, enum.Enum):
    """Enum representing the possible states of a circuit breaker."""
    CLOSED = "closed"      # Normal operation - requests are allowed
    OPEN = "open"          # Failing state - requests are blocked
    HALF_OPEN = "half_open"  # Testing state - limited requests allowed


class CircuitBreakerConfig(BaseModel):
    """Configuration for a circuit breaker."""
    
    # Failure threshold before opening circuit
    failure_threshold: int = 5
    
    # Success threshold to close circuit when in half-open state
    success_threshold: int = 3
    
    # Time (ms) to wait before transitioning from open to half-open
    reset_timeout_ms: int = 30000  # 30 seconds
    
    # Maximum number of concurrent requests allowed
    max_concurrent_requests: int = 10
    
    # Time window (ms) for counting failures
    failure_window_ms: int = 60000  # 60 seconds
    
    # Whether to include timeouts in failure count
    count_timeouts_as_failures: bool = True
    
    # Request timeout (ms)
    request_timeout_ms: Optional[int] = None
    
    # Optional whitelist of exception types that should not be counted as failures
    excluded_exceptions: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class CircuitBreakerMetrics(BaseModel):
    """Metrics for circuit breaker monitoring."""
    
    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Timing statistics
    total_elapsed_ms: int = 0
    
    # Circuit state transitions
    last_state_change_time: int = Field(default_factory=get_current_time_ms)
    open_count: int = 0
    
    # Failure tracking
    recent_failures: List[Tuple[int, str]] = Field(default_factory=list)  # [(timestamp, error_type)]
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Concurrency tracking
    current_concurrent_requests: int = 0
    max_observed_concurrency: int = 0
    
    class Config:
        arbitrary_types_allowed = True


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls.
    
    The circuit breaker pattern prevents cascading failures by stopping
    calls to a failing service and allowing it time to recover.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """Initialize the circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker.
            config: Optional configuration, uses defaults if not provided.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitState.CLOSED
        self._state_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        logger.info(
            f"Circuit breaker initialized: {name}",
            extra={
                "circuit_name": name,
                "initial_state": self._state,
                "failure_threshold": self.config.failure_threshold,
                "reset_timeout_ms": self.config.reset_timeout_ms
            }
        )
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if the circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if the circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if the circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    @property
    def failure_count(self) -> int:
        """Get the number of recent failures within the failure window."""
        current_time = get_current_time_ms()
        window_start = current_time - self.config.failure_window_ms
        
        # Filter failures to only include those within the window
        recent_failures = [f for f in self.metrics.recent_failures if f[0] >= window_start]
        
        # Update the recent failures list
        self.metrics.recent_failures = recent_failures
        
        return len(recent_failures)
    
    async def allow_request(self) -> bool:
        """Check if a request should be allowed based on the circuit state."""
        # If circuit is closed, allow the request
        if self._state == CircuitState.CLOSED:
            return self.metrics.current_concurrent_requests < self.config.max_concurrent_requests
        
        # If circuit is open, check if it's time to transition to half-open
        if self._state == CircuitState.OPEN:
            time_in_open_state = get_current_time_ms() - self.metrics.last_state_change_time
            
            if time_in_open_state >= self.config.reset_timeout_ms:
                async with self._state_lock:
                    # Double-check after acquiring lock
                    if self._state == CircuitState.OPEN:
                        await self._transition_to(CircuitState.HALF_OPEN)
                        # Immediately increment to prevent race conditions
                        self.metrics.current_concurrent_requests += 1
                        return True
            
            # Still in open state and not ready to test
            return False
        
        # For HALF_OPEN state, use a lock to ensure atomic check and update
        if self._state == CircuitState.HALF_OPEN:
            async with self._state_lock:
                if self.metrics.current_concurrent_requests == 0:
                    self.metrics.current_concurrent_requests += 1
                    return True
                return False
                
        return False
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition the circuit to a new state.
        
        Args:
            new_state: The new state for the circuit.
        """
        if self._state == new_state:
            return
        
        old_state = self._state
        self._state = new_state
        self.metrics.last_state_change_time = get_current_time_ms()
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self.metrics.consecutive_failures = 0
        elif new_state == CircuitState.OPEN:
            self.metrics.consecutive_successes = 0
            self.metrics.open_count += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.metrics.consecutive_successes = 0
        
        logger.info(
            f"Circuit state changed: {self.name} - {old_state} -> {new_state}",
            extra={
                "circuit_name": self.name,
                "old_state": old_state,
                "new_state": new_state,
                "failure_count": self.failure_count,
                "total_requests": self.metrics.total_requests
            }
        )
    
    async def on_success(self) -> None:
        """Record a successful request and update circuit state if needed."""
        async with self._state_lock:
            # Update metrics
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            
            # Decrease concurrent request count
            self.metrics.current_concurrent_requests = max(0, self.metrics.current_concurrent_requests - 1)
            
            # If circuit is half-open and enough successes, close it
            if (self._state == CircuitState.HALF_OPEN and 
                self.metrics.consecutive_successes >= self.config.success_threshold):
                await self._transition_to(CircuitState.CLOSED)
    
    async def on_failure(self, error_type: str, count_towards_threshold: bool = True) -> None:
        """Record a failed request and update circuit state if needed.
        
        Args:
            error_type: Type of error that occurred.
        """
        current_time = get_current_time_ms()
        
        async with self._state_lock:
            # Update metrics
            self.metrics.failed_requests += 1
            
            if count_towards_threshold:
                self.metrics.consecutive_failures += 1
                self.metrics.consecutive_successes = 0
            
                # Add to recent failures
                self.metrics.recent_failures.append((current_time, error_type))
            
                # Clean up old failures outside the window
                window_start = current_time - self.config.failure_window_ms
                self.metrics.recent_failures = [
                    f for f in self.metrics.recent_failures if f[0] >= window_start
                ]
            
                # Check if we need to open the circuit
                if self._state == CircuitState.CLOSED:
                    if self.failure_count >= self.config.failure_threshold:
                        await self._transition_to(CircuitState.OPEN)
            
                # If in half-open state, any failure opens the circuit again
                elif self._state == CircuitState.HALF_OPEN:
                    await self._transition_to(CircuitState.OPEN)
                    
            self.metrics.current_concurrent_requests = max(0, self.metrics.current_concurrent_requests - 1)
    
    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function with circuit breaker protection."""
        # Check if the request should be allowed
        if not await self.allow_request():
            logger.warning(
                f"Circuit breaker {self.name} blocked request: circuit is {self._state}",
                extra={"circuit_name": self.name, "state": self._state}
            )
            raise CircuitOpenError(
                circuit_name=self.name,
                message=f"Circuit breaker {self.name} is {self._state}"
            )
        
        # Only increment for CLOSED state (HALF_OPEN increments in allow_request)
        if self._state == CircuitState.CLOSED:
            async with self._state_lock:
                self.metrics.current_concurrent_requests += 1
                self.metrics.max_observed_concurrency = max(
                    self.metrics.max_observed_concurrency,
                    self.metrics.current_concurrent_requests
                )
        
        self.metrics.total_requests += 1
        start_time = get_current_time_ms()
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            await self.on_success()
            
            return result
        except Exception as e:
            # Check if this exception type should be counted as a failure
            error_type = type(e).__name__
            count_towards_threshold = error_type not in self.config.excluded_exceptions
            
            # Record failure
            await self.on_failure(error_type, count_towards_threshold)
            
            # Re-raise the exception
            raise
        finally:
            # Update timing metrics
            execution_time = get_current_time_ms() - start_time
            self.metrics.total_elapsed_ms += execution_time
    
    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name}, state={self._state}, "
            f"failures={self.failure_count}/{self.config.failure_threshold})"
        )


class CircuitOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    
    def __init__(self, circuit_name: str, message: str):
        self.circuit_name = circuit_name
        self.message = message
        super().__init__(message)


# Circuit breaker registry for global access
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker by name.
    
    Args:
        name: Name of the circuit breaker.
        config: Optional configuration for new circuit breakers.
        
    Returns:
        CircuitBreaker: The requested circuit breaker instance.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    elif config is not None:
        cb = _circuit_breakers[name]
        cb.config = config
        cb.metrics.consecutive_failures = 0
    
    return _circuit_breakers[name]


async def with_circuit_breaker(
    circuit_name: str,
    func: Callable[..., R],
    *args: Any,
    config: Optional[CircuitBreakerConfig] = None,
    **kwargs: Any
) -> R:
    """Execute a function with circuit breaker protection.
    
    Args:
        circuit_name: Name of the circuit breaker to use.
        func: Function to execute.
        *args: Positional arguments for the function.
        config: Optional circuit breaker configuration.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        R: The result of the function execution.
        
    Raises:
        Exception: If the circuit is open or the function fails.
    """
    circuit = get_circuit_breaker(circuit_name, config)
    return await circuit.execute(func, *args, **kwargs)
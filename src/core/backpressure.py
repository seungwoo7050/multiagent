import asyncio
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

from pydantic import BaseModel, Field

from src.config.logger import get_logger
from src.utils.timing import get_current_time_ms

# Module logger
logger = get_logger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


class BackpressureStrategy(str, Enum):
    """Strategies for handling backpressure."""
    REJECT = "reject"         # Reject new requests
    THROTTLE = "throttle"     # Delay processing of new requests
    SHED_LOAD = "shed_load"   # Drop lower priority requests
    QUEUE = "queue"           # Queue requests with a limit


class TokenBucketConfig(BaseModel):
    """Configuration for token bucket rate limiter."""
    
    # Maximum number of tokens in the bucket
    max_tokens: int = 100
    
    # Number of tokens added per refill
    refill_rate: float = 10.0
    
    # Time between refills in milliseconds
    refill_interval_ms: int = 1000
    
    # Number of tokens to consume per request
    tokens_per_request: int = 1


class BackpressureConfig(BaseModel):
    """Configuration for backpressure handling."""
    
    # Strategy for handling backpressure
    strategy: BackpressureStrategy = BackpressureStrategy.THROTTLE
    
    # Maximum number of concurrent requests
    max_concurrent_requests: int = 100
    
    # Maximum queue size for queued requests
    max_queue_size: int = 1000
    
    # Maximum wait time in milliseconds for queued requests
    max_wait_time_ms: int = 30000
    
    # Threshold percentage of max_concurrent_requests at which to start applying backpressure
    threshold_percentage: float = 0.8
    
    # Token bucket configuration for rate limiting
    token_bucket: TokenBucketConfig = Field(default_factory=TokenBucketConfig)
    
    # Priority thresholds for load shedding
    min_priority_threshold: int = 0  # Minimum priority to accept when shedding load


class BackpressureMetrics(BaseModel):
    """Metrics for backpressure monitoring."""
    
    # Request tracking
    total_requests: int = 0
    rejected_requests: int = 0
    throttled_requests: int = 0
    shed_requests: int = 0
    queued_requests: int = 0
    
    # Concurrency tracking
    current_concurrent_requests: int = 0
    max_observed_concurrency: int = 0
    
    # Queue metrics
    current_queue_size: int = 0
    max_observed_queue_size: int = 0
    total_queue_wait_time_ms: int = 0
    
    # Token bucket metrics
    current_tokens: float = 0.0
    last_refill_time: int = Field(default_factory=get_current_time_ms)
    
    class Config:
        arbitrary_types_allowed = True


class Backpressure:
    """Utility for implementing backpressure control.
    
    This helps prevent system overload by controlling the rate
    at which new requests are accepted and processed.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[BackpressureConfig] = None
    ):
        """Initialize the backpressure controller.
        
        Args:
            name: Identifier for this backpressure controller.
            config: Optional configuration, uses defaults if not provided.
        """
        self.name = name
        self.config = config or BackpressureConfig()
        
        # Initialize metrics
        self.metrics = BackpressureMetrics()
        self.metrics.current_tokens = float(self.config.token_bucket.max_tokens)
        
        # Create lock for thread safety
        self._lock = asyncio.Lock()
        
        # Queue for queued requests
        self._queue: List[asyncio.Event] = []
        
        logger.info(
            f"Backpressure controller initialized: {name}",
            extra={
                "controller_name": name,
                "strategy": self.config.strategy,
                "max_concurrent": self.config.max_concurrent_requests,
                "token_bucket_max": self.config.token_bucket.max_tokens,
                "token_bucket_rate": self.config.token_bucket.refill_rate
            }
        )
    
    async def _refill_tokens(self) -> None:
        """Refill tokens in the token bucket based on elapsed time."""
        current_time = get_current_time_ms()
        elapsed_ms = current_time - self.metrics.last_refill_time
        
        # Calculate number of refills that should have occurred
        refills = elapsed_ms / self.config.token_bucket.refill_interval_ms
        
        if refills >= 1:
            # Add tokens based on refill rate and elapsed time
            tokens_to_add = refills * self.config.token_bucket.refill_rate
            
            # Update metrics
            self.metrics.current_tokens = min(
                self.metrics.current_tokens + tokens_to_add,
                self.config.token_bucket.max_tokens
            )
            self.metrics.last_refill_time = current_time
    
    async def _consume_tokens(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume.
            
        Returns:
            bool: True if tokens were consumed, False otherwise.
        """
        async with self._lock:
            # Refill tokens first
            await self._refill_tokens()
            
            # Check if we have enough tokens
            if self.metrics.current_tokens >= tokens:
                self.metrics.current_tokens -= tokens
                return True
            
            return False
    
    async def _process_queue(self) -> None:
        """Process the request queue, releasing waiters when possible."""
        async with self._lock:
            # Check if we can process any queued requests
            if (self.metrics.current_concurrent_requests < self.config.max_concurrent_requests and
                    self._queue):
                # Release the first waiter
                wait_event = self._queue.pop(0)
                self.metrics.current_queue_size = len(self._queue)
                wait_event.set()
    
    async def _check_strategy(self, priority: int = 0) -> Tuple[bool, str]:
        """Check if a request should be accepted based on the configured strategy."""
        # Update metrics
        self.metrics.total_requests += 1
        
        # Apply strategy-specific logic
        if self.config.strategy == BackpressureStrategy.REJECT:
            # Only reject when we've reached max_concurrent_requests
            if self.metrics.current_concurrent_requests >= self.config.max_concurrent_requests:
                self.metrics.rejected_requests += 1
                return False, "rejected_max_concurrency"
            return True, "accepted"
        
        elif self.config.strategy == BackpressureStrategy.THROTTLE:
            # Always consume tokens for THROTTLE, regardless of concurrency
            if not await self._consume_tokens(self.config.token_bucket.tokens_per_request):
                self.metrics.throttled_requests += 1
                return False, "throttled_rate_limit"
            return True, "accepted"
            
        # Other strategies remain the same
        
        elif self.config.strategy == BackpressureStrategy.SHED_LOAD:
            # If we're at max concurrency, check priority
            if self.metrics.current_concurrent_requests >= self.config.max_concurrent_requests:
                # Reject low priority requests
                if priority < self.config.min_priority_threshold:
                    self.metrics.shed_requests += 1
                    return False, "shed_low_priority"
            return True, "accepted"
        
        elif self.config.strategy == BackpressureStrategy.QUEUE:
            # If we're at max concurrency, queue the request
            if self.metrics.current_concurrent_requests >= self.config.max_concurrent_requests:
                # Reject if queue is full
                if self.metrics.current_queue_size >= self.config.max_queue_size:
                    self.metrics.rejected_requests += 1
                    return False, "rejected_queue_full"
                
                self.metrics.queued_requests += 1
                return True, "queued"
            
            return True, "accepted"
        
        # Default to accepting
        return True, "accepted"
    
    async def acquire(self, priority: int = 0) -> bool:
        """Attempt to acquire permission to process a request.
        
        Args:
            priority: Priority of the request (higher is more important).
            
        Returns:
            bool: True if the request can proceed, False if it should be rejected.
        """
        # Check strategy first
        accepted, reason = await self._check_strategy(priority)
        
        if not accepted:
            logger.debug(
                f"Backpressure controller {self.name} rejected request: {reason}",
                extra={
                    "controller_name": self.name,
                    "reason": reason,
                    "current_concurrency": self.metrics.current_concurrent_requests,
                    "max_concurrency": self.config.max_concurrent_requests
                }
            )
            return False
        
        # Handle queuing if needed
        if reason == "queued":
            # Create an event for this request
            wait_event = asyncio.Event()
            
            # Add to queue
            async with self._lock:
                self._queue.append(wait_event)
                self.metrics.current_queue_size = len(self._queue)
                self.metrics.max_observed_queue_size = max(
                    self.metrics.max_observed_queue_size,
                    self.metrics.current_queue_size
                )
            
            # Wait for our turn, with timeout
            queue_start_time = get_current_time_ms()
            try:
                wait_result = await asyncio.wait_for(
                    wait_event.wait(),
                    timeout=self.config.max_wait_time_ms / 1000.0
                )
                
                # Update queue wait time metrics
                queue_wait_time = get_current_time_ms() - queue_start_time
                self.metrics.total_queue_wait_time_ms += queue_wait_time
                
                if not wait_result:
                    # Shouldn't happen with asyncio.Event, but handle anyway
                    return False
                
            except asyncio.TimeoutError:
                # Remove from queue if still there
                async with self._lock:
                    if wait_event in self._queue:
                        self._queue.remove(wait_event)
                        self.metrics.current_queue_size = len(self._queue)
                
                logger.debug(
                    f"Backpressure controller {self.name} queue wait timeout",
                    extra={
                        "controller_name": self.name,
                        "wait_time_ms": self.config.max_wait_time_ms
                    }
                )
                return False
        
        # Increment concurrency counter
        async with self._lock:
            self.metrics.current_concurrent_requests += 1
            self.metrics.max_observed_concurrency = max(
                self.metrics.max_observed_concurrency,
                self.metrics.current_concurrent_requests
            )
        
        return True
    
    async def release(self) -> None:
        """Release a request, decrementing the concurrency counter.
        
        This should be called after a request is complete.
        """
        async with self._lock:
            # Decrement concurrency counter
            self.metrics.current_concurrent_requests = max(
                0,  # Prevent negative counts
                self.metrics.current_concurrent_requests - 1
            )
        
        # Process queue if we have queued requests
        if self._queue:
            await self._process_queue()
    
    async def execute(
        self,
        func: Callable[..., R],
        *args: Any,
        priority: int = 0,
        **kwargs: Any
    ) -> Optional[R]:
        """Execute a function with backpressure control.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            priority: Priority of the request (higher is more important).
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Optional[R]: The function result, or None if rejected.
        """
        # Try to acquire permission
        if not await self.acquire(priority):
            return None
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        finally:
            # Always release
            await self.release()
    
    async def __aenter__(self) -> "Backpressure":
        """Context manager entry point."""
        # Try to acquire permission
        if not await self.acquire():
            raise BackpressureRejectedError(
                f"Request rejected by backpressure controller {self.name}"
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        await self.release()


class BackpressureRejectedError(Exception):
    """Exception raised when a request is rejected due to backpressure."""
    pass


# Backpressure controller registry for global access
_backpressure_controllers: Dict[str, Backpressure] = {}


def get_backpressure_controller(
    name: str,
    config: Optional[BackpressureConfig] = None
) -> Backpressure:
    """Get or create a backpressure controller by name.
    
    Args:
        name: Name of the backpressure controller.
        config: Optional configuration for new controllers.
        
    Returns:
        Backpressure: The requested backpressure controller.
    """
    if name not in _backpressure_controllers:
        _backpressure_controllers[name] = Backpressure(name, config)
    return _backpressure_controllers[name]


async def with_backpressure(
    controller_name: str,
    func: Callable[..., R],
    *args: Any,
    priority: int = 0,
    config: Optional[BackpressureConfig] = None,
    **kwargs: Any
) -> Optional[R]:
    """Execute a function with backpressure control.
    
    Args:
        controller_name: Name of the backpressure controller to use.
        func: Function to execute.
        *args: Positional arguments for the function.
        priority: Priority of the request (higher is more important).
        config: Optional backpressure configuration.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        Optional[R]: The function result, or None if rejected.
    """
    controller = get_backpressure_controller(controller_name, config)
    return await controller.execute(func, *args, priority=priority, **kwargs)
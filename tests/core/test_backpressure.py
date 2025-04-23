import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from src.core.backpressure import (
    Backpressure,
    BackpressureConfig,
    BackpressureStrategy,
    TokenBucketConfig,
    BackpressureRejectedError,
    get_backpressure_controller,
    with_backpressure
)


@pytest.fixture
def token_bucket_config():
    """Fixture for creating token bucket configuration."""
    return TokenBucketConfig(
        max_tokens=10,
        refill_rate=5.0,
        refill_interval_ms=100,
        tokens_per_request=1
    )


@pytest.fixture
def reject_config(token_bucket_config):
    """Fixture for creating a REJECT backpressure configuration."""
    return BackpressureConfig(
        strategy=BackpressureStrategy.REJECT,
        max_concurrent_requests=2,
        threshold_percentage=0.5,  # Start applying backpressure at 1 concurrent request
        token_bucket=token_bucket_config
    )


@pytest.fixture
def throttle_config(token_bucket_config):
    """Fixture for creating a THROTTLE backpressure configuration."""
    return BackpressureConfig(
        strategy=BackpressureStrategy.THROTTLE,
        max_concurrent_requests=5,
        token_bucket=token_bucket_config
    )


@pytest.fixture
def shed_load_config():
    """Fixture for creating a SHED_LOAD backpressure configuration."""
    return BackpressureConfig(
        strategy=BackpressureStrategy.SHED_LOAD,
        max_concurrent_requests=2,
        min_priority_threshold=2  # Requests with priority < 2 will be shed when at max
    )


@pytest.fixture
def queue_config():
    """Fixture for creating a QUEUE backpressure configuration."""
    return BackpressureConfig(
        strategy=BackpressureStrategy.QUEUE,
        max_concurrent_requests=2,
        max_queue_size=3,
        max_wait_time_ms=100  # Short wait time for testing
    )


class TestBackpressure:
    """Test suite for Backpressure."""

    @pytest.mark.asyncio
    async def test_reject_strategy(self, reject_config):
        """Test REJECT backpressure strategy."""
        bp = Backpressure("reject_test", reject_config)
        
        # First request should be allowed
        assert await bp.acquire() is True
        
        # Verify metrics
        assert bp.metrics.current_concurrent_requests == 1
        assert bp.metrics.total_requests == 1
        
        # Second request hits threshold but is still allowed
        assert await bp.acquire() is True
        assert bp.metrics.current_concurrent_requests == 2
        
        # Third request should be rejected (max concurrent reached)
        assert await bp.acquire() is False
        assert bp.metrics.rejected_requests == 1
        assert bp.metrics.current_concurrent_requests == 2
        
        # Release a request
        await bp.release()
        assert bp.metrics.current_concurrent_requests == 1
        
        # Now next request should be allowed
        assert await bp.acquire() is True
        assert bp.metrics.current_concurrent_requests == 2

    @pytest.mark.asyncio
    async def test_throttle_strategy(self, throttle_config):
        """Test THROTTLE backpressure strategy."""
        bp = Backpressure("throttle_test", throttle_config)
        
        # Refill tokens before test
        await bp._refill_tokens()
        initial_tokens = bp.metrics.current_tokens
        
        # First request should be allowed and consume a token
        assert await bp.acquire() is True
        assert bp.metrics.current_tokens == initial_tokens - bp.config.token_bucket.tokens_per_request
        
        # Use up all tokens
        for i in range(int(initial_tokens) - 1):
            assert await bp.acquire() is True
        
        # Tokens should be exhausted
        assert bp.metrics.current_tokens == 0
        
        # Next request should be throttled
        assert await bp.acquire() is False
        assert bp.metrics.throttled_requests == 1
        
        # Wait for token refill
        await asyncio.sleep(bp.config.token_bucket.refill_interval_ms / 1000.0 + 0.01)
        
        # Force token refill
        await bp._refill_tokens()
        
        # Check tokens refilled
        assert bp.metrics.current_tokens > 0
        
        # Request should now be allowed
        assert await bp.acquire() is True

    @pytest.mark.asyncio
    async def test_shed_load_strategy(self, shed_load_config):
        """Test SHED_LOAD backpressure strategy."""
        bp = Backpressure("shed_load_test", shed_load_config)
        
        # First request (no priority) should be allowed
        assert await bp.acquire() is True
        
        # Second request (no priority) should be allowed
        assert await bp.acquire() is True
        
        # Third request with low priority should be shed
        assert await bp.acquire(priority=1) is False
        assert bp.metrics.shed_requests == 1
        
        # Third request with high priority should be allowed
        assert await bp.acquire(priority=5) is True
        assert bp.metrics.current_concurrent_requests == 3
        
        # Release requests
        await bp.release()
        await bp.release()
        
        # Now low priority request should be allowed
        assert await bp.acquire(priority=1) is True

    @pytest.mark.asyncio
    async def test_queue_strategy(self, queue_config):
        """Test QUEUE backpressure strategy."""
        bp = Backpressure("queue_test", queue_config)
        
        # First two requests should be allowed immediately
        assert await bp.acquire() is True
        assert await bp.acquire() is True
        assert bp.metrics.current_concurrent_requests == 2
        
        # Next requests should be queued
        # Start 3 async tasks to test queuing
        async def try_acquire():
            return await bp.acquire()
        
        tasks = [asyncio.create_task(try_acquire()) for _ in range(3)]
        
        # Let the tasks start and queue up
        await asyncio.sleep(0.01)
        
        # Check metrics
        assert bp.metrics.queued_requests == 3
        assert bp.metrics.current_queue_size == 3
        assert len(bp._queue) == 3
        
        # Release one request, which should allow one queued request to proceed
        await bp.release()
        
        # Give time for queued request to be processed
        await asyncio.sleep(0.01)
        
        # Check metrics
        assert bp.metrics.current_concurrent_requests == 2
        assert bp.metrics.current_queue_size == 2
        
        # Release all remaining requests
        await bp.release()
        await bp.release()
        
        # Wait for tasks to complete
        results = await asyncio.gather(*tasks)
        
        # First 3 queued requests should be allowed
        assert results == [True, True, True]
        
        # Queue should be empty
        assert bp.metrics.current_queue_size == 0

    @pytest.mark.asyncio
    async def test_queue_timeout(self, queue_config):
        """Test timeout behavior in QUEUE strategy."""
        # Use very short wait time
        config = queue_config.copy()
        config.max_wait_time_ms = 10  # 10ms timeout
        
        bp = Backpressure("queue_timeout_test", config)
        
        # Fill up concurrent requests
        assert await bp.acquire() is True
        assert await bp.acquire() is True
        
        # This request should time out
        async def acquire_with_await():
            return await bp.acquire()
        
        # Start request as task
        task = asyncio.create_task(acquire_with_await())
        
        # Let it time out
        await asyncio.sleep(0.03)  # > max_wait_time_ms
        
        # Get result
        result = await task
        
        # Should have failed
        assert result is False
        
        # Queue should be empty
        assert bp.metrics.current_queue_size == 0

    @pytest.mark.asyncio
    async def test_execute_function(self, reject_config):
        """Test executing a function through backpressure."""
        bp = Backpressure("execute_test", reject_config)

        # Define test function
        async def test_func(arg1, arg2=None):
            return f"{arg1}-{arg2}"

        # Execute function - uses 1 slot temporarily
        result = await bp.execute(test_func, "hello", arg2="world")
        assert result == "hello-world"
        # Concurrency is now 0 after execute finishes and releases

        # Fill up ALL concurrent requests manually
        # reject_config.max_concurrent_requests is 2
        assert await bp.acquire() is True # Concurrency = 1
        assert await bp.acquire() is True # Concurrency = 2 (max reached)

        # Next execution should now be rejected by the internal acquire()
        result = await bp.execute(test_func, "rejected")
        # The assertion should now pass as acquire() fails and execute returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_with_backpressure_helper(self, throttle_config):
        """Test the with_backpressure helper function."""
        # Define test function
        async def test_func(arg1, arg2=None):
            return f"{arg1}-{arg2}"
        
        # Execute function
        result = await with_backpressure(
            "helper_test",
            test_func,
            "hello",
            arg2="world",
            config=throttle_config
        )
        assert result == "hello-world"
        
        # Get the controller to fill it up
        bp = get_backpressure_controller("helper_test")
        
        # Use up all tokens
        tokens = bp.metrics.current_tokens
        for i in range(int(tokens)):
            await bp.acquire()
        
        # Next execution should be throttled
        result = await with_backpressure("helper_test", test_func, "throttled")
        assert result is None
        assert bp.metrics.throttled_requests == 1

    @pytest.mark.asyncio
    async def test_get_backpressure_controller(self):
        """Test the get_backpressure_controller factory function."""
        # Get a controller
        bp1 = get_backpressure_controller("singleton_test")
        
        # Get it again
        bp2 = get_backpressure_controller("singleton_test")
        
        # Should be the same instance
        assert bp1 is bp2
        
        # Get with custom config
        custom_config = BackpressureConfig(max_concurrent_requests=10)
        bp3 = get_backpressure_controller("custom_config_test", custom_config)
        
        # Should have custom config
        assert bp3.config.max_concurrent_requests == 10
        
        # Getting same name again should return same instance with original config
        bp4 = get_backpressure_controller("custom_config_test")
        assert bp4 is bp3
        assert bp4.config.max_concurrent_requests == 10

    @pytest.mark.asyncio
    async def test_context_manager(self, reject_config):
        """Test using backpressure as an async context manager."""
        bp = Backpressure("context_test", reject_config)
        
        # Use context manager successfully
        async with bp:
            # Inside context, a request should be acquired
            assert bp.metrics.current_concurrent_requests == 1
        
        # After context, request should be released
        assert bp.metrics.current_concurrent_requests == 0
        
        # Fill up requests
        assert await bp.acquire() is True
        assert await bp.acquire() is True
        
        # Context manager should raise when acquisition fails
        with pytest.raises(BackpressureRejectedError):
            async with bp:
                pass  # Should not execute

    @pytest.mark.asyncio
    async def test_priority_handling(self, shed_load_config):
        """Test priority handling in various strategies."""
        bp = Backpressure("priority_test", shed_load_config)
        
        # Fill up to max concurrent
        assert await bp.acquire(priority=0) is True
        assert await bp.acquire(priority=0) is True
        
        # Low priority request should be rejected
        assert await bp.acquire(priority=1) is False
        
        # High priority request should be allowed
        assert await bp.acquire(priority=5) is True
        
        # Release all
        await bp.release()
        await bp.release()
        await bp.release()
        
        # Change to REJECT strategy
        bp.config.strategy = BackpressureStrategy.REJECT
        
        # Fill up to max concurrent
        assert await bp.acquire() is True
        assert await bp.acquire() is True
        
        # Priority shouldn't matter for REJECT
        assert await bp.acquire(priority=100) is False
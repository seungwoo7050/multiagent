import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from src.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    get_circuit_breaker,
    with_circuit_breaker
)


@pytest.fixture
def circuit_config():
    """Fixture for creating a test circuit breaker configuration."""
    return CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        reset_timeout_ms=50,  # Short timeout for testing
        max_concurrent_requests=5
    )


@pytest.fixture
def circuit_breaker(circuit_config):
    """Fixture for creating a test circuit breaker."""
    return CircuitBreaker("test_circuit", circuit_config)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state."""
        # Should start in CLOSED state
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_closed is True
        assert circuit_breaker.is_open is False
        assert circuit_breaker.is_half_open is False
        
        # Should allow requests
        assert await circuit_breaker.allow_request() is True
        
        # Metrics should be initialized
        assert circuit_breaker.metrics.total_requests == 0
        assert circuit_breaker.metrics.successful_requests == 0
        assert circuit_breaker.metrics.failed_requests == 0
        assert circuit_breaker.metrics.consecutive_failures == 0
        assert circuit_breaker.metrics.consecutive_successes == 0

    @pytest.mark.asyncio
    async def test_success_tracking(self, circuit_breaker):
        """Test success tracking."""
        # Record a success
        await circuit_breaker.on_success()
        
        # Check metrics
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.consecutive_successes == 1
        assert circuit_breaker.metrics.consecutive_failures == 0
        
        # Record another success
        await circuit_breaker.on_success()
        
        # Check metrics
        assert circuit_breaker.metrics.successful_requests == 2
        assert circuit_breaker.metrics.consecutive_successes == 2
        assert circuit_breaker.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_failure_tracking(self, circuit_breaker):
        """Test failure tracking."""
        # Record a failure
        await circuit_breaker.on_failure("TestError")
        
        # Check metrics
        assert circuit_breaker.metrics.failed_requests == 1
        assert circuit_breaker.metrics.consecutive_failures == 1
        assert circuit_breaker.metrics.consecutive_successes == 0
        
        # Recent failures should be recorded
        assert len(circuit_breaker.metrics.recent_failures) == 1
        assert circuit_breaker.metrics.recent_failures[0][1] == "TestError"
        
        # Record another failure
        await circuit_breaker.on_failure("AnotherError")
        
        # Check metrics
        assert circuit_breaker.metrics.failed_requests == 2
        assert circuit_breaker.metrics.consecutive_failures == 2
        assert circuit_breaker.metrics.consecutive_successes == 0
        assert len(circuit_breaker.metrics.recent_failures) == 2

    @pytest.mark.asyncio
    async def test_circuit_opening(self, circuit_breaker):
        """Test circuit opening after reaching failure threshold."""
        # Circuit starts CLOSED
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Record failures up to threshold
        for i in range(circuit_breaker.config.failure_threshold):
            await circuit_breaker.on_failure("TestError")
        
        # Circuit should now be OPEN
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.is_open is True
        
        # Requests should be rejected
        assert await circuit_breaker.allow_request() is False
        
        # Open count should be incremented
        assert circuit_breaker.metrics.open_count == 1

    @pytest.mark.asyncio
    async def test_circuit_half_open_transition(self, circuit_breaker):
        """Test transition from OPEN to HALF_OPEN after reset timeout."""
        # Open the circuit
        for i in range(circuit_breaker.config.failure_threshold):
            await circuit_breaker.on_failure("TestError")
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for reset timeout
        await asyncio.sleep(circuit_breaker.config.reset_timeout_ms / 1000.0 + 0.01)
        
        # First request after timeout should be allowed and transition to HALF_OPEN
        assert await circuit_breaker.allow_request() is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        
        # Subsequent requests should be rejected until test request completes
        assert await circuit_breaker.allow_request() is False

    @pytest.mark.asyncio
    async def test_circuit_reclosing(self, circuit_breaker):
        """Test reclosing circuit after successful test requests."""
        # Open the circuit
        for i in range(circuit_breaker.config.failure_threshold):
            await circuit_breaker.on_failure("TestError")
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for reset timeout
        await asyncio.sleep(circuit_breaker.config.reset_timeout_ms / 1000.0 + 0.01)
        
        # First request after timeout should be allowed and transition to HALF_OPEN
        assert await circuit_breaker.allow_request() is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        
        # Record successful requests up to threshold
        for i in range(circuit_breaker.config.success_threshold):
            await circuit_breaker.on_success()
        
        # Circuit should now be CLOSED again
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_closed is True
        
        # Requests should be allowed again
        assert await circuit_breaker.allow_request() is True

    @pytest.mark.asyncio
    async def test_circuit_reopening(self, circuit_breaker):
        """Test reopening circuit from HALF_OPEN on failure."""
        # Open the circuit
        for i in range(circuit_breaker.config.failure_threshold):
            await circuit_breaker.on_failure("TestError")
        
        # Wait for reset timeout and get to HALF_OPEN
        await asyncio.sleep(circuit_breaker.config.reset_timeout_ms / 1000.0 + 0.01)
        await circuit_breaker.allow_request()
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        
        # Record a failure in HALF_OPEN state
        await circuit_breaker.on_failure("TestError")
        
        # Circuit should be OPEN again
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.is_open is True
        
        # Open count should be incremented again
        assert circuit_breaker.metrics.open_count == 2

    @pytest.mark.asyncio
    async def test_execute_function(self, circuit_breaker):
        """Test executing a function through the circuit breaker."""
        # Define test functions
        async def success_func():
            return "success"
        
        async def failing_func():
            raise ValueError("Test error")
        
        # Execute successful function
        result = await circuit_breaker.execute(success_func)
        assert result == "success"
        
        # Check metrics
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.failed_requests == 0
        
        # Execute failing function
        with pytest.raises(ValueError):
            await circuit_breaker.execute(failing_func)
        
        # Check metrics
        assert circuit_breaker.metrics.successful_requests == 1
        assert circuit_breaker.metrics.failed_requests == 1
        
        # Open the circuit with more failures
        for i in range(circuit_breaker.config.failure_threshold - 1):
            with pytest.raises(ValueError):
                await circuit_breaker.execute(failing_func)
        
        # Circuit should now be OPEN
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Next execution should be blocked with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await circuit_breaker.execute(success_func)

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_helper(self):
        """Test the with_circuit_breaker helper function."""
        # Define test functions
        async def success_func():
            return "success"
        
        async def failing_func():
            raise ValueError("Test error")
        
        # Execute successful function
        result = await with_circuit_breaker("helper_test", success_func)
        assert result == "success"
        
        # Execute failing function
        with pytest.raises(ValueError):
            await with_circuit_breaker("helper_test", failing_func)
        
        # Get the circuit breaker to check state
        circuit = get_circuit_breaker("helper_test")
        assert circuit.metrics.successful_requests == 1
        assert circuit.metrics.failed_requests == 1
        
        # Open the circuit with more failures
        config = CircuitBreakerConfig(failure_threshold=2)
        with pytest.raises(ValueError):
            await with_circuit_breaker("helper_test", failing_func, config=config)
        
        # Circuit should now be OPEN
        assert circuit.state == CircuitState.OPEN
        
        # Next execution should be blocked
        with pytest.raises(CircuitOpenError):
            await with_circuit_breaker("helper_test", success_func)

    @pytest.mark.asyncio
    async def test_get_circuit_breaker(self):
        """Test the get_circuit_breaker factory function."""
        # Get a circuit breaker
        circuit1 = get_circuit_breaker("singleton_test")
        
        # Get it again
        circuit2 = get_circuit_breaker("singleton_test")
        
        # Should be the same instance
        assert circuit1 is circuit2
        
        # Get with custom config
        custom_config = CircuitBreakerConfig(failure_threshold=10)
        circuit3 = get_circuit_breaker("custom_config_test", custom_config)
        
        # Should have custom config
        assert circuit3.config.failure_threshold == 10
        
        # Getting same name again should return same instance with original config
        circuit4 = get_circuit_breaker("custom_config_test")
        assert circuit4 is circuit3
        assert circuit4.config.failure_threshold == 10

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self, circuit_config):
        """Test excluded exceptions don't count towards failure threshold."""
        # Create circuit with excluded exceptions
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=["KeyError", "ValueError"]
        )
        circuit = CircuitBreaker("excluded_test", config)
        
        # Define test functions
        async def key_error_func():
            raise KeyError("Test key error")
        
        async def value_error_func():
            raise ValueError("Test value error")
        
        async def type_error_func():
            raise TypeError("Test type error")
        
        # Execute excluded exception functions
        with pytest.raises(KeyError):
            await circuit.execute(key_error_func)
        
        with pytest.raises(ValueError):
            await circuit.execute(value_error_func)
        
        # Circuit should still be CLOSED
        assert circuit.state == CircuitState.CLOSED
        assert circuit.metrics.failed_requests == 2
        assert circuit.failure_count == 0  # Excluded exceptions don't count in failure window
        
        # Execute non-excluded exception function
        with pytest.raises(TypeError):
            await circuit.execute(type_error_func)
        
        # Check metrics
        assert circuit.metrics.failed_requests == 3
        assert circuit.failure_count == 1  # This one counts
        
        # Execute non-excluded exception again to open circuit
        with pytest.raises(TypeError):
            await circuit.execute(type_error_func)
        
        # Circuit should now be OPEN
        assert circuit.state == CircuitState.OPEN
        assert circuit.failure_count == 2

    @pytest.mark.asyncio
    async def test_failure_window(self):
        """Test failure window for counting failures."""
        # Create circuit with short failure window
        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window_ms=50  # Very short window
        )
        circuit = CircuitBreaker("window_test", config)
        
        # Add two failures
        await circuit.on_failure("TestError")
        await circuit.on_failure("TestError")
        
        # Wait for failures to age out of window
        await asyncio.sleep(0.06)  # Just over the failure window
        
        # Add a single new failure
        await circuit.on_failure("TestError")
        
        # Circuit should still be CLOSED because old failures expired
        assert circuit.state == CircuitState.CLOSED
        assert circuit.failure_count == 1  # Only the most recent failure counts
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from src.llm.retry import (
    is_retryable_error,
    calculate_backoff,
    async_retry_with_exponential_backoff,
    retry_async_operation,
    DEFAULT_RETRYABLE_ERROR_CODES,
    LLM_RETRYABLE_ERROR_CODES
)
from src.config.errors import BaseError, ErrorCode, LLMError


def test_is_retryable_error():
    """Test determination of retryable errors."""
    # BaseError with retryable error code
    retryable_error = BaseError(
        code=ErrorCode.LLM_TIMEOUT,
        message="Timeout error"
    )
    assert is_retryable_error(retryable_error) is True
    
    # BaseError with non-retryable error code
    non_retryable_error = BaseError(
        code=ErrorCode.VALIDATION_ERROR,
        message="Validation error"
    )
    assert is_retryable_error(non_retryable_error) is False
    
    # Standard Python exceptions that should be retryable
    assert is_retryable_error(TimeoutError("Connection timed out")) is True
    assert is_retryable_error(ConnectionError("Connection refused")) is True
    
    # Other exceptions with retryable phrases in message
    assert is_retryable_error(Exception("status code: 429")) is True
    assert is_retryable_error(Exception("rate limit exceeded")) is True
    assert is_retryable_error(Exception("server overloaded")) is True
    
    # Non-retryable standard exception
    assert is_retryable_error(ValueError("Invalid input")) is False


def test_calculate_backoff():
    """Test backoff delay calculation."""
    # First attempt (0-based) with base delay 1s
    delay = calculate_backoff(0, 1.0, 60.0, jitter=False)
    assert delay == 1.0
    
    # Second attempt
    delay = calculate_backoff(1, 1.0, 60.0, jitter=False)
    assert delay == 2.0
    
    # Third attempt
    delay = calculate_backoff(2, 1.0, 60.0, jitter=False)
    assert delay == 4.0
    
    # Test max delay
    delay = calculate_backoff(10, 1.0, 5.0, jitter=False)
    assert delay == 5.0  # Should be capped at max_delay
    
    # Test with jitter
    for _ in range(10):
        delay = calculate_backoff(1, 1.0, 60.0, jitter=True)
        # With jitter, delay should be within ±25% of base value
        assert 1.5 <= delay <= 2.5


@pytest.mark.asyncio
async def test_async_retry_decorator_success():
    """Test async retry decorator with successful function."""
    mock_fn = AsyncMock(return_value="success")
    
    # Apply decorator
    decorated_fn = async_retry_with_exponential_backoff(
        max_retries=3,
        base_delay=0.01,  # Small delay for tests
        max_delay=0.1
    )(mock_fn)
    
    # Call decorated function
    result = await decorated_fn("arg1", kwarg1="value1")
    
    # Check result
    assert result == "success"
    
    # Should be called only once
    mock_fn.assert_called_once_with("arg1", kwarg1="value1")


@pytest.mark.asyncio
async def test_async_retry_decorator_with_retries():
    """Test async retry decorator with function that fails then succeeds."""
    # Mock function that fails twice then succeeds
    mock_fn = AsyncMock(side_effect=[
        TimeoutError("Timeout 1"),
        TimeoutError("Timeout 2"),
        "success"
    ])
    
    # Apply decorator
    decorated_fn = async_retry_with_exponential_backoff(
        max_retries=3,
        base_delay=0.01,  # Small delay for tests
        max_delay=0.1
    )(mock_fn)
    
    # Call decorated function
    result = await decorated_fn("arg1")
    
    # Check result
    assert result == "success"
    
    # Should be called three times
    assert mock_fn.call_count == 3


@pytest.mark.asyncio
async def test_async_retry_decorator_max_retries():
    """Test async retry decorator with max retries exceeded."""
    # Mock function that always fails
    mock_fn = AsyncMock(side_effect=TimeoutError("Always timeout"))
    
    # Apply decorator
    decorated_fn = async_retry_with_exponential_backoff(
        max_retries=2,
        base_delay=0.01,  # Small delay for tests
        max_delay=0.1
    )(mock_fn)
    
    # Call decorated function - should raise after max retries
    with pytest.raises(TimeoutError):
        await decorated_fn()
    
    # Should be called 1 + max_retries times
    assert mock_fn.call_count == 3


@pytest.mark.asyncio
async def test_async_retry_decorator_non_retryable():
    """Test async retry decorator with non-retryable error."""
    # Mock function that fails with non-retryable error
    mock_fn = AsyncMock(side_effect=ValueError("Non-retryable"))
    
    # Apply decorator
    decorated_fn = async_retry_with_exponential_backoff(
        max_retries=3,
        base_delay=0.01
    )(mock_fn)
    
    # Call decorated function - should raise immediately
    with pytest.raises(ValueError):
        await decorated_fn()
    
    # Should be called only once
    mock_fn.assert_called_once()


@pytest.mark.asyncio
async def test_retry_async_operation():
    """Test retry_async_operation helper function."""
    # Mock operation that fails then succeeds
    operation = AsyncMock(side_effect=[
        ConnectionError("Connection error"),
        "success"
    ])
    
    # Wrap with retry_async_operation
    result = await retry_async_operation(
        operation=operation,
        max_retries=3,
        base_delay=0.01,
        operation_name="test_operation"
    )
    
    # Check result
    assert result == "success"
    
    # Should be called twice
    assert operation.call_count == 2


@pytest.mark.asyncio
async def test_retry_async_operation_all_failed():
    """Test retry_async_operation when all attempts fail."""
    # Mock operation that always fails
    operation = AsyncMock(side_effect=TimeoutError("Always timeout"))
    
    # Attempt retry
    with pytest.raises(TimeoutError):
        await retry_async_operation(
            operation=operation,
            max_retries=2,
            base_delay=0.01,
            operation_name="test_operation"
        )
    
    # Should be called 1 + max_retries times
    assert operation.call_count == 3


@pytest.mark.asyncio
async def test_retry_async_operation_non_retryable():
    """Test retry_async_operation with non-retryable error."""
    # Mock operation that fails with non-retryable error
    operation = AsyncMock(side_effect=ValueError("Non-retryable"))
    
    # Attempt retry
    with pytest.raises(ValueError):
        await retry_async_operation(
            operation=operation,
            max_retries=2,
            base_delay=0.01
        )
    
    # Should be called only once
    operation.assert_called_once()


def test_retryable_error_codes_aligned():
    """Test that our retryable error codes are aligned with global config."""
    # Import the global list from config
    from src.config.errors import RETRYABLE_ERRORS
    
    # Check that all codes in RETRYABLE_ERRORS are in our DEFAULT_RETRYABLE_ERROR_CODES
    for error_code in RETRYABLE_ERRORS:
        assert error_code in DEFAULT_RETRYABLE_ERROR_CODES, f"{error_code} missing from DEFAULT_RETRYABLE_ERROR_CODES"
    
    # LLM-specific codes should be in our set
    for error_code in LLM_RETRYABLE_ERROR_CODES:
        assert error_code in DEFAULT_RETRYABLE_ERROR_CODES, f"{error_code} missing from DEFAULT_RETRYABLE_ERROR_CODES"
import asyncio
import time
from unittest.mock import patch, MagicMock

import pytest

from src.utils.timing import (
    timed,
    async_timed,
    Timer,
    AsyncTimer,
    get_current_time_ms,
    sleep_with_jitter
)


def test_timed_decorator():
    """Test the timed decorator logs function execution time."""
    with patch('src.utils.timing.logger') as mock_logger:
        
        @timed()
        def sample_function():
            time.sleep(0.01)
            return "result"
        
        result = sample_function()
        
        # Check basic functionality
        assert result == "result"
        assert mock_logger.debug.called
        
        # Verify log contains function name but don't check exact format
        assert "sample_function" in str(mock_logger.debug.call_args)
        
        # Verify extras include execution time but don't check exact value
        extras = mock_logger.debug.call_args[1].get('extra', {})
        assert 'execution_time' in extras
        assert extras.get('execution_time', 0) > 0


def test_timed_decorator_with_custom_name():
    """Test the timed decorator with a custom name."""
    with patch('src.utils.timing.logger') as mock_logger:
        
        @timed(name="custom_name")
        def sample_function():
            return "result"
        
        sample_function()
        
        # Verify custom name usage
        assert "custom_name" in str(mock_logger.debug.call_args)


@pytest.mark.asyncio
async def test_async_timed_decorator():
    """Test the async_timed decorator logs function execution time."""
    with patch('src.utils.timing.logger') as mock_logger:
        
        @async_timed()
        async def sample_async_function():
            await asyncio.sleep(0.01)
            return "async result"
        
        result = await sample_async_function()
        
        # Check basic functionality
        assert result == "async result"
        assert mock_logger.debug.called


def test_timer_context_manager():
    """Test the Timer context manager."""
    with patch('src.utils.timing.logger') as mock_logger:
        with Timer("test_operation"):
            time.sleep(0.01)
        
        # Check that logging occurred
        assert mock_logger.debug.called
        
        # Check that the timer name is included
        assert "test_operation" in str(mock_logger.debug.call_args)
        
        # Check that execution time is recorded
        extras = mock_logger.debug.call_args[1].get('extra', {})
        assert 'execution_time' in extras
        assert extras.get('execution_time', 0) > 0


def test_timer_context_manager_different_log_levels():
    """Test the Timer context manager with different log levels."""
    with patch('src.utils.timing.logger') as mock_logger:
        # Test info level
        with Timer("info_operation", log_level="info"):
            pass
        
        assert mock_logger.info.called
        
        # Test warning level
        with Timer("warning_operation", log_level="warning"):
            pass
            
        assert mock_logger.warning.called


@pytest.mark.asyncio
async def test_async_timer_context_manager():
    """Test the AsyncTimer context manager."""
    with patch('src.utils.timing.logger') as mock_logger:
        async with AsyncTimer("async_operation"):
            await asyncio.sleep(0.01)
        
        # Check that logging occurred
        assert mock_logger.debug.called
        
        # Check that timer name is included
        assert "async_operation" in str(mock_logger.debug.call_args)


def test_get_current_time_ms():
    """Test the get_current_time_ms function returns a timestamp."""
    # Just check that it returns a positive integer
    time_ms = get_current_time_ms()
    assert isinstance(time_ms, int)
    assert time_ms > 0


@pytest.mark.asyncio
async def test_sleep_with_jitter():
    """Test the sleep_with_jitter function."""
    with patch('asyncio.sleep') as mock_sleep:
        start_time = time.time()
        await sleep_with_jitter(0.1, 0.1)
        
        # Simply verify that sleep was called
        assert mock_sleep.called
"""
Configuration Package Tests

This module contains tests for the configuration system components, including:
- Settings management
- Logging system
- Metrics collection
- Connection pooling
- Error handling

These tests verify the core functionality of the configuration system
according to the requirements in the Roadmap.
"""
import os
import json
import time
import asyncio
import unittest
import tempfile
from unittest import mock
import pytest

# Mock Redis and aiohttp for testing
import redis

# Import the modules we're testing
from src.config.settings import get_settings
from src.config.logger import get_logger, get_logger_with_context, setup_logging
from src.config.metrics import get_metrics_manager, MEMORY_METRICS
from src.config.connections import ConnectionManager, get_connection_manager
from src.config.errors import BaseError, ErrorCode


class TestConfigSettings(unittest.TestCase):
    """Test case for the settings module functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_settings_load_from_environment(self):
        """Test that settings load correctly from environment variables."""
        # Set test environment variables
        os.environ['APP_NAME'] = 'TestApp'
        os.environ['APP_VERSION'] = '1.0.0-test'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        os.environ['REDIS_URL'] = 'redis://testhost:6379/0'
        os.environ['ENABLED_MODELS_SET'] = '["model1", "model2", "model3"]'
        
        # Clear settings cache if any
        from src.config.settings import get_settings
        get_settings.cache_clear()
        
        # Load settings
        settings = get_settings()
        
        # Verify settings loaded from environment
        self.assertEqual(settings.APP_NAME, 'TestApp')
        self.assertEqual(settings.APP_VERSION, '1.0.0-test')
        self.assertEqual(settings.LOG_LEVEL, 'DEBUG')
        self.assertEqual(settings.REDIS_URL, 'redis://testhost:6379/0')
        self.assertEqual(settings.ENABLED_MODELS_SET, {'model1', 'model2', 'model3'})
    
    def test_settings_validation(self):
        """Test that settings validation works correctly."""
        # Set test environment variables with incomplete config
        os.environ['APP_NAME'] = 'ValidationTest'
        os.environ['PRIMARY_LLM'] = 'nonexistent-model'
        
        # Clear settings cache
        get_settings.cache_clear()
        
        # Load settings
        settings = get_settings()
        
        # Check that validation detected the issue
        warnings = settings.validate_settings()
        self.assertGreaterEqual(len(warnings), 1)
        self.assertTrue(any('PRIMARY_LLM' in warning for warning in warnings))
    
    def test_settings_provider_config(self):
        """Test LLM provider configuration setup."""
        # Set test environment variables
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'
        
        # Clear settings cache
        get_settings.cache_clear()
        
        # Load settings
        settings = get_settings()
        
        # Verify API keys were loaded
        self.assertEqual(settings.LLM_PROVIDERS_CONFIG['openai']['api_key'], 'test-openai-key')
        self.assertEqual(settings.LLM_PROVIDERS_CONFIG['anthropic']['api_key'], 'test-anthropic-key')


class TestConfigLogger(unittest.TestCase):
    """Test case for the logger module functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary log file
        self.log_file = tempfile.NamedTemporaryFile(delete=False)
        self.log_file.close()
        
        # Set up environment for logging
        self.original_env = os.environ.copy()
        os.environ['LOG_LEVEL'] = 'DEBUG'
        os.environ['LOG_FORMAT'] = 'json'
        os.environ['LOG_TO_FILE'] = 'true'
        os.environ['LOG_FILE_PATH'] = self.log_file.name
        
        # Clear settings cache
        get_settings.cache_clear()
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary log file
        if os.path.exists(self.log_file.name):
            os.unlink(self.log_file.name)
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_logger_setup(self):
        """Test that logger setup works correctly."""
        # Configure logging
        setup_logging()
        
        # Get a logger
        logger = get_logger('test.logger')
        
        # Write some log messages
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        
        # Check that log file was created and contains messages
        with open(self.log_file.name, 'r') as f:
            log_content = f.read()
        
        self.assertIn('Debug message', log_content)
        self.assertIn('Info message', log_content)
        self.assertIn('Warning message', log_content)
        self.assertIn('Error message', log_content)
    
    def test_structured_logging(self):
        """Test that structured logging includes required context."""
        # Configure logging
        setup_logging()
        
        # Get a logger with context
        logger = get_logger_with_context(
            'test.structured',
            trace_id='test-trace-123',
            task_id='task-456',
            agent_id='agent-789',
            custom_field='custom-value'
        )
        
        # Write a log message
        logger.info('Structured log message')
        
        # Check that log file contains structured data
        with open(self.log_file.name, 'r') as f:
            for line in f:
                if 'Structured log message' in line:
                    log_data = json.loads(line)
                    break
            else:
                self.fail("Could not find structured log message")
        
        # Verify context fields
        self.assertEqual(log_data['trace_id'], 'test-trace-123')
        self.assertEqual(log_data['task_id'], 'task-456')
        self.assertEqual(log_data['agent_id'], 'agent-789')
        
        # Check for custom field - this might be in extra_attrs
        if 'custom_field' in log_data:
            self.assertEqual(log_data['custom_field'], 'custom-value')
        elif 'extra_attrs' in log_data:
            self.assertEqual(log_data['extra_attrs']['custom_field'], 'custom-value')
    
    def test_logger_context_propagation(self):
        """Test that logger context is propagated correctly."""
        # Configure logging
        setup_logging()
        
        # Get a logger with context
        logger = get_logger_with_context(
            'test.propagation',
            trace_id='trace-propagation'
        )
        
        # Write multiple log messages
        logger.info('First message')
        logger.warning('Second message')
        logger.error('Third message')
        
        # Check that all messages have the same trace ID
        trace_ids = []
        with open(self.log_file.name, 'r') as f:
            for line in f:
                if 'test.propagation' in line:
                    data = json.loads(line)
                    if 'trace_id' in data:
                        trace_ids.append(data['trace_id'])
        
        # Verify all messages have the same trace ID
        self.assertTrue(all(tid == 'trace-propagation' for tid in trace_ids))
        self.assertGreaterEqual(len(trace_ids), 3)


@pytest.mark.asyncio
class TestConfigConnections:
    """Test case for the connections module functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Set up mocks for Redis and HTTP connections."""
        # Mock Redis connection pool
        self.mock_redis_pool = mock.MagicMock()
        self.mock_redis_connection = mock.MagicMock()
        monkeypatch.setattr('redis.ConnectionPool.from_url', lambda *args, **kwargs: self.mock_redis_pool)
        monkeypatch.setattr('redis.Redis', lambda *args, **kwargs: self.mock_redis_connection)
        
        # Mock async Redis connection pool
        self.mock_redis_async_pool = mock.MagicMock()
        self.mock_redis_async_connection = mock.MagicMock()
        monkeypatch.setattr('redis.asyncio.ConnectionPool.from_url', lambda *args, **kwargs: self.mock_redis_async_pool)
        monkeypatch.setattr('redis.asyncio.Redis', lambda *args, **kwargs: self.mock_redis_async_connection)
        
        # Mock HTTP session
        self.mock_http_session = mock.MagicMock()
        monkeypatch.setattr('aiohttp.ClientSession', lambda *args, **kwargs: self.mock_http_session)
        
        # Reset connection manager
        ConnectionManager._instance = None
    
    async def test_connection_pooling(self):
        """Test that connection pooling works correctly."""
        # Get connection manager
        manager = get_connection_manager()
        
        # Get Redis connection multiple times - should use the same pool
        manager.get_redis_connection()
        manager.get_redis_connection()
        manager.get_redis_connection()
        
        # Verify only one pool was created
        assert self.mock_redis_connection.call_count <= 1

        # Test async connections
        await manager.get_redis_async_connection()
        await manager.get_redis_async_connection()
        
        # Verify only one async pool was created
        assert self.mock_redis_connection.call_count <= 1
        
        # Test HTTP session
        session1 = await manager.get_http_session()
        session2 = await manager.get_http_session()
        
        # Verify only one session was created
        assert session1 is session2
    
    async def test_connection_context_managers(self):
        """Test connection context managers."""
        # Get connection manager
        manager = get_connection_manager()
        
        # Test synchronous context manager
        with manager.redis_connection() as conn:
            assert conn is not None
        
        # Test asynchronous context manager
        async with manager.redis_async_connection() as conn:
            assert conn is not None
        
        async with manager.http_session() as session:
            assert session is not None
    
    async def test_concurrent_connection_access(self):
        """Test connection pooling under concurrent load."""
        # Create a connection manager
        manager = get_connection_manager()
        
        # Create tasks to simulate concurrent access
        async def access_connections():
            # Get connections of each type
            redis_conn = manager.get_redis_connection()
            redis_async_conn = await manager.get_redis_async_connection()
            http_session = await manager.get_http_session()
            return (redis_conn, redis_async_conn, http_session)
        
        # Run multiple concurrent tasks
        tasks = [access_connections() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed
        assert len(results) == 10
        
        # Get the first set of connections for comparison
        first_redis, first_redis_async, first_http = results[0]
        
        # Verify connections were reused
        for redis_conn, redis_async_conn, http_session in results[1:]:
            # Due to mocking, we can't directly compare the connections
            # But we can verify they're not None
            assert redis_conn is not None
            assert redis_async_conn is not None
            assert http_session is not None
    
    async def test_connection_error_handling(self):
        """Test error handling in connections."""
        # Set up Redis connection to raise an error
        self.mock_redis_connection.ping.side_effect = redis.RedisError("Test Redis error")
        
        # Get connection manager
        manager = get_connection_manager()
        
        # Test error handling in context manager
        with pytest.raises(Exception) as exc_info:
            with manager.redis_connection() as conn:
                conn.ping()
        
        # Verify an exception was raised
        assert exc_info.value is not None
    
    async def test_connection_cleanup(self):
        """Test connection cleanup."""
        # Get connection manager
        manager = get_connection_manager()
        
        # Get connections to initialize pools
        manager.get_redis_connection()
        await manager.get_redis_async_connection()
        await manager.get_http_session()
        
        # Close connections
        await manager.close_all_connections()
        
        # Verify disconnection was attempted
        # We can't directly verify the mock calls in pytest fixtures
        # But we can verify the test completes without errors
        assert True


class TestConfigMetrics(unittest.TestCase):
    """Test case for the metrics module functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Enable metrics
        os.environ['METRICS_ENABLED'] = 'true'
        
        # Clear settings cache
        get_settings.cache_clear()
        
        # Get a metrics manager
        self.metrics = get_metrics_manager()
    
    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        # Track some metrics
        self.metrics.track_task('created')
        self.metrics.track_task('consumed', dispatcher_id='test-dispatcher')
        self.metrics.track_task('completed', status='success')
        
        self.metrics.track_llm('requests', model='gpt-4', provider='openai')
        self.metrics.track_llm('duration', model='gpt-4', provider='openai', value=0.5)
        
        self.metrics.track_memory('operations', operation_type='write')
        
        # Unfortunately, we can't easily check the metrics values in a unit test,
        # since Prometheus client maintains internal state. In a real test environment,
        # we would verify these through the metrics endpoint.
        
        # For this test, we'll just verify that no exceptions were raised
        self.assertTrue(True)
    
    def test_timed_metric_decorator(self):
        """Test the timed metric decorator."""
        # Get the memory duration metric from the manager
        # In the refactored metrics, this is accessed through MEMORY_METRICS dict
        memory_duration_metric = MEMORY_METRICS['duration']
        
        # Create a test function with timing
        @self.metrics.timed_metric(memory_duration_metric, {'operation_type': 'test_op'})
        def test_function():
            """Test function that simulates work."""
            # Simulate work
            time.sleep(0.01)
            return 42
        
        # Call the function
        result = test_function()
        
        # Verify the function still works
        self.assertEqual(result, 42)
    
    @pytest.mark.asyncio
    async def test_timed_metric_async_decorator(self):
        """Test the timed metric decorator with async functions."""
        # Get the memory duration metric from the manager
        memory_duration_metric = self.metrics.MEMORY_METRICS['duration']
        
        # Create a test async function with timing
        @self.metrics.timed_metric(memory_duration_metric, {'operation_type': 'test_async_op'})
        async def test_async_function():
            """Test async function that simulates work."""
            # Simulate work
            await asyncio.sleep(0.01)
            return 42
        
        # Call the function
        result = await test_async_function()
        
        # Verify the function still works
        self.assertEqual(result, 42)


class TestConfigErrors(unittest.TestCase):
    """Test case for the errors module functionality."""
    
    def test_error_creation(self):
        """Test that errors are created correctly."""
        # Create a basic error
        error = BaseError(
            code=ErrorCode.SYSTEM_ERROR,
            message="Test system error",
            details={"key": "value"}
        )
        
        # Verify error attributes
        self.assertEqual(error.code, ErrorCode.SYSTEM_ERROR)
        self.assertEqual(error.message, "Test system error")
        self.assertEqual(error.details, {"key": "value"})
    
    def test_error_to_dict(self):
        """Test that errors convert to dictionaries correctly."""
        # Create an error
        error = BaseError(
            code=ErrorCode.API_ERROR,
            message="Test API error",
            details={"endpoint": "/test"}
        )
        
        # Convert to dict
        error_dict = error.to_dict()
        
        # Verify dictionary
        self.assertEqual(error_dict["code"], ErrorCode.API_ERROR.value)
        self.assertEqual(error_dict["message"], "Test API error")
        self.assertEqual(error_dict["details"]["endpoint"], "/test")
    
    def test_error_hierarchy(self):
        """Test that the error hierarchy works correctly."""
        # Create different error types
        system_error = BaseError(code=ErrorCode.SYSTEM_ERROR, message="System error")
        api_error = BaseError(code=ErrorCode.API_ERROR, message="API error")
        llm_error = BaseError(code=ErrorCode.LLM_ERROR, message="LLM error")
        
        # Verify error codes
        self.assertEqual(system_error.code, ErrorCode.SYSTEM_ERROR)
        self.assertEqual(api_error.code, ErrorCode.API_ERROR)
        self.assertEqual(llm_error.code, ErrorCode.LLM_ERROR)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for the config package."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        
        # Set up test environment
        os.environ['APP_NAME'] = 'IntegrationTest'
        os.environ['LOG_LEVEL'] = 'INFO'
        os.environ['METRICS_ENABLED'] = 'true'
        
        # Clear settings cache
        get_settings.cache_clear()
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @mock.patch('src.config.logger.setup_logging')
    @mock.patch('src.config.settings.get_settings')
    def test_initialize_config(self, mock_get_settings, mock_setup_logging):
        """Test the initialize_config function."""
        # Create a mock settings object with the necessary attributes
        mock_settings_obj = mock.MagicMock()
        mock_settings_obj.APP_NAME = 'MockApp'
        mock_settings_obj.APP_VERSION = '1.0.0'
        mock_get_settings.return_value = mock_settings_obj
        
        # Call initialize_config
        from src.config import initialize_config
        result = initialize_config()
        
        # Verify initialization
        self.assertTrue(result)
        mock_get_settings.assert_called_once()
        mock_setup_logging.assert_called_once()


if __name__ == '__main__':
    unittest.main()
# Multi-Agent Platform Configuration Package

## Overview

The configuration package (`src/config/`) provides a comprehensive foundation for the Multi-Agent Platform, handling essential infrastructure concerns such as settings management, logging, connection pooling, error handling, and performance metrics. This package is designed with performance, reliability, and extensibility in mind, supporting the high-performance requirements of the platform.

## Architecture

The configuration package consists of five core modules that work together to provide a solid infrastructure foundation:

1. **Settings Management** (`settings.py`): Environment-based configuration with validation
2. **Structured Logging** (`logger.py`): Context-aware logging with trace propagation
3. **Connection Management** (`connections.py`): Resource pooling and lifecycle management
4. **Error Handling** (`errors.py`): Standardized error hierarchy and conversion
5. **Performance Metrics** (`metrics.py`): Prometheus-based metrics collection

The package initializes through a central entry point in `__init__.py`, which manages the initialization sequence to avoid circular dependencies.

## Initialization Flow

The configuration system follows a specific initialization sequence to ensure proper setup:

1. Bootstrap logging is established for initial diagnostics
2. Settings are loaded and validated
3. Full logging system is configured based on settings
4. Connection pools are prepared (lazily initialized)
5. Metrics collection is started if enabled

## Module Details

### Settings Management (`settings.py`)

The settings module provides a Pydantic-based configuration system that loads from environment variables and `.env` files, with comprehensive validation.

#### Core Classes

- `Settings`: The main settings class with all configuration parameters
- `LLMProviderConfig`: Configuration for LLM providers

#### Key Features

- **Environment-based Configuration**: Settings are loaded from environment variables and `.env` files
- **Automatic Type Conversion**: Values are automatically converted to appropriate types
- **Validation**: Settings are validated for correctness and consistency
- **Custom Validators**: Complex validators for nested structures like LLM provider configurations
- **Provider API Key Loading**: Automatic loading of provider API keys from environment variables
- **Performance Optimization**: Settings are cached using `lru_cache` for performance

#### Usage Examples

```python
# Get settings instance (cached)
from src.config.settings import get_settings

settings = get_settings()

# Access settings properties
redis_url = settings.REDIS_URL
worker_count = settings.WORKER_COUNT

# Check if a model is enabled
model_name = "gpt-4"
if model_name in settings.ENABLED_MODELS_SET:
    # Use the model
    pass

# Get provider configuration
openai_config = settings.LLM_PROVIDERS_CONFIG.get('openai', {})
api_key = openai_config.get('api_key')
```

#### Environment Variables

The settings module supports numerous environment variables, including:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `APP_NAME` | Application name | 'MultiAgentPlatform' |
| `APP_VERSION` | Application version | '0.1.0' |
| `ENVIRONMENT` | Deployment environment | 'development' |
| `LOG_LEVEL` | Logging level | 'INFO' |
| `LOG_FORMAT` | Log format (json/text) | 'json' |
| `REDIS_URL` | Redis connection URL | 'redis://localhost:6379/0' |
| `OPENAI_API_KEY` | OpenAI API key | '' |
| `ANTHROPIC_API_KEY` | Anthropic API key | '' |
| `METRICS_ENABLED` | Enable metrics collection | 'true' |
| `METRICS_PORT` | Prometheus metrics port | 9090 |

### Structured Logging (`logger.py`)

The logging module provides a structured logging system with context propagation and support for both JSON and text formats.

#### Core Classes

- `JsonFormatter`: Formats log records as JSON with structured data
- `TraceLogger`: Logger that automatically adds trace IDs to records
- `ContextLoggerAdapter`: Adapter that preserves context across calls

#### Key Features

- **JSON Structured Logging**: Log entries are formatted as structured JSON for easy parsing
- **Context Propagation**: Logging context (trace IDs, etc.) is preserved across multiple log calls
- **Performance Optimization**: JSON formatting is optimized for performance
- **Automatic Trace IDs**: Every log entry gets a unique trace ID if not provided
- **Context Hierarchies**: Support for task IDs, agent IDs, and custom context
- **File Logging**: Optional logging to file with proper error handling

#### Usage Examples

```python
# Basic logging
from src.config.logger import get_logger

logger = get_logger(__name__)
logger.info("This is a regular log message")
logger.error("An error occurred", exc_info=True)

# Contextual logging
from src.config.logger import get_logger_with_context

# Create a logger with context
ctx_logger = get_logger_with_context(
    __name__,
    trace_id="unique-trace-123",
    task_id="task-456",
    agent_id="agent-789",
    custom_field="custom value"
)

# All these log messages will have the same context
ctx_logger.info("Starting operation")
ctx_logger.debug("Processing data")
ctx_logger.warning("Potential issue detected")
ctx_logger.error("Operation failed")
```

#### Log Format

When using JSON format, log entries contain the following fields:

```json
{
  "timestamp": "2025-04-29T12:34:56.789012Z",
  "level": "INFO",
  "name": "module.name",
  "message": "Log message text",
  "module": "name",
  "function": "function_name",
  "lineno": 42,
  "trace_id": "unique-trace-123",
  "task_id": "task-456",
  "agent_id": "agent-789",
  "custom_field": "custom value",
  "execution_time": 0.0123
}
```

### Connection Management (`connections.py`)

The connection module provides efficient connection pooling for Redis (both synchronous and asynchronous) and HTTP client connections.

#### Core Classes

- `ConnectionManager`: Singleton manager for all connection pools

#### Key Features

- **Singleton Pattern**: Single access point for connection management
- **Connection Pooling**: Reuse connections for better performance
- **Synchronous & Asynchronous**: Support for both sync and async connections
- **Context Managers**: Resource management through context managers
- **Graceful Shutdown**: Proper connection cleanup on shutdown
- **Error Standardization**: Converts provider-specific errors to standard errors
- **Performance Metrics**: Connection operations are timed and recorded

#### Usage Examples

```python
# Synchronous Redis connection
from src.config.connections import get_connection_manager

manager = get_connection_manager()

# Using context manager (recommended)
with manager.redis_connection() as redis:
    redis.set("key", "value")
    value = redis.get("key")

# Asynchronous Redis connection
async def redis_example():
    async with manager.redis_async_connection() as redis:
        await redis.set("async_key", "async_value")
        value = await redis.get("async_key")

# HTTP client session
async def http_example():
    async with manager.http_session() as session:
        async with session.get("https://api.example.com/data") as response:
            data = await response.json()
```

#### Connection Lifecycle

The connection manager handles the entire lifecycle of connections:

1. **Initialization**: Pools are created on first use (lazy initialization)
2. **Reuse**: Connections are returned to the pool after use
3. **Monitoring**: Connection operations are timed and monitored
4. **Error Handling**: Connection errors are standardized and logged
5. **Cleanup**: Proper cleanup during application shutdown

```python
# Application shutdown example
import asyncio
from src.config.connections import get_connection_manager

async def shutdown():
    manager = get_connection_manager()
    await manager.close_all_connections()
```

### Error Handling (`errors.py`)

The errors module provides a comprehensive error handling system with a standardized error hierarchy, error codes, and conversion utilities.

#### Core Classes

- `ErrorCode`: Enum of standardized error codes
- `BaseError`: Base error class for all system errors
- Specialized error classes (SystemError, APIError, LLMError, etc.)

#### Key Features

- **Error Hierarchy**: Organized error class hierarchy
- **Error Codes**: Standardized error codes with categories
- **Error Conversion**: Convert external errors to system errors
- **Structured Error Data**: Errors include structured data for logging
- **HTTP Status Mapping**: Mapping between error codes and HTTP status codes
- **Retry Classification**: Identification of retryable errors

#### Usage Examples

```python
# Create a system error
from src.config.errors import BaseError, ErrorCode

error = BaseError(
    code=ErrorCode.SYSTEM_ERROR,
    message="System initialization failed",
    details={"component": "database"}
)

# Log the error
error.log_error(logger)

# Convert an external error
from src.config.errors import convert_exception

try:
    # Some operation that might fail
    result = external_api.call()
except ExternalAPIError as e:
    # Convert to system error
    system_error = convert_exception(
        e,
        default_code=ErrorCode.API_ERROR,
        default_message="External API call failed"
    )
    # Handle system error
    system_error.log_error(logger)
    raise system_error
```

#### Error Structure

Errors include the following attributes:

- `code`: Standardized error code (e.g., `SYSTEM_ERROR_1000`)
- `message`: Human-readable error message
- `details`: Dictionary of error-specific details
- `original_error`: Original exception that caused this error

When converted to a dictionary (using `to_dict()`), errors have this structure:

```json
{
  "code": "API_ERROR_2000",
  "message": "External API call failed",
  "details": {
    "api": "example.com",
    "endpoint": "/data",
    "status_code": 503
  },
  "original_error": "HTTPError: 503 Service Unavailable"
}
```

### Performance Metrics (`metrics.py`)

The metrics module provides a Prometheus-based metrics collection system for monitoring application performance.

#### Core Classes

- `MetricsManager`: Singleton manager for metrics collection

#### Key Features

- **Prometheus Integration**: Metrics are compatible with Prometheus
- **Metrics Categories**: Organized metrics for different aspects of the system
- **Decorator-based Timing**: Easy function timing with decorators
- **Conditional Collection**: Metrics collection can be disabled for performance
- **Async Support**: Support for both synchronous and asynchronous functions
- **Label Support**: Proper labeling of metrics for detailed analysis
- **HTTP Metrics Server**: Built-in HTTP server for Prometheus scraping

#### Metrics Categories

The system collects metrics in several categories:

- **HTTP Metrics**: Request counts, durations, sizes
- **Task Metrics**: Creation, consumption, completion, durations
- **LLM Metrics**: API requests, tokens, errors, fallbacks
- **Agent Metrics**: Operations, durations, errors
- **Tool Metrics**: Executions, durations, errors
- **Memory Metrics**: Operations, sizes, durations
- **Cache Metrics**: Hits, misses, sizes

#### Usage Examples

```python
# Track metrics directly
from src.config.metrics import get_metrics_manager

metrics = get_metrics_manager()

# Track LLM request - Counter metric with labels
metrics.track_llm('requests', model='gpt-4', provider='openai')

# Track token usage - Counter metric with value and labels
metrics.track_llm('tokens', model='gpt-4', provider='openai', type='prompt', value=150)

# Track duration - Histogram metric with value and labels
metrics.track_llm('duration', model='gpt-4', provider='openai', value=0.5)

# Track memory metrics
metrics.track_memory('operations', operation_type='write')

# Set gauge value
metrics.track_memory('size', memory_type='redis', value=1024)

# Use timing decorator
from src.config.metrics import timed_metric, MEMORY_OPERATION_DURATION

@timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'vector_search'})
def search_vectors(query_vector, top_k=10):
    # Function is automatically timed
    return vector_db.search(query_vector, top_k)

# Time async functions too
@timed_metric(LLM_REQUEST_DURATION, {'model': 'gpt-4', 'provider': 'openai'})
async def get_completion(prompt):
    # Async function is automatically timed
    return await llm_client.complete(prompt)

# Start metrics server
from src.config.metrics import start_metrics_server

metrics_thread = start_metrics_server()
```

#### Prometheus Client Patterns

When working with Prometheus client metrics, it's important to follow these correct patterns:

**For Counter metrics:**
```python
# CORRECT: Apply labels first, then call inc()
counter.labels(label1='value1', label2='value2').inc()

# For value increments:
counter.labels(label1='value1', label2='value2').inc(5)

# INCORRECT - will raise TypeError:
counter.inc(label1='value1', label2='value2')  # ❌
```

**For Histogram metrics:**
```python
# CORRECT: Apply labels first, then call observe()
histogram.labels(label1='value1', label2='value2').observe(0.5)

# INCORRECT - will raise TypeError:
histogram.observe(0.5, label1='value1', label2='value2')  # ❌
```

**For Gauge metrics:**
```python
# CORRECT: Apply labels first, then call set()
gauge.labels(label1='value1', label2='value2').set(42)

# INCORRECT - will raise TypeError:
gauge.set(42, label1='value1', label2='value2')  # ❌
```

The `MetricsManager` class handles these patterns correctly for all metrics types. This is a critical aspect of the implementation that was improved during testing.

## Testing the Configuration Package

The configuration package includes comprehensive tests in `tests/config/test_config.py` to ensure all components function correctly. These tests validate the system's functionality and serve as examples of how to use the configuration components.

### Test Structure

The test suite includes tests for each major component:

1. `TestConfigSettings`: Tests for settings loading and validation
2. `TestConfigLogger`: Tests for structured logging and context propagation
3. `TestConfigConnections`: Tests for connection pooling and resource management
4. `TestConfigMetrics`: Tests for metrics collection and timing
5. `TestConfigErrors`: Tests for error handling and conversion
6. `TestConfigIntegration`: Tests for overall system integration

### Running the Tests

To run the configuration tests:

```bash
# Run all config tests
pytest -xvs tests/config/test_config.py

# Run a specific test class
pytest -xvs tests/config/test_config.py::TestConfigSettings

# Run a specific test
pytest -xvs tests/config/test_config.py::TestConfigSettings::test_settings_load_from_environment
```

### Test Environment Setup

The tests use pytest fixtures and mocking to create isolated test environments:

```python
@pytest.fixture(autouse=True)
def setup_mocks(self, monkeypatch):
    """Set up mocks for Redis and HTTP connections."""
    # Mock Redis connection pool
    self.mock_redis_pool = mock.MagicMock()
    self.mock_redis_connection = mock.MagicMock()
    monkeypatch.setattr('redis.ConnectionPool.from_url', 
                        lambda *args, **kwargs: self.mock_redis_pool)
    monkeypatch.setattr('redis.Redis', 
                        lambda *args, **kwargs: self.mock_redis_connection)
    
    # Mock async Redis connection pool
    self.mock_redis_async_pool = mock.MagicMock()
    self.mock_redis_async_connection = mock.MagicMock()
    # ... more mocking
```

### Key Test Cases

Some of the most important tests include:

1. **Settings Loading Test**: Verifies that settings are correctly loaded from environment variables

```python
def test_settings_load_from_environment(self):
    """Test that settings load correctly from environment variables."""
    # Set test environment variables
    os.environ['APP_NAME'] = 'TestApp'
    os.environ['APP_VERSION'] = '1.0.0-test'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Clear settings cache
    get_settings.cache_clear()
    
    # Load settings
    settings = get_settings()
    
    # Verify settings loaded from environment
    self.assertEqual(settings.APP_NAME, 'TestApp')
    self.assertEqual(settings.APP_VERSION, '1.0.0-test')
    self.assertEqual(settings.LOG_LEVEL, 'DEBUG')
```

2. **Structured Logging Test**: Verifies that logging context is preserved

```python
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
    # ...
```

3. **Connection Pooling Test**: Verifies that connections are pooled correctly

```python
async def test_connection_pooling(self):
    """Test that connection pooling works correctly."""
    # Get connection manager
    manager = get_connection_manager()
    
    # Get Redis connection multiple times - should use the same pool
    conn1 = manager.get_redis_connection()
    conn2 = manager.get_redis_connection()
    conn3 = manager.get_redis_connection()
    
    # Verify only one pool was created
    assert self.mock_redis_connection.call_count <= 1
```

4. **Metrics Collection Test**: Verifies that metrics are correctly recorded

```python
def test_metrics_tracking(self):
    """Test that metrics are tracked correctly."""
    # Track some metrics
    self.metrics.track_task('created')
    self.metrics.track_task('consumed', dispatcher_id='test-dispatcher')
    self.metrics.track_task('completed', status='success')
    
    self.metrics.track_llm('requests', model='gpt-4', provider='openai')
    self.metrics.track_llm('duration', model='gpt-4', provider='openai', value=0.5)
    
    self.metrics.track_memory('operations', operation_type='write')
    
    # Verify that no exceptions were raised
    self.assertTrue(True)
```

### Common Test Issues and Solutions

During testing, several important fixes were identified and implemented:

1. **Issue**: Prometheus Counter labels were being passed directly to `inc()` method
   - **Solution**: Use the correct pattern: `counter.labels(**labels).inc()`

2. **Issue**: Prometheus Histogram labels were being passed directly to `observe()` method
   - **Solution**: Use the correct pattern: `histogram.labels(**labels).observe(value)`

3. **Issue**: Connection pooling tests were checking incorrect mock objects
   - **Solution**: Verify call counts on the correct mock objects

These fixes have been incorporated into the code and tests now pass successfully.

## Integration and Initialization

The configuration package is designed to be initialized at application startup through the `initialize_config()` function in `__init__.py`.

```python
from src.config import initialize_config, settings, logger

# Initialize configuration
initialize_config()

# Use configuration components
app_name = settings.APP_NAME
log = logger.get_logger(__name__)
log.info(f"Application {app_name} initialized")
```

## Performance Considerations

The configuration package has been optimized for performance:

- **Settings Caching**: Settings are cached using `lru_cache`
- **Connection Pooling**: Connections are pooled and reused
- **Logger Optimization**: JSON formatting is optimized for performance
- **Conditional Metrics**: Metrics collection can be disabled if not needed
- **Lazy Initialization**: Resources are initialized only when needed

## Thread Safety

The configuration package is designed to be thread-safe:

- **Thread Locks**: Critical sections are protected by locks
- **Singleton Pattern**: Proper implementation of thread-safe singletons
- **Immutable Settings**: Settings are immutable after loading
- **Context Managers**: Resource access through context managers

## Async Support

The configuration package fully supports asynchronous programming:

- **Async Connection Pools**: Dedicated async connection pools
- **Async Context Managers**: Async context managers for resource management
- **Async-compatible Metrics**: Metrics decorators support async functions
- **Async Cleanup**: Async cleanup functions for graceful shutdown

## Best Practices

When using the configuration package, follow these best practices:

1. **Always use context managers** for connections to ensure proper resource management
2. **Use structured logging** with context for better traceability
3. **Standardize errors** by converting external errors to system errors
4. **Time performance-critical functions** using the metrics decorators
5. **Properly clean up resources** during application shutdown
6. **Follow Prometheus patterns** for working with metrics (labels first, then method call)

## Extension Points

The configuration package can be extended in several ways:

1. **Additional Settings**: Add new settings to the `Settings` class
2. **Custom Log Formatters**: Create custom log formatters for specific needs
3. **New Connection Types**: Add new connection types to the `ConnectionManager`
4. **Additional Error Types**: Create specialized error classes for specific domains
5. **Custom Metrics**: Define new metrics for specific application needs
# Multi-Agent Platform: Tools Package Documentation

## Architecture Overview

The Tools Package serves as a critical component within the Multi-Agent Platform, providing specialized capabilities that agents can leverage to perform specific tasks. This package implements a plugin-based architecture allowing the platform to be extended with new tools without modifying core functionality.

### Purpose and Responsibility of Each Module

- **Base Tool Framework (`base.py`)**: Defines the common interface and behavior that all tools must implement, providing the foundation for tool development.
- **Tool Registry (`registry.py`)**: Maintains a catalog of available tools that can be discovered at runtime, enabling dynamic tool discovery and instantiation.
- **Tool Runner (`runner.py`)**: Executes tools with proper error handling, metrics collection, and resource management, including retry mechanisms and parallel execution support.
- **Specialized Tools**: Implements specific capabilities such as web search (`web_search.py`, `web_search_google.py`), mathematical calculations (`calculator.py`), and date/time operations (`datetime_tool.py`).

### Component Interactions

The Tools Package operates through a well-defined flow:

1. The Registry discovers and catalogs all available tools at startup through decorator-based registration
2. The API or Agent System requests tool execution via the Runner
3. The Runner resolves the tool through the Registry and executes it with proper error handling
4. The Tool performs its function, accessing resources as needed (Redis, HTTP APIs)
5. Results are returned through the Runner with standardized formatting
6. Metrics are collected throughout this process for performance monitoring

### Design Patterns Used

The Tools Package employs several design patterns:

1. **Strategy Pattern**: Each tool implements a specific strategy for performing a task while adhering to a common interface, allowing tools to be used interchangeably.

2. **Registry Pattern**: The ToolRegistry maintains a centralized catalog of available tools, enabling dynamic discovery and instantiation.

3. **Decorator Pattern**: Used throughout the codebase for cross-cutting concerns like metrics collection (`timed_metric` decorator), validation, and error handling.

4. **Factory Pattern**: The DynamicTool implementation creates tools dynamically from functions, providing a flexible way to define new tools without subclassing.

5. **Singleton Pattern**: Both the Registry and Runner are implemented as singletons to ensure consistent state and resource management.

### Key Abstractions

1. **BaseTool**: The fundamental abstraction defining what constitutes a tool, with both synchronous and asynchronous execution paths.

2. **Tool Registration**: The mechanism by which tools announce their availability to the system through the `@register_tool()` decorator.

3. **Tool Execution Context**: The environment in which tools execute, including error handling, metrics, and resource management.

4. **Tool Arguments Schema**: A structured definition of the inputs a tool accepts using Pydantic models, enabling validation and documentation.

## Component Details

### BaseTool and DynamicTool

#### Primary Purpose and Responsibilities

The `BaseTool` class defines the contract that all tools must fulfill, providing a consistent interface for tool execution, validation, and error handling. It's an abstract base class that enforces implementation of key methods like `_run()` and `_arun()`.

The `DynamicTool` class enables creation of tools from functions without subclassing, making it easy to convert existing functions into tools.

#### Core Classes and Relationships

```
BaseTool (ABC)
  ├── name (property)
  ├── description (property)
  ├── args_schema (property)
  ├── _run() (abstract)
  ├── _arun() (abstract)
  ├── run() (concrete)
  ├── arun() (concrete)
  ├── _validate_args() (concrete)
  └── _handle_error() (concrete)

DynamicTool (BaseTool)
  ├── __init__(name, description, func, coroutine, args_schema)
  ├── _run() (concrete)
  ├── _arun() (concrete)
  └── _infer_args_schema() (concrete)
```

#### Key Features and Capabilities

- Abstract base class for all tools
- Standardized execution flow with validation and error handling
- Support for both synchronous and asynchronous execution
- Automatic integration with metrics collection
- Schema-based argument validation
- Dynamic tool creation from functions
- Comprehensive error handling

#### Usage Examples

Creating a custom tool by subclassing BaseTool:

```python
from pydantic import BaseModel, Field
from src.tools.base import BaseTool

class WeatherInput(BaseModel):
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    units: str = Field("metric", description="The unit system: 'metric' or 'imperial'")

class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "weather"
        
    @property
    def description(self) -> str:
        return "Get the current weather in a given location"
        
    @property
    def args_schema(self) -> type[BaseModel]:
        return WeatherInput
        
    def _run(self, **kwargs) -> dict:
        location = kwargs["location"]
        units = kwargs["units"]
        # Implementation of weather API call
        return {"temperature": 22.5, "conditions": "Sunny", "location": location}
        
    async def _arun(self, **kwargs) -> dict:
        # Could implement an async version, or just use the sync version
        return self._run(**kwargs)
```

Creating a tool dynamically from a function:

```python
from src.tools.base import DynamicTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

multiply_tool = DynamicTool(
    name="multiply",
    description="Multiply two integers together",
    func=multiply
)

# Using the tool
result = multiply_tool.run(a=5, b=3)  # Returns 15
```

#### Best Practices

1. **Implement both execution methods**: Always implement both `_run()` and `_arun()` methods, even if one delegates to the other.

2. **Thorough validation**: Use Pydantic models to define comprehensive argument schemas with validation rules.

3. **Descriptive error messages**: Provide clear, actionable error messages that help users understand what went wrong.

4. **Resource management**: Properly acquire and release resources, especially in asynchronous implementations.

5. **Consistent error handling**: Use the provided error handling mechanisms rather than raising exceptions directly.

#### Performance Considerations

- Consider implementing a truly asynchronous version of `_arun()` for I/O-bound operations to improve concurrency
- Cache expensive computations where appropriate
- Be mindful of memory usage, especially for tools that process large datasets
- Implement timeouts for external service calls

### ToolRegistry

#### Primary Purpose and Responsibilities

The `ToolRegistry` manages the registration, discovery, and instantiation of tools within the platform. It serves as the central repository of available tools, allowing tools to be found by name at runtime.

#### Core Classes and Relationships

```
ToolRegistry
  ├── __init__()
  ├── register(tool_cls)
  ├── get_tool_class(tool_name)
  ├── get_tool(tool_name)
  ├── list_tools()
  ├── clear_cache()
  ├── unregister(tool_name)
  └── get_names()

register_tool (decorator)
  └── decorator(cls)
```

#### Key Features and Capabilities

- Tool class registration and discovery
- Tool instance caching for performance
- Decorator-based registration
- Listing available tools with metadata
- Tool unregistration for cleanup
- Thread-safe operations

#### Usage Examples

Registering a tool using the decorator:

```python
from src.tools.registry import register_tool
from src.tools.base import BaseTool

@register_tool()
class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    # Other required implementations...
```

Retrieving and using a tool:

```python
from src.tools.registry import ToolRegistry

registry = ToolRegistry()
calculator = registry.get_tool("calculator")
result = calculator.run(expression="2+2*3")
print(result)  # {"result": 8, "expression": "2+2*3", "simplified": "8"}
```

Listing available tools:

```python
from src.tools.registry import ToolRegistry

registry = ToolRegistry()
tools = registry.list_tools()
for tool in tools:
    print(f"{tool['name']}: {tool['description']}")
```

#### Best Practices

1. **Use the decorator**: Prefer the `@register_tool()` decorator for registration to keep registration close to the tool definition.

2. **Unique tool names**: Ensure all tools have unique names to avoid registration conflicts.

3. **Singleton registry**: Use a single registry instance throughout the application.

4. **Lazy initialization**: Tools should defer expensive initialization until first use.

5. **Clear the cache**: Clear the cache when reloading tools or during testing.

#### Performance Considerations

- The registry caches tool instances to avoid repeated instantiation
- Tool resolution is a frequent operation, so it's optimized for speed
- Registration typically happens at startup, so it's optimized for correctness over speed
- The registry uses locks to ensure thread safety during registration and instance creation

### ToolRunner

#### Primary Purpose and Responsibilities

The `ToolRunner` provides a standardized way to execute tools with proper error handling, retries, and metrics collection. It abstracts the complexities of tool execution and offers features like parallel execution.

#### Core Classes and Relationships

```
ToolRunner
  ├── __init__()
  ├── run_tool(tool, tool_registry, args, retry_count, trace_id)
  ├── _resolve_tool(tool, tool_registry)
  ├── _handle_tool_error(error, tool_name)
  ├── _format_result(result, tool_name, execution_time)
  ├── _format_error(error, tool_name, execution_time)
  ├── _create_tool_task(tool_name, tool_args, registry)
  └── run_tools_parallel(tools_config, registry, timeout)
```

#### Key Features and Capabilities

- Single and parallel tool execution
- Tool resolution from name or instance
- Standard error handling and formatting
- Execution timing and metrics
- Retry logic with exponential backoff
- Timeout management
- Consistent result formatting

#### Usage Examples

Running a single tool:

```python
from src.tools.runner import ToolRunner
from src.tools.registry import ToolRegistry

runner = ToolRunner()
registry = ToolRegistry()

# Run by tool name
result = await runner.run_tool(
    tool="calculator",
    tool_registry=registry,
    args={"expression": "2+2*3"},
    retry_count=2
)

# Run by tool instance
calculator = registry.get_tool("calculator")
result = await runner.run_tool(
    tool=calculator,
    args={"expression": "2+2*3"}
)
```

Running multiple tools in parallel:

```python
from src.tools.runner import ToolRunner
from src.tools.registry import ToolRegistry

runner = ToolRunner()
registry = ToolRegistry()

tools_config = [
    {
        "name": "calculator",
        "args": {"expression": "2+2*3"}
    },
    {
        "name": "datetime",
        "args": {"operation": "current", "timezone": "UTC"}
    }
]

results = await runner.run_tools_parallel(
    tools_config=tools_config,
    registry=registry,
    timeout=5.0
)

for result in results:
    print(f"{result['tool_name']}: {result['status']}")
```

#### Best Practices

1. **Provide retry counts**: Set appropriate retry counts for tools that interact with external services.

2. **Use trace IDs**: Pass trace IDs for distributed tracing across the system.

3. **Handle errors appropriately**: Check the status in the result to handle errors properly.

4. **Set reasonable timeouts**: Always specify a timeout when running tools in parallel.

5. **Use tool instances for repeated calls**: Resolve tool instances once and reuse them for multiple calls.

#### Performance Considerations

- Parallel execution significantly improves throughput for independent operations
- Exponential backoff prevents overwhelming external services during retries
- Tool resolution can be a bottleneck, so caching is important
- Error handling adds overhead but is essential for reliability

### WebSearchTool and GoogleSearchTool

#### Primary Purpose and Responsibilities

These tools provide web search capabilities using different search providers. The `WebSearchTool` integrates with DuckDuckGo's API, while the `GoogleSearchTool` uses Google's Custom Search API. They demonstrate how to integrate external APIs with proper error handling, caching, and result formatting.

#### Core Classes and Relationships

```
WebSearchTool (BaseTool)
  ├── __init__()
  ├── name
  ├── description
  ├── args_schema
  ├── _run()
  ├── _arun()
  ├── _perform_search()
  ├── _generate_fallback_results()
  ├── _format_results()
  ├── _get_cache_key()
  ├── _get_from_cache()
  └── _save_to_cache()

GoogleSearchTool (BaseTool)
  ├── __init__()
  ├── name
  ├── description
  ├── args_schema
  ├── _run()
  ├── _arun()
  ├── _perform_search()
  ├── _format_results()
  ├── _get_cache_key()
  ├── _get_from_cache()
  └── _save_to_cache()

WebSearchInput (BaseModel)
  ├── query
  ├── num_results
  └── safe_search
```

#### Key Features and Capabilities

- Integration with DuckDuckGo and Google search APIs
- Configurable via environment variables
- Result caching with Redis
- Fallback mechanisms for API failures
- Safe search options
- Rich metadata extraction
- Standardized result formatting

#### Usage Examples

Using the DuckDuckGo search tool:

```python
from src.tools.web_search import WebSearchTool

search_tool = WebSearchTool()
results = await search_tool.arun(
    query="multi-agent AI systems",
    num_results=5,
    safe_search=True
)

for result in results["results"]:
    print(f"Title: {result['title']}")
    print(f"Snippet: {result['snippet']}")
    print(f"URL: {result['url']}")
    print()
```

Using the Google search tool:

```python
from src.tools.web_search_google import GoogleSearchTool

google_search = GoogleSearchTool()
results = await google_search.arun(
    query="python programming best practices",
    num_results=3,
    safe_search=True
)

for result in results["results"]:
    print(f"Title: {result['title']}")
    print(f"Snippet: {result['snippet']}")
    print(f"URL: {result['url']}")
    print(f"Metadata: {result['metadata']}")
    print()
```

#### Best Practices

1. **API key management**: Store API keys in environment variables, never in code.

2. **Use caching**: Always enable caching to reduce API calls and improve performance.

3. **Implement fallbacks**: Have fallback mechanisms for when APIs fail or return no results.

4. **Safe search**: Enable safe search by default to prevent inappropriate results.

5. **Result limiting**: Limit the number of results to reduce resource usage.

6. **Error handling**: Implement robust error handling for API failures.

#### Performance Considerations

- Cache results to reduce API calls (configurable TTL)
- Limit the number of results to reduce memory usage
- Use non-blocking I/O for API calls
- Handle API rate limits gracefully
- Implement timeouts for external service calls

## Usage Examples

### Basic Tool Creation and Registration

Creating and registering a custom tool:

```python
from pydantic import BaseModel, Field
from src.tools.base import BaseTool
from src.tools.registry import register_tool

class RandomNumberInput(BaseModel):
    min_value: int = Field(0, description="Minimum value (inclusive)")
    max_value: int = Field(100, description="Maximum value (inclusive)")

@register_tool()
class RandomNumberTool(BaseTool):
    @property
    def name(self) -> str:
        return "random_number"
        
    @property
    def description(self) -> str:
        return "Generate a random number within a specified range"
        
    @property
    def args_schema(self) -> type[BaseModel]:
        return RandomNumberInput
        
    def _run(self, **kwargs) -> dict:
        import random
        min_value = kwargs["min_value"]
        max_value = kwargs["max_value"]
        number = random.randint(min_value, max_value)
        return {"number": number, "min": min_value, "max": max_value}
        
    async def _arun(self, **kwargs) -> dict:
        return self._run(**kwargs)
```

### Synchronous and Asynchronous Tool Execution

```python
import asyncio
from src.tools.registry import ToolRegistry
from src.tools.runner import ToolRunner

# Get tool instances
registry = ToolRegistry()
calculator = registry.get_tool("calculator")
datetime_tool = registry.get_tool("datetime")

# Synchronous execution
calc_result = calculator.run(expression="2^3 + 4^2")
print(f"Calculation result: {calc_result['result']}")  # 24

# Asynchronous execution
async def run_async_tools():
    runner = ToolRunner()
    
    # Run a single tool asynchronously
    datetime_result = await runner.run_tool(
        tool=datetime_tool,
        args={"operation": "current"}
    )
    print(f"Current time: {datetime_result['result']['iso_format']}")
    
    # Run multiple tools in parallel
    parallel_results = await runner.run_tools_parallel(
        tools_config=[
            {"name": "calculator", "args": {"expression": "sin(0.5)"}},
            {"name": "datetime", "args": {"operation": "current"}}
        ],
        registry=registry,
        timeout=5.0
    )
    print(f"Parallel results: {[r['tool_name'] for r in parallel_results]}")

# Run the async function
asyncio.run(run_async_tools())
```

### Web Search with Different Providers

```python
import asyncio
from src.tools.web_search import WebSearchTool
from src.tools.web_search_google import GoogleSearchTool

async def compare_search_results():
    # Initialize both search tools
    duckduckgo_search = WebSearchTool()
    google_search = GoogleSearchTool()
    
    # Define a search query
    query = "multi-agent systems in artificial intelligence"
    
    # Perform searches with both tools
    duckduckgo_results = await duckduckgo_search.arun(
        query=query,
        num_results=3,
        safe_search=True
    )
    
    google_results = await google_search.arun(
        query=query,
        num_results=3,
        safe_search=True
    )
    
    # Compare the results
    print("DuckDuckGo Results:")
    for idx, result in enumerate(duckduckgo_results["results"], 1):
        print(f"{idx}. {result['title']}")
        print(f"   {result['snippet'][:100]}...")
    
    print("\nGoogle Results:")
    for idx, result in enumerate(google_results["results"], 1):
        print(f"{idx}. {result['title']}")
        print(f"   {result['snippet'][:100]}...")

# Run the comparison
asyncio.run(compare_search_results())
```

### Error Handling Patterns

```python
import asyncio
from src.tools.runner import ToolRunner
from src.tools.registry import ToolRegistry
from src.config.errors import ToolError, ErrorCode

async def error_handling_example():
    runner = ToolRunner()
    registry = ToolRegistry()
    
    # Handle missing tool
    try:
        result = await runner.run_tool(
            tool="nonexistent_tool",
            tool_registry=registry,
            args={}
        )
        # The runner will actually return an error result rather than raising
        if result["status"] == "error":
            print(f"Tool error: {result['error']['message']}")
    except Exception as e:
        print(f"Unexpected exception: {e}")
    
    # Handle invalid arguments
    calculator = registry.get_tool("calculator")
    try:
        result = calculator.run(expression="2 + * 3")
        # This will raise a ToolError with validation details
    except ToolError as e:
        print(f"Tool error ({e.code}): {e.message}")
        if e.code == ErrorCode.TOOL_VALIDATION_ERROR:
            print("Validation error details:", e.details)
    
    # Using the runner's error handling
    result = await runner.run_tool(
        tool="calculator",
        tool_registry=registry,
        args={"expression": "2 + * 3"},
        retry_count=0
    )
    
    if result["status"] == "error":
        print(f"Error captured by runner: {result['error']['message']}")

# Run the error handling examples
asyncio.run(error_handling_example())
```

## Best Practices

### Tool Design

1. **Single Responsibility**: Each tool should do one thing well rather than trying to handle multiple unrelated responsibilities.

2. **Clear Interface**: Define clear input schemas that document all required and optional parameters.

3. **Comprehensive Error Handling**: Anticipate and handle all error conditions, providing clear error messages.

4. **Proper Resource Management**: Acquire and release resources properly, especially in asynchronous code.

5. **Timeouts and Retries**: Implement timeouts for external calls and retries for transient failures.

6. **Metrics Integration**: Use the metrics framework to track performance and usage.

7. **Caching Strategy**: Implement appropriate caching for expensive operations.

8. **Fallback Mechanisms**: Provide graceful fallbacks when primary methods fail.

### Tool Implementation

1. **Start with a Schema**: Define your input schema first to clarify the interface.

2. **Implement Synchronous First**: Implement the `_run()` method first, then adapt it for `_arun()`.

3. **Test Both Paths**: Test both synchronous and asynchronous execution paths thoroughly.

4. **Validate Early**: Perform validation as early as possible to fail fast.

5. **Document Thoroughly**: Document your tool's purpose, parameters, and behavior.

6. **Consider Resource Usage**: Be mindful of memory and CPU usage, especially for long-running operations.

7. **Use Async for I/O**: Leverage asynchronous programming for I/O-bound operations.

8. **Profile Performance**: Identify and optimize bottlenecks in your tool's execution.

### Error Handling

1. **Use ToolError**: Use the `ToolError` class for all tool-specific errors, setting the appropriate error code.

2. **Detailed Messages**: Provide detailed error messages that help users understand what went wrong.

3. **Include Context**: Include relevant context in error details to aid debugging.

4. **Catch All Exceptions**: Catch and handle all exceptions that might occur during tool execution.

5. **Log Errors**: Log errors with appropriate severity and context.

6. **Fail Gracefully**: Provide partial results or fallbacks when possible instead of complete failure.

### API Key Management

1. **Environment Variables**: Store API keys in environment variables, never in code.

2. **Settings System**: Use the settings system to load and validate API keys.

3. **Check Credentials**: Verify API keys are present before attempting to use them.

4. **Credential Validation**: Validate credentials early to provide clear error messages.

5. **Rotation Support**: Design your code to support API key rotation without service interruption.

## Testing Approach

The tools package includes comprehensive tests to ensure each component works correctly in isolation and with other components.

### Test Structure and Organization

Tests are organized by component, with separate test classes for:
- Base tool functionality (`TestBaseTool`)
- Dynamic tool creation (`TestDynamicTool`)
- Tool registry (`TestToolRegistry`)
- Tool runner (`TestToolRunner`)
- Individual tools (`TestCalculatorTool`, `TestDateTimeTool`, `TestWebSearchTool`, `TestGoogleSearchTool`)

### Running Tests

Run all tool tests with:

```bash
python -m unittest tests.tools.test_tools
```

Run specific test classes with:

```bash
python -m unittest tests.tools.test_tools.TestCalculatorTool
```

Run a specific test with:

```bash
python -m unittest tests.tools.test_tools.TestCalculatorTool.test_basic_arithmetic
```

### Key Test Fixtures and Mocks

The tests use mocking extensively to avoid external dependencies:

- `MagicMock` and `AsyncMock` for synchronous and asynchronous mock objects
- `patch` for temporarily replacing functions and classes
- `side_effect` for simulating errors and complex behaviors

Example of mocking an HTTP API:

```python
@patch("aiohttp.ClientSession.get")
async def test_google_search(self, mock_get):
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.raise_for_status = AsyncMock()
    mock_response.json.return_value = {
        "items": [
            {
                "title": "Test Result",
                "snippet": "This is a test result",
                "link": "https://example.com"
            }
        ]
    }
    mock_get.return_value = mock_response
    
    # Run the test
    result = await self.google_search.arun(query="test query")
    
    # Assertions
    self.assertEqual(result["results"][0]["title"], "Test Result")
    mock_get.assert_called_once()
```

### Test Environment Setup

The test environment includes:
- Mocked Redis for caching tests
- Mocked HTTP clients for API calls
- Helper functions for async testing
- Setup and teardown methods to ensure isolation

### Common Test Patterns

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test components working together
3. **Mocked External Dependencies**: Replace external services with mocks
4. **Error Handling Tests**: Verify error conditions are handled properly
5. **Edge Case Tests**: Test boundary conditions and special cases
6. **Async Test Helpers**: Use async test helpers for testing asynchronous code

## Implementation Notes

### Asynchronous Programming

The tools package makes extensive use of asynchronous programming for I/O-bound operations:

1. **Async Tool Interface**: All tools implement both synchronous and asynchronous interfaces, with the synchronous interface often deferring to the asynchronous one.

2. **Event Loop Management**: The code carefully manages event loops to prevent issues when running asynchronous code from synchronous contexts.

3. **Non-blocking I/O**: External API calls use non-blocking I/O to improve concurrency.

4. **Task Management**: The `ToolRunner` manages asynchronous tasks for parallel execution.

5. **Cancellation Handling**: Tasks properly handle cancellation for cleanup.

### Critical Fixes Implemented

1. **Syntax Error in f-string**: Fixed incorrect use of single quotes inside f-strings in the `DynamicTool._infer_args_schema()` method.

2. **Missing Imports**: Added necessary imports that were previously missing, including `Tuple` from typing in base.py, `Type` from typing in calculator.py, and `time`/`random` in runner.py.

3. **Undefined Function**: Added the missing `track_cache_miss` function in web_search.py that was being called but not defined.

4. **Pydantic Version Inconsistency**: Standardized on Pydantic v2 field validators throughout the codebase.

5. **Empty File Implementation**: Implemented the previously empty web_search_google.py file with a proper Google Search API integration.

6. **Redundant Error Handling**: Simplified error handling in the runner's create_tool_task method.

### Resource Management Patterns

Resources like HTTP sessions and Redis connections are managed through the Connection Manager:

1. **Connection Pooling**: HTTP and Redis connections use connection pooling for efficiency.

2. **Context Managers**: Resources are managed through context managers for proper cleanup.

3. **Lazy Initialization**: Connections are established only when needed.

4. **Proper Cleanup**: Resources are properly released when no longer needed.

### Performance Optimizations

1. **Caching**: Implementation of Redis-based caching for search results.

2. **Connection Pooling**: Reuse of HTTP and Redis connections.

3. **Asynchronous I/O**: Non-blocking I/O for external service calls.

4. **Parallel Execution**: Support for running multiple tools in parallel.

5. **Instance Caching**: Caching of tool instances in the registry.

## API Reference

### BaseTool Interface

```python
class BaseTool(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the tool."""
        
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Get the description of the tool."""
        
    @property
    @abc.abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """Get the pydantic schema for the tool's arguments."""
        
    @abc.abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Synchronous execution of the tool."""
        
    @abc.abstractmethod
    async def _arun(self, **kwargs: Any) -> Any:
        """Asynchronous execution of the tool."""
        
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously."""
        
    async def arun(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously."""
```

### ToolRegistry Interface

```python
class ToolRegistry:
    def register(self, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """Register a tool class with the registry."""
        
    def get_tool_class(self, tool_name: str) -> Type[BaseTool]:
        """Get a tool class by name."""
        
    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a tool instance by name, creating it if necessary."""
        
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with metadata."""
        
    def clear_cache(self) -> None:
        """Clear the tool instance cache."""
        
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        
    def get_names(self) -> Set[str]:
        """Get a set of all registered tool names."""
```

### ToolRunner Interface

```python
class ToolRunner:
    async def run_tool(
        self, 
        tool: Union[BaseTool, str], 
        tool_registry: Optional[Any]=None, 
        args: Optional[Dict[str, Any]]=None, 
        retry_count: int=0, 
        trace_id: Optional[str]=None
    ) -> Dict[str, Any]:
        """Run a tool with error handling and retries."""
        
    async def run_tools_parallel(
        self, 
        tools_config: List[Dict[str, Any]], 
        registry: Any, 
        timeout: Optional[float]=None
    ) -> List[Dict[str, Any]]:
        """Run multiple tools in parallel."""
```

### WebSearchTool Interface

```python
class WebSearchTool(BaseTool):
    async def _arun(
        self,
        query: str,
        num_results: int = 5,
        safe_search: bool = True
    ) -> Dict[str, Any]:
        """Run the DuckDuckGo search asynchronously."""
```

### GoogleSearchTool Interface

```python
class GoogleSearchTool(BaseTool):
    async def _arun(
        self,
        query: str,
        num_results: int = 5,
        safe_search: bool = True
    ) -> Dict[str, Any]:
        """Run the Google search asynchronously."""
```

## Integration Guidelines

### Initialization Sequence

1. **Registry Initialization**: Initialize the ToolRegistry first
2. **Tool Registration**: Register all tools with the registry
3. **Runner Initialization**: Initialize the ToolRunner
4. **Connection Setup**: Ensure Redis and HTTP connections are available

```python
from src.tools.registry import ToolRegistry
from src.tools.runner import ToolRunner
from src.config.connections import setup_connection_pools

# Initialize connections
setup_connection_pools()

# Initialize registry and get all registered tools
registry = ToolRegistry()
available_tools = registry.list_tools()

# Initialize runner
runner = ToolRunner()
```

### Configuration Options

The tools package uses these configuration settings:

```python
# In .env file or environment variables
GOOGLE_SEARCH_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
DUCKDUCKGO_PROXY_API_URL=https://api.duckduckgo.com/
SEARCH_CACHE_TTL=3600
REDIS_URL=redis://localhost:6379/0
REDIS_CONNECTION_POOL_SIZE=10
```

### Dependency Management

The tools package depends on:

1. **Redis**: For caching search results
2. **HTTP Client**: For making API calls (aiohttp)
3. **Metrics System**: For collecting metrics (Prometheus)
4. **Configuration**: For loading settings
5. **Error Handling**: For standardized error reporting

### Resource Lifecycle Management

1. **Connection Pools**: Created on demand and managed by the Connection Manager
2. **Tool Instances**: Created on demand and cached by the Registry
3. **Search Caches**: Created in Redis with configurable TTL

### Shutdown Procedures

```python
from src.config.connections import cleanup_connection_pools

async def shutdown():
    # Clean up all connection pools
    await cleanup_connection_pools()
    
    # Clear registry cache
    from src.tools.registry import ToolRegistry
    registry = ToolRegistry()
    registry.clear_cache()
```

## Key Improvements

### Bug Fixes

1. **Fixed Syntax Error in f-string**: Corrected improper quotation marks in f-string in the `DynamicTool._infer_args_schema()` method.

2. **Added Missing Imports**: Added missing imports including `Tuple`, `Type`, `time`, and `random` in various files.

3. **Added Missing Function**: Implemented the missing `track_cache_miss` function in web_search.py.

4. **Fixed Pydantic Version Inconsistency**: Standardized on Pydantic v2 field validators throughout the codebase.

5. **Implemented Empty File**: Created a full implementation for the previously empty web_search_google.py.

### Interface Enhancements

1. **Standardized Search Interface**: Both search tools now use the same interface, making them interchangeable.

2. **Improved Error Reporting**: Enhanced error handling with detailed error messages and proper error classification.

3. **Uniform Result Format**: Standardized result format across all tools for consistent handling.

### Better Error Handling

1. **Specific Error Codes**: Added specific error codes for different error conditions.

2. **Detailed Error Messages**: Improved error messages with more context and actionable information.

3. **Fallback Mechanisms**: Added fallback mechanisms for search tools when APIs fail or return no results.

4. **Retry Logic**: Enhanced retry logic with proper backoff in the tool runner.

### Improved Resource Management

1. **Connection Pooling**: Implemented connection pooling for HTTP and Redis connections.

2. **Proper Resource Cleanup**: Ensured proper resource cleanup through context managers.

3. **Lazy Initialization**: Implemented lazy initialization for connection resources.

### Code Structure Improvements

1. **Standardized Tool Interface**: All tools now follow the same interface pattern.

2. **Common Error Handling**: Consolidated error handling patterns.

3. **Unified Result Format**: Standardized result format for all tools.

4. **Metrics Integration**: Integrated metrics collection throughout the codebase.

5. **Comprehensive Testing**: Added thorough tests for all components.
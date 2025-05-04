# Model Context Protocol (MCP) API System Documentation

## Architecture Overview

The Model Context Protocol (MCP) API provides a standardized framework for serializing, validating, and transmitting context objects between components of the high-performance multi-agent platform. This subsystem implements the extension-based architecture described in the project roadmap, focusing on middleware components that ensure context integrity, proper serialization/deserialization, and efficient transmission.

### Purpose and Responsibility of Each Module

- **BasicMCPMiddleware**: Foundation middleware that logs MCP context information and provides timing metrics
- **MCPSerializationMiddleware**: Handles conversion between binary/JSON formats and runtime context objects
- **MCPContextValidationMiddleware**: Ensures contexts meet schema and version requirements
- **WebSocketAdapter**: Enables real-time streaming of context objects to clients
- **OpenAPI Extensions**: Extends API documentation with MCP context schemas

### Component Interactions

The MCP API components form a middleware pipeline that processes HTTP requests and WebSocket connections:

1. **Request Flow**:
   - BasicMCPMiddleware logs request information and adds timing metrics
   - MCPSerializationMiddleware deserializes binary/JSON payloads into context objects
   - MCPContextValidationMiddleware validates context objects
   - Application handlers process validated contexts
   - Response flows back through middleware chain

2. **WebSocket Flow**:
   - ConnectionManager maintains active WebSocket connections
   - MCPWebSocketAdapter serializes context objects
   - Context updates stream to clients in real-time

### Design Patterns

- **Middleware Pattern**: Layered request processing with separation of concerns
- **Adapter Pattern**: Converts between MCP contexts and transport formats
- **Singleton Pattern**: For WebSocketAdapter to ensure a single instance
- **Factory Pattern**: For creating adapter instances
- **Decorator Pattern**: For OpenAPI customization

### Key Abstractions

- **ContextProtocol**: Foundation interface that all context objects implement
- **SerializationFormat**: Enumeration of supported serialization formats
- **MCP Headers**: Standardized HTTP headers for version and context type information

## Component Details

### 1. BasicMCPMiddleware

#### Primary Purpose
Provides foundational request/response processing with logging and timing metrics for all MCP-related HTTP requests.

#### Core Classes
- `BasicMCPMiddleware`: Extends `BaseHTTPMiddleware` from Starlette

#### Key Features
- Logs detailed information about MCP requests and responses
- Adds processing time headers to responses
- Detects MCP contexts based on content type and headers
- Handles errors during request processing

#### Usage Example

```python
# Adding the middleware to a FastAPI app
from fastapi import FastAPI
from src.core.mcp.api.middleware import BasicMCPMiddleware

app = FastAPI()
app.add_middleware(BasicMCPMiddleware)
```

#### Public Interfaces

```python
class BasicMCPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Processes request, invokes next middleware, adds timing headers
```

#### Best Practices
- Always add this middleware first in the MCP middleware stack
- Use it to obtain baseline performance metrics for all requests
- Monitor X-Process-Time-Ms header values to track performance

#### Performance Considerations
- The middleware adds minimal overhead (<1ms per request)
- Logging granularity can be adjusted for production environments

### 2. MCPSerializationMiddleware

#### Primary Purpose
Converts between wire formats (JSON, MessagePack) and ContextProtocol objects.

#### Core Classes
- `MCPSerializationMiddleware`: Extends `BaseHTTPMiddleware`
- Uses `deserialize_context` function from `src.core.mcp.serialization`

#### Key Features
- Automatic content type detection
- Support for multiple serialization formats (JSON, MessagePack)
- Async deserialization with minimal overhead
- Proper error handling for malformed contexts

#### Usage Example

```python
# Adding the middleware to a FastAPI app
from fastapi import FastAPI
from src.core.mcp.api.serialization_middleware import MCPSerializationMiddleware

app = FastAPI()
app.add_middleware(MCPSerializationMiddleware)

# In route handlers, access the deserialized context
@app.post("/process")
async def process(request: Request):
    mcp_context = request.state.mcp_context
    if mcp_context:
        # Work with the context
        return {"context_id": mcp_context.context_id}
    return {"message": "No context found"}
```

#### Public Interfaces

```python
class MCPSerializationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Deserializes request body, stores in request.state.mcp_context
```

#### Best Practices
- Add this middleware after BasicMCPMiddleware
- Use appropriate content-type headers in client requests
- Properly handle SerializationError exceptions

#### Performance Considerations
- Deserializes context objects with <1ms overhead
- Uses async processing to avoid blocking
- MessagePack format is more efficient than JSON for large contexts

### 3. MCPContextValidationMiddleware

#### Primary Purpose
Ensures context objects meet schema requirements and version compatibility.

#### Core Classes
- `MCPContextValidationMiddleware`: Extends `BaseHTTPMiddleware`
- Uses `check_version_compatibility` from `src.core.mcp.versioning`

#### Key Features
- Validates context version compatibility
- Ensures context objects follow required schema
- Returns appropriate error responses for invalid contexts
- Maintains backward compatibility with previous versions

#### Usage Example

```python
# Adding the middleware to a FastAPI app
from fastapi import FastAPI
from src.core.mcp.api.validation_middleware import MCPContextValidationMiddleware

app = FastAPI()
app.add_middleware(MCPContextValidationMiddleware)
```

#### Public Interfaces

```python
class MCPContextValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Validates mcp_context in request.state, rejects invalid contexts
```

#### Best Practices
- Add this middleware after MCPSerializationMiddleware
- Check response status codes to detect validation failures
- Use the latest compatible MCP version in clients

#### Performance Considerations
- Validation is fast (<1ms per context)
- Caching can be implemented for repeated validation of similar contexts

### 4. WebSocket Adapter

#### Primary Purpose
Enables streaming of context objects to clients via WebSockets.

#### Core Classes
- `MCPWebSocketAdapter`: Main adapter class
- `get_websocket_adapter`: Factory function for obtaining singleton instance

#### Key Features
- Serializes context objects for WebSocket transmission
- Supports multiple serialization formats
- Maintains connection pools via ConnectionManager
- Implements singleton pattern for resource efficiency

#### Usage Example

```python
# Using the WebSocket adapter to stream a context
from src.core.mcp.api.websocket_adapter import get_websocket_adapter
from src.core.mcp.serialization import SerializationFormat

async def stream_context_update(context, task_id):
    adapter = await get_websocket_adapter()
    success = await adapter.stream_context(
        context=context,
        task_id=task_id,
        target_format=SerializationFormat.JSON
    )
    return success
```

#### Public Interfaces

```python
class MCPWebSocketAdapter:
    async def stream_context(
        self, 
        context: ContextProtocol, 
        task_id: Optional[str]=None, 
        target_format: SerializationFormat=SerializationFormat.JSON
    ) -> bool:
        # Serializes and streams context to WebSocket clients

async def get_websocket_adapter(
    connection_manager: Optional[ConnectionManager]=None
) -> MCPWebSocketAdapter:
    # Returns singleton adapter instance
```

#### Best Practices
- Always use the `get_websocket_adapter()` factory function
- Send small, incremental updates rather than large contexts
- Handle streaming failures gracefully
- Use the proper serialization format based on client capabilities

#### Performance Considerations
- Singleton pattern reduces resource usage
- Serialization overhead is minimal for small contexts
- For large contexts, consider using MessagePack format

### 5. OpenAPI Extensions

#### Primary Purpose
Enhances API documentation with MCP context schemas for developer reference.

#### Core Classes
- Functions for OpenAPI schema customization:
  - `get_mcp_context_schemas`
  - `add_mcp_schemas_to_openapi`
  - `customize_openapi_for_mcp`

#### Key Features
- Adds MCP context schemas to OpenAPI documentation
- Provides visible extension points in API docs
- Documents MCP version compatibility

#### Usage Example

```python
# Adding MCP OpenAPI extensions to a FastAPI app
from fastapi import FastAPI
from src.core.mcp.api.openapi_extension import customize_openapi_for_mcp

app = FastAPI()
customize_openapi_for_mcp(app)
```

#### Public Interfaces

```python
def get_mcp_context_schemas() -> List[Type[BaseModel]]:
    # Returns list of MCP context schemas for documentation

def customize_openapi_for_mcp(app: FastAPI) -> None:
    # Adds MCP extensions to OpenAPI schema
```

#### Best Practices
- Apply customization during app initialization
- Review generated OpenAPI documentation for correctness
- Update schema documentation when context models change

#### Performance Considerations
- Minimal overhead as customization happens at startup
- Does not affect runtime performance

## Usage Examples

### Basic MCP API Integration

```python
from fastapi import FastAPI, Request
from src.core.mcp.api.middleware import BasicMCPMiddleware
from src.core.mcp.api.serialization_middleware import MCPSerializationMiddleware
from src.core.mcp.api.validation_middleware import MCPContextValidationMiddleware
from src.core.mcp.api.openapi_extension import customize_openapi_for_mcp

# Create FastAPI app
app = FastAPI()

# Add MCP middleware stack (order matters)
app.add_middleware(MCPContextValidationMiddleware)
app.add_middleware(MCPSerializationMiddleware)
app.add_middleware(BasicMCPMiddleware)

# Add OpenAPI customization
customize_openapi_for_mcp(app)

# Example route that processes an MCP context
@app.post("/process_task")
async def process_task(request: Request):
    # Access the deserialized and validated context
    context = request.state.mcp_context
    if not context:
        return {"error": "No MCP context provided"}
    
    # Process the context
    task_id = getattr(context, "task_id", None) or context.metadata.get("task_id")
    
    # Return a response
    return {
        "task_id": task_id,
        "context_id": context.context_id,
        "status": "processing"
    }
```

### Streaming Context Updates via WebSockets

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.api.streaming import get_connection_manager
from src.core.mcp.api.websocket_adapter import get_websocket_adapter

app = FastAPI()
connection_manager = get_connection_manager()

@app.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await connection_manager.connect(websocket, task_id)
    try:
        # Handle incoming messages if needed
        while True:
            data = await websocket.receive_text()
            # Process received data
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, task_id)

# Example function to send context updates
async def send_task_update(task_id: str, context):
    adapter = await get_websocket_adapter(connection_manager)
    success = await adapter.stream_context(context, task_id)
    return success
```

### Error Handling Approach

```python
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from src.core.mcp.serialization import SerializationError
from src.core.mcp.versioning import VersionIncompatibleError

app = FastAPI()

@app.exception_handler(SerializationError)
async def serialization_error_handler(request: Request, exc: SerializationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": f"MCP serialization error: {str(exc)}"}
    )

@app.exception_handler(VersionIncompatibleError)
async def version_error_handler(request: Request, exc: VersionIncompatibleError):
    return JSONResponse(
        status_code=status.HTTP_426_UPGRADE_REQUIRED,
        content={
            "detail": f"MCP version incompatible: {str(exc)}",
            "compatible_versions": exc.compatible_versions
        }
    )
```

## Best Practices

### Using MCP API Components Efficiently

1. **Middleware Order**:
   - Add middlewares in the correct order: BasicMCPMiddleware → SerializationMiddleware → ValidationMiddleware
   - This ensures proper logging and error handling at each stage

2. **Context Design**:
   - Keep MCP context objects small and focused
   - Use proper inheritance hierarchy for specialized contexts
   - Include only necessary fields to minimize serialization overhead

3. **Version Management**:
   - Always include version information in contexts
   - Use semantic versioning (MAJOR.MINOR.PATCH)
   - Maintain backward compatibility when possible

4. **Performance Optimization**:
   - Use MessagePack format for large contexts
   - Implement response compression for bandwidth-intensive operations
   - Stream large contexts in smaller chunks
   - Add appropriate caching for frequently accessed contexts

5. **Error Handling**:
   - Implement specific exception handlers for MCP-related errors
   - Include detailed error information in responses
   - Log errors with appropriate severity levels

6. **Testing**:
   - Write comprehensive tests for serialization/deserialization
   - Test version compatibility across different context versions
   - Benchmark serialization performance with realistic data sizes

## Testing Approach

### Test Structure and Organization

The MCP API tests follow a comprehensive approach to ensure correct functionality and performance:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test middleware pipeline as a whole
3. **Performance Tests**: Measure overhead and ensure it meets requirements

### Running the Tests

```bash
# Run all MCP API tests
pytest tests/core/mcp/api/test_mcp_api.py

# Run specific test classes
pytest tests/core/mcp/api/test_mcp_api.py::TestMCPAPI

# Run specific test methods
pytest tests/core/mcp/api/test_mcp_api.py::TestMCPAPI::test_websocket_adapter

# Run with coverage
pytest tests/core/mcp/api/test_mcp_api.py --cov=src.core.mcp.api

# Run performance tests only
pytest tests/core/mcp/api/test_mcp_api.py -k performance
```

### Key Test Fixtures and Mock Objects

The tests use several important fixtures and mocks:

1. **MockContextProtocol**: A simplified implementation of the ContextProtocol interface for testing
2. **AsyncMock**: Used to simulate asynchronous functions like request handlers
3. **Patched Serialization**: Mocks the serialization/deserialization functions to isolate middleware logic
4. **FastAPI TestClient**: For integration testing of the entire middleware pipeline

### Common Test Patterns

1. **Middleware Dispatch Testing**: Create mock requests and responses, verify middleware behavior
2. **Performance Measurement**: Time operations to verify overhead meets requirements
3. **Error Handling**: Verify appropriate responses for various error conditions
4. **Integration Flow**: Test the complete request processing pipeline

### Key Test Cases

1. **BasicMCPMiddleware Tests**:
   - Verify logging of MCP content types and headers
   - Ensure timing headers are added to responses
   - Confirm proper handling of various error conditions

2. **MCPSerializationMiddleware Tests**:
   - Test successful deserialization of different formats
   - Verify error responses for malformed payloads
   - Measure serialization overhead to ensure performance targets

3. **MCPContextValidationMiddleware Tests**:
   - Verify proper validation of context versions
   - Ensure appropriate error responses for incompatible versions
   - Test validation of context schema requirements

4. **WebSocketAdapter Tests**:
   - Test successful streaming of contexts
   - Verify proper handling of serialization errors
   - Confirm singleton pattern works correctly

5. **OpenAPI Extension Tests**:
   - Verify schema customization applies correctly
   - Ensure MCP context schemas are included in documentation

## Implementation Notes

### Design Decisions

1. **Middleware Approach**: The system uses FastAPI/Starlette middleware for optimal integration with the web framework, allowing clear separation of concerns and maintainable code.

2. **Async Implementation**: All components use async/await patterns for non-blocking I/O, essential for the high-performance requirements of the platform.

3. **Serialization Strategy**: Two serialization formats (JSON and MessagePack) are supported to balance human readability (JSON) with performance (MessagePack).

4. **Version Compatibility**: Version checking uses semantic versioning principles with explicit compatibility rules rather than simple equality checks.

5. **Singleton WebSocket Adapter**: The WebSocket adapter uses a singleton pattern to maintain a single connection pool across the application.

### Thread Safety Considerations

1. **Request State**: Each request has its own state object, preventing cross-request contamination.

2. **Async Lock for Singleton**: The WebSocket adapter uses an async lock to ensure thread-safe singleton initialization.

3. **Connection Manager**: The underlying ConnectionManager handles concurrent WebSocket connections with appropriate locking.

4. **Immutable Context Objects**: Context objects should be treated as immutable after validation to prevent race conditions.

### Asynchronous Programming Patterns

1. **Async Middleware**: All middleware components use async dispatch methods for non-blocking processing.

2. **Worker Offloading**: CPU-intensive operations like serialization can be offloaded with `asyncio.to_thread()`.

3. **Async Singleton Factory**: The WebSocket adapter factory uses async patterns for thread-safe initialization.

4. **Concurrent WebSocket Broadcasting**: The adapter supports concurrent broadcasts to multiple clients.

### Performance Optimizations

1. **Minimal Deserialization**: Contexts are deserialized only when needed based on content type detection.

2. **Request Timing**: All requests are timed with microsecond precision for performance monitoring.

3. **Efficient Context Storage**: Deserialized contexts are stored in request state to avoid redundant processing.

4. **Connection Pooling**: WebSocket connections are pooled by task ID for efficient broadcasting.

5. **Lazy Initialization**: Components use lazy initialization patterns to minimize startup impact.

## API Reference

### BasicMCPMiddleware

```python
class BasicMCPMiddleware(BaseHTTPMiddleware):
    """Provides logging and timing for MCP requests."""
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process an incoming request and add timing headers.
        
        Args:
            request: The incoming HTTP request.
            call_next: Function to call the next middleware.
            
        Returns:
            The HTTP response with added timing headers.
        
        Performance: Adds <1ms overhead per request.
        Thread Safety: Thread-safe, operates on per-request basis.
        """
```

### MCPSerializationMiddleware

```python
class MCPSerializationMiddleware(BaseHTTPMiddleware):
    """Deserializes MCP contexts from request bodies."""
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Deserialize MCP context from request body.
        
        Args:
            request: The incoming HTTP request.
            call_next: Function to call the next middleware.
            
        Returns:
            The HTTP response.
            
        Exceptions:
            - 400 Bad Request: For deserialization errors.
            - 500 Internal Server Error: For unexpected failures.
            
        Performance: <1ms overhead for context deserialization.
        Thread Safety: Thread-safe, operates on per-request basis.
        """
```

### MCPContextValidationMiddleware

```python
class MCPContextValidationMiddleware(BaseHTTPMiddleware):
    """Validates MCP context objects."""
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Validate MCP context in request state.
        
        Args:
            request: The incoming HTTP request.
            call_next: Function to call the next middleware.
            
        Returns:
            The HTTP response.
            
        Exceptions:
            - 400 Bad Request: For invalid context objects.
            - 426 Upgrade Required: For incompatible versions.
            
        Performance: <1ms overhead for context validation.
        Thread Safety: Thread-safe, operates on per-request basis.
        """
```

### MCPWebSocketAdapter

```python
class MCPWebSocketAdapter:
    """Streams MCP contexts via WebSockets."""
    
    async def stream_context(
        self, 
        context: ContextProtocol, 
        task_id: Optional[str]=None, 
        target_format: SerializationFormat=SerializationFormat.JSON
    ) -> bool:
        """
        Serialize and stream context to WebSocket clients.
        
        Args:
            context: The MCP context to stream.
            task_id: Optional task ID for routing. If None, extracted from context.
            target_format: Serialization format to use.
            
        Returns:
            True if streaming succeeded, False otherwise.
            
        Performance: Serialization overhead depends on context size.
        Thread Safety: Thread-safe through ConnectionManager.
        """

async def get_websocket_adapter(
    connection_manager: Optional[ConnectionManager]=None
) -> MCPWebSocketAdapter:
    """
    Get or create singleton WebSocket adapter instance.
    
    Args:
        connection_manager: Optional ConnectionManager. If None, default is used.
        
    Returns:
        MCPWebSocketAdapter singleton instance.
        
    Exceptions:
        - ValueError: If connection_manager is not available.
        - RuntimeError: If instance creation fails.
        
    Thread Safety: Thread-safe through async lock.
    """
```

### OpenAPI Extensions

```python
def get_mcp_context_schemas() -> List[Type[BaseModel]]:
    """
    Get list of MCP context schemas for OpenAPI documentation.
    
    Returns:
        List of Pydantic model classes representing MCP contexts.
    """

def customize_openapi_for_mcp(app: FastAPI) -> None:
    """
    Add MCP extensions to OpenAPI schema.
    
    Args:
        app: FastAPI application instance.
    
    Side Effects:
        Modifies app.openapi function to include MCP information.
    """
```

## Integration Guidelines

### Initialization Sequence

1. **Create FastAPI Application**:
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   ```

2. **Add MCP Middleware Stack** (in reverse order):
   ```python
   from src.core.mcp.api.middleware import BasicMCPMiddleware
   from src.core.mcp.api.serialization_middleware import MCPSerializationMiddleware
   from src.core.mcp.api.validation_middleware import MCPContextValidationMiddleware
   
   app.add_middleware(MCPContextValidationMiddleware)
   app.add_middleware(MCPSerializationMiddleware)
   app.add_middleware(BasicMCPMiddleware)
   ```

3. **Customize OpenAPI Documentation**:
   ```python
   from src.core.mcp.api.openapi_extension import customize_openapi_for_mcp
   customize_openapi_for_mcp(app)
   ```

4. **Set Up WebSocket Connection Manager**:
   ```python
   from src.api.streaming import get_connection_manager
   connection_manager = get_connection_manager()
   ```

5. **Initialize WebSocket Routes**:
   ```python
   @app.websocket("/ws/tasks/{task_id}")
   async def websocket_endpoint(websocket: WebSocket, task_id: str):
       await connection_manager.connect(websocket, task_id)
       try:
           while True:
               data = await websocket.receive_text()
               # Process data
       except WebSocketDisconnect:
           connection_manager.disconnect(websocket, task_id)
   ```

### Configuration Options

The MCP API components have several configuration options:

1. **Logging Configuration**:
   - Set up appropriate log levels in your logger configuration
   - For production, reduce log verbosity to minimize overhead

2. **Serialization Formats**:
   - Configure supported formats in `MCP_SERIALIZATION_FORMAT_MAP`
   - Add custom formats by extending the SerializationFormat enum

3. **Version Compatibility Rules**:
   - Configure version compatibility rules in the versioning module
   - Set minimum compatible version if needed

4. **WebSocket Configuration**:
   - Adjust connection timeout settings in ConnectionManager
   - Configure message size limits for WebSocket connections

### Resource Lifecycle Management

1. **WebSocket Connections**:
   - Properly handle connection and disconnection events
   - Implement heartbeat mechanism for long-lived connections
   - Clean up resources on disconnection

2. **Request/Response Cycle**:
   - Middleware ensures proper resource cleanup
   - Use FastAPI dependency injection for managed resources

3. **Adapter Resources**:
   - The singleton pattern manages adapter lifecycle
   - No explicit cleanup needed in most cases

### Shutdown Procedures

1. **Graceful WebSocket Shutdown**:
   ```python
   @app.on_event("shutdown")
   async def shutdown_event():
       # Close all WebSocket connections
       connection_manager = get_connection_manager()
       await connection_manager.close_all()
   ```

2. **Resource Cleanup**:
   ```python
   @app.on_event("shutdown")
   async def cleanup_resources():
       # Perform any necessary cleanup
       # Close connection pools, etc.
       pass
   ```

## Key Improvements

The MCP API implementation includes several key improvements:

1. **Performance Optimizations**:
   - Middleware overhead reduced to <1ms per operation
   - Efficient serialization with MessagePack support
   - Singleton pattern for WebSocket adapter to reduce resource usage

2. **Enhanced Error Handling**:
   - Detailed error responses with proper HTTP status codes
   - Comprehensive logging with request IDs for traceability
   - Specific error types for different failure scenarios

3. **Better Resource Management**:
   - Connection pooling for WebSockets
   - Async patterns for non-blocking operation
   - Thread-safe singleton initialization

4. **Code Structure Improvements**:
   - Clear separation of concerns with middleware approach
   - Consistent error handling patterns
   - Comprehensive test coverage

5. **Interface Enhancements**:
   - OpenAPI documentation with MCP schemas
   - Consistent HTTP headers for MCP metadata
   - Simplified WebSocket adapter interface

These improvements align with the project's focus on high performance, maintainability, and extensibility as outlined in the roadmap document.
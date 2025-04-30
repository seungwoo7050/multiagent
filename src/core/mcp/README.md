# Multi-Agent Platform Model Context Protocol (MCP) Package

## Overview

The Model Context Protocol (MCP) package (`src/core/mcp/`) provides a standardized communication framework for the Multi-Agent Platform, enabling different components to exchange data through a common protocol. The MCP establishes a unified approach to context representation, serialization, optimization, and adaptation, supporting the high-performance requirements of the multi-agent system. This package serves as the communication backbone of the platform, allowing components to interact seamlessly regardless of their internal implementations.

The MCP is designed with performance, reliability, and extensibility in mind, offering optimized serialization formats, context compression, version management, and adapter-based component integration.

## Architecture

The MCP package consists of five core modules that work together to create a communication infrastructure:

1. **Protocol and Schema** (`protocol.py`, `schema.py`): Core abstractions and data models
2. **Serialization** (`serialization.py`): Binary conversion for transport and storage
3. **Compression** (`compression.py`): Size optimization and data reduction
4. **Versioning** (`versioning.py`): Version compatibility and schema migration
5. **Adapters** (`adapter_base.py`): Component integration and interface adaptation

These modules are designed to work together as a system while maintaining clear separations of concerns:

- The **Protocol and Schema** define what a context is
- The **Serialization** handles how contexts are converted to transportable formats
- The **Compression** optimizes the size of serialized contexts
- The **Versioning** manages compatibility between different schema iterations
- The **Adapters** connect non-MCP components to the MCP system

## Module Details

### Protocol and Schema (`protocol.py`, `schema.py`)

The protocol and schema modules define the core abstractions and data models for the MCP system.

#### Core Classes

- `ContextProtocol`: Abstract base class that defines the contract for all context objects
- `BaseContextSchema`: Concrete implementation of the protocol with common fields and behaviors
- `TaskContext`: Specialized context for representing tasks in the system

#### Key Features

- **Type-safe Context Objects**: Pydantic-based models with validation
- **Standardized Interface**: Common methods for all context objects
- **Serialization Contract**: Defined methods for serialization and deserialization
- **Metadata Support**: Flexible metadata for context-specific attributes
- **Optimization Capabilities**: Support for context optimization
- **JSON Conversion**: Utilities for JSON conversion

#### Usage Examples

```python
from src.core.mcp.schema import BaseContextSchema, TaskContext

# Create a basic context
context = BaseContextSchema(
    metadata={"source": "user_input", "priority": "high"}
)

# Access context properties
context_id = context.context_id
timestamp = context.timestamp
metadata = context.metadata

# Create a task context with more specific fields
task_context = TaskContext(
    task_id="calculation-123",
    task_type="math_operation",
    input_data={"operation": "add", "values": [1, 2, 3]},
    current_step="initialization"
)

# Serialize to a dictionary
context_dict = context.serialize()

# Deserialize from a dictionary
restored_context = BaseContextSchema.deserialize(context_dict)

# Convert to JSON
json_str = context.to_json()

# Create from JSON
from_json = BaseContextSchema.from_json(json_str)
```

### Serialization (`serialization.py`)

The serialization module handles the conversion between context objects and binary formats for storage or transmission.

#### Core Functions

- `serialize_context()`: Converts a context object to binary data
- `deserialize_context()`: Converts binary data back to a context object
- `_deserialize_with_target_class()`: Helper for typed deserialization
- `_infer_and_deserialize()`: Helper for automatic type inference

#### Key Features

- **Multiple Serialization Formats**: Support for MessagePack and JSON
- **Type Safety**: Type hints and validation throughout
- **Automatic Type Inference**: Ability to infer context type during deserialization
- **Error Handling**: Comprehensive error handling and reporting
- **Performance Optimization**: Optimized for fast serialization and deserialization
- **Metrics Integration**: Performance tracking for serialization operations

#### Usage Examples

```python
from src.core.mcp.serialization import serialize_context, deserialize_context
from src.core.mcp.schema import TaskContext
from src.utils.serialization import SerializationFormat

# Create a context
task_context = TaskContext(
    task_id="process-456",
    task_type="data_processing",
    input_data={"data": [1, 2, 3, 4, 5]}
)

# Serialize to binary (default is MessagePack)
binary_data = serialize_context(task_context)

# Serialize to JSON
json_binary = serialize_context(task_context, format=SerializationFormat.JSON)

# Deserialize with known type (preferred for performance)
restored = deserialize_context(binary_data, target_class=TaskContext)

# Deserialize with type inference (more flexible)
inferred = deserialize_context(binary_data)
```

### Compression (`compression.py`)

The compression module optimizes context data size through context-specific optimization and general-purpose compression.

#### Core Functions

- `optimize_context_data()`: Applies context-specific optimizations
- `compress_context()`: Compresses serialized context data
- `decompress_context()`: Decompresses compressed context data

#### Key Features

- **Two-Level Optimization**: Context-level and binary-level optimization
- **Adaptive Compression**: Different compression levels based on data size
- **Performance Metrics**: Tracking of compression ratios and times
- **Error Handling**: Robust error handling with fallbacks
- **Size Checks**: Only applies compression when beneficial
- **Logging**: Detailed logging of compression operations

#### Usage Examples

```python
from src.core.mcp.compression import optimize_context_data, compress_context, decompress_context
from src.core.mcp.schema import BaseContextSchema

# Create and optimize a context
context = BaseContextSchema(metadata={"large_data": "A" * 1000})
optimized = optimize_context_data(context)

# Serialize to a dictionary
context_dict = optimized.serialize()

# Compress the dictionary
compressed_data = compress_context(context_dict)

# Decompress the data
decompressed_dict = decompress_context(compressed_data)

# Verify data integrity
assert decompressed_dict["metadata"]["large_data"] == "A" * 1000
```

### Versioning (`versioning.py`)

The versioning module manages compatibility between different versions of context schemas to ensure smooth upgrades and backward compatibility.

#### Core Functions

- `check_version_compatibility()`: Verifies if a version is compatible
- `upgrade_context()`: Upgrades a context to a specified version
- `get_latest_supported_version()`: Returns the latest supported version

#### Key Features

- **Semantic Versioning**: Version compatibility based on semantic versioning
- **Version Compatibility Checking**: Automatic checking of version compatibility
- **Context Upgrading**: Support for upgrading contexts between versions
- **Explicit Upgrade Paths**: Defined paths for version transitions
- **Error Handling**: Clear error messages for incompatible versions

#### Usage Examples

```python
from src.core.mcp.versioning import check_version_compatibility, upgrade_context, get_latest_supported_version

# Check if a version is compatible
is_compatible = check_version_compatibility("1.0.0")

# Get the latest supported version
latest_version = get_latest_supported_version()

# Upgrade a context from an older version
old_context_data = {
    "version": "0.0.0",  # Unversioned context
    "data": {"value": 42}
}

# Upgrade to the latest version
upgraded = upgrade_context(old_context_data, latest_version)
```

### Adapters (`adapter_base.py`)

The adapter module provides a foundation for building adapters that connect non-MCP components to the MCP ecosystem, translating between component-specific interfaces and the MCP context protocol.

#### Core Classes

- `MCPAdapterBase`: Abstract base class for all MCP adapters

#### Key Features

- **Abstract Adapter Interface**: Common interface for all adapters
- **Component Method Detection**: Automatic detection of execute/process methods
- **Synchronous and Asynchronous Support**: Support for both types of components
- **Input/Output Adaptation**: Standard methods for transforming inputs and outputs
- **Comprehensive Error Handling**: Proper error propagation and logging
- **Performance Tracking**: Metrics for adapter operations

#### Usage Examples

```python
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.schema import BaseContextSchema
import asyncio

# Example component that isn't MCP-aware
class DataProcessor:
    async def execute(self, input_data):
        # Process the data
        result = [x * 2 for x in input_data]
        return {"processed_data": result, "count": len(result)}

# Custom adapter for the data processor
class DataProcessorAdapter(MCPAdapterBase):
    async def adapt_input(self, context, **kwargs):
        # Extract data from context
        return context.metadata.get("input_data", [])
    
    async def adapt_output(self, component_output, original_context=None, **kwargs):
        # Create a new context with the result
        return BaseContextSchema(
            metadata={
                "result": component_output["processed_data"],
                "count": component_output["count"],
                "original_id": original_context.context_id if original_context else None
            }
        )

# Usage
async def process_data():
    # Create the component and adapter
    processor = DataProcessor()
    adapter = DataProcessorAdapter(processor)
    
    # Create input context
    input_context = BaseContextSchema(
        metadata={"input_data": [1, 2, 3, 4, 5]}
    )
    
    # Process through the adapter
    result_context = await adapter.process_with_mcp(input_context)
    
    # Access results
    print(f"Result: {result_context.metadata['result']}")  # [2, 4, 6, 8, 10]
    print(f"Count: {result_context.metadata['count']}")    # 5

# Run the example
asyncio.run(process_data())
```

## Testing Approach

The MCP package includes comprehensive tests in `tests/core/test_mcp.py` to ensure all components function correctly and maintain performance standards.

### Test Structure

The tests are organized by component, with separate test classes for each major module:

1. `TestProtocolAndSchema`: Tests for context protocols and schema implementations
2. `TestSerialization`: Tests for serialization and deserialization functionality
3. `TestCompression`: Tests for compression and optimization functionality
4. `TestVersioning`: Tests for version compatibility and context upgrading
5. `TestAdapter`: Tests for adapter base functionality
6. `TestIntegration`: Tests for end-to-end workflows
7. `TestPerformance`: Benchmarks for performance-critical operations

### Key Test Fixtures

The test module includes several fixtures to provide common test objects:

- `base_context`: A basic BaseContextSchema instance
- `task_context`: A TaskContext instance with test data
- `context_with_large_data`: A context containing larger datasets for compression testing
- `mock_metrics_manager`: A fixture that mocks the metrics system for testing

### Running the Tests

To run the MCP tests:

```bash
# Run all MCP tests
pytest tests/core/test_mcp.py -v

# Run a specific test class
pytest tests/core/test_mcp.py::TestCompression -v

# Run a specific test method
pytest tests/core/test_mcp.py::TestCompression::test_compression_roundtrip -v

# Run performance tests (normally skipped)
pytest tests/core/test_mcp.py::TestPerformance -v
```

### Key Test Cases

Some of the most important test cases include:

1. **Context Serialization Roundtrip**: Verifies that contexts can be serialized and deserialized without data loss
2. **Compression Effectiveness**: Tests compression with different data sizes
3. **Version Compatibility Checking**: Verifies version compatibility checks
4. **Adapter Component Execution**: Tests adapters with different component types
5. **Error Handling**: Verifies proper error handling in all components
6. **Performance Benchmarking**: Measures serialization and compression performance

### Testing Metrics Integration

Testing the metrics integration requires special handling since it involves mocking the metrics manager:

```python
import unittest.mock

@pytest.fixture
def mock_metrics_manager():
    """Fixture providing a mock metrics manager."""
    with unittest.mock.patch('src.core.mcp.compression.get_metrics_manager') as mock_get_metrics:
        mock_manager = unittest.mock.MagicMock()
        mock_get_metrics.return_value = mock_manager
        yield mock_manager

def test_compression_with_metrics(self, mock_metrics_manager):
    """Test that compression properly records metrics."""
    # Setup test data
    data = {"id": "test", "value": "A" * 1000}
    
    # Execute compression
    compressed = compress_context(data)
    
    # Verify metrics were recorded
    assert mock_metrics_manager.track_memory.call_count >= 1
    
    # Check that the right metrics were tracked
    metric_types = [call[0][0] for call in mock_metrics_manager.track_memory.call_args_list 
                   if call[0]]
    
    # Ensure both operations and duration metrics were tracked
    assert 'operations' in metric_types
    assert 'duration' in metric_types
```

## Integration and Usage

### Basic Integration

To integrate the MCP package into your application:

1. **Define Context Types**: Create specialized context classes if needed
2. **Create Adapters**: Build adapters for your components
3. **Set Up Communication**: Use serialization and compression for data transfer

### Complete Integration Example

Here's a comprehensive example of integrating the MCP into a multi-agent system:

```python
import asyncio
from src.core.mcp.schema import BaseContextSchema, TaskContext
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.serialization import serialize_context, deserialize_context
from src.core.mcp.compression import optimize_context_data, compress_context, decompress_context
from src.core.mcp.versioning import check_version_compatibility

# 1. Define a specialized context (if needed)
class AnalysisContext(BaseContextSchema):
    analysis_id: str
    data_source: str
    parameters: dict
    results: Optional[dict] = None

# 2. Create a component adapter
class AnalysisAdapter(MCPAdapterBase):
    async def adapt_input(self, context, **kwargs):
        if not isinstance(context, AnalysisContext):
            raise TypeError(f"Expected AnalysisContext, got {type(context).__name__}")
        
        return {
            "id": context.analysis_id,
            "source": context.data_source,
            "params": context.parameters
        }
    
    async def adapt_output(self, component_output, original_context=None, **kwargs):
        if not original_context or not isinstance(original_context, AnalysisContext):
            # Create new context if needed
            return AnalysisContext(
                analysis_id=component_output.get("id", "unknown"),
                data_source=component_output.get("source", "unknown"),
                parameters={},
                results=component_output.get("results", {})
            )
        
        # Update the original context
        updated = original_context.model_copy()
        updated.results = component_output.get("results", {})
        return updated

# 3. Communication between components
async def process_analysis_request(analysis_data, target_component):
    # Create the context
    context = AnalysisContext(
        analysis_id=analysis_data["id"],
        data_source=analysis_data["source"],
        parameters=analysis_data["parameters"]
    )
    
    # Optimize and serialize for transmission
    optimized = optimize_context_data(context)
    binary_data = serialize_context(optimized)
    compressed = compress_context(context.serialize())
    
    # Simulate sending to another system
    received_data = compressed  # In reality, this would be sent over network
    
    # Decompress and deserialize
    decompressed = decompress_context(received_data)
    received_context = deserialize_context(
        serialize_context(decompressed), 
        target_class=AnalysisContext
    )
    
    # Create adapter and process
    adapter = AnalysisAdapter(target_component)
    result_context = await adapter.process_with_mcp(received_context)
    
    # Return the results
    return result_context.results
```

### Integration with Existing Components

To integrate an existing component with the MCP:

1. **Create an adapter**: Implement an adapter for your component
2. **Convert inputs/outputs**: Translate between contexts and component formats
3. **Handle errors**: Properly propagate and convert errors

For example, integrating an existing machine learning model:

```python
class MLModelAdapter(MCPAdapterBase):
    async def adapt_input(self, context, **kwargs):
        # Extract features from context
        features = context.metadata.get("features", [])
        
        # Convert to model input format (e.g., numpy array)
        import numpy as np
        return np.array(features)
    
    async def adapt_output(self, component_output, original_context=None, **kwargs):
        # Create result context
        return BaseContextSchema(
            metadata={
                "prediction": component_output.tolist(),  # Convert numpy to list
                "model_version": self.target_component.version,
                "original_context_id": original_context.context_id if original_context else None
            }
        )
```

## Performance Considerations

The MCP package is designed for high performance in a multi-agent environment:

### Serialization Performance

- **MessagePack vs JSON**: MessagePack is significantly faster and more compact than JSON
- **Targeted Deserialization**: Providing the target_class during deserialization avoids type inference overhead
- **Batch Processing**: Process contexts in batches when possible to reduce overhead

### Compression Strategies

- **Data-Size-Based Compression**: The system uses different compression levels based on data size:
  - Small data (<1KB): Level 1 (fastest, less compression)
  - Medium data (1-10KB): Level 6 (balanced)
  - Large data (>10KB): Level 9 (maximum compression)
- **Compression Thresholds**: Compression is only applied if it actually reduces size

### Adapter Efficiency

- **Method Detection Caching**: Component method detection is optimized to avoid repeated introspection
- **Minimal Copy**: Data is copied only when necessary
- **Direct Execution**: Adapters call component methods directly without intermediate layers

### Memory Usage

- **Context Optimization**: Contexts are optimized to reduce memory footprint
- **Lazy Evaluation**: Operations are performed only when needed
- **Explicit Cleanup**: Resources are cleaned up promptly after use

### Performance Benchmarks

The performance tests ensure the MCP meets performance requirements:

- **Serialization**: <10ms for typical contexts
- **Deserialization**: <10ms for typical contexts
- **Compression**: <20ms for typical contexts
- **Decompression**: <20ms for typical contexts
- **Adapter Overhead**: <1ms for method execution

## Best Practices

### Context Design

1. **Keep contexts focused and minimal**:
   - Include only essential data in context objects
   - Avoid large nested structures
   - Use references to data rather than embedding large objects

2. **Use typed contexts for specific domains**:
   - Create specialized context classes for different purposes
   - Inherit from BaseContextSchema for consistency
   - Add domain-specific fields with appropriate type hints

3. **Leverage the metadata field appropriately**:
   - Use metadata for context-specific attributes
   - Keep metadata keys consistent throughout your application
   - Document the expected structure of metadata

### Serialization and Compression

1. **Choose the right serialization format**:
   - Use MessagePack for efficiency and performance
   - Use JSON when human readability is important
   - Benchmark both formats with your typical data

2. **Optimize before serializing**:
   - Always call optimize_context_data before serialization
   - Implement custom optimize() methods for specialized contexts
   - Remove unnecessary data before serialization

3. **Use compression strategically**:
   - Compress large contexts (>10KB)
   - Monitor compression ratios to ensure effectiveness
   - Balance CPU usage against bandwidth/storage savings

### Adapter Implementation

1. **Create adapters for logical boundaries**:
   - Make adapters for components with different interfaces
   - Prefer coarse-grained adapters over fine-grained ones
   - Group related functionality in a single adapter

2. **Implement robust error handling**:
   - Handle errors in adapt_input and adapt_output
   - Provide meaningful error messages
   - Never silently fail in adapters

3. **Preserve context metadata**:
   - Maintain important metadata when adapting outputs
   - Include references to source contexts where appropriate
   - Document the metadata transformation in your adapter

### Version Management

1. **Follow semantic versioning**:
   - Increment major version for backward-incompatible changes
   - Increment minor version for backward-compatible new features
   - Increment patch version for backward-compatible bug fixes

2. **Plan version upgrade paths**:
   - Implement upgrade functions for all supported version pairs
   - Test upgrade paths thoroughly
   - Document breaking changes

3. **Check version compatibility early**:
   - Verify version compatibility before processing
   - Log warnings for deprecated versions
   - Have fallback strategies for incompatible versions

## Extension Points

The MCP package can be extended in several ways:

### Additional Context Types

Create specialized context classes for specific domains:

```python
from src.core.mcp.schema import BaseContextSchema
from typing import List, Dict, Optional

class ConversationContext(BaseContextSchema):
    """Context for conversation history tracking."""
    conversation_id: str
    messages: List[Dict[str, str]]
    user_id: str
    agent_id: str
    sentiment: Optional[str] = None
    
    def optimize(self) -> 'ConversationContext':
        """Optimize by limiting message history."""
        optimized = super().optimize()
        # Keep only last 10 messages if there are more
        if len(optimized.messages) > 10:
            optimized.messages = optimized.messages[-10:]
        return optimized
```

### Custom Adapters

Implement specialized adapters for different component types:

```python
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.schema import BaseContextSchema

class APIServiceAdapter(MCPAdapterBase):
    """Adapter for external API services."""
    
    async def adapt_input(self, context, **kwargs):
        # Convert context to API request format
        return {
            "endpoint": context.metadata.get("endpoint"),
            "method": context.metadata.get("method", "GET"),
            "params": context.metadata.get("params", {}),
            "headers": context.metadata.get("headers", {})
        }
    
    async def adapt_output(self, component_output, original_context=None, **kwargs):
        # Convert API response to context
        return BaseContextSchema(
            metadata={
                "status_code": component_output.get("status"),
                "response_data": component_output.get("data"),
                "request_id": original_context.metadata.get("request_id") if original_context else None
            }
        )
```

### Serialization Plugins

Extend the serialization system with additional formats:

```python
from src.core.mcp.serialization import serialize_context, deserialize_context
from src.utils.serialization import SerializationFormat
import cbor2  # Example external library

# Add CBOR format to serialization system
class ExtendedSerializationFormat(SerializationFormat):
    CBOR = 'cbor'

def serialize_context_cbor(context):
    """Serialize context to CBOR format."""
    context_dict = context.serialize()
    return cbor2.dumps(context_dict)

def deserialize_context_cbor(data, target_class=None):
    """Deserialize context from CBOR format."""
    context_dict = cbor2.loads(data)
    if target_class:
        return target_class.deserialize(context_dict)
    # Fall back to type inference...
```

### Compression Algorithms

Add support for different compression algorithms:

```python
import lzma
from src.core.mcp.compression import compress_context, decompress_context

def compress_context_lzma(context_data):
    """Compress context data using LZMA."""
    import json
    json_data = json.dumps(context_data).encode('utf-8')
    return lzma.compress(json_data)

def decompress_context_lzma(compressed_data):
    """Decompress LZMA compressed context data."""
    decompressed = lzma.decompress(compressed_data)
    import json
    return json.loads(decompressed.decode('utf-8'))
```

## Key Improvements from Refactoring

The MCP package underwent several improvements during refactoring:

### Critical Fixes

1. **Missing Import Fix**:
   - Added missing `asyncio` import in adapter_base.py
   - Fixed potential runtime errors when checking if methods are coroutine functions

2. **Incomplete Schema Method Fix**:
   - Completed the `optimize()` method in BaseContextSchema
   - Added proper optimization flag setting

3. **Error Propagation Fix**:
   - Updated compression error handling to properly propagate TypeErrors
   - Improved error handling in deserialization for better diagnostics

4. **Testing Infrastructure**:
   - Fixed metrics mocking in tests to properly patch the relevant modules
   - Added missing self parameter to test method

### Performance Improvements

1. **Optimized Compression Strategy**:
   - Implemented data size-based compression level selection
   - Added checks to only use compression when it actually reduces size
   - Improved compression ratio tracking

2. **Enhanced Serialization**:
   - Refactored serialization with helper methods for better maintainability
   - Improved type inference during deserialization
   - Added direct deserialization with target class for better performance

3. **Metrics Integration**:
   - Added detailed performance metrics for serialization and compression operations
   - Improved timing precision for performance monitoring

### Code Structure Improvements

1. **Refactored Adapter Base**:
   - Improved method detection and execution logic
   - Enhanced error handling and logging
   - Added support for both synchronous and asynchronous components

2. **Enhanced Versioning**:
   - Implemented version compatibility checking based on semantic versioning principles
   - Added support for upgrading unversioned contexts
   - Improved version handling utilities

3. **Better Deserialization Logic**:
   - Split monolithic function into focused helper methods
   - Improved error messages for deserialization failures
   - Enhanced type inference capabilities

These improvements have resulted in a more robust, maintainable, and performant MCP system that better serves the needs of the multi-agent platform.

## Conclusion

The Model Context Protocol (MCP) package provides a solid foundation for communication between components in the multi-agent platform. By standardizing context representation, serialization, and adaptation, it enables seamless integration of diverse components while maintaining high performance.

The modular design with clear separation of concerns allows for easy extension and customization, while the comprehensive testing ensures reliability and performance. With proper use of the MCP, developers can focus on building specialized components without worrying about the complexities of inter-component communication.
# Multi-Agent Platform: Model Context Protocol (MCP) LLM Integration

## Architecture Overview

The Model Context Protocol (MCP) LLM Integration package provides a robust framework for integrating Large Language Models (LLMs) into the multi-agent platform. This subsystem handles the complex interactions between different components and LLM providers while maintaining context integrity throughout the process.

### Purpose and Responsibilities

The core/mcp/llm package serves several critical functions:

1. **Context-Aware Model Selection**: Intelligently selects the most appropriate LLM model based on task context attributes.
2. **Context Transformation**: Adapts input formats between different LLM providers to maintain compatibility.
3. **Context Preservation During Failures**: Implements sophisticated fallback mechanisms to handle LLM provider failures without losing context.
4. **Performance Monitoring**: Tracks and labels context for performance metrics and observability.

### Component Interactions

![MCP LLM Components](https://via.placeholder.com/800x400?text=MCP+LLM+Component+Interactions)

The components interact as follows:

1. The `ContextModelSelector` determines the appropriate model for a given context
2. The `transform_llm_input_for_model` function adapts the input format for the selected model
3. The `execute_mcp_llm_with_context_preserving_fallback` function handles the LLM interaction, with fallback to alternative models if needed
4. The `get_context_labels` function extracts relevant context information for metrics and monitoring

### Design Patterns

The package leverages several design patterns:

- **Adapter Pattern**: Used in the `context_transform.py` module to adapt between different LLM provider input formats
- **Strategy Pattern**: Implemented in `context_model_selector.py` to select models based on contextual rules
- **Singleton Pattern**: Applied to the `ContextModelSelector` to ensure consistent model selection logic
- **Circuit Breaker Pattern**: Used in fallback mechanisms to prevent cascading failures

### Key Abstractions

- **Context Protocol**: A standardized interface for representing and manipulating context
- **Model Selection Rules**: Configuration-driven rules for determining the appropriate model
- **Fallback Chain**: A prioritized sequence of alternative models to try when the primary model fails

## Component Details

### 1. Context Model Selector (`context_model_selector.py`)

#### Primary Purpose

The `ContextModelSelector` evaluates various aspects of a context object to determine the most appropriate LLM model for processing that context.

#### Core Classes and Relationships

- **ContextModelSelector**: The main class that implements model selection logic
- **get_context_model_selector()**: Factory function that implements the Singleton pattern

#### Key Features

- Rule-based model selection based on context attributes
- Support for task type, estimated tokens, and specific content requirements
- Fallback to default models when no specific rules match
- Singleton pattern to ensure consistent rule application

#### Usage Example

```python
from src.core.mcp.llm.context_model_selector import get_context_model_selector
from src.core.mcp.schema import TaskContext

async def process_task(task_context: TaskContext):
    # Get the singleton model selector instance
    model_selector = await get_context_model_selector()
    
    # Select the appropriate model based on context
    selected_model = await model_selector.select_model(task_context)
    
    # Use the selected model for processing
    print(f"Selected model: {selected_model}")
```

#### Public Interfaces

```python
class ContextModelSelector:
    async def select_model(
        self, 
        context: ContextProtocol,
        available_models: Optional[List[str]] = None
    ) -> str:
        """
        Select the most appropriate model based on context attributes.
        
        Args:
            context: The context object containing task information
            available_models: Optional list of available models to choose from
            
        Returns:
            str: The name of the selected model
        """
        
async def get_context_model_selector() -> ContextModelSelector:
    """
    Get the singleton instance of the ContextModelSelector.
    
    Returns:
        ContextModelSelector: The singleton instance
    """
```

#### Best Practices

- Add new selection rules to handle specific task types or requirements
- Keep selection rules prioritized with most specific first
- Ensure rules handle edge cases gracefully
- Log selection decisions for debugging and monitoring

#### Performance Considerations

- Rule evaluation is performed sequentially; keep rules lightweight
- The selector uses a singleton pattern to avoid repeated instantiation
- Consider caching results for similar contexts to improve performance

### 2. Context Performance Tracking (`context_performance.py`)

#### Primary Purpose

Extracts relevant context information for performance tracking and monitoring, ensuring observability across the system.

#### Core Functions

- **get_context_labels()**: Extracts labeled attributes from context objects

#### Key Features

- Extracts task type, context class, and other relevant metadata
- Returns standardized labels for metrics collection
- Handles different context types appropriately

#### Usage Example

```python
from src.core.mcp.llm.context_performance import get_context_labels
from src.config.metrics import get_metrics_manager

def record_context_metrics(context):
    # Extract labels from context
    labels = get_context_labels(context)
    
    # Use labels for metrics recording
    metrics = get_metrics_manager()
    metrics.track_llm('requests', **labels)
```

#### Public Interfaces

```python
def get_context_labels(context: Optional[ContextProtocol]) -> Dict[str, str]:
    """
    Extract relevant labels from a context object for metrics and monitoring.
    
    Args:
        context: The context object to extract labels from, or None
        
    Returns:
        Dict[str, str]: Dictionary of label keys and values
    """
```

#### Best Practices

- Extend with additional context attributes as needed for monitoring
- Keep label values simple and string-based for compatibility with metrics systems
- Filter out empty or None values to keep labels clean

### 3. Context Transformation (`context_transform.py`)

#### Primary Purpose

Transforms context and prompts between different formats required by various LLM providers.

#### Core Functions

- **transform_llm_input_for_model()**: Adapts input formats based on the target LLM provider

#### Key Features

- Converts between string prompts and message lists
- Handles provider-specific format requirements
- Preserves content while changing structure

#### Usage Example

```python
from src.core.mcp.llm.context_transform import transform_llm_input_for_model
from src.llm.adapters import get_adapter

async def process_with_different_models(prompt, models):
    results = []
    
    for model_name in models:
        # Get the adapter for this model
        adapter = await get_adapter(model_name)
        
        # Transform the prompt for this specific model/provider
        transformed_input = await transform_llm_input_for_model(prompt, adapter)
        
        # Use the transformed input with the model
        response = await adapter.generate(transformed_input)
        results.append(response)
        
    return results
```

#### Public Interfaces

```python
async def transform_llm_input_for_model(
    original_input: Union[str, List[Dict[str, str]]],
    target_adapter: BaseLLMAdapter
) -> Union[str, List[Dict[str, str]]]:
    """
    Transform input format based on target LLM adapter requirements.
    
    Args:
        original_input: Original prompt as string or message list
        target_adapter: The adapter for the target LLM model
        
    Returns:
        Union[str, List[Dict[str, str]]]: Transformed input in the appropriate format
    """
```

#### Best Practices

- Add support for new providers as they are integrated
- Ensure transformations preserve all content and meaning
- Add logging for format changes to aid debugging
- Handle edge cases like mixed content types

#### Performance Considerations

- String concatenation can be expensive for very large contexts
- Consider streaming approaches for very large inputs

### 4. Context-Preserving Fallback (`context_preserving_fallback.py`)

#### Primary Purpose

Implements robust fallback mechanisms to handle LLM provider failures while preserving context integrity.

#### Core Functions

- **execute_mcp_llm_with_context_preserving_fallback()**: Executes LLM requests with automatic fallback

#### Key Features

- Tries multiple models in sequence when failures occur
- Preserves context across fallback attempts
- Tracks metrics for fallback operations
- Handles various error conditions appropriately

#### Usage Example

```python
from src.core.mcp.llm.context_preserving_fallback import execute_mcp_llm_with_context_preserving_fallback
from src.core.mcp.adapters.llm_adapter import LLMAdapter

async def process_with_fallback(prompt, mcp_adapter):
    try:
        # Execute with fallback support
        model, output_context = await execute_mcp_llm_with_context_preserving_fallback(
            requested_model="gpt-4",
            original_prompt_or_messages=prompt,
            mcp_llm_adapter=mcp_adapter,
            parameters={"temperature": 0.7},
            metadata={"trace_id": "request-123"}
        )
        
        print(f"Successfully processed with model: {model}")
        return output_context
        
    except LLMError as e:
        print(f"All fallback attempts failed: {e}")
        raise
```

#### Public Interfaces

```python
async def execute_mcp_llm_with_context_preserving_fallback(
    requested_model: Optional[str],
    original_prompt_or_messages: Union[str, List[Dict[str, str]]],
    mcp_llm_adapter: LLMAdapter,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overall_timeout: Optional[float] = None,
    track_metrics: bool = True,
) -> Tuple[str, LLMOutputContext]:
    """
    Execute LLM request with automatic fallback to alternative models.
    
    Args:
        requested_model: Preferred model to use, or None for default
        original_prompt_or_messages: The prompt or message list to send
        mcp_llm_adapter: The MCP LLM adapter to use
        parameters: Optional LLM parameters like temperature
        metadata: Optional metadata for the request
        overall_timeout: Maximum time for all fallback attempts
        track_metrics: Whether to track metrics for this operation
        
    Returns:
        Tuple[str, LLMOutputContext]: The model used and output context
        
    Raises:
        LLMError: If all fallback attempts fail
    """
```

#### Best Practices

- Configure fallback chains appropriately for your use case
- Set appropriate timeouts to prevent hung requests
- Monitor fallback metrics to identify provider issues
- Consider cost implications of fallback models

#### Performance Considerations

- Fallbacks introduce latency; consider using faster models in the fallback chain
- Track timeout budgets across fallback attempts
- Consider circuit breaking for persistently failing providers

## Usage Examples

### Basic Model Selection

```python
from src.core.mcp.llm.context_model_selector import get_context_model_selector
from src.core.mcp.schema import TaskContext

async def select_model_for_task():
    # Create a task context
    task_context = TaskContext(
        task_id="task-123",
        task_type="coding",
        metadata={"estimated_tokens": 60000},
        input_data={"goal": "Write a Django web application"}
    )
    
    # Get the model selector
    selector = await get_context_model_selector()
    
    # Select model based on context
    model = await selector.select_model(task_context)
    
    print(f"Selected model for coding task: {model}")  # Expected: gpt-4-turbo
```

### Transforming Input for Different Providers

```python
from src.core.mcp.llm.context_transform import transform_llm_input_for_model
from src.llm.adapters.openai import OpenAIAdapter
from src.llm.adapters.anthropic import AnthropicAdapter

async def demonstrate_transformations():
    # Create adapters
    openai_adapter = OpenAIAdapter(model="gpt-4")
    anthropic_adapter = AnthropicAdapter(model="claude-3-opus")
    
    # Original input as string
    string_input = "Explain quantum computing in simple terms."
    
    # Original input as messages
    messages_input = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
    
    # Transform string for OpenAI
    openai_from_string = await transform_llm_input_for_model(string_input, openai_adapter)
    print(f"String transformed for OpenAI: {openai_from_string}")
    # Expected: List of message dictionaries
    
    # Transform messages for Anthropic
    anthropic_from_messages = await transform_llm_input_for_model(messages_input, anthropic_adapter)
    print(f"Messages preserved for Anthropic: {anthropic_from_messages}")
    # Expected: Same message list structure
```

### Implementing Fallback with Metrics

```python
from src.core.mcp.llm.context_preserving_fallback import execute_mcp_llm_with_context_preserving_fallback
from src.core.mcp.adapters.llm_adapter import LLMAdapter
from src.config.metrics import get_metrics_manager

async def process_with_fallback_and_metrics():
    # Get metrics manager
    metrics = get_metrics_manager()
    metrics.start_metrics_server()
    
    # Create LLM adapter
    llm_adapter = LLMAdapter()
    
    try:
        # Execute with fallback
        model, output = await execute_mcp_llm_with_context_preserving_fallback(
            requested_model="gpt-4",
            original_prompt_or_messages="Explain the theory of relativity.",
            mcp_llm_adapter=llm_adapter,
            parameters={
                "temperature": 0.7,
                "max_tokens": 500
            },
            metadata={
                "trace_id": "request-456",
                "user_id": "user-789"
            },
            track_metrics=True
        )
        
        # Success case
        print(f"Successfully processed with model: {model}")
        print(f"Output content: {output.content}")
        
    except LLMError as e:
        # Handle the case where all models failed
        print(f"All models failed: {e}")
        
        # Record failure metric
        metrics.track_llm('errors', error_type='all_models_failed')
```

### Advanced Context Performance Monitoring

```python
from src.core.mcp.llm.context_performance import get_context_labels
from src.config.metrics import get_metrics_manager
from src.core.mcp.schema import TaskContext
import time

def monitor_task_performance():
    metrics = get_metrics_manager()
    
    # Create a task context
    task_context = TaskContext(
        task_id="task-987",
        task_type="summarization",
        metadata={
            "estimated_tokens": 8000,
            "priority": "high",
            "source": "email"
        },
        input_data={"content": "Long document text..."}
    )
    
    # Start timing
    start_time = time.time()
    
    try:
        # Extract labels for metrics
        labels = get_context_labels(task_context)
        
        # Simulate processing
        time.sleep(1.5)  # Simulate work
        
        # Record processing time with context labels
        duration = time.time() - start_time
        metrics.track_llm('duration', value=duration, **labels)
        
        # Record success
        metrics.track_llm('requests', **labels)
        
    except Exception as e:
        # Record failure with context
        metrics.track_llm('errors', error_type=type(e).__name__, **labels)
        raise
```

## Best Practices

### Model Selection Configuration

1. **Define Clear Selection Rules**:
   - Each rule should have a specific purpose
   - Rules should be ordered from most to least specific
   - Include sensible defaults for when no rule matches

2. **Test Selection Rules Thoroughly**:
   - Verify with different context types
   - Test edge cases like empty metadata
   - Confirm fallback behavior

3. **Monitor Selection Decisions**:
   - Log which rule matched and why
   - Track model usage metrics
   - Evaluate selection effectiveness over time

### Error Handling in LLM Interactions

1. **Categorize Errors Appropriately**:
   - Distinguish between retryable and non-retryable errors
   - Tag errors with specific error codes
   - Preserve error details for debugging

2. **Set Appropriate Timeouts**:
   - Use timeouts for both individual requests and overall operations
   - Implement progressive backoff for retries
   - Account for network latency in timeout calculations

3. **Implement Circuit Breakers**:
   - Detect failing providers early
   - Avoid overwhelming already stressed services
   - Automatically resume when service recovers

### Token Usage Optimization

1. **Monitor Token Consumption**:
   - Track prompt and completion tokens separately
   - Set budgets for different operation types
   - Alert on unexpected token usage patterns

2. **Optimize Prompts**:
   - Remove unnecessary context
   - Use token-efficient instruction patterns
   - Consider compression for repeated content

3. **Select Models Based on Requirements**:
   - Use smaller models for simpler tasks
   - Reserve large context models for when needed
   - Consider token cost in model selection

### Context Preservation Strategies

1. **Maintain Context Integrity**:
   - Preserve essential context during transformations
   - Validate context after operations
   - Use checksums for critical context elements

2. **Implement Clean Fallbacks**:
   - Ensure context is fully preserved during fallbacks
   - Validate model outputs after fallbacks
   - Propagate context metadata through all operations

3. **Handle Context Size Limits**:
   - Implement context pruning when needed
   - Prioritize what context to keep
   - Split large contexts for models with smaller limits

## Testing Approach

### Test Structure and Organization

The testing framework for the MCP LLM package follows a component-based approach, with specialized test classes for each module:

- **TestContextModelSelector**: Tests for model selection logic
- **TestContextPerformance**: Tests for context labeling and metrics
- **TestContextTransform**: Tests for input format transformation
- **TestContextPreservingFallback**: Tests for fallback mechanisms

### Running the Tests

Tests can be run using pytest:

```bash
# Run all MCP LLM tests
pytest tests/core/mcp/test_llm.py -v

# Run a specific test class
pytest tests/core/mcp/test_llm.py::TestContextModelSelector -v

# Run a specific test method
pytest tests/core/mcp/test_llm.py::TestContextModelSelector::test_select_model_defaults -v
```

### Key Test Fixtures

The test suite uses several fixtures:

- **model_selector**: Creates a fresh ContextModelSelector instance
- **mock_task_context**: Creates a mock TaskContext for testing selection rules
- **mock_llm_adapter**: Mock LLM adapter for testing transformations
- **mock_mcp_adapter**: Mock MCP adapter for testing fallbacks

### Mock Objects

Mock objects are extensively used to simulate components like:

- Task contexts with various attributes
- LLM adapters for different providers
- MCP adapters for processing
- Error conditions and responses

### Test Environment Setup

Each test sets up its environment with:

- Patched external dependencies
- Mock objects for components
- Controlled test data
- Expected output values

### Common Test Patterns

Several patterns are used consistently:

- **Happy Path Testing**: Verifying expected behavior under normal conditions
- **Error Path Testing**: Confirming proper error handling and fallbacks
- **Edge Case Testing**: Testing boundary conditions and unusual inputs
- **Mock-Based Testing**: Using mocks to isolate components

### Examples of Key Test Cases

#### Model Selection Testing

```python
@pytest.mark.asyncio
async def test_select_model_coding_task(self, model_selector):
    """Test that coding task with large token count selects the expected model."""
    context = MagicMock(spec=TaskContext)
    context.task_type = "coding"
    context.metadata = {"estimated_tokens": 60000}
    
    with patch('src.core.mcp.llm.context_model_selector.list_available_models', 
               return_value=['gpt-3.5-turbo', 'gpt-4-turbo', 'claude-3-haiku']):
        selected_model = await model_selector.select_model(context)
        
        assert selected_model == 'gpt-4-turbo'
```

#### Fallback Testing

```python
@pytest.mark.asyncio
async def test_fallback_to_secondary_model(self, mock_llm_adapter, mock_mcp_adapter):
    """Test fallback to secondary model when primary fails."""
    # Setup mocks
    with patch('src.core.mcp.llm.context_preserving_fallback.select_models', 
               return_value=("gpt-4", ["gpt-3.5-turbo"])), \
         patch('src.core.mcp.llm.context_preserving_fallback.get_llm_adapter_instance') as mock_get_adapter:
        
        # Create adapters with primary failing
        primary_adapter = AsyncMock()
        primary_adapter.ensure_initialized.side_effect = LLMError(
            code=ErrorCode.LLM_RATE_LIMIT, 
            message="Rate limit exceeded"
        )
        
        secondary_adapter = AsyncMock()
        secondary_adapter.ensure_initialized.return_value = True
        
        # Return different adapters based on model name
        mock_get_adapter.side_effect = lambda model, **kwargs: \
            primary_adapter if model == "gpt-4" else secondary_adapter
            
        # Mock successful execution on secondary model
        mock_output_context = MagicMock()
        mock_output_context.success = True
        mock_mcp_adapter.process_with_mcp.return_value = mock_output_context
        
        # Execute and verify fallback
        model, output = await execute_mcp_llm_with_context_preserving_fallback(
            requested_model="gpt-4",
            original_prompt_or_messages="Hello",
            mcp_llm_adapter=mock_mcp_adapter
        )
        
        assert model == "gpt-3.5-turbo"
        assert output == mock_output_context
```

## Implementation Notes

### Design Decisions

#### Context-Based Model Selection

The system uses a rule-based approach for model selection rather than a more complex machine learning approach. This decision was made because:

1. Rule-based selection is deterministic and explainable
2. The selection criteria are well-defined and relatively straightforward
3. Rules can be easily updated as new models or requirements emerge

#### Fallback Mechanism Design

The fallback mechanism was implemented as a single function rather than a class hierarchy for several reasons:

1. The operation is fundamentally a pipeline process
2. State management between fallback attempts is minimal
3. The implementation is more closely tied to configuration than inheritance

#### Format Transformation Strategy

The format transformation module uses adapter-aware transformation rather than standardizing on a single format because:

1. Different LLM providers have fundamentally different input formats
2. Preserving native formats optimizes for each provider's strengths
3. This approach minimizes unnecessary transformations

### Critical Fixes and Improvements

#### Provider Support

Added support for Gemini models in the context transformation module to handle all supported LLM providers.

#### Model Selection Configuration

Made model selection rules configurable through settings rather than hardcoded, enabling:
- Environment-specific rule configurations
- Dynamic updates without code changes
- Separation of policy from implementation

#### Enhanced Context Labeling

Improved context labeling for metrics to include:
- Task-specific metadata
- Performance indicators
- User-definable attributes

### Thread Safety Considerations

The package is designed with asyncio in mind:

- The model selector uses a mutex for singleton instantiation
- Fallback operations properly manage async resources
- Transformations are stateless and thread-safe

### Asynchronous Programming Patterns

The codebase follows several async patterns:

- **Async Factory Functions**: For obtaining shared instances
- **Task-Based Concurrency**: For parallel operations
- **Async Context Managers**: For resource management
- **Async Mutex Locks**: For thread-safety in critical sections

### Resource Management

Resources are managed carefully to avoid leaks:

- Model selection is handled by a singleton to avoid proliferation
- Connection pooling is used at the LLM adapter level
- Timeouts are enforced to prevent resource exhaustion
- Fallbacks are bounded to prevent infinite retries

### Performance Optimizations

Several optimizations are implemented:

- **Caching Model Selection**: Results can be cached for similar contexts
- **Efficient Transformations**: Minimal copying during format conversions
- **Strategic Fallbacks**: Quick failure detection to avoid waiting for timeouts
- **Request Batching**: When possible, similar requests are batched

## API Reference

### Context Model Selector

#### `ContextModelSelector.select_model`

```python
async def select_model(
    self, 
    context: ContextProtocol, 
    available_models: Optional[List[str]] = None
) -> str
```

**Purpose**: Selects the most appropriate LLM model based on context attributes.

**Parameters**:
- `context`: The context object containing task information (required)
- `available_models`: Optional list of available models to choose from (defaults to all enabled models)

**Returns**: String name of the selected model.

**Exceptions**:
- Logs warnings for rule evaluation errors but doesn't raise exceptions
- Returns default model if selection fails

**Thread Safety**: Thread-safe, can be called concurrently.

**Performance**: O(n) where n is the number of selection rules.

#### `get_context_model_selector`

```python
async def get_context_model_selector() -> ContextModelSelector
```

**Purpose**: Provides the singleton instance of the ContextModelSelector.

**Parameters**: None

**Returns**: The shared ContextModelSelector instance.

**Exceptions**: May raise RuntimeError if selector creation fails.

**Thread Safety**: Thread-safe, protected by async mutex.

**Performance**: O(1) after first call, initialization is O(n) where n is number of rules.

### Context Performance

#### `get_context_labels`

```python
def get_context_labels(context: Optional[ContextProtocol]) -> Dict[str, str]
```

**Purpose**: Extracts standardized labels from context for metrics and monitoring.

**Parameters**:
- `context`: The context object to extract labels from, can be None

**Returns**: Dictionary with string keys and values representing labels.

**Exceptions**: None, handles missing attributes gracefully.

**Thread Safety**: Thread-safe, stateless function.

**Performance**: O(1), simple property extraction.

### Context Transform

#### `transform_llm_input_for_model`

```python
async def transform_llm_input_for_model(
    original_input: Union[str, List[Dict[str, str]]],
    target_adapter: BaseLLMAdapter
) -> Union[str, List[Dict[str, str]]]
```

**Purpose**: Transforms input format based on the requirements of the target LLM adapter.

**Parameters**:
- `original_input`: The original prompt or message list
- `target_adapter`: The adapter for the target LLM model

**Returns**: Transformed input in either string or message list format.

**Exceptions**: Logs warnings for unexpected input types but attempts conversion.

**Thread Safety**: Thread-safe, stateless function.

**Performance**: O(n) where n is the size of the input.

### Context Preserving Fallback

#### `execute_mcp_llm_with_context_preserving_fallback`

```python
async def execute_mcp_llm_with_context_preserving_fallback(
    requested_model: Optional[str],
    original_prompt_or_messages: Union[str, List[Dict[str, str]]],
    mcp_llm_adapter: LLMAdapter,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overall_timeout: Optional[float] = None,
    track_metrics: bool = True
) -> Tuple[str, LLMOutputContext]
```

**Purpose**: Executes an LLM request with automatic fallback to alternative models if needed.

**Parameters**:
- `requested_model`: Preferred model to use, or None for default
- `original_prompt_or_messages`: The prompt or message list to send
- `mcp_llm_adapter`: The MCP LLM adapter to use
- `parameters`: Optional parameters like temperature and max tokens
- `metadata`: Optional metadata for the request
- `overall_timeout`: Maximum time for all fallback attempts
- `track_metrics`: Whether to track metrics for this operation

**Returns**: Tuple containing the model name used and the output context.

**Exceptions**:
- `LLMError`: If all fallback attempts fail

**Thread Safety**: Thread-safe, maintains its own state.

**Performance**: O(m*r) where m is message size and r is number of retries needed.

## Integration Guidelines

### Initialization Sequence

1. **Environment Configuration**:
   - Set API keys for LLM providers
   - Configure enabled models
   - Define model selection rules

2. **Component Initialization**:
   - Initialize metrics manager
   - Initialize logging
   - Set up connection pools

3. **MCP LLM Integration**:
   - Get model selector singleton
   - Create LLM adapters for required models
   - Initialize MCP adapters

### Configuration Options

#### Model Selection Rules

Configure model selection in settings:

```python
# In settings.py or environment variables
CONTEXT_MODEL_SELECTION_RULES = [
    {
        "condition": "lambda ctx: isinstance(ctx, TaskContext) and ctx.task_type == 'coding'",
        "preferred_model": "gpt-4-turbo" 
    },
    {
        "condition": "lambda ctx: ctx.metadata.get('low_latency', False)",
        "preferred_model": "claude-3-haiku"
    }
]
```

#### Fallback Configuration

Configure fallback behavior:

```python
# In settings.py or environment variables
PRIMARY_LLM = "gpt-4"
FALLBACK_LLM = "claude-3-opus"
LLM_RETRY_MAX_ATTEMPTS = 3
LLM_RETRY_BACKOFF_FACTOR = 0.5
LLM_RETRY_JITTER = True
```

### Resource Lifecycle Management

1. **Startup**:
   ```python
   # Start metrics server
   metrics = get_metrics_manager()
   metrics.start_metrics_server()
   
   # Initialize connection pools
   await initialize_connection_pools()
   ```

2. **Shutdown**:
   ```python
   # Close LLM adapters
   for adapter in active_adapters:
       await adapter.close()
       
   # Close connection pools
   await close_connection_pools()
   ```

## Key Improvements

### Provider Support Enhancement

#### Before:
The context transformation only supported OpenAI and Anthropic formats.

#### After:
Added support for Gemini models, enabling full coverage of all supported LLM providers:
```python
elif isinstance(target_adapter, GeminiAdapter):
    expects_messages_list = True
```

### Configurable Model Selection

#### Before:
Model selection rules were hardcoded in the `ContextModelSelector` class.

#### After:
Rules can be loaded from configuration:
```python
def _load_selection_rules(self):
    # Try to load from settings first
    custom_rules = getattr(settings, 'CONTEXT_MODEL_SELECTION_RULES', None)
    if custom_rules:
        logger.info('Loading custom model selection rules from settings')
        return custom_rules
    
    # Default rules
    return [...]
```

### Enhanced Context Labeling

#### Before:
Context labeling for metrics was minimal.

#### After:
Added support for richer context labels:
```python
def get_context_labels(context: Optional[ContextProtocol]) -> Dict[str, str]:
    # ... existing code ...
    
    # Extract relevant metadata as labels
    if hasattr(context, 'metadata') and context.metadata:
        for key in ['priority', 'source', 'user_id']:
            if key in context.metadata:
                labels[f'metadata_{key}'] = str(context.metadata[key])
        
        # Performance metrics
        if 'estimated_tokens' in context.metadata:
            labels['estimated_tokens'] = str(context.metadata['estimated_tokens'])
    
    return {k: str(v) for k, v in labels.items() if v}
```

### Improved Testing

#### Before:
Limited test coverage for edge cases and failure scenarios.

#### After:
Added comprehensive test coverage:
- Tests for model selection with various context types
- Tests for fallback behavior with different error scenarios
- Tests for format transformation across providers
- Tests for context labeling with different context structures

These improvements have resulted in a more robust, maintainable, and extensible MCP LLM integration system.
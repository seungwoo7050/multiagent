# Prompt Template System - Technical Documentation

## Architecture Overview

The Prompt Template System provides a flexible, efficient way to manage and render prompts with variable substitution for the high-performance multi-agent platform. It's designed to support asynchronous operations, efficient caching, and robust template management.

### Key Components

1. **PromptTemplate**: Handles individual templates with parsing, validation, and rendering capabilities
2. **PromptTemplateLoader**: Manages template loading from various sources with caching support
3. **DictFormatter**: Custom string formatter that enables dot notation access to nested dictionaries
4. **Global Accessor Functions**: Provide singleton access to the template loader

### Design Patterns

- **Singleton Pattern**: Applied to the template loader to ensure a single global instance
- **Factory Pattern**: Used in the loader to create appropriate template instances
- **Decorator Pattern**: Applied in logging and performance tracking
- **Template Method Pattern**: Used in the template loading process

### Key Abstractions

- **Template**: A string with placeholders for variables, stored with metadata
- **Variable**: Placeholders within templates that can be substituted with values
- **Rendering**: The process of replacing variables with actual values
- **Template Source**: The location from which templates are loaded (files, directories)

## Component Details

### DictFormatter

#### Purpose
Extends Python's string.Formatter to support dictionary access using dot notation and array indexing.

#### Features
- Dot notation for accessing nested dictionary keys (`person.name`)
- Array/list indexing support (`hobbies[0]`)
- Graceful handling of missing keys with informative placeholders
- Custom parsing of complex field expressions

#### Example
```python
formatter = DictFormatter()
data = {"person": {"name": "Alice", "hobbies": ["reading", "hiking"]}}
result = formatter.format("Name: {person.name}, Hobby: {person.hobbies[0]}", **data)
# Result: "Name: Alice, Hobby: reading"
```

#### Implementation Details
The formatter parses field names into parts (split by dots and brackets) and traverses the object hierarchy accordingly, handling both dictionary keys and object attributes.

### PromptTemplate

#### Purpose
Represents a prompt template with variable extraction, validation, and rendering capabilities.

#### Features
- Variable extraction from template strings
- Template validation for properly matched braces
- Efficient template rendering with variable substitution
- Support for nested dictionary access via dot notation
- Comprehensive error reporting

#### Public Interface
```python
class PromptTemplate:
    def __init__(self, template_id: str, template_string: str, input_variables: Optional[List[str]]=None)
    def render(self, **kwargs: Any) -> str
```

#### Example
```python
template = PromptTemplate(
    "greeting",
    "Hello {name}, welcome to {service}!"
)
result = template.render(name="Alice", service="AI Platform")
# Result: "Hello Alice, welcome to AI Platform!"
```

#### Best Practices
- Give templates descriptive IDs that reflect their purpose
- Keep templates focused on a single responsibility
- Document expected variables in comments or separate documentation
- Handle rendering errors gracefully

### PromptTemplateLoader

#### Purpose
Manages loading and caching of templates from various sources.

#### Features
- Asynchronous template loading
- Caching of loaded templates for performance
- Support for multiple template sources (files, directories)
- Multiple file formats (JSON, plain text)
- Fallback to hardcoded templates

#### Public Interface
```python
class PromptTemplateLoader:
    def __init__(self, template_source: str)
    async def load_template(self, template_id: str) -> Optional[PromptTemplate]
```

#### Example
```python
loader = PromptTemplateLoader("templates/")
template = await loader.load_template("greeting")
if template:
    result = template.render(name="Alice", service="AI Platform")
```

#### Best Practices
- Organize templates in logical directory structures
- Use JSON for templates with metadata, plain text for simple templates
- Prefer loading from files over hardcoded templates
- Implement proper error handling for missing templates

### Global Accessor Functions

#### Purpose
Provide convenient access to a singleton template loader instance.

#### Features
- Thread-safe initialization
- Configurable template source
- Environment variable support
- Template cache clearing/reloading

#### Public Interface
```python
async def get_prompt_loader(template_source: Optional[str]=None) -> PromptTemplateLoader
async def reload_templates() -> None
```

#### Example
```python
# Get the global loader
loader = await get_prompt_loader("templates/")

# Load and render a template
template = await loader.load_template("greeting")
result = template.render(name="Alice", service="AI Platform")

# Reload templates (clear cache)
await reload_templates()
```

## Usage Examples

### Basic Template Usage

```python
# Create a template directly
template = PromptTemplate(
    "simple_greeting",
    "Hello {name}!"
)

# Render the template
greeting = template.render(name="World")
# Result: "Hello World!"
```

### Nested Dictionary Access

```python
# Create a template with nested access
template = PromptTemplate(
    "user_profile",
    "Name: {user.name}\nAge: {user.age}\nFavorite Color: {user.preferences.color}"
)

# Prepare data with nested structure
user_data = {
    "user": {
        "name": "Alice",
        "age": 30,
        "preferences": {
            "color": "blue",
            "theme": "dark"
        }
    }
}

# Render with nested data
profile = template.render(**user_data)
# Result:
# Name: Alice
# Age: 30
# Favorite Color: blue
```

### Loading Templates from Files

```python
# JSON file with multiple templates (templates.json):
# {
#   "greeting": {
#     "template": "Hello {name}, welcome to {service}!",
#     "input_variables": ["name", "service"]
#   },
#   "farewell": "Goodbye {name}, thank you for using {service}!"
# }

# Load the template loader
loader = PromptTemplateLoader("templates/")

# Load a template by ID
greeting_template = await loader.load_template("greeting")

# Render the template
result = greeting_template.render(name="Alice", service="AI Platform")
# Result: "Hello Alice, welcome to AI Platform!"
```

### Using the Global Loader

```python
# In application startup:
app_config = {"template_path": "templates/"}

# In a module that needs templates:
async def generate_greeting(name: str, service: str) -> str:
    # Get the global loader
    loader = await get_prompt_loader()
    
    # Load the template
    template = await loader.load_template("greeting")
    if not template:
        return f"Hello {name}"  # Fallback
    
    # Render the template
    return template.render(name=name, service=service)
```

## Best Practices

### Template Organization

1. **Logical Grouping**: Organize templates by functional area or agent type
2. **File Structure**: Use a consistent file naming convention (e.g., `{agent}_{purpose}.json`)
3. **Hierarchy**: For large systems, use subdirectories to organize templates by category
4. **Version Control**: Store templates in version control alongside code

### Template Design

1. **Atomicity**: Each template should have a single, focused purpose
2. **Variable Naming**: Use descriptive variable names that reflect their content
3. **Documentation**: Include comments in template files explaining expected variables
4. **Consistency**: Maintain consistent style and terminology across templates
5. **Validation**: Validate templates during testing to catch syntax errors early

### Error Handling

1. **Missing Templates**: Always handle the case where a template might not exist
2. **Missing Variables**: Validate that all required variables are provided before rendering
3. **Graceful Degradation**: Provide fallback templates or default values when possible
4. **Logging**: Log template rendering errors with sufficient context for debugging

### Performance Considerations

1. **Caching**: Leverage the template cache for frequently used templates
2. **Lazy Loading**: Only load templates when needed
3. **Template Size**: Keep templates concise and focused for better performance
4. **Benchmarking**: Measure template rendering time for performance-critical paths

## Testing Approach

### Test Structure

The test suite for the Prompt Template System is organized into the following components:

1. **Unit Tests**: Testing individual classes and methods
   - `TestPromptTemplate`: Tests for the PromptTemplate class
   - `TestPromptTemplateLoader`: Tests for the PromptTemplateLoader class
   - `TestGlobalLoader`: Tests for the global accessor functions

2. **Integration Tests**: Testing interactions between components
   - `TestTemplateIntegration`: End-to-end tests for template loading and rendering

### Key Test Fixtures

1. **Template Files**: Temporary directories with test template files
2. **Mock Template Loaders**: For testing template loading behavior
3. **Sample Templates**: Various template formats for testing rendering

### Running Tests

Tests can be run using pytest:

```bash
# Run all tests
pytest tests/prompts/

# Run specific test classes
pytest tests/prompts/test_prompts.py::TestPromptTemplate

# Run with verbose output
pytest -v tests/prompts/
```

### Mock Objects

1. **File System Mocks**: Using `patch` and `mock_open` to simulate file system operations
2. **Template Loader Mocks**: Mocking the loader's file reading methods for predictable test behavior
3. **Logger Mocks**: Intercepting logging calls to verify error handling

### Test Environment Setup

For integration tests, a temporary directory structure is created:

```python
@pytest.fixture
def setup_template_files(self, tmp_path):
    # Create a temporary directory structure
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    
    # Create test template files
    templates_json = templates_dir / "templates.json"
    templates_json.write_text(json.dumps({
        "greeting": {
            "template": "Hello {name}, welcome to {service}!",
            "input_variables": ["name", "service"]
        }
    }))
    
    return str(templates_dir)
```

## Implementation Notes

### Design Decisions

1. **Asynchronous Loading**: Template loading is implemented with async/await to support non-blocking I/O
2. **Custom Formatter**: A custom string formatter was implemented to support dot notation for dictionaries
3. **Caching Strategy**: Templates are cached by ID for efficient reuse
4. **Singleton Loader**: A global loader instance is provided for convenience and consistency

### Critical Fixes

1. **Template Validation**: Improved brace validation logic to correctly handle nested and escaped braces
2. **Dot Notation Access**: Implemented custom formatter to support accessing nested dictionary keys
3. **File Loading Implementation**: Added comprehensive file loading from multiple sources and formats
4. **Error Message Formatting**: Fixed syntax errors in f-string formatting
5. **Singleton Pattern**: Corrected thread-safety issues in the global loader initialization

### Asynchronous Programming Patterns

1. **Async File I/O**: File operations are performed in a thread pool to prevent blocking
2. **Lock for Singleton**: An asyncio.Lock ensures thread-safe initialization of the global loader
3. **Async Factory Function**: The global loader accessor is async to support non-blocking initialization

### Resource Management

1. **Efficient Caching**: Templates are cached to minimize file system operations
2. **Lazy Loading**: Templates are only loaded when requested, not at initialization
3. **Cache Invalidation**: A reload function is provided to clear the cache when needed

## API Reference

### PromptTemplate

#### `__init__(template_id: str, template_string: str, input_variables: Optional[List[str]]=None)`

Creates a new template instance.

- **Parameters**:
  - `template_id`: Unique identifier for the template
  - `template_string`: The template string with {variable} placeholders
  - `input_variables`: Optional list of variable names required by the template
- **Raises**: None

#### `render(**kwargs: Any) -> str`

Renders the template with the provided variables.

- **Parameters**:
  - `**kwargs`: Variables to substitute in the template
- **Returns**: The rendered template string
- **Raises**:
  - `KeyError`: If a required variable is missing
  - `ValueError`: If the template format is invalid

### PromptTemplateLoader

#### `__init__(template_source: str)`

Creates a new template loader.

- **Parameters**:
  - `template_source`: Path to the template source (file or directory)
- **Raises**: None

#### `async load_template(template_id: str) -> Optional[PromptTemplate]`

Loads a template by its ID, either from cache or from the source.

- **Parameters**:
  - `template_id`: The unique identifier for the template
- **Returns**: The loaded template, or None if not found
- **Raises**: None (errors are logged but not propagated)

### Global Functions

#### `async get_prompt_loader(template_source: Optional[str]=None) -> PromptTemplateLoader`

Gets or creates the global PromptTemplateLoader instance.

- **Parameters**:
  - `template_source`: Optional path to the template source
- **Returns**: The global PromptTemplateLoader instance
- **Raises**:
  - `RuntimeError`: If the loader cannot be created

#### `async reload_templates() -> None`

Clears the template cache to force reloading from source.

- **Parameters**: None
- **Returns**: None
- **Raises**: None

## Integration Guidelines

### Initialization

The template system should be initialized during application startup:

```python
async def startup_event():
    # Configure the template loader
    template_source = os.getenv("PROMPT_TEMPLATE_PATH", "prompts/")
    loader = await get_prompt_loader(template_source)
    logger.info(f"Prompt template system initialized with source: {template_source}")
```

### Configuration Options

The template system can be configured through:

1. **Environment Variables**:
   - `PROMPT_TEMPLATE_PATH`: Path to template files

2. **Direct Configuration**:
   - Passing a template source to `get_prompt_loader()`
   - Creating a custom PromptTemplateLoader instance

### Dependency Management

The template system has the following dependencies:

1. **Logger**: Requires the application's logging system
2. **File System Access**: Requires read access to template files
3. **Asyncio**: Requires an asyncio event loop for async operations

### Resource Lifecycle Management

1. **Initialization**: The template loader should be initialized during application startup
2. **Cache Management**: The template cache can be cleared with `reload_templates()`
3. **Shutdown**: No special shutdown procedures are required

## Key Improvements

### Bug Fixes

1. **Template Validation**: Completely rewrote brace validation logic to properly handle escaped braces and nesting
2. **Error Message Formatting**: Corrected f-string syntax error in error messages
3. **Import Organization**: Fixed import order to prevent module loading issues

### Feature Enhancements

1. **File Loading**: Implemented comprehensive file loading from JSON and TXT files
2. **Dictionary Access**: Added support for dot notation access to nested dictionaries
3. **Template Caching**: Enhanced caching mechanism with proper invalidation
4. **Template Formats**: Added support for multiple template formats and structures

### Code Quality Improvements

1. **Documentation**: Added comprehensive docstrings and comments
2. **Error Handling**: Enhanced error reporting and recovery
3. **Type Annotations**: Added consistent type hints throughout the codebase
4. **Testability**: Improved code structure for better testability

### Performance Optimizations

1. **Efficient Parsing**: Optimized template parsing for faster initialization
2. **Caching Strategy**: Implemented efficient template caching to reduce file I/O
3. **Async Operations**: Added asynchronous loading to prevent blocking
4. **Resource Reuse**: Implemented singleton pattern for resource efficiency
"""Unit tests for the tools base module."""

import asyncio
import pytest
from pydantic import BaseModel, Field

from src.config.errors import ToolError
from src.tools.base import BaseTool, DynamicTool


class TestArgs(BaseModel):
    """Test argument schema."""
    
    input_string: str = Field(..., description="A test input string")
    input_int: int = Field(42, description="A test input integer")


class TestTool(BaseTool):
    """Test tool implementation."""
    
    @property
    def name(self) -> str:
        return "test_tool"
    
    @property
    def description(self) -> str:
        return "A test tool"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestArgs
    
    def _run(self, **kwargs):
        input_string = kwargs["input_string"]
        input_int = kwargs["input_int"]
        return f"{input_string}-{input_int}"
    
    async def _arun(self, **kwargs):
        input_string = kwargs["input_string"]
        input_int = kwargs["input_int"]
        return f"{input_string}-{input_int}"


class TestErrorTool(BaseTool):
    """Test tool that raises errors."""
    
    @property
    def name(self) -> str:
        return "error_tool"
    
    @property
    def description(self) -> str:
        return "A tool that raises errors"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestArgs
    
    def _run(self, **kwargs):
        raise ValueError("Test error")
    
    async def _arun(self, **kwargs):
        raise ValueError("Test error")


def test_base_tool_run():
    """Test synchronous tool execution."""
    tool = TestTool()
    result = tool.run(input_string="hello", input_int=123)
    assert result == "hello-123"


@pytest.mark.asyncio
async def test_base_tool_arun():
    """Test asynchronous tool execution."""
    tool = TestTool()
    result = await tool.arun(input_string="hello", input_int=123)
    assert result == "hello-123"


def test_base_tool_run_validation_error():
    """Test tool execution with validation error."""
    tool = TestTool()
    with pytest.raises(ToolError) as excinfo:
        tool.run(input_int=123)  # Missing required input_string
    
    assert "Invalid arguments for tool" in str(excinfo.value)


def test_base_tool_run_execution_error():
    """Test tool execution with execution error."""
    tool = TestErrorTool()
    with pytest.raises(ToolError) as excinfo:
        tool.run(input_string="hello", input_int=123)
    
    assert "Error executing tool" in str(excinfo.value)


@pytest.mark.asyncio
async def test_base_tool_arun_execution_error():
    """Test async tool execution with execution error."""
    tool = TestErrorTool()
    with pytest.raises(ToolError) as excinfo:
        await tool.arun(input_string="hello", input_int=123)
    
    assert "Error executing tool" in str(excinfo.value)


def test_base_tool_empty_args_schema():
    """Test the empty args schema utility."""
    empty_schema = BaseTool.get_empty_args_schema("TestEmptySchema")
    assert empty_schema.__name__ == "TestEmptySchema"
    
    # Should be able to instantiate with no args
    instance = empty_schema()
    assert instance is not None


def test_dynamic_tool_creation():
    """Test creating a dynamic tool from functions."""
    def sync_func(x: int, y: str = "default") -> str:
        return f"{y}-{x}"
    
    async def async_func(x: int, y: str = "default") -> str:
        return f"{y}-{x}"
    
    # Create dynamic tool
    tool = DynamicTool(
        name="dynamic_test",
        description="A dynamic test tool",
        func=sync_func,
        coroutine=async_func
    )
    
    # Check properties
    assert tool.name == "dynamic_test"
    assert tool.description == "A dynamic test tool"
    
    # Check schema inference
    assert hasattr(tool.args_schema, "__fields__")
    assert "x" in tool.args_schema.__fields__
    assert "y" in tool.args_schema.__fields__
    
    # Test execution
    result = tool.run(x=42, y="hello")
    assert result == "hello-42"
    
    # Test with default
    result = tool.run(x=42)
    assert result == "default-42"


@pytest.mark.asyncio
async def test_dynamic_tool_async_execution():
    """Test async execution of a dynamic tool."""
    def sync_func(x: int, y: str = "default") -> str:
        return f"{y}-{x}"
    
    async def async_func(x: int, y: str = "default") -> str:
        return f"{y}-{x}:async"
    
    # Create dynamic tool with async function
    tool_with_async = DynamicTool(
        name="dynamic_async",
        description="A dynamic async tool",
        func=sync_func,
        coroutine=async_func
    )
    
    result = await tool_with_async.arun(x=42, y="hello")
    assert result == "hello-42:async"
    
    # Create tool without async function (should wrap sync function)
    tool_without_async = DynamicTool(
        name="dynamic_sync_only",
        description="A dynamic sync-only tool",
        func=sync_func,
    )
    
    result = await tool_without_async.arun(x=42, y="hello")
    assert result == "hello-42"  # Same as sync result


def test_dynamic_tool_validation():
    """Test validation in dynamic tool."""
    def test_func(x: int, y: str = "default") -> str:
        return f"{y}-{x}"
    
    tool = DynamicTool(
        name="validation_test",
        description="Validation test tool",
        func=test_func
    )
    
    # Valid input
    result = tool.run(x=42, y="hello")
    assert result == "hello-42"
    
    # Invalid input (x should be int)
    with pytest.raises(ToolError) as excinfo:
        tool.run(x="not-an-int", y="hello")
    
    assert "Invalid arguments for tool" in str(excinfo.value) 
"""Unit tests for the tool runner module."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field

from src.config.errors import ErrorCode, ToolError
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry
from src.tools.runner import ToolRunner


class TestRunnerArgs(BaseModel):
    """Test argument schema."""
    
    arg1: str = Field(..., description="Test argument 1")
    arg2: int = Field(42, description="Test argument 2")


class TestRunnerTool(BaseTool):
    """Test tool for runner tests."""
    
    @property
    def name(self) -> str:
        return "test_runner_tool"
    
    @property
    def description(self) -> str:
        return "A test tool for runner tests"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestRunnerArgs
    
    def _run(self, **kwargs):
        return f"Run: {kwargs['arg1']}-{kwargs['arg2']}"
    
    async def _arun(self, **kwargs):
        return f"Async: {kwargs['arg1']}-{kwargs['arg2']}"


class TestErrorTool(BaseTool):
    """Test tool that raises errors."""
    
    @property
    def name(self) -> str:
        return "test_error_tool"
    
    @property
    def description(self) -> str:
        return "A test tool that raises errors"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestRunnerArgs
    
    def _run(self, **kwargs):
        raise ValueError("Test run error")
    
    async def _arun(self, **kwargs):
        raise ValueError("Test async error")


class TestRetryTool(BaseTool):
    """Test tool that succeeds after retries."""
    
    def __init__(self):
        self.attempts = 0
    
    @property
    def name(self) -> str:
        return "test_retry_tool"
    
    @property
    def description(self) -> str:
        return "A test tool that succeeds after retries"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestRunnerArgs
    
    def _run(self, **kwargs):
        self.attempts += 1
        if self.attempts < 2:
            raise ValueError(f"Retry attempt {self.attempts}")
        return f"Success after {self.attempts} attempts"
    
    async def _arun(self, **kwargs):
        self.attempts += 1
        if self.attempts < 2:
            raise ValueError(f"Retry attempt {self.attempts}")
        return f"Async success after {self.attempts} attempts"


@pytest.fixture
def runner():
    """Create a fresh tool runner for each test."""
    return ToolRunner()


@pytest.fixture
def registry():
    """Create a registry with test tools."""
    registry = ToolRegistry()
    registry.register(TestRunnerTool)
    registry.register(TestErrorTool)
    registry.register(TestRetryTool)
    return registry


@pytest.mark.asyncio
async def test_run_tool_with_instance(runner):
    """Test running a tool using an instance."""
    tool = TestRunnerTool()
    result = await runner.run_tool(tool, args={"arg1": "test"})
    
    assert result["status"] == "success"
    assert result["tool_name"] == "test_runner_tool"
    assert result["result"] == "Async: test-42"
    assert "execution_time" in result


@pytest.mark.asyncio
async def test_run_tool_with_name(runner, registry):
    """Test running a tool using its name."""
    result = await runner.run_tool("test_runner_tool", registry, args={"arg1": "test"})
    
    assert result["status"] == "success"
    assert result["tool_name"] == "test_runner_tool"
    assert result["result"] == "Async: test-42"


@pytest.mark.asyncio
async def test_run_tool_with_error(runner):
    """Test running a tool that raises an error."""
    tool = TestErrorTool()
    result = await runner.run_tool(tool, args={"arg1": "test"})
    
    assert result["status"] == "error"
    assert result["tool_name"] == "test_error_tool"
    assert "error" in result
    assert result["error"]["code"] == str(ErrorCode.TOOL_EXECUTION_ERROR)


@pytest.mark.asyncio
async def test_run_tool_with_retry(runner):
    """Test running a tool that succeeds after retry."""
    tool = TestRetryTool()
    result = await runner.run_tool(tool, args={"arg1": "test"}, retry_count=2)
    
    assert result["status"] == "success"
    assert result["tool_name"] == "test_retry_tool"
    assert "Async success after" in result["result"]


@pytest.mark.asyncio
async def test_run_tool_with_retry_failure(runner):
    """Test running a tool that fails despite retries."""
    tool = TestErrorTool()
    result = await runner.run_tool(tool, args={"arg1": "test"}, retry_count=1)
    
    assert result["status"] == "error"
    assert result["tool_name"] == "test_error_tool"


@pytest.mark.asyncio
async def test_run_tool_with_validation_error(runner):
    """Test running a tool with invalid arguments."""
    tool = TestRunnerTool()
    result = await runner.run_tool(tool, args={})  # Missing required arg1
    
    assert result["status"] == "error"
    assert result["tool_name"] == "test_runner_tool"
    assert "error" in result
    assert result["error"]["code"] == str(ErrorCode.TOOL_VALIDATION_ERROR)


@pytest.mark.asyncio
async def test_run_tool_nonexistent(runner, registry):
    """Test running a non-existent tool."""
    with pytest.raises(ToolError) as excinfo:
        await runner.run_tool("nonexistent_tool", registry)
    
    assert "could not be resolved" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_tools_parallel(runner, registry):
    """Test running multiple tools in parallel."""
    tools = [
        {"name": "test_runner_tool", "args": {"arg1": "test1"}},
        {"name": "test_runner_tool", "args": {"arg1": "test2"}},
    ]
    
    results = await runner.run_tools_parallel(tools, registry)
    
    assert len(results) == 2
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "success"
    assert "test1" in results[0]["result"]
    assert "test2" in results[1]["result"]


@pytest.mark.asyncio
async def test_run_tools_parallel_with_error(runner, registry):
    """Test running multiple tools in parallel with one error."""
    tools = [
        {"name": "test_runner_tool", "args": {"arg1": "test"}},
        {"name": "test_error_tool", "args": {"arg1": "test"}},
    ]
    
    results = await runner.run_tools_parallel(tools, registry)
    
    assert len(results) == 2
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "error"


@pytest.mark.asyncio
async def test_run_tools_parallel_timeout(runner, registry):
    """Test running tools in parallel with timeout."""
    # Create a slow tool
    class SlowTool(BaseTool):
        @property
        def name(self) -> str:
            return "slow_tool"
        
        @property
        def description(self) -> str:
            return "A slow tool"
        
        @property
        def args_schema(self) -> type[BaseModel]:
            return TestRunnerArgs
        
        def _run(self, **kwargs):
            return "Slow run"
        
        async def _arun(self, **kwargs):
            await asyncio.sleep(0.5)  # Simulate slow operation
            return "Slow async"
    
    # Mock the sleep to avoid actual delays in tests
    original_sleep = asyncio.sleep
    
    # Create patched sleep that only delays for the slow tool
    async def patched_sleep(delay):
        if delay == 0.5:  # This is our slow tool
            await original_sleep(0.1)  # Small delay to ensure proper test flow
        else:
            # Other sleeps (like backoff) proceed normally but faster
            await original_sleep(0.01)
    
    with patch('asyncio.sleep', side_effect=patched_sleep):
        # Register the slow tool
        registry.register(SlowTool)
        
        tools = [
            {"name": "slow_tool", "args": {"arg1": "test"}},
        ]
        
        # Run with very short timeout
        results = await runner.run_tools_parallel(tools, registry, timeout=0.05)
        
        # Should timeout
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "timeout" in results[0]["error"]["message"].lower()


@pytest.mark.asyncio
async def test_format_result(runner):
    """Test the result formatting."""
    # Test different result types
    string_result = runner._format_result("string result", "test_tool", 0.1)
    assert string_result["result"] == "string result"
    assert string_result["result_type"] == "str"
    
    dict_result = runner._format_result({"key": "value"}, "test_tool", 0.1)
    assert dict_result["result"] == {"key": "value"}
    assert dict_result["result_type"] == "dict"
    
    # Test object result
    class TestObject:
        def __str__(self):
            return "test_object_str"
    
    obj_result = runner._format_result(TestObject(), "test_tool", 0.1)
    assert obj_result["result"] == "test_object_str"
    assert obj_result["result_type"] == "TestObject"
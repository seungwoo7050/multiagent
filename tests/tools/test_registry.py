"""Unit tests for the tools registry module."""

import pytest
from pydantic import BaseModel, Field

from src.config.errors import ToolError
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry, register_tool


class TestRegistryArgs(BaseModel):
    """Test argument schema."""
    
    arg1: str = Field(..., description="Test argument 1")


class TestRegistryTool(BaseTool):
    """Test tool for registry tests."""
    
    @property
    def name(self) -> str:
        return "test_registry_tool"
    
    @property
    def description(self) -> str:
        return "A test tool for registry tests"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestRegistryArgs
    
    def _run(self, **kwargs):
        return f"Executed: {kwargs['arg1']}"
    
    async def _arun(self, **kwargs):
        return f"Executed async: {kwargs['arg1']}"


class TestRegistryTool2(BaseTool):
    """Second test tool for registry tests."""
    
    @property
    def name(self) -> str:
        return "test_registry_tool_2"
    
    @property
    def description(self) -> str:
        return "A second test tool for registry tests"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestRegistryArgs
    
    def _run(self, **kwargs):
        return f"Tool 2: {kwargs['arg1']}"
    
    async def _arun(self, **kwargs):
        return f"Tool 2 async: {kwargs['arg1']}"


class DuplicateNameTool(BaseTool):
    """Test tool with duplicate name."""
    
    @property
    def name(self) -> str:
        return "test_registry_tool"  # Same as TestRegistryTool
    
    @property
    def description(self) -> str:
        return "A tool with duplicate name"
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return TestRegistryArgs
    
    def _run(self, **kwargs):
        return "Duplicate"
    
    async def _arun(self, **kwargs):
        return "Duplicate async"


class InvalidTool:
    """Invalid tool that doesn't extend BaseTool."""
    
    def run(self):
        return "Invalid"


def test_registry_initialization():
    """Test registry initialization."""
    registry = ToolRegistry()
    assert isinstance(registry, ToolRegistry)
    assert len(registry.get_names()) == 0


def test_registry_register_tool():
    """Test registering a tool with the registry."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    
    assert "test_registry_tool" in registry.get_names()
    assert len(registry.get_names()) == 1


def test_registry_register_multiple_tools():
    """Test registering multiple tools."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    registry.register(TestRegistryTool2)
    
    assert "test_registry_tool" in registry.get_names()
    assert "test_registry_tool_2" in registry.get_names()
    assert len(registry.get_names()) == 2


def test_registry_register_duplicate_tool():
    """Test registering a tool with a duplicate name."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    
    with pytest.raises(ToolError) as excinfo:
        registry.register(DuplicateNameTool)
    
    assert "is already registered" in str(excinfo.value)


def test_registry_register_invalid_tool():
    """Test registering an invalid tool."""
    registry = ToolRegistry()
    
    with pytest.raises(ToolError) as excinfo:
        registry.register(InvalidTool)
    
    assert "must inherit from BaseTool" in str(excinfo.value)


def test_registry_get_tool_class():
    """Test getting a tool class from the registry."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    
    tool_cls = registry.get_tool_class("test_registry_tool")
    assert tool_cls is TestRegistryTool


def test_registry_get_nonexistent_tool_class():
    """Test getting a non-existent tool class."""
    registry = ToolRegistry()
    
    with pytest.raises(ToolError) as excinfo:
        registry.get_tool_class("nonexistent_tool")
    
    assert "not found" in str(excinfo.value)


def test_registry_get_tool():
    """Test getting a tool instance from the registry."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    
    tool = registry.get_tool("test_registry_tool")
    assert isinstance(tool, TestRegistryTool)
    assert tool.name == "test_registry_tool"


def test_registry_get_tool_caching():
    """Test that tool instances are cached."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    
    tool1 = registry.get_tool("test_registry_tool")
    tool2 = registry.get_tool("test_registry_tool")
    
    assert tool1 is tool2  # Same instance, not just equal


def test_registry_clear_cache():
    """Test clearing the tool instance cache."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    
    tool1 = registry.get_tool("test_registry_tool")
    registry.clear_cache()
    tool2 = registry.get_tool("test_registry_tool")
    
    assert tool1 is not tool2  # Different instances after cache clear


def test_registry_unregister():
    """Test unregistering a tool."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    registry.register(TestRegistryTool2)
    
    assert len(registry.get_names()) == 2
    
    registry.unregister("test_registry_tool")
    assert len(registry.get_names()) == 1
    assert "test_registry_tool" not in registry.get_names()
    assert "test_registry_tool_2" in registry.get_names()


def test_registry_unregister_nonexistent():
    """Test unregistering a non-existent tool."""
    registry = ToolRegistry()
    
    with pytest.raises(ToolError) as excinfo:
        registry.unregister("nonexistent_tool")
    
    assert "not found for unregistration" in str(excinfo.value)


def test_registry_list_tools():
    """Test listing all registered tools."""
    registry = ToolRegistry()
    registry.register(TestRegistryTool)
    registry.register(TestRegistryTool2)
    
    tool_list = registry.list_tools()
    assert len(tool_list) == 2
    
    # Check tool information
    tool_names = [t["name"] for t in tool_list]
    assert "test_registry_tool" in tool_names
    assert "test_registry_tool_2" in tool_names
    
    # Check schema information is included
    for tool in tool_list:
        assert "schema" in tool
        assert "properties" in tool["schema"]


def test_registry_decorator():
    """Test the register_tool decorator."""
    test_registry = ToolRegistry()
    
    @register_tool(registry=test_registry)
    class DecoratedTool(BaseTool):
        @property
        def name(self) -> str:
            return "decorated_tool"
        
        @property
        def description(self) -> str:
            return "A tool registered with decorator"
        
        @property
        def args_schema(self) -> type[BaseModel]:
            return TestRegistryArgs
        
        def _run(self, **kwargs):
            return "Decorated run"
        
        async def _arun(self, **kwargs):
            return "Decorated arun"
    
    assert "decorated_tool" in test_registry.get_names()
    tool = test_registry.get_tool("decorated_tool")
    assert isinstance(tool, DecoratedTool)
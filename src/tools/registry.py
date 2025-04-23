"""
Tool Registry - High-Performance Implementation.

This module implements a registry for tools with O(1) lookup performance.
It provides a decorator for automatic tool registration and fast retrieval.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, cast, get_type_hints

from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.config.metrics import CACHE_OPERATIONS_TOTAL, CACHE_SIZE
from src.tools.base import BaseTool

logger = get_logger(__name__)

# Type for the decorator
T = TypeVar("T", bound=Type[BaseTool])


class ToolRegistry:
    """
    Registry for tool classes with O(1) lookup.
    
    This registry is designed for fast lookups and tool discovery.
    It validates tools on registration to ensure they meet the interface.
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        # Main storage: tool_name -> tool_class
        self._tools: Dict[str, Type[BaseTool]] = {}
        
        # Cache for instantiated tools
        self._instance_cache: Dict[str, BaseTool] = {}
        
        # Register self as a metric target
        CACHE_SIZE.labels(cache_type="tool_registry").set(0)
    
    def register(self, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """
        Register a tool class with the registry.
        
        Args:
            tool_cls: The tool class to register.
            
        Returns:
            The registered tool class (unchanged).
            
        Raises:
            ToolError: If registration fails validation.
        """
        # Create a temporary instance to get the name
        try:
            # Verify it's a BaseTool subclass
            if not issubclass(tool_cls, BaseTool):
                raise ToolError(
                    code=ErrorCode.TOOL_VALIDATION_ERROR,
                    message=f"Tool class {tool_cls.__name__} must inherit from BaseTool",
                    details={"class": tool_cls.__name__}
                )
            
            # Create temporary instance to validate interface
            tool_instance = tool_cls()
            tool_name = tool_instance.name
            
            # Check name not already registered
            if tool_name in self._tools:
                raise ToolError(
                    code=ErrorCode.TOOL_VALIDATION_ERROR,
                    message=f"Tool name '{tool_name}' is already registered",
                    details={"name": tool_name, "existing_class": self._tools[tool_name].__name__}
                )
            
            # Register the tool class
            self._tools[tool_name] = tool_cls
            
            # Update metrics
            CACHE_OPERATIONS_TOTAL.labels(operation_type="tool_registration").inc()
            CACHE_SIZE.labels(cache_type="tool_registry").set(len(self._tools))
            
            logger.info(f"Tool '{tool_name}' registered successfully", 
                       extra={"tool_name": tool_name, "tool_class": tool_cls.__name__})
            
            return tool_cls
            
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            
            # Handle other errors during registration
            logger.error(
                f"Failed to register tool class {tool_cls.__name__}",
                extra={"class": tool_cls.__name__},
                exc_info=e
            )
            
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message=f"Failed to register tool class {tool_cls.__name__}",
                details={"class": tool_cls.__name__},
                original_error=e
            )
    
    def get_tool_class(self, tool_name: str) -> Type[BaseTool]:
        """
        Get a tool class by name with O(1) lookup.
        
        Args:
            tool_name: The name of the tool to retrieve.
            
        Returns:
            The tool class.
            
        Raises:
            ToolError: If the tool is not found.
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool '{tool_name}' not found in registry", 
                          extra={"tool_name": tool_name})
            
            raise ToolError(
                code=ErrorCode.TOOL_NOT_FOUND,
                message=f"Tool '{tool_name}' not found",
                details={"name": tool_name, "available_tools": list(self._tools.keys())},
                tool_name=tool_name
            )
        
        return self._tools[tool_name]
    
    def get_tool(self, tool_name: str) -> BaseTool:
        """
        Get or create a tool instance by name with caching.
        
        Args:
            tool_name: The name of the tool to retrieve.
            
        Returns:
            An instance of the requested tool.
            
        Raises:
            ToolError: If the tool is not found or initialization fails.
        """
        # Check cache first
        if tool_name in self._instance_cache:
            CACHE_OPERATIONS_TOTAL.labels(operation_type="tool_cache_hit").inc()
            return self._instance_cache[tool_name]
        
        try:
            # Get class and instantiate
            tool_cls = self.get_tool_class(tool_name)
            tool_instance = tool_cls()
            
            # Cache the instance
            self._instance_cache[tool_name] = tool_instance
            CACHE_OPERATIONS_TOTAL.labels(operation_type="tool_cache_set").inc()
            
            return tool_instance
            
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            
            logger.error(
                f"Failed to instantiate tool '{tool_name}'",
                extra={"tool_name": tool_name},
                exc_info=e
            )
            
            raise ToolError(
                code=ErrorCode.TOOL_CREATION_ERROR,
                message=f"Failed to instantiate tool '{tool_name}'",
                details={"name": tool_name},
                original_error=e,
                tool_name=tool_name
            )
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools with their metadata.
        
        Returns:
            A list of dictionaries containing tool metadata.
        """
        result = []
        
        for tool_name, tool_cls in self._tools.items():
            try:
                # Get instance for metadata
                tool = self.get_tool(tool_name)
                
                # Extract metadata
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": self._get_schema_dict(tool.args_schema),
                }
                
                result.append(tool_info)
                
            except Exception as e:
                logger.warning(
                    f"Error getting metadata for tool '{tool_name}'",
                    extra={"tool_name": tool_name},
                    exc_info=e
                )
                
                # Add partial info
                result.append({
                    "name": tool_name,
                    "error": str(e),
                })
        
        return result
    
    def _get_schema_dict(self, schema_cls: Any) -> Dict[str, Any]:
        """
        Convert a Pydantic model class to a dictionary representation.
        
        Args:
            schema_cls: The Pydantic model class.
            
        Returns:
            A dictionary representation of the schema.
        """
        # Create an empty instance to get the schema
        schema = schema_cls.schema()
        
        # Return only the relevant parts
        return {
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
            "title": schema.get("title", ""),
            "type": schema.get("type", "object"),
        }
    
    def clear_cache(self) -> None:
        """Clear the instance cache."""
        self._instance_cache.clear()
        CACHE_OPERATIONS_TOTAL.labels(operation_type="tool_cache_clear").inc()
        logger.debug("Tool instance cache cleared")
    
    def unregister(self, tool_name: str) -> None:
        """
        Unregister a tool by name.
        
        Args:
            tool_name: The name of the tool to unregister.
            
        Raises:
            ToolError: If the tool is not found.
        """
        if tool_name not in self._tools:
            raise ToolError(
                code=ErrorCode.TOOL_NOT_FOUND,
                message=f"Tool '{tool_name}' not found for unregistration",
                details={"name": tool_name},
                tool_name=tool_name
            )
        
        # Remove from registry and cache
        del self._tools[tool_name]
        if tool_name in self._instance_cache:
            del self._instance_cache[tool_name]
        
        # Update metrics
        CACHE_OPERATIONS_TOTAL.labels(operation_type="tool_unregistration").inc()
        CACHE_SIZE.labels(cache_type="tool_registry").set(len(self._tools))
        
        logger.info(f"Tool '{tool_name}' unregistered successfully", 
                   extra={"tool_name": tool_name})
    
    def get_names(self) -> Set[str]:
        """
        Get the set of all registered tool names.
        
        Returns:
            A set of tool names.
        """
        return set(self._tools.keys())


def register_tool(registry: Optional[ToolRegistry] = None) -> Callable[[T], T]:
    """
    Decorator to register a tool class with the registry.
    
    Args:
        registry: The registry to register with. If None, a global registry
                 will be used when it becomes available.
    
    Returns:
        A decorator function that registers the tool class.
    
    Example:
        @register_tool()
        class MyTool(BaseTool):
            ...
    """
    def decorator(cls: T) -> T:
        # If registry provided, register immediately
        if registry is not None:
            return registry.register(cls)
        
        # Otherwise, we'll register it when the module is imported
        # The __init__.py will handle this with the global registry
        return cls
    
    return decorator
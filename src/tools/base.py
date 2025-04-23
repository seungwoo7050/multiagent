"""
Base Tool Interface - High-Performance Implementation.

This module defines the BaseTool interface that all tools must implement.
The interface provides both synchronous and asynchronous execution paths
with comprehensive error handling and performance tracking.
"""

import abc
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Type, Union

from pydantic import BaseModel, ValidationError, create_model

from src.config.errors import ErrorCode, ToolError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import (
    TOOL_EXECUTION_DURATION,
    TOOL_ERRORS_TOTAL,
    timed_metric,
)
from src.utils.timing import Timer, AsyncTimer

logger = get_logger(__name__)


class BaseTool(abc.ABC):
    """
    Abstract base class for all tools.
    
    Tools provide functionality that can be called by agents. Each tool
    must define synchronous and asynchronous execution methods, as well
    as metadata properties like name, description, and arg schema.
    """
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass
    
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """A human-readable description of what the tool does."""
        pass
    
    @property
    @abc.abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """The Pydantic model defining the arguments for this tool."""
        pass
    
    @abc.abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """
        Synchronous execution method.
        
        This should implement the tool's logic for synchronous execution.
        Must be implemented by all tool classes.
        
        Args:
            **kwargs: The arguments for the tool, validated against args_schema.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ToolError: If tool execution fails.
        """
        pass
    
    @abc.abstractmethod
    async def _arun(self, **kwargs: Any) -> Any:
        """
        Asynchronous execution method.
        
        This should implement the tool's logic for asynchronous execution.
        Must be implemented by all tool classes.
        
        Args:
            **kwargs: The arguments for the tool, validated against args_schema.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ToolError: If tool execution fails.
        """
        pass
    
    @timed_metric(TOOL_EXECUTION_DURATION, {"tool_name": lambda self: self.name})
    def run(self, **kwargs: Any) -> Any:
        """
        Execute the tool synchronously with validation and error handling.
        
        Args:
            **kwargs: The arguments for the tool.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ToolError: If validation or execution fails.
        """
        try:
            # Validate input against schema
            validated_args = self._validate_args(kwargs)
            
            # Execute tool with timing
            with Timer(f"tool_{self.name}_run"):
                result = self._run(**validated_args)
            
            return result
        except Exception as e:
            error = self._handle_error(e)
            raise error
    
    @timed_metric(TOOL_EXECUTION_DURATION, {"tool_name": lambda self: self.name})
    async def arun(self, **kwargs: Any) -> Any:
        """
        Execute the tool asynchronously with validation and error handling.
        
        Args:
            **kwargs: The arguments for the tool.
            
        Returns:
            The result of the tool execution.
            
        Raises:
            ToolError: If validation or execution fails.
        """
        try:
            # Validate input against schema
            validated_args = self._validate_args(kwargs)
            
            # Execute tool with timing
            async with AsyncTimer(f"tool_{self.name}_arun"):
                result = await self._arun(**validated_args)
            
            return result
        except Exception as e:
            error = self._handle_error(e)
            raise error
    
    def _validate_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool arguments against the args_schema.
        
        Args:
            kwargs: The arguments for the tool.
            
        Returns:
            Validated arguments dictionary.
            
        Raises:
            ValidationError: If validation fails.
        """
        try:
            # Create and validate the model instance
            args_instance = self.args_schema(**kwargs)
            # Convert back to dict for passing to execution methods
            return args_instance.dict()
        except ValidationError as e:
            logger.warning(
                f"Tool argument validation failed for {self.name}",
                extra={"error": str(e), "args": kwargs}
            )
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message=f"Invalid arguments for tool '{self.name}'",
                details={"errors": e.errors(), "args": kwargs},
                original_error=e,
                tool_name=self.name
            )
    
    def _handle_error(self, error: Exception) -> ToolError:
        """
        Handle and convert errors that occur during tool execution.
        
        Args:
            error: The exception that occurred.
            
        Returns:
            A standardized ToolError.
        """
        # Already a ToolError, just return it
        if isinstance(error, ToolError):
            TOOL_ERRORS_TOTAL.labels(
                tool_name=self.name,
                error_type=error.code
            ).inc()
            return error
        
        # Convert to ToolError
        error_code = ErrorCode.TOOL_EXECUTION_ERROR
        message = f"Error executing tool '{self.name}': {str(error)}"
        
        # Log the error
        logger.error(
            message,
            extra={"tool_name": self.name, "error_type": type(error).__name__},
            exc_info=error
        )
        
        # Track metric
        TOOL_ERRORS_TOTAL.labels(
            tool_name=self.name,
            error_type=error_code
        ).inc()
        
        # Convert and return
        return ToolError(
            code=error_code,
            message=message,
            details={"tool_name": self.name, "error_type": type(error).__name__},
            original_error=error,
            tool_name=self.name
        )
    
    @classmethod
    def get_empty_args_schema(cls, name: str = "EmptyArgsSchema") -> Type[BaseModel]:
        """
        Create an empty Pydantic model for tools with no arguments.
        
        Args:
            name: The name for the empty schema class.
            
        Returns:
            An empty Pydantic model class.
        """
        return create_model(name, __base__=BaseModel)


class DynamicTool(BaseTool):
    """
    A tool that can be created dynamically from functions.
    
    This allows creating tools from existing functions without
    having to create a new class for each tool.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        args_schema: Optional[Type[BaseModel]] = None,
    ):
        """
        Initialize a dynamic tool.
        
        Args:
            name: The unique name of the tool.
            description: A human-readable description of what the tool does.
            func: The synchronous function to execute.
            coroutine: The asynchronous function to execute. If None, a wrapper
                       around the synchronous function will be created.
            args_schema: The Pydantic model defining the arguments. If None,
                         an attempt will be made to infer it from function signatures.
        """
        self._name = name
        self._description = description
        self._func = func
        self._coroutine = coroutine
        self._args_schema = args_schema or self._infer_args_schema()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def args_schema(self) -> Type[BaseModel]:
        return self._args_schema
    
    def _run(self, **kwargs: Any) -> Any:
        return self._func(**kwargs)
    
    async def _arun(self, **kwargs: Any) -> Any:
        if self._coroutine:
            return await self._coroutine(**kwargs)
        # If no coroutine was provided, wrap the sync function
        return self._func(**kwargs)
    
    def _infer_args_schema(self) -> Type[BaseModel]:
        """
        Attempt to infer a Pydantic model from the function signature.
        
        Returns:
            A Pydantic model matching the function's parameters.
        """
        # Get function signature
        sig = inspect.signature(self._func)
        
        # Create field definitions for model
        fields = {}
        for name, param in sig.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            
            # Add field with annotation if available, or Any
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = param.default if param.default != inspect.Parameter.empty else ...
            
            fields[name] = (annotation, default)
        
        # Create model class name
        model_name = f"{self.name.title().replace(' ', '')}Schema"
        
        # Create and return model
        return create_model(model_name, **fields)
import abc
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Type, Union
from pydantic import BaseModel, ValidationError, create_model
from src.config.errors import ErrorCode, ToolError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import TOOL_METRICS, timed_metric
from src.utils.timing import Timer, AsyncTimer

logger = get_logger(__name__)

class BaseTool(abc.ABC):
    """Abstract base class for all tools."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the tool."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Get the description of the tool."""
        pass

    @property
    @abc.abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """Get the pydantic schema for the tool's arguments."""
        pass

    @abc.abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Synchronous execution of the tool."""
        pass

    @abc.abstractmethod
    async def _arun(self, **kwargs: Any) -> Any:
        """Asynchronous execution of the tool."""
        pass

    @timed_metric(TOOL_METRICS['duration'], {'tool_name': lambda self: self.name})
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously."""
        try:
            validated_args = self._validate_args(kwargs)
            result = self._run(**validated_args)
            return result
        except Exception as e:
            error = self._handle_error(e)
            raise error

    @timed_metric(TOOL_METRICS['duration'], {'tool_name': lambda self: self.name})
    async def arun(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously."""
        try:
            validated_args = self._validate_args(kwargs)
            result = await self._arun(**validated_args)
            return result
        except Exception as e:
            error = self._handle_error(e)
            raise error

    def _validate_args(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate arguments against the schema."""
        try:
            args_instance = self.args_schema(**kwargs)
            return args_instance.dict()
        except ValidationError as e:
            logger.warning(
                f'Tool argument validation failed for {self.name}', 
                extra={'error': str(e), 'args': kwargs, 'tool_name': self.name}
            )
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR, 
                message=f"Invalid arguments for tool '{self.name}'", 
                details={'errors': e.errors(), 'args': kwargs}, 
                original_error=e, 
                tool_name=self.name
            )

    def _handle_error(self, error: Exception) -> ToolError:
        """Handle tool execution errors."""
        if isinstance(error, ToolError):
            TOOL_METRICS['errors'].labels(tool_name=self.name, error_type=str(error.code)).inc()
            return error
            
        error_code = ErrorCode.TOOL_EXECUTION_ERROR
        message = f"Error executing tool '{self.name}': {str(error)}"
        logger.error(
            message, 
            extra={'tool_name': self.name, 'error_type': type(error).__name__}, 
            exc_info=error
        )
        TOOL_METRICS['errors'].labels(tool_name=self.name, error_type=str(error_code)).inc()
        return ToolError(
            code=error_code, 
            message=message, 
            details={'tool_name': self.name, 'error_type': type(error).__name__}, 
            original_error=error, 
            tool_name=self.name
        )

    @classmethod
    def get_empty_args_schema(cls, name: str='EmptyArgsSchema') -> Type[BaseModel]:
        """Create an empty argument schema."""
        return create_model(name, __base__=BaseModel)

class DynamicTool(BaseTool):
    """Dynamically created tool from a function."""

    def __init__(
        self, 
        name: str, 
        description: str, 
        func: Callable[..., Any], 
        coroutine: Optional[Callable[..., Awaitable[Any]]]=None, 
        args_schema: Optional[Type[BaseModel]]=None
    ):
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
        logger.debug(f"Executing synchronous function '{self._func.__name__}' for async call in DynamicTool '{self.name}'.")
        return self._func(**kwargs)

    def _infer_args_schema(self) -> Type[BaseModel]:
        """Infer a pydantic schema from the function signature."""
        logger.debug(f"Inferring args schema for dynamic tool '{self.name}' from function '{self._func.__name__}'")
        try:
            sig = inspect.signature(self._func)
            fields: Dict[str, Tuple[Type, Any]] = {}
            
            for name, param in sig.parameters.items():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                    
                annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
                default = param.default if param.default != inspect.Parameter.empty else ...
                
                fields[name] = (annotation, default)
                
            model_name = f'{self.name.title().replace("_", "").replace(" ", "")}Schema'
            inferred_schema = create_model(model_name, **fields)
            
            logger.debug(f"Inferred schema for '{self.name}': {inferred_schema.schema()}")
            return inferred_schema
            
        except Exception as e:
            logger.warning(
                f"Could not infer args_schema for tool '{self.name}': {e}. Using empty schema.", 
                exc_info=True
            )
            return self.get_empty_args_schema(f'{self.name.title().replace(" ", "")}Schema')
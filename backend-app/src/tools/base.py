import abc
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Type

from pydantic import BaseModel, create_model
from langchain_core.tools import BaseTool as LangchainBaseTool

from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_default_args_schema(name: str = "DefaultToolInputSchema") -> Type[BaseModel]:
    """Creates a default empty Pydantic schema if none is provided."""
    return create_model(name, __base__=BaseModel)


class BaseTool(LangchainBaseTool, abc.ABC):
    """
    Abstract base class for all tools, inheriting from LangChain's BaseTool.

    Ensures tools have a name, description, and an argument schema (args_schema).
    Subclasses must implement the _run or _arun method for execution logic.

    Attributes:
        name (str): The unique name of the tool. Used to identify the tool.
        description (str): A detailed description of what the tool does,
                           its parameters, and when to use it. Crucial for LLM agent tool selection.
        args_schema (Type[BaseModel]): Pydantic model defining the arguments
                                       this tool accepts. Used for validation and guiding LLMs.
    """

    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None

    return_direct: bool = False

    @abc.abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the tool's logic synchronously. MUST be implemented by subclasses.
        Use Pydantic models for input validation within this method if complex validation is needed beyond the schema.
        """
        raise NotImplementedError(f"{self.__class__.__name__}._run not implemented")

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the tool's logic asynchronously. Override if the tool involves async I/O.
        The default implementation runs the sync version in a thread pool.
        Use Pydantic models for input validation within this method if complex validation is needed beyond the schema.
        """

        logger.debug(
            f"Default _arun called for {self.name}, will likely execute _run in executor."
        )

    @classmethod
    def get_empty_args_schema(cls, name: str = "EmptyArgsSchema") -> Type[BaseModel]:
        """Create an empty argument schema."""
        return create_model(name, __base__=BaseModel)

    class Config:
        """Pydantic V1 style config."""

        arbitrary_types_allowed = True


class DynamicTool(BaseTool):
    """
    Dynamically created tool from a function.
    Inherits from the updated BaseTool (which inherits from Langchain's BaseTool).
    """

    func: Callable[..., Any]
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ):
        inferred_schema = args_schema or self._infer_args_schema(func, name)
        super().__init__(
            name=name,
            description=description,
            args_schema=inferred_schema,
            func=func,
            coroutine=coroutine,
            **kwargs,
        )

        self.func = func
        self.coroutine = coroutine

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Executes the synchronous function."""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            logger.error(
                f"Error executing dynamic tool '{self.name}': {e}", exc_info=True
            )

            raise ToolError(
                message=f"Error in tool '{self.name}': {str(e)}",
                tool_name=self.name,
                original_error=e,
                code=ErrorCode.TOOL_EXECUTION_ERROR,
            )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Executes the asynchronous function if available, otherwise the sync function."""
        try:
            if self.coroutine:
                result = await self.coroutine(**kwargs)
                return str(result)
            else:
                logger.debug(
                    f"Executing synchronous function '{self.func.__name__}' for async call in DynamicTool '{self.name}'."
                )

                result = self._run(*args, **kwargs)
                return result
        except Exception as e:
            if not isinstance(e, ToolError):
                logger.error(
                    f"Error executing async dynamic tool '{self.name}': {e}",
                    exc_info=True,
                )
                raise ToolError(
                    message=f"Error in async tool '{self.name}': {str(e)}",
                    tool_name=self.name,
                    original_error=e,
                    code=ErrorCode.TOOL_EXECUTION_ERROR,
                )
            else:
                raise

    @staticmethod
    def _infer_args_schema(func: Callable[..., Any], name: str) -> Type[BaseModel]:
        """Infer a pydantic schema from the function signature."""
        logger.debug(
            f"Inferring args schema for dynamic tool '{name}' from function '{func.__name__}'"
        )
        try:
            sig = inspect.signature(func)
            fields: Dict[str, Tuple[Type, Any]] = {}

            for param_name, param in sig.parameters.items():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue

                annotation = (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else Any
                )
                default = (
                    param.default if param.default != inspect.Parameter.empty else ...
                )

                fields[param_name] = (annotation, default)

            model_name = f"{name.title().replace('_', '').replace(' ', '')}Schema"
            inferred_schema = create_model(model_name, **fields)

            logger.debug(f"Inferred schema for '{name}': {inferred_schema.schema()}")
            return inferred_schema

        except Exception as e:
            logger.warning(
                f"Could not infer args_schema for tool '{name}': {e}. Using empty schema.",
                exc_info=True,
            )

            return create_model(
                f"{name.title().replace(' ', '')}Schema", __base__=BaseModel
            )

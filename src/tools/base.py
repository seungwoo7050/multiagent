import abc
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Type, List, Union, ClassVar

# Pydantic과 LangChain Core 의존성 추가
from pydantic import BaseModel, Field, ValidationError, create_model
from langchain_core.tools import BaseTool as LangchainBaseTool # 이름 충돌 방지

from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger
# from src.config.metrics import TOOL_METRICS, timed_metric # BaseTool에서 제거, 필요시 개별 도구에 적용

logger = get_logger(__name__)

# 빈 Pydantic 모델 생성 함수 (기존 유지)
def _get_default_args_schema(name: str = 'DefaultToolInputSchema') -> Type[BaseModel]:
    """Creates a default empty Pydantic schema if none is provided."""
    return create_model(name, __base__=BaseModel)

# BaseTool 정의 (LangChain BaseTool 상속)
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

    # LangChain BaseTool에서 name과 description은 필수 필드이므로 @property 제거
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    args_schema: ClassVar[Optional[Type[BaseModel]]] = None


    # return_direct는 LangChain BaseTool의 속성. 필요에 따라 설정.
    return_direct: bool = False

    # _run은 LangChain BaseTool에 이미 정의된 추상 메서드.
    # 동기 실행 로직을 여기에 구현.
    @abc.abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the tool's logic synchronously. MUST be implemented by subclasses.
        Use Pydantic models for input validation within this method if complex validation is needed beyond the schema.
        """
        raise NotImplementedError(f"{self.__class__.__name__}._run not implemented")

    # _arun은 LangChain BaseTool에 정의된 비동기 실행 메서드.
    # 기본 구현은 _run을 호출. 비동기 I/O가 필요하면 재정의.
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the tool's logic asynchronously. Override if the tool involves async I/O.
        The default implementation runs the sync version in a thread pool.
        Use Pydantic models for input validation within this method if complex validation is needed beyond the schema.
        """
        # 기본 구현은 LangChain BaseTool의 구현을 따름 (sync _run 호출)
        # 필요시, 여기서 직접 비동기 로직 구현
        # 예: return await self._run_async_logic(*args, **kwargs)
        # 또는 validation 후 _run 호출
        # validated_args = self._validate_and_parse_args(*args, **kwargs) # 예시
        # result = await self._perform_async_operation(**validated_args)
        # return result

        # LangChain의 기본 동작에 맡기거나, 필요시 여기에 로직 추가
        # 여기서는 명시적으로 _run을 호출하는 대신 BaseTool의 기본 동작을 활용
        # (BaseTool은 _arun이 구현되지 않으면 _run을 executor에서 실행)
        logger.debug(f"Default _arun called for {self.name}, will likely execute _run in executor.")
        # 만약 _run 로직만 있고, 비동기 실행을 명시적으로 제어하고 싶다면:
        # loop = asyncio.get_event_loop()
        # return await loop.run_in_executor(None, functools.partial(self._run, *args, **kwargs))

        # 여기서는 Langchain의 기본 동작을 따르도록 pass 또는 _run 호출 코드를 제거합니다.
        # LangChain BaseTool이 _arun 미구현 시 _run을 처리합니다.
        # 만약 이 클래스에서 직접 _run을 호출해야 한다면 다음처럼 합니다.
        # return self._run(*args, **kwargs) # 이 경우 BaseTool 기본 비동기 처리 무시

        # Pydantic 스키마 유효성 검사는 LangChain Tool 실행 메커니즘에서 처리됨
        # 오류 처리도 _run/_arun 내부에서 try...except로 처리하고 ToolError 발생시키는 것을 권장
        pass # LangChain BaseTool의 기본 _arun 동작에 맡김

    # run/arun 래퍼 메서드 제거 -> LangChain의 invoke/ainvoke 등 사용
    # _validate_args 제거 -> args_schema와 Pydantic으로 처리 또는 _run/_arun 내부에서 처리
    # _handle_error 제거 -> _run/_arun 내부에서 try...except 및 ToolError 사용

    # 빈 스키마 생성 헬퍼 유지
    @classmethod
    def get_empty_args_schema(cls, name: str = 'EmptyArgsSchema') -> Type[BaseModel]:
        """Create an empty argument schema."""
        return create_model(name, __base__=BaseModel)

    # model_config은 LangChain BaseTool에서 상속받은 것을 사용할 수 있음
    # 필요시 여기서 추가 설정 가능
    class Config:
        """Pydantic V1 style config."""
        arbitrary_types_allowed = True


# DynamicTool 유지 및 수정 (BaseTool 상속 관계 수정)
class DynamicTool(BaseTool):
    """
    Dynamically created tool from a function.
    Inherits from the updated BaseTool (which inherits from Langchain's BaseTool).
    """
    # 클래스 변수로 속성들 정의 (Pydantic V2 및 Langchain BaseTool 스타일에 맞춤)
    # name: str - 생성자에서 설정
    # description: str - 생성자에서 설정
    # args_schema: Type[BaseModel] - 생성자에서 설정 또는 추론
    func: Callable[..., Any]
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None

    # Pydantic 모델이므로 __init__ 대신 model_post_init 사용 가능 (또는 __init__ 유지)
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        args_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any # BaseTool의 다른 필드(예: return_direct)를 받을 수 있도록
    ):
        # BaseTool의 필드를 초기화하기 위해 kwargs 사용 가능
        inferred_schema = args_schema or self._infer_args_schema(func, name)
        super().__init__(
            name=name,
            description=description,
            args_schema=inferred_schema,
            func=func, # DynamicTool 고유 필드
            coroutine=coroutine, # DynamicTool 고유 필드
            **kwargs # BaseTool의 다른 필드들 전달
        )
        # func와 coroutine은 BaseTool 필드가 아니므로 직접 할당
        self.func = func
        self.coroutine = coroutine


    # _run 메서드 구현 (BaseTool의 추상 메서드 충족)
    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Executes the synchronous function."""
        try:
            # args_schema를 통해 이미 유효성 검사가 되었거나, 여기서 추가 검사 가능
            # Langchain 실행 흐름에서 kwargs에 인자들이 매핑되어 들어옴
            result = self.func(**kwargs)
            return str(result) # 결과는 문자열로 반환
        except Exception as e:
            logger.error(f"Error executing dynamic tool '{self.name}': {e}", exc_info=True)
            # ToolError를 발생시켜 에이전트/LLM이 처리하도록 함
            raise ToolError(
                message=f"Error in tool '{self.name}': {str(e)}",
                tool_name=self.name,
                original_error=e,
                code=ErrorCode.TOOL_EXECUTION_ERROR
            )

    # _arun 메서드 구현 (BaseTool의 추상 메서드 충족)
    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Executes the asynchronous function if available, otherwise the sync function."""
        try:
            if self.coroutine:
                result = await self.coroutine(**kwargs)
                return str(result)
            else:
                logger.debug(f"Executing synchronous function '{self.func.__name__}' for async call in DynamicTool '{self.name}'.")
                # BaseTool의 기본 _arun 동작과 유사하게 처리하거나 직접 실행
                # 여기서는 BaseTool의 기본 동작 활용 대신 직접 실행
                result = self._run(*args, **kwargs) # _run 내부에서 예외 처리됨
                return result
        except Exception as e:
            # _run에서 ToolError를 발생시키지 않은 경우 여기서 처리
            if not isinstance(e, ToolError):
                logger.error(f"Error executing async dynamic tool '{self.name}': {e}", exc_info=True)
                raise ToolError(
                    message=f"Error in async tool '{self.name}': {str(e)}",
                    tool_name=self.name,
                    original_error=e,
                    code=ErrorCode.TOOL_EXECUTION_ERROR
                )
            else:
                raise # 이미 ToolError인 경우 그대로 전달

    # _infer_args_schema는 내부 헬퍼이므로 private 유지
    @staticmethod
    def _infer_args_schema(func: Callable[..., Any], name: str) -> Type[BaseModel]:
        """Infer a pydantic schema from the function signature."""
        logger.debug(f"Inferring args schema for dynamic tool '{name}' from function '{func.__name__}'")
        try:
            sig = inspect.signature(func)
            fields: Dict[str, Tuple[Type, Any]] = {}

            for param_name, param in sig.parameters.items():
                # self, args, kwargs 등은 스키마에서 제외 (일반 함수 기준)
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue

                annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
                default = param.default if param.default != inspect.Parameter.empty else ...

                fields[param_name] = (annotation, default)

            model_name = f'{name.title().replace("_", "").replace(" ", "")}Schema'
            inferred_schema = create_model(model_name, **fields)

            logger.debug(f"Inferred schema for '{name}': {inferred_schema.schema()}")
            return inferred_schema

        except Exception as e:
            logger.warning(
                f"Could not infer args_schema for tool '{name}': {e}. Using empty schema.",
                exc_info=True
            )
            # BaseTool 클래스 메서드 직접 호출 대신, 빈 모델 생성 사용
            return create_model(f'{name.title().replace(" ", "")}Schema', __base__=BaseModel)

    # DynamicTool에 필요한 추가 설정이나 메서드 정의 가능
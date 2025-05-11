# src/services/tool_manager.py
import importlib
import inspect
import os
import pkgutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, get_origin, cast

from pydantic import BaseModel

# BaseTool import 경로 확인 및 수정 (이제 base는 tools 아래에 있음)
from src.tools.base import BaseTool
from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger
# 메트릭 관련 코드는 필요시 활성화
# from src.config.metrics import REGISTRY_OPERATION_DURATION, get_metrics_manager

logger = get_logger(__name__)
# metrics_manager = get_metrics_manager() # 필요시 활성화

T = TypeVar('T', bound=BaseTool) # 제네릭 타입을 BaseTool로 제한
F = TypeVar('F', bound=Callable[..., Any])

# 클래스 이름을 ToolManager로 변경
class ToolManager:
    """Manages registration, loading, and retrieval of tool classes and instances."""

    def __init__(self, name: str = "global_tools"): # 매니저에 이름 부여 가능
        self._name = name # 매니저 인스턴스 이름 저장 (로깅용)
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instance_cache: Dict[str, BaseTool] = {}
        self._lock = threading.RLock() # RLock으로 변경 (재진입 가능)
        logger.debug(f"ToolManager '{self._name}' initialized.")

    @property
    def name(self) -> str:
        """Returns the name of this ToolManager instance."""
        return self._name

    def register(self, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """Registers a tool class, typically via the @register_tool decorator."""
        logger.debug(f"Attempting to register tool class: {tool_cls.__name__} with ToolManager '{self.name}'")
        try:
            if not issubclass(tool_cls, BaseTool):
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f"Tool class '{tool_cls.__name__}' must inherit from BaseTool.", details={'class': tool_cls.__name__})

            # 클래스 변수로 name 확인 시도
            tool_name = getattr(tool_cls, 'name', None)
            if not tool_name or not isinstance(tool_name, str):
                # 클래스 변수에 없으면 임시 인스턴스 생성 시도 (기존 방식)
                try:
                    tool_instance_for_name = tool_cls()
                    tool_name = tool_instance_for_name.name
                except Exception as inst_err:
                    raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f"Could not determine 'name' for tool class '{tool_cls.__name__}': {inst_err}", details={'class': tool_cls.__name__}, original_error=inst_err)

            if not tool_name:
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f"Tool class '{tool_cls.__name__}' has an empty 'name' property.", details={'class': tool_cls.__name__})

            with self._lock:
                if tool_name in self._tools:
                    if self._tools[tool_name] == tool_cls:
                        logger.debug(f"Tool class '{tool_cls.__name__}' (name: '{tool_name}') is already registered in '{self.name}'. Skipping.")
                        return tool_cls
                    else:
                        raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f"Tool name '{tool_name}' is already registered by a different class '{self._tools[tool_name].__name__}' in manager '{self.name}'. Cannot register '{tool_cls.__name__}'.", details={'name': tool_name, 'new_class': tool_cls.__name__, 'existing_class': self._tools[tool_name].__name__})

                self._tools[tool_name] = tool_cls
                logger.info(f"Tool '{tool_name}' (Class: {tool_cls.__name__}) registered successfully in manager '{self.name}'.")
            return tool_cls
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to register tool class '{tool_cls.__name__}' in manager '{self.name}' due to an unexpected error.", extra={'class': tool_cls.__name__}, exc_info=e)
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Failed to register tool class {tool_cls.__name__} in {self.name}: {str(e)}', details={'class': tool_cls.__name__}, original_error=e)
    
    def has(self, tool_name: str) -> bool:
        """Checks if a tool with the given name is registered."""
        with self._lock:
            return tool_name in self._tools    

    def get_tool_class(self, tool_name: str) -> Type[BaseTool]:
        """Gets the registered tool class by name."""
        with self._lock:
            tool_class = self._tools.get(tool_name)
        if tool_class is None:
            available = list(self.get_names()) # 스레드 안전
            logger.warning(f"Tool class '{tool_name}' not found in manager '{self.name}'. Available: {available}", extra={'tool_name': tool_name})
            raise ToolError(code=ErrorCode.TOOL_NOT_FOUND, message=f"Tool class '{tool_name}' not found.", details={'name': tool_name, 'available': available}, tool_name=tool_name)
        return tool_class

    def get_tool(self, tool_name: str) -> BaseTool:
        """Gets a tool instance by name, using cache if available."""
        with self._lock:
            if tool_name in self._instance_cache:
                logger.debug(f"Returning cached tool instance for: {tool_name} from manager '{self.name}'")
                return self._instance_cache[tool_name]

        logger.debug(f"Tool instance cache miss for: {tool_name} in manager '{self.name}'. Creating new instance.")
        try:
            tool_cls = self.get_tool_class(tool_name) # 내부에서 lock 사용
            tool_instance = tool_cls()
            with self._lock:
                # 인스턴스 생성 후 캐시에 넣기 전에 다시 확인 (다른 스레드가 먼저 넣었을 수 있음)
                if tool_name not in self._instance_cache:
                     self._instance_cache[tool_name] = tool_instance
                     logger.debug(f"Created and cached new tool instance for: {tool_name} in manager '{self.name}'")
                else:
                     logger.debug(f"Instance for {tool_name} was created concurrently, using existing cache entry.")
                     tool_instance = self._instance_cache[tool_name] # 이미 캐시된 것 사용
            return tool_instance
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to instantiate tool '{tool_name}' in manager '{self.name}'", extra={'tool_name': tool_name}, exc_info=e)
            raise ToolError(code=ErrorCode.TOOL_CREATION_ERROR, message=f"Failed to create instance of tool '{tool_name}': {str(e)}", details={'name': tool_name}, original_error=e, tool_name=tool_name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        현재 매니저에 등록된 모든 Tool 메타데이터를 정리해 반환한다.

        Returns
        -------
        List[Dict[str, Any]]
            각 항목은 다음 형태의 딕셔너리이다.
            {
                "name": "calculator",
                "description": "사칙연산 계산기",
                "class_name": "CalculatorTool",
                "args_schema_summary": {"expression": "string (required)"}
            }
        """
        result: List[Dict[str, Any]] = []

        for tool_name, tool_cls in self._tools.items():
            # ── 1) 기본 메타
            entry: Dict[str, Any] = {
                "name": tool_name,                       # slug (e.g. "mock_tool")
                "description": getattr(tool_cls, "description", ""),
                "class_name": tool_cls.__name__,
            }

            # ── 2) args_schema 요약
            schema_summary: Dict[str, str] = {}

            # 2-1) 클래스 속성에서 직접 찾기
            raw_schema = getattr(tool_cls, "args_schema", None)
            if isinstance(raw_schema, BaseModel):
                schema_cls = raw_schema.__class__
            elif inspect.isclass(raw_schema) and issubclass(raw_schema, BaseModel):
                schema_cls = raw_schema
            else:
                schema_cls = None

            # 2-2) 백업 로직 ─ 인스턴스를 한 번 만들어 필드 값에서 재탐색
            if schema_cls is None:
                try:
                    sig = inspect.signature(tool_cls.__init__)
                    # 매개변수(self 제외)에 필수 positional 인자가 없을 때만 안전 생성
                    if all(
                        p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                        for p in list(sig.parameters.values())[1:]  # self 제외
                    ):
                        inst = tool_cls()
                        inst_schema = getattr(inst, "args_schema", None)
                        if inspect.isclass(inst_schema) and issubclass(inst_schema, BaseModel):
                            schema_cls = inst_schema
                except Exception:
                    # 생성 실패 시 요약 없이 넘어감
                    pass

            # 2-3) 필드 요약 생성
            if schema_cls:
                for fname, field in schema_cls.model_fields.items():  # Pydantic v2 API
                    typ = field.annotation
                    origin = get_origin(typ)
                    typ_name = origin.__name__ if origin else getattr(typ, "__name__", str(typ))
                    if typ_name == "str":
                        typ_name = "string"
                    if field.is_required():
                        typ_name += " (required)"
                    schema_summary[fname] = typ_name

            entry["args_schema_summary"] = schema_summary
            result.append(entry)

        return result


    def get_tool_summaries_for_llm(self) -> List[Dict[str, str]]:
         """Provides concise summaries of tools suitable for LLM prompts."""
         summaries = []
         # list_tools는 내부적으로 get_names 등을 호출하며 lock을 사용하므로 여기선 추가 lock 불필요
         for tool_info in self.list_tools():
             if 'error' not in tool_info:
                 # args_schema_summary를 문자열로 변환하여 포함 가능
                 args_str = ", ".join([f"{k}: {v}" for k, v in tool_info.get('args_schema_summary', {}).items()])
                 summary = {
                     "name": tool_info.get('name', 'unknown'),
                     "description": f"{tool_info.get('description', 'No description.')} (Arguments: {args_str if args_str else 'None'})"
                 }
                 summaries.append(summary)
         return summaries

    def clear_cache(self) -> None:
        """Clears the tool instance cache."""
        with self._lock:
            cache_size_before = len(self._instance_cache)
            self._instance_cache.clear()
        logger.debug(f"Tool instance cache cleared for manager '{self.name}' ({cache_size_before} items removed).")

    def unregister(self, tool_name: str) -> None:
        """Unregisters a tool class and removes its instance from cache."""
        with self._lock:
            if tool_name not in self._tools:
                raise ToolError(code=ErrorCode.TOOL_NOT_FOUND, message=f"Tool '{tool_name}' not found for unregistration in manager '{self.name}'.", details={'name': tool_name}, tool_name=tool_name)
            del self._tools[tool_name]
            if tool_name in self._instance_cache:
                del self._instance_cache[tool_name]
                logger.debug(f"Removed cached instance for unregistered tool: {tool_name} from manager '{self.name}'")
        logger.info(f"Tool '{tool_name}' unregistered successfully from manager '{self.name}'.")

    def get_names(self) -> Set[str]:
        """Returns a set of all registered tool names."""
        with self._lock:
            return set(self._tools.keys())

    def load_tools_from_directory(
        self,
        directory: str,
        *,
        auto_register: bool = True,
        recursive: bool = False,
    ) -> int:
        """
        주어진 ``directory`` 의 모든 ``*.py`` 모듈을 import 한 뒤
        (옵션) 그 안에서 발견한 ``BaseTool`` 하위 클래스를 전역 ToolManager
        (이름: ``global_tools``) 에 자동 등록한다.

        Parameters
        ----------
        directory : str
            탐색할 디렉토리 경로 (절대/상대 모두 허용).
        auto_register : bool, default=True
            True 이면 로드된 모듈 안의 Tool 클래스를 전역 매니저에 등록.
        recursive : bool, default=False
            True 이면 서브 패키지까지 재귀적으로 탐색.

        Returns
        -------
        int
            *성공적으로 import* 된 **모듈**의 수.
        """
        src_path = Path(directory).resolve()
        if not src_path.exists():
            logger.error("Path %s does not exist – load aborted.", src_path)
            return 0

        logger.info("ToolManager '%s' loading tools from: %s", self.name, src_path)

        # ── import 를 위해 부모 디렉터리를 sys.path 에 잠시 추가
        parent_dir = str(src_path.parent)
        path_added = False
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            path_added = True

        imported_count = 0
        search_paths: List[str] = [str(src_path)]

        try:
            walker = (
                pkgutil.walk_packages(search_paths) if recursive
                else pkgutil.iter_modules(search_paths)
            )

            for finder, mod_name, ispkg in walker:
                # (recursive=False) 이고 ispkg=True 면 건너뜀
                if ispkg and not recursive:
                    continue

                module_file = Path(finder.path) / f"{mod_name}.py"
                spec = importlib.util.spec_from_file_location(mod_name, module_file)
                if not spec or not spec.loader:
                    logger.error("No import spec for module '%s'", mod_name)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module  # 임시 등록 (순환 import 대비)

                try:
                    spec.loader.exec_module(module)  # 실제 import 실행
                except Exception as exc:
                    logger.error(
                        "Failed to import module '%s': %s",
                        mod_name, exc, exc_info=True
                    )
                    sys.modules.pop(mod_name, None)
                    continue  # 실패 → 다음 모듈로

                imported_count += 1
                logger.debug("Imported tool module: %s", mod_name)

                # ── (선택) 자동 등록
                if auto_register:
                    global_mgr = get_tool_manager("global_tools")
                    for attr in module.__dict__.values():
                        if (
                            inspect.isclass(attr) and
                            issubclass(attr, BaseTool) and
                            attr is not BaseTool
                        ):
                            try:
                                global_mgr.register(attr)
                            except Exception as reg_exc:
                                logger.warning(
                                    "Auto-register skipped for %s: %s",
                                    attr.__name__, reg_exc
                                )

        finally:
            if path_added:
                sys.path.remove(parent_dir)

        logger.info(
            "ToolManager '%s' finished loading. %d module(s) imported.",
            self.name, imported_count
        )
        return imported_count






# --- 싱글톤 인스턴스 관리 ---
_managers: Dict[str, ToolManager] = {}
_manager_lock = threading.RLock()

# --- 데코레이터 및 팩토리 함수 ---
# register_tool 데코레이터 (ToolManager 인스턴스를 받도록 수정)
def register_tool(manager: Optional[ToolManager] = None) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a tool class with a specific ToolManager (or the default 'global_tools' one).
    """
    def decorator(cls: Type[T]) -> Type[T]:
        target_manager = manager
        if target_manager is None:
            # 기본 매니저 이름 설정 (예: 'global_tools')
            # get_tool_manager는 async가 아니므로 바로 호출 가능 (단, 첫 호출 시 생성될 수 있음)
            target_manager = get_tool_manager('global_tools') # get_tool_manager 사용

        target_manager.register(cls) # register 메서드가 이름 추출 및 등록 수행
        # logger.debug(f"Tool class '{cls.__name__}' registered via decorator to manager '{target_manager.name}'.") # manager.name 사용
        return cls
    return decorator

# get_tool_manager 함수 (get_tool_manager 대체)
def get_tool_manager(name: str = 'global_tools') -> ToolManager:
    """
    Gets or creates a ToolManager instance by name.
    Uses 'global_tools' as the default name. Thread-safe.
    """
    global _managers
    with _manager_lock:
        if name not in _managers:
            logger.info(f"Creating new ToolManager instance named: {name}")
            _managers[name] = ToolManager(name=name) # 생성 시 이름 전달
        else:
            logger.debug(f"Returning existing ToolManager instance named: {name}")
        return _managers[name]

# --- 기존 get_tool_manager 함수는 제거하거나 deprecated 처리 ---
# def get_tool_manager(name: str = 'default') -> ToolManager: ...
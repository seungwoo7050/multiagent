import importlib
import inspect
import pkgutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, get_origin

from pydantic import BaseModel


from src.tools.base import BaseTool
from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger


logger = get_logger(__name__)


T = TypeVar("T", bound=BaseTool)
F = TypeVar("F", bound=Callable[..., Any])


class ToolManager:
    """Manages registration, loading, and retrieval of tool classes and instances."""

    def __init__(self, name: str = "global_tools"):
        self._name = name
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instance_cache: Dict[str, BaseTool] = {}
        self._lock = threading.RLock()
        logger.debug(f"ToolManager '{self._name}' initialized.")

    @property
    def name(self) -> str:
        """Returns the name of this ToolManager instance."""
        return self._name

    def register(self, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """Registers a tool class, typically via the @register_tool decorator."""
        logger.debug(
            f"Attempting to register tool class: {tool_cls.__name__} with ToolManager '{self.name}'"
        )
        try:
            if not issubclass(tool_cls, BaseTool):
                raise ToolError(
                    code=ErrorCode.TOOL_VALIDATION_ERROR,
                    message=f"Tool class '{tool_cls.__name__}' must inherit from BaseTool.",
                    details={"class": tool_cls.__name__},
                )

            tool_name = getattr(tool_cls, "name", None)
            if not tool_name or not isinstance(tool_name, str):
                try:
                    tool_instance_for_name = tool_cls()
                    tool_name = tool_instance_for_name.name
                except Exception as inst_err:
                    raise ToolError(
                        code=ErrorCode.TOOL_VALIDATION_ERROR,
                        message=f"Could not determine 'name' for tool class '{tool_cls.__name__}': {inst_err}",
                        details={"class": tool_cls.__name__},
                        original_error=inst_err,
                    )

            if not tool_name:
                raise ToolError(
                    code=ErrorCode.TOOL_VALIDATION_ERROR,
                    message=f"Tool class '{tool_cls.__name__}' has an empty 'name' property.",
                    details={"class": tool_cls.__name__},
                )

            with self._lock:
                if tool_name in self._tools:
                    if self._tools[tool_name] == tool_cls:
                        logger.debug(
                            f"Tool class '{tool_cls.__name__}' (name: '{tool_name}') is already registered in '{self.name}'. Skipping."
                        )
                        return tool_cls
                    else:
                        raise ToolError(
                            code=ErrorCode.TOOL_VALIDATION_ERROR,
                            message=f"Tool name '{tool_name}' is already registered by a different class '{self._tools[tool_name].__name__}' in manager '{self.name}'. Cannot register '{tool_cls.__name__}'.",
                            details={
                                "name": tool_name,
                                "new_class": tool_cls.__name__,
                                "existing_class": self._tools[tool_name].__name__,
                            },
                        )

                self._tools[tool_name] = tool_cls
                logger.info(
                    f"Tool '{tool_name}' (Class: {tool_cls.__name__}) registered successfully in manager '{self.name}'."
                )
            return tool_cls
        except ToolError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to register tool class '{tool_cls.__name__}' in manager '{self.name}' due to an unexpected error.",
                extra={"class": tool_cls.__name__},
                exc_info=e,
            )
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message=f"Failed to register tool class {tool_cls.__name__} in {self.name}: {str(e)}",
                details={"class": tool_cls.__name__},
                original_error=e,
            )

    def has(self, tool_name: str) -> bool:
        """Checks if a tool with the given name is registered."""
        with self._lock:
            return tool_name in self._tools

    def get_tool_class(self, tool_name: str) -> Type[BaseTool]:
        """Gets the registered tool class by name."""
        with self._lock:
            tool_class = self._tools.get(tool_name)
        if tool_class is None:
            available = list(self.get_names())
            logger.warning(
                f"Tool class '{tool_name}' not found in manager '{self.name}'. Available: {available}",
                extra={"tool_name": tool_name},
            )
            raise ToolError(
                code=ErrorCode.TOOL_NOT_FOUND,
                message=f"Tool class '{tool_name}' not found.",
                details={"name": tool_name, "available": available},
                tool_name=tool_name,
            )
        return tool_class

    def get_tool(self, tool_name: str) -> BaseTool:
        """Gets a tool instance by name, using cache if available."""
        with self._lock:
            if tool_name in self._instance_cache:
                logger.debug(
                    f"Returning cached tool instance for: {tool_name} from manager '{self.name}'"
                )
                return self._instance_cache[tool_name]

        logger.debug(
            f"Tool instance cache miss for: {tool_name} in manager '{self.name}'. Creating new instance."
        )
        try:
            tool_cls = self.get_tool_class(tool_name)
            tool_instance = tool_cls()
            with self._lock:
                if tool_name not in self._instance_cache:
                    self._instance_cache[tool_name] = tool_instance
                    logger.debug(
                        f"Created and cached new tool instance for: {tool_name} in manager '{self.name}'"
                    )
                else:
                    logger.debug(
                        f"Instance for {tool_name} was created concurrently, using existing cache entry."
                    )
                    tool_instance = self._instance_cache[tool_name]
            return tool_instance
        except ToolError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to instantiate tool '{tool_name}' in manager '{self.name}'",
                extra={"tool_name": tool_name},
                exc_info=e,
            )
            raise ToolError(
                code=ErrorCode.TOOL_CREATION_ERROR,
                message=f"Failed to create instance of tool '{tool_name}': {str(e)}",
                details={"name": tool_name},
                original_error=e,
                tool_name=tool_name,
            )

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
            entry: Dict[str, Any] = {
                "name": tool_name,
                "description": getattr(tool_cls, "description", ""),
                "class_name": tool_cls.__name__,
            }

            schema_summary: Dict[str, str] = {}

            raw_schema = getattr(tool_cls, "args_schema", None)
            if isinstance(raw_schema, BaseModel):
                schema_cls = raw_schema.__class__
            elif inspect.isclass(raw_schema) and issubclass(raw_schema, BaseModel):
                schema_cls = raw_schema
            else:
                schema_cls = None

            if schema_cls is None:
                try:
                    sig = inspect.signature(tool_cls.__init__)

                    if all(
                        p.default is not p.empty
                        or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                        for p in list(sig.parameters.values())[1:]
                    ):
                        inst = tool_cls()
                        inst_schema = getattr(inst, "args_schema", None)
                        if inspect.isclass(inst_schema) and issubclass(
                            inst_schema, BaseModel
                        ):
                            schema_cls = inst_schema
                except Exception:
                    pass

            if schema_cls:
                for fname, field in schema_cls.model_fields.items():
                    typ = field.annotation
                    origin = get_origin(typ)
                    typ_name = (
                        origin.__name__
                        if origin
                        else getattr(typ, "__name__", str(typ))
                    )
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

        for tool_info in self.list_tools():
            if "error" not in tool_info:
                args_str = ", ".join(
                    [
                        f"{k}: {v}"
                        for k, v in tool_info.get("args_schema_summary", {}).items()
                    ]
                )
                summary = {
                    "name": tool_info.get("name", "unknown"),
                    "description": f"{tool_info.get('description', 'No description.')} (Arguments: {args_str if args_str else 'None'})",
                }
                summaries.append(summary)
        return summaries

    def clear_cache(self) -> None:
        """Clears the tool instance cache."""
        with self._lock:
            cache_size_before = len(self._instance_cache)
            self._instance_cache.clear()
        logger.debug(
            f"Tool instance cache cleared for manager '{self.name}' ({cache_size_before} items removed)."
        )

    def unregister(self, tool_name: str) -> None:
        """Unregisters a tool class and removes its instance from cache."""
        with self._lock:
            if tool_name not in self._tools:
                raise ToolError(
                    code=ErrorCode.TOOL_NOT_FOUND,
                    message=f"Tool '{tool_name}' not found for unregistration in manager '{self.name}'.",
                    details={"name": tool_name},
                    tool_name=tool_name,
                )
            del self._tools[tool_name]
            if tool_name in self._instance_cache:
                del self._instance_cache[tool_name]
                logger.debug(
                    f"Removed cached instance for unregistered tool: {tool_name} from manager '{self.name}'"
                )
        logger.info(
            f"Tool '{tool_name}' unregistered successfully from manager '{self.name}'."
        )

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

        parent_dir = str(src_path.parent)
        path_added = False
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            path_added = True

        imported_count = 0
        search_paths: List[str] = [str(src_path)]

        try:
            walker = (
                pkgutil.walk_packages(search_paths)
                if recursive
                else pkgutil.iter_modules(search_paths)
            )

            for finder, mod_name, ispkg in walker:
                if ispkg and not recursive:
                    continue

                module_file = Path(finder.path) / f"{mod_name}.py"
                spec = importlib.util.spec_from_file_location(mod_name, module_file)
                if not spec or not spec.loader:
                    logger.error("No import spec for module '%s'", mod_name)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module

                try:
                    spec.loader.exec_module(module)
                except Exception as exc:
                    logger.error(
                        "Failed to import module '%s': %s", mod_name, exc, exc_info=True
                    )
                    sys.modules.pop(mod_name, None)
                    continue

                imported_count += 1
                logger.debug("Imported tool module: %s", mod_name)

                if auto_register:
                    global_mgr = get_tool_manager("global_tools")
                    for attr in module.__dict__.values():
                        if (
                            inspect.isclass(attr)
                            and issubclass(attr, BaseTool)
                            and attr is not BaseTool
                        ):
                            try:
                                global_mgr.register(attr)
                            except Exception as reg_exc:
                                logger.warning(
                                    "Auto-register skipped for %s: %s",
                                    attr.__name__,
                                    reg_exc,
                                )

        finally:
            if path_added:
                sys.path.remove(parent_dir)

        logger.info(
            "ToolManager '%s' finished loading. %d module(s) imported.",
            self.name,
            imported_count,
        )
        return imported_count


_managers: Dict[str, ToolManager] = {}
_manager_lock = threading.RLock()


def register_tool(
    manager: Optional[ToolManager] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a tool class with a specific ToolManager (or the default 'global_tools' one).
    """

    def decorator(cls: Type[T]) -> Type[T]:
        target_manager = manager
        if target_manager is None:
            target_manager = get_tool_manager("global_tools")

        target_manager.register(cls)

        return cls

    return decorator


def get_tool_manager(name: str = "global_tools") -> ToolManager:
    """
    Gets or creates a ToolManager instance by name.
    Uses 'global_tools' as the default name. Thread-safe.
    """
    global _managers
    with _manager_lock:
        if name not in _managers:
            logger.info(f"Creating new ToolManager instance named: {name}")
            _managers[name] = ToolManager(name=name)
        else:
            logger.debug(f"Returning existing ToolManager instance named: {name}")
        return _managers[name]

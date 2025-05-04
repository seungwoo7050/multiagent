import threading
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from pydantic import BaseModel

from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.core.registry import Registry
from src.tools.base import BaseTool

metrics = get_metrics_manager()
logger = get_logger(__name__)
_registries: Dict[str, Registry[Any]] = {}
_registries_lock = threading.RLock()

T = TypeVar('T', bound=Type[BaseTool])

class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instance_cache: Dict[str, BaseTool] = {}
        metrics.track_registry('size', registry_name='tool_registry', value=0)
        logger.debug('ToolRegistry initialized.')

    def register(self, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        logger.debug(f'Attempting to register tool class: {tool_cls.__name__}')
        try:
            if not issubclass(tool_cls, BaseTool):
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f"Tool class '{tool_cls.__name__}' must inherit from BaseTool.", details={'class': tool_cls.__name__})
            tool_instance: BaseTool = tool_cls()
            tool_name: str = tool_instance.name
            if not tool_name or not isinstance(tool_name, str):
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f"Tool class '{tool_cls.__name__}' has an invalid 'name' property.", details={'class': tool_cls.__name__, 'name_type': type(tool_name).__name__})
            if tool_name in self._tools:
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f"Tool name '{tool_name}' is already registered by class '{self._tools[tool_name].__name__}'.", details={'name': tool_name, 'new_class': tool_cls.__name__, 'existing_class': self._tools[tool_name].__name__})
            self._tools[tool_name] = tool_cls
            metrics.track_registry('operations', registry_name='tool_registry', operation_type='registration')
            metrics.track_registry('size', registry_name='tool_registry', value=len(self._tools))
            logger.info(f"Tool '{tool_name}' (Class: {tool_cls.__name__}) registered successfully.")
            return tool_cls
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to register tool class '{tool_cls.__name__}' due to an unexpected error.", extra={'class': tool_cls.__name__}, exc_info=e)
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Failed to register tool class {tool_cls.__name__}: {str(e)}', details={'class': tool_cls.__name__}, original_error=e)

    def get_tool_class(self, tool_name: str) -> Type[BaseTool]:
        tool_class = self._tools.get(tool_name)
        if tool_class is None:
            logger.warning(f"Tool '{tool_name}' not found in registry.", extra={'tool_name': tool_name, 'available_tools': list(self._tools.keys())})
            raise ToolError(code=ErrorCode.TOOL_NOT_FOUND, message=f"Tool '{tool_name}' not found in registry.", details={'name': tool_name, 'available_tools': list(self._tools.keys())}, tool_name=tool_name)
        return tool_class

    def get_tool(self, tool_name: str) -> BaseTool:
        if tool_name in self._instance_cache:
            metrics.track_registry('operations', registry_name='tool_registry', operation_type='cache_hit')
            logger.debug(f'Returning cached tool instance for: {tool_name}')
            return self._instance_cache[tool_name]
        logger.debug(f'Tool instance cache miss for: {tool_name}. Creating new instance.')
        try:
            tool_cls = self.get_tool_class(tool_name)
            tool_instance = tool_cls()
            self._instance_cache[tool_name] = tool_instance
            metrics.track_registry('operations', registry_name='tool_registry', operation_type='cache_set')
            logger.debug(f'Created and cached new tool instance for: {tool_name}')
            return tool_instance
        except ToolError:
            raise
        except Exception as e:
            logger.error(f"Failed to instantiate tool '{tool_name}'", extra={'tool_name': tool_name}, exc_info=e)
            raise ToolError(code=ErrorCode.TOOL_CREATION_ERROR, message=f"Failed to create instance of tool '{tool_name}': {str(e)}", details={'name': tool_name}, original_error=e, tool_name=tool_name)

    def list_tools(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for tool_name in self._tools.keys():
            try:
                tool = self.get_tool(tool_name)
                tool_info = {'name': tool.name, 'description': tool.description, 'schema': self._get_schema_dict(tool.args_schema)}
                result.append(tool_info)
            except Exception as e:
                logger.warning(f"Error getting metadata for tool '{tool_name}'", extra={'tool_name': tool_name}, exc_info=e)
                result.append({'name': tool_name, 'error': f'Failed to retrieve metadata: {str(e)}'})
        return result

    def _get_schema_dict(self, schema_cls: Type[BaseModel]) -> Dict[str, Any]:
        try:
            schema = schema_cls.schema()
        except Exception as e:
            logger.error(f'Failed to get schema dictionary for class {schema_cls.__name__}: {e}')
            return {'error': f'Could not generate schema: {str(e)}'}
        return {'properties': schema.get('properties', {}), 'required': schema.get('required', []), 'title': schema.get('title', schema_cls.__name__), 'type': schema.get('type', 'object')}

    def clear_cache(self) -> None:
        cache_size_before = len(self._instance_cache)
        self._instance_cache.clear()
        metrics.track_cache('operations', operation_type='tool_cache_clear')
        logger.debug(f'Tool instance cache cleared ({cache_size_before} items removed).')

    def unregister(self, tool_name: str) -> None:
        if tool_name not in self._tools:
            raise ToolError(code=ErrorCode.TOOL_NOT_FOUND, message=f"Tool '{tool_name}' not found for unregistration.", details={'name': tool_name}, tool_name=tool_name)
        del self._tools[tool_name]
        if tool_name in self._instance_cache:
            del self._instance_cache[tool_name]
            logger.debug(f'Removed cached instance for unregistered tool: {tool_name}')
        metrics.track_registry('operations', registry_name='tool_registry', operation_type='unregistration')
        metrics.track_registry('size', registry_name='tool_registry', value=len(self._tools))
        logger.info(f"Tool '{tool_name}' unregistered successfully.")
        
    def list_names(self) -> List[str]:
        """
        Return a list of all registered tool names.
        Alias for get_names() to maintain backward compatibility.
        
        Returns:
            List[str]: Names of all registered tools
        """
        return list(self.get_names())

    def get_names(self) -> Set[str]:
        return set(self._tools.keys())

def register_tool(registry: Optional[ToolRegistry]=None) -> Callable[[T], T]:

    def decorator(cls: T) -> T:
        if registry is not None:
            registry.register(cls)
            logger.debug(f"Tool class '{cls.__name__}' registered via decorator to specific registry.")
        else:
            logger.debug(f"Tool class '{cls.__name__}' marked for registration (decorator used without specific registry). Ensure a global registry processes this later.")
        return cls
    return decorator

def get_registry(name: str = 'default') -> ToolRegistry:
    """
    지정된 이름의 ToolRegistry 인스턴스를 가져오거나 생성합니다.
    이름이 제공되지 않으면 'default' 레지스트리를 사용합니다.
    """
    global _registries
    if name not in _registries:
        logger.info(f"Creating new ToolRegistry instance named: {name}")
        _registries[name] = ToolRegistry()
        # 생성된 레지스트리에 대한 추가 초기화 로직이 필요하면 여기에 추가
    else:
        logger.debug(f"Returning existing ToolRegistry instance named: {name}")
    return _registries[name]
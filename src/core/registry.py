import functools
import inspect
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, Union, cast, get_type_hints
from src.config.logger import get_logger
from src.utils.timing import async_timed
logger = get_logger(__name__)
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class Registry(Generic[T]):

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, T] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        logger.info(f'Registry initialized: {name}')

    def register(self, name: str, item: T, **metadata) -> T:
        if name in self._registry:
            logger.warning(f'Overriding existing item in {self._name} registry: {name}')
        self._registry[name] = item
        self._metadata[name] = metadata
        logger.debug(f'Registered item in {self._name} registry: {name}', extra={'registry': self._name, 'item_name': name, 'item_type': type(item).__name__, 'metadata': metadata})
        return item

    def unregister(self, name: str) -> Optional[T]:
        if name not in self._registry:
            logger.warning(f'Attempted to unregister non-existent item in {self._name} registry: {name}')
            return None
        item = self._registry.pop(name)
        self._metadata.pop(name, None)
        logger.debug(f'Unregistered item from {self._name} registry: {name}', extra={'registry': self._name, 'item_name': name})
        return item

    @async_timed(name='registry_get_item')
    async def get(self, name: str) -> Optional[T]:
        return self.get_sync(name)

    def get_sync(self, name: str) -> Optional[T]:
        item = self._registry.get(name)
        if item is None:
            logger.debug(f'Item not found in {self._name} registry: {name}', extra={'registry': self._name, 'item_name': name})
        return item

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        metadata = self._metadata.get(name)
        return metadata.copy() if metadata is not None else None

    def has(self, name: str) -> bool:
        return name in self._registry

    def list_names(self) -> List[str]:
        return list(self._registry.keys())

    def list_items(self) -> List[T]:
        return list(self._registry.values())

    def list_all(self) -> Dict[str, T]:
        return self._registry.copy()

    def clear(self) -> None:
        count = len(self._registry)
        self._registry.clear()
        self._metadata.clear()
        logger.info(f'Cleared {self._name} registry ({count} items)')

    def size(self) -> int:
        return len(self._registry)

    def decorator(self, name: Optional[str]=None, **metadata) -> Callable[[T], T]:

        def decorator_func(item: T) -> T:
            item_name = name
            if item_name is None:
                if hasattr(item, '__name__'):
                    item_name = cast(Any, item).__name__
                elif hasattr(item, '__class__'):
                    item_name = item.__class__.__name__
                else:
                    raise ValueError(f'Cannot determine name for item of type {type(item).__name__}. Please provide a name explicitly using @registry.decorator(name=...).')
            return self.register(item_name, item, **metadata)
        return decorator_func

class FunctionRegistry(Registry[Callable[..., Any]]):

    def register_function(self, name: Optional[str]=None, **metadata) -> Callable[[F], F]:

        def decorator(func: F) -> F:
            func_name = name or func.__name__
            try:
                sig = inspect.signature(func)
                param_info = {}
                for param_name, param in sig.parameters.items():
                    param_info[param_name] = {'kind': str(param.kind), 'has_default': param.default is not param.empty, 'default': None if param.default is param.empty else param.default, 'annotation': str(param.annotation) if param.annotation is not param.empty else None}
                extended_metadata = {'module': func.__module__, 'doc': func.__doc__, 'parameters': param_info, 'return_type': str(sig.return_annotation) if sig.return_annotation is not sig.empty else None, 'is_async': inspect.iscoroutinefunction(func), **metadata}
            except Exception as e:
                logger.warning(f'Could not automatically extract metadata for function {func_name}: {e}')
                extended_metadata = metadata
            self.register(func_name, func, **extended_metadata)
            return func
        return decorator

class ClassRegistry(Registry[Type[Any]]):

    def register_class(self, name: Optional[str]=None, **metadata) -> Callable[[Type[T]], Type[T]]:

        def decorator(cls: Type[T]) -> Type[T]:
            class_name = name or cls.__name__
            try:
                extended_metadata = {'module': cls.__module__, 'doc': cls.__doc__, 'bases': [base.__name__ for base in cls.__bases__], **metadata}
            except Exception as e:
                logger.warning(f'Could not automatically extract metadata for class {class_name}: {e}')
                extended_metadata = metadata
            self.register(class_name, cls, **extended_metadata)
            return cls
        return decorator

    async def create_instance(self, name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
        cls = await self.get(name)
        if cls is None:
            logger.error(f"Class '{name}' not found in {self._name} registry for instantiation.")
            return None
        try:
            instance = cls(*args, **kwargs)
            return instance
        except Exception as e:
            logger.error(f"Failed to instantiate class '{name}' from {self._name} registry: {e}", exc_info=True)
            return None
_registries: Dict[str, Registry[Any]] = {}

def get_registry(name: str, registry_type: str='generic') -> Registry[Any]:
    registry_key = f'{name}:{registry_type}'
    if registry_key not in _registries:
        if registry_type == 'generic':
            _registries[registry_key] = Registry[Any](name)
        elif registry_type == 'function':
            _registries[registry_key] = FunctionRegistry(name)
        elif registry_type == 'class':
            _registries[registry_key] = ClassRegistry(name)
        else:
            raise ValueError(f"Invalid registry type specified: {registry_type}. Must be 'generic', 'function', or 'class'.")
        logger.info(f"Created new registry: Key='{registry_key}', Name='{name}', Type='{registry_type}'")
    return _registries[registry_key]

def clear_all_registries() -> None:
    count = len(_registries)
    logger.info(f'Clearing all ({count}) registries...')
    for registry in _registries.values():
        try:
            registry.clear()
        except Exception as e:
            logger.error(f"Error clearing registry '{registry._name}': {e}")
    logger.info('Finished clearing all registries.')

def get_function_registry(name: str) -> FunctionRegistry:
    registry = get_registry(name, 'function')
    return cast(FunctionRegistry, registry)

def get_class_registry(name: str) -> ClassRegistry:
    registry = get_registry(name, 'class')
    return cast(ClassRegistry, registry)
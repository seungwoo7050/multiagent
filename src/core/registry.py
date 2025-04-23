import functools
import inspect
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar, Union, cast, get_type_hints

from src.config.logger import get_logger
from src.utils.timing import async_timed

# Module logger
logger = get_logger(__name__)

# Type variables for generic registry
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class Registry(Generic[T]):
    """Dynamic registry pattern for registering and retrieving components.
    
    This generic registry can be used for any type of component,
    with O(1) lookup performance and decorator support.
    """
    
    def __init__(self, name: str):
        """Initialize a new registry.
        
        Args:
            name: Registry name for logging and debugging.
        """
        self._name = name
        self._registry: Dict[str, T] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Registry initialized: {name}")
    
    def register(
        self, 
        name: str, 
        item: T, 
        **metadata
    ) -> T:
        """Register an item with the registry.
        
        Args:
            name: Unique identifier for the item.
            item: The item to register.
            **metadata: Additional metadata to store with the item.
            
        Returns:
            T: The registered item (for method chaining).
            
        Raises:
            ValueError: If an item with the same name already exists
                        and override is False.
        """
        if name in self._registry:
            logger.warning(
                f"Overriding existing item in {self._name} registry: {name}"
            )
        
        self._registry[name] = item
        self._metadata[name] = metadata
        
        logger.debug(
            f"Registered item in {self._name} registry: {name}",
            extra={
                "registry": self._name,
                "item_name": name,
                "item_type": type(item).__name__,
                "metadata": metadata
            }
        )
        
        return item
    
    def unregister(self, name: str) -> Optional[T]:
        """Unregister an item from the registry.
        
        Args:
            name: Identifier of the item to unregister.
            
        Returns:
            Optional[T]: The unregistered item, or None if not found.
        """
        if name not in self._registry:
            logger.warning(
                f"Attempted to unregister non-existent item in {self._name} registry: {name}"
            )
            return None
        
        item = self._registry.pop(name)
        self._metadata.pop(name, None)
        
        logger.debug(
            f"Unregistered item from {self._name} registry: {name}",
            extra={"registry": self._name, "item_name": name}
        )
        
        return item
    
    @async_timed(name="registry_get_item")
    async def get(self, name: str) -> Optional[T]:
        """Get an item from the registry asynchronously.
        
        This is an async wrapper around get_sync for consistency in async contexts.
        It includes performance metrics for tracking access patterns.
        
        Args:
            name: Identifier of the item to retrieve.
            
        Returns:
            Optional[T]: The requested item, or None if not found.
        """
        return self.get_sync(name)
    
    def get_sync(self, name: str) -> Optional[T]:
        """Get an item from the registry synchronously.
        
        Args:
            name: Identifier of the item to retrieve.
            
        Returns:
            Optional[T]: The requested item, or None if not found.
        """
        if name not in self._registry:
            logger.debug(
                f"Item not found in {self._name} registry: {name}",
                extra={"registry": self._name, "item_name": name}
            )
            return None
        
        return self._registry[name]
    
    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an item.
        
        Args:
            name: Identifier of the item.
            
        Returns:
            Optional[Dict[str, Any]]: The metadata, or None if item not found.
        """
        if name not in self._metadata:
            return None
        
        return self._metadata[name].copy()
    
    def has(self, name: str) -> bool:
        """Check if an item exists in the registry.
        
        Args:
            name: Identifier of the item.
            
        Returns:
            bool: True if the item exists, False otherwise.
        """
        return name in self._registry
    
    def list_names(self) -> List[str]:
        """Get a list of all registered item names.
        
        Returns:
            List[str]: List of registered item names.
        """
        return list(self._registry.keys())
    
    def list_items(self) -> List[T]:
        """Get a list of all registered items.
        
        Returns:
            List[T]: List of registered items.
        """
        return list(self._registry.values())
    
    def list_all(self) -> Dict[str, T]:
        """Get a copy of the entire registry.
        
        Returns:
            Dict[str, T]: Copy of the registry mapping.
        """
        return self._registry.copy()
    
    def clear(self) -> None:
        """Clear the registry."""
        count = len(self._registry)
        self._registry.clear()
        self._metadata.clear()
        logger.info(f"Cleared {self._name} registry ({count} items)")
    
    def size(self) -> int:
        """Get the number of items in the registry.
        
        Returns:
            int: Number of registered items.
        """
        return len(self._registry)
    
    def decorator(self, name: Optional[str] = None, **metadata) -> Callable[[T], T]:
        """Create a decorator for registering items.
        
        Args:
            name: Optional name for the item. If not provided, uses the item's name.
            **metadata: Additional metadata to store with the item.
            
        Returns:
            Callable[[T], T]: Decorator function.
        """
        def decorator_func(item: T) -> T:
            item_name = name
            
            # If name not provided, use item's name or class name
            if item_name is None:
                if hasattr(item, "__name__"):
                    item_name = cast(Any, item).__name__
                elif hasattr(item, "__class__"):
                    item_name = item.__class__.__name__
                else:
                    raise ValueError(
                        f"Cannot determine name for item of type {type(item).__name__}. "
                        "Please provide a name explicitly."
                    )
            
            return self.register(item_name, item, **metadata)
        
        return decorator_func


class FunctionRegistry(Registry[Callable[..., Any]]):
    """Specialized registry for function registration.
    
    Provides additional functionality specific to functions,
    such as argument inspection and type checking.
    """
    
    def register_function(
        self,
        name: Optional[str] = None,
        **metadata
    ) -> Callable[[F], F]:
        """Decorator for registering functions with metadata.
        
        Args:
            name: Optional name override (defaults to function name).
            **metadata: Additional metadata about the function.
            
        Returns:
            Callable[[F], F]: Decorator function.
        """
        def decorator(func: F) -> F:
            func_name = name or func.__name__
            
            # Extract function signature for metadata
            sig = inspect.signature(func)
            param_info = {}
            
            for param_name, param in sig.parameters.items():
                param_info[param_name] = {
                    "kind": str(param.kind),
                    "has_default": param.default is not param.empty,
                    "default": None if param.default is param.empty else param.default,
                    "annotation": str(param.annotation) if param.annotation is not param.empty else None
                }
            
            # Add function signature to metadata
            extended_metadata = {
                "module": func.__module__,
                "doc": func.__doc__,
                "parameters": param_info,
                "return_type": str(sig.return_annotation) if sig.return_annotation is not sig.empty else None,
                "is_async": inspect.iscoroutinefunction(func),
                **metadata
            }
            
            self.register(func_name, func, **extended_metadata)
            return func
        
        return decorator


class ClassRegistry(Registry[Type[Any]]):
    """Specialized registry for class registration.
    
    Provides additional functionality specific to classes,
    such as inheritance checking and instance creation.
    """
    
    def register_class(
        self,
        name: Optional[str] = None,
        **metadata
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator for registering classes with metadata.
        
        Args:
            name: Optional name override (defaults to class name).
            **metadata: Additional metadata about the class.
            
        Returns:
            Callable[[Type[T]], Type[T]]: Decorator function.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            class_name = name or cls.__name__
            
            # Extract class information for metadata
            extended_metadata = {
                "module": cls.__module__,
                "doc": cls.__doc__,
                "bases": [base.__name__ for base in cls.__bases__],
                **metadata
            }
            
            self.register(class_name, cls, **extended_metadata)
            return cls
        
        return decorator
    
    async def create_instance(
        self,
        name: str,
        *args: Any,
        **kwargs: Any
    ) -> Optional[Any]:
        """Create an instance of a registered class.
        
        Args:
            name: Name of the registered class.
            *args: Positional arguments for the constructor.
            **kwargs: Keyword arguments for the constructor.
            
        Returns:
            Optional[Any]: Instance of the class, or None if class not found.
        """
        cls = await self.get(name)
        if cls is None:
            return None
        
        return cls(*args, **kwargs)


# Central registry instances
_registries: Dict[str, Registry[Any]] = {}


def get_registry(name: str, registry_type: str = "generic") -> Registry[Any]:
    """Get or create a registry instance.
    
    Args:
        name: Name of the registry.
        registry_type: Type of registry ('generic', 'function', or 'class').
        
    Returns:
        Registry[Any]: The requested registry instance.
        
    Raises:
        ValueError: If registry_type is invalid.
    """
    registry_key = f"{name}:{registry_type}"
    
    if registry_key not in _registries:
        if registry_type == "generic":
            _registries[registry_key] = Registry[Any](name)
        elif registry_type == "function":
            _registries[registry_key] = FunctionRegistry(name)
        elif registry_type == "class":
            _registries[registry_key] = ClassRegistry(name)
        else:
            raise ValueError(f"Invalid registry type: {registry_type}")
    
    return _registries[registry_key]


def clear_all_registries() -> None:
    """Clear all registries."""
    for registry in _registries.values():
        registry.clear()


def get_function_registry(name: str) -> FunctionRegistry:
    """Get or create a function registry.
    
    Args:
        name: Name of the registry.
        
    Returns:
        FunctionRegistry: The function registry instance.
    """
    registry = get_registry(name, "function")
    return cast(FunctionRegistry, registry)


def get_class_registry(name: str) -> ClassRegistry:
    """Get or create a class registry.
    
    Args:
        name: Name of the registry.
        
    Returns:
        ClassRegistry: The class registry instance.
    """
    registry = get_registry(name, "class")
    return cast(ClassRegistry, registry)
import pytest
import asyncio
from typing import Dict, Any, List, Optional

from src.core.registry import (
    Registry,
    FunctionRegistry,
    ClassRegistry,
    get_registry,
    get_function_registry,
    get_class_registry,
    clear_all_registries
)


# Test classes and functions
class TestItem:
    """Simple class for testing class registry."""
    
    def __init__(self, value: str = "default"):
        self.value = value
    
    def get_value(self) -> str:
        return self.value


class TestItemWithFromDict:
    """Class with from_dict method for testing deserialization."""
    
    def __init__(self, value: str = "default"):
        self.value = value
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestItemWithFromDict":
        return cls(value=data.get("value", "from_dict_default"))


def sample_function(a: int, b: int) -> int:
    """Simple function for testing function registry."""
    return a + b


async def sample_async_function(a: int, b: int) -> int:
    """Simple async function for testing function registry."""
    await asyncio.sleep(0.01)
    return a + b


class TestRegistry:
    """Test suite for Registry classes."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear all registries before each test
        clear_all_registries()

    def test_basic_registry_operations(self):
        """Test basic registry operations: register, get, has, unregister."""
        # Create registry
        # 참고: 실제 코드에서는 Registry를 임포트해야 합니다.
        # from src.core.registry import Registry, Any
        registry = Registry[Any]("test_registry")

        # Register items
        item1 = "test_item"
        item2 = 42
        item3 = {"key": "value"}

        registry.register("item1", item1)
        registry.register("item2", item2)
        registry.register("item3", item3)

        # Check registration
        assert registry.has("item1") is True
        assert registry.has("item2") is True
        assert registry.has("item3") is True
        assert registry.has("non_existent") is False

        # Get items
        assert registry.get_sync("item1") == "test_item"
        assert registry.get_sync("item2") == 42
        assert registry.get_sync("item3") == {"key": "value"} # [cite: 3]
        assert registry.get_sync("non_existent") is None

        # List operations
        items = registry.list_items()
        assert len(items) == 3  # Check length
        # Check membership individually
        assert item1 in items
        assert item2 in items
        assert item3 in items # [cite: 4]

        # Unregister
        removed_item = registry.unregister("item2")
        assert removed_item == 42
        assert registry.has("item2") is False
        assert registry.size() == 2

        # Unregister non-existent
        assert registry.unregister("non_existent") is None

        # Clear all
        registry.clear()
        assert registry.size() == 0
        assert len(registry.list_names()) == 0

    def test_registry_metadata(self):
        """Test registry metadata handling."""
        registry = Registry[str]("test_registry")
        
        # Register with metadata
        registry.register("item1", "value1", category="test", priority=1)
        registry.register("item2", "value2", category="prod", priority=2)
        
        # Get metadata
        metadata1 = registry.get_metadata("item1")
        assert metadata1 is not None
        assert metadata1["category"] == "test"
        assert metadata1["priority"] == 1
        
        # Get non-existent metadata
        assert registry.get_metadata("non_existent") is None
        
        # Overwrite item and check metadata is updated
        registry.register("item1", "new_value", category="updated")
        metadata1_updated = registry.get_metadata("item1")
        assert metadata1_updated["category"] == "updated"
        assert "priority" not in metadata1_updated

    @pytest.mark.asyncio
    async def test_registry_async_get(self):
        """Test async get method."""
        registry = Registry[str]("test_registry")
        registry.register("item1", "value1")
        
        # Get item asynchronously
        item = await registry.get("item1")
        assert item == "value1"
        
        # Get non-existent item
        item = await registry.get("non_existent")
        assert item is None

    def test_registry_decorator(self):
        """Test registry decorator."""
        registry = Registry[Any]("test_registry")
        
        # Use decorator with name
        @registry.decorator(name="decorated_item", metadata_key="from_decorator")
        class DecoratedClass:
            pass
        
        # Use decorator with auto-name
        @registry.decorator(metadata_key="from_decorator")
        def decorated_function():
            pass
        
        # Check registration
        assert registry.has("decorated_item") is True
        assert registry.has("decorated_function") is True
        
        # Check metadata
        assert registry.get_metadata("decorated_item")["metadata_key"] == "from_decorator"
        assert registry.get_metadata("decorated_function")["metadata_key"] == "from_decorator"

    def test_function_registry(self):
        """Test function registry specific features."""
        func_registry = FunctionRegistry("test_functions")
        
        # Register functions with decorator
        @func_registry.register_function()
        def add(a: int, b: int) -> int:
            return a + b
        
        @func_registry.register_function(name="custom_name", description="Custom multiply")
        def multiply(a: int, b: int) -> int:
            return a * b
        
        # Check registration
        assert func_registry.has("add") is True
        assert func_registry.has("custom_name") is True
        assert func_registry.has("multiply") is False  # Registered with custom name
        
        # Get functions
        add_func = func_registry.get_sync("add")
        multiply_func = func_registry.get_sync("custom_name")
        
        assert add_func(2, 3) == 5
        assert multiply_func(2, 3) == 6
        
        # Check metadata
        add_metadata = func_registry.get_metadata("add")
        assert add_metadata is not None
        assert add_metadata["is_async"] is False
        assert "parameters" in add_metadata
        assert "return_type" in add_metadata
        
        # Parameter info should be captured
        assert "a" in add_metadata["parameters"]
        assert "b" in add_metadata["parameters"]
        assert add_metadata["parameters"]["a"]["annotation"] == "<class 'int'>"

    @pytest.mark.asyncio
    async def test_function_registry_async(self):
        """Test function registry with async functions."""
        func_registry = FunctionRegistry("test_async_functions")
        
        # Register async function
        @func_registry.register_function()
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b
        
        # Check registration
        assert func_registry.has("async_add") is True
        
        # Get function
        async_add_func = func_registry.get_sync("async_add")
        
        # Execute
        result = await async_add_func(2, 3)
        assert result == 5
        
        # Check metadata
        metadata = func_registry.get_metadata("async_add")
        assert metadata is not None
        assert metadata["is_async"] is True

    def test_class_registry(self):
        """Test class registry specific features."""
        class_registry = ClassRegistry("test_classes")
        
        # Register classes with decorator
        @class_registry.register_class()
        class TestClass:
            def __init__(self, name: str = "default"):
                self.name = name
                
            def get_name(self) -> str:
                return self.name
        
        @class_registry.register_class(name="CustomItem", category="special")
        class SpecialItem:
            def __init__(self, value: int = 0):
                self.value = value
        
        # Check registration
        assert class_registry.has("TestClass") is True
        assert class_registry.has("CustomItem") is True
        
        # Get classes
        test_class = class_registry.get_sync("TestClass")
        custom_item_class = class_registry.get_sync("CustomItem")
        
        assert test_class is not None
        assert custom_item_class is not None
        
        # Check metadata
        test_metadata = class_registry.get_metadata("TestClass")
        assert test_metadata is not None
        assert "bases" in test_metadata
        assert test_metadata["module"] == test_class.__module__
        
        custom_metadata = class_registry.get_metadata("CustomItem")
        assert custom_metadata is not None
        assert custom_metadata["category"] == "special"

    @pytest.mark.asyncio
    async def test_class_registry_create_instance(self):
        """Test class registry's create_instance method."""
        class_registry = ClassRegistry("test_create_instance")
        
        # Register a class
        @class_registry.register_class()
        class Item:
            def __init__(self, name: str = "default", value: int = 0):
                self.name = name
                self.value = value
                
            def get_data(self) -> Dict[str, Any]:
                return {"name": self.name, "value": self.value}
        
        # Create instance with default params
        instance1 = await class_registry.create_instance("Item")
        assert instance1 is not None
        assert instance1.name == "default"
        assert instance1.value == 0
        
        # Create instance with custom params
        instance2 = await class_registry.create_instance("Item", "custom", value=42)
        assert instance2 is not None
        assert instance2.name == "custom"
        assert instance2.value == 42
        
        # Try to create instance of non-existent class
        instance3 = await class_registry.create_instance("NonExistentClass")
        assert instance3 is None

    def test_get_registry_functions(self):
        """Test the global registry getter functions."""
        # Get registries of different types
        registry1 = get_registry("global_test", registry_type="generic")
        func_registry = get_function_registry("global_functions")
        class_registry = get_class_registry("global_classes")
        
        # Register some items
        registry1.register("item1", "value1")
        
        @func_registry.register_function()
        def test_func():
            return "test"
        
        @class_registry.register_class()
        class TestClass:
            pass
        
        # Get same registries again
        registry2 = get_registry("global_test", registry_type="generic")
        func_registry2 = get_function_registry("global_functions")
        class_registry2 = get_class_registry("global_classes")
        
        # They should be the same instances
        assert registry1 is registry2
        assert func_registry is func_registry2
        assert class_registry is class_registry2
        
        # Check registered items are accessible
        assert registry2.get_sync("item1") == "value1"
        assert func_registry2.has("test_func") is True
        assert class_registry2.has("TestClass") is True
        
        # Clear all registries
        clear_all_registries()
        
        # Registries should be empty
        assert registry1.size() == 0
        assert func_registry.size() == 0
        assert class_registry.size() == 0
        
        # Invalid registry type should raise ValueError
        with pytest.raises(ValueError):
            get_registry("invalid_type_test", registry_type="invalid_type")

    def test_registry_threading(self):
        """Test registry operations from multiple threads."""
        import threading
        
        registry = Registry[str]("threading_test")
        
        # Define function to run in threads
        def register_items(prefix: str, count: int):
            for i in range(count):
                registry.register(f"{prefix}_{i}", f"value_{prefix}_{i}")
        
        # Create and start threads
        threads = []
        prefixes = ["a", "b", "c"]
        for prefix in prefixes:
            thread = threading.Thread(target=register_items, args=(prefix, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all items were registered
        assert registry.size() == 30
        for prefix in prefixes:
            for i in range(10):
                key = f"{prefix}_{i}"
                assert registry.has(key) is True
                assert registry.get_sync(key) == f"value_{prefix}_{i}"
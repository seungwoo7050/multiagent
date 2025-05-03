import unittest
import asyncio
from unittest.mock import MagicMock, patch

from src.tools.base import BaseTool, DynamicTool
from src.tools.calculator import CalculatorTool
from src.tools.datetime_tool import DateTimeTool, DateTimeOperation
from src.tools.registry import ToolRegistry
from src.tools.runner import ToolRunner
from src.config.errors import ToolError


class TestBaseTool(unittest.TestCase):
    """Tests for the BaseTool abstract base class and DynamicTool implementation"""
    
    def test_dynamic_tool_creation(self):
        """Test creating a DynamicTool with basic functionality"""
        def sample_func(x: int, y: int) -> int:
            return x + y
            
        async def sample_coro(x: int, y: int) -> int:
            return x * y
            
        tool = DynamicTool(
            name="test_tool",
            description="A test tool",
            func=sample_func,
            coroutine=sample_coro
        )
        
        # Check basic properties
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "A test tool")
        
        # Test sync execution
        result = tool.run(x=5, y=3)
        self.assertEqual(result, 8)
        
        # Test async execution
        async def test_async():
            return await tool.arun(x=5, y=3)
            
        result = asyncio.run(test_async())
        self.assertEqual(result, 15)
        
    def test_args_schema_inference(self):
        """Test the schema inference from function signature"""
        def func_with_types(x: int, y: str = "default") -> str:
            return f"{x} - {y}"
            
        tool = DynamicTool(
            name="schema_test",
            description="Testing schema inference",
            func=func_with_types
        )
        
        # Check the generated schema
        schema = tool.args_schema.schema()
        self.assertIn("properties", schema)
        self.assertIn("x", schema["properties"])
        self.assertIn("y", schema["properties"])
        self.assertEqual(schema["properties"]["y"]["default"], "default")


class TestCalculatorTool(unittest.TestCase):
    """Tests for the Calculator Tool implementation"""
    
    def setUp(self):
        self.calculator = CalculatorTool()
        
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations"""
        result = self.calculator.run(expression="2 + 2")
        self.assertEqual(result["result"], 4)
        
        result = self.calculator.run(expression="10 - 5")
        self.assertEqual(result["result"], 5)
        
        result = self.calculator.run(expression="3 * 4")
        self.assertEqual(result["result"], 12)
        
        result = self.calculator.run(expression="20 / 4")
        self.assertEqual(result["result"], 5)
        
    def test_complex_expressions(self):
        """Test more complex mathematical expressions"""
        result = self.calculator.run(expression="2 * (3 + 4)")
        self.assertEqual(result["result"], 14)
        
        result = self.calculator.run(expression="sin(0)")
        self.assertEqual(result["result"], 0)
        
        result = self.calculator.run(expression="cos(0)")
        self.assertEqual(result["result"], 1)
        
        result = self.calculator.run(expression="sqrt(16)")
        self.assertEqual(result["result"], 4)
        
    def test_invalid_expressions(self):
        """Test handling of invalid expressions"""
        with self.assertRaises(ToolError):
            self.calculator.run(expression="2 +/ 3")
            
        with self.assertRaises(ToolError):
            self.calculator.run(expression="invalid_function(5)")


class TestDateTimeTool(unittest.TestCase):
    """Tests for the DateTime Tool implementation"""
    
    def setUp(self):
        self.datetime_tool = DateTimeTool()
        
    def test_current_time(self):
        """Test getting current time"""
        result = self.datetime_tool.run(operation=DateTimeOperation.CURRENT)
        
        # Verify the structure of the response
        self.assertEqual(result["operation"], "current")
        self.assertIn("iso_format", result)
        self.assertIn("timestamp", result)
        self.assertIn("components", result)
        self.assertIn("year", result["components"])
        
    def test_date_parsing(self):
        """Test parsing date strings"""
        result = self.datetime_tool.run(
            operation=DateTimeOperation.PARSE,
            date_string="2023-01-15"
        )
        
        self.assertEqual(result["operation"], "parse")
        self.assertIn("iso_format", result)
        self.assertEqual(result["components"]["year"], 2023)
        self.assertEqual(result["components"]["month"], 1)
        self.assertEqual(result["components"]["day"], 15)
        
    def test_date_formatting(self):
        """Test date formatting operations"""
        result = self.datetime_tool.run(
            operation=DateTimeOperation.FORMAT,
            date_string="2023-01-15",
            format_string="%Y/%m/%d"
        )
        
        self.assertEqual(result["operation"], "format")
        self.assertEqual(result["formatted_string"], "2023/01/15")
        
    def test_date_operations(self):
        """Test date arithmetic operations"""
        # Test adding days
        result = self.datetime_tool.run(
            operation=DateTimeOperation.ADD,
            date_string="2023-01-15",
            days=5
        )
        
        self.assertEqual(result["operation"], "add")
        self.assertIn("2023-01-20", result["result_iso"])


class TestToolRegistry(unittest.TestCase):
    """Tests for the Tool Registry"""
    
    def setUp(self):
        self.registry = ToolRegistry()
        
    def test_tool_registration(self):
        """Test registering and retrieving tools"""
        # Register a tool class
        @self.registry.register
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"
                
            @property
            def description(self) -> str:
                return "A test tool"
                
            @property
            def args_schema(self):
                return self.get_empty_args_schema()
                
            def _run(self, **kwargs):
                return {"message": "Tool executed"}
                
            async def _arun(self, **kwargs):
                return self._run(**kwargs)
        
        # Check if tool was registered
        self.assertIn("test_tool", self.registry.get_names())
        
        # Get tool class and instance
        tool_class = self.registry.get_tool_class("test_tool")
        self.assertEqual(tool_class, TestTool)
        
        tool_instance = self.registry.get_tool("test_tool")
        self.assertIsInstance(tool_instance, TestTool)
        
        # Test tool execution
        result = tool_instance.run()
        self.assertEqual(result, {"message": "Tool executed"})
        
    def test_tool_listing(self):
        """Test listing all registered tools"""
        # Register a couple of tools
        @self.registry.register
        class Tool1(BaseTool):
            @property
            def name(self) -> str:
                return "tool1"
                
            @property
            def description(self) -> str:
                return "Tool 1"
                
            @property
            def args_schema(self):
                return self.get_empty_args_schema()
                
            def _run(self, **kwargs):
                return {"message": "Tool 1 executed"}
                
            async def _arun(self, **kwargs):
                return self._run(**kwargs)
                
        @self.registry.register
        class Tool2(BaseTool):
            @property
            def name(self) -> str:
                return "tool2"
                
            @property
            def description(self) -> str:
                return "Tool 2"
                
            @property
            def args_schema(self):
                return self.get_empty_args_schema()
                
            def _run(self, **kwargs):
                return {"message": "Tool 2 executed"}
                
            async def _arun(self, **kwargs):
                return self._run(**kwargs)
        
        # List all tools
        tools_list = self.registry.list_tools()
        
        # Check the structure of the list
        self.assertEqual(len(tools_list), 2)
        tool_names = [t["name"] for t in tools_list]
        self.assertIn("tool1", tool_names)
        self.assertIn("tool2", tool_names)


class TestToolRunner(unittest.TestCase):
    """Tests for the Tool Runner"""
    
    def setUp(self):
        self.runner = ToolRunner()
        self.registry = MagicMock()
        
    @patch("time.monotonic")
    async def test_run_tool(self, mock_monotonic):
        """Test running a tool through the runner"""
        # Mock time.monotonic() to return predictable values
        mock_monotonic.side_effect = [0.0, 1.0]
        
        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "mock_tool"
        mock_tool.arun.return_value = {"result": "success"}
        
        # Mock registry.get_tool to return our mock tool
        self.registry.get_tool.return_value = mock_tool
        
        # Run the tool
        result = await self.runner.run_tool(
            tool="mock_tool",
            tool_registry=self.registry,
            args={"param": "value"}
        )
        
        # Verify the result
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool_name"], "mock_tool")
        self.assertEqual(result["result"], {"result": "success"})
        self.assertEqual(result["execution_time"], 1.0)
        
        # Verify the tool was called correctly
        mock_tool.arun.assert_called_once_with(param="value")
        
    @patch("time.monotonic")
    async def test_run_tool_error(self, mock_monotonic):
        """Test error handling when running a tool"""
        # Mock time.monotonic() to return predictable values
        mock_monotonic.side_effect = [0.0, 1.0]
        
        # Create a mock tool that raises an exception
        mock_tool = MagicMock()
        mock_tool.name = "error_tool"
        mock_tool.arun.side_effect = Exception("Tool execution failed")
        
        # Mock registry.get_tool to return our mock tool
        self.registry.get_tool.return_value = mock_tool
        
        # Run the tool
        result = await self.runner.run_tool(
            tool="error_tool",
            tool_registry=self.registry,
            args={}
        )
        
        # Verify the error result
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["tool_name"], "error_tool")
        self.assertEqual(result["execution_time"], 1.0)
        self.assertIn("error", result)
        self.assertIn("Tool execution failed", result["error"]["message"])
        
        # Verify the tool was called
        mock_tool.arun.assert_called_once()


if __name__ == "__main__":
    unittest.main()
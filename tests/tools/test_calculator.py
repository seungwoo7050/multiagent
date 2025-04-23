"""Unit tests for the calculator tool."""

import pytest
import math
from pydantic import ValidationError

from src.config.errors import ToolError
from src.tools.calculator import CalculatorTool, CalculatorInput


@pytest.fixture
def calculator():
    """Create a calculator tool for tests."""
    return CalculatorTool()


def test_calculator_initialization(calculator):
    """Test calculator tool initialization."""
    assert calculator.name == "calculator"
    assert "mathematical expressions" in calculator.description.lower()
    assert calculator.args_schema == CalculatorInput


def test_calculator_input_validation():
    """Test calculator input validation."""
    # Valid input
    valid_input = CalculatorInput(expression="2 + 2")
    assert valid_input.expression == "2 + 2"
    
    # Empty expression
    with pytest.raises(ValidationError):
        CalculatorInput(expression="")
    
    # Too long expression
    with pytest.raises(ValidationError):
        CalculatorInput(expression="2 + 2" * 500)  # Over 1000 chars


def test_calculator_basic_arithmetic(calculator):
    """Test basic arithmetic operations."""
    # Addition
    result = calculator.run(expression="2 + 3")
    assert result["result"] == 5
    
    # Subtraction
    result = calculator.run(expression="5 - 3")
    assert result["result"] == 2
    
    # Multiplication
    result = calculator.run(expression="4 * 5")
    assert result["result"] == 20
    
    # Division
    result = calculator.run(expression="10 / 2")
    assert result["result"] == 5.0
    
    # Integer division
    result = calculator.run(expression="10 // 3")
    assert result["result"] == 3
    
    # Modulo
    result = calculator.run(expression="10 % 3")
    assert result["result"] == 1
    
    # Exponentiation
    result = calculator.run(expression="2 ** 3")
    assert result["result"] == 8


def test_calculator_complex_expressions(calculator):
    """Test more complex expressions with operator precedence."""
    # Mixed operations
    result = calculator.run(expression="2 + 3 * 4")
    assert result["result"] == 14
    
    # Parentheses
    result = calculator.run(expression="(2 + 3) * 4")
    assert result["result"] == 20
    
    # Nested parentheses
    result = calculator.run(expression="2 * (3 + (4 - 1))")
    assert result["result"] == 12
    
    # Multiple operations
    result = calculator.run(expression="10 - 2 * 3 + 4 / 2")
    assert result["result"] == 6.0


def test_calculator_math_functions(calculator):
    """Test math functions."""
    # Square root
    result = calculator.run(expression="sqrt(16)")
    assert result["result"] == 4.0
    
    # Sine
    result = calculator.run(expression="sin(0)")
    assert result["result"] == 0.0
    
    # Cosine
    result = calculator.run(expression="cos(0)")
    assert result["result"] == 1.0
    
    # Logarithm
    result = calculator.run(expression="log(10)")
    assert result["result"] == math.log(10)
    
    # Exp
    result = calculator.run(expression="exp(2)")
    assert result["result"] == math.exp(2)
    
    # Absolute value
    result = calculator.run(expression="abs(-5)")
    assert result["result"] == 5
    
    # Pi constant
    result = calculator.run(expression="pi")
    assert result["result"] == math.pi
    
    # E constant
    result = calculator.run(expression="e")
    assert result["result"] == math.e
    
    # Power function
    result = calculator.run(expression="pow(2, 3)")
    assert result["result"] == 8.0


def test_calculator_function_chaining(calculator):
    """Test chaining multiple math functions."""
    # Nested functions
    result = calculator.run(expression="sqrt(abs(-16))")
    assert result["result"] == 4.0
    
    # Functions with operators
    result = calculator.run(expression="sin(pi/2)")
    assert abs(result["result"] - 1.0) < 1e-10  # Allow for floating point precision
    
    # Multiple functions
    result = calculator.run(expression="sqrt(16) + log10(100)")
    assert result["result"] == 6.0


def test_calculator_comparison_operators(calculator):
    """Test comparison operators."""
    # Equal
    result = calculator.run(expression="2 == 2")
    assert result["result"] is True
    
    # Not equal
    result = calculator.run(expression="2 != 3")
    assert result["result"] is True
    
    # Greater than
    result = calculator.run(expression="5 > 3")
    assert result["result"] is True
    
    # Less than
    result = calculator.run(expression="2 < 3")
    assert result["result"] is True
    
    # Greater than or equal
    result = calculator.run(expression="5 >= 5")
    assert result["result"] is True
    
    # Less than or equal
    result = calculator.run(expression="3 <= 5")
    assert result["result"] is True
    
    # Chained comparison
    result = calculator.run(expression="1 < 2 < 3")
    assert result["result"] is True
    
    result = calculator.run(expression="1 < 2 > 3")
    assert result["result"] is False


def test_calculator_invalid_expressions(calculator):
    """Test invalid expressions."""
    # Syntax error
    with pytest.raises(ToolError) as excinfo:
        calculator.run(expression="2 +* 3")
    assert "Failed to evaluate expression" in str(excinfo.value)
    
    # Division by zero doesn't raise, follows Python's behavior
    result = calculator.run(expression="1/0")
    assert math.isinf(result["result"])
    
    # Invalid function
    with pytest.raises(ToolError) as excinfo:
        calculator.run(expression="invalid_func(2)")
    assert "Unknown variable" in str(excinfo.value.details["error"]) or "Function not allowed" in str(excinfo.value.details["error"])


def test_calculator_format_result(calculator):
    """Test result formatting."""
    # Integer result
    result = calculator._format_result(42)
    assert result == "42"
    
    # Float result
    result = calculator._format_result(3.14159)
    assert result == "3.14159"
    
    # Float that's almost an integer
    result = calculator._format_result(5.000000001)
    assert result == "5"
    
    # Very large number
    result = calculator._format_result(1e10)
    assert "e" in result.lower()
    
    # Very small number
    result = calculator._format_result(1e-10)
    assert "e" in result.lower()
    
    # Special values
    result = calculator._format_result(float('inf'))
    assert "infinity" in result.lower()
    
    result = calculator._format_result(float('nan'))
    assert "nan" in result.lower()
    
    # Boolean
    result = calculator._format_result(True)
    assert result == "True"


def test_calculator_security(calculator):
    """Test that unsafe operations are blocked."""
    # Attempt to use built-in functions
    with pytest.raises(ToolError) as excinfo:
        calculator.run(expression="__import__('os').system('echo hack')")
    assert "Failed to evaluate expression" in str(excinfo.value)
    
    # Attempt to access globals
    with pytest.raises(ToolError) as excinfo:
        calculator.run(expression="globals()")
    assert "Failed to evaluate expression" in str(excinfo.value)
    
    # Attempt to define a function
    with pytest.raises(ToolError) as excinfo:
        calculator.run(expression="def f(): return 1")
    assert "Failed to evaluate expression" in str(excinfo.value)
    
    # Attempt with exec
    with pytest.raises(ToolError) as excinfo:
        calculator.run(expression="exec('print(1)')")
    assert "Failed to evaluate expression" in str(excinfo.value)


@pytest.mark.asyncio
async def test_calculator_async(calculator):
    """Test the async execution path."""
    # Calculator doesn't have a true async implementation, but we test the interface
    result = await calculator.arun(expression="2 + 2")
    assert result["result"] == 4
    assert "expression" in result
    assert "simplified" in result
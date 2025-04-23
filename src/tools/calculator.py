"""
Calculator Tool - High-Performance Implementation.

This module provides a safe arithmetic calculator tool that can evaluate
mathematical expressions with proper validation and sanitization.
"""

import ast
import math
import operator
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator

from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.tools.base import BaseTool
from src.tools.registry import register_tool

logger = get_logger(__name__)


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""
    
    expression: str = Field(
        ...,
        description="The mathematical expression to evaluate"
    )
    
    @validator("expression")
    def validate_expression(cls, v: str) -> str:
        """Validate the expression to prevent code execution."""
        if not v or not isinstance(v, str):
            raise ValueError("Expression must be a non-empty string")
        
        # Basic validation
        if len(v) > 1000:
            raise ValueError("Expression is too long (max 1000 chars)")
        
        # Remove whitespace
        v = v.strip()
        
        # Additional security checks could be added here
        return v


@register_tool()
class CalculatorTool(BaseTool):
    """
    A tool for safely evaluating mathematical expressions.
    
    This tool provides a secure way to evaluate arithmetic expressions
    without allowing arbitrary code execution.
    """
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Evaluates mathematical expressions. Can handle basic arithmetic, trigonometry, and other math functions."
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return CalculatorInput
    
    def _run(self, **kwargs: Any) -> Any:
        """Execute the calculator synchronously."""
        expression = kwargs["expression"]
        return self._evaluate_expression(expression)
    
    async def _arun(self, **kwargs: Any) -> Any:
        """Execute the calculator asynchronously."""
        # Calculator is CPU-bound so no async benefit
        return self._run(**kwargs)
    
    def _evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: The string expression to evaluate.
            
        Returns:
            A dictionary with the result and intermediate steps.
            
        Raises:
            ToolError: If the expression cannot be evaluated safely.
        """
        logger.debug(f"Evaluating expression: {expression}")
        
        try:
            # Parse the expression into an AST
            node = ast.parse(expression, mode="eval")
            
            # Set up environment with safe operations
            safe_env = {
                # Math functions
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "asin": math.asin,
                "acos": math.acos,
                "atan": math.atan,
                "atan2": math.atan2,
                "sinh": math.sinh,
                "cosh": math.cosh,
                "tanh": math.tanh,
                "exp": math.exp,
                "log": math.log,
                "log10": math.log10,
                "sqrt": math.sqrt,
                "pow": pow,
                "abs": abs,
                "ceil": math.ceil,
                "floor": math.floor,
                "round": round,
                "pi": math.pi,
                "e": math.e,
                # Constants
                "nan": float("nan"),
                "inf": float("inf"),
            }
            
            # Evaluate safely
            result = self._safe_eval(node, safe_env)
            
            # Create a structured response
            return {
                "result": result,
                "expression": expression,
                "simplified": self._format_result(result)
            }
            
        except Exception as e:
            logger.warning(
                f"Calculator evaluation failed: {str(e)}",
                extra={"expression": expression, "error": str(e)},
                exc_info=e
            )
            
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to evaluate expression: {str(e)}",
                details={"expression": expression, "error": str(e)},
                original_error=e,
                tool_name=self.name
            )
    
    def _safe_eval(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        """
        Safely evaluate an AST node.
        
        Args:
            node: The AST node to evaluate.
            env: The environment with allowed functions.
            
        Returns:
            The evaluation result.
            
        Raises:
            ValueError: If the node contains unsafe operations.
        """
        # Validate node type for safety
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body, env)
        
        # Binary operations
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left, env)
            right = self._safe_eval(node.right, env)
            
            # Map operators to functions
            op_map = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            
            op_func = op_map.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            try:
                return op_func(left, right)
            except ZeroDivisionError:
                if type(node.op) in (ast.Div, ast.FloorDiv):
                    return float("inf") if left >= 0 else float("-inf")
                raise
            
        # Unary operations
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand, env)
            
            # Map operators to functions
            op_map = {
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }
            
            op_func = op_map.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            
            return op_func(operand)
        
        # Function calls
        elif isinstance(node, ast.Call):
            # Only allow known math functions
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls to known math functions are allowed")
            
            func_name = node.func.id
            if func_name not in env:
                raise ValueError(f"Function not allowed: {func_name}")
            
            # Evaluate arguments
            args = [self._safe_eval(arg, env) for arg in node.args]
            
            # Call the function
            return env[func_name](*args)
        
        # Constants
        elif isinstance(node, ast.Constant):
            return node.value
        
        # Names (variables)
        elif isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"Unknown variable: {node.id}")
            return env[node.id]
        
        # Tuples (for multi-argument functions)
        elif isinstance(node, ast.Tuple):
            return tuple(self._safe_eval(elt, env) for elt in node.elts)
        
        # Lists
        elif isinstance(node, ast.List):
            return [self._safe_eval(elt, env) for elt in node.elts]
        
        # Comparison operators
        elif isinstance(node, ast.Compare):
            # Get left operand
            left = self._safe_eval(node.left, env)
            
            # Supported comparison operators
            op_map = {
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
            }
            
            # Evaluate each comparison in sequence
            result = True
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                # Get operator function
                op_func = op_map.get(type(op))
                if op_func is None:
                    raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
                
                # Get right operand
                right = self._safe_eval(comparator, env)
                
                # For the first comparison, use left from the node
                # For subsequent comparisons, use the right from the previous comparison
                if i == 0:
                    result = op_func(left, right)
                else:
                    # This handles chained comparisons (e.g., a < b < c)
                    result = result and op_func(left, right)
                
                # Update left for the next comparison
                left = right
                
                # Short-circuit if result is already False
                if not result:
                    break
            
            return result
        
        # Parentheses (handled by AST structure)
        
        # Anything else is not allowed
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")
    
    def _format_result(self, result: Any) -> str:
        """
        Format the result for human readability.
        
        Args:
            result: The raw calculation result.
            
        Returns:
            A formatted string representation.
        """
        # Handle different result types
        if isinstance(result, (int, float)):
            # Check for special values FIRST
            if isinstance(result, float):
                if math.isnan(result):
                    return "NaN (Not a Number)"
                if math.isinf(result):
                    return "Infinity" if result > 0 else "-Infinity"
            
            # Check for close-to-integer values
            if isinstance(result, float) and abs(result - round(result)) < 1e-10:
                return str(int(round(result)))
            
            # Format floats with appropriate precision
            if isinstance(result, float):
                # Determine appropriate precision
                abs_result = abs(result)
                if abs_result >= 1e6 or (abs_result < 1e-4 and abs_result > 0):
                    # Scientific notation for very large or small numbers
                    return f"{result:.6e}"
                elif abs_result >= 1000:
                    # Fewer decimals for larger numbers
                    return f"{result:.2f}"
                else:
                    # More decimals for regular numbers
                    return f"{result:.6f}".rstrip("0").rstrip(".")
            
            # For very large integers, use scientific notation
            if isinstance(result, int) and abs(result) >= 1e10:
                return f"{result:.6e}"
                
            # Regular integers
            return str(result)
        
        # Boolean results
        elif isinstance(result, bool):
            return "True" if result else "False"
        
        # Other types
        return str(result)
import ast
import math
import operator
import json
from typing import Any, Dict, Type

from pydantic import BaseModel, Field, field_validator

from src.tools.base import BaseTool
from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger
from src.services.tool_manager import register_tool

logger = get_logger(__name__)


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="The mathematical expression to evaluate")

    @field_validator("expression")
    def validate_expression(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("Expression must be a non-empty string")
        if len(v) > 1000:
            raise ValueError("Expression is too long (max 1000 chars)")
        v = v.strip()
        if not v:
            raise ValueError("Expression cannot be empty after stripping whitespace")
        return v


@register_tool()
class CalculatorTool(BaseTool):
    """
    Evaluates mathematical expressions safely.
    Can handle basic arithmetic (+, -, *, /), powers (**), trigonometry (sin, cos, tan, asin, acos, atan, atan2),
    logarithms (log, log10), square root (sqrt), exponentiation (exp), absolute value (abs),
    ceiling (ceil), floor (floor), rounding (round), and constants (pi, e).
    Input 'expression' should be a valid Python-like math expression string.
    Example: "3 * (sin(pi/2) + 1)"
    """

    name: str = "calculator"
    description: str = (
        "Evaluates mathematical expressions like '2 + 2', 'sqrt(16)', '3 * (sin(pi/2) + 1)', or 'pow(2, 5)'. "
        "Handles basic arithmetic, trigonometry, powers, logarithms, constants (pi, e). "
        "Input 'expression' must be a valid math expression string."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """
        Synchronously evaluates the mathematical expression.
        LangChain Tool 실행 메커니즘은 이 메서드를 호출합니다.
        kwargs에서 'expression' 인자를 직접 받아 사용합니다.
        """
        logger.debug(f"CalculatorTool: Attempting to evaluate expression: {expression}")
        try:
            node: ast.AST = ast.parse(expression, mode="eval")

            safe_env: Dict[str, Any] = {
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
            }

            result: Any = self._safe_eval(node, safe_env)
            formatted_result: str = self._format_result(result)

            logger.info(
                f"CalculatorTool: Evaluated '{expression}' = {formatted_result} (raw: {result})"
            )

            return formatted_result

        except SyntaxError as se:
            logger.warning(
                f"CalculatorTool: Syntax error evaluating expression: {expression}",
                exc_info=se,
            )
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message=f"Invalid mathematical expression syntax: {str(se)}",
                details={"expression": expression, "error": str(se)},
                original_error=se,
                tool_name=self.name,
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            RecursionError,
            OverflowError,
            ZeroDivisionError,
        ) as eval_err:
            logger.warning(
                f"CalculatorTool: Evaluation failed for expression '{expression}': {str(eval_err)}"
            )
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to evaluate expression: {str(eval_err)}",
                details={"expression": expression, "error": str(eval_err)},
                original_error=eval_err,
                tool_name=self.name,
            )
        except Exception as e:
            logger.exception(
                f"CalculatorTool: Unexpected error evaluating expression '{expression}': {str(e)}"
            )
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Unexpected error during calculation: {str(e)}",
                details={"expression": expression},
                original_error=e,
                tool_name=self.name,
            )

    async def _arun(self, expression: str) -> str:
        """
        Asynchronously evaluates the mathematical expression.
        Since the calculation is CPU-bound and likely fast, running the sync version is usually sufficient.
        """

        try:
            return self._run(expression=expression)
        except ToolError:
            raise
        except Exception as e:
            logger.exception(
                f"CalculatorTool: Unexpected error during async evaluation wrapper for '{expression}': {str(e)}"
            )
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Unexpected async wrapper error: {str(e)}",
                details={"expression": expression},
                original_error=e,
                tool_name=self.name,
            )

    def _safe_eval(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body, env)
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left, env)
            right = self._safe_eval(node.right, env)
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
                raise ValueError(
                    f"Unsupported binary operator: {type(node.op).__name__}"
                )

            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand, env)
            op_map = {ast.USub: operator.neg, ast.UAdd: operator.pos}
            op_func = op_map.get(type(node.op))
            if op_func is None:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
            return op_func(operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    "Function calls must be direct name calls (e.g., sin(x))."
                )
            func_name = node.func.id
            if func_name not in env:
                raise ValueError(f"Function not allowed: {func_name}")
            args = [self._safe_eval(arg, env) for arg in node.args]
            return env[func_name](*args)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, complex, bool)):
                pass
            return node.value

        elif isinstance(node, ast.Name):
            var_name = node.id
            if var_name not in env:
                raise ValueError(f"Unknown or disallowed variable/constant: {var_name}")
            return env[var_name]

        else:
            raise ValueError(
                f"Unsupported operation or node type: {type(node).__name__}"
            )

    def _format_result(self, result: Any) -> str:
        if isinstance(result, (int, float)):
            if isinstance(result, float):
                if math.isnan(result):
                    return "NaN"
                if math.isinf(result):
                    return "Infinity" if result > 0 else "-Infinity"

                if abs(result - round(result)) < 1e-10:
                    return str(int(round(result)))

                abs_result = abs(result)
                if abs_result >= 1e12 or (abs_result < 1e-6 and abs_result > 0):
                    return f"{result:.6e}"
                else:
                    formatted = f"{result:.6f}".rstrip("0").rstrip(".")
                    return formatted if "." in formatted else formatted + ".0"

            if isinstance(result, int) and abs(result) >= 1e12:
                return f"{result:.6e}"
            return str(result)
        elif isinstance(result, bool):
            return str(result)
        else:
            try:
                return json.dumps(result)
            except TypeError:
                return str(result)

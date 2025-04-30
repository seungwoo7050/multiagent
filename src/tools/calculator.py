import ast
import math
import operator
from typing import Any, Dict, List, Optional, Set, Type, Union
from pydantic import BaseModel, Field, field_validator
from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.tools.base import BaseTool
from src.tools.registry import register_tool
logger = get_logger(__name__)

class CalculatorInput(BaseModel):
    expression: str = Field(..., description='The mathematical expression to evaluate')

    @field_validator('expression')
    def validate_expression(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError('Expression must be a non-empty string')
        if len(v) > 1000:
            raise ValueError('Expression is too long (max 1000 chars)')
        v = v.strip()
        return v

@register_tool()
class CalculatorTool(BaseTool):

    @property
    def name(self) -> str:
        return 'calculator'

    @property
    def description(self) -> str:
        return 'Evaluates mathematical expressions. Can handle basic arithmetic, trigonometry, and other math functions.'

    @property
    def args_schema(self) -> Type[BaseModel]:
        return CalculatorInput

    def _run(self, **kwargs: Any) -> Any:
        expression = kwargs['expression']
        return self._evaluate_expression(expression)

    async def _arun(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def _evaluate_expression(self, expression: str) -> Dict[str, Any]:
        logger.debug(f'Attempting to evaluate expression: {expression}')
        try:
            node: ast.AST = ast.parse(expression, mode='eval')
            safe_env: Dict[str, Any] = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2, 'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh, 'exp': math.exp, 'log': math.log, 'log10': math.log10, 'sqrt': math.sqrt, 'pow': pow, 'abs': abs, 'ceil': math.ceil, 'floor': math.floor, 'round': round, 'pi': math.pi, 'e': math.e, 'nan': float('nan'), 'inf': float('inf')}
            result: Any = self._safe_eval(node, safe_env)
            simplified_result: str = self._format_result(result)
            logger.info(f"Evaluated expression '{expression}' = {simplified_result} (raw: {result})")
            return {'result': result, 'expression': expression, 'simplified': simplified_result}
        except SyntaxError as se:
            logger.warning(f'Syntax error evaluating expression: {expression}', exc_info=se)
            raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=f'Invalid mathematical expression syntax: {str(se)}', details={'expression': expression, 'error': str(se)}, original_error=se, tool_name=self.name)
        except Exception as e:
            logger.warning(f"Calculator evaluation failed for expression '{expression}': {str(e)}", extra={'expression': expression, 'error': str(e)}, exc_info=e)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Failed to evaluate expression: {str(e)}', details={'expression': expression, 'error': str(e)}, original_error=e, tool_name=self.name)

    def _safe_eval(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body, env)
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left, env)
            right = self._safe_eval(node.right, env)
            op_map = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
            op_func = op_map.get(type(node.op))
            if op_func is None:
                raise ValueError(f'Unsupported binary operator: {type(node.op).__name__}')
            try:
                return op_func(left, right)
            except ZeroDivisionError:
                if type(node.op) in (ast.Div, ast.FloorDiv):
                    logger.debug('Division by zero encountered, returning infinity.')
                    return float('inf') if left >= 0 else float('-inf')
                else:
                    raise
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand, env)
            op_map = {ast.USub: operator.neg, ast.UAdd: operator.pos}
            op_func = op_map.get(type(node.op))
            if op_func is None:
                raise ValueError(f'Unsupported unary operator: {type(node.op).__name__}')
            return op_func(operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError('Function calls must be direct name calls (e.g., sin(x)), not method calls or complex expressions.')
            func_name = node.func.id
            if func_name not in env:
                raise ValueError(f'Function not allowed: {func_name}')
            args = [self._safe_eval(arg, env) for arg in node.args]
            try:
                return env[func_name](*args)
            except Exception as func_call_err:
                logger.debug(f"Error calling function '{func_name}' with args {args}: {func_call_err}")
                raise ValueError(f"Error in function '{func_name}': {str(func_call_err)}")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, complex, bool)):
                logger.warning(f'Encountered non-numeric constant: {node.value} ({type(node.value)}). Evaluation might fail.')
            return node.value
        elif isinstance(node, ast.Name):
            var_name = node.id
            if var_name not in env:
                raise ValueError(f'Unknown or disallowed variable/constant: {var_name}')
            return env[var_name]
        elif isinstance(node, ast.Tuple):
            return tuple((self._safe_eval(elt, env) for elt in node.elts))
        elif isinstance(node, ast.List):
            return [self._safe_eval(elt, env) for elt in node.elts]
        elif isinstance(node, ast.Compare):
            left = self._safe_eval(node.left, env)
            result = True
            op_map = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                op_func = op_map.get(type(op))
                if op_func is None:
                    raise ValueError(f'Unsupported comparison operator: {type(op).__name__}')
                right = self._safe_eval(comparator, env)
                current_comparison_result = op_func(left, right)
                if i > 0:
                    result = result and current_comparison_result
                else:
                    result = current_comparison_result
                left = right
                if not result:
                    break
            return result
        else:
            raise ValueError(f'Unsupported operation or node type in expression: {type(node).__name__}')

    def _format_result(self, result: Any) -> str:
        if isinstance(result, (int, float)):
            if isinstance(result, float):
                if math.isnan(result):
                    return 'NaN (Not a Number)'
                if math.isinf(result):
                    return 'Infinity' if result > 0 else '-Infinity'
            if isinstance(result, float) and abs(result - round(result)) < 1e-10:
                return str(int(round(result)))
            if isinstance(result, float):
                abs_result = abs(result)
                if abs_result >= 1000000.0 or (abs_result < 0.0001 and abs_result > 0):
                    return f'{result:.6e}'
                elif abs_result >= 1000:
                    return f'{result:.2f}'
                else:
                    formatted = f'{result:.6f}'
                    return formatted.rstrip('0').rstrip('.')
            if isinstance(result, int) and abs(result) >= 10000000000.0:
                return f'{result:.6e}'
            return str(result)
        elif isinstance(result, bool):
            return 'True' if result else 'False'
        else:
            return str(result)
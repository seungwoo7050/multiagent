# src/tools/calculator.py
import ast
import math
import operator
from typing import Any, Dict, Type, Optional # Optional 추가

from pydantic import BaseModel, Field, field_validator
# BaseTool 및 LangChain 관련 import 추가
from src.tools.base import BaseTool
# from langchain_core.tools import Tool # 필요시 LangChain Tool 직접 사용 가능

from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger
from src.services.tool_manager import register_tool

logger = get_logger(__name__)

# 입력 스키마는 Pydantic 모델로 유지
class CalculatorInput(BaseModel):
    expression: str = Field(..., description='The mathematical expression to evaluate')

    @field_validator('expression')
    def validate_expression(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError('Expression must be a non-empty string')
        if len(v) > 1000:
            raise ValueError('Expression is too long (max 1000 chars)')
        v = v.strip()
        if not v:
             raise ValueError('Expression cannot be empty after stripping whitespace')
        return v

# @register_tool() 데코레이터 유지 (ToolManager 사용)
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
    # 클래스 변수로 name, description, args_schema 정의
    name: str = "calculator"
    description: str = (
        "Evaluates mathematical expressions like '2 + 2', 'sqrt(16)', '3 * (sin(pi/2) + 1)', or 'pow(2, 5)'. "
        "Handles basic arithmetic, trigonometry, powers, logarithms, constants (pi, e). "
        "Input 'expression' must be a valid math expression string."
    )
    args_schema: Type[BaseModel] = CalculatorInput
    # return_direct: bool = False # 필요시 설정 (LangChain BaseTool 속성)

    # _evaluate_expression 로직을 _run 안으로 통합하고 ToolError 처리 추가
    def _run(self, expression: str) -> str:
        """
        Synchronously evaluates the mathematical expression.
        LangChain Tool 실행 메커니즘은 이 메서드를 호출합니다.
        kwargs에서 'expression' 인자를 직접 받아 사용합니다.
        """
        logger.debug(f"CalculatorTool: Attempting to evaluate expression: {expression}")
        try:
            # ast.parse는 비교적 안전하지만, 복잡한 코드는 여전히 위험할 수 있음
            # 주의: 매우 복잡하거나 악의적인 입력에 대한 완전한 안전을 보장하지는 않음
            node: ast.AST = ast.parse(expression, mode='eval')

            # 허용되는 함수 및 상수 정의 (기존 로직 유지)
            safe_env: Dict[str, Any] = {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
                'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
                'exp': math.exp, 'log': math.log, 'log10': math.log10,
                'sqrt': math.sqrt, 'pow': pow, 'abs': abs,
                'ceil': math.ceil, 'floor': math.floor, 'round': round,
                'pi': math.pi, 'e': math.e,
                # 'nan': float('nan'), 'inf': float('inf') # 필요시 추가
                # 사용자 정의 함수 추가 가능 (주의해서)
            }

            result: Any = self._safe_eval(node, safe_env)
            formatted_result: str = self._format_result(result)

            logger.info(f"CalculatorTool: Evaluated '{expression}' = {formatted_result} (raw: {result})")
            # 결과는 항상 문자열로 반환 (LangChain Tool 기본 요구사항)
            # JSON 문자열로 구조화된 정보를 반환할 수도 있음
            # 예: return json.dumps({'result': result, 'simplified': formatted_result})
            return formatted_result

        except SyntaxError as se:
            logger.warning(f'CalculatorTool: Syntax error evaluating expression: {expression}', exc_info=se)
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR,
                message=f'Invalid mathematical expression syntax: {str(se)}',
                details={'expression': expression, 'error': str(se)},
                original_error=se,
                tool_name=self.name
            )
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError, ZeroDivisionError) as eval_err:
             # _safe_eval 내부 또는 여기서 발생할 수 있는 다양한 실행 오류 처리
             logger.warning(f"CalculatorTool: Evaluation failed for expression '{expression}': {str(eval_err)}")
             raise ToolError(
                 code=ErrorCode.TOOL_EXECUTION_ERROR,
                 message=f'Failed to evaluate expression: {str(eval_err)}',
                 details={'expression': expression, 'error': str(eval_err)},
                 original_error=eval_err,
                 tool_name=self.name
             )
        except Exception as e:
            # 예상치 못한 다른 모든 예외 처리
            logger.exception(f"CalculatorTool: Unexpected error evaluating expression '{expression}': {str(e)}")
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f'Unexpected error during calculation: {str(e)}',
                details={'expression': expression},
                original_error=e,
                tool_name=self.name
            )

    # _arun 구현 (BaseTool 요구사항 충족)
    async def _arun(self, expression: str) -> str:
        """
        Asynchronously evaluates the mathematical expression.
        Since the calculation is CPU-bound and likely fast, running the sync version is usually sufficient.
        """
        # 계산 로직 자체는 CPU 바운드이므로, 별도 스레드에서 실행하는 것이
        # 이벤트 루프를 막지 않는 좋은 방법일 수 있으나, 간단한 계산은 동기 호출도 가능.
        # 여기서는 간단하게 동기 메서드 호출. 복잡하다면 asyncio.to_thread 사용 고려.
        try:
            # loop = asyncio.get_event_loop()
            # return await loop.run_in_executor(None, self._run, expression=expression)
            return self._run(expression=expression) # 직접 호출 (매우 빠른 연산 가정)
        except ToolError:
             raise # ToolError는 그대로 전달
        except Exception as e:
             # _run 에서 처리되지 않은 예외가 혹시 있다면 여기서 처리
             logger.exception(f"CalculatorTool: Unexpected error during async evaluation wrapper for '{expression}': {str(e)}")
             raise ToolError(
                 code=ErrorCode.TOOL_EXECUTION_ERROR,
                 message=f'Unexpected async wrapper error: {str(e)}',
                 details={'expression': expression},
                 original_error=e,
                 tool_name=self.name
             )

    # _safe_eval과 _format_result는 내부 헬퍼 메서드로 유지
    def _safe_eval(self, node: ast.AST, env: Dict[str, Any]) -> Any:
        # ... (기존 _safe_eval 코드 유지) ...
        # ZeroDivisionError 처리 등 개선 가능
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body, env)
        elif isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left, env)
            right = self._safe_eval(node.right, env)
            op_map = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
            op_func = op_map.get(type(node.op))
            if op_func is None:
                raise ValueError(f'Unsupported binary operator: {type(node.op).__name__}')
            # ZeroDivisionError를 여기서 잡아서 처리하거나, _run에서 잡도록 둘 수 있음
            # 여기서는 위로 던져서 _run에서 ToolError로 변환되도록 함
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand, env)
            op_map = {ast.USub: operator.neg, ast.UAdd: operator.pos}
            op_func = op_map.get(type(node.op))
            if op_func is None:
                raise ValueError(f'Unsupported unary operator: {type(node.op).__name__}')
            return op_func(operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError('Function calls must be direct name calls (e.g., sin(x)).')
            func_name = node.func.id
            if func_name not in env:
                raise ValueError(f'Function not allowed: {func_name}')
            args = [self._safe_eval(arg, env) for arg in node.args]
            return env[func_name](*args) # 함수 호출 시 발생하는 예외는 상위로 전달
        elif isinstance(node, ast.Constant): # Python 3.8+
            # 숫자, bool 외의 상수는 허용하지 않도록 제한 강화 가능
            if not isinstance(node.value, (int, float, complex, bool)):
                 # logger.warning(f'Non-numeric constant encountered: {node.value}. Allowing it.')
                 pass # 혹은 여기서 ValueError 발생
            return node.value
        # Python 3.8 미만에서는 ast.Num, ast.Str, ast.NameConstant 등 사용
        # elif isinstance(node, ast.Num): return node.n
        # elif isinstance(node, ast.NameConstant): return node.value
        elif isinstance(node, ast.Name):
            var_name = node.id
            if var_name not in env:
                raise ValueError(f'Unknown or disallowed variable/constant: {var_name}')
            return env[var_name]
        # Compare, Tuple, List 등은 필요시 추가 지원 (기존 코드 참고)
        else:
            raise ValueError(f'Unsupported operation or node type: {type(node).__name__}')

    def _format_result(self, result: Any) -> str:
        # ... (기존 _format_result 코드 유지) ...
        if isinstance(result, (int, float)):
            if isinstance(result, float):
                if math.isnan(result): return 'NaN'
                if math.isinf(result): return 'Infinity' if result > 0 else '-Infinity'
                # 부동소수점 정밀도 문제 해결 시도
                if abs(result - round(result)) < 1e-10:
                    return str(int(round(result)))
                # 너무 크거나 작은 수는 지수 표기법 사용
                abs_result = abs(result)
                if abs_result >= 1e12 or (abs_result < 1e-6 and abs_result > 0):
                    return f'{result:.6e}'
                else:
                    # 유효숫자를 고려하여 소수점 이하 자릿수 조절 (예: 6자리)
                    formatted = f'{result:.6f}'.rstrip('0').rstrip('.')
                    return formatted if '.' in formatted else formatted + '.0' # 정수 뒤에 .0 추가 방지
            # 매우 큰 정수도 지수 표기법 사용 가능
            if isinstance(result, int) and abs(result) >= 1e12:
                 return f'{result:.6e}'
            return str(result)
        elif isinstance(result, bool):
            return str(result) # 'True' or 'False'
        else:
            # 다른 타입(예: 리스트, 튜플 - _safe_eval에서 허용 시) 처리
            try:
                 # 복잡한 객체는 JSON으로 변환 시도
                 return json.dumps(result)
            except TypeError:
                 return str(result) # 최후의 수단
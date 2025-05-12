Q=Exception
P=TypeError
O=float
N=round
M=int
L=abs
J='expression'
I=type
E=ValueError
D=isinstance
C=str
import ast as B,math as A,operator as F,json
from typing import Any,Dict,Type
from pydantic import BaseModel as R,Field,field_validator as S
from src.tools.base import BaseTool as T
from src.config.errors import ErrorCode as K,ToolError as H
from src.utils.logger import get_logger as U
from src.services.tool_manager import register_tool as V
G=U(__name__)
class W(R):
	expression=Field(...,description='The mathematical expression to evaluate')
	@S(J)
	def validate_expression(cls,v):
		if not v or not D(v,C):raise E('Expression must be a non-empty string')
		if len(v)>1000:raise E('Expression is too long (max 1000 chars)')
		v=v.strip()
		if not v:raise E('Expression cannot be empty after stripping whitespace')
		return v
@V()
class X(T):
	name='calculator';description="Evaluates mathematical expressions like '2 + 2', 'sqrt(16)', '3 * (sin(pi/2) + 1)', or 'pow(2, 5)'. Handles basic arithmetic, trigonometry, powers, logarithms, constants (pi, e). Input 'expression' must be a valid math expression string.";args_schema=W
	def _run(F,expression):
		T='error';D=expression;G.debug(f"CalculatorTool: Attempting to evaluate expression: {D}")
		try:U=B.parse(D,mode='eval');V={'sin':A.sin,'cos':A.cos,'tan':A.tan,'asin':A.asin,'acos':A.acos,'atan':A.atan,'atan2':A.atan2,'sinh':A.sinh,'cosh':A.cosh,'tanh':A.tanh,'exp':A.exp,'log':A.log,'log10':A.log10,'sqrt':A.sqrt,'pow':pow,'abs':L,'ceil':A.ceil,'floor':A.floor,'round':N,'pi':A.pi,'e':A.e};R=F._safe_eval(U,V);S=F._format_result(R);G.info(f"CalculatorTool: Evaluated '{D}' = {S} (raw: {R})");return S
		except SyntaxError as I:G.warning(f"CalculatorTool: Syntax error evaluating expression: {D}",exc_info=I);raise H(code=K.TOOL_VALIDATION_ERROR,message=f"Invalid mathematical expression syntax: {C(I)}",details={J:D,T:C(I)},original_error=I,tool_name=F.name)
		except(E,P,KeyError,RecursionError,OverflowError,ZeroDivisionError)as M:G.warning(f"CalculatorTool: Evaluation failed for expression '{D}': {C(M)}");raise H(code=K.TOOL_EXECUTION_ERROR,message=f"Failed to evaluate expression: {C(M)}",details={J:D,T:C(M)},original_error=M,tool_name=F.name)
		except Q as O:G.exception(f"CalculatorTool: Unexpected error evaluating expression '{D}': {C(O)}");raise H(code=K.TOOL_EXECUTION_ERROR,message=f"Unexpected error during calculation: {C(O)}",details={J:D},original_error=O,tool_name=F.name)
	async def _arun(D,expression):
		A=expression
		try:return D._run(expression=A)
		except H:raise
		except Q as B:G.exception(f"CalculatorTool: Unexpected error during async evaluation wrapper for '{A}': {C(B)}");raise H(code=K.TOOL_EXECUTION_ERROR,message=f"Unexpected async wrapper error: {C(B)}",details={J:A},original_error=B,tool_name=D.name)
	def _safe_eval(G,node,env):
		C=env;A=node
		if D(A,B.Expression):return G._safe_eval(A.body,C)
		elif D(A,B.BinOp):
			N=G._safe_eval(A.left,C);P=G._safe_eval(A.right,C);J={B.Add:F.add,B.Sub:F.sub,B.Mult:F.mul,B.Div:F.truediv,B.FloorDiv:F.floordiv,B.Mod:F.mod,B.Pow:F.pow};H=J.get(I(A.op))
			if H is None:raise E(f"Unsupported binary operator: {I(A.op).__name__}")
			return H(N,P)
		elif D(A,B.UnaryOp):
			Q=G._safe_eval(A.operand,C);J={B.USub:F.neg,B.UAdd:F.pos};H=J.get(I(A.op))
			if H is None:raise E(f"Unsupported unary operator: {I(A.op).__name__}")
			return H(Q)
		elif D(A,B.Call):
			if not D(A.func,B.Name):raise E('Function calls must be direct name calls (e.g., sin(x)).')
			K=A.func.id
			if K not in C:raise E(f"Function not allowed: {K}")
			R=[G._safe_eval(A,C)for A in A.args];return C[K](*R)
		elif D(A,B.Constant):
			if not D(A.value,(M,O,complex,bool)):0
			return A.value
		elif D(A,B.Name):
			L=A.id
			if L not in C:raise E(f"Unknown or disallowed variable/constant: {L}")
			return C[L]
		else:raise E(f"Unsupported operation or node type: {I(A).__name__}")
	def _format_result(G,result):
		B=result
		if D(B,(M,O)):
			if D(B,O):
				if A.isnan(B):return'NaN'
				if A.isinf(B):return'Infinity'if B>0 else'-Infinity'
				if L(B-N(B))<1e-10:return C(M(N(B)))
				E=L(B)
				if E>=1e12 or E<1e-06 and E>0:return f"{B:.6e}"
				else:F=f"{B:.6f}".rstrip('0').rstrip('.');return F if'.'in F else F+'.0'
			if D(B,M)and L(B)>=1e12:return f"{B:.6e}"
			return C(B)
		elif D(B,bool):return C(B)
		else:
			try:return json.dumps(B)
			except P:return C(B)
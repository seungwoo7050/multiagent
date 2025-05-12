I=Exception
E=True
A=None
C=str
import abc,inspect as F
from typing import Any,Awaitable,Callable,Dict,Optional,Tuple,Type
from pydantic import BaseModel as G,create_model as D
from langchain_core.tools import BaseTool as K
from src.config.errors import ErrorCode as J,ToolError as H
from src.utils.logger import get_logger as L
B=L(__name__)
def N(name='DefaultToolInputSchema'):return D(name,__base__=G)
class M(K,abc.ABC):
	name:0;description:0;args_schema=A;return_direct=False
	@abc.abstractmethod
	def _run(self,*A,**B):raise NotImplementedError(f"{self.__class__.__name__}._run not implemented")
	async def _arun(A,*C,**D):B.debug(f"Default _arun called for {A.name}, will likely execute _run in executor.")
	@classmethod
	def get_empty_args_schema(A,name='EmptyArgsSchema'):return D(name,__base__=G)
	class Config:arbitrary_types_allowed=E
class O(M):
	func:0;coroutine=A
	def __init__(A,name,description,func,coroutine=A,args_schema=A,**D):C=coroutine;B=func;E=args_schema or A._infer_args_schema(B,name);super().__init__(name=name,description=description,args_schema=E,func=B,coroutine=C,**D);A.func=B;A.coroutine=C
	def _run(A,*K,**F):
		try:G=A.func(**F);return C(G)
		except I as D:B.error(f"Error executing dynamic tool '{A.name}': {D}",exc_info=E);raise H(message=f"Error in tool '{A.name}': {C(D)}",tool_name=A.name,original_error=D,code=J.TOOL_EXECUTION_ERROR)
	async def _arun(A,*K,**G):
		try:
			if A.coroutine:F=await A.coroutine(**G);return C(F)
			else:B.debug(f"Executing synchronous function '{A.func.__name__}' for async call in DynamicTool '{A.name}'.");F=A._run(*K,**G);return F
		except I as D:
			if not isinstance(D,H):B.error(f"Error executing async dynamic tool '{A.name}': {D}",exc_info=E);raise H(message=f"Error in async tool '{A.name}': {C(D)}",tool_name=A.name,original_error=D,code=J.TOOL_EXECUTION_ERROR)
			else:raise
	@staticmethod
	def _infer_args_schema(func,name):
		C=name;B.debug(f"Inferring args schema for dynamic tool '{C}' from function '{func.__name__}'")
		try:
			K=F.signature(func);H={}
			for(L,A)in K.parameters.items():
				if A.kind in(A.VAR_POSITIONAL,A.VAR_KEYWORD):continue
				M=A.annotation if A.annotation!=F.Parameter.empty else Any;N=A.default if A.default!=F.Parameter.empty else...;H[L]=M,N
			O=f"{C.title().replace("_","").replace(" ","")}Schema";J=D(O,**H);B.debug(f"Inferred schema for '{C}': {J.schema()}");return J
		except I as P:B.warning(f"Could not infer args_schema for tool '{C}': {P}. Using empty schema.",exc_info=E);return D(f"{C.title().replace(" ","")}Schema",__base__=G)
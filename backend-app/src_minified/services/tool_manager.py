a=False
Z='args_schema_summary'
Y='tool_name'
X=isinstance
R=True
P='description'
O='global_tools'
N=issubclass
K=getattr
J=Exception
I=str
H='name'
F=None
import importlib as S,inspect as L,pkgutil as T,sys as C,threading as U
from pathlib import Path as V
from typing import Any,Callable as b,Dict,List,Optional,Set,Type,TypeVar as B,get_origin as c
from pydantic import BaseModel as Q
from src.tools.base import BaseTool as M
from src.config.errors import ErrorCode as G,ToolError as E
from src.utils.logger import get_logger as d
A=d(__name__)
g=B('T',bound=M)
h=B('F',bound=b[...,Any])
class e:
	def __init__(B,name=O):B._name=name;B._tools={};B._instance_cache={};B._lock=U.RLock();A.debug(f"ToolManager '{B._name}' initialized.")
	@property
	def name(self):return self._name
	def register(D,tool_cls):
		L='class';B=tool_cls;A.debug(f"Attempting to register tool class: {B.__name__} with ToolManager '{D.name}'")
		try:
			if not N(B,M):raise E(code=G.TOOL_VALIDATION_ERROR,message=f"Tool class '{B.__name__}' must inherit from BaseTool.",details={L:B.__name__})
			C=K(B,H,F)
			if not C or not X(C,I):
				try:Q=B();C=Q.name
				except J as P:raise E(code=G.TOOL_VALIDATION_ERROR,message=f"Could not determine 'name' for tool class '{B.__name__}': {P}",details={L:B.__name__},original_error=P)
			if not C:raise E(code=G.TOOL_VALIDATION_ERROR,message=f"Tool class '{B.__name__}' has an empty 'name' property.",details={L:B.__name__})
			with D._lock:
				if C in D._tools:
					if D._tools[C]==B:A.debug(f"Tool class '{B.__name__}' (name: '{C}') is already registered in '{D.name}'. Skipping.");return B
					else:raise E(code=G.TOOL_VALIDATION_ERROR,message=f"Tool name '{C}' is already registered by a different class '{D._tools[C].__name__}' in manager '{D.name}'. Cannot register '{B.__name__}'.",details={H:C,'new_class':B.__name__,'existing_class':D._tools[C].__name__})
				D._tools[C]=B;A.info(f"Tool '{C}' (Class: {B.__name__}) registered successfully in manager '{D.name}'.")
			return B
		except E:raise
		except J as O:A.error(f"Failed to register tool class '{B.__name__}' in manager '{D.name}' due to an unexpected error.",extra={L:B.__name__},exc_info=O);raise E(code=G.TOOL_VALIDATION_ERROR,message=f"Failed to register tool class {B.__name__} in {D.name}: {I(O)}",details={L:B.__name__},original_error=O)
	def has(A,tool_name):
		with A._lock:return tool_name in A._tools
	def get_tool_class(C,tool_name):
		B=tool_name
		with C._lock:D=C._tools.get(B)
		if D is F:I=list(C.get_names());A.warning(f"Tool class '{B}' not found in manager '{C.name}'. Available: {I}",extra={Y:B});raise E(code=G.TOOL_NOT_FOUND,message=f"Tool class '{B}' not found.",details={H:B,'available':I},tool_name=B)
		return D
	def get_tool(C,tool_name):
		B=tool_name
		with C._lock:
			if B in C._instance_cache:A.debug(f"Returning cached tool instance for: {B} from manager '{C.name}'");return C._instance_cache[B]
		A.debug(f"Tool instance cache miss for: {B} in manager '{C.name}'. Creating new instance.")
		try:
			K=C.get_tool_class(B);D=K()
			with C._lock:
				if B not in C._instance_cache:C._instance_cache[B]=D;A.debug(f"Created and cached new tool instance for: {B} in manager '{C.name}'")
				else:A.debug(f"Instance for {B} was created concurrently, using existing cache entry.");D=C._instance_cache[B]
			return D
		except E:raise
		except J as F:A.error(f"Failed to instantiate tool '{B}' in manager '{C.name}'",extra={Y:B},exc_info=F);raise E(code=G.TOOL_CREATION_ERROR,message=f"Failed to create instance of tool '{B}': {I(F)}",details={H:B},original_error=F,tool_name=B)
	def list_tools(V):
		U='args_schema';M=[]
		for(W,B)in V._tools.items():
			O={H:W,P:K(B,P,''),'class_name':B.__name__};R={};C=K(B,U,F)
			if X(C,Q):A=C.__class__
			elif L.isclass(C)and N(C,Q):A=C
			else:A=F
			if A is F:
				try:
					Y=L.signature(B.__init__)
					if all(A.default is not A.empty or A.kind in(A.VAR_POSITIONAL,A.VAR_KEYWORD)for A in list(Y.parameters.values())[1:]):
						a=B();E=K(a,U,F)
						if L.isclass(E)and N(E,Q):A=E
				except J:pass
			if A:
				for(b,S)in A.model_fields.items():
					G=S.annotation;T=c(G);D=T.__name__ if T else K(G,'__name__',I(G))
					if D=='str':D='string'
					if S.is_required():D+=' (required)'
					R[b]=D
			O[Z]=R;M.append(O)
		return M
	def get_tool_summaries_for_llm(D):
		B=[]
		for A in D.list_tools():
			if'error'not in A:C=', '.join([f"{A}: {B}"for(A,B)in A.get(Z,{}).items()]);E={H:A.get(H,'unknown'),P:f"{A.get(P,"No description.")} (Arguments: {C if C else"None"})"};B.append(E)
		return B
	def clear_cache(B):
		with B._lock:C=len(B._instance_cache);B._instance_cache.clear()
		A.debug(f"Tool instance cache cleared for manager '{B.name}' ({C} items removed).")
	def unregister(C,tool_name):
		B=tool_name
		with C._lock:
			if B not in C._tools:raise E(code=G.TOOL_NOT_FOUND,message=f"Tool '{B}' not found for unregistration in manager '{C.name}'.",details={H:B},tool_name=B)
			del C._tools[B]
			if B in C._instance_cache:del C._instance_cache[B];A.debug(f"Removed cached instance for unregistered tool: {B} from manager '{C.name}'")
		A.info(f"Tool '{B}' unregistered successfully from manager '{C.name}'.")
	def get_names(A):
		with A._lock:return set(A._tools.keys())
	def load_tools_from_directory(Q,directory,*,auto_register=R,recursive=a):
		U=recursive;D=V(directory).resolve()
		if not D.exists():A.error('Path %s does not exist – load aborted.',D);return 0
		A.info("ToolManager '%s' loading tools from: %s",Q.name,D);H=I(D.parent);X=a
		if H not in C.path:C.path.insert(0,H);X=R
		K=0;Y=[I(D)]
		try:
			Z=T.walk_packages(Y)if U else T.iter_modules(Y)
			for(b,B,c)in Z:
				if c and not U:continue
				d=V(b.path)/f"{B}.py";G=S.util.spec_from_file_location(B,d)
				if not G or not G.loader:A.error("No import spec for module '%s'",B);continue
				P=S.util.module_from_spec(G);C.modules[B]=P
				try:G.loader.exec_module(P)
				except J as e:A.error("Failed to import module '%s': %s",B,e,exc_info=R);C.modules.pop(B,F);continue
				K+=1;A.debug('Imported tool module: %s',B)
				if auto_register:
					f=W(O)
					for E in P.__dict__.values():
						if L.isclass(E)and N(E,M)and E is not M:
							try:f.register(E)
							except J as g:A.warning('Auto-register skipped for %s: %s',E.__name__,g)
		finally:
			if X:C.path.remove(H)
		A.info("ToolManager '%s' finished loading. %d module(s) imported.",Q.name,K);return K
D={}
f=U.RLock()
def i(manager=F):
	def A(cls):
		A=manager
		if A is F:A=W(O)
		A.register(cls);return cls
	return A
def W(name=O):
	B=name;global D
	with f:
		if B not in D:A.info(f"Creating new ToolManager instance named: {B}");D[B]=e(name=B)
		else:A.debug(f"Returning existing ToolManager instance named: {B}")
		return D[B]
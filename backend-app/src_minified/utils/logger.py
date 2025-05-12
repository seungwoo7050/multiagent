b='threadName'
a='thread'
Z='processName'
Y='process'
X='module'
W='lineno'
S=format
R='extra_attrs'
Q='trace_id'
P='otel_span_id'
O='name'
N=Exception
M=isinstance
L='agent_id'
K='task_id'
J='otel_trace_id'
F='message'
H=False
G=str
E=hasattr
D=None
C=getattr
import json as T,logging as A,sys
from datetime import datetime as U,timezone as V
from typing import Optional,Set,Dict,Any
from opentelemetry import trace
from src.schemas.config import AppSettings
f={'args','asctime','created','exc_info','exc_text','filename','funcName','levelname','levelno',W,X,'msecs',F,'msg',O,'pathname',Y,Z,'relativeCreated','stack_info',a,b,J,P}
class c(A.Formatter):
	def __init__(A,include_thread_info=H,include_process_info=H):super().__init__();A.include_thread_info=include_thread_info;A.include_process_info=include_process_info
	def format(I,record):
		e='execution_time';d='level';c='timestamp';A=record;B={c:U.fromtimestamp(A.created,V.utc).isoformat(),d:A.levelname,O:A.name,F:A.getMessage(),X:A.module,'function':A.funcName,W:A.lineno}
		if E(A,J)and A.otel_trace_id:B[J]=A.otel_trace_id
		if E(A,P)and A.otel_span_id:B[P]=A.otel_span_id
		if I.include_thread_info:B[a]=A.thread;B[b]=A.threadName
		if I.include_process_info:B[Y]=A.process;B[Z]=A.processName
		if A.exc_info:B['exception']={'type':A.exc_info[0].__name__,F:G(A.exc_info[1]),'traceback':I.formatException(A.exc_info)}
		if E(A,Q)and J not in B:B[Q]=A.trace_id
		if E(A,K):B[K]=A.task_id
		if E(A,L):B[L]=A.agent_id
		if E(A,e):B[e]=A.execution_time
		S=C(A,R,D)
		if S and M(S,dict):B.update(S)
		else:
			for(H,g)in A.__dict__.items():
				if H not in f and not H.startswith('_')and H not in B:B[H]=g
		try:return T.dumps(B,default=G,separators=(',',':'))
		except N as h:return T.dumps({c:U.now(V.utc).isoformat(),d:'ERROR',O:'logger.JsonFormatter',F:f"Error serializing log record: {G(h)}",'original_message':G(A.getMessage())},default=G)
class d(A.LoggerAdapter):
	def process(E,msg,kwargs):
		C='extra';A=kwargs
		if C not in A:A[C]={}
		B=A[C].get(R,{})
		if not M(B,dict):B={}
		if E.extra:
			for(D,F)in E.extra.items():
				if D in(K,L):A[C][D]=F
				elif D!=Q:B[D]=F
		if B:A[C][R]=B
		return msg,A
I=H
def B(settings_obj=D):
	W='LOG_FILE_PATH';V='LOG_TO_FILE';U='DEBUG';T='json';R='INFO';B=settings_obj;global I
	if I:A.getLogger(__name__).debug('Logging already configured.');return
	if B is D:from src.config.settings import get_settings as X;B=X()
	Y=A.getLogRecordFactory()
	def Z(*D,**F):
		A=Y(*D,**F);A.otel_trace_id='';A.otel_span_id=''
		try:
			C=trace.get_current_span()
			if C and E(C,'get_span_context'):
				B=C.get_span_context()
				if B and B.is_valid:A.otel_trace_id=S(B.trace_id,'032x');A.otel_span_id=S(B.span_id,'016x');import sys;sys.stderr.write(f"[OTEL-DEBUG] Added trace_id={A.otel_trace_id} to LogRecord '{A.name}'\n");sys.stderr.flush()
		except N as H:import sys;sys.stderr.write(f"[OTEL-ERROR] Failed to add trace ID: {G(H)}\n");sys.stderr.flush()
		return A
	A.setLogRecordFactory(Z);F=A.getLogger();J=C(B,'LOG_LEVEL',R);a=C(A,J.upper(),A.INFO);F.setLevel(a)
	for b in list(F.handlers):F.removeHandler(b)
	L=A.StreamHandler(sys.stdout);O=C(B,'LOG_FORMAT',T)
	if O==T:d=J.upper()==U;f=J.upper()in(U,R);K=c(include_thread_info=f,include_process_info=d)
	else:g='%(asctime)s - %(name)s - %(levelname)s - [%(otel_trace_id)s:%(otel_span_id)s] - %(message)s';K=A.Formatter(g,datefmt='%Y-%m-%d %H:%M:%S')
	L.setFormatter(K);F.addHandler(L)
	if C(B,V,H)and C(B,W,D):
		try:P=A.FileHandler(B.LOG_FILE_PATH);P.setFormatter(K);F.addHandler(P)
		except N as h:F.error(f"Failed to set up file logging to {B.LOG_FILE_PATH}: {h}")
	e();Q=A.getLogger(__name__);Q.info(f"Logging setup complete. Level: {J}, Format: {O}")
	if C(B,V,H)and C(B,W,D)and any(M(B,A.FileHandler)for B in F.handlers):Q.info(f"Log file configured at {B.LOG_FILE_PATH}")
	I=True
def e():
	B={'uvicorn':A.WARNING,'uvicorn.access':A.WARNING,'uvicorn.error':A.ERROR,'fastapi':A.WARNING,'aiohttp':A.WARNING,'redis':A.WARNING,'httpx':A.WARNING,'openai':A.WARNING,'anthropic':A.WARNING,'langchain':A.INFO,'langgraph':A.INFO,'opentelemetry':A.INFO}
	for(C,D)in B.items():A.getLogger(C).setLevel(D)
def g(name):
	if not I:B()
	return A.getLogger(name)
def h(name,trace_id=D,task_id=D,agent_id=D,**E):
	D=agent_id;C=task_id;B=trace_id;F=g(name);A={}
	if B:A['app_trace_id']=B
	if C:A[K]=C
	if D:A[L]=D
	A.update(E);return d(F,A)
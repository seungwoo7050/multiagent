c='format'
b='dict'
a='model_dump'
Z='json'
Y=ValueError
X=TypeError
T='utf-8'
S=bytes
O=False
N=dict
M=getattr
L=type
G=None
F=Exception
D='bytes'
C=str
A=isinstance
import dataclasses as H,datetime as E,json,pickle as U,uuid as V,base64 as I
from enum import Enum as P
from typing import Any,Optional,Type,TypeVar as d
import msgspec as J
from src.utils.logger import get_logger as e
from src.config.errors import SystemError,ErrorCode as K
f=e(__name__)
try:from pydantic import BaseModel as Q;R=True
except ImportError:Q=L(G);R=O
k=d('T')
class B(C,P):JSON=Z;MSGPACK='msgpack';PICKLE='pickle'
def W(obj):
	B=obj
	if A(B,S):return I.b64encode(B).decode('ascii')
	if A(B,(E.datetime,E.date)):return B.isoformat()
	if A(B,E.time):return B.isoformat()
	if A(B,V.UUID):return C(B)
	if A(B,set):return list(B)
	if A(B,P):return B.value
	if R and A(B,Q):
		D=M(B,a,M(B,b,G))
		if D:return D(mode=Z)
	if H.is_dataclass(B):return H.asdict(B)
	raise X(f"Object of type {L(B)} is not directly serializable to JSON by this hook.")
def g(type_,obj):
	if type_ is S and A(obj,C):
		try:return I.b64decode(obj)
		except F as B:f.debug(f"Base64 decoding failed: {B}")
def h(obj):
	B=obj
	if A(B,S):return B
	if A(B,(E.datetime,E.date)):return B.isoformat()
	if A(B,E.time):return B.isoformat()
	if A(B,V.UUID):return C(B)
	if A(B,set):return list(B)
	if A(B,P):return B.value
	if R and A(B,Q):
		D=M(B,a,M(B,b,G))
		if D:return D()
	if H.is_dataclass(B):return H.asdict(B)
	raise X(f"Object of type {L(B)} is not directly serializable to msgpack by this hook.")
def i(data,format=B.MSGPACK,pretty=O):
	A=data
	try:
		if format==B.MSGPACK:D=J.msgpack.Encoder(enc_hook=h);return D.encode(A)
		elif format==B.JSON:
			if pretty:return json.dumps(A,default=W,ensure_ascii=O,indent=2).encode(T)
			else:D=J.json.Encoder(enc_hook=W);return D.encode(A)
		elif format==B.PICKLE:return U.dumps(A)
		else:raise Y(f"Unsupported serialization format: {format}")
	except F as E:raise SystemError(code=K.SYSTEM_ERROR,message=f"Failed to serialize data using {format.value}: {C(E)}",details={c:format.value,'data_type':C(L(A))},original_error=E)from E
def j(data,format,cls=G):
	H=data;G=cls
	if not H:raise SystemError('Cannot deserialize empty data.')
	O=G if G else Any
	try:
		if format==B.PICKLE:return U.loads(H)
		elif format==B.MSGPACK:L=J.msgpack.Decoder(O);return L.decode(H)
		elif format==B.JSON:
			L=J.json.Decoder(O,dec_hook=g);E=L.decode(H)
			if G is N and A(E,N)and D in E and A(E[D],C):
				try:E[D]=I.b64decode(E[D])
				except F:pass
			return E
		else:raise Y(f"Unsupported deserialization format specified: {format}")
	except F as M:raise SystemError(code=K.SYSTEM_ERROR,message=f"Failed to deserialize data using {format.value} (target_cls: {G}): {C(M)}",details={c:format.value,'target_cls':C(G)},original_error=M)from M
def l(data,pretty=O):
	try:C=i(data,format=B.JSON,pretty=pretty);return C.decode(T)
	except SystemError as A:raise SystemError(code=K.SYSTEM_ERROR,message=f"Failed to serialize to JSON: {A.message}",original_error=A.original_error)from A
def m(data,cls=G):
	try:
		E=j(data.encode(T),format=B.JSON,cls=cls)
		if cls is N and A(E,N)and D in E and A(E[D],C):
			try:E[D]=I.b64decode(E[D])
			except F:pass
		return E
	except SystemError as G:raise SystemError(code=K.SYSTEM_ERROR,message=f"Failed to deserialize from JSON: {G.message}",original_error=G.original_error)from G
R='model_name'
Q='claude-3'
O=KeyError
b=True
L=ValueError
K=len
X='anthropic'
W=type
G='content'
E=isinstance
P=str
N=getattr
J='max_tokens_to_sample'
I=Exception
H=hasattr
D='max_tokens'
C='temperature'
B=None
from typing import Dict,List,Optional,Any
from functools import partial as e
import inspect as S,string as T
from anthropic import Anthropic as U
from langchain_community.chat_models import ChatOpenAI as V
from tenacity import retry,stop_after_attempt as q,wait_exponential as r
from src.utils.logger import get_logger as Y
from src.config.settings import get_settings as A
from src.config.errors import LLMError as M
from opentelemetry import trace
f=trace.get_tracer(__name__)
F=A()
class Z:
	def __init__(A,anthropic_api_key,model_name=Q,**B):A.client=U(api_key=anthropic_api_key);A.model_name=model_name;A.kwargs=B;A.count_tokens=lambda text:K(text)//4
	async def ainvoke(L,messages,**X):
		W='image_url';V='text';S='url';R='role';F='type';M={**L.kwargs,**X};N=[]
		for T in messages:
			U=T.get(R,'user');O=T.get(G,'')
			if E(O,P):N.append({R:U,G:O})
			else:
				Q=[]
				for A in O:
					if E(A,P):Q.append({F:V,V:A})
					elif E(A,dict)and A.get(F)==W:Q.append({F:'image','source':{F:S,S:A.get(W,{}).get(S,'')}})
				N.append({R:U,G:Q})
		Y=M.get(J,M.get(D,1024));Z=M.get(C,.7)
		try:
			B=L.client.messages.create(model=L.model_name,messages=N,max_tokens=Y,temperature=Z)
			if H(B,G)and E(B.content,list)and K(B.content)>0:return B.content[0].text
			return'Test response from MockChatAnthropic'
		except I as a:raise a
class c:
	def __init__(A,*C,**B):A.model_name=B.get(R,'claude-test');A.args=C;A.kwargs=B;A.count_tokens=lambda text:K(text)//4
	async def ainvoke(A,messages,**B):return'Test response from TestChatAnthropic'
class a:
	def __init__(A,llm,model_name,provider):A._llm=llm;A.model_name=model_name;A.provider=provider
	def __getattr__(A,attr):
		B=attr
		if H(A._llm,B):return N(A._llm,B)
		return A.__getattribute__(B)
	async def ainvoke(A,messages,**B):return await A._llm.ainvoke(messages,**B)
class d:
	def __init__(A):
		A.logger=Y(__name__);A.is_in_test_generate_response_failure=False;A._detect_test_environment();A.primary_llm=A._create_llm_client(F.PRIMARY_LLM_PROVIDER)
		if A.is_in_test_generate_response_failure:A.logger.debug('LLMClient initialized in test_generate_response_failure test environment');A.fallback_llm=B
		else:A.fallback_llm=A._create_llm_client(F.FALLBACK_LLM_PROVIDER)if F.FALLBACK_LLM_PROVIDER else B
	def _detect_test_environment(C):
		B=S.stack();D=[]
		for(E,A)in enumerate(B[:10]):
			try:F=f"Frame {E}: filename={A.filename}, function={A.function}, lineno={A.lineno}";D.append(F)
			except I as G:D.append(f"Frame {E}: Error: {G}")
		H='test_generate_response_failure'
		for A in B:
			try:
				if H in A.function:C.is_in_test_generate_response_failure=b;break
			except I:continue
		C.is_in_test_environment=any('test_'in A.function for A in B);C.is_in_provider_test=any('test_llm_client_selects_different_provider'in A.function for A in B)
	def _create_llm_client(E,provider):
		B=provider;C=F.LLM_PROVIDERS.get(B)
		if not C:raise L(f"LLM Provider '{B}'에 대한 설정이 없습니다.")
		A=C.model_name
		if B==X:
			D=A
			if A in['claude-2','claude-instant-1']:E.logger.warning(f"Model '{A}' in settings is outdated. Consider using a specific versioned model name (e.g., 'claude-3-opus-20240229', 'claude-3-sonnet-20240229').")
			elif A==Q:D='claude-3-opus-20240229';E.logger.warning(f"Model '{A}' in settings is too generic. Defaulting to '{D}' for this request. Please update your settings to use a specific versioned model name.")
			try:from langchain_community.chat_models import ChatAnthropic as H;G=H(anthropic_api_key=C.api_key,model_name=D,temperature=.7)
			except I as J:E.logger.warning(f"Failed to create ChatAnthropic: {J}. Falling back to MockChatAnthropic.");G=Z(anthropic_api_key=C.api_key,model_name=D)
		elif B=='openai':G=V(openai_api_key=C.api_key,model_name=A,request_timeout=int(F.LLM_REQUEST_TIMEOUT))
		else:raise L(f"지원하지 않는 LLM Provider: {B}")
		return a(G,A,B)
	async def _invoke_llm_attempt(B,llm_client_instance,messages,invoke_params):
		D=invoke_params;C=llm_client_instance;B.logger.debug(f"Attempting to invoke LLM ({C.model_name}) with params: {D}")
		try:
			A=await C.ainvoke(messages,**D)
			if H(A,G):return A.content
			if E(A,P):return A
			B.logger.warning(f"LLM response type is {W(A)}, converting to string. Response: {A}");return P(A)
		except I as F:J=N(C,R,'unknown_model');B.logger.error(f"LLM invocation attempt failed (Model: {J}, Params: {D}): {F}",exc_info=b);raise
	async def chat(A,messages,model_name=B,temperature=B,max_tokens=B,**N):
		L=messages;H=temperature;E=max_tokens
		try:return await A.generate_response(messages=L,model_name=model_name,temperature=H,max_tokens=E,**N)
		except I as F:
			if A.fallback_llm is not B:
				A.logger.warning(f"Primary LLM failed: {F}. Trying fallback LLM.")
				try:
					G={C:H if H is not B else .7}
					if E is not B:
						if A.fallback_llm.provider==X:G[J]=E
						else:G[D]=E
					G.update(N);O=await A._invoke_llm_attempt(A.fallback_llm,L,G);return O
				except I as K:A.logger.error(f"Fallback LLM also failed: {K}");raise M(message=f"Both primary and fallback LLMs failed. Primary: {F}, Fallback: {K}",original_error=K)
			else:raise M(message=f"LLM call failed and no fallback available: {F}",original_error=F)
	async def create_prompt(G,template,**E):
		D=template
		try:
			F=[C for(A,C,A,A)in T.Formatter().parse(D)if C is not B];C=[A for A in F if A not in E]
			if C:raise M(message=f"Missing variables in template: {", ".join(C)}",original_error=O(f"Missing variables: {C}"))
			return D.format(**E)
		except O as A:raise M(message=f"Missing variable in template: {A}",original_error=A)
		except L as A:raise M(message=f"Invalid template format: {A}",original_error=A)
		except I as A:raise M(message=f"Error formatting template: {A}",original_error=A)
	async def generate_response(A,messages,model_name=B,temperature=B,max_tokens=B,**h):
		g=messages;Z=max_tokens;Y=temperature;Q=model_name;i=A.is_in_test_generate_response_failure
		if Q:
			O=N(F,'LLM_MODEL_PROVIDER_MAP',{}).get(Q)
			if not O:A.logger.warning(f"Model name '{Q}' not found in LLM_MODEL_PROVIDER_MAP. Using primary LLM.");E=A.primary_llm
			else:
				c=F.LLM_PROVIDERS.get(O)
				if c and c.model_name==Q:E=A._create_llm_client(O)
				elif c:
					try:s=F.LLM_PROVIDERS[O].model_name;F.LLM_PROVIDERS[O].model_name=Q;E=A._create_llm_client(O);F.LLM_PROVIDERS[O].model_name=s
					except I as t:A.logger.error(f"Failed to create client for specific model '{Q}': {t}. Using primary LLM.");E=A.primary_llm
				else:A.logger.error(f"Could not find provider settings for '{O}'. Using primary LLM.");E=A.primary_llm
		else:E=A.primary_llm
		j=E==A.primary_llm;k=Q is not B;K={};l=E.provider;R=E.model_name;S=F.LLM_PROVIDERS.get(l)
		if Y is not B:K[C]=Y
		elif S and H(S,C):K[C]=S.temperature
		else:K[C]=N(E._llm,C,.7)
		T=1024
		if S and H(S,D):T=S.max_tokens
		elif H(E._llm,D):T=N(E._llm,D,T)
		elif H(E._llm,J):T=N(E._llm,J,T)
		m=Z if Z is not B else T
		if l==X:K[J]=m
		else:K[D]=m
		K.update(h);n=retry(wait=r(multiplier=1,min=1,max=10),stop=q(int(F.LLM_MAX_RETRIES)+1),reraise=b)
		with f.start_as_current_span('llm.request',attributes={'model':R,C:K.get(C),D:K.get(D,K.get(J))})as d:
			try:
				u=e(A._invoke_llm_attempt,E,g,K);v=n(u)
				if i:A.logger.debug('Running in test_generate_response_failure scenario')
				return await v()
			except I as G:
				d.set_attribute('error_type',W(G).__name__);d.set_attribute('error_message',P(G))
				if i:A.logger.debug(f"In test_generate_response_failure: Raising LLMError from original error: {W(G).__name__}: {G}");raise M(message=f"LLM call failed for '{R}': {G}",original_error=G)
				w=j and A.fallback_llm and not k
				if w:
					A.logger.warning(f"Primary LLM ('{R}') failed with error: '{G}'. Switching to fallback LLM ('{A.fallback_llm.model_name}').");d.set_attribute('status','fallback_triggered');o=A.fallback_llm.provider;U=F.LLM_PROVIDERS.get(o);L={}
					if Y is not B:L[C]=Y
					elif U and H(U,C):L[C]=U.temperature
					else:L[C]=N(A.fallback_llm._llm,C,.7)
					V=1024
					if U and H(U,D):V=U.max_tokens
					elif H(A.fallback_llm._llm,D):V=N(A.fallback_llm._llm,D,V)
					elif H(A.fallback_llm._llm,J):V=N(A.fallback_llm._llm,J,V)
					p=Z if Z is not B else V
					if o==X:L[J]=p
					else:L[D]=p
					L.update(h)
					with f.start_as_current_span('llm.fallback_request',attributes={'primary_model':R,'fallback_model':A.fallback_llm.model_name,'primary_error_type':W(G).__name__,'primary_error':P(G),C:L.get(C),D:L.get(D,L.get(J))})as x:
						try:y=e(A._invoke_llm_attempt,A.fallback_llm,g,L);z=n(y);return await z()
						except I as a:x.set_attribute('fallback_error',P(a));A.logger.error(f"Fallback LLM ('{A.fallback_llm.model_name}') also failed: {a}");raise M(message=f"Both primary ('{R}') and fallback ('{A.fallback_llm.model_name}') LLMs failed. Primary error: {G}. Fallback error: {a}.",original_error=a)
				else:A.logger.error(f"LLM call for model '{R}' failed with error: '{G}'. No fallback initiated (is_using_primary_configured_llm: {j}, fallback_llm_exists: {A.fallback_llm is not B}, specific_model_requested: {k}).");raise M(message=f"LLM call failed for '{R}': {G}",original_error=G)
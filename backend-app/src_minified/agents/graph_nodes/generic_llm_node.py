AS='subtask_description'
AR='subtask_answer'
Z='metadata'
AQ='description'
Y='available_tools'
AA='input'
A9='action_input'
A8='action'
U='\n'
A0='content'
z='role'
y='tool_call_history'
Q=hasattr
s=type
q='scratchpad'
p=len
K=getattr
o=list
h=False
g=Exception
d='result'
c='args'
X=''
V='.'
S='tool_name'
R='dynamic_data'
N=True
L=dict
J=str
H=isinstance
E=None
import os,json as b,re as F
from typing import Any,Dict,List,Optional,Union,TypedDict as A
from langchain_core.prompts import PromptTemplate as I
from src.config.settings import get_settings as B
from src.utils.logger import get_logger as D
from src.config.errors import ToolError as AP
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager
from src.memory.memory_manager import MemoryManager
from src.schemas.mcp_models import AgentGraphState as Ac,ConversationTurn as T
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage as n,IntermediateResultMessage as A7
from opentelemetry import trace as f
Ad=f.get_tracer(__name__)
C=D(__name__)
G=B()
class M(A):action:J;action_input:Union[Dict[J,Any],J]
class O:
	def __init__(A,llm_client,tool_manager,memory_manager,notification_service,prompt_template_path,output_field_name=E,input_keys_for_prompt=E,model_name=E,temperature=E,max_tokens=E,node_id='generic_llm_node',max_react_iterations=5,enable_tool_use=h,allowed_tools=E,history_prompt_key='conversation_history',summary_prompt_key='conversation_summary',history_key_prefix='chat_history',max_history_items=10):
		G=allowed_tools;F=prompt_template_path;D=notification_service;A.llm_client=llm_client;A.tool_manager=tool_manager;A.memory_manager=memory_manager;A.notification_service=D;A.prompt_template_path=F;A.output_field_name=output_field_name;A.input_keys_for_prompt=input_keys_for_prompt or[];A.model_name=model_name;A.temperature=temperature;A.max_tokens=max_tokens;A.node_id=node_id;A.max_react_iterations=max_react_iterations;A.enable_tool_use=enable_tool_use;A.allowed_tools=set(G)if G is not E else E;A.history_prompt_key=history_prompt_key;A.summary_prompt_key=summary_prompt_key;A.history_key_prefix=history_key_prefix;A.max_history_items=max_history_items;A.prompt_template_str=A._load_prompt_template();B=set(A.input_keys_for_prompt)
		if A.enable_tool_use:B.update([Y,y,q])
		if A.history_prompt_key:B.add(A.history_prompt_key)
		if A.summary_prompt_key:B.add(A.summary_prompt_key)
		A.prompt_template=I(template=A.prompt_template_str,input_variables=o(B));C.info(f"GenericLLMNode '{A.node_id}' initialized. Tool use: {A.enable_tool_use}. History key: '{A.history_prompt_key}', Summary key: '{A.summary_prompt_key}', Max hist items: {A.max_history_items}. Prompt: {F}. NotificationService injected: {"Yes"if D else"No"}")
	def _load_prompt_template(B):
		D=K(G,'PROMPT_TEMPLATE_DIR','config/prompts')
		if os.path.isabs(B.prompt_template_path):A=B.prompt_template_path
		else:A=os.path.join(D,B.prompt_template_path)
		try:
			with open(A,'r',encoding='utf-8')as E:return E.read()
		except FileNotFoundError:C.error(f"Prompt template file not found: {A}");raise
		except g as F:C.error(f"Error loading prompt template from {A}: {F}");raise
	def _get_available_tools_for_prompt(A):
		I='name'
		if not A.enable_tool_use:return'Tool use is disabled for this node.'
		C=A.tool_manager.get_names();B=[]
		if A.allowed_tools is not E:
			D=C.intersection(A.allowed_tools)
			if not D:return'No tools available (allowed list is empty or no matches found).'
			B=o(D)
		else:B=o(C)
		if not B:return'No tools available.'
		J=A.tool_manager.get_tool_summaries_for_llm();F=[A for A in J if A.get(I)in B]
		if not F:return'No details found for available tools.'
		G=['Available Tools:']
		for H in F:G.append(f"- {H.get(I,"Unknown Tool")}: {H.get(AQ,"No description.")}")
		return U.join(G)
	async def _load_conversation_history(A,state):
		R='Unknown';P='conversation_id';D=state
		if not A.history_prompt_key or not A.history_key_prefix:return'No conversation history configured for retrieval.'
		try:
			F=D.task_id
			if Q(D,Z)and H(D.metadata,L)and D.metadata.get(P):F=D.metadata[P]
			if not F:C.warning(f"Node '{A.node_id}': context_id (task_id or conversation_id) not found in state for history retrieval.");return'History retrieval skipped: context_id is missing.'
			M=await A.memory_manager.get_history(context_id=F,history_key_prefix=A.history_key_prefix,limit=A.max_history_items)
			if not M:return'No conversation history found.'
			E=[]
			for B in M:
				if H(B,T):G=K(B,z,R).capitalize();I=K(B,A0,X);E.append(f"{G}: {I}")
				elif H(B,L):G=B.get(z,R).capitalize();I=B.get(A0,X);E.append(f"{G}: {I}")
				else:C.warning(f"Node '{A.node_id}': Encountered unknown history item type: {s(B)}. Converting to string.");E.append(J(B))
			return U.join(E)
		except g as O:C.error(f"Node '{A.node_id}': Failed to load conversation history: {O}",exc_info=N);return f"Error loading conversation history: {O}"
	async def _prepare_prompt_input(B,state):
		D=state;M={};i=B.prompt_template.input_variables;a={AR:['dynamic_data.current_subtask.final_answer'],AS:['dynamic_data.current_subtask.description'],'score_threshold':[]}
		for A in i:
			F=E
			if A in B.__dict__:F=B.__dict__[A];C.debug(f"Node '{B.node_id}': Using parameter value for '{A}'")
			elif Q(D,A):F=K(D,A)
			elif A in a:
				for W in a[A]:
					if F is not E:break
					if V in W:
						I=W.split(V)
						if I[0]==R and H(D.dynamic_data,L):
							O=D.dynamic_data;e=N
							for f in I[1:]:
								if H(O,L)and f in O:O=O[f]
								else:e=h;break
							if e:F=O;C.debug(f"Node '{B.node_id}': Found '{A}' via special mapping path '{W}'")
			elif V in A:
				I=A.split(V)
				if I[0]==R and H(D.dynamic_data,L):G=D.dynamic_data;T=I[1:]
				elif I[0]==Z and H(D.metadata,L):G=D.metadata;T=I[1:]
				elif Q(D,I[0]):G=K(D,I[0]);T=I[1:]
				else:G=E;T=[]
				for P in T:
					if H(G,L)and P in G:G=G[P]
					elif Q(G,P):G=K(G,P)
					else:C.debug(f"Node '{B.node_id}': Path '{A}' could not be fully resolved. Part '{P}' not found in {s(G).__name__}.");G=E;break
				F=G
			elif H(D.dynamic_data,L)and A in D.dynamic_data:F=D.dynamic_data[A]
			elif H(D.metadata,L)and A in D.metadata:F=D.metadata[A]
			if A==B.history_prompt_key and B.history_key_prefix:F=await B._load_conversation_history(D)
			elif A==B.summary_prompt_key and B.summary_prompt_key:
				if D.dynamic_data and H(D.dynamic_data.get(B.summary_prompt_key),J):F=D.dynamic_data[B.summary_prompt_key];C.debug(f"Node '{B.node_id}': Using '{B.summary_prompt_key}' from dynamic_data for prompt.")
				else:F='No conversation summary available for this turn.';C.debug(f"Node '{B.node_id}': '{B.summary_prompt_key}' not found in dynamic_data or not a string. Using default.")
			elif B.enable_tool_use:
				if A==Y:F=B._get_available_tools_for_prompt()
				elif A==y:g=D.tool_call_history if D.tool_call_history is not E else[];F=U.join(f"Tool: {A.get(S,"UnknownTool")}, Args: {b.dumps(A.get(c,{}))}, Result: {J(A.get(d,"No result"))[:200]}"for A in g if H(A,L))if g else'No tool calls yet.'
				elif A==q:F=D.scratchpad if D.scratchpad is not E else X
			if F is E:
				if A in B.input_keys_for_prompt or A==B.history_prompt_key and B.history_prompt_key or A==B.summary_prompt_key and B.summary_prompt_key:C.warning(f"Node '{B.node_id}': Key '{A}' required for prompt (from input_keys, history, or summary) was not found or resolved to None; using empty string.")
				M[A]=X
			elif H(F,(o,L))and A!='messages':
				try:M[A]=b.dumps(F,indent=2,ensure_ascii=h,default=J)
				except TypeError:C.warning(f"Node '{B.node_id}': Could not JSON serialize value for key '{A}' (type: {s(F).__name__}). Using str().");M[A]=J(F)
			else:M[A]=J(F)
		C.debug(f"Node '{B.node_id}': Prepared prompt input keys: {o(M.keys())}");return M
	def _parse_llm_response(B,response_str):
		K='```';A=response_str;A=A.strip();C.debug(f"Node '{B.node_id}': Parsing LLM response: {A[:300]}...")
		try:
			if A.startswith('```json')and A.endswith(K):A=A[7:-3].strip()
			elif A.startswith(K)and A.endswith(K):A=A[3:-3].strip()
			D=b.loads(A)
			if H(D,L)and A8 in D and A9 in D:G=D[A8];P=D[A9];C.info(f"Node '{B.node_id}': Successfully parsed JSON response. Action: {G}");return M(action=J(G),action_input=P)
			else:C.warning(f"Node '{B.node_id}': Parsed JSON lacks 'action' or 'action_input' keys. Content: {J(D)[:200]}");return
		except b.JSONDecodeError:
			C.debug(f"Node '{B.node_id}': Response is not valid JSON. Attempting ReAct style text parsing.");N=F.search('Action:\\s*([^\\n]+)',A,F.IGNORECASE);O=F.search('Action Input:\\s*([\\s\\S]+)',A,F.IGNORECASE|F.DOTALL)
			if N:
				G=N.group(1).strip();I=O.group(1).strip()if O else'{}';E:0
				try:
					E=b.loads(I)
					if not H(E,L):E={AA:I}
				except b.JSONDecodeError:E=I
				C.info(f"Node '{B.node_id}': Successfully parsed ReAct style text response (Action: {G}). Input type: {s(E).__name__}");return M(action=G,action_input=E)
			C.error(f"Node '{B.node_id}': Failed to parse LLM response using known formats. Response preview: {A[:500]}...");return
	async def __call__(A,state):
		Ab='Max iterations reached';Aa='tool.result_length';AZ='tool_calling';AY='dynamic_data.';AX='current_subtask';AW='max_tokens';AV='temperature';AO='tool_args';AN='node_completed';AM='last_llm_output';AL='last_llm_input';A6='tool.name';A5='error_message';m='error';e='final_answer';B=state
		with Ad.start_as_current_span('graph.node.generic_llm',attributes={'node_id':A.node_id,'task_id':B.task_id,'model':A.model_name or'default_from_llm_client','enable_tool_use':A.enable_tool_use})as F:
			C.info(f"GenericLLMNode '{A.node_id}' execution started. Task ID: {B.task_id}. Tool use enabled: {A.enable_tool_use}");F.set_attribute('app.node.id',A.node_id);await A.notification_service.broadcast_to_task(B.task_id,n(task_id=B.task_id,status='node_executing',detail=f"Node '{A.node_id}' started.",current_node=A.node_id))
			if not A.enable_tool_use:
				C.debug(f"Node '{A.node_id}' (Task: {B.task_id}): Executing simple LLM call (tool use disabled).");P={};A1=E
				try:
					AB=await A._prepare_prompt_input(B);Y=A.prompt_template.format(**AB);F.set_attribute('app.llm.prompt_length',p(Y));C.debug(f"Node '{A.node_id}' (Task: {B.task_id}): Formatted prompt (tools disabled):\n{Y[:500]}...");i={}
					if A.temperature is not E:i[AV]=A.temperature
					if A.max_tokens is not E:i[AW]=A.max_tokens
					AC=[{z:'user',A0:Y}];Q=await A.llm_client.generate_response(messages=AC,model_name=A.model_name,**i);F.set_attribute('app.llm.response_length',p(Q));C.debug(f"Node '{A.node_id}' (Task: {B.task_id}): LLM raw response (tools disabled): {Q[:200]}...");P={A5:E,AL:Y,AM:Q,R:B.dynamic_data.copy()if B.dynamic_data else{}}
					if B.dynamic_data and AX in B.dynamic_data and A.node_id=='initial_responder_subtask':P[AS]=B.dynamic_data[AX].get(AQ,X);P[AR]=Q
					Z=A.output_field_name or e
					if V in Z:
						a=Z.split(V);Ae=a[0]
						if Ae==R:
							r=P[R]
							for(Ag,A2)in enumerate(a[1:-1]):
								if A2 not in r or not H(r[A2],L):r[A2]={}
								r=r[A2]
							r[a[-1]]=Q
						else:C.warning(f"Node '{A.node_id}': Cannot set output to non-dynamic_data nested field '{Z}'. Storing in 'final_answer'.");P[e]=Q
					else:P[Z]=Q
					P[R][AL]=Y;P[R][AM]=Q
				except g as AD:C.error(f"Node '{A.node_id}' (Task: {B.task_id}): Error during simple LLM call: {AD}",exc_info=N);A1=f"Error in node '{A.node_id}' (tools disabled): {AD}";F.set_status(f.Status(f.StatusCode.ERROR,description=A1));F.record_exception(AD);P={A5:A1,R:B.dynamic_data.copy()if B.dynamic_data else{}}
				await A.notification_service.broadcast_to_task(B.task_id,n(task_id=B.task_id,status=AN,detail=f"Node '{A.node_id}' (Simple LLM Call) finished. Error: {A1 or"None"}",current_node=A.node_id,next_node=E));return P
			C.debug(f"Node '{A.node_id}' (Task: {B.task_id}): Executing ReAct loop (tool use enabled).");F.set_attribute('app.react.max_iterations',A.max_react_iterations);T=E;U=B.dynamic_data.copy()if B.dynamic_data else{};U.setdefault(q,B.scratchpad or X);U.setdefault(y,o(B.tool_call_history or[]))
			for M in range(A.max_react_iterations):
				F.add_event(f"ReAct Iteration Start: {M+1}");C.info(f"Node '{A.node_id}' (Task: {B.task_id}): ReAct Iteration {M+1}/{A.max_react_iterations}");await A.notification_service.broadcast_to_task(B.task_id,n(task_id=B.task_id,status='node_iterating',detail=f"Node '{A.node_id}' ReAct iteration {M+1}.",current_node=A.node_id));Af=Ac(task_id=B.task_id,original_input=B.original_input,current_iteration=M,thoughts=B.thoughts,current_thoughts_to_evaluate=B.current_thoughts_to_evaluate,current_best_thought_id=B.current_best_thought_id,search_depth=B.search_depth,max_search_depth=B.max_search_depth,dynamic_data=U,metadata=B.metadata)
				try:AB=await A._prepare_prompt_input(Af);Y=A.prompt_template.format(**AB);F.set_attribute(f"app.react.iteration.{M+1}.prompt_length",p(Y));C.debug(f"Node '{A.node_id}' (Task: {B.task_id}) Iteration {M+1}: Formatted prompt ready.")
				except g as AE:C.error(f"Node '{A.node_id}' (Task: {B.task_id}): Error preparing prompt in iteration {M+1}: {AE}",exc_info=N);T=f"Error preparing prompt in node '{A.node_id}': {AE}";F.record_exception(AE);break
				try:
					i={}
					if A.temperature is not E:i[AV]=A.temperature
					i[AW]=A.max_tokens if A.max_tokens else 1000;AC=[{z:'user',A0:Y}];Q=await A.llm_client.generate_response(messages=AC,model_name=A.model_name,**i);F.set_attribute(f"app.react.iteration.{M+1}.response_length",p(Q));C.debug(f"Node '{A.node_id}' (Task: {B.task_id}) Iteration {M+1}: LLM raw response received.");U[AL]=Y;U[AM]=Q
				except g as AF:C.error(f"Node '{A.node_id}' (Task: {B.task_id}): LLM call failed in iteration {M+1}: {AF}",exc_info=N);T=f"LLM call failed in node '{A.node_id}': {AF}";F.record_exception(AF);break
				AG=A._parse_llm_response(Q)
				if AG is E:C.error(f"Node '{A.node_id}' (Task: {B.task_id}): Failed to parse LLM response in iteration {M+1}. Response: {Q[:500]}...");T=f"Failed to parse LLM response in node '{A.node_id}'.";F.set_attribute(f"app.react.iteration.{M+1}.parse_error",N);await A.notification_service.broadcast_to_task(B.task_id,n(task_id=B.task_id,status='node_error',detail=f"Node '{A.node_id}' failed to parse LLM response.",current_node=A.node_id));break
				G=AG[A8];K=AG[A9];F.set_attribute(f"app.react.iteration.{M+1}.action",G);C.info(f"Node '{A.node_id}' (Task: {B.task_id}) Iteration {M+1}: Parsed Action: {G}, Input type: {s(K).__name__}");D=X;W=E;A3=G.lower().replace(' ','_')
				if A3==e or A3=='finish':
					C.info(f"Node '{A.node_id}' (Task: {B.task_id}): ReAct loop: Received '{G}' action. Finishing.");F.add_event('ReAct Action: Finish');AT=K if H(K,J)else b.dumps(K,default=J);P={A5:E,R:U};Z=A.output_field_name or e
					if V in Z and Z.startswith(AY):
						a=Z.split(V);j=P[R]
						for AH in a[1:-1]:j=j.setdefault(AH,{})
						j[a[-1]]=AT
					else:P[Z]=AT
					await A.notification_service.broadcast_to_task(B.task_id,n(task_id=B.task_id,status=AN,detail=f"Node '{A.node_id}' ReAct loop finished successfully.",current_node=A.node_id));return P
				elif A3=='think':C.info(f"Node '{A.node_id}' (Task: {B.task_id}): ReAct loop: Received 'think' action.");F.add_event('ReAct Action: Think');AU=K if H(K,J)else b.dumps(K,default=J);D=f"Thought processed: {AU}";U[q]+=f"\nThought: {AU}"
				elif A3=='tool_call'and H(K,L):
					I=K.get(S);t=K.get(AO,{});F.add_event(f"ReAct Action: Tool Call ({I})")
					if not I or not H(I,J):D="Error: 'tool_call' action input missing 'tool_name' or it's not a string.";C.error(f"Node '{A.node_id}': {D} Input: {K}");W={S:'unknown_format',c:K,d:D,m:N}
					elif not A.tool_manager.has(I):D=f"Error: Tool '{I}' does not exist.";C.warning(f"Node '{A.node_id}': {D}");W={S:I,c:t,d:D,m:N}
					elif A.allowed_tools is not E and I not in A.allowed_tools:D=f"Error: Tool '{I}' is not allowed for this node.";C.warning(f"Node '{A.node_id}': {D}");W={S:I,c:t,d:D,m:N}
					else:
						C.info(f"Node '{A.node_id}' (Task: {B.task_id}): ReAct loop: Executing tool '{I}' via 'tool_call' action.");await A.notification_service.broadcast_to_task(B.task_id,A7(task_id=B.task_id,node_id=A.node_id,result_step_name=AZ,data={S:I,AO:t}));O=X;k=h
						try:AI=A.tool_manager.get_tool(I);AJ=await AI.ainvoke(t);O=J(AJ);D=f"Tool {I} execution successful.";C.debug(f"Node '{A.node_id}': Tool '{I}' Result: {O[:200]}...");F.add_event(f"Tool Executed: {I}",attributes={Aa:p(O)})
						except AP as l:D=f"Error executing tool '{I}': {l.message}";C.error(f"Node '{A.node_id}': ToolError during execution of '{I}': {l.message}",exc_info=h);k=N;O=D;F.record_exception(l,attributes={A6:I})
						except g as u:D=f"Unexpected error running tool '{I}': {J(u)}";C.exception(f"Node '{A.node_id}': Unexpected error during execution of tool '{I}'");k=N;O=D;F.record_exception(u,attributes={A6:I})
						W={S:I,c:t,d:O,m:k};D=f"Observation: {O}"
				elif A.tool_manager.has(G):
					F.add_event(f"ReAct Action: Tool Call (Direct - {G})")
					if A.allowed_tools is not E and G not in A.allowed_tools:D=f"Error: Tool '{G}' is not allowed for this node.";C.warning(f"Node '{A.node_id}': {D}");W={S:G,c:K,d:D,m:N}
					else:
						C.info(f"Node '{A.node_id}' (Task: {B.task_id}): ReAct loop: Executing tool '{G}' (direct action).");await A.notification_service.broadcast_to_task(B.task_id,A7(task_id=B.task_id,node_id=A.node_id,result_step_name=AZ,data={S:G,AO:K if H(K,L)else{AA:K}}));A4=K if H(K,L)else{}
						if H(K,J)and not A4:A4={AA:K}
						O=X;k=h
						try:AI=A.tool_manager.get_tool(G);AJ=await AI.ainvoke(A4);O=J(AJ);D=f"Tool {G} execution successful.";C.debug(f"Node '{A.node_id}': Tool '{G}' Result: {O[:200]}...");F.add_event(f"Tool Executed: {G}",attributes={Aa:p(O)})
						except AP as l:D=f"Error executing tool '{G}': {l.message}";C.error(f"Node '{A.node_id}': ToolError during execution of '{G}': {l.message}",exc_info=h);k=N;O=D;F.record_exception(l,attributes={A6:G})
						except g as u:D=f"Unexpected error running tool '{G}': {J(u)}";C.exception(f"Node '{A.node_id}': Unexpected error during execution of tool '{G}'");k=N;O=D;F.record_exception(u,attributes={A6:G})
						W={S:G,c:A4,d:O,m:k};D=f"Observation: {O}"
				else:D=f"Error: Unknown action '{G}'. LLM response was: {Q[:200]}";C.error(f"Node '{A.node_id}': {D}");F.add_event(f"ReAct Action: Unknown ({G})");W={S:G,c:K,d:D,m:N}
				U[q]+=f"\n{D}"
				if W:await A.notification_service.broadcast_to_task(B.task_id,A7(task_id=B.task_id,node_id=A.node_id,result_step_name='tool_result',data=W));U[y].append(W)
			F.set_attribute('app.react.iterations_completed',M+1);AK=f"Node '{A.node_id}' (Task: {B.task_id}): ReAct loop "
			if T:AK+=f"finished due to error after {M+1} iterations: {T}";F.set_status(f.Status(f.StatusCode.ERROR,description=T))
			else:AK+=f"reached max iterations ({A.max_react_iterations}).";F.set_status(f.Status(f.StatusCode.OK,description=Ab))
			C.warning(AK);v={A5:T if T else f"Reached max ReAct iterations ({A.max_react_iterations}).",R:U};w=U.get(q,'No answer generated after max iterations.')
			if T:w=f"Processing stopped due to error: {T}. Last scratchpad: {w}"
			x=A.output_field_name or e
			if V in x and x.startswith(AY):
				a=x.split(V);j=v[R]
				for AH in a[1:-1]:j=j.setdefault(AH,{})
				j[a[-1]]=w
			else:v[x]=w
			if e not in v and x!=e:v[e]=w
			await A.notification_service.broadcast_to_task(B.task_id,n(task_id=B.task_id,status=AN,detail=f"Node '{A.node_id}' ReAct loop finished. Error: {T or Ab}",current_node=A.node_id));return v
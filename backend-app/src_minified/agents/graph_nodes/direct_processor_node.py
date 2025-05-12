P='task'
O=Exception
M=getattr
L=len
K=dict
J=list
I=isinstance
G=str
F=None
import os
from typing import Any,Dict,List,Optional
from langchain_core.prompts import PromptTemplate as E
from src.utils.logger import get_logger as A
from src.config.settings import get_settings as B
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage as N,IntermediateResultMessage as Q
from opentelemetry import trace as D
import json
R=D.get_tracer(__name__)
C=A(__name__)
H=B()
class S:
	def __init__(A,llm_client,notification_service,prompt_template_path='generic/direct_processor.txt',temperature=.7,max_tokens=100,model_name=F,node_id='direct_processor_node',summary_prompt_key='conversation_summary',input_keys_for_prompt=F):
		B=prompt_template_path;A.llm_client=llm_client;A.notification_service=notification_service;A.temperature=temperature;A.max_tokens=max_tokens;A.model_name=model_name;A.node_id=node_id;A.prompt_template_path_relative=B;A.summary_prompt_key=summary_prompt_key;A.input_keys_for_prompt=input_keys_for_prompt or[P];A.prompt_template_str=A._load_prompt_template();D=set(A.input_keys_for_prompt)
		if A.summary_prompt_key:D.add(A.summary_prompt_key)
		A.prompt_template_engine=E(template=A.prompt_template_str,input_variables=J(D));C.info(f"DirectProcessorNode '{A.node_id}' initialized. Prompt: '{B}', Summary key: '{A.summary_prompt_key}', Max tokens: {A.max_tokens}")
	def _load_prompt_template(A):
		if not A.prompt_template_path_relative:C.error(f"Node '{A.node_id}': No prompt template path provided.");return'User request: {task}\n\nAssistant response:'
		D=G(M(H,'PROMPT_TEMPLATE_DIR','config/prompts'));B=os.path.join(D,A.prompt_template_path_relative)
		try:
			with open(B,'r',encoding='utf-8')as E:C.debug(f"Node '{A.node_id}': Successfully loaded prompt template from: {B}");return E.read()
		except FileNotFoundError:C.error(f"Node '{A.node_id}': Prompt template file not found at {B}.");raise
		except O as F:C.error(f"Node '{A.node_id}': Error loading prompt template from {B}: {F}",exc_info=True);raise
	async def _prepare_prompt_input(B,state):
		D=state;H={};L=B.prompt_template_engine.input_variables
		for A in L:
			E=F
			if A==P:E=D.original_input
			elif hasattr(D,A):E=M(D,A)
			elif I(D.dynamic_data,K)and A in D.dynamic_data:E=D.dynamic_data[A]
			elif I(D.metadata,K)and A in D.metadata:E=D.metadata[A]
			if A==B.summary_prompt_key and B.summary_prompt_key:
				if D.dynamic_data and I(D.dynamic_data.get(B.summary_prompt_key),G):E=D.dynamic_data[B.summary_prompt_key];C.debug(f"Node '{B.node_id}': Using '{B.summary_prompt_key}' from dynamic_data.")
				else:E='No conversation summary available.';C.debug(f"Node '{B.node_id}': '{B.summary_prompt_key}' not found in dynamic_data. Using default.")
			if E is F:
				if A in B.input_keys_for_prompt or A==B.summary_prompt_key and B.summary_prompt_key:C.warning(f"Node '{B.node_id}': Key '{A}' for prompt was not found; using empty string.")
				H[A]=''
			elif I(E,(J,K)):
				try:H[A]=json.dumps(E,indent=2,ensure_ascii=False,default=G)
				except TypeError:H[A]=G(E)
			else:H[A]=G(E)
		C.debug(f"Node '{B.node_id}': Prepared prompt input keys: {J(H.keys())}");return H
	async def __call__(A,state):
		P='__end__';B=state
		with R.start_as_current_span('graph.node.direct_processor',attributes={'node_id':A.node_id,'task_id':B.task_id})as E:
			C.info(f"DirectProcessorNode '{A.node_id}' execution started. Task ID: {B.task_id}");E.set_attribute('app.node.id',A.node_id);await A.notification_service.broadcast_to_task(B.task_id,N(task_id=B.task_id,status='node_executing',detail=f"Node '{A.node_id}' started.",current_node=A.node_id));H=F;I=F
			try:
				S=await A._prepare_prompt_input(B);J=A.prompt_template_engine.format(**S);E.set_attribute('app.llm.prompt_length',L(J));C.debug(f"Node '{A.node_id}' (Task: {B.task_id}): Formatted prompt:\n{J[:500]}...");T=[{'role':'user','content':J}];K={}
				if A.temperature is not F:K['temperature']=A.temperature
				if A.max_tokens is not F:K['max_tokens']=A.max_tokens
				I=await A.llm_client.generate_response(messages=T,model_name=A.model_name,**K);E.set_attribute('app.llm.response_length',L(I or''));C.info(f"Node '{A.node_id}' (Task: {B.task_id}): Direct processing completed successfully.");await A.notification_service.broadcast_to_task(B.task_id,Q(task_id=B.task_id,node_id=A.node_id,result_step_name='direct_processing_complete',data={'result_length':L(I)if I else 0}))
			except O as M:H=f"Error during direct processing in node '{A.node_id}': {G(M)}";C.error(f"Node '{A.node_id}' (Task: {B.task_id}): {H}",exc_info=True);E.set_status(D.Status(D.StatusCode.ERROR,description=H));E.record_exception(M);I=f"An error occurred while processing your request: {G(M)}"
			await A.notification_service.broadcast_to_task(B.task_id,N(task_id=B.task_id,status='node_completed',detail=f"Node '{A.node_id}' finished. Error: {H or"None"}",current_node=A.node_id,next_node=P))
			if H:E.set_status(D.Status(D.StatusCode.ERROR,description=H))
			else:E.set_status(D.Status(D.StatusCode.OK))
			return{'final_answer':I,'error_message':H,'dynamic_data':B.dynamic_data,'next_action':P}
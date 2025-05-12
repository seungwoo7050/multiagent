g='subtask_title'
V='original_input'
U='task_complexity_evaluator'
T=Exception
M='description'
L='title'
E=None
import os
from typing import Any,Dict,Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate as H
from src.utils.logger import get_logger as A
from src.config.settings import get_settings as B
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage as K,IntermediateResultMessage as f
from opentelemetry import trace
o=trace.get_tracer(__name__)
D=A(__name__)
C=B()
class F:
	def __init__(A,llm_client,notification_service,temperature=.3,prompt_template_path='generic/task_complexity_evaluation.txt',model_name=E,node_id=U):A.llm_client=llm_client;A.notification_service=notification_service;A.temperature=temperature;A.prompt_template_path=prompt_template_path;A.model_name=model_name;A.node_id=node_id;A.prompt_template_str=A._load_prompt_template_if_path_exists();D.info(f"TaskComplexityEvaluatorNode '{A.node_id}' initialized. Prompt: '{A.prompt_template_path if A.prompt_template_path else"Default internal"}'")
	def _load_prompt_template_if_path_exists(A):
		if not A.prompt_template_path:return
		E=getattr(C,'PROMPT_TEMPLATE_DIR','config/prompts')
		if os.path.isabs(A.prompt_template_path):B=A.prompt_template_path
		else:B=os.path.join(E,A.prompt_template_path)
		try:
			with open(B,'r',encoding='utf-8')as F:D.debug(f"Successfully loaded prompt template from: {B} for node '{A.node_id}'");return F.read()
		except FileNotFoundError:D.warning(f"Prompt template file not found for TaskComplexityEvaluatorNode '{A.node_id}': {B}. Using default internal prompt.");return
		except T as G:D.error(f"Error loading prompt template from {B} for node '{A.node_id}': {G}. Using default internal prompt.");return
	def _construct_prompt(B,subtask,state):
		G='No description provided';F='Untitled Subtask';C=state;A=subtask
		if B.prompt_template_str:
			try:E={V:C.original_input,g:A.get(L,F),'subtask_description':A.get(M,G)};I=H(template=B.prompt_template_str,input_variables=list(E.keys()));return I.format(**E)
			except T as J:D.error(f"Error formatting prompt template in TaskComplexityEvaluatorNode '{B.node_id}': {J}. Falling back to default internal prompt.")
		return f'''
    You are a task complexity evaluator. Your job is to determine whether a subtask requires complex reasoning (Tree of Thoughts) or is straightforward enough for a single-step process (Generic LLM).

    Original Task: {C.original_input}

    Subtask to Evaluate:
    Title: {A.get(L,F)}
    Description: {A.get(M,G)}

    Instructions:
    1. Analyze the subtask carefully considering its nature, difficulty, and requirements.
    2. Determine if it requires complex reasoning with multiple steps of thought (Tree of Thoughts)
       OR if it\'s straightforward enough for a single, direct response (Generic LLM).
    3. Consider these factors:
       - Does it require breaking down into multiple reasoning steps?
       - Does it involve comparing multiple possible approaches?
       - Does it benefit from exploring different thought paths?
       - Does it need evaluation of intermediate results?
       - Is it creative/open-ended or more factual/direct?

    You must respond with EXACTLY ONE of these two answers:
    - "COMPLEX": This task requires Tree of Thoughts for multi-step reasoning
    - "SIMPLE": This task can be handled with a single Generic LLM call

    Your evaluation:
    '''
	async def __call__(B,state,config=E):
		n='process_simple_subtask';m='complex';l='subtask_index';k='complexity_eval_retries';j='final_answer';i='Untitled';h='result';e='process_complex_subtask';d=True;c='node_completed';b='__end__';a='current_subtask';S='is_complex';R='subtasks';Q='current_subtask_index';J='next_action';I='dynamic_data';A=state
		if A.dynamic_data and Q in A.dynamic_data:p=A.dynamic_data[Q];A.dynamic_data[a]=A.dynamic_data[R][p]
		with o.start_as_current_span('graph.node.task_complexity_evaluator',attributes={'node_id':B.node_id,'task_id':A.task_id}):
			D.info(f"TaskComplexityEvaluatorNode '{B.node_id}' execution started. Task ID: {A.task_id}");await B.notification_service.broadcast_to_task(A.task_id,K(task_id=A.task_id,status='node_executing',detail=f"Node '{B.node_id}' (Complexity Evaluator) started.",current_node=B.node_id));N=E
			if not A.dynamic_data or R not in A.dynamic_data or not A.dynamic_data[R]:N='No subtasks available to evaluate';D.error(f"Node '{B.node_id}' (Task: {A.task_id}): {N}");return{I:A.dynamic_data,'error_message':N,J:b}
			C=A.dynamic_data.get(Q,0);F=A.dynamic_data[R]
			if C is E or C>=len(F):
				D.info(f"Node '{B.node_id}' (Task: {A.task_id}): All subtasks have been evaluated");W=[]
				for X in F:
					if h in X:W.append(f"Subtask: {X.get(L,i)}\nResult: {X[h]}")
				q='\n\n'.join(W)if W else'No results were produced for the subtasks.';await B.notification_service.broadcast_to_task(A.task_id,K(task_id=A.task_id,status=c,detail=f"Node '{B.node_id}' (Complexity Evaluator) finished. All subtasks processed.",current_node=B.node_id,next_node=b));A.dynamic_data['processing_complete']=d;return{I:A.dynamic_data,j:q,J:b}
			H=F[C]
			try:
				r=B._construct_prompt(H,A);D.debug(f"Node '{B.node_id}' (Task: {A.task_id}): Evaluation prompt constructed for subtask {C}");O=A.dynamic_data.get(k,{});Y=str(C);O[Y]=O.get(Y,0)+1;A.dynamic_data[k]=O
				if O[Y]>2:D.warning(f"Node '{B.node_id}' (Task: {A.task_id}): Maximum retry attempts reached for subtask {C}. Using default complexity.");F[C][S]=d;await B.notification_service.broadcast_to_task(task_id=A.task_id,message=f(task_id=A.task_id,node_id=B.node_id,result_step_name='subtask_complexity_default',data={l:C,'default_complexity':m}));P=e;A.dynamic_data[a]=H;return{I:A.dynamic_data.copy(),V:H.get(M,A.original_input),J:P}
				s=await B.llm_client.generate_response(messages=[{'role':'user','content':r}],model_name=B.model_name,temperature=B.temperature,max_tokens=100);D.debug(f"Node '{B.node_id}' (Task: {A.task_id}): LLM evaluation received for subtask {C}");t=s.strip().upper();G='COMPLEX'in t;D.debug(f"[TCE] task_id={A.task_id} idx={C} is_complex={G} next_action={e if G else n}");F[C][S]=G;D.info(f"Node '{B.node_id}' (Task: {A.task_id}): Subtask {C} evaluated as {"complex (ToT)"if G else"simple (GenericLLM)"}");await B.notification_service.broadcast_to_task(A.task_id,f(task_id=A.task_id,node_id=B.node_id,result_step_name='subtask_evaluated',data={l:C,g:H.get(L,i),S:G}));P=e if G else n;A.dynamic_data[a]=H;await B.notification_service.broadcast_to_task(A.task_id,K(task_id=A.task_id,status=c,detail=f"Node '{B.node_id}' (Complexity Evaluator) finished. Subtask {C} is {m if G else"simple"}.",current_node=B.node_id,next_node=P));u=H.get(M,A.original_input);return{I:A.dynamic_data.copy()if A.dynamic_data else{},V:u,J:P}
			except T as Z:D.error(f"Node '{B.node_id}' (Task: {A.task_id}): Error during complexity evaluation: {Z}",exc_info=d);N=f"Error in TaskComplexityEvaluatorNode '{B.node_id}': {Z}";F[C][S]=E;F[C]['error']=str(Z);A.dynamic_data[Q]=C+1;await B.notification_service.broadcast_to_task(A.task_id,K(task_id=A.task_id,status=c,detail=f"Node '{B.node_id}' (Complexity Evaluator) encountered an error. Moving to next subtask.",current_node=B.node_id,next_node=U));return{I:A.dynamic_data,j:E,'thoughts':[],'current_thoughts_to_evaluate':[],'current_best_thought_id':E,'search_depth':0,J:U}
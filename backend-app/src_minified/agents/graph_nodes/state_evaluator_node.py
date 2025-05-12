J=ValueError
H=float
N=Exception
I=str
G=None
import os,re as C
from typing import Any,Dict,Optional,Tuple
from langchain_core.runnables import RunnableConfig
from src.utils.logger import get_logger as A
from src.config.settings import get_settings as B
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState,Thought as T
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage as M,IntermediateResultMessage as U
from opentelemetry import trace
f=trace.get_tracer(__name__)
D=A(__name__)
E=B()
class F:
	def __init__(A,llm_client,notification_service,max_tokens_per_eval=150,temperature=.4,prompt_template_path=G,model_name=G,node_id='state_evaluator',default_score=.5):B=notification_service;A.llm_client=llm_client;A.notification_service=B;A.max_tokens_per_eval=max_tokens_per_eval;A.temperature=temperature;A.prompt_template_path=prompt_template_path;A.model_name=model_name;A.node_id=node_id;A.default_score=default_score;A.prompt_template_str=A._load_prompt_template_if_path_exists();D.info(f"StateEvaluatorNode '{A.node_id}' initialized. NotificationService injected: {"Yes"if B else"No"}. Prompt: '{A.prompt_template_path if A.prompt_template_path else"Default internal"}'")
	def _load_prompt_template_if_path_exists(A):
		if not A.prompt_template_path:return
		C=getattr(E,'PROMPT_TEMPLATE_DIR','config/prompts')
		if os.path.isabs(A.prompt_template_path):B=A.prompt_template_path
		else:B=os.path.join(C,A.prompt_template_path)
		try:
			with open(B,'r',encoding='utf-8')as F:D.debug(f"Successfully loaded prompt template from: {B} for node '{A.node_id}'");return F.read()
		except FileNotFoundError:D.warning(f"Prompt template file not found for StateEvaluatorNode '{A.node_id}': {B}. Using default internal prompt.");return
		except N as G:D.error(f"Error loading prompt template from {B} for node '{A.node_id}': {G}. Using default internal prompt.");return
	def _construct_evaluation_prompt(A,state,thought_content,parent_thought_content):
		E=parent_thought_content;C=thought_content;B=state
		if A.prompt_template_str:
			from langchain_core.prompts import PromptTemplate as G
			try:F={'original_input':B.original_input,'parent_thought_content':E,'thought_to_evaluate_content':C};H=G(template=A.prompt_template_str,input_variables=list(F.keys()));return H.format(**F)
			except KeyError as I:D.error(f"Missing key for prompt template in StateEvaluatorNode '{A.node_id}': {I}. Falling back to default internal prompt.")
			except N as J:D.error(f"Error formatting prompt template in StateEvaluatorNode '{A.node_id}': {J}. Falling back to default internal prompt.")
		K='\n    Evaluation Criteria:\n    1. Relevance to Goal (Weight: 40%): How directly and significantly does this thought contribute to achieving the "Overall Goal"?\n    2. Progress Likelihood (Weight: 30%): How likely is pursuing this thought to lead to tangible progress or valuable new information?\n    3. Feasibility & Actionability (Weight: 20%): How practical, actionable, and achievable is this thought given typical constraints (time, resources, information availability)? Is it specific enough to act upon?\n    4. Novelty/Insight (Weight: 10%): Does this thought offer a new perspective, a creative approach, or a particularly insightful next step? (Bonus, not strictly required for a good score if other criteria are met).\n    \n    IMPORTANT: Be generous with your scoring! If a thought has any merit at all, score it at least 0.3.\n    - Score 0.7-1.0: An excellent thought that directly addresses the goal\n    - Score 0.5-0.69: A good thought with clear value \n    - Score 0.3-0.49: An average thought with some potential\n    - Score 0.1-0.29: A weak thought with limited value\n    - Score 0-0.09: A completely irrelevant or impractical thought\n    ';return f'''
    You are an expert AI Evaluator. Your task is to critically assess the promise and quality of a given "Thought" in the context of achieving an "Overall Goal", potentially originating from a "Parent Thought".

    Overall Goal:
    {B.original_input}

    Parent Thought (Context for the thought being evaluated, if applicable):
    {E}
    (Note: If the thought is an initial thought, Parent Thought might be "Initial problem analysis" or similar.)

    Thought to Evaluate:
    "{C}"

    {K}

    Instructions for Evaluation:
    1. Provide a numerical score between 0.0 (not at all promising) and 1.0 (highly promising) for the "Thought to Evaluate".
    2. Remember: Be generous in your scoring. If the thought is directly useful, score it 0.7 or higher.
    3. Provide a concise reasoning for your score, briefly touching upon the relevant criteria. Your reasoning should justify the score.

    Output Format:
    Score: [A single float value between 0.0 and 1.0]
    Reasoning: [Your concise textual reasoning, typically 1-3 sentences.]

    Begin Evaluation:
    '''
	def _parse_evaluation_response(G,response_str):
		L='No reasoning provided.';A=response_str;A=A.strip();I=C.search('Score:\\s*([0-9.]+)',A,C.IGNORECASE)
		if I:
			try:
				B=H(I.group(1))
				if 0<=B<=1:K=C.search('Reasoning:\\s*([\\s\\S]+)',A,C.IGNORECASE|C.DOTALL);E=K.group(1).strip()if K else L;return B,E
			except J:pass
		F=A.split('\n')
		if F and F[0].strip().replace('.','',1).isdigit():
			try:
				B=H(F[0].strip())
				if 0<=B<=1:E='\n'.join(F[1:]).strip()or L;return B,E
			except J:pass
		M=C.findall('([0-9](?:\\.[0-9]+)?)',A)
		for N in M:
			try:
				B=H(N)
				if 0<=B<=1:E=f"Score extracted from text. Original response: {A[:200]}...";return B,E
			except J:continue
		D.warning(f"Could not parse score from evaluation response. Using default score {G.default_score}. Response: {A[:100]}...");return G.default_score,f"Score parsing failed. Original response: {A[:200]}..."
	async def __call__(A,state,config=G):
		e='thought_id';d='evaluated';c='eval_reasoning';b='current_thoughts_to_evaluate';a='search_strategy';Z='node_completed';S=True;B=state
		with f.start_as_current_span('graph.node.state_evaluator',attributes={'node_id':A.node_id,'task_id':B.task_id,'thoughts_to_eval':len(B.current_thoughts_to_evaluate)}):
			D.info(f"StateEvaluatorNode '{A.node_id}' execution started. Task ID: {B.task_id}");await A.notification_service.broadcast_to_task(B.task_id,M(task_id=B.task_id,status='node_executing',detail=f"Node '{A.node_id}' (State Evaluator) started.",current_node=A.node_id));J={A.id:A for A in B.thoughts};K=G;O=False;L=[]
			if not B.current_thoughts_to_evaluate:D.info(f"Node '{A.node_id}' (Task: {B.task_id}): No thoughts to evaluate.");await A.notification_service.broadcast_to_task(B.task_id,M(task_id=B.task_id,status=Z,detail=f"Node '{A.node_id}' finished: No thoughts to evaluate.",current_node=A.node_id,next_node=a if B.thoughts else G));return{b:[]}
			for E in B.current_thoughts_to_evaluate:
				C=J.get(E)
				if not C:D.warning(f"Node '{A.node_id}' (Task: {B.task_id}): Thought with ID '{E}' not found. Skipping.");continue
				V='N/A (Initial thought or parent not found)'
				if C.parent_id:
					W=B.get_thought_by_id(C.parent_id)
					if W:V=W.content
				try:g=A._construct_evaluation_prompt(B,C.content,V);D.debug(f"Node '{A.node_id}' (Task: {B.task_id}): Evaluation prompt for thought '{C.id}'.");h=[{'role':'user','content':g}];P=await A.llm_client.generate_response(messages=h,model_name=A.model_name,temperature=A.temperature,max_tokens=A.max_tokens_per_eval);D.debug(f"Node '{A.node_id}' (Task: {B.task_id}): Evaluation response for '{C.id}': {P[:100]}...");Q,X=A._parse_evaluation_response(P);F=C.metadata.copy()if C.metadata else{};F[c]=X;F['raw_eval_response']=P;i=T(id=C.id,parent_id=C.parent_id,content=C.content,evaluation_score=Q,status=d,metadata=F);J[E]=i;O=S;L.append(f"Thought '{C.id}': Score={Q:.2f}");await A.notification_service.broadcast_to_task(B.task_id,U(task_id=B.task_id,node_id=A.node_id,result_step_name='thought_evaluated',data={e:E,'score':Q,'reasoning':X[:150]+'...'}))
				except N as H:D.error(f"Node '{A.node_id}' (Task: {B.task_id}): Error evaluating thought '{E}': {H}",exc_info=S);F=C.metadata.copy()if C.metadata else{};F['eval_error']=I(H);F[c]=f"Evaluation failed, using default score {A.default_score}. Error: {I(H)}";j=T(id=C.id,parent_id=C.parent_id,content=C.content,evaluation_score=A.default_score,status=d,metadata=F);J[E]=j;O=S;L.append(f"Thought '{E}': Score={A.default_score:.2f} (fallback)");K=(K or'')+f"Error evaluating {E}: {I(H)}; ";await A.notification_service.broadcast_to_task(B.task_id,U(task_id=B.task_id,node_id=A.node_id,result_step_name='thought_evaluation_fallback',data={e:E,'fallback_score':A.default_score,'error':I(H)}))
			k=list(J.values());Y=', '.join(L)if L else'No new evaluations.';D.info(f"Node '{A.node_id}' (Task: {B.task_id}): Evaluations completed. Summary: {Y}");R={'thoughts':k,b:[]}
			if O:R['last_llm_output']=f"Evaluations completed for thoughts: {", ".join(B.current_thoughts_to_evaluate)}. Summary: {Y}"
			if K:R['error_message']=K
			await A.notification_service.broadcast_to_task(B.task_id,M(task_id=B.task_id,status=Z,detail=f"Node '{A.node_id}' (State Evaluator) finished. Evaluated {len(B.current_thoughts_to_evaluate)} thoughts.",current_node=A.node_id,next_node=a));return R
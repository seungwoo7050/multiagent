N='content'
M='search_depth'
L='num_thoughts'
K='error_message'
J=len
I=Exception
E=None
import os,re
from typing import Any,Dict,List,Optional
from langchain_core.runnables import RunnableConfig
from src.utils.logger import get_logger as A
from src.config.settings import get_settings as B
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage as H,IntermediateResultMessage as S
from opentelemetry import trace
T=trace.get_tracer(__name__)
C=A(__name__)
D=B()
class U:
	def __init__(A,llm_client,notification_service,num_thoughts=3,max_tokens_per_thought=500,temperature=.7,prompt_template_path=E,model_name=E,node_id='thought_generator'):B=notification_service;A.llm_client=llm_client;A.notification_service=B;A.num_thoughts=num_thoughts;A.max_tokens_per_thought=max_tokens_per_thought;A.temperature=temperature;A.prompt_template_path=prompt_template_path;A.model_name=model_name;A.node_id=node_id;A.prompt_template_str=A._load_prompt_template_if_path_exists();C.info(f"ThoughtGeneratorNode '{A.node_id}' initialized. Num thoughts: {A.num_thoughts}. NotificationService injected: {"Yes"if B else"No"}. Prompt: '{A.prompt_template_path if A.prompt_template_path else"Default internal"}'")
	def _load_prompt_template_if_path_exists(A):
		if not A.prompt_template_path:return
		E=getattr(D,'PROMPT_TEMPLATE_DIR','config/prompts')
		if os.path.isabs(A.prompt_template_path):B=A.prompt_template_path
		else:B=os.path.join(E,A.prompt_template_path)
		try:
			with open(B,'r',encoding='utf-8')as F:C.debug(f"Successfully loaded prompt template from: {B} for node '{A.node_id}'");return F.read()
		except FileNotFoundError:C.warning(f"Prompt template file not found for ThoughtGeneratorNode '{A.node_id}': {B}. Using default internal prompt.");return
		except I as G:C.error(f"Error loading prompt template from {B} for node '{A.node_id}': {G}. Using default internal prompt.");return
	def _construct_prompt(D,state):
		O='N/A (Initial thought generation)';A=state;F=O
		if A.current_best_thought_id:
			H=A.get_thought_by_id(A.current_best_thought_id)
			if H:F=H.content
		G=[]
		if F!=O and A.current_best_thought_id:
			for B in A.thoughts:
				if B.parent_id==A.current_best_thought_id and B.status=='evaluated'and B.evaluation_score is not E:G.append(f"Sibling Thought (ID: {B.id}, Score: {B.evaluation_score:.2f}): {B.content[:100]}...")
				elif B.parent_id==A.current_best_thought_id and B.status=='evaluation_failed':G.append(f"Sibling Thought (ID: {B.id}, Status: evaluation_failed): {B.content[:100]}...")
		J='\n'.join(G)if G else'No previously explored sibling thoughts from this parent.'
		if D.prompt_template_str:
			from langchain_core.prompts import PromptTemplate as P
			try:N={'original_input':A.original_input,'parent_thought_content':F,'sibling_thoughts_summary':J,L:D.num_thoughts,M:A.search_depth,'max_search_depth':A.max_search_depth,K:f"Current Error (if any, try to overcome or sidestep this): {A.error_message}"if A.error_message else''};Q=P(template=D.prompt_template_str,input_variables=list(N.keys()));return Q.format(**N)
			except KeyError as R:C.error(f"Missing key for prompt template in ThoughtGeneratorNode '{D.node_id}': {R}. Falling back to default internal prompt.")
			except I as S:C.error(f"Error formatting prompt template in ThoughtGeneratorNode '{D.node_id}': {S}. Falling back to default internal prompt.")
		return f'''
    You are a creative and methodical problem solver. Your task is to generate a set of diverse and promising next steps (thoughts) to achieve a given goal, based on the current context.

    Overall Goal: {A.original_input}

    Current Context / Parent Thought for Expansion:
    {F}

    Previously Explored Sibling Thoughts and Their Outcomes (if any, for context):
    {J}

    Current Search Depth: {A.search_depth} / {A.max_search_depth}
    {"Current Error (if any, try to generate thoughts to overcome or sidestep this): "+A.error_message if A.error_message else""}

    Instructions:
    1. Analyze the "Overall Goal" and the "Current Context/Parent Thought".
    2. Generate exactly {D.num_thoughts} distinct, actionable, and forward-looking thoughts.
    3. Each thought should represent a potential next step, a hypothesis to test, a question to answer, or a sub-problem to solve.
    4. Thoughts should be diverse, exploring different angles or approaches if possible.
    5. Avoid thoughts that are too vague, too broad, or simply restate the current problem. Be specific and constructive.
    6. If "Previously Explored Sibling Thoughts" are provided, try not to generate highly similar thoughts unless you have a significantly new angle or refinement.
    7. Ensure thoughts are concise and clearly phrased.

    Output Format:
    Provide each thought on a new line, prefixed with "Thought: ".
    Example (if Overall Goal is "Plan a 3-day trip to Paris" and Parent Thought is "Day 1: Focus on iconic landmarks"):
    Thought: Research opening hours and ticket prices for the Eiffel Tower and Louvre Museum.
    Thought: Plan a walking route connecting the Eiffel Tower, Arc de Triomphe, and Champs-Élysées.
    Thought: Identify potential lunch spots near the Louvre with good reviews.

    Begin Generating Thoughts:
    '''
	@staticmethod
	def _extract_thoughts(raw):
		D=re.compile('\n      ^\\s* # 앞 공백\n      (?:thought\\s*:|[-\\d.)]+)?\\s* # Thought: 또는 불릿/번호\n      (?P<content>.+?)         # 실제 텍스트\n      \\s*$              # 뒤 공백\n      ',re.IGNORECASE|re.VERBOSE);A=[]
		for E in raw.splitlines():
			B=D.match(E)
			if B:
				C=B.group(N).strip()
				if C:A.append(C)
		return A
	async def __call__(A,state,config=E):
		R='current_thoughts_to_evaluate';Q='node_completed';B=state
		with T.start_as_current_span('graph.node.thought_generator',attributes={'node_id':A.node_id,'task_id':B.task_id,L:A.num_thoughts,M:B.search_depth}):
			C.info(f"ThoughtGeneratorNode '{A.node_id}' execution started. Task ID: {B.task_id}");await A.notification_service.broadcast_to_task(B.task_id,H(task_id=B.task_id,status='node_executing',detail=f"Node '{A.node_id}' (Thought Generator) started.",current_node=A.node_id));G=E;D=[]
			if B.search_depth>=B.max_search_depth:C.info(f"Node '{A.node_id}': Max search depth ({B.max_search_depth}) reached. No new thoughts generated for task {B.task_id}.");await A.notification_service.broadcast_to_task(B.task_id,H(task_id=B.task_id,status=Q,detail=f"Node '{A.node_id}' finished: Max search depth reached.",current_node=A.node_id));return{R:[],K:'Max search depth reached.'}
			try:
				V=A._construct_prompt(B);C.debug(f"Node '{A.node_id}' (Task: {B.task_id}): Generation prompt constructed.");O=await A.llm_client.generate_response(messages=[{'role':'user',N:V}],model_name=A.model_name,temperature=A.temperature,max_tokens=A.max_tokens_per_thought*A.num_thoughts);C.debug(f"Node '{A.node_id}' (Task: {B.task_id}): LLM response received.");D=U._extract_thoughts(O);D=D[:A.num_thoughts]
				if not D:C.warning(f"Node '{A.node_id}' (Task: {B.task_id}): LLM did not generate any valid thoughts from response: {O[:200]}...");G=f"LLM did not generate thoughts in node '{A.node_id}'."
				else:await A.notification_service.broadcast_to_task(B.task_id,S(task_id=B.task_id,node_id=A.node_id,result_step_name='thoughts_generated',data={'generated_count':J(D),'thoughts_preview':[A[:100]+'...'for A in D]}))
			except I as P:C.error(f"Node '{A.node_id}' (Task: {B.task_id}): Error during thought generation: {P}",exc_info=True);G=f"Error in ThoughtGeneratorNode '{A.node_id}': {P}";D=[]
			F=[]
			if D:
				for W in D:X=B.add_thought(content=W,parent_id=B.current_best_thought_id);F.append(X.id)
				if F:C.info(f"Node '{A.node_id}' (Task: {B.task_id}): Generated and added {J(F)} new thoughts to state.")
			await A.notification_service.broadcast_to_task(B.task_id,H(task_id=B.task_id,status=Q,detail=f"Node '{A.node_id}' (Thought Generator) finished. Generated {J(F)} thoughts.",current_node=A.node_id,next_node='state_evaluator'));return{'thoughts':B.thoughts,R:F,'last_llm_output':D if D else G or'No thoughts generated.',K:G}
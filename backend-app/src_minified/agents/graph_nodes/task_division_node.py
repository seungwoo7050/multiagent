_A=None
import os
from typing import Any,Dict,List,Optional
import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage,IntermediateResultMessage
from opentelemetry import trace
tracer=trace.get_tracer(__name__)
logger=get_logger(__name__)
settings=get_settings()
class TaskDivisionNode:
	def __init__(self,llm_client,notification_service,max_subtasks=4,min_subtasks=2,temperature=.7,prompt_template_path='generic/task_division.txt',model_name=_A,node_id='task_divider'):self.llm_client=llm_client;self.notification_service=notification_service;self.max_subtasks=max_subtasks;self.min_subtasks=min_subtasks;self.temperature=temperature;self.prompt_template_path=prompt_template_path;self.model_name=model_name;self.node_id=node_id;self.prompt_template_str=self._load_prompt_template_if_path_exists();logger.info(f"TaskDivisionNode '{self.node_id}' initialized. Max subtasks: {self.max_subtasks}, Min subtasks: {self.min_subtasks}. Prompt: '{self.prompt_template_path if self.prompt_template_path else"Default internal"}'")
	def _load_prompt_template_if_path_exists(self):
		if not self.prompt_template_path:return
		base_prompt_dir=getattr(settings,'PROMPT_TEMPLATE_DIR','config/prompts')
		if os.path.isabs(self.prompt_template_path):full_path=self.prompt_template_path
		else:full_path=os.path.join(base_prompt_dir,self.prompt_template_path)
		try:
			with open(full_path,'r',encoding='utf-8')as f:logger.debug(f"Successfully loaded prompt template from: {full_path} for node '{self.node_id}'");return f.read()
		except FileNotFoundError:logger.warning(f"Prompt template file not found for TaskDivisionNode '{self.node_id}': {full_path}. Using default internal prompt.");return
		except Exception as e:logger.error(f"Error loading prompt template from {full_path} for node '{self.node_id}': {e}. Using default internal prompt.");return
	def _construct_prompt(self,state):
		if self.prompt_template_str:
			try:prompt_data={'original_input':state.original_input,'max_subtasks':self.max_subtasks,'min_subtasks':self.min_subtasks};template=PromptTemplate(template=self.prompt_template_str,input_variables=list(prompt_data.keys()));return template.format(**prompt_data)
			except Exception as e:logger.error(f"Error formatting prompt template in TaskDivisionNode '{self.node_id}': {e}. Falling back to default internal prompt.")
		return f"""
    You are a task breakdown specialist. Your job is to divide a complex task into smaller, more manageable subtasks.

    Original Task: {state.original_input}

    Instructions:
    1. Analyze the original task and break it down into {self.min_subtasks}-{self.max_subtasks} subtasks.
    2. Each subtask should be self-contained and focused on a specific aspect of the problem.
    3. Subtasks should collectively cover all aspects of the original task.
    4. Order the subtasks logically - earlier subtasks may provide inputs to later ones.
    5. Give each subtask a short but descriptive title.

    Output Format:
    For each subtask, provide:
    Subtask #[number]: [Brief title]
    Description: [Detailed description of what this subtask involves]

    Begin Task Division:
    """
	def _parse_subtasks(self,response):
		C=':';B='title';A='description';lines=response.strip().split('\n');subtasks=[];current_subtask=_A
		for line in lines:
			line=line.strip()
			if not line:continue
			if line.lower().startswith('subtask #')or line.lower().startswith('subtask:'):
				if current_subtask and B in current_subtask and A in current_subtask:subtasks.append(current_subtask)
				title_part=line.split(C,1)[1].strip()if C in line else'';current_subtask={'id':str(uuid.uuid4()),B:title_part,A:'','is_complex':_A}
			elif current_subtask and line.lower().startswith('description:'):current_subtask[A]=line.split(C,1)[1].strip()
			elif current_subtask and A in current_subtask and current_subtask[A]:current_subtask[A]+=' '+line
			elif current_subtask and not current_subtask[A]:
				if not current_subtask[B]:current_subtask[B]=line
				else:current_subtask[A]=line
		if current_subtask and B in current_subtask and A in current_subtask:subtasks.append(current_subtask)
		return subtasks
	async def __call__(self,state,config=_A):
		A='subtasks'
		with tracer.start_as_current_span('graph.node.task_division',attributes={'node_id':self.node_id,'task_id':state.task_id}):
			logger.info(f"TaskDivisionNode '{self.node_id}' execution started. Task ID: {state.task_id}");await self.notification_service.broadcast_to_task(state.task_id,StatusUpdateMessage(task_id=state.task_id,status='node_executing',detail=f"Node '{self.node_id}' (Task Division) started.",current_node=self.node_id));error_message=_A;subtasks=[]
			try:
				division_prompt=self._construct_prompt(state);logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): Division prompt constructed.");full_response=await self.llm_client.generate_response(messages=[{'role':'user','content':division_prompt}],model_name=self.model_name,temperature=self.temperature,max_tokens=1000);logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): LLM response received.");subtasks=self._parse_subtasks(full_response)
				if not subtasks:logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): No subtasks could be parsed from LLM response.");error_message='Failed to divide task into subtasks.'
				else:logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Successfully divided task into {len(subtasks)} subtasks.");await self.notification_service.broadcast_to_task(state.task_id,IntermediateResultMessage(task_id=state.task_id,node_id=self.node_id,result_step_name='subtasks_created',data={'subtask_count':len(subtasks),A:subtasks}))
			except Exception as e:logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): Error during task division: {e}",exc_info=True);error_message=f"Error in TaskDivisionNode '{self.node_id}': {e}";subtasks=[]
			if not state.dynamic_data:state.dynamic_data={}
			state.dynamic_data[A]=subtasks;state.dynamic_data['current_subtask_index']=0 if subtasks else _A;await self.notification_service.broadcast_to_task(state.task_id,StatusUpdateMessage(task_id=state.task_id,status='node_completed',detail=f"Node '{self.node_id}' (Task Division) finished. Created {len(subtasks)} subtasks.",current_node=self.node_id,next_node='task_complexity_evaluator'));return{'dynamic_data':state.dynamic_data,'error_message':error_message,'last_llm_output':full_response if'full_response'in locals()else _A}
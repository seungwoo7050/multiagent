M=len
K=None
from typing import Any,Dict,Optional
from langchain_core.runnables import RunnableConfig
from src.utils.logger import get_logger as A
from src.config.settings import get_settings as B
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage as L,IntermediateResultMessage as k
from opentelemetry import trace
l=trace.get_tracer(__name__)
E=A(__name__)
C=B()
class D:
	def __init__(A,llm_client,notification_service,node_id='subtask_processor',early_exit_threshold=.9):A.llm_client=llm_client;A.notification_service=notification_service;A.node_id=node_id;A.early_exit_threshold=early_exit_threshold;E.info(f"SubtaskProcessorNode '{A.node_id}' initialized with early exit threshold: {A.early_exit_threshold}.")
	async def __call__(B,state,config=K):
		j='task_complexity_evaluator';i='processing_complete';h='\n\n';g='Untitled';f='title';e='processed_subtasks_count';d='current_best_score';c='decision';b='current_subtask';Z='final_answer';V='synthesis_node';U='__end__';T=True;S='current_subtask_index';Q='next_action';P='dynamic_data';O='node_completed';J='search_strategy_decision';I='result';D='subtasks';A=state
		with l.start_as_current_span('graph.node.subtask_processor',attributes={'node_id':B.node_id,'task_id':A.task_id}):
			E.info(f"SubtaskProcessorNode '{B.node_id}' execution started. Task ID: {A.task_id}")
			if A.dynamic_data and b in A.dynamic_data:
				C=A.dynamic_data.get(S,0)
				if D in A.dynamic_data and C<M(A.dynamic_data[D]):A.dynamic_data[D][C][I]=A.final_answer;E.info(f"Node '{B.node_id}' (Task: {A.task_id}): Stored result for subtask {C}")
			W=False;m=B.early_exit_threshold;R=K
			if A.dynamic_data and isinstance(A.dynamic_data.get(J),dict):
				R=A.dynamic_data.get(J,{}).get(c);X=A.dynamic_data.get(J,{}).get(d,0)
				if R=='finish_very_high_score'and X>=m or X>=.95:W=T;E.info(f"Node '{B.node_id}' (Task: {A.task_id}): Detected excellent result with score {X}. Considering early workflow completion.")
			if A.dynamic_data and S in A.dynamic_data:
				C=A.dynamic_data[S];N=A.dynamic_data.get(D,[]);F=A.dynamic_data.get(e,0);F+=1;A.dynamic_data[e]=F
				if A.next_action=='finish'and b in A.dynamic_data:
					E.info(f"Node '{B.node_id}' (Task: {A.task_id}): Search strategy determined to finish with high score. Current subtask result quality: {W}.")
					if D in A.dynamic_data and C<M(A.dynamic_data[D]):
						if not A.dynamic_data[D][C].get(I):A.dynamic_data[D][C][I]=A.final_answer or'Completed via search strategy'
				if R:
					if J not in A.dynamic_data:A.dynamic_data[J]={}
					A.dynamic_data[J][c]=R
				a=10
				if F>=a:
					E.warning(f"Node '{B.node_id}' (Task: {A.task_id}): Maximum subtask processing limit ({a}) reached.");G=[]
					for H in N:
						if I in H:G.append(f"Subtask: {H.get(f,g)}\nResult: {H[I]}")
					Y=h.join(G)if G else'Processing stopped due to maximum subtask limit.';await B.notification_service.broadcast_to_task(A.task_id,L(task_id=A.task_id,status=O,detail=f"Node '{B.node_id}' finished. Maximum subtasks reached.",current_node=B.node_id,next_node=U));return{P:A.dynamic_data,Z:Y,Q:U}
				if W and F>=3:E.info(f"Node '{B.node_id}' (Task: {A.task_id}): Found excellent result after processing {F} subtasks. Proceeding to synthesis early.");A.dynamic_data[i]=T;A.dynamic_data['early_termination']=T;A.dynamic_data['early_termination_reason']=f"Found excellent result with score {A.dynamic_data.get(J,{}).get(d,"high")} after processing {F}/{M(N)} subtasks";await B.notification_service.broadcast_to_task(A.task_id,k(task_id=A.task_id,node_id=B.node_id,result_step_name='early_workflow_completion',data={'processed_subtasks':F,'total_subtasks':M(N),'reason':'excellent_result'}));await B.notification_service.broadcast_to_task(A.task_id,L(task_id=A.task_id,status=O,detail=f"Node '{B.node_id}' finished. Moving to synthesis with excellent early results.",current_node=B.node_id,next_node=V));return{P:A.dynamic_data,Q:V}
				if C>=M(N)-1:
					A.dynamic_data[i]=T;G=[]
					for H in N:
						if I in H:G.append(f"Subtask: {H.get(f,g)}\nResult: {H[I]}")
					Y=h.join(G)if G else'No results were produced for the subtasks.';await B.notification_service.broadcast_to_task(A.task_id,L(task_id=A.task_id,status=O,detail=f"Node '{B.node_id}' finished. All subtasks processed.",current_node=B.node_id,next_node=V));return{P:A.dynamic_data,Z:Y,Q:V}
				A.dynamic_data[S]=C+1;A.final_answer=K;A.thoughts=[];A.current_thoughts_to_evaluate=[];A.current_best_thought_id=K;A.search_depth=0;await B.notification_service.broadcast_to_task(A.task_id,L(task_id=A.task_id,status=O,detail=f"Node '{B.node_id}' finished. Moving to next subtask.",current_node=B.node_id,next_node=j));return{P:A.dynamic_data,Z:K,'thoughts':[],'current_thoughts_to_evaluate':[],'current_best_thought_id':K,'search_depth':0,Q:j}
			await B.notification_service.broadcast_to_task(A.task_id,L(task_id=A.task_id,status=O,detail=f"Node '{B.node_id}' finished. Missing subtask index data.",current_node=B.node_id,next_node=U));return{P:A.dynamic_data,Q:U}
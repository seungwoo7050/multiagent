g='evaluated'
e=dict
Z=isinstance
C=None
from typing import Any,Dict,List,Optional
from langchain_core.runnables import RunnableConfig
from src.utils.logger import get_logger as A
from src.config.settings import get_settings as B
from src.schemas.mcp_models import AgentGraphState,Thought
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage as d,IntermediateResultMessage as t
from opentelemetry import trace
u=trace.get_tracer(__name__)
F=A(__name__)
D=B()
class E:
	def __init__(A,notification_service,beam_width=2,score_threshold_to_finish=.7,min_score_to_continue=.1,node_id='search_strategy',min_depth_before_finish=1):
		C=beam_width;B=notification_service
		if C<1:raise ValueError('Beam width must be at least 1.')
		A.notification_service=B;A.beam_width=C;A.score_threshold_to_finish=score_threshold_to_finish;A.min_score_to_continue=min_score_to_continue;A.node_id=node_id;A.min_depth_before_finish=min_depth_before_finish;F.info(f"SearchStrategyNode '{A.node_id}' initialized. Beam width: {A.beam_width}, Finish Threshold: {A.score_threshold_to_finish}, Min Continue Score: {A.min_score_to_continue}, Min Depth: {A.min_depth_before_finish}. NotificationService injected: {"Yes"if B else"No"}")
	def _select_best_thought(D,thoughts):
		A=thoughts
		if not A:return
		B=[A for A in A if A.status==g and A.evaluation_score is not C]
		if not B:return
		return max(B,key=lambda t:t.evaluation_score or .0)
	async def __call__(B,state,config=C):
		s='dynamic_data';r='thoughts';q='finish_low_score';p='finish_high_score';o='finish_max_depth';n='finish_recursion_limit';m='prev_best_score';l='no_improvement_count';k='continue_search';f='continue';c='next_action';b='current_best_thought_id';Y='recursion_limit';W='final_answer';V='search_depth';Q='finish';P='error_message';O=True;A=state
		with u.start_as_current_span('graph.node.search_strategy',attributes={'node_id':B.node_id,'task_id':A.task_id,'beam_width':B.beam_width,V:A.search_depth}):
			F.info(f"SearchStrategyNode '{B.node_id}' execution started. Task ID: {A.task_id}, Depth: {A.search_depth}");await B.notification_service.broadcast_to_task(A.task_id,d(task_id=A.task_id,status='node_executing',detail=f"Node '{B.node_id}' (Search Strategy) started.",current_node=B.node_id));h=sorted([A for A in A.thoughts if A.status==g and A.evaluation_score is not C],key=lambda t:t.evaluation_score or-1.,reverse=O)
			if not h:
				F.warning(f"Node '{B.node_id}' (Task: {A.task_id}): No evaluated thoughts with scores found.");await B.notification_service.broadcast_to_task(A.task_id,d(task_id=A.task_id,status='node_error',detail=f"Node '{B.node_id}': No evaluated thoughts to process.",current_node=B.node_id));H='No evaluated thoughts were found to solve this problem.';D=B._select_best_thought(A.thoughts)
				if D:H=D.content
				elif A.original_input:H=f"Regarding '{A.original_input}', I couldn't develop a complete solution. Please provide more details or try a different approach."
				return{W:H,P:'No thoughts were evaluated successfully.',b:A.current_best_thought_id,c:Q}
			i=h[:B.beam_width];J=i[0];D=B._select_best_thought(A.thoughts);M=D.id if D else A.current_best_thought_id;S=A.get_thought_by_id(A.current_best_thought_id)if A.current_best_thought_id else C
			if S is C or D and D.evaluation_score is not C and(S.evaluation_score is C or D.evaluation_score>S.evaluation_score):M=D.id;F.info(f"Node '{B.node_id}' (Task: {A.task_id}): New global best thought ID: {M} (Score: {D.evaluation_score})")
			elif S:F.info(f"Node '{B.node_id}' (Task: {A.task_id}): Global best thought remains: {S.id} (Score: {S.evaluation_score})")
			E=False;R=A.search_depth+1;H=C;G=k;N='thought_generator';K=f;a=getattr(A,Y,15)
			if Z(A.dynamic_data,e)and Y in A.dynamic_data:a=A.dynamic_data[Y]
			elif Z(A.metadata,e)and Y in A.metadata:a=A.metadata[Y]
			if J.evaluation_score is not C and J.evaluation_score>=.85:F.warning(f"Node '{B.node_id}' (Task: {A.task_id}): Very high score ({J.evaluation_score}) detected. Forcing termination despite depth.");E=O;H=J.content;G='finish_very_high_score';N=C;K=Q
			X=0
			if not E and Z(A.dynamic_data,e):
				X=A.dynamic_data.get(l,0);v=A.dynamic_data.get(m,0)
				if D and D.evaluation_score is not C:
					if D.evaluation_score<=v:X+=1
					else:X=0;A.dynamic_data[m]=D.evaluation_score
				A.dynamic_data[l]=X
			if not E and X>=3 and A.search_depth>=2:F.warning(f"Node '{B.node_id}' (Task: {A.task_id}): No score improvement for 3 iterations. Forcing termination.");E=O;L=D or J;H=L.content if L else'Search stopped due to no progress.';G='finish_no_improvement';N=C;K=Q
			if not E and A.search_depth<B.min_depth_before_finish:F.info(f"Node '{B.node_id}' (Task: {A.task_id}): Continuing search as minimum depth {B.min_depth_before_finish} not yet reached.");G='continue_min_depth';K=f
			elif not E and R>=a:F.warning(f"Node '{B.node_id}' (Task: {A.task_id}): Recursion limit ({a}) nearly reached. Forcing termination.");E=O;L=D or J;H=L.content if L else'Search stopped due to recursion limit.';G=n;N=C;K=Q
			elif not E and R>=A.max_search_depth:F.info(f"Node '{B.node_id}' (Task: {A.task_id}): Max search depth ({A.max_search_depth}) reached.");E=O;L=D or J;H=L.content if L else'Reached max depth, no definitive answer.';G=o;N=C;K=Q
			elif not E and J.evaluation_score is not C and J.evaluation_score>=B.score_threshold_to_finish:F.info(f"Node '{B.node_id}' (Task: {A.task_id}): High-confidence thought found (Score: {J.evaluation_score}). Finalizing.");E=O;H=J.content;G=p;N=C;K=Q
			elif not E and D and D.evaluation_score is not C and D.evaluation_score>=B.score_threshold_to_finish*.85 and A.search_depth>=3:F.info(f"Node '{B.node_id}' (Task: {A.task_id}): Decent solution found after sufficient exploration (Score: {D.evaluation_score}). Finalizing.");E=O;H=D.content;G='finish_sufficient_score';N=C;K=Q
			elif not E and A.search_depth>=2 and all(A.evaluation_score is not C and A.evaluation_score<B.min_score_to_continue for A in i):F.warning(f"Node '{B.node_id}' (Task: {A.task_id}): All top {B.beam_width} thoughts below threshold. Stopping.");E=O;L=D or J;H=L.content;G=q;N=C;K=Q
			elif not E:F.info(f"Node '{B.node_id}' (Task: {A.task_id}): Proceeding to next search depth ({R}). Best thought to expand based on: {M}");G=k;K=f
			T={r:A.thoughts,b:M,c:K}
			if E:
				U=H
				if A.original_input and Z(U,str):
					if not U.startswith('Based on')and not U.startswith('Regarding'):U=f"Based on your request: '{A.original_input}', here is my solution:\n\n{U}"
				T[V]=A.search_depth if G==p else R;T[P]=C;T[W]=U;T[s]=A.dynamic_data.copy()if A.dynamic_data else{}
			else:T[V]=R;T[W]=C
			await B.notification_service.broadcast_to_task(A.task_id,t(task_id=A.task_id,node_id=B.node_id,result_step_name='search_strategy_decision',data={'decision':G,b:M,'current_best_score':A.get_thought_by_id(M).evaluation_score if M and A.get_thought_by_id(M)else C,'next_depth':R if not E else A.search_depth,'final_answer_preview':H[:100]+'...'if H else C,'is_terminal':E}));await B.notification_service.broadcast_to_task(A.task_id,d(task_id=A.task_id,status='node_completed',detail=f"Node '{B.node_id}' (Search Strategy) finished. Decision: {G}.",current_node=B.node_id,next_node=N));F.debug(f"[Search-Strategy] task_id={A.task_id} depth={A.search_depth} decision={G} next_action={K}");w=A.dynamic_data.copy()if A.dynamic_data else{};j=A.error_message;I={r:A.thoughts,b:M,c:K,P:C,s:w}
			if E:
				I[V]=A.search_depth;I[W]=H
				if G==n:I[P]='Search for subtask stopped due to recursion limit within ToT.'
				elif G==o:I[P]='Reached max search depth in ToT for subtask.'
				elif G==q:I[P]='Stopping ToT for subtask due to consistently low scores.'
			else:I[V]=R;I[W]=C
			if j and not I.get(P):I[P]=j
			F.info(f"Node '{B.node_id}' (Task: {A.task_id}): Returning update. Subtask ToT Final Answer Present: {I.get(W)is not C}, Next Action: {I.get(c)}, Search Depth: {I.get(V)}");return I
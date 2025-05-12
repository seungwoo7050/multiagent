D='엣지의 시작 노드 ID'
G=isinstance
C=ValueError
from typing import Any,Dict,List,Optional,Union,Literal
from pydantic import BaseModel as B,Field as A,model_validator as E
class F(B):id=A(...,description='노드의 고유 식별자 (LangGraph 노드 이름)');node_type=A(...,description="노드의 유형 (예: 'planner_node', 'executor_node', 'tool_node'). agents/graph_nodes/ 디렉토리의 구현과 매칭되어야 함.");parameters=A(default_factory=dict,description='노드 초기화 또는 실행에 필요한 파라미터')
class J(B):type='standard';source=A(...,description=D);target=A(...,description='엣지의 도착 노드 ID')
class H(B):context_key=A(...,description='StateGraph 상태에서 비교할 키');expected_value=A(...,description='키가 가져야 하는 예상 값');operator=A(default='==',description='비교 연산자')
class K(B):type='conditional';source=A(...,description=D);condition_key=A(...,description='상태(StateGraph)에서 조건을 판단할 키 (Orchestrator가 이 키를 보고 라우팅 결정)');targets=A(...,description="조건 값에 따른 타겟 노드 ID 매핑 (예: {'continue': 'executor_node', 'replan': 'planner_node', '__end__': '__end__'})");default_target=A(None,description='매핑되는 조건 값이 없을 경우 이동할 기본 타겟 노드 ID (없으면 에러 처리될 수 있음)')
class I(B):
	name=A(...,description='그래프(워크플로우)의 이름');description=A(None,description='그래프에 대한 설명');entry_point=A(...,description='그래프 실행 시작 노드 ID');nodes=A(...,description='그래프를 구성하는 노드 목록');edges=A(...,description='노드 간의 연결(엣지) 목록')
	@E(mode='after')
	def check_node_ids_exist(self):
		I='__end__';B=self;D={A.id for A in B.nodes}
		if B.entry_point not in D:raise C(f"Entry point '{B.entry_point}' does not match any defined node ID.")
		for(E,A)in enumerate(B.edges):
			if A.source not in D:raise C(f"Edge {E}: Source node ID '{A.source}' not found in defined nodes.")
			if G(A,J):
				if A.target!=I and A.target not in D:raise C(f"Edge {E}: Target node ID '{A.target}' not found in defined nodes.")
			elif G(A,K):
				H=list(A.targets.values())
				if A.default_target:H.append(A.default_target)
				for F in H:
					if F!=I and F not in D:raise C(f"Edge {E}: Conditional target node ID '{F}' not found in defined nodes.")
		return B
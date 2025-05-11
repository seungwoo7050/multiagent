# src/schemas/agent_graph_config.py
"""
동적 에이전트 그래프(LangGraph StateGraph) 설정을 위한 Pydantic 스키마 정의.
이 스키마는 config/agent_graphs/ 디렉토리의 JSON 파일 구조를 정의합니다.
"""
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator

class NodeConfig(BaseModel):
    """에이전트 그래프 내 노드 설정"""
    id: str = Field(..., description="노드의 고유 식별자 (LangGraph 노드 이름)")
    node_type: str = Field(..., description="노드의 유형 (예: 'planner_node', 'executor_node', 'tool_node'). agents/graph_nodes/ 디렉토리의 구현과 매칭되어야 함.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="노드 초기화 또는 실행에 필요한 파라미터")
    # 예시: "parameters": {"llm_model": "gpt-4o", "max_iterations": 3}

class EdgeConfig(BaseModel):
    """그래프 내 일반적인 엣지(연결) 설정"""
    type: Literal["standard"] = "standard"
    source: str = Field(..., description="엣지의 시작 노드 ID")
    target: str = Field(..., description="엣지의 도착 노드 ID")

class ConditionalEdgeCondition(BaseModel):
    """조건부 엣지의 조건을 정의하는 방식 (예시)"""
    # 실제 조건 로직은 구현에 따라 달라질 수 있음
    # 예1: 특정 키 값 비교
    context_key: str = Field(..., description="StateGraph 상태에서 비교할 키")
    expected_value: Any = Field(..., description="키가 가져야 하는 예상 값")
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not in"] = Field(default="==", description="비교 연산자")
    # 예2: 특정 함수 호출 (Orchestrator에서 해석 필요)
    # function_name: str = Field(..., description="호출할 조건 함수 이름")
    # function_args: List[str] = Field(default_factory=list, description="조건 함수에 전달할 StateGraph 상태 키 목록")

class ConditionalEdgeConfig(BaseModel):
    """그래프 내 조건부 엣지 설정"""
    type: Literal["conditional"] = "conditional"
    source: str = Field(..., description="엣지의 시작 노드 ID")
    condition_key: str = Field(..., description="상태(StateGraph)에서 조건을 판단할 키 (Orchestrator가 이 키를 보고 라우팅 결정)")
    # condition: Union[str, ConditionalEdgeCondition] = Field(..., description="조건 정의 (간단한 문자열 또는 상세 조건 객체)") # 복잡도를 위해 condition_key 사용
    targets: Dict[str, str] = Field(..., description="조건 값에 따른 타겟 노드 ID 매핑 (예: {'continue': 'executor_node', 'replan': 'planner_node', '__end__': '__end__'})")
    default_target: Optional[str] = Field(None, description="매핑되는 조건 값이 없을 경우 이동할 기본 타겟 노드 ID (없으면 에러 처리될 수 있음)")

class AgentGraphConfig(BaseModel):
    """동적 에이전트 그래프 전체 설정 스키마"""
    name: str = Field(..., description="그래프(워크플로우)의 이름")
    description: Optional[str] = Field(None, description="그래프에 대한 설명")
    entry_point: str = Field(..., description="그래프 실행 시작 노드 ID")
    nodes: List[NodeConfig] = Field(..., description="그래프를 구성하는 노드 목록")
    edges: List[Union[EdgeConfig, ConditionalEdgeConfig]] = Field(..., description="노드 간의 연결(엣지) 목록")
    # state_schema: Optional[Dict[str, Any]] = Field(None, description="LangGraph StateGraph의 상태 스키마 정의 (선택 사항)")

    @model_validator(mode='after')
    def check_node_ids_exist(self) -> 'AgentGraphConfig':
        """엣지에서 참조하는 노드 ID가 실제로 노드 목록에 존재하는지 검증"""
        node_ids = {node.id for node in self.nodes}
        if self.entry_point not in node_ids:
            raise ValueError(f"Entry point '{self.entry_point}' does not match any defined node ID.")

        for i, edge in enumerate(self.edges):
            if edge.source not in node_ids:
                raise ValueError(f"Edge {i}: Source node ID '{edge.source}' not found in defined nodes.")

            if isinstance(edge, EdgeConfig):
                if edge.target != '__end__' and edge.target not in node_ids: # __end__는 LangGraph의 예약어
                    raise ValueError(f"Edge {i}: Target node ID '{edge.target}' not found in defined nodes.")
            elif isinstance(edge, ConditionalEdgeConfig):
                all_targets = list(edge.targets.values())
                if edge.default_target:
                    all_targets.append(edge.default_target)
                for target_id in all_targets:
                     if target_id != '__end__' and target_id not in node_ids:
                         raise ValueError(f"Edge {i}: Conditional target node ID '{target_id}' not found in defined nodes.")
        return self
    
"""
NodeConfig: 그래프의 각 단계를 나타냅니다. id는 고유 이름, node_type은 agents/graph_nodes/ 아래 구현될 파이썬 클래스 이름과 연결되며, parameters는 해당 노드 실행에 필요한 설정을 담습니다.
EdgeConfig: 두 노드 간의 기본적인 연결을 나타냅니다. source에서 target으로 제어가 이동합니다.
ConditionalEdgeConfig: source 노드의 실행 결과(StateGraph 상태의 특정 condition_key 값)에 따라 다음 노드를 결정합니다. targets 딕셔너리는 key값(조건 결과 문자열)과 value(다음 노드 ID)를 매핑합니다. __end__는 LangGraph에서 워크플로우 종료를 나타내는 특별한 값입니다. default_target은 매칭되는 조건 값이 없을 때 이동할 노드를 지정합니다.
AgentGraphConfig: 전체 그래프 설정을 담습니다. 그래프 이름, 설명, 시작점(entry_point), 노드 목록, 엣지 목록을 포함합니다.
model_validator: Pydantic v2의 검증 기능으로, 엣지에 정의된 source 및 target 노드 ID가 실제로 nodes 목록에 존재하는지 확인하여 설정 파일의 오류를 미리 방지합니다.
"""
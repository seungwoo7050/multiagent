from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator


class NodeConfig(BaseModel):
    """에이전트 그래프 내 노드 설정"""

    id: str = Field(..., description="노드의 고유 식별자 (LangGraph 노드 이름)")
    node_type: str = Field(
        ...,
        description="노드의 유형 (예: 'planner_node', 'executor_node', 'tool_node'). agents/graph_nodes/ 디렉토리의 구현과 매칭되어야 함.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="노드 초기화 또는 실행에 필요한 파라미터"
    )


class EdgeConfig(BaseModel):
    """그래프 내 일반적인 엣지(연결) 설정"""

    type: Literal["standard"] = "standard"
    source: str = Field(..., description="엣지의 시작 노드 ID")
    target: str = Field(..., description="엣지의 도착 노드 ID")


class ConditionalEdgeCondition(BaseModel):
    """조건부 엣지의 조건을 정의하는 방식 (예시)"""

    context_key: str = Field(..., description="StateGraph 상태에서 비교할 키")
    expected_value: Any = Field(..., description="키가 가져야 하는 예상 값")
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in", "not in"] = Field(
        default="==", description="비교 연산자"
    )


class ConditionalEdgeConfig(BaseModel):
    """그래프 내 조건부 엣지 설정"""

    type: Literal["conditional"] = "conditional"
    source: str = Field(..., description="엣지의 시작 노드 ID")
    condition_key: str = Field(
        ...,
        description="상태(StateGraph)에서 조건을 판단할 키 (Orchestrator가 이 키를 보고 라우팅 결정)",
    )

    targets: Dict[str, str] = Field(
        ...,
        description="조건 값에 따른 타겟 노드 ID 매핑 (예: {'continue': 'executor_node', 'replan': 'planner_node', '__end__': '__end__'})",
    )
    default_target: Optional[str] = Field(
        None,
        description="매핑되는 조건 값이 없을 경우 이동할 기본 타겟 노드 ID (없으면 에러 처리될 수 있음)",
    )


class AgentGraphConfig(BaseModel):
    """동적 에이전트 그래프 전체 설정 스키마"""

    name: str = Field(..., description="그래프(워크플로우)의 이름")
    description: Optional[str] = Field(None, description="그래프에 대한 설명")
    entry_point: str = Field(..., description="그래프 실행 시작 노드 ID")
    nodes: List[NodeConfig] = Field(..., description="그래프를 구성하는 노드 목록")
    edges: List[Union[EdgeConfig, ConditionalEdgeConfig]] = Field(
        ..., description="노드 간의 연결(엣지) 목록"
    )

    @model_validator(mode="after")
    def check_node_ids_exist(self) -> "AgentGraphConfig":
        """엣지에서 참조하는 노드 ID가 실제로 노드 목록에 존재하는지 검증"""
        node_ids = {node.id for node in self.nodes}
        if self.entry_point not in node_ids:
            raise ValueError(
                f"Entry point '{self.entry_point}' does not match any defined node ID."
            )

        for i, edge in enumerate(self.edges):
            if edge.source not in node_ids:
                raise ValueError(
                    f"Edge {i}: Source node ID '{edge.source}' not found in defined nodes."
                )

            if isinstance(edge, EdgeConfig):
                if edge.target != "__end__" and edge.target not in node_ids:
                    raise ValueError(
                        f"Edge {i}: Target node ID '{edge.target}' not found in defined nodes."
                    )
            elif isinstance(edge, ConditionalEdgeConfig):
                all_targets = list(edge.targets.values())
                if edge.default_target:
                    all_targets.append(edge.default_target)
                for target_id in all_targets:
                    if target_id != "__end__" and target_id not in node_ids:
                        raise ValueError(
                            f"Edge {i}: Conditional target node ID '{target_id}' not found in defined nodes."
                        )
        return self

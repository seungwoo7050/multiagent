from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class RunWorkflowRequest(BaseModel):
    """
    `/run` 엔드포인트에 대한 워크플로우 실행 요청 모델.
    실행할 그래프 설정 이름과 초기 입력을 받습니다.
    """

    graph_config_name: str = Field(
        ...,
        description="실행할 에이전트 그래프 설정 파일의 이름 (예: 'default_tot_workflow', 'react_tool_workflow'). '.json' 확장자는 제외합니다.",
        examples=["default_tot_workflow", "react_tool_workflow"],
    )
    original_input: Any = Field(
        ...,
        description="워크플로우의 주 입력 데이터입니다. 그래프의 첫 노드가 처리할 내용입니다 (예: 사용자 질문, 처리할 데이터 객체 등).",
    )
    task_id: Optional[str] = Field(
        None,
        description="선택적으로 지정할 작업 ID. 지정하지 않으면 시스템에서 생성됩니다.",
    )
    initial_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="워크플로우 시작 시 AgentGraphState의 metadata 필드에 전달될 초기 메타데이터입니다.",
    )


class ToolExecutionRequest(BaseModel):
    """'/tools/{tool_name}/execute' 엔드포인트에 대한 도구 실행 요청 모델 (기존 유지)"""

    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="도구 실행에 필요한 인자 (도구의 args_schema와 일치해야 함)",
    )

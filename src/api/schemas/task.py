from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class CreateTaskRequest(BaseModel):
    goal: str = Field(..., description='작업의 최종 목표 설명')
    task_type: Optional[str] = Field(None, description="실행할 에이전트 유형 (예: 'planner', 'executor'). 미지정 시 기본 로직 따름.")
    input_data: Dict[str, Any] = Field(default_factory=dict, description='작업 실행에 필요한 추가 입력 데이터')
    priority: int = Field(default=2, ge=1, le=4, description='작업 우선순위 (1=Low, 2=Normal, 3=High, 4=Critical)')
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description='작업에 첨부할 추가 메타데이터')

class CreateTaskResponse(BaseModel):
    task_id: str = Field(..., description='새로 생성된 작업의 고유 ID')
    status: str = Field(default='submitted', description='작업 제출 상태')
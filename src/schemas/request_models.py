"""
API 요청에 사용되는 Pydantic 모델 정의
"""
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator
from src.schemas.enums import TaskPriority

class CreateTaskRequest(BaseModel):
    """'/tasks' 엔드포인트에 대한 작업 생성 요청 모델"""
    goal: str = Field(..., description="작업의 최종 목표")
    task_type: Optional[str] = Field(None, description="실행할 작업/에이전트 유형 (예: 'planning', 'code_generation')")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="작업 실행에 필요한 입력 데이터")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="작업 우선순위 (LOW, NORMAL, HIGH, CRITICAL 또는 정수 1-4)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="추가 메타데이터")
    
    # priority 필드에 대한 validator 추가
    @field_validator('priority', mode='before')
    @classmethod
    def validate_priority(cls, v: Union[TaskPriority, int, str]) -> TaskPriority:
        """입력값을 TaskPriority Enum으로 변환하고 유효성을 검사합니다."""
        if isinstance(v, TaskPriority):
            return v
        if isinstance(v, str):
            # 대소문자 구분 없이 문자열 이름으로 Enum 멤버 찾기
            try:
                return TaskPriority[v.upper()]
            except KeyError:
                # 이름으로 못 찾으면 값(value)으로 시도 (Enum 값이 문자열일 경우)
                try:
                    return TaskPriority(v.upper())
                except ValueError:
                     raise ValueError(f"Invalid priority string: '{v}'. Must be one of {list(TaskPriority.__members__.keys())}")
        if isinstance(v, int):
            # 정수 값을 Enum 멤버로 매핑 (일반적인 관례: LOW=1, NORMAL=2, HIGH=3, CRITICAL=4)
            mapping = {
                1: TaskPriority.LOW,
                2: TaskPriority.NORMAL,
                3: TaskPriority.HIGH,
                4: TaskPriority.CRITICAL
            }
            if v in mapping:
                return mapping[v]
            else:
                 raise ValueError(f"Invalid integer priority: {v}. Must be 1 (LOW), 2 (NORMAL), 3 (HIGH), or 4 (CRITICAL).")
        # 허용되지 않는 타입인 경우 에러 발생
        raise TypeError(f"Invalid type for priority: {type(v)}. Expected TaskPriority, int, or str.")


class ToolExecutionRequest(BaseModel):
    """'/tools/{tool_name}/execute' 엔드포인트에 대한 도구 실행 요청 모델"""
    args: Dict[str, Any] = Field(default_factory=dict, description="도구 실행에 필요한 인자 (도구의 args_schema와 일치해야 함)")
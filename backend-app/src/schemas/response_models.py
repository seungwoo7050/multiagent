"""
API 응답에 사용되는 Pydantic 모델 정의
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field
# TaskState Enum 대신 문자열 상태를 사용하도록 변경 가능 (AgentGraphState 와 맞추기 위해)
# from src.schemas.enums import TaskState

# --- Task Related Responses ---

class TaskSubmittedResponse(BaseModel):
    """작업 제출 성공 시 응답 모델"""
    task_id: str = Field(..., description="새로 생성된 작업의 고유 ID")
    # status를 'accepted'로 고정하여 비동기 처리를 명확히 함
    status: Literal["accepted"] = Field(default="accepted", description="작업 접수 상태")

class WorkflowStatusResponse(BaseModel):
    """워크플로우 상태 조회 응답 모델 (`/tasks/{task_id}` 엔드포인트)"""
    task_id: str = Field(..., description="작업 ID")
    status: str = Field(
        ...,
        description="워크플로우의 현재 상태 (예: 'running', 'completed', 'failed', 'pending')",
        examples=["running", "completed", "failed"]
    )
    final_answer: Optional[str] = Field(
        None,
        description="워크플로우 최종 결과 (완료 시). AgentGraphState의 final_answer 필드에 해당합니다."
    )
    error_message: Optional[str] = Field(
        None,
        description="오류 발생 시 메시지. AgentGraphState의 error_message 필드에 해당합니다."
    )
    # AgentGraphState의 주요 필드를 추가하여 더 자세한 정보 제공 가능 (선택 사항)
    current_iteration: Optional[int] = Field(
        None,
        description="현재 반복 횟수 (AgentGraphState의 current_iteration)"
    )
    search_depth: Optional[int] = Field(
        None,
        description="현재 탐색 깊이 (ToT 경우, AgentGraphState의 search_depth)"
    )
    last_llm_output: Optional[str] = Field(
        None,
        description="마지막 LLM 출력 (디버깅용, AgentGraphState의 last_llm_output)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="워크플로우 관련 메타데이터 (AgentGraphState의 metadata)"
    )
    # dynamic_data는 너무 클 수 있으므로 기본적으로 제외하고, 필요시 API 파라미터로 요청하도록 설계 가능
    # dynamic_data: Optional[Dict[str, Any]] = Field(None, description="워크플로우의 동적 데이터")


# 기존 TaskStatusResponse는 WorkflowStatusResponse로 대체되므로 주석 처리 또는 삭제
# class TaskStatusResponse(BaseModel):
#     id: str = Field(..., description="작업 ID")
#     state: TaskState = Field(..., description="현재 작업 상태 (pending, running, completed, failed, canceled)")
#     result: Optional[Dict[str, Any]] = Field(None, description="작업 성공 시 결과 데이터")
#     error: Optional[Union[str, Dict[str, Any]]] = Field(None, description="작업 실패 시 에러 정보")

# --- Tool Related Responses (기존 유지 또는 개선) ---

class ToolSchemaProperty(BaseModel):
    """도구 인자 스키마의 속성(property) 정보 (기존 유지)"""
    title: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    format: Optional[str] = None
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None

class ToolSchema(BaseModel):
    """도구의 인자 스키마 구조 (기존 유지)"""
    title: str = Field(..., description="도구 인자 스키마 제목 (보통 도구 이름)")
    type: str = Field(default='object', description="스키마 타입 (일반적으로 'object')")
    properties: Dict[str, ToolSchemaProperty] = Field(default_factory=dict, description="인자 속성 정의")
    required: List[str] = Field(default_factory=list, description="필수 인자 목록")

class ToolInfo(BaseModel):
    """도구 목록 조회 시 사용되는 기본 정보 모델 (기존 유지)"""
    name: str = Field(..., description="도구 이름")
    description: str = Field(..., description="도구 설명")
    args_schema_summary: Optional[Dict[str, str]] = Field(None, description="인자 스키마 요약 (인자명: 타입)")

class ToolDetail(BaseModel):
    """도구 상세 정보 모델 (스키마 포함) (기존 유지)"""
    name: str = Field(..., description="도구 이름")
    description: str = Field(..., description="도구 설명")
    args_schema: ToolSchema = Field(..., description="도구의 상세 인자 스키마")

class ToolExecutionResponse(BaseModel):
    """도구 실행 결과 응답 모델 (기존 유지)"""
    status: str = Field(..., description="'success' 또는 'error'")
    tool_name: str = Field(..., description="실행된 도구 이름")
    execution_time: Optional[float] = Field(None, description="실행 시간(초)")
    result: Optional[Any] = Field(None, description="도구 실행 결과 (성공 시)")
    error: Optional[Dict[str, Any]] = Field(None, description="에러 상세 정보 (실패 시)")


# --- Context Related Responses (기존 유지) ---

class ContextResponse(BaseModel):
    """Context 조회 응답 모델"""
    context_id: str = Field(..., description="Context ID")
    data: Dict[str, Any] = Field(..., description="Context 데이터")

class ContextOperationResponse(BaseModel):
    """Context 생성/수정 결과 응답 모델"""
    context_id: str = Field(..., description="Context ID")
    status: Literal["created", "updated"] = Field(..., description="작업 상태")
    message: str = Field(..., description="결과 메시지")


# --- Agent Related Responses (기존 유지) ---

class AgentInfo(BaseModel):
    """에이전트 목록 조회 시 사용되는 기본 정보 모델"""
    name: str = Field(..., description="에이전트 이름")
    agent_type: str = Field(..., description="에이전트 타입")
    description: Optional[str] = Field(None, description="에이전트 설명")
    version: str = Field(..., description="에이전트 버전")

class AgentDetailResponse(BaseModel):
    """에이전트 상세 설정 응답 모델"""
    name: str
    description: Optional[str] = None
    version: str
    agent_type: str
    model: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int
    timeout: float
    allowed_tools: List[str] = Field(default_factory=list)
    memory_keys: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    mcp_enabled: bool
    mcp_context_types: List[str] = Field(default_factory=list)


# --- System Related Responses (기존 유지) ---

class HealthCheckResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str = Field(default='ok', description="시스템 상태")

# --- (선택 사항) Graph/Tool List Responses ---
class GraphInfo(BaseModel):
    """사용 가능한 그래프 설정 정보"""
    name: str = Field(..., description="그래프 설정 파일 이름 (확장자 제외)")
    description: Optional[str] = Field(None, description="그래프 설명 (설정 파일 내)")
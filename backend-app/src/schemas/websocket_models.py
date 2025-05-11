from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field
import datetime

class WebSocketMessageBase(BaseModel):
    """모든 WebSocket 메시지의 기본 모델"""
    event_type: str = Field(..., description="메시지 이벤트 유형")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="메시지 생성 타임스탬프 (UTC 권장)")
    task_id: str = Field(..., description="관련 작업 ID")

class StatusUpdateMessage(WebSocketMessageBase):
    """워크플로우 상태 변경 알림 메시지"""
    event_type: Literal["status_update"] = "status_update"
    status: str = Field(..., description="새로운 작업 상태 (예: 'running', 'tool_called', 'completed', 'failed')")
    detail: Optional[str] = Field(None, description="상태에 대한 추가 설명")
                                        
    current_node: Optional[str] = Field(None, description="현재 실행 중이거나 완료된 노드 ID")
    next_node: Optional[str] = Field(None, description="다음에 실행될 것으로 예상되는 노드 ID")

class IntermediateResultMessage(WebSocketMessageBase):
    """중간 결과 알림 메시지"""
    event_type: Literal["intermediate_result"] = "intermediate_result"
    node_id: str = Field(..., description="결과를 생성한 노드 ID")
    result_step_name: str = Field(..., description="중간 결과 단계 이름 또는 설명")
    data: Dict[str, Any] = Field(..., description="중간 결과 데이터")
                                        
                                            

class FinalResultMessage(WebSocketMessageBase):
    """최종 결과 알림 메시지"""
    event_type: Literal["final_result"] = "final_result"
    final_answer: Optional[str] = Field(None, description="워크플로우의 최종 답변")
    error_message: Optional[str] = Field(None, description="오류 발생 시 최종 오류 메시지")
                                       
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="최종 결과 관련 추가 메타데이터")

class ErrorMessage(WebSocketMessageBase):
    """WebSocket 통신 또는 처리 중 오류 알림 메시지"""
    event_type: Literal["error"] = "error"
    error_code: str = Field(..., description="오류 코드 (내부 정의)")
    message: str = Field(..., description="오류 메시지")
    details: Optional[Dict[str, Any]] = Field(None, description="오류 상세 정보")
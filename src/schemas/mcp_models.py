# src/schemas/mcp_models.py
"""
Model Context Protocol (MCP)을 따르는 구체적인 데이터 모델 정의.
성능이 중요한 경우 msgspec.Struct를 사용하여 정의합니다.
"""
import msgspec
import time
from typing import Any, Dict, List, Optional, Union

# --- LLM 관련 컨텍스트 모델 (msgspec 사용 예시) ---

class LLMInputMessage(msgspec.Struct, forbid_unknown_fields=True):
    """LLM 입력 메시지 구조체 (msgspec)"""
    role: str  # 예: "user", "assistant", "system"
    content: Union[str, List[Dict[str, Any]]] # 텍스트 또는 멀티모달 콘텐츠

class LLMParameters(msgspec.Struct, omit_defaults=True, forbid_unknown_fields=True):
    """LLM 호출 파라미터 구조체 (msgspec)"""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    # 추가적인 LLM 특정 파라미터 필드를 여기에 추가할 수 있습니다.
    # 예: presence_penalty: Optional[float] = None

# BaseContextSchema의 필드를 포함하도록 msgspec.Struct 정의
# 참고: msgspec은 직접 상속보다 합성을 권장하는 경우가 많으나,
#       여기서는 BaseContextSchema의 필드를 명시적으로 다시 정의합니다.
class LLMInputContext(msgspec.Struct, tag='llm_input', omit_defaults=True, forbid_unknown_fields=True):
    """LLM 입력을 위한 MCP 컨텍스트 (msgspec)"""
    model: str # 대상 LLM 모델 이름
    context_id: str = msgspec.field(default_factory=lambda: generate_uuid()) # 기본값 생성기 사용
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)
    version: str = '1.0.0'

    prompt: Optional[str] = None # 간단한 텍스트 프롬프트 (messages와 상호 배타적일 수 있음)
    messages: Optional[List[LLMInputMessage]] = None # 채팅 기반 모델용 메시지 리스트
    parameters: Optional[LLMParameters] = None # LLM 호출 파라미터
    use_cache: bool = True # 캐시 사용 여부
    retry_on_failure: bool = True # 실패 시 재시도 여부

class LLMOutputChoice(msgspec.Struct, forbid_unknown_fields=True):
    """LLM 응답 선택지 구조체 (msgspec)"""
    text: Optional[str] = None # 생성된 텍스트
    index: int = 0
    finish_reason: Optional[str] = None # 예: "stop", "length", "content_filter"

class LLMUsage(msgspec.Struct, omit_defaults=True, forbid_unknown_fields=True):
    """LLM 토큰 사용량 구조체 (msgspec)"""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class LLMOutputContext(msgspec.Struct, tag='llm_output', omit_defaults=True, forbid_unknown_fields=True):
    """LLM 응답을 위한 MCP 컨텍스트 (msgspec)"""
    success: bool # LLM 호출 성공 여부
    context_id: str = msgspec.field(default_factory=lambda: generate_uuid())
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)
    version: str = '1.0.0'

    result_text: Optional[str] = None # 주 응답 텍스트 (choices[0].text 와 동일할 수 있음)
    choices: Optional[List[LLMOutputChoice]] = None # LLM 응답 선택지 리스트
    usage: Optional[LLMUsage] = None # 토큰 사용량 정보
    error_message: Optional[str] = None # 에러 발생 시 메시지
    model_used: Optional[str] = None # 실제로 사용된 모델 이름


# --- 에이전트 상태 또는 메모리 저장을 위한 모델 예시 (msgspec) ---

class ConversationTurn(msgspec.Struct, tag='conversation_turn', omit_defaults=True, forbid_unknown_fields=True):
    """대화 턴을 저장하기 위한 구조체"""
    role: str # "user", "assistant" 등
    content: str
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

# --- 필수 임포트 추가 ---
# generate_uuid와 time을 사용하기 위해 임포트 추가
from src.utils.ids import generate_uuid
import time
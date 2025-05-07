# src/schemas/mcp_models.py
"""
Model Context Protocol (MCP)을 따르는 구체적인 데이터 모델 정의.
성능이 중요한 경우 msgspec.Struct를 사용하여 정의합니다.
"""
import msgspec
import time
from typing import Any, Dict, List, Optional, Union

from src.utils.ids import generate_uuid


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

class ConversationTurn(msgspec.Struct, tag='conversation_turn', omit_defaults=True, forbid_unknown_fields=True):
    """대화 턴을 저장하기 위한 구조체"""
    role: str # "user", "assistant" 등
    content: str
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

class Thought(msgspec.Struct, forbid_unknown_fields=True):
    """ToT: 개별 생각 또는 추론 단계"""
    content: str # 생각의 내용
    id: str = msgspec.field(default_factory=generate_uuid)
    parent_id: Optional[str] = None # 부모 생각의 ID (트리 구조용)
    evaluation_score: Optional[float] = None # 평가 점수
    status: str = "generated" # 예: "generated", "evaluated", "pruned", "explored"
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict) # 추가 정보 (예: 사용된 프롬프트)

class AgentGraphState(msgspec.Struct, omit_defaults=True, forbid_unknown_fields=True):
    """
    LangGraph StateGraph에서 사용될 상태 객체.
    ToT 및 GenericLLMNode 워크플로우를 지원합니다.
    """
    task_id: str # 현재 처리 중인 작업의 ID
    original_input: Any # 워크플로우 시작 시의 원본 입력
    current_iteration: int = 0 # 현재 반복 횟수 (ToT의 깊이 또는 일반 루프용)
    
    # ToT 관련 필드
    thoughts: List[Thought] = msgspec.field(default_factory=list) # 생성된 모든 생각/노드
    current_thoughts_to_evaluate: List[str] = msgspec.field(default_factory=list) # 현재 평가 대기 중인 생각 ID 목록
    current_best_thought_id: Optional[str] = None # 현재까지 가장 좋은 생각의 ID
    search_depth: int = 0 # ToT: 현재 탐색 깊이
    max_search_depth: int = 5 # ToT: 최대 탐색 깊이
    
    # GenericLLMNode 또는 일반적인 LLM 호출 결과 저장용
    last_llm_input: Optional[Union[str, List[Dict[str, Any]]]] = None # 마지막 LLM 호출 입력
    last_llm_output: Optional[str] = None # 마지막 LLM 호출 결과
    
    # 워크플로우 제어 및 결과
    next_node_override: Optional[str] = None # 특정 다음 노드를 지정 (조건부 엣지 외)
    final_answer: Optional[str] = None # 최종 결과
    error_message: Optional[str] = None # 오류 발생 시 메시지
    
    # 추가적인 동적 데이터 (노드별 데이터, 중간 결과 등)
    # 이 필드는 매우 유연하게 사용될 수 있습니다.
    # 예를 들어, 'planner_output': {'plan': ...}, 'executor_intermediate_result': ...
    dynamic_data: Dict[str, Any] = msgspec.field(default_factory=dict)
    
    # MCP 컨텍스트 정보
    context_id: str = msgspec.field(default_factory=generate_uuid)
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict) # 요청 ID, 사용자 정보 등
    version: str = '1.0.0' # 상태 객체 버전

    def get_thought_by_id(self, thought_id: str) -> Optional[Thought]:
        """ID로 특정 생각을 찾습니다."""
        for thought in self.thoughts:
            if thought.id == thought_id:
                return thought
        return None

    def add_thought(self, content: str, parent_id: Optional[str] = None, metadata: Optional[Dict[str,Any]] = None) -> Thought:
        """새로운 생각을 추가합니다."""
        new_thought = Thought(content=content, parent_id=parent_id, metadata=metadata or {})
        self.thoughts.append(new_thought)
        return new_thought

    
"""
AgentGraphState 설명:
task_id: 현재 그래프가 처리 중인 작업의 ID입니다.
original_input: 워크플로우가 시작될 때 받은 초기 입력 데이터입니다.
current_iteration: 반복적인 프로세스(예: ToT의 생각 생성-평가 루프 또는 일반적인 에이전트 루프)에서 현재 몇 번째 반복인지를 나타냅니다.
ToT 관련 필드:
thoughts: Thought 객체의 리스트로, 생성된 모든 생각/추론 단계를 저장합니다. Thought 객체는 자체 ID, 부모 ID, 내용, 평가 점수, 상태 등을 가집니다.
current_thoughts_to_evaluate: 현재 평가해야 할 Thought들의 ID 리스트입니다.
current_best_thought_id: 평가를 통해 현재까지 가장 유망하다고 판단된 생각의 ID입니다.
search_depth, max_search_depth: ToT의 탐색 깊이를 제어합니다.
last_llm_input, last_llm_output: GenericLLMNode와 같이 일반적인 LLM 호출을 수행하는 노드가 자신의 입력과 출력을 기록하는 데 사용할 수 있습니다.
next_node_override: 특정 조건에 따라 다음에 실행될 노드를 명시적으로 지정하고 싶을 때 사용합니다 (LangGraph의 조건부 엣지 메커니즘과 함께 또는 별도로 사용 가능).
final_answer, error_message: 워크플로우의 최종 결과 또는 오류 상태를 저장합니다.
dynamic_data: 다양한 노드들이 중간 결과나 자신만의 상태 정보를 저장할 수 있는 유연한 딕셔너리입니다. 예를 들어, 플래너 노드는 여기에 생성된 계획을 저장할 수 있고, 실행기 노드는 실행 중간 결과를 저장할 수 있습니다.
MCP 컨텍스트 정보: context_id, timestamp, metadata, version은 MCP의 BaseContextSchema와 유사한 표준 필드입니다.
"""
import msgspec
import time
from typing import Any, Dict, List, Optional, Union

from src.utils.ids import generate_uuid

                                        

class LLMInputMessage(msgspec.Struct, forbid_unknown_fields=True):
    """LLM 입력 메시지 구조체 (msgspec)"""
    role: str                                    
    content: Union[str, List[Dict[str, Any]]]                  

class LLMParameters(msgspec.Struct, omit_defaults=True, forbid_unknown_fields=True):
    """LLM 호출 파라미터 구조체 (msgspec)"""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
                                          
class LLMInputContext(msgspec.Struct, tag='llm_input', omit_defaults=True, forbid_unknown_fields=True):
    """LLM 입력을 위한 MCP 컨텍스트 (msgspec)"""
    model: str               
    context_id: str = msgspec.field(default_factory=lambda: generate_uuid())             
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)
    version: str = '1.0.0'

    prompt: Optional[str] = None                                        
    messages: Optional[List[LLMInputMessage]] = None                    
    parameters: Optional[LLMParameters] = None              
    use_cache: bool = True           
    retry_on_failure: bool = True              

class LLMOutputChoice(msgspec.Struct, forbid_unknown_fields=True):
    """LLM 응답 선택지 구조체 (msgspec)"""
    text: Optional[str] = None          
    index: int = 0
    finish_reason: Optional[str] = None                                        

class LLMUsage(msgspec.Struct, omit_defaults=True, forbid_unknown_fields=True):
    """LLM 토큰 사용량 구조체 (msgspec)"""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class LLMOutputContext(msgspec.Struct, tag='llm_output', omit_defaults=True, forbid_unknown_fields=True):
    """LLM 응답을 위한 MCP 컨텍스트 (msgspec)"""
    success: bool               
    context_id: str = msgspec.field(default_factory=lambda: generate_uuid())
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)
    version: str = '1.0.0'

    result_text: Optional[str] = None                                        
    choices: Optional[List[LLMOutputChoice]] = None                 
    usage: Optional[LLMUsage] = None            
    error_message: Optional[str] = None              
    model_used: Optional[str] = None                

class ConversationTurn(msgspec.Struct, tag='conversation_turn', omit_defaults=True, forbid_unknown_fields=True):
    """대화 턴을 저장하기 위한 구조체"""
    role: str                        
    content: str
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None

class Thought(msgspec.Struct, forbid_unknown_fields=True):
    """ToT: 개별 생각 또는 추론 단계"""
    content: str         
    id: str = msgspec.field(default_factory=generate_uuid)
    parent_id: Optional[str] = None                     
    evaluation_score: Optional[float] = None        
    status: str = "generated"                                                    
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)                      

class AgentGraphState(msgspec.Struct, omit_defaults=True, forbid_unknown_fields=True):
    """
    LangGraph StateGraph에서 사용될 상태 객체.
    ToT 및 GenericLLMNode 워크플로우를 지원합니다.
    """
    task_id: str                  
    original_input: Any                    
    next_action: Optional[str] = None     
    current_iteration: int = 0                               
    
               
    thoughts: List[Thought] = msgspec.field(default_factory=list)               
    current_thoughts_to_evaluate: List[str] = msgspec.field(default_factory=list)                       
    current_best_thought_id: Optional[str] = None                    
    search_depth: int = 0                
    max_search_depth: int = 5                
    
                                     
    last_llm_input: Optional[Union[str, List[Dict[str, Any]]]] = None
    last_llm_output: Optional[str] = None
    scratchpad: str = ""                               
    tool_call_history: List[Dict] = msgspec.field(default_factory=list)              
    
                   
    next_node_override: Optional[str] = None                          
    final_answer: Optional[str] = None        
    error_message: Optional[str] = None              
    
                                    
                               
                                                                                 
    dynamic_data: Dict[str, Any] = msgspec.field(default_factory=dict)
    
                 
    context_id: str = msgspec.field(default_factory=generate_uuid)
    timestamp: float = msgspec.field(default_factory=time.time)
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)                  
    version: str = '1.0.0'           

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
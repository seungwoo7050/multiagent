# src/agents/graph_nodes/generic_llm_node.py

import os
import builtins as _bt # builtins 임포트 유지
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.config.errors import ToolError # ToolError는 유지
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager
from src.memory.memory_manager import MemoryManager
from src.schemas.mcp_models import AgentGraphState, ConversationTurn # ConversationTurn도 유지 (일반 히스토리용)
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

# _bt.os = os # 이 줄은 중복되거나 필요 없을 수 있습니다. os는 이미 직접 임포트됩니다.
logger = get_logger(__name__)
settings = get_settings()

# ParsedLLMResponse 타입 정의는 그대로 유지
class ParsedLLMResponse(TypedDict):
    action: str
    action_input: Union[Dict[str, Any], str]

class GenericLLMNode:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: ToolManager,
        memory_manager: MemoryManager,
        notification_service: NotificationService,
        prompt_template_path: str,
        output_field_name: Optional[str] = None,
        input_keys_for_prompt: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        node_id: str = "generic_llm_node",
        max_react_iterations: int = 5,
        enable_tool_use: bool = False,
        allowed_tools: Optional[List[str]] = None,
        history_prompt_key: Optional[str] = "conversation_history", # 일반 대화 기록용
        # --- [추가] 대화 요약 컨텍스트를 위한 설정 ---
        summary_prompt_key: Optional[str] = "conversation_summary", # 프롬프트 내 요약 변수명
        # --- [추가 끝] ---
        # history_key_prefix 와 max_history_items는 _load_conversation_history 에서 사용되므로 유지
        history_key_prefix: Optional[str] = "chat_history",
        max_history_items: Optional[int] = 10
    ):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager
        self.notification_service = notification_service
        self.prompt_template_path = prompt_template_path
        self.output_field_name = output_field_name
        self.input_keys_for_prompt = input_keys_for_prompt or []
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.node_id = node_id
        self.max_react_iterations = max_react_iterations
        self.enable_tool_use = enable_tool_use
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None

        self.history_prompt_key = history_prompt_key
        # --- [추가] 요약 키 저장 ---
        self.summary_prompt_key = summary_prompt_key
        # --- [추가 끝] ---
        self.history_key_prefix = history_key_prefix
        self.max_history_items = max_history_items

        self.prompt_template_str = self._load_prompt_template()
        prompt_vars_set = set(self.input_keys_for_prompt) # 중복 제거를 위해 set으로 시작

        if self.enable_tool_use:
            prompt_vars_set.update(['available_tools', 'tool_call_history', 'scratchpad'])
        if self.history_prompt_key:
            prompt_vars_set.add(self.history_prompt_key)
        # --- [추가] 요약 키도 프롬프트 변수에 추가 ---
        if self.summary_prompt_key:
            prompt_vars_set.add(self.summary_prompt_key)
        # --- [추가 끝] ---

        self.prompt_template = PromptTemplate(
            template=self.prompt_template_str,
            input_variables=list(prompt_vars_set) # 최종적으로 list로 변환
        )
        logger.info(
            f"GenericLLMNode '{self.node_id}' initialized. Tool use: {self.enable_tool_use}. "
            f"History key: '{self.history_prompt_key}', Summary key: '{self.summary_prompt_key}', Max hist items: {self.max_history_items}. Prompt: {prompt_template_path}. "
            f"NotificationService injected: {'Yes' if notification_service else 'No'}"
        )

    def _load_prompt_template(self) -> str:
        # ... (기존 _load_prompt_template 메서드 코드는 변경 없음) ...
        base_prompt_dir = getattr(settings, 'PROMPT_TEMPLATE_DIR', 'config/prompts')
        if os.path.isabs(self.prompt_template_path):
            full_path = self.prompt_template_path
        else:
            full_path = os.path.join(base_prompt_dir, self.prompt_template_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {full_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt template from {full_path}: {e}")
            raise

    def _get_available_tools_for_prompt(self) -> str:
        # ... (기존 _get_available_tools_for_prompt 메서드 코드는 변경 없음) ...
        if not self.enable_tool_use:
             return "Tool use is disabled for this node."

        available_tool_names_set = self.tool_manager.get_names() # Set으로 반환됨
        tools_to_list_names: List[str] = []

        if self.allowed_tools is not None: # self.allowed_tools도 set임
            allowed_and_available = available_tool_names_set.intersection(self.allowed_tools)
            if not allowed_and_available:
                 return "No tools available (allowed list is empty or no matches found)."
            tools_to_list_names = list(allowed_and_available)
        else:
            tools_to_list_names = list(available_tool_names_set)

        if not tools_to_list_names:
             return "No tools available."

        tool_summaries = self.tool_manager.get_tool_summaries_for_llm() # List[Dict[str,str]] 반환
        # 이름으로 필터링
        filtered_summaries = [s for s in tool_summaries if s.get('name') in tools_to_list_names]


        if not filtered_summaries:
             return "No details found for available tools."

        prompt_lines = ["Available Tools:"]
        for summary_dict in filtered_summaries:
             prompt_lines.append(f"- {summary_dict.get('name', 'Unknown Tool')}: {summary_dict.get('description', 'No description.')}")
        return "\n".join(prompt_lines)


    async def _load_conversation_history(self, state: AgentGraphState) -> str:
        # ... (기존 _load_conversation_history 메서드 코드는 변경 없음) ...
        # 이 메서드는 일반적인 대화 기록(ConversationTurn 객체 리스트)을 로드하고 포맷팅합니다.
        # 요약된 대화는 Orchestrator가 dynamic_data에 직접 넣어줄 것이므로, 이 메서드에서는 처리하지 않습니다.
        if not self.history_prompt_key or not self.history_key_prefix:
            return "No conversation history configured for retrieval." # 또는 "Conversation history is not enabled."

        try:
            context_id = state.task_id # 또는 conversation_id가 있다면 그것을 사용
            if hasattr(state, 'metadata') and isinstance(state.metadata, dict) and state.metadata.get('conversation_id'):
                context_id = state.metadata['conversation_id']
            
            if not context_id:
                logger.warning(f"Node '{self.node_id}': context_id (task_id or conversation_id) not found in state for history retrieval.")
                return "History retrieval skipped: context_id is missing."

            history_items: List[Union[ConversationTurn, Dict[str, Any]]] = await self.memory_manager.get_history(
                context_id=context_id,
                history_key_prefix=self.history_key_prefix,
                limit=self.max_history_items
            )

            if not history_items:
                return "No conversation history found." # 또는 "This is the start of the conversation."

            formatted_history = []
            for item in history_items: # item은 ConversationTurn 객체 또는 dict일 수 있음
                if isinstance(item, ConversationTurn): # msgspec.Struct는 isinstance로 확인
                    role = getattr(item, 'role', 'Unknown').capitalize()
                    content = getattr(item, 'content', '')
                    formatted_history.append(f"{role}: {content}")
                elif isinstance(item, dict):
                    role = item.get('role', 'Unknown').capitalize()
                    content = item.get('content', '')
                    formatted_history.append(f"{role}: {content}")
                else:
                    logger.warning(f"Node '{self.node_id}': Encountered unknown history item type: {type(item)}. Converting to string.")
                    formatted_history.append(str(item))

            return "\n".join(formatted_history)

        except Exception as e:
            logger.error(f"Node '{self.node_id}': Failed to load conversation history: {e}", exc_info=True)
            return f"Error loading conversation history: {e}" # 오류 메시지를 프롬프트에 포함시킬 수 있음

    # src/agents/graph_nodes/generic_llm_node.py에 추가할 수정 - _prepare_prompt_input 메서드

    async def _prepare_prompt_input(self, state: AgentGraphState) -> Dict[str, Any]:
        prompt_input: Dict[str, Any] = {}
        all_expected_vars = self.prompt_template.input_variables
        
        # 특수 매핑 - 일반적으로 사용되는 키들과 중첩된 데이터 구조 간의 매핑
        special_mappings = {
            # 결과 평가기용 특수 매핑
            "subtask_answer": ["dynamic_data.current_subtask.final_answer"],
            "subtask_description": ["dynamic_data.current_subtask.description"],
            "score_threshold": [] # 파라미터에서 직접 로드
        }

        for key in all_expected_vars:
            value: Any = None

            # 1. node parameters에서 값 로드 시도 (score_threshold 등)
            if key in self.__dict__:
                value = self.__dict__[key]
                logger.debug(f"Node '{self.node_id}': Using parameter value for '{key}'")
            
            # 2. state 객체의 직접적인 속성 확인
            elif hasattr(state, key):
                value = getattr(state, key)
            
            # 3. 특수 매핑 확인 - 중첩된 경로 우선 시도
            elif key in special_mappings:
                for path in special_mappings[key]:
                    if value is not None:
                        break
                    
                    if '.' in path:
                        parts = path.split('.')
                        # dynamic_data로 시작하는 경우
                        if parts[0] == 'dynamic_data' and isinstance(state.dynamic_data, dict):
                            current = state.dynamic_data
                            found = True
                            
                            for p in parts[1:]:
                                if isinstance(current, dict) and p in current:
                                    current = current[p]
                                else:
                                    found = False
                                    break
                            
                            if found:
                                value = current
                                logger.debug(f"Node '{self.node_id}': Found '{key}' via special mapping path '{path}'")
            
            # 4. 점(.) 표기법으로 중첩된 키 지원 (기존 로직)
            elif '.' in key:
                parts = key.split('.')
                # dynamic_data.xxx.yyy...
                if parts[0] == 'dynamic_data' and isinstance(state.dynamic_data, dict):
                    current_val = state.dynamic_data
                    path_to_value = parts[1:]
                # metadata.xxx.yyy...
                elif parts[0] == 'metadata' and isinstance(state.metadata, dict):
                    current_val = state.metadata
                    path_to_value = parts[1:]
                # state.xxx.yyy... (AgentGraphState의 다른 중첩 가능 필드)
                elif hasattr(state, parts[0]):
                    current_val = getattr(state, parts[0])
                    path_to_value = parts[1:]
                else:
                    current_val = None
                    path_to_value = []

                for p_key in path_to_value:
                    if isinstance(current_val, dict) and p_key in current_val:
                        current_val = current_val[p_key]
                    elif hasattr(current_val, p_key): # 객체 속성 접근
                        current_val = getattr(current_val, p_key)
                    else:
                        logger.debug(f"Node '{self.node_id}': Path '{key}' could not be fully resolved. Part '{p_key}' not found in {type(current_val).__name__}.")
                        current_val = None
                        break
                value = current_val

            # 5. dynamic_data의 최상위 키 확인
            elif isinstance(state.dynamic_data, dict) and key in state.dynamic_data:
                value = state.dynamic_data[key]

            # 6. metadata의 최상위 키 확인
            elif isinstance(state.metadata, dict) and key in state.metadata:
                value = state.metadata[key]

            # 7. 특별 처리 키들 (히스토리, 요약, 도구 목록 등)
            # 일반 대화 기록
            if key == self.history_prompt_key and self.history_key_prefix: # history_prompt_key가 None이 아닐 때만
                value = await self._load_conversation_history(state)
            # --- 요약된 대화 내용 가져오기 ---
            elif key == self.summary_prompt_key and self.summary_prompt_key: # summary_prompt_key가 None이 아닐 때만
                if state.dynamic_data and isinstance(state.dynamic_data.get(self.summary_prompt_key), str):
                    value = state.dynamic_data[self.summary_prompt_key]
                    logger.debug(f"Node '{self.node_id}': Using '{self.summary_prompt_key}' from dynamic_data for prompt.")
                else:
                    # Orchestrator에서 주입하지 않았거나, dynamic_data에 없는 경우
                    value = "No conversation summary available for this turn." # 또는 빈 문자열
                    logger.debug(f"Node '{self.node_id}': '{self.summary_prompt_key}' not found in dynamic_data or not a string. Using default.")
            # 도구 사용 관련
            elif self.enable_tool_use:
                if key == 'available_tools':
                    value = self._get_available_tools_for_prompt()
                elif key == 'tool_call_history':
                    # AgentGraphState의 tool_call_history는 List[Dict] 여야 함
                    history_list = state.tool_call_history if state.tool_call_history is not None else []
                    # state.dynamic_data.get('tool_call_history', []) 대신 state.tool_call_history 직접 사용
                    value = "\n".join(
                        f"Tool: {c.get('tool_name', 'UnknownTool')}, Args: {json.dumps(c.get('args', {}))}, Result: {str(c.get('result', 'No result'))[:200]}" # 결과 길이 제한
                        for c in history_list if isinstance(c, dict) # 타입 체크 추가
                    ) if history_list else "No tool calls yet."
                elif key == 'scratchpad':
                    value = state.scratchpad if state.scratchpad is not None else "" # state.scratchpad 직접 사용

            # 8. 값이 없으면 기본값 (빈 문자열)으로 설정
            if value is None:
                # input_keys_for_prompt는 명시적으로 채워져야 하는 키들
                if key in self.input_keys_for_prompt or \
                (key == self.history_prompt_key and self.history_prompt_key) or \
                (key == self.summary_prompt_key and self.summary_prompt_key): # 요약 키도 로그 대상
                    logger.warning(f"Node '{self.node_id}': Key '{key}' required for prompt (from input_keys, history, or summary) was not found or resolved to None; using empty string.")
                prompt_input[key] = ""
            elif isinstance(value, (list, dict)) and key != 'messages': # 'messages'는 보통 특별 처리됨
                try:
                    # 복잡한 객체는 JSON 문자열로 변환하여 프롬프트에 주입
                    prompt_input[key] = json.dumps(value, indent=2, ensure_ascii=False, default=str)
                except TypeError:
                    logger.warning(f"Node '{self.node_id}': Could not JSON serialize value for key '{key}' (type: {type(value).__name__}). Using str().")
                    prompt_input[key] = str(value) # 최후의 수단
            else:
                prompt_input[key] = str(value) # 모든 값을 문자열로 변환 (LLM 프롬프트는 보통 문자열)

        logger.debug(f"Node '{self.node_id}': Prepared prompt input keys: {list(prompt_input.keys())}")
        return prompt_input

    def _parse_llm_response(self, response_str: str) -> Optional[ParsedLLMResponse]:
        # ... (기존 _parse_llm_response 메서드 코드는 변경 없음) ...
        response_str = response_str.strip()
        logger.debug(f"Node '{self.node_id}': Parsing LLM response: {response_str[:300]}...")

        try:
            # LangChain의 StructuredOutputParser 또는 PydanticOutputParser 사용 고려
            # 여기서는 기존의 JSON 또는 ReAct 스타일 텍스트 파싱 유지
            if response_str.startswith("```json") and response_str.endswith("```"):
                response_str = response_str[7:-3].strip()
            elif response_str.startswith("```") and response_str.endswith("```"): # ``` 만 있는 경우
                 response_str = response_str[3:-3].strip()

            parsed_json = json.loads(response_str) # 표준 json 모듈 사용
            if isinstance(parsed_json, dict) and 'action' in parsed_json and 'action_input' in parsed_json:
                 action_val = parsed_json['action']
                 action_input_val = parsed_json['action_input']
                 logger.info(f"Node '{self.node_id}': Successfully parsed JSON response. Action: {action_val}")
                 return ParsedLLMResponse(action=str(action_val), action_input=action_input_val) # action도 str로 캐스팅
            else:
                 logger.warning(f"Node '{self.node_id}': Parsed JSON lacks 'action' or 'action_input' keys. Content: {str(parsed_json)[:200]}")
                 return None

        except json.JSONDecodeError:
            logger.debug(f"Node '{self.node_id}': Response is not valid JSON. Attempting ReAct style text parsing.")
            # ReAct 스타일 파싱 (더 견고하게 수정 가능)
            action_match = re.search(r"Action:\s*([^\n]+)", response_str, re.IGNORECASE)
            action_input_match = re.search(r"Action Input:\s*([\s\S]+)", response_str, re.IGNORECASE | re.DOTALL) # 여러 줄 Action Input 처리

            if action_match:
                action_val = action_match.group(1).strip()
                action_input_str = action_input_match.group(1).strip() if action_input_match else "{}" # 기본값은 빈 JSON 객체 문자열

                action_input_parsed: Union[Dict, str]
                try:
                    # Action Input이 JSON 문자열일 수 있음
                    action_input_parsed = json.loads(action_input_str)
                    if not isinstance(action_input_parsed, dict): # JSON은 맞으나 dict가 아니면
                         action_input_parsed = {"input": action_input_str} # 단순 문자열로 처리
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시, 문자열 그대로 사용 (예: finish 액션의 최종 답변)
                    action_input_parsed = action_input_str

                logger.info(f"Node '{self.node_id}': Successfully parsed ReAct style text response (Action: {action_val}). Input type: {type(action_input_parsed).__name__}")
                return ParsedLLMResponse(action=action_val, action_input=action_input_parsed)

            logger.error(f"Node '{self.node_id}': Failed to parse LLM response using known formats. Response preview: {response_str[:500]}...")
            return None # 어떤 형식으로도 파싱 실패


    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        # ... (기존 __call__ 메서드의 ReAct 루프 이전 부분, 즉 enable_tool_use == False 인 경우는 변경 없음) ...
        # ... (단, _prepare_prompt_input 호출 시 await 추가된 것 확인) ...
        with tracer.start_as_current_span(
            "graph.node.generic_llm",
            attributes={
                "node_id": self.node_id,
                "task_id": state.task_id,
                "model": self.model_name or "default_from_llm_client",
                "enable_tool_use": self.enable_tool_use
            }
        ) as current_node_span: # Span 객체 저장
            logger.info(f"GenericLLMNode '{self.node_id}' execution started. Task ID: {state.task_id}. Tool use enabled: {self.enable_tool_use}")
            current_node_span.set_attribute("app.node.id", self.node_id) # 커스텀 속성 추가

            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' started.", current_node=self.node_id)
            )

            if not self.enable_tool_use:
                logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): Executing simple LLM call (tool use disabled).")
                final_update_dict: Dict[str, Any] = {} # 최종 반환될 딕셔너리
                error_for_ws: Optional[str] = None # 웹소켓 알림용 에러 메시지

                try:
                    prompt_input_values = await self._prepare_prompt_input(state)
                    formatted_prompt = self.prompt_template.format(**prompt_input_values)
                    current_node_span.set_attribute("app.llm.prompt_length", len(formatted_prompt)) # 프롬프트 길이 추적
                    logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): Formatted prompt (tools disabled):\n{formatted_prompt[:500]}...")

                    llm_params_for_call: Dict[str, Any] = {} # LLM 호출 파라미터
                    if self.temperature is not None: llm_params_for_call['temperature'] = self.temperature
                    if self.max_tokens is not None: llm_params_for_call['max_tokens'] = self.max_tokens

                    messages_for_llm = [{"role": "user", "content": formatted_prompt}]
                    llm_response_str = await self.llm_client.generate_response(
                        messages=messages_for_llm,
                        model_name=self.model_name, # llm_client가 내부적으로 기본 모델 사용
                        **llm_params_for_call
                    )
                    current_node_span.set_attribute("app.llm.response_length", len(llm_response_str))
                    logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): LLM raw response (tools disabled): {llm_response_str[:200]}...")

                    # 결과 저장을 위한 기본 필드
                    final_update_dict = {
                        "error_message": None,
                        "last_llm_input": formatted_prompt,
                        "last_llm_output": llm_response_str,
                        "dynamic_data": state.dynamic_data.copy() if state.dynamic_data else {} # dynamic_data는 항상 복사본으로 시작
                    }
                    
                    # subtask 관련 정보가 있다면 final_update_dict에 추가 (기존 로직과 유사하게)
                    if state.dynamic_data and "current_subtask" in state.dynamic_data and self.node_id == "initial_responder_subtask":
                        # 이 노드가 initial_responder_subtask일 경우, 결과 평가를 위해 subtask 정보 전달
                        final_update_dict["subtask_description"] = state.dynamic_data["current_subtask"].get("description", "")
                        final_update_dict["subtask_answer"] = llm_response_str # LLM 응답을 subtask_answer로 설정

                    # output_field_name에 따라 결과 저장 위치 결정
                    output_key = self.output_field_name or "final_answer" # 기본값은 final_answer

                    if '.' in output_key: # 예: "dynamic_data.some_field" 또는 "dynamic_data.current_subtask.final_answer"
                        parts = output_key.split('.')
                        parent_key = parts[0]
                        if parent_key == "dynamic_data":
                            # dynamic_data는 이미 final_update_dict에 복사되어 있음
                            current_dict_level = final_update_dict["dynamic_data"]
                            for part_idx, part_key in enumerate(parts[1:-1]): # 마지막 키 제외하고 경로 탐색
                                if part_key not in current_dict_level or not isinstance(current_dict_level[part_key], dict):
                                    current_dict_level[part_key] = {} # 경로 없으면 생성
                                current_dict_level = current_dict_level[part_key]
                            current_dict_level[parts[-1]] = llm_response_str # 마지막 키에 값 할당
                        else:
                            logger.warning(f"Node '{self.node_id}': Cannot set output to non-dynamic_data nested field '{output_key}'. Storing in 'final_answer'.")
                            final_update_dict["final_answer"] = llm_response_str
                    else: # 예: "final_answer" 또는 "some_other_top_level_key"
                        final_update_dict[output_key] = llm_response_str

                    # last_llm_input/output도 dynamic_data에 저장 (디버깅/추적용)
                    final_update_dict["dynamic_data"]["last_llm_input"] = formatted_prompt
                    final_update_dict["dynamic_data"]["last_llm_output"] = llm_response_str


                except Exception as e:
                    logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): Error during simple LLM call: {e}", exc_info=True)
                    error_for_ws = f"Error in node '{self.node_id}' (tools disabled): {e}"
                    current_node_span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_for_ws))
                    current_node_span.record_exception(e)
                    # dynamic_data를 보존하면서 에러 메시지 설정
                    final_update_dict = {
                        "error_message": error_for_ws,
                        "dynamic_data": state.dynamic_data.copy() if state.dynamic_data else {}
                    }

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id, status="node_completed",
                        detail=f"Node '{self.node_id}' (Simple LLM Call) finished. Error: {error_for_ws or 'None'}",
                        current_node=self.node_id,
                        next_node=None # 단순 호출은 보통 여기서 종료되거나, 그래프 엣지가 다음 노드 결정
                    )
                )
                return final_update_dict


            # --- ReAct Loop Logic (enable_tool_use == True 인 경우) ---
            logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): Executing ReAct loop (tool use enabled).")
            current_node_span.set_attribute("app.react.max_iterations", self.max_react_iterations)
            current_error_message: Optional[str] = None
            
            # 현재 AgentGraphState의 dynamic_data를 복사하여 이번 호출의 ReAct 루프 동안 사용할 로컬 복사본 생성
            # 이렇게 하면 루프 내에서 scratchpad, tool_call_history 등을 업데이트할 때 원본 state 객체를 직접 수정하지 않음
            current_dynamic_data_for_loop = state.dynamic_data.copy() if state.dynamic_data else {}
            current_dynamic_data_for_loop.setdefault('scratchpad', state.scratchpad or "") # state의 scratchpad 우선 사용
            current_dynamic_data_for_loop.setdefault('tool_call_history', list(state.tool_call_history or [])) # state의 tool_call_history 우선 사용

            for i in range(self.max_react_iterations):
                current_node_span.add_event(f"ReAct Iteration Start: {i+1}")
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): ReAct Iteration {i + 1}/{self.max_react_iterations}")
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(task_id=state.task_id, status="node_iterating", detail=f"Node '{self.node_id}' ReAct iteration {i+1}.", current_node=self.node_id)
                )

                # 프롬프트 준비를 위한 임시 상태 객체 생성
                # 이 객체는 현재 루프 반복에 대한 스냅샷이며, current_dynamic_data_for_loop를 사용
                temp_state_for_prompt_preparation = AgentGraphState(
                     task_id=state.task_id,
                     original_input=state.original_input,
                     current_iteration=i,
                     thoughts=state.thoughts, # 이전 상태의 thoughts는 그대로 전달 (ToT 연계 시)
                     current_thoughts_to_evaluate=state.current_thoughts_to_evaluate,
                     current_best_thought_id=state.current_best_thought_id,
                     search_depth=state.search_depth,
                     max_search_depth=state.max_search_depth,
                     dynamic_data=current_dynamic_data_for_loop, # 현재 루프의 dynamic_data 사용
                     metadata=state.metadata,
                     # 다음 필드들은 루프 내에서 업데이트되거나, 프롬프트에 직접 사용되지 않음
                     # last_llm_input, last_llm_output, final_answer, error_message 등
                     # AgentGraphState의 scratchpad와 tool_call_history 필드는 이제 dynamic_data 내부로 이동했으므로,
                     # temp_state_for_prompt_preparation 생성 시 이 필드들을 직접 설정할 필요는 없음.
                     # _prepare_prompt_input에서 dynamic_data['scratchpad'] 등을 참조하게 됨.
                )

                try:
                    prompt_input_values = await self._prepare_prompt_input(temp_state_for_prompt_preparation)
                    formatted_prompt = self.prompt_template.format(**prompt_input_values)
                    current_node_span.set_attribute(f"app.react.iteration.{i+1}.prompt_length", len(formatted_prompt))
                    logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}) Iteration {i+1}: Formatted prompt ready.")
                except Exception as prompt_err:
                     logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): Error preparing prompt in iteration {i+1}: {prompt_err}", exc_info=True)
                     current_error_message = f"Error preparing prompt in node '{self.node_id}': {prompt_err}"
                     current_node_span.record_exception(prompt_err)
                     break # 루프 중단

                try:
                    llm_params_for_call = {}
                    if self.temperature is not None: llm_params_for_call['temperature'] = self.temperature
                    llm_params_for_call['max_tokens'] = self.max_tokens if self.max_tokens else 1000 # ReAct는 응답이 길 수 있음

                    messages_for_llm = [{"role": "user", "content": formatted_prompt}]
                    llm_response_str = await self.llm_client.generate_response(
                        messages=messages_for_llm, model_name=self.model_name, **llm_params_for_call
                    )
                    current_node_span.set_attribute(f"app.react.iteration.{i+1}.response_length", len(llm_response_str))
                    logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}) Iteration {i+1}: LLM raw response received.")
                    # current_dynamic_data_for_loop 업데이트
                    current_dynamic_data_for_loop["last_llm_input"] = formatted_prompt
                    current_dynamic_data_for_loop["last_llm_output"] = llm_response_str
                except Exception as llm_err:
                    logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): LLM call failed in iteration {i+1}: {llm_err}", exc_info=True)
                    current_error_message = f"LLM call failed in node '{self.node_id}': {llm_err}"
                    current_node_span.record_exception(llm_err)
                    break

                parsed_response = self._parse_llm_response(llm_response_str)
                if parsed_response is None:
                    logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): Failed to parse LLM response in iteration {i+1}. Response: {llm_response_str[:500]}...")
                    current_error_message = f"Failed to parse LLM response in node '{self.node_id}'."
                    current_node_span.set_attribute(f"app.react.iteration.{i+1}.parse_error", True)
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        StatusUpdateMessage(task_id=state.task_id, status="node_error", detail=f"Node '{self.node_id}' failed to parse LLM response.", current_node=self.node_id)
                    )
                    break

                action = parsed_response['action']
                action_input = parsed_response['action_input']
                current_node_span.set_attribute(f"app.react.iteration.{i+1}.action", action)
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}) Iteration {i+1}: Parsed Action: {action}, Input type: {type(action_input).__name__}")

                observation = ""
                tool_call_entry: Optional[Dict[str, Any]] = None # 타입 명시

                normalized_action = action.lower().replace(" ", "_") # 공백 제거 및 소문자화

                if normalized_action == "final_answer" or normalized_action == "finish":
                    logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Received '{action}' action. Finishing.")
                    current_node_span.add_event("ReAct Action: Finish")
                    final_answer_content = action_input if isinstance(action_input, str) else json.dumps(action_input, default=str)
                    
                    # 최종 상태 업데이트 준비
                    final_update_dict = {
                        "error_message": None, # 성공적 종료
                        "dynamic_data": current_dynamic_data_for_loop # 최종 scratchpad, tool_call_history 등 포함
                    }
                    # output_field_name에 따라 final_answer 저장
                    output_key = self.output_field_name or "final_answer"
                    if '.' in output_key and output_key.startswith("dynamic_data."):
                        parts = output_key.split('.')
                        current_level = final_update_dict["dynamic_data"]
                        for p_key in parts[1:-1]:
                            current_level = current_level.setdefault(p_key, {})
                        current_level[parts[-1]] = final_answer_content
                    else:
                        final_update_dict[output_key] = final_answer_content
                    
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        StatusUpdateMessage(task_id=state.task_id, status="node_completed", detail=f"Node '{self.node_id}' ReAct loop finished successfully.", current_node=self.node_id)
                    )
                    return final_update_dict

                elif normalized_action == "think":
                    logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Received 'think' action.")
                    current_node_span.add_event("ReAct Action: Think")
                    thought_content = action_input if isinstance(action_input, str) else json.dumps(action_input, default=str)
                    observation = f"Thought processed: {thought_content}"
                    current_dynamic_data_for_loop['scratchpad'] += f"\nThought: {thought_content}"

                # 'tool_call' 액션 핸들러 (로드맵 Stage 4.5, 5에서 정의된 ReAct 패턴에 부합)
                elif normalized_action == "tool_call" and isinstance(action_input, dict):
                    tool_name_from_action = action_input.get("tool_name")
                    tool_args_from_action = action_input.get("tool_args", {}) # 기본값 빈 dict
                    current_node_span.add_event(f"ReAct Action: Tool Call ({tool_name_from_action})")

                    if not tool_name_from_action or not isinstance(tool_name_from_action, str):
                        observation = "Error: 'tool_call' action input missing 'tool_name' or it's not a string."
                        logger.error(f"Node '{self.node_id}': {observation} Input: {action_input}")
                        tool_call_entry = {"tool_name": "unknown_format", "args": action_input, "result": observation, "error": True}
                    elif not self.tool_manager.has(tool_name_from_action):
                        observation = f"Error: Tool '{tool_name_from_action}' does not exist."
                        logger.warning(f"Node '{self.node_id}': {observation}")
                        tool_call_entry = {"tool_name": tool_name_from_action, "args": tool_args_from_action, "result": observation, "error": True}
                    elif self.allowed_tools is not None and tool_name_from_action not in self.allowed_tools:
                        observation = f"Error: Tool '{tool_name_from_action}' is not allowed for this node."
                        logger.warning(f"Node '{self.node_id}': {observation}")
                        tool_call_entry = {"tool_name": tool_name_from_action, "args": tool_args_from_action, "result": observation, "error": True}
                    else:
                        logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Executing tool '{tool_name_from_action}' via 'tool_call' action.")
                        await self.notification_service.broadcast_to_task(
                            state.task_id,
                            IntermediateResultMessage(
                                task_id=state.task_id, node_id=self.node_id,
                                result_step_name="tool_calling",
                                data={"tool_name": tool_name_from_action, "tool_args": tool_args_from_action}
                            )
                        )
                        tool_result_str = ""
                        tool_error = False
                        try:
                            tool_instance = self.tool_manager.get_tool(tool_name_from_action)
                            tool_result = await tool_instance.ainvoke(tool_args_from_action) # tool_args_from_action 사용
                            tool_result_str = str(tool_result)
                            observation = f"Tool {tool_name_from_action} execution successful."
                            logger.debug(f"Node '{self.node_id}': Tool '{tool_name_from_action}' Result: {tool_result_str[:200]}...")
                            current_node_span.add_event(f"Tool Executed: {tool_name_from_action}", attributes={"tool.result_length": len(tool_result_str)})
                        except ToolError as tool_err:
                            observation = f"Error executing tool '{tool_name_from_action}': {tool_err.message}"
                            logger.error(f"Node '{self.node_id}': ToolError during execution of '{tool_name_from_action}': {tool_err.message}", exc_info=False)
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(tool_err, attributes={"tool.name": tool_name_from_action})
                        except Exception as tool_run_e:
                            observation = f"Unexpected error running tool '{tool_name_from_action}': {str(tool_run_e)}"
                            logger.exception(f"Node '{self.node_id}': Unexpected error during execution of tool '{tool_name_from_action}'")
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(tool_run_e, attributes={"tool.name": tool_name_from_action})
                        
                        tool_call_entry = {
                            "tool_name": tool_name_from_action,
                            "args": tool_args_from_action,
                            "result": tool_result_str,
                            "error": tool_error
                        }
                        observation = f"Observation: {tool_result_str}" # Observation은 도구 결과 자체 또는 그 요약

                # 이전 ReAct 스타일 (action이 직접 도구 이름인 경우)
                elif self.tool_manager.has(action): # action 자체가 도구 이름인 경우
                    current_node_span.add_event(f"ReAct Action: Tool Call (Direct - {action})")
                    if self.allowed_tools is not None and action not in self.allowed_tools:
                        observation = f"Error: Tool '{action}' is not allowed for this node."
                        logger.warning(f"Node '{self.node_id}': {observation}")
                        tool_call_entry = {"tool_name": action, "args": action_input, "result": observation, "error": True}
                    else:
                        logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Executing tool '{action}' (direct action).")
                        await self.notification_service.broadcast_to_task(
                            state.task_id,
                            IntermediateResultMessage(
                                task_id=state.task_id, node_id=self.node_id,
                                result_step_name="tool_calling",
                                data={"tool_name": action, "tool_args": action_input if isinstance(action_input, dict) else {"input": action_input}}
                            )
                        )
                        tool_args_direct = action_input if isinstance(action_input, dict) else {} # 도구 인자가 dict가 아니면 빈 dict로
                        if isinstance(action_input, str) and not tool_args_direct : # 문자열이고 dict가 아니면 input 키로 래핑 시도
                            tool_args_direct = {"input": action_input}


                        tool_result_str = ""
                        tool_error = False
                        try:
                            tool_instance = self.tool_manager.get_tool(action)
                            tool_result = await tool_instance.ainvoke(tool_args_direct)
                            tool_result_str = str(tool_result)
                            observation = f"Tool {action} execution successful."
                            logger.debug(f"Node '{self.node_id}': Tool '{action}' Result: {tool_result_str[:200]}...")
                            current_node_span.add_event(f"Tool Executed: {action}", attributes={"tool.result_length": len(tool_result_str)})
                        except ToolError as tool_err:
                            observation = f"Error executing tool '{action}': {tool_err.message}"
                            logger.error(f"Node '{self.node_id}': ToolError during execution of '{action}': {tool_err.message}", exc_info=False)
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(tool_err, attributes={"tool.name": action})
                        except Exception as tool_run_e:
                            observation = f"Unexpected error running tool '{action}': {str(tool_run_e)}"
                            logger.exception(f"Node '{self.node_id}': Unexpected error during execution of tool '{action}'")
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(tool_run_e, attributes={"tool.name": action})

                        tool_call_entry = {
                             "tool_name": action,
                             "args": tool_args_direct,
                             "result": tool_result_str,
                             "error": tool_error
                        }
                        observation = f"Observation: {tool_result_str}"
                else: # 알 수 없는 액션
                    observation = f"Error: Unknown action '{action}'. LLM response was: {llm_response_str[:200]}"
                    logger.error(f"Node '{self.node_id}': {observation}")
                    current_node_span.add_event(f"ReAct Action: Unknown ({action})")
                    tool_call_entry = {"tool_name": action, "args": action_input, "result": observation, "error": True} # tool_call_history에 기록

                # 스크래치패드 및 도구 호출 기록 업데이트 (current_dynamic_data_for_loop 사용)
                current_dynamic_data_for_loop['scratchpad'] += f"\n{observation}"
                if tool_call_entry:
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        IntermediateResultMessage(
                            task_id=state.task_id, node_id=self.node_id,
                            result_step_name="tool_result", data=tool_call_entry
                        )
                    )
                    current_dynamic_data_for_loop['tool_call_history'].append(tool_call_entry)

            # --- 루프 종료 (최대 반복 도달 또는 오류 발생) ---
            current_node_span.set_attribute("app.react.iterations_completed", i + 1) # i는 0부터 시작
            final_log_message = f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop "
            if current_error_message:
                final_log_message += f"finished due to error after {i+1} iterations: {current_error_message}"
                current_node_span.set_status(trace.Status(trace.StatusCode.ERROR, description=current_error_message))
            else: # 최대 반복 도달
                final_log_message += f"reached max iterations ({self.max_react_iterations})."
                current_node_span.set_status(trace.Status(trace.StatusCode.OK, description="Max iterations reached")) # OK로 하되, 설명 추가

            logger.warning(final_log_message)

            # 최종 반환될 상태 업데이트 딕셔너리 구성
            final_state_update_dict: Dict[str, Any] = {
                "error_message": current_error_message if current_error_message else f"Reached max ReAct iterations ({self.max_react_iterations}).",
                "dynamic_data": current_dynamic_data_for_loop # 루프 동안 업데이트된 dynamic_data 포함
            }

            # 최종 답변은 output_field_name에 따라 설정. 루프가 정상 종료되지 않았으므로,
            # scratchpad의 마지막 내용이나 에러 메시지를 사용.
            final_answer_content_on_loop_end = current_dynamic_data_for_loop.get('scratchpad', "No answer generated after max iterations.")
            if current_error_message:
                final_answer_content_on_loop_end = f"Processing stopped due to error: {current_error_message}. Last scratchpad: {final_answer_content_on_loop_end}"

            output_key_on_loop_end = self.output_field_name or "final_answer"
            if '.' in output_key_on_loop_end and output_key_on_loop_end.startswith("dynamic_data."):
                parts = output_key_on_loop_end.split('.')
                current_level = final_state_update_dict["dynamic_data"]
                for p_key in parts[1:-1]:
                    current_level = current_level.setdefault(p_key, {})
                current_level[parts[-1]] = final_answer_content_on_loop_end
            else:
                final_state_update_dict[output_key_on_loop_end] = final_answer_content_on_loop_end
            
            # final_answer 키가 명시적으로 설정되지 않았다면 (output_field_name이 다른 경우)
            # 그리고 루프가 오류나 반복 초과로 끝났다면, final_answer에도 요약된 정보를 넣어주는 것이 좋을 수 있음
            if "final_answer" not in final_state_update_dict and output_key_on_loop_end != "final_answer":
                 final_state_update_dict["final_answer"] = final_answer_content_on_loop_end


            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_completed", detail=f"Node '{self.node_id}' ReAct loop finished. Error: {current_error_message or 'Max iterations reached'}", current_node=self.node_id)
            )
            return final_state_update_dict
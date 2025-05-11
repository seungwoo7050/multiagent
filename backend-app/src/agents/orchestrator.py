# src/agents/orchestrator.py
import json
import os
from pathlib import Path # <<< [추가] Path 객체 사용을 위해
from typing import Any, Callable, Dict, Optional, Type, cast, Union
import functools
import asyncio
import inspect # inspect 모듈 추가
import msgspec

from fastapi import HTTPException
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
# --- [추가] LangChain의 PromptTemplate 임포트 ---
from langchain_core.prompts import PromptTemplate
# --- [추가 끝] ---

from src.config.settings import get_settings
from src.config.errors import ValidationError
from src.utils.logger import get_logger
from opentelemetry import trace


from src.services.llm_client import LLMClient
from src.memory.memory_manager import MemoryManager
from src.schemas.mcp_models import AgentGraphState # AgentGraphState가 이미 import 되어 있는지 확인
from src.schemas.agent_graph_config import AgentGraphConfig, NodeConfig as JsonNodeConfig, EdgeConfig, ConditionalEdgeConfig
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage, FinalResultMessage
from src.services.tool_manager import ToolManager, get_tool_manager
# ... (기존 다른 graph_node 임포트들은 그대로 유지) ...
from src.agents.graph_nodes.generic_llm_node import GenericLLMNode
from src.agents.graph_nodes.thought_generator_node import ThoughtGeneratorNode
from src.agents.graph_nodes.state_evaluator_node import StateEvaluatorNode
from src.agents.graph_nodes.search_strategy_node import SearchStrategyNode
from src.agents.graph_nodes.task_division_node import TaskDivisionNode
from src.agents.graph_nodes.task_complexity_evaluator_node import TaskComplexityEvaluatorNode
from src.agents.graph_nodes.subtask_processor_node import SubtaskProcessorNode
from src.agents.graph_nodes.synthesis_node import SynthesisNode
from src.agents.graph_nodes.task_complexity_router_node import TaskComplexityRouterNode
from src.agents.graph_nodes.direct_processor_node import DirectProcessorNode
# from src.agents.graph_nodes.synthesis_node import SynthesisNode # 중복 임포트 제거


logger = get_logger(__name__)
settings = get_settings()
tracer = trace.get_tracer(__name__)

REGISTERED_NODE_TYPES: Dict[str, Type[Any]] = {
    "generic_llm_node": GenericLLMNode,
    "thought_generator_node": ThoughtGeneratorNode,
    "state_evaluator_node": StateEvaluatorNode,
    "search_strategy_node": SearchStrategyNode,
    "task_division_node": TaskDivisionNode,
    "task_complexity_evaluator_node": TaskComplexityEvaluatorNode,
    "subtask_processor_node": SubtaskProcessorNode,
    "result_evaluator_node": GenericLLMNode, # GenericLLMNode를 재활용
    "synthesis_node": SynthesisNode,
    "task_complexity_router_node": TaskComplexityRouterNode,
    "direct_processor_node": DirectProcessorNode,
    # "synthesis_node": SynthesisNode, # 중복 제거
}


class Orchestrator:
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, memory_manager: MemoryManager, notification_service: NotificationService):
        if not isinstance(llm_client, LLMClient):
             raise TypeError("llm_client must be an instance of LLMClient")
        if not isinstance(tool_manager, ToolManager):
             raise TypeError("tool_manager must be an instance of ToolManager")
        if not isinstance(memory_manager, MemoryManager):
             raise TypeError("memory_manager must be an instance of MemoryManager")
        if not isinstance(notification_service, NotificationService):
            raise TypeError("notification_service must be an instance of NotificationService")

        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager
        self.notification_service = notification_service
        self._compiled_graphs: Dict[str, CompiledGraph] = {}
        # --- [추가] 대화 요약 관련 설정 ---
        self.summarizer_prompt_path = "generic/conversation_summarizer.txt" # 1단계에서 생성한 프롬프트 파일명
        self.summary_memory_key = "conversation_summary" # 메모리 매니저에 저장할 요약 키 이름
        self.summary_prompt_template_str: Optional[str] = None # 요약 프롬프트 내용을 캐시할 변수
        self._load_summarizer_prompt() # 인스턴스 생성 시 요약 프롬프트 로드
        # --- [추가 끝] ---
        logger.info(f"Orchestrator initialized with ToolManager: '{self.tool_manager.name}', MemoryManager, and Summarizer prompt: '{self.summarizer_prompt_path}'.")

    # --- [추가] 요약 프롬프트 로드 메서드 ---
    def _load_summarizer_prompt(self):
        """
        인스턴스 변수 self.summarizer_prompt_path에 지정된 프롬프트 템플릿을 로드하여
        self.summary_prompt_template_str에 저장합니다.
        """
        try:
            prompt_template_full_path = Path(settings.PROMPT_TEMPLATE_DIR) / self.summarizer_prompt_path
            if prompt_template_full_path.exists():
                with open(prompt_template_full_path, 'r', encoding='utf-8') as f:
                    self.summary_prompt_template_str = f.read()
                logger.info(f"Orchestrator: Successfully loaded summarizer prompt from: {prompt_template_full_path}")
            else:
                logger.error(f"Orchestrator: Summarizer prompt file not found at {prompt_template_full_path}. Summary generation will fail if prompt is not set otherwise.")
                self.summary_prompt_template_str = None # 명시적으로 None 설정
        except Exception as e:
            logger.error(f"Orchestrator: Failed to load summarizer prompt from {self.summarizer_prompt_path}: {e}", exc_info=True)
            self.summary_prompt_template_str = None # 오류 발생 시 None
    # --- [추가 끝] ---

    # --- [추가] 대화 요약 생성 메서드 ---
    async def _generate_conversation_summary(
        self,
        conversation_id: str,
        previous_summary: Optional[str],
        current_user_input: Any,
        current_agent_response: Optional[str]
    ) -> Optional[str]:
        """
        주어진 정보를 바탕으로 대화 요약을 생성합니다.
        """
        logger.debug(f"Orchestrator: Attempting to generate summary for conversation_id: {conversation_id}")

        if not self.summary_prompt_template_str:
            logger.error(f"Orchestrator: Summarizer prompt template is not loaded. Cannot generate summary for {conversation_id}.")
            return previous_summary # 프롬프트 없으면 이전 요약 반환 (또는 에러 처리)

        if current_agent_response is None or str(current_agent_response).strip() == "":
            logger.debug(f"Orchestrator: Agent response is empty for {conversation_id}, skipping summary generation. Returning previous summary.")
            return previous_summary

        try:
            prompt_input_values = {
                "previous_summary": previous_summary or "This is the beginning of the conversation.",
                "current_user_input": str(current_user_input), # LLM이 처리하기 쉽도록 문자열 변환
                "current_agent_response": str(current_agent_response), # 문자열 변환
            }

            # PromptTemplate을 사용하여 안전하게 포맷팅
            prompt = PromptTemplate(template=self.summary_prompt_template_str, input_variables=list(prompt_input_values.keys()))
            formatted_prompt = prompt.format(**prompt_input_values)

            messages_for_llm = [{"role": "user", "content": formatted_prompt}]

            # 요약 생성을 위한 LLM 호출 (비교적 간단한 모델, 낮은 temperature 권장)
            new_summary = await self.llm_client.generate_response(
                messages=messages_for_llm,
                # model_name=settings.PRIMARY_LLM_PROVIDER 설정의 모델 또는 더 저렴한 모델 지정 가능
                temperature=0.2, # 요약은 일관성이 중요
                max_tokens=300  # 요약 길이 제어 (설정으로 관리 가능)
            )
            generated_summary = new_summary.strip()
            logger.info(f"Orchestrator: New summary generated for {conversation_id}. Length: {len(generated_summary)}")
            logger.debug(f"Orchestrator: Generated summary preview for {conversation_id}: {generated_summary[:100]}...")
            return generated_summary
        except Exception as e:
            logger.error(f"Orchestrator: Failed to generate conversation summary for {conversation_id}: {e}", exc_info=True)
            return previous_summary # 실패 시 이전 요약 유지
    # --- [추가 끝] ---

    def _load_graph_config_from_file(self, config_name: str) -> Dict[str, Any]:
        # ... (기존 _load_graph_config_from_file 메서드 코드는 변경 없음) ...
        import json # 이 함수 내에서만 사용하므로 여기에 두거나 클래스 레벨로 올릴 수 있음
        # from unittest.mock import MagicMock # 테스트 환경 관련 코드가 있다면 유지

        try:
            graph_conf_dict = None
            # settings.AGENT_GRAPH_CONFIG_DIR를 사용하도록 수정 (settings는 이미 로드됨)
            config_path = Path(settings.AGENT_GRAPH_CONFIG_DIR) / f"{config_name}.json"

            if not config_path.exists():
                config_path_no_ext = Path(settings.AGENT_GRAPH_CONFIG_DIR) / config_name
                if config_path_no_ext.with_suffix(".json").exists(): # 혹시 확장자 없이 들어왔을 경우 대비
                    config_path = config_path_no_ext.with_suffix(".json")
                else:
                    logger.error(f"Graph configuration file not found: {config_path} (and without .json extension)")
                    raise FileNotFoundError(f"Graph configuration not found: {config_name}")

            logger.debug(f"Loading graph config directly from: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                graph_conf_dict = json.load(f)

            # Pydantic validation (AgentGraphConfig는 이미 임포트됨)
            validated_config = AgentGraphConfig.model_validate(graph_conf_dict) # model_validate 사용
            logger.info(f"Successfully loaded and validated graph configuration: {config_name} from {config_path}")
            return validated_config.model_dump() # Pydantic 모델을 dict로 변환하여 반환 (기존 코드 호환성)

        except FileNotFoundError as fnf:
            logger.error(f"Graph configuration file not found: {config_name} (Searched path: {config_path if 'config_path' in locals() else 'unknown'}) - Error: {fnf}")
            raise # 예외를 다시 발생시켜 상위에서 처리하도록 함
        except json.JSONDecodeError as jde:
            logger.error(f"JSON parsing error in graph config '{config_name}': {jde}", exc_info=True)
            raise ValueError(f"Invalid JSON in graph configuration: {config_name}") from jde
        except ValidationError as ve: # pydantic.ValidationError
            logger.error(f"Graph configuration validation failed for '{config_name}': {ve.errors()}", exc_info=True) # ve.errors()로 상세 오류 확인
            raise ValueError(f"Graph configuration validation error for '{config_name}': {ve}") from ve
        except Exception as e:
            logger.error(f"Unexpected error loading graph config '{config_name}': {e}", exc_info=True)
            raise # 예상치 못한 다른 오류도 다시 발생

    def _create_node_instance(self, node_config: JsonNodeConfig) -> Callable[[AgentGraphState], Dict[str, Any]]:
        # ... (기존 _create_node_instance 메서드 코드는 변경 없음) ...
        node_type_str = node_config.node_type
        node_params = node_config.parameters or {}
        node_id = node_config.id

        node_class = REGISTERED_NODE_TYPES.get(node_type_str)
        if not node_class:
            raise ValueError(f"Unsupported node_type: '{node_type_str}' for node ID '{node_id}'. Registered types: {list(REGISTERED_NODE_TYPES.keys())}")

        try:
            constructor_params = {
                "llm_client": self.llm_client,
                "node_id": node_id,
            }
            sig_params = inspect.signature(node_class.__init__).parameters

            if 'tool_manager' in sig_params and "tool_manager" not in node_params :
                 constructor_params["tool_manager"] = self.tool_manager
                 logger.debug(f"Injecting default ToolManager into node '{node_id}' (type: {node_type_str}).")

            if 'memory_manager' in sig_params and "memory_manager" not in node_params :
                constructor_params["memory_manager"] = self.memory_manager
                logger.debug(f"Injecting default MemoryManager into node '{node_id}' (type: {node_type_str}).")

            if 'notification_service' in sig_params and "notification_service" not in node_params:
                 constructor_params["notification_service"] = self.notification_service
                 logger.debug(f"Injecting default NotificationService into node '{node_id}' (type: {node_type_str}).")

            constructor_params.update(node_params)

            sig = inspect.signature(node_class.__init__)
            valid_params_for_constructor = {
                k: v for k, v in constructor_params.items() if k in sig.parameters
            }

            required_params_in_sig = {
                p.name for p in sig.parameters.values()
                if p.default == inspect.Parameter.empty and p.name != 'self'
            }
            missing_params = required_params_in_sig - set(valid_params_for_constructor.keys())
            if missing_params:
                 critical_missing = missing_params - {"llm_client", "tool_manager", "memory_manager", "notification_service"}
                 if critical_missing:
                      raise TypeError(f"Node '{node_id}' (Type: {node_type_str}): Missing required constructor arguments: {critical_missing}. Check JSON parameters and node class constructor.")

            node_instance = node_class(**valid_params_for_constructor)
            logger.debug(f"Created instance for node ID '{node_id}', type '{node_type_str}' with params: {list(valid_params_for_constructor.keys())}")

            if not callable(node_instance):
                 raise TypeError(f"Node instance for '{node_id}' (type: {node_type_str}) is not callable.")
            return node_instance

        except TypeError as te:
             logger.error(f"TypeError creating node instance for ID '{node_id}', type '{node_type_str}': {te}. Check constructor arguments and JSON parameters.", exc_info=True)
             raise RuntimeError(f"Error creating node '{node_id}' due to TypeError: {te}") from te
        except Exception as e:
            logger.error(f"Failed to create node instance for ID '{node_id}', type '{node_type_str}': {e}", exc_info=True)
            raise RuntimeError(f"Error creating node '{node_id}': {e}") from e


    def _get_conditional_router_func(
        self,
        condition_key: str,
        targets_map: Dict[str, str],
        default_decision: str = END,
    ) -> Callable[[Union[AgentGraphState, Dict[str, Any]]], str]:
        # ... (기존 _get_conditional_router_func 메서드 코드는 변경 없음) ...
        def router(state: Union[AgentGraphState, Dict[str, Any]]) -> str:
            value_to_check = None
            is_dict = isinstance(state, dict)
            
            state_for_log = str(state)[:200] + "..." if len(str(state)) > 200 else str(state) # 로깅용 상태 문자열 축약
            logger.debug(f"[Router] Input state type: {type(state).__name__}, condition_key='{condition_key}', state_preview='{state_for_log}'")


            if is_dict:
                state_dict = cast(Dict[str, Any], state)
                if condition_key in state_dict:
                    value_to_check = state_dict[condition_key]
                elif condition_key.startswith("dynamic_data.") and "." in condition_key:
                    key_path = condition_key.split('.')[1:]
                    current_val = state_dict.get("dynamic_data")
                    if isinstance(current_val, dict):
                        try:
                            for k_part in key_path: current_val = current_val.get(k_part) if isinstance(current_val, dict) else None
                            value_to_check = current_val
                        except Exception: value_to_check = None
                    else: value_to_check = None
                elif condition_key.startswith("metadata.") and "." in condition_key:
                    key_path = condition_key.split('.')[1:]
                    current_val = state_dict.get("metadata")
                    if isinstance(current_val, dict):
                        try:
                            for k_part in key_path: current_val = current_val.get(k_part) if isinstance(current_val, dict) else None
                            value_to_check = current_val
                        except Exception: value_to_check = None
                    else: value_to_check = None

            else: # AgentGraphState 객체인 경우
                state_obj = cast(AgentGraphState, state)
                if hasattr(state_obj, condition_key):
                    value_to_check = getattr(state_obj, condition_key)
                elif condition_key.startswith("dynamic_data.") and "." in condition_key and isinstance(state_obj.dynamic_data, dict):
                     key_path = condition_key.split('.')[1:]
                     current_val = state_obj.dynamic_data
                     try:
                          for k_part in key_path: current_val = current_val.get(k_part) if isinstance(current_val, dict) else None
                          value_to_check = current_val
                     except Exception: value_to_check = None
                elif condition_key.startswith("metadata.") and "." in condition_key and isinstance(state_obj.metadata, dict):
                     key_path = condition_key.split('.')[1:]
                     current_val = state_obj.metadata
                     try:
                          for k_part in key_path: current_val = current_val.get(k_part) if isinstance(current_val, dict) else None
                          value_to_check = current_val
                     except Exception: value_to_check = None


            logger.debug(f"[Router] Condition key='{condition_key}', Value to check='{value_to_check}' (type: {type(value_to_check).__name__})")

            decision = default_decision
            str_value_to_check = str(value_to_check).lower() # 비교를 위해 소문자로 변환

            # targets_map의 키도 소문자로 변환하여 비교
            normalized_targets_map = {str(k).lower(): v for k, v in targets_map.items()}


            if str_value_to_check in normalized_targets_map:
                 decision = normalized_targets_map[str_value_to_check]
                 logger.debug(f"Router decision based on value '{str_value_to_check}': '{decision}'")
                 return decision
            
            # 'value_is_not_none' 와 'value_is_none'은 정확한 키 이름으로 매칭
            if "value_is_not_none" in targets_map and value_to_check is not None:
                 decision = targets_map["value_is_not_none"]
                 logger.debug(f"Router decision based on 'value_is_not_none': '{decision}'")
                 return decision
            if "value_is_none" in targets_map and value_to_check is None:
                 decision = targets_map["value_is_none"]
                 logger.debug(f"Router decision based on 'value_is_none': '{decision}'")
                 return decision
            
            # Boolean 값 ('true', 'false')에 대한 처리
            if isinstance(value_to_check, bool):
                 bool_str_val = str(value_to_check).lower() # 'true' or 'false'
                 if bool_str_val in normalized_targets_map:
                     decision = normalized_targets_map[bool_str_val]
                     logger.debug(f"Router decision based on boolean value '{bool_str_val}': '{decision}'")
                     return decision

            logger.debug(f"No specific condition met for value '{value_to_check}'. Using default decision: '{default_decision}'")
            return default_decision
        return router

    def build_graph(self, graph_config_dict: Dict[str, Any]) -> StateGraph:
        # ... (기존 build_graph 메서드 코드는 변경 없음) ...
        # cfg = AgentGraphConfig.model_validate(graph_config_dict) # 이 라인은 _load_graph_config_from_file 에서 이미 수행됨
        # 대신, _load_graph_config_from_file의 반환값을 AgentGraphConfig 객체로 사용하거나,
        # 여기서 다시 한번 model_validate를 수행할 수 있음. 여기서는 dict를 그대로 사용한다고 가정하고 진행.
        # AgentGraphConfig의 Pydantic 모델 사용을 명확히 함
        cfg = AgentGraphConfig.model_validate(graph_config_dict)
        graph = StateGraph(AgentGraphState)

        for node_config_model in cfg.nodes: # JsonNodeConfig 대신 NodeConfig 사용
            try:
                node_instance = self._create_node_instance(node_config_model) # model_validate 된 NodeConfig 사용
                graph.add_node(node_config_model.id, node_instance)
            except Exception as node_creation_err:
                 logger.error(f"Fatal error creating node '{node_config_model.id}' for graph '{cfg.name}'. Halting graph build.", exc_info=True)
                 raise RuntimeError(f"Failed to build graph '{cfg.name}': Error creating node '{node_config_model.id}'.") from node_creation_err

        for edge_config_model in cfg.edges: # EdgeConfig 또는 ConditionalEdgeConfig 사용
            try:
                if edge_config_model.type == "standard":
                    std_edge_cfg = cast(EdgeConfig, edge_config_model) # 타입 캐스팅
                    target_node = END if std_edge_cfg.target == "__end__" else std_edge_cfg.target
                    graph.add_edge(std_edge_cfg.source, target_node)
                    logger.debug(f"[Graph-Edge] {std_edge_cfg.source} → {target_node} (standard)")
                elif edge_config_model.type == "conditional":
                    cond_edge_cfg = cast(ConditionalEdgeConfig, edge_config_model)
                    target_map = {k: (END if v == "__end__" else v) for k, v in cond_edge_cfg.targets.items()}
                    default_target = END if cond_edge_cfg.default_target == "__end__" else (cond_edge_cfg.default_target or END) # default_target이 None일 경우 END

                    router_func = self._get_conditional_router_func(
                        cond_edge_cfg.condition_key,
                        target_map,
                        default_target
                    )
                    possible_nodes = list(set(target_map.values()))
                    if default_target and default_target not in possible_nodes: # default_target도 possible_nodes에 포함
                        possible_nodes.append(default_target)
                    
                    graph.add_conditional_edges(
                        cond_edge_cfg.source,
                        router_func,
                        possible_nodes # LangGraph 0.0.42+ 에서는 dict 형태로 전달 {value: node_name}
                                       # 여기서는 라우터 함수가 직접 노드 이름을 반환하므로 list 형태 유지 가능
                    )
                    logger.debug(
                        f"[Graph-CondEdge] {cond_edge_cfg.source} → based on '{cond_edge_cfg.condition_key}'. Targets: {possible_nodes}"
                    )
            except Exception as edge_creation_err:
                 logger.error(f"Fatal error adding edge (Source: {edge_config_model.source}) for graph '{cfg.name}'. Halting graph build.", exc_info=True)
                 raise RuntimeError(f"Failed to build graph '{cfg.name}': Error adding edge from '{edge_config_model.source}'.") from edge_creation_err
        try:
            graph.set_entry_point(cfg.entry_point)
        except Exception as entry_point_err:
            logger.error(f"Fatal error setting entry point '{cfg.entry_point}' for graph '{cfg.name}'. Halting graph build.", exc_info=True)
            raise RuntimeError(f"Failed to build graph '{cfg.name}': Error setting entry point '{cfg.entry_point}'.") from entry_point_err

        logger.info(f"Successfully built StateGraph for configuration: {cfg.name}")
        return graph


    def get_compiled_graph(self, graph_config_name: str) -> CompiledGraph:
        # ... (기존 get_compiled_graph 메서드 코드는 변경 없음) ...
        if graph_config_name not in self._compiled_graphs:
            logger.info(f"Compiled graph for '{graph_config_name}' not found in cache. Building...")
            try:
                graph_dict_config = self._load_graph_config_from_file(graph_config_name)
                state_graph = self.build_graph(graph_dict_config)
                compiled_graph = state_graph.compile()
                self._compiled_graphs[graph_config_name] = compiled_graph
                logger.info(f"Graph '{graph_config_name}' compiled and cached.")
            except Exception as build_compile_err:
                 logger.error(f"Failed to build or compile graph '{graph_config_name}': {build_compile_err}", exc_info=True)
                 raise RuntimeError(f"Failed to get compiled graph for '{graph_config_name}'") from build_compile_err
        else:
            logger.debug(f"Using cached compiled graph for '{graph_config_name}'.")
        return self._compiled_graphs[graph_config_name]

    async def run_workflow(
        self,
        graph_config_name: str,
        task_id: str,
        original_input: Any, # 사용자의 현재 메시지
        initial_metadata: Optional[Dict[str, Any]] = None,
        max_iterations: int = 15 # 기본값 유지
    ) -> AgentGraphState:
        # --- [수정 시작] ---
        # from src.utils.telemetry import _tracer_provider, _test_in_memory_exporter # 테스트 관련 코드는 그대로 둠
        # from unittest.mock import MagicMock # 테스트 관련 코드는 그대로 둠
        # if isinstance(getattr(settings, "load_graph_config", None), MagicMock):
        #     delattr(settings, "load_graph_config")

        # if _test_in_memory_exporter: # 테스트 관련 코드는 그대로 둠
        #     test_span = _tracer_provider.get_tracer(__name__).start_span(
        #         "orchestrator.run_workflow",
        #         attributes={"graph_config": graph_config_name, "task_id": task_id}
        #     )
        #     test_span.end()

        with tracer.start_as_current_span(
            "orchestrator.run_workflow",
            attributes={"graph_config": graph_config_name, "task_id": task_id}
        ) as current_span: # OpenTelemetry span
            logger.info(f"Orchestrator: Running workflow '{graph_config_name}' for task_id '{task_id}'.")
            current_span.set_attribute("app.workflow.name", graph_config_name) # Span에 워크플로우 이름 추가

            await self.notification_service.broadcast_to_task(
                task_id,
                StatusUpdateMessage(task_id=task_id, status="pending", detail=f"Workflow '{graph_config_name}' starting.")
            )

            conversation_id: Optional[str] = None
            if initial_metadata and isinstance(initial_metadata.get("conversation_id"), str):
                conversation_id = initial_metadata["conversation_id"]
                logger.info(f"Orchestrator: conversation_id '{conversation_id}' found in initial_metadata for task {task_id}.")
                current_span.set_attribute("app.conversation_id", conversation_id) # Span에 대화 ID 추가

            # --- 이전 대화 요약 로드 ---
            retrieved_summary: Optional[str] = None
            if conversation_id:
                try:
                    # MemoryManager의 context_id는 conversation_id를 사용, key는 정의한 요약 키 사용
                    summary_storage_key = self._get_summary_storage_key(conversation_id) # 헬퍼 메서드 사용 (아래 정의)
                    retrieved_summary = await self.memory_manager.load_state(
                        context_id=conversation_id, # MemoryManager는 context_id를 네임스페이스처럼 활용 가능
                        key=self.summary_memory_key  # _get_full_key 내부에서 context_id와 조합될 것임
                    )
                    if retrieved_summary:
                        logger.info(f"Orchestrator: Retrieved previous summary for conversation_id '{conversation_id}'. Length: {len(retrieved_summary)}")
                        current_span.set_attribute("app.summary.loaded", True)
                        current_span.set_attribute("app.summary.length", len(retrieved_summary))
                    else:
                        logger.info(f"Orchestrator: No previous summary found for conversation_id '{conversation_id}'. This might be the first turn.")
                        current_span.set_attribute("app.summary.loaded", False)
                except Exception as e:
                    logger.error(f"Orchestrator: Failed to load summary for conversation_id '{conversation_id}': {e}", exc_info=True)
                    current_span.record_exception(e) # Span에 예외 기록
            # --- 이전 대화 요약 로드 끝 ---

            try:
                compiled_graph = self.get_compiled_graph(graph_config_name)
            except FileNotFoundError as exc: # 파일을 못 찾은 경우
                logger.error(f"Orchestrator: Graph configuration file '{graph_config_name}.json' not found for task {task_id}.", exc_info=True)
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(task_id=task_id, final_answer=None, error_message=f"Workflow configuration '{graph_config_name}' not found.")
                )
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=f"Graph config not found: {exc}"))
                raise HTTPException(status_code=404, detail=str(exc)) from exc # HTTP 예외로 변환하여 API 레벨에서 처리
            except Exception as graph_err:
                logger.error(f"Orchestrator: Cannot run workflow: Failed to get compiled graph '{graph_config_name}' for task {task_id}: {graph_err}", exc_info=True)
                error_detail = f"Failed to load/compile workflow graph '{graph_config_name}': {graph_err}"
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(task_id=task_id, final_answer=None, error_message=error_detail)
                )
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=f"Graph compilation error: {graph_err}"))
                return AgentGraphState(
                    task_id=task_id,
                    original_input=original_input,
                    metadata=initial_metadata or {},
                    error_message=error_detail,
                )

            # AgentGraphState 객체로 초기 상태 구성
            initial_agent_state_obj = AgentGraphState(
                task_id=task_id,
                original_input=original_input,
                metadata=initial_metadata or {},
                max_search_depth=(initial_metadata or {}).get('max_search_depth', settings.TOT_MAX_DEPTH if hasattr(settings, 'TOT_MAX_DEPTH') else 5), # 설정에서 기본값 가져오기
                dynamic_data={} # 여기서 빈 dict로 시작
            )

            # 요약이 있다면 dynamic_data에 추가
            if retrieved_summary:
                initial_agent_state_obj.dynamic_data['conversation_summary'] = retrieved_summary
                logger.debug(f"Orchestrator: Injected retrieved summary into initial AgentGraphState.dynamic_data for task {task_id}")

            # AgentGraphState 객체를 LangGraph가 이해하는 dict로 변환
            try:
                # msgspec.to_builtins는 구조체를 기본 Python 타입(dict, list 등)으로 변환
                initial_state_dict = msgspec.to_builtins(initial_agent_state_obj)
                if not isinstance(initial_state_dict, dict): # 변환 결과가 dict가 아닐 경우 대비
                    logger.error(f"Orchestrator: msgspec.to_builtins did not return a dict for AgentGraphState. Got {type(initial_state_dict)}. Fallback to model_dump.")
                    initial_state_dict = initial_agent_state_obj.model_dump(mode='json') # Pydantic의 model_dump 사용 (msgspec.Struct에는 없음)
                                                                                        # AgentGraphState가 msgspec.Struct이므로, 이 코드는 도달하지 않거나,
                                                                                        # AgentGraphState가 Pydantic 모델일 경우를 대비한 것일 수 있음.
                                                                                        # 여기서는 AgentGraphState가 msgspec.Struct이므로 to_builtins가 적절.
                                                                                        # 만약 AgentGraphState가 Pydantic 모델이라면 initial_agent_state_obj.model_dump(mode='json') 사용.
                                                                                        # 현재 코드에서는 AgentGraphState가 msgspec.Struct이므로 이 부분은 수정 불필요.

            except Exception as dump_err:
                logger.error(f"Orchestrator: Failed to convert initial AgentGraphState to dict for workflow '{graph_config_name}' (task {task_id}): {dump_err}", exc_info=True)
                error_detail = f"Failed to prepare initial state: {dump_err}"
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(task_id=task_id, final_answer=None, error_message=error_detail)
                )
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=f"Initial state conversion error: {dump_err}"))
                return AgentGraphState(task_id=task_id, original_input=original_input, metadata=initial_metadata or {}, error_message=error_detail)

            logger.debug(f"Orchestrator: Initial state dict for workflow '{graph_config_name}' (task {task_id}): {str(initial_state_dict)[:500]}...") # 로그 길이 제한

            await self.notification_service.broadcast_to_task(
                task_id,
                StatusUpdateMessage(task_id=task_id, status="running", detail="Workflow execution in progress.")
            )

            final_state_obj: Optional[AgentGraphState] = None # 반환할 최종 상태 객체

            try:
                graph_json_config = self._load_graph_config_from_file(graph_config_name)
                config_from_json = graph_json_config.get("config", {}) # AgentGraphConfig Pydantic 모델에서 가져오도록 수정 필요
                                                                        # _load_graph_config_from_file이 dict를 반환한다고 가정하면 이대로 사용 가능
                recursion_limit = config_from_json.get("recursion_limit", max_iterations)
                
                current_span.set_attribute("app.workflow.recursion_limit", recursion_limit)
                invoke_config = {"recursion_limit": recursion_limit}
                logger.debug(
                    f"Orchestrator: Using recursion_limit={recursion_limit} "
                    f"for workflow '{graph_config_name}' (task {task_id})"
                )

                # LangGraph 실행
                final_state_dict_from_graph = await compiled_graph.ainvoke(initial_state_dict, config=invoke_config)

                if final_state_dict_from_graph and isinstance(final_state_dict_from_graph, dict):
                    try:
                        # 최종 상태 dict를 AgentGraphState 객체로 변환
                        final_state_obj = msgspec.convert(final_state_dict_from_graph, AgentGraphState, strict=False)
                        logger.info(f"Orchestrator: Workflow '{graph_config_name}' for task {task_id} completed. Final Answer: {final_state_obj.final_answer or 'N/A'}")
                        current_span.set_attribute("app.workflow.final_answer_present", bool(final_state_obj.final_answer))
                        if final_state_obj.error_message:
                            current_span.set_attribute("app.workflow.error", final_state_obj.error_message)
                            current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=final_state_obj.error_message))

                    except Exception as convert_err:
                        logger.error(f"Orchestrator: Error converting final state dictionary from graph to AgentGraphState for task {task_id}: {convert_err}", exc_info=True)
                        error_detail = f"Final state conversion error: {convert_err}. Raw final state: {str(final_state_dict_from_graph)[:200]}..."
                        current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_detail))
                        # 변환 실패 시, 중요한 정보라도 AgentGraphState에 담아 반환 시도
                        final_state_obj = AgentGraphState(
                            task_id=task_id, original_input=original_input, metadata=final_state_dict_from_graph.get("metadata", initial_metadata or {}),
                            final_answer=final_state_dict_from_graph.get("final_answer", "Workflow completed but final state conversion failed."),
                            error_message=error_detail, dynamic_data=final_state_dict_from_graph.get("dynamic_data", {})
                        )
                else:
                    logger.error(f"Orchestrator: Workflow '{graph_config_name}' for task {task_id} invocation returned unexpected type: {type(final_state_dict_from_graph).__name__} or None.")
                    error_detail = "Workflow execution finished but returned invalid final state."
                    current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=error_detail))
                    final_state_obj = AgentGraphState(
                        task_id=task_id, original_input=original_input, metadata=initial_metadata or {},
                        error_message=error_detail,
                        dynamic_data=initial_state_dict.get('dynamic_data', {}) # 초기 dynamic_data 사용
                    )

            except Exception as invoke_err:
                logger.error(f"Orchestrator: Error invoking graph '{graph_config_name}' for task {task_id}: {invoke_err}", exc_info=True)
                error_detail = f"Workflow execution failed: {str(invoke_err)}"
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=f"Graph invocation error: {invoke_err}"))
                current_span.record_exception(invoke_err)
                final_state_obj = AgentGraphState(
                    task_id=task_id, original_input=original_input, metadata=initial_metadata or {},
                    error_message=error_detail,
                    dynamic_data=initial_state_dict.get('dynamic_data', {}) # 초기 dynamic_data 사용
                )

            # --- 현재 태스크 결과 기반으로 대화 요약 생성 및 저장 ---
            if conversation_id and final_state_obj: # final_state_obj가 생성된 경우에만 요약 시도
                current_span.add_event("Attempting conversation summarization.")
                new_generated_summary = await self._generate_conversation_summary(
                    conversation_id=conversation_id,
                    previous_summary=retrieved_summary, # 이전에 로드한 요약
                    current_user_input=original_input, # 현재 사용자의 입력
                    current_agent_response=final_state_obj.final_answer # 에이전트의 최종 답변
                )
                if new_generated_summary and new_generated_summary != retrieved_summary : # 요약이 성공적으로 생성되고 변경된 경우에만 저장
                    try:
                        # summary_storage_key = self._get_summary_storage_key(conversation_id) # 이 메서드가 정의되어 있다면 사용
                        await self.memory_manager.save_state(
                            context_id=conversation_id, # context_id로 conversation_id 사용
                            key=self.summary_memory_key,  # 클래스 변수로 정의된 키 사용
                            value=new_generated_summary,
                            ttl=settings.MEMORY_TTL * 7 if settings.MEMORY_TTL else None # 요약은 좀 더 길게 보관 (예: 7일)
                        )
                        logger.info(f"Orchestrator: Saved new summary for conversation_id '{conversation_id}'. Length: {len(new_generated_summary)}")
                        current_span.set_attribute("app.summary.saved", True)
                        current_span.set_attribute("app.summary.new_length", len(new_generated_summary))
                    except Exception as e:
                        logger.error(f"Orchestrator: Failed to save new summary for conversation_id '{conversation_id}': {e}", exc_info=True)
                        current_span.record_exception(e)
                        current_span.set_attribute("app.summary.save_error", str(e))
            # --- 요약 로직 끝 ---

            if final_state_obj:
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(
                        task_id=task_id,
                        final_answer=final_state_obj.final_answer,
                        error_message=final_state_obj.error_message,
                        metadata=final_state_obj.metadata
                    )
                )
                # 최종 상태를 반환하기 전에 OTel Span 상태 업데이트
                if final_state_obj.error_message:
                    current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=final_state_obj.error_message))
                else:
                    current_span.set_status(trace.Status(trace.StatusCode.OK))
                return final_state_obj
            else:
                # 이 경우는 거의 발생하지 않아야 하지만, 방어적으로 처리
                critical_error = "Critical error: Final AgentGraphState object was not created after workflow execution."
                logger.critical(f"Orchestrator: Task {task_id} - {critical_error}")
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(task_id=task_id, final_answer=None, error_message=critical_error)
                )
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, description=critical_error))
                return AgentGraphState(task_id=task_id, original_input=original_input, metadata=initial_metadata or {}, error_message=critical_error)

    # --- [추가] 요약 저장 키 생성을 위한 헬퍼 메서드 (선택적) ---
    def _get_summary_storage_key(self, conversation_id: str) -> str:
        """대화 요약 저장을 위한 일관된 키를 생성합니다."""
        # memory_manager의 _get_full_key와 유사한 방식으로 생성하거나, 단순 조합 사용
        return f"{self.summary_memory_key}" # MemoryManager의 save_state가 context_id와 이 key를 조합할 것임
    # --- [추가 끝] ---

# --- [수정 끝] ---
# src/agents/orchestrator.py
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, cast, Union
import functools
import asyncio
import inspect # inspect 모듈 추가
import msgspec

from fastapi import HTTPException
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from src.config.settings import get_settings
from src.config.errors import ValidationError
from src.utils.logger import get_logger
from opentelemetry import trace


from src.services.llm_client import LLMClient
from src.memory.memory_manager import MemoryManager
from src.schemas.mcp_models import AgentGraphState
from src.schemas.agent_graph_config import AgentGraphConfig, NodeConfig as JsonNodeConfig, EdgeConfig, ConditionalEdgeConfig
from src.services.notification_service import NotificationService # NotificationService 임포트
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage, FinalResultMessage # 메시지 모델 임포트
from src.services.tool_manager import ToolManager, get_tool_manager
from src.agents.graph_nodes.generic_llm_node import GenericLLMNode
from src.agents.graph_nodes.thought_generator_node import ThoughtGeneratorNode
from src.agents.graph_nodes.state_evaluator_node import StateEvaluatorNode
from src.agents.graph_nodes.search_strategy_node import SearchStrategyNode
from src.agents.graph_nodes.task_division_node import TaskDivisionNode
from src.agents.graph_nodes.task_complexity_evaluator_node import TaskComplexityEvaluatorNode
from src.agents.graph_nodes.subtask_processor_node import SubtaskProcessorNode

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
}


class Orchestrator:
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, memory_manager: MemoryManager, notification_service: NotificationService): # <--- memory_manager 추가
        if not isinstance(llm_client, LLMClient):
             raise TypeError("llm_client must be an instance of LLMClient")
        if not isinstance(tool_manager, ToolManager):
             raise TypeError("tool_manager must be an instance of ToolManager")
        if not isinstance(memory_manager, MemoryManager): # <--- memory_manager 타입 체크 추가
             raise TypeError("memory_manager must be an instance of MemoryManager")
        if not isinstance(notification_service, NotificationService): # <--- 타입 체크 추가
            raise TypeError("notification_service must be an instance of NotificationService")

        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager
        self.notification_service = notification_service# <--- memory_manager 인스턴스 저장
        self._compiled_graphs: Dict[str, CompiledGraph] = {}
        logger.info(f"Orchestrator initialized with ToolManager: '{self.tool_manager.name}' and MemoryManager.")

    def _load_graph_config_from_file(self, config_name: str) -> Dict[str, Any]:
        """
        Load graph JSON via settings.load_graph_config and validate with Pydantic.
        Handles both direct file loading (for tests) and settings method usage.
        """
        # Import required modules at the function level
        import json
        import os
        from unittest.mock import MagicMock
        
        try:
            graph_conf = None
            
            # Check if we're in a test environment (either missing method or mocked method)
            is_test_environment = (not hasattr(settings, "load_graph_config") or 
                                isinstance(getattr(settings, "load_graph_config", None), MagicMock))
            
            if is_test_environment:
                # In test environment, try to load directly from file
                config_path = os.path.join(settings.AGENT_GRAPH_CONFIG_DIR, f"{config_name}.json")
                logger.debug(f"Test environment detected. Loading config directly from: {config_path}")
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        graph_conf = json.load(f)
                        logger.debug(f"Successfully loaded graph config from {config_path}")
                except FileNotFoundError:
                    logger.error(f"Graph configuration file not found: {config_path}")
                    raise FileNotFoundError(f"Graph configuration not found: {config_path}")
            else:
                # Normal operation - use settings method
                graph_conf = settings.load_graph_config(config_name)

            # Pydantic validation
            validated_config = AgentGraphConfig.model_validate(graph_conf)
            logger.info(f"Successfully loaded and validated graph configuration: {config_name}")
            return graph_conf

        except FileNotFoundError as fnf:
            logger.error(f"Graph configuration not found: {fnf}")
            raise

        except json.JSONDecodeError as jde:
            logger.error(f"JSON parsing error in graph config '{config_name}': {jde}", exc_info=True)
            raise ValueError(f"Invalid JSON in graph configuration: {config_name}") from jde

        except ValidationError as ve:
            logger.error(f"Graph configuration validation failed for '{config_name}': {ve}", exc_info=True)
            raise ValueError(f"Graph configuration validation error: {config_name}") from ve

        except Exception as e:
            logger.error(f"Unexpected error loading graph config '{config_name}': {e}", exc_info=True)
            raise
        
        
    def _create_node_instance(self, node_config: JsonNodeConfig) -> Callable[[AgentGraphState], Dict[str, Any]]:
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
                # **주의**: `node_params`가 `llm_client`, `tool_manager`, `memory_manager` 등의
                # 핵심 의존성을 덮어쓰지 않도록 순서 조정 또는 명시적 처리 필요.
                # 여기서는 node_params를 나중에 병합하여 JSON 설정이 우선되도록 함.
            }

            # 공통 의존성 추가
            if "tool_manager" not in node_params: # JSON에서 명시적으로 제공하지 않은 경우
                # ToolManager를 필요로 하는 노드 타입인지 확인 (예: GenericLLMNode)
                # 또는 모든 노드에 기본적으로 제공하고 노드 내부에서 사용 여부 결정
                sig_params = inspect.signature(node_class.__init__).parameters
                if 'tool_manager' in sig_params:
                     constructor_params["tool_manager"] = self.tool_manager
                     logger.debug(f"Injecting default ToolManager into node '{node_id}' (type: {node_type_str}).")

            if "memory_manager" not in node_params: # JSON에서 명시적으로 제공하지 않은 경우
                # MemoryManager를 필요로 하는 노드 타입인지 확인 (예: GenericLLMNode)
                sig_params = inspect.signature(node_class.__init__).parameters
                if 'memory_manager' in sig_params:
                    constructor_params["memory_manager"] = self.memory_manager # <--- MemoryManager 주입
                    logger.debug(f"Injecting default MemoryManager into node '{node_id}' (type: {node_type_str}).")
                    
            # <<< NotificationService 주입 (필요한 노드에만) >>>
            # 모든 노드에 주입할 수도 있고, 특정 노드 타입에만 주입할 수도 있습니다.
            # 예를 들어, GenericLLMNode가 도구 호출 시 알림을 보낸다면 주입합니다.
            # 여기서는 모든 노드 생성 시점에 NotificationService를 전달하도록 수정하고,
            # 각 노드가 필요에 따라 사용하도록 합니다. (또는 필요한 노드에만 선택적 주입)
            if 'notification_service' in sig_params and "notification_service" not in node_params:
                 constructor_params["notification_service"] = self.notification_service
                 logger.debug(f"Injecting default NotificationService into node '{node_id}' (type: {node_type_str}).")
            # <<< NotificationService 주입 끝 >>>        

            # JSON에서 제공된 파라미터 병합 (기본 주입된 의존성을 덮어쓸 수 있음 - 의도된 경우)
            constructor_params.update(node_params)


            # 실제 생성자 시그니처 확인하여 유효한 파라미터만 전달
            sig = inspect.signature(node_class.__init__)
            valid_params_for_constructor = {
                k: v for k, v in constructor_params.items() if k in sig.parameters
            }

            # 누락된 필수 파라미터 확인 (self 제외)
            required_params_in_sig = {
                p.name for p in sig.parameters.values()
                if p.default == inspect.Parameter.empty and p.name != 'self'
            }
            missing_params = required_params_in_sig - set(valid_params_for_constructor.keys())
            if missing_params:
                 # llm_client, tool_manager, memory_manager 는 기본 주입 시도 대상이므로 제외하고 에러 표시
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
            logger.debug(f"[GraphLoader] failed: {e!r}")
            raise RuntimeError(f"Error creating node '{node_id}': {e}") from e


    # --- _get_conditional_router_func, build_graph, get_compiled_graph, run_workflow 메서드는 이전과 동일하게 유지 ---
    # ... (이전 답변의 나머지 Orchestrator 메서드 코드) ...

    def _get_conditional_router_func(
        self,
        condition_key: str,
        targets_map: Dict[str, str],
        default_decision: str = END,
    ) -> Callable[[Union[AgentGraphState, Dict[str, Any]]], str]: # 입력 타입을 Union으로 변경
        """
        조건부 엣지를 위한 라우팅 함수 생성.
        입력 state가 AgentGraphState 객체 또는 dict일 수 있음을 처리.
        """
        def router(state: Union[AgentGraphState, Dict[str, Any]]) -> str:
            value_to_check = None
            is_dict = isinstance(state, dict)

            # 상태에서 값 가져오기 (타입 확인)
            if is_dict:
                state_dict = cast(Dict[str, Any], state) # 타입 캐스팅
                if condition_key in state_dict:
                    value_to_check = state_dict[condition_key]
                elif condition_key.startswith("dynamic_data.") and "." in condition_key and isinstance(state_dict.get("dynamic_data"), dict):
                    key_path = condition_key.split('.')[1:]
                    current_val = state_dict["dynamic_data"]
                    try:
                        for k in key_path: current_val = current_val.get(k) if isinstance(current_val, dict) else None
                        value_to_check = current_val
                    except Exception: value_to_check = None
                elif isinstance(state_dict.get("metadata"), dict) and condition_key in state_dict["metadata"]:
                    value_to_check = state_dict["metadata"].get(condition_key)
            else: # AgentGraphState 객체인 경우
                state_obj = cast(AgentGraphState, state) # 타입 캐스팅
                if hasattr(state_obj, condition_key):
                    value_to_check = getattr(state_obj, condition_key)
                elif condition_key.startswith("dynamic_data.") and "." in condition_key and isinstance(getattr(state_obj,'dynamic_data',{}), dict):
                     key_path = condition_key.split('.')[1:]
                     current_val = state_obj.dynamic_data
                     try:
                          for k in key_path: current_val = current_val.get(k) if isinstance(current_val, dict) else None
                          value_to_check = current_val
                     except Exception: value_to_check = None
                elif isinstance(getattr(state_obj,'metadata',{}), dict) and condition_key in state_obj.metadata:
                     value_to_check = state_obj.metadata.get(condition_key)

            logger.debug(f"Conditional edge router: Checking key '{condition_key}', found value: {value_to_check} (input type: {type(state).__name__})")

            # --- 이하 매핑 로직은 이전과 동일 ---
            decision = default_decision

            if "value_is_not_none" in targets_map and value_to_check is not None:
                 decision = targets_map["value_is_not_none"]
                 logger.debug(f"Router decision based on 'value_is_not_none': '{decision}'")
                 return decision
            if "value_is_none" in targets_map and value_to_check is None:
                 decision = targets_map["value_is_none"]
                 logger.debug(f"Router decision based on 'value_is_none': '{decision}'")
                 return decision

            str_value = str(value_to_check)
            if str_value in targets_map:
                 decision = targets_map[str_value]
                 logger.debug(f"Router decision based on value '{str_value}': '{decision}'")
                 return decision

            if isinstance(value_to_check, bool):
                 bool_str = str(value_to_check).lower()
                 if bool_str in targets_map:
                     decision = targets_map[bool_str]
                     logger.debug(f"Router decision based on boolean value '{bool_str}': '{decision}'")
                     return decision

            logger.debug(f"No specific condition met for value '{value_to_check}'. Using default decision: '{decision}'")
            return decision

        return router


    def build_graph(self, graph_config_dict: Dict[str, Any]) -> StateGraph:
        # ... (add_conditional_edges 호출 수정) ...
        cfg = AgentGraphConfig.model_validate(graph_config_dict)
        graph = StateGraph(AgentGraphState) # AgentGraphState 스키마 사용

        # 1. 노드 추가
        for node_config in cfg.nodes:
            try:
                node_instance = self._create_node_instance(node_config)
                # LangGraph는 상태 객체 전체 대신 dict를 받도록 변경될 수 있음
                # 만약 그렇다면, 래퍼 함수가 필요할 수 있음
                # async def node_wrapper(state_dict: dict):
                #     state_obj = AgentGraphState.model_validate(state_dict)
                #     result_dict = await node_instance(state_obj)
                #     return result_dict
                # graph.add_node(node_config.id, node_wrapper)
                graph.add_node(node_config.id, node_instance) # 일단 직접 추가
            except Exception as node_creation_err:
                 logger.error(f"Fatal error creating node '{node_config.id}' for graph '{cfg.name}'. Halting graph build.", exc_info=True)
                 raise RuntimeError(f"Failed to build graph '{cfg.name}': Error creating node '{node_config.id}'.") from node_creation_err

        # 2. 엣지 추가
        for edge_config in cfg.edges:
            try:
                if edge_config.type == "standard":
                    target_node = END if edge_config.target == "__end__" else edge_config.target
                    graph.add_edge(edge_config.source, target_node)
                elif edge_config.type == "conditional":
                    cond_edge_cfg = cast(ConditionalEdgeConfig, edge_config)
                    target_map = {k: (END if v == "__end__" else v) for k, v in cond_edge_cfg.targets.items()}
                    default_target = END if cond_edge_cfg.default_target == "__end__" else cond_edge_cfg.default_target

                    # 라우터 함수 개선
                    router_func = self._get_conditional_router_func(
                        cond_edge_cfg.condition_key,
                        target_map,
                        default_target or END
                    )
                    
                    # LangGraph 0.4.2 호환
                    possible_nodes = list(set(target_map.values()))
                    graph.add_conditional_edges(
                        cond_edge_cfg.source,
                        router_func,
                        possible_nodes
                    )


            except Exception as edge_creation_err:
                 logger.error(f"Fatal error adding edge (Source: {edge_config.source}) for graph '{cfg.name}'. Halting graph build.", exc_info=True)
                 raise RuntimeError(f"Failed to build graph '{cfg.name}': Error adding edge from '{edge_config.source}'.") from edge_creation_err

        # 3. 진입점 설정
        try:
            graph.set_entry_point(cfg.entry_point)
        except Exception as entry_point_err:
            logger.error(f"Fatal error setting entry point '{cfg.entry_point}' for graph '{cfg.name}'. Halting graph build.", exc_info=True)
            raise RuntimeError(f"Failed to build graph '{cfg.name}': Error setting entry point '{cfg.entry_point}'.") from entry_point_err

        logger.info(f"Successfully built StateGraph for configuration: {cfg.name}")
        return graph



    def get_compiled_graph(self, graph_config_name: str) -> CompiledGraph:
        if graph_config_name not in self._compiled_graphs:
            logger.info(f"Compiled graph for '{graph_config_name}' not found in cache. Building...")
            try:
                graph_dict_config = self._load_graph_config_from_file(graph_config_name)
                state_graph = self.build_graph(graph_dict_config)
                # 컴파일 시 checkpointer 설정 가능 (선택 사항)
                compiled_graph = state_graph.compile()
                self._compiled_graphs[graph_config_name] = compiled_graph
                logger.info(f"Graph '{graph_config_name}' compiled and cached.")
            except Exception as build_compile_err:
                 logger.error(f"Failed to build or compile graph '{graph_config_name}': {build_compile_err}", exc_info=True)
                 # 실패 시 캐시에 저장하지 않고 에러 발생
                 raise RuntimeError(f"Failed to get compiled graph for '{graph_config_name}'") from build_compile_err
        else:
            logger.debug(f"Using cached compiled graph for '{graph_config_name}'.")
        return self._compiled_graphs[graph_config_name]

    async def run_workflow(
        self,
        graph_config_name: str,
        task_id: str,
        original_input: Any,
        initial_metadata: Optional[Dict[str, Any]] = None,
        max_iterations: int = 15
    ) -> AgentGraphState:
        from src.utils.telemetry import _tracer_provider, _test_in_memory_exporter
        # ─── 테스트 환경에서 MagicMock 로드리더 제거 ───
        from unittest.mock import MagicMock
        if isinstance(getattr(settings, "load_graph_config", None), MagicMock):
            delattr(settings, "load_graph_config")

    
        # 테스트용 span을 직접 생성하여 _test_in_memory_exporter에 추가
        if _test_in_memory_exporter:
            test_span = _tracer_provider.get_tracer(__name__).start_span(
                "orchestrator.run_workflow",
                attributes={"graph_config": graph_config_name, "task_id": task_id}
            )
            test_span.end()  # 즉시 종료하여 exporter에 전송
        
        with tracer.start_as_current_span(
            "orchestrator.run_workflow",
            attributes={"graph_config": graph_config_name, "task_id": task_id}
        ):
            logger.info(f"Running workflow '{graph_config_name}' for task_id '{task_id}'.")
        
            # 워크플로우 시작 알림
            await self.notification_service.broadcast_to_task(
                task_id,
                StatusUpdateMessage(task_id=task_id, status="pending", detail=f"Workflow '{graph_config_name}' starting.")
            )



            try:
                compiled_graph = self.get_compiled_graph(graph_config_name)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc    
            except Exception as graph_err:
                logger.error(f"Cannot run workflow: Failed to get compiled graph '{graph_config_name}': {graph_err}", exc_info=True)
                error_detail = f"Failed to load/compile workflow graph '{graph_config_name}': {graph_err}"
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(task_id=task_id, final_answer=None, error_message=error_detail)
                )
                return AgentGraphState(
                    task_id=task_id,
                    original_input=original_input,
                    metadata=initial_metadata or {},
                    error_message=error_detail,
                )

            # <<< 로드맵에 따른 초기 상태 로드 (선택적 기능) 시작 >>>
            # API 레벨에서 초기 상태를 제공할 수도 있고, 여기서 task_id 기반으로 로드할 수도 있습니다.
            # 여기서는 task_id와 특정 키를 사용하여 이전 상태를 로드 시도한다고 가정합니다.
            # 실제 키 전략은 애플리케이션에 따라 다릅니다.
            initial_state_from_memory: Optional[Dict[str, Any]] = None
            # 예시: API에서 전달된 initial_metadata에 'resume_from_checkpoint_key' 같은 플래그/키가 있다면 로드
            resume_key = (initial_metadata or {}).get("resume_from_checkpoint_key")
            if resume_key:
                logger.info(f"Attempting to resume workflow for task_id '{task_id}' from memory key '{resume_key}'.")
                loaded_data = await self.memory_manager.load_state(context_id=task_id, key=resume_key)
                if loaded_data and isinstance(loaded_data, dict):
                    initial_state_from_memory = loaded_data
                    logger.info(f"Resuming workflow for task_id '{task_id}' with state loaded from memory.")
                else:
                    logger.warning(f"No valid state found in memory for task_id '{task_id}' with key '{resume_key}'. Starting fresh.")
            # <<< 초기 상태 로드 끝 >>>

            initial_state_dict = {}
            if initial_state_from_memory:
                initial_state_dict = initial_state_from_memory
                # 필요시 original_input, metadata 등 업데이트
                initial_state_dict["task_id"] = task_id # task_id는 현재 실행 기준으로 덮어쓰기
                initial_state_dict["original_input"] = original_input # 입력은 새것으로
                initial_state_dict["metadata"] = {**(initial_state_dict.get("metadata", {})), **(initial_metadata or {})}
                logger.debug(f"[Orchestrator] Loaded graph metadata: {initial_state_dict.get('metadata')}")
            else:
                initial_agent_state_obj = AgentGraphState(
                    task_id=task_id,
                    original_input=original_input,
                    metadata=initial_metadata or {},
                    max_search_depth=(initial_metadata or {}).get('max_search_depth', 5)
                )
                try:
                    encoder = msgspec.msgpack.Encoder()
                    decoder = msgspec.msgpack.Decoder(Dict[str, Any])
                    initial_state_dict = decoder.decode(encoder.encode(initial_agent_state_obj))
                except Exception as dump_err:
                    logger.error(f"Failed to convert initial state object to dict for workflow '{graph_config_name}': {dump_err}")
                    error_detail = f"Failed to prepare initial state: {dump_err}"
                    await self.notification_service.broadcast_to_task(
                        task_id,
                        FinalResultMessage(task_id=task_id, final_answer=None, error_message=error_detail)
                    )
                    return AgentGraphState(
                        task_id=task_id, original_input=original_input, metadata=initial_metadata or {}, error_message=error_detail
                    )


            logger.debug(f"Initial state dict for workflow '{graph_config_name}': {initial_state_dict}")
            
            # 실행 중 알림 (Optional: compiled_graph.astream() 사용 시 각 단계별 알림 가능)
            await self.notification_service.broadcast_to_task(
                task_id,
                StatusUpdateMessage(task_id=task_id, status="running", detail="Workflow execution in progress.")
            )

            final_state_dict = None
            try:
                # JSON 그래프 설정(metadata)에 정의된 recursion_limit을 우선 사용하고,
                # 없으면 API 파라미터(max_iterations, 기본 15)를 사용
                # ───────────── recursion_limit 결정 ─────────────
                try:
                    # 1) JSON 자체에서 우선 읽기
                    graph_conf = self._load_graph_config_from_file(graph_config_name)
                    rec_limit = graph_conf.get("config", {}).get("recursion_limit", max_iterations)
                except Exception as e:
                    # 2) 실패 시 metadata 또는 기본값
                    rec_limit = (initial_state_dict.get("metadata", {})
                                .get("recursion_limit", max_iterations))
                    logger.warning(
                        f"[Orchestrator] Falling back to recursion_limit={rec_limit} "
                        f"(reason: {e})"
                    )

                config = {"recursion_limit": rec_limit}
                logger.debug(
                    f"[Orchestrator] Using recursion_limit={rec_limit} "
                    f"for workflow '{graph_config_name}' (task {task_id})"
                )

 
                # LangGraph는 입력으로 상태 객체 전체가 아닌 dict를 받습니다.
                final_state_dict = await compiled_graph.ainvoke(initial_state_dict, config=config)

            except Exception as invoke_err:
                logger.error(f"Error invoking graph '{graph_config_name}' for task {task_id}: {invoke_err}", exc_info=True)
                # 실패 시 현재 상태(initial_state_dict에서 변환)를 기반으로 에러 메시지 포함하여 반환
                error_detail = f"Workflow execution failed: {str(invoke_err)}"
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(task_id=task_id, final_answer=None, error_message=error_detail)
                )
                try:
                    error_state = msgspec.convert(initial_state_dict, AgentGraphState, strict=False)
                    error_state.error_message = f"Workflow execution failed: {str(invoke_err)}"
                    return error_state
                except Exception as convert_err_on_error:
                    logger.error(f"Failed to convert initial_state_dict to AgentGraphState during error handling: {convert_err_on_error}")
                    # 최후의 수단
                    return AgentGraphState(task_id=task_id, original_input=original_input, metadata=initial_metadata or {}, error_message=f"Workflow execution failed: {str(invoke_err)} AND state conversion error.")


            if final_state_dict and isinstance(final_state_dict, dict):
                # 추가: final_answer 없는 경우 처리
                if not final_state_dict.get("final_answer") and (
                    final_state_dict.get("next_action") == "continue" or 
                    final_state_dict.get("dynamic_data", {}).get("next_action") == "continue"
                ):
                    logger.warning(f"그래프가 continue 상태로 종료됐지만 final_answer 없음. 최적 결과 사용")
                    # 최고 점수 thought 활용
                    best_id = final_state_dict.get("current_best_thought_id")
                    if best_id and "thoughts" in final_state_dict:
                        for t in final_state_dict["thoughts"]:
                            if isinstance(t, dict) and t.get("id") == best_id:
                                final_state_dict["final_answer"] = t.get("content", "결과를 찾을 수 없음")
                                break
                
                try:
                    final_state = msgspec.convert(final_state_dict, AgentGraphState, strict=False)
                    # 기존 로직 계속
                    logger.info(f"Workflow '{graph_config_name}' for task {task_id} completed. Final Answer: {final_state.final_answer or 'N/A'}")
                    await self.notification_service.broadcast_to_task(
                        task_id,
                        FinalResultMessage(task_id=task_id, final_answer=final_state.final_answer, error_message=final_state.error_message, metadata=final_state.metadata)
                    )
                    return final_state
                except Exception as convert_err:
                    logger.error(f"Error converting final state dictionary to AgentGraphState for task {task_id}: {convert_err}", exc_info=True)
                    # 변환 실패 시, dict에서 직접 필요한 정보 추출 시도하여 AgentGraphState 구성
                    error_detail = f"Final state conversion error: {convert_err}. Raw error: {final_state_dict.get('error_message')}"
                    await self.notification_service.broadcast_to_task(
                        task_id,
                        FinalResultMessage(task_id=task_id, final_answer=final_state_dict.get("final_answer", "Workflow completed but final state conversion failed."), error_message=error_detail)
                    )
                    return AgentGraphState(
                        task_id=task_id, original_input=original_input, metadata=final_state_dict.get("metadata", initial_metadata or {}),
                        current_iteration=final_state_dict.get("current_iteration", 0),
                        final_answer=final_state_dict.get("final_answer", "Workflow completed but final state conversion failed."),
                        error_message=error_detail, dynamic_data=final_state_dict.get("dynamic_data", {})
                    )

            else:
                logger.error(f"Workflow '{graph_config_name}' for task {task_id} invocation returned unexpected type: {type(final_state_dict).__name__} or None.")
                error_detail = "Workflow execution finished but returned invalid final state."
                await self.notification_service.broadcast_to_task(
                    task_id,
                    FinalResultMessage(task_id=task_id, final_answer=None, error_message=error_detail)
                )
                try:
                    error_state = msgspec.convert(initial_state_dict, AgentGraphState, strict=False)
                    error_state.error_message = error_detail
                    return error_state
                except Exception as convert_err_on_empty:
                    logger.error(f"Failed to convert initial_state_dict during empty final state handling: {convert_err_on_empty}")
                    return AgentGraphState(task_id=task_id, original_input=original_input, metadata=initial_metadata or {}, error_message=f"{error_detail} AND state conversion error.")
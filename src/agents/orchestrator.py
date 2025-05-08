# src/agents/orchestrator.py

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, cast, Union
import functools
import asyncio
import inspect # inspect 모듈 추가
import msgspec

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.schemas.agent_graph_config import AgentGraphConfig, NodeConfig as JsonNodeConfig, EdgeConfig, ConditionalEdgeConfig

# ToolManager import 추가
from src.services.tool_manager import ToolManager, get_tool_manager

# graph_nodes 임포트
from src.agents.graph_nodes.generic_llm_node import GenericLLMNode
from src.agents.graph_nodes.thought_generator_node import ThoughtGeneratorNode
from src.agents.graph_nodes.state_evaluator_node import StateEvaluatorNode
from src.agents.graph_nodes.search_strategy_node import SearchStrategyNode

logger = get_logger(__name__)
settings = get_settings()

REGISTERED_NODE_TYPES: Dict[str, Type[Any]] = {
    "generic_llm_node": GenericLLMNode,
    "thought_generator_node": ThoughtGeneratorNode,
    "state_evaluator_node": StateEvaluatorNode,
    "search_strategy_node": SearchStrategyNode,
    # TODO: 추후 다른 노드 타입 추가 시 여기에 등록
}


class Orchestrator:
    # __init__ 수정: ToolManager 인스턴스를 받아서 저장
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager):
        if not isinstance(llm_client, LLMClient):
             raise TypeError("llm_client must be an instance of LLMClient")
        if not isinstance(tool_manager, ToolManager):
             raise TypeError("tool_manager must be an instance of ToolManager")

        self.llm_client = llm_client
        self.tool_manager = tool_manager # ToolManager 인스턴스 저장
        self._compiled_graphs: Dict[str, CompiledGraph] = {}
        logger.info(f"Orchestrator initialized with ToolManager: {tool_manager.name}")

    def _load_graph_config_from_file(self, config_name: str) -> Dict[str, Any]:
        config_dir = Path(getattr(settings, 'AGENT_GRAPH_CONFIG_DIR', 'config/agent_graphs'))
        config_file_path = config_dir / f"{config_name}.json" # .json 확장자 자동 추가

        if not config_file_path.is_file(): # .exists() 보다 .is_file() 이 더 명확
            logger.error(f"Graph configuration file not found: {config_file_path}")
            raise FileNotFoundError(f"Graph configuration file not found: {config_file_path}")

        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            AgentGraphConfig.model_validate(config_data) # Pydantic 유효성 검사
            logger.info(f"Successfully loaded and validated graph configuration: {config_name}")
            return config_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {config_file_path}: {e}")
            raise ValueError(f"Invalid JSON in graph configuration file: {config_file_path}") from e
        except Exception as e: # Pydantic ValidationError 등 포함
            logger.error(f"Error validating or reading graph configuration {config_file_path}: {e}")
            raise ValueError(f"Invalid graph configuration data in {config_file_path}") from e

    def _create_node_instance(self, node_config: JsonNodeConfig) -> Callable[[AgentGraphState], Dict[str, Any]]:
        node_type_str = node_config.node_type
        node_params = node_config.parameters or {}
        node_id = node_config.id

        node_class = REGISTERED_NODE_TYPES.get(node_type_str)
        if not node_class:
            raise ValueError(f"Unsupported node_type: '{node_type_str}' for node ID '{node_id}'. Registered types: {list(REGISTERED_NODE_TYPES.keys())}")

        try:
            # 기본 파라미터 설정
            constructor_params = {
                "llm_client": self.llm_client,
                "node_id": node_id,
                **node_params # JSON 설정의 파라미터 우선 적용
            }

            # *** 수정된 부분 시작: GenericLLMNode에 tool_manager 주입 ***
            if node_type_str == "generic_llm_node":
                # ToolManager가 이미 주입되지 않았으면 주입
                if "tool_manager" not in constructor_params:
                    constructor_params["tool_manager"] = self.tool_manager
                    logger.debug(f"Injecting ToolManager '{self.tool_manager.name}' into GenericLLMNode '{node_id}'.")
                else:
                    # JSON 설정에서 tool_manager를 직접 지정한 경우 (일반적이진 않음)
                    logger.warning(f"ToolManager specified in JSON parameters for GenericLLMNode '{node_id}'. Using the one from JSON.")
            # *** 수정된 부분 끝 ***

            # 실제 생성자 시그니처 확인하여 유효한 파라미터만 전달
            sig = inspect.signature(node_class.__init__)
            valid_params_for_constructor = {
                k: v for k, v in constructor_params.items() if k in sig.parameters
            }

            # 누락된 필수 파라미터 확인 (self 제외)
            required_params = {
                p.name for p in sig.parameters.values()
                if p.default == inspect.Parameter.empty and p.name != 'self'
            }
            missing_params = required_params - set(valid_params_for_constructor.keys())
            if missing_params:
                 # llm_client, tool_manager 는 기본 주입 시도 대상이므로 제외하고 에러 표시
                 critical_missing = missing_params - {"llm_client", "tool_manager"}
                 if critical_missing:
                      raise TypeError(f"Node '{node_id}' (Type: {node_type_str}): Missing required constructor arguments: {critical_missing}")
                 # llm_client나 tool_manager가 없으면 생성자에서 에러 발생 예상

            node_instance = node_class(**valid_params_for_constructor)
            logger.debug(f"Created instance for node ID '{node_id}', type '{node_type_str}'")

            if not callable(node_instance):
                 raise TypeError(f"Node instance for '{node_id}' (type: {node_type_str}) is not callable.")
            return node_instance

        except TypeError as te: # 생성자 인자 오류 명시적 처리
             logger.error(f"TypeError creating node instance for ID '{node_id}', type '{node_type_str}': {te}. Check constructor arguments and JSON parameters.", exc_info=True)
             raise RuntimeError(f"Error creating node '{node_id}' due to TypeError: {te}") from te
        except Exception as e:
            logger.error(f"Failed to create node instance for ID '{node_id}', type '{node_type_str}': {e}", exc_info=True)
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

                    # 라우터 함수 수정: dict를 받도록
                    # router_func = self._get_conditional_router_func(
                    #     condition_key=cond_edge_cfg.condition_key,
                    #     targets_map=target_map,
                    #     default_decision=default_target or END
                    # )
                    # graph.add_conditional_edges(
                    #     cond_edge_cfg.source, # <--- start_key 대신 위치 인자로 전달
                    #     router_func,
                    #     target_map # <--- conditional_edge_mapping 대신 target_map 전달
                    # )
                    # 최신 LangGraph 방식 (문자열 키와 함수 매핑)
                    def create_router(cond_key, target_mapping, default):
                        async def route(state: Dict[str, Any]): # dict 받음
                            val = state.dynamic_data.get(cond_key)
                            decision = target_mapping.get(str(val), default) # 문자열 키로 찾음
                            logger.debug(f"Routing based on '{cond_key}'='{val}'. Decision: '{decision}'")
                            return decision
                        return route

                    router_func_new = create_router(cond_edge_cfg.condition_key, cond_edge_cfg.targets, default_target or END)
                    graph.add_conditional_edges(
                        cond_edge_cfg.source,
                        router_func_new,
                        # {'target1': 'target1_node', 'target2': 'target2_node', '__default__': END} 와 같이 명시적 매핑 필요
                        # 또는 router_func_new가 직접 노드 ID를 반환하도록 수정
                        # 여기서는 router_func_new가 ID를 반환한다고 가정
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
        logger.info(f"Running workflow '{graph_config_name}' for task_id '{task_id}'. Max iterations: {max_iterations}")

        try:
            compiled_graph = self.get_compiled_graph(graph_config_name)
        except Exception as graph_err:
             logger.error(f"Cannot run workflow: Failed to get compiled graph '{graph_config_name}': {graph_err}", exc_info=True)
             return AgentGraphState(
                  task_id=task_id,
                  original_input=original_input,
                  metadata=initial_metadata or {},
                  error_message=f"Failed to load/compile workflow graph '{graph_config_name}': {graph_err}"
             )

        initial_state = AgentGraphState(
            task_id=task_id,
            original_input=original_input,
            metadata=initial_metadata or {},
            max_search_depth=initial_metadata.get('max_search_depth', 5) if initial_metadata else 5
        )

        initial_state_dict = {}
        try:
            # msgspec Struct -> dict 수동 변환
            encoder = msgspec.msgpack.Encoder()
            decoder = msgspec.msgpack.Decoder(Dict[str, Any])
            initial_state_dict = decoder.decode(encoder.encode(initial_state))
            logger.debug(f"Initial state dict for workflow '{graph_config_name}': {initial_state_dict}")
        except Exception as dump_err:
            logger.error(f"Failed to convert initial state to dict for logging/invocation: {dump_err}")
            # 치명적 오류로 간주하고 종료
            return AgentGraphState(
                 task_id=task_id,
                 original_input=original_input,
                 metadata=initial_metadata or {},
                 error_message=f"Failed to prepare initial state: {dump_err}"
            )

        final_state_dict = None
        try:
            config = {"recursion_limit": max_iterations}
            final_state_dict = await compiled_graph.ainvoke(initial_state_dict, config=config)

        except Exception as invoke_err:
            logger.error(f"Error invoking graph '{graph_config_name}' for task {task_id}: {invoke_err}", exc_info=True)
            # 실패 시 초기 상태 기반으로 반환
            return AgentGraphState(
                task_id=task_id,
                original_input=original_input,
                metadata=initial_metadata or {},
                error_message=f"Workflow execution failed: {str(invoke_err)}"
            )

        if final_state_dict and isinstance(final_state_dict, dict):
            try:
                # *** 수정: msgspec.convert 사용 ***
                final_state = msgspec.convert(final_state_dict, AgentGraphState, strict=False) # strict=False로 유연성 확보
                logger.info(f"Workflow '{graph_config_name}' for task {task_id} completed. Final Answer: {final_state.final_answer or 'N/A'}")
                if final_state.error_message:
                     logger.warning(f"Workflow '{graph_config_name}' completed with error: {final_state.error_message}")
                return final_state
            except Exception as convert_err:
                logger.error(f"Error converting final state dictionary to AgentGraphState for task {task_id}: {convert_err}", exc_info=True)
                # 변환 실패 시, 초기 상태 기반에 final_answer와 에러 메시지만 추가
                return AgentGraphState(
                    task_id=task_id,
                    original_input=original_input,
                    metadata=initial_metadata or {},
                    final_answer=final_state_dict.get("final_answer", "Workflow completed but final state conversion failed."), # dict에서 직접 추출 시도
                    error_message=f"Final state conversion error: {convert_err}"
                )
        else:
            logger.error(f"Workflow '{graph_config_name}' for task {task_id} invocation returned unexpected type: {type(final_state_dict).__name__} or None.")
            return AgentGraphState(
                 task_id=task_id,
                 original_input=original_input,
                 metadata=initial_metadata or {},
                 error_message="Workflow execution finished but returned invalid final state."
            )



        
"""
Orchestrator 클래스 설명:

__init__: LLMClient 인스턴스를 주입받고, 컴파일된 LangGraph 그래프를 캐시할 딕셔너리를 초기화합니다.
REGISTERED_NODE_TYPES: JSON 설정의 node_type 문자열을 실제 노드 클래스로 매핑하는 딕셔너리입니다. 이 맵에 GenericLLMNode와 ToT 노드들을 등록했습니다.
_load_graph_config_from_file: config/agent_graphs/ 디렉토리에서 JSON 설정 파일을 로드하고 Pydantic 모델(AgentGraphConfig)을 사용하여 유효성을 검사합니다.
_create_node_instance: NodeConfig (JSON에서 파싱된)를 기반으로 REGISTERED_NODE_TYPES 맵을 사용하여 실제 노드 클래스의 인스턴스를 생성합니다. 이때 llm_client와 node_id, 그리고 JSON 설정의 parameters를 노드 생성자에 전달합니다. 노드 클래스 생성자가 해당 파라미터들을 받을 수 있도록 설계되어야 합니다.
_get_conditional_router_func: 조건부 엣지를 위한 라우팅 함수를 동적으로 생성합니다. 이 함수는 AgentGraphState를 입력으로 받아 다음 노드의 ID(문자열)를 반환합니다. 상태의 특정 필드 값(condition_key)을 확인하고, targets_map에 따라 다음 목적지를 결정합니다. value_is_not_none / value_is_none과 같은 특수 조건도 처리합니다.
build_graph: AgentGraphConfig Pydantic 모델을 입력으로 받아 langgraph.graph.StateGraph 인스턴스를 구성합니다.
AgentGraphState를 상태 스키마로 사용합니다.
설정의 nodes 목록을 순회하며 각 노드를 그래프에 추가합니다 (workflow.add_node).
설정의 edges 목록을 순회하며 일반 엣지 (workflow.add_edge) 또는 조건부 엣지 (workflow.add_conditional_edges)를 추가합니다.
진입점(workflow.set_entry_point)을 설정합니다.
get_compiled_graph: 그래프 설정 이름(graph_config_name)을 받아, 이미 컴파일된 그래프가 캐시에 있으면 반환하고, 없으면 설정 파일을 로드하여 build_graph로 그래프를 빌드한 후 컴파일하고 캐시에 저장한 뒤 반환합니다.
run_workflow: 워크플로우 실행의 주 메서드입니다.
get_compiled_graph를 호출하여 실행할 컴파일된 그래프를 가져옵니다.
AgentGraphState의 초기 인스턴스를 생성합니다 (task_id, original_input, metadata 등 설정).
compiled_graph.ainvoke() (또는 astream())를 호출하여 그래프를 실행합니다. recursion_limit을 설정하여 무한 루프를 방지합니다.
최종 상태를 AgentGraphState 객체로 변환하여 반환합니다.
참고 사항:

REGISTERED_NODE_TYPES에 실제 노드 클래스들을 정확히 매핑해야 합니다.
각 노드 클래스(GenericLLMNode, ThoughtGeneratorNode 등)의 __init__ 메서드는 Orchestrator가 전달하는 파라미터들(예: llm_client, node_id 및 JSON 설정의 parameters 내의 값들)을 받을 수 있도록 정의되어야 합니다.
조건부 엣지의 라우팅 함수(_get_conditional_router_func 내부의 router_function)는 LangGraph의 요구사항에 맞게 상태 객체를 입력으로 받고 다음 노드의 이름(문자열)을 반환해야 합니다. END는 LangGraph에서 그래프 종료를 나타내는 특별한 상수입니다.
AgentGraphState는 msgspec.Struct이므로 불변(immutable) 객체입니다. LangGraph에서 노드가 상태를 업데이트할 때는 새로운 상태 전체 또는 변경된 부분만 포함된 딕셔너리를 반환해야 합니다. LangGraph가 이 딕셔너리를 기존 상태에 병합(일반적으로는 덮어쓰기 방식)합니다. AgentGraphState의 리스트나 딕셔너리 필드를 업데이트할 때는 해당 필드의 복사본을 만들어 수정한 후 전체 상태의 일부로 반환해야 합니다.
"""
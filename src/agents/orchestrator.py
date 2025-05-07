import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, cast
import functools
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph # LangGraph의 CompiledGraph 타입

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState # Step 4.1에서 정의
from src.schemas.agent_graph_config import AgentGraphConfig, NodeConfig as JsonNodeConfig, EdgeConfig, ConditionalEdgeConfig # Step 4.1에서 검토

# graph_nodes 임포트 (실제 프로젝트 구조에 맞게 필요시 수정)
from src.agents.graph_nodes.generic_llm_node import GenericLLMNode
from src.agents.graph_nodes.thought_generator_node import ThoughtGeneratorNode
from src.agents.graph_nodes.state_evaluator_node import StateEvaluatorNode
from src.agents.graph_nodes.search_strategy_node import SearchStrategyNode

logger = get_logger(__name__)
settings = get_settings()

# 사용 가능한 노드 타입과 해당 클래스 매핑
# 이 맵은 Orchestrator가 JSON 설정에서 node_type을 보고 실제 클래스를 찾는데 사용됩니다.
REGISTERED_NODE_TYPES: Dict[str, Type[Any]] = {
    "generic_llm_node": GenericLLMNode,
    "thought_generator_node": ThoughtGeneratorNode,
    "state_evaluator_node": StateEvaluatorNode,
    "search_strategy_node": SearchStrategyNode,
    # TODO: 추후 다른 노드 타입(예: planner_node, executor_node, tool_node) 추가 시 여기에 등록
}


class Orchestrator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self._compiled_graphs: Dict[str, CompiledGraph] = {} # 컴파일된 그래프 캐시
        logger.info("Orchestrator initialized.")

    def _load_graph_config_from_file(self, config_name: str) -> Dict[str, Any]:
        """지정된 이름의 그래프 설정을 JSON 파일에서 로드합니다."""
        config_dir = Path(getattr(settings, 'AGENT_GRAPH_CONFIG_DIR', 'config/agent_graphs'))
        config_file_path = config_dir / f"{config_name}.json"

        if not config_file_path.exists():
            logger.error(f"Graph configuration file not found: {config_file_path}")
            raise FileNotFoundError(f"Graph configuration file not found: {config_file_path}")

        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            # Pydantic 모델로 유효성 검사 (선택 사항이지만 권장)
            AgentGraphConfig.model_validate(config_data)
            logger.info(f"Successfully loaded and validated graph configuration: {config_name}")
            return config_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {config_file_path}: {e}")
            raise ValueError(f"Invalid JSON in graph configuration file: {config_file_path}") from e
        except Exception as e: # Pydantic ValidationError 등
            logger.error(f"Error validating graph configuration {config_file_path}: {e}")
            raise ValueError(f"Invalid graph configuration data in {config_file_path}") from e

    def _create_node_instance(self, node_config: JsonNodeConfig) -> Callable[[AgentGraphState], Dict[str, Any]]:
        """JSON 설정에서 노드 인스턴스를 생성합니다."""
        node_type_str = node_config.node_type
        node_params = node_config.parameters or {}
        node_id = node_config.id

        node_class = REGISTERED_NODE_TYPES.get(node_type_str)
        if not node_class:
            raise ValueError(f"Unsupported node_type: '{node_type_str}' in graph config for node ID '{node_id}'. "
                             f"Registered types are: {list(REGISTERED_NODE_TYPES.keys())}")

        try:
            # 모든 노드에 llm_client와 node_id를 기본적으로 전달 시도 (필요한 경우)
            # 노드 클래스 생성자가 이를 받도록 설계되어 있어야 함.
            constructor_params = {"llm_client": self.llm_client, "node_id": node_id, **node_params}
            
            # 실제 생성자에 있는 파라미터만 전달
            import inspect
            sig = inspect.signature(node_class.__init__)
            valid_params_for_constructor = {
                k: v for k, v in constructor_params.items() if k in sig.parameters
            }
            
            node_instance = node_class(**valid_params_for_constructor)
            logger.debug(f"Created instance for node ID '{node_id}', type '{node_type_str}' with params: {valid_params_for_constructor}")
            
            # LangGraph 노드는 호출 가능한 객체여야 함 (대부분 __call__ 메서드를 가짐)
            if not callable(node_instance):
                 raise TypeError(f"Node instance for '{node_id}' (type: {node_type_str}) is not callable.")
            return node_instance # __call__을 직접 반환할 필요 없음. 인스턴스 자체가 callable.

        except Exception as e:
            logger.error(f"Failed to create node instance for ID '{node_id}', type '{node_type_str}': {e}", exc_info=True)
            raise RuntimeError(f"Error creating node '{node_id}': {e}") from e
            
    def _get_conditional_router_func(
        self,
        condition_key: str,
        targets_map: Dict[str, str],
        default_decision: str = "finish",          # ← 인자 의미를 “결정 라벨”로 바꿈
    ) -> Callable[[AgentGraphState], str]:
        """
        LangGraph 규칙: router 함수는 **타깃 노드가 아니라
        decision 라벨**(= targets_map 의 key)을 반환해야 한다.
        """
        def router(state: AgentGraphState) -> str:
            # 1) 상태 속성 → 2) dynamic_data.some_key → 3) metadata
            if hasattr(state, condition_key):
                value = getattr(state, condition_key)
            elif condition_key.startswith("dynamic_data.") and "." in condition_key:
                _, child = condition_key.split(".", 1)
                value = state.dynamic_data.get(child)
            else:
                value = state.metadata.get(condition_key)

            # 특수 처리: value_is_(not_)none
            if "value_is_not_none" in targets_map and "value_is_none" in targets_map:
                return "value_is_not_none" if value is not None else "value_is_none"

            # 값이 문자열이고 그 라벨이 존재하면 그대로 반환
            if isinstance(value, str) and value in targets_map:
                return value

            # 그 밖에는 기본(decision) 사용
            return default_decision

        return router

    def build_graph(self, graph_config_dict: Dict[str, Any]) -> StateGraph:
        cfg = AgentGraphConfig.model_validate(graph_config_dict)
        sg = StateGraph(AgentGraphState)

        # 1. 노드
        for n in cfg.nodes:
            sg.add_node(n.id, self._create_node_instance(n))

        # 2. 엣지
        for e in cfg.edges:
            if e.type == "standard":
                tgt = END if e.target == "__end__" else e.target
                sg.add_edge(e.source, tgt)

            elif e.type == "conditional":
                ce = cast(ConditionalEdgeConfig, e)

                # node ID 변환
                converted = {k: (END if v == "__end__" else v) for k, v in ce.targets.items()}

                router = self._get_conditional_router_func(
                    condition_key=ce.condition_key,
                    targets_map=converted,
                    default_decision="finish",          # decision 라벨
                )
                sg.add_conditional_edges(ce.source, router, converted)

        # 3. 진입점
        sg.set_entry_point(cfg.entry_point)
        return sg

    def get_compiled_graph(self, graph_config_name: str) -> CompiledGraph:
        """컴파일된 그래프를 가져오거나, 없으면 빌드하고 컴파일하여 캐시합니다."""
        if graph_config_name not in self._compiled_graphs:
            logger.info(f"Compiled graph for '{graph_config_name}' not found in cache. Building...")
            graph_dict_config = self._load_graph_config_from_file(graph_config_name)
            state_graph = self.build_graph(graph_dict_config)
            self._compiled_graphs[graph_config_name] = state_graph.compile()
            logger.info(f"Graph '{graph_config_name}' compiled and cached.")
        else:
            logger.debug(f"Using cached compiled graph for '{graph_config_name}'.")
        return self._compiled_graphs[graph_config_name]

    async def run_workflow(
        self,
        graph_config_name: str,
        task_id: str,
        original_input: Any,
        initial_metadata: Optional[Dict[str, Any]] = None,
        max_iterations: int = 15 # ToT 루프 등의 최대 반복 횟수
    ) -> AgentGraphState:
        """
        지정된 그래프 설정을 사용하여 워크플로우를 실행합니다.

        Args:
            graph_config_name: config/agent_graphs/ 디렉토리의 JSON 파일 이름 (확장자 제외).
            task_id: 현재 실행 중인 작업의 ID.
            original_input: 워크플로우의 초기 입력. AgentGraphState.original_input에 저장됩니다.
            initial_metadata: AgentGraphState.metadata에 저장될 초기 메타데이터.
            max_iterations: 그래프 실행의 최대 반복 횟수 (무한 루프 방지).

        Returns:
            최종 AgentGraphState 객체.
        """
        logger.info(f"Running workflow '{graph_config_name}' for task_id '{task_id}'.")
        compiled_graph = self.get_compiled_graph(graph_config_name)

        initial_state = AgentGraphState(
            task_id=task_id,
            original_input=original_input,
            metadata=initial_metadata or {}
        )
        
        # 로깅 부분 개선
        try:
            # 표준 json 라이브러리 사용 (msgspec 대신)
            initial_state_dict = {}
            for field in initial_state.__annotations__:
                if hasattr(initial_state, field):
                    value = getattr(initial_state, field)
                    initial_state_dict[field] = value
                    
            pretty_json_for_log = json.dumps(initial_state_dict, default=str, indent=2)
            logger.debug(f"Initial state for workflow '{graph_config_name}': {pretty_json_for_log}")
        except Exception as log_e:
            logger.warning(f"Could not pretty print initial_state for logging: {log_e}")
            logger.debug(f"Initial state (raw struct) for workflow '{graph_config_name}': {str(initial_state)}")

        final_state_dict = None
        try:
            config = {"recursion_limit": max_iterations}
            
            # Dict 변환 (더 안전한 방법)
            initial_state_dict = {}
            for field in initial_state.__annotations__:
                if hasattr(initial_state, field):
                    value = getattr(initial_state, field)
                    initial_state_dict[field] = value
            
            final_state_dict = await compiled_graph.ainvoke(initial_state_dict, config=config)

        except Exception as e:
            logger.error(f"Error invoking graph '{graph_config_name}' for task {task_id}: {e}", exc_info=True)
            current_state_approx = AgentGraphState(
                task_id=task_id,
                original_input=original_input,
                metadata=initial_metadata or {},
                error_message=f"Workflow execution failed: {str(e)}"
            )
            return current_state_approx

        if final_state_dict:
            final_state_dict["error_message"] = None
            
            try:
                # 직접 JSON 직렬화 후 역직렬화 대신 직접 객체 생성
                final_state = AgentGraphState(
                    task_id=final_state_dict.get("task_id", task_id),
                    original_input=final_state_dict.get("original_input", original_input),
                    final_answer=final_state_dict.get("final_answer"),
                    error_message=None,
                    metadata=final_state_dict.get("metadata", {}),
                    thoughts=final_state_dict.get("thoughts", []),
                    max_search_depth=final_state_dict.get("max_search_depth", 3),
                    dynamic_data=final_state_dict.get("dynamic_data", {})
                )
            except Exception as e:
                logger.warning(f"Error creating final state object: {e}, using partial data")
                # 문제가 있어도 최소한의 정보는 반환
                final_state = AgentGraphState(
                    task_id=task_id,
                    original_input=original_input,
                    final_answer=final_state_dict.get("final_answer", "Workflow completed but couldn't parse full results"),
                    error_message=f"Completed with data conversion error: {str(e)}",
                    metadata=initial_metadata or {}
                )
            
            logger.info(f"Workflow '{graph_config_name}' for task {task_id} completed. {final_state.final_answer}")
            if final_state.error_message:
                logger.warning(f"Workflow '{graph_config_name}' for task {task_id} completed with error: {final_state.error_message}")
            return final_state
        else:
            logger.error(f"Workflow '{graph_config_name}' for task {task_id} did not return a final state dictionary after invocation.")
            current_state_approx = AgentGraphState(
                task_id=task_id,
                original_input=original_input,
                metadata=initial_metadata or {},
                error_message="Workflow did not produce a final state."
            )
            return current_state_approx

        
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
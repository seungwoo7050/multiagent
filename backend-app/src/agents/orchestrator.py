import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, cast, Union
import inspect
import msgspec

from fastapi import HTTPException
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from langchain_core.prompts import PromptTemplate


from src.config.settings import get_settings
from src.config.errors import ValidationError
from src.utils.logger import get_logger
from opentelemetry import trace

from src.services.llm_client import LLMClient
from src.memory.memory_manager import MemoryManager
from src.schemas.mcp_models import AgentGraphState
from src.schemas.agent_graph_config import (
  AgentGraphConfig,
  NodeConfig as JsonNodeConfig,
  EdgeConfig,
  ConditionalEdgeConfig,
)
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, FinalResultMessage
from src.services.tool_manager import ToolManager

from src.agents.graph_nodes.generic_llm_node import GenericLLMNode
from src.agents.graph_nodes.thought_generator_node import ThoughtGeneratorNode
from src.agents.graph_nodes.state_evaluator_node import StateEvaluatorNode
from src.agents.graph_nodes.search_strategy_node import SearchStrategyNode
from src.agents.graph_nodes.task_division_node import TaskDivisionNode
from src.agents.graph_nodes.task_complexity_evaluator_node import (
  TaskComplexityEvaluatorNode,
)
from src.agents.graph_nodes.subtask_processor_node import SubtaskProcessorNode
from src.agents.graph_nodes.synthesis_node import SynthesisNode
from src.agents.graph_nodes.task_complexity_router_node import TaskComplexityRouterNode
from src.agents.graph_nodes.direct_processor_node import DirectProcessorNode


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
  "result_evaluator_node": GenericLLMNode,
  "synthesis_node": SynthesisNode,
  "task_complexity_router_node": TaskComplexityRouterNode,
  "direct_processor_node": DirectProcessorNode,
}


class Orchestrator:
  def __init__(
    self,
    llm_client: LLMClient,
    tool_manager: ToolManager,
    memory_manager: MemoryManager,
    notification_service: NotificationService,
  ):
    if not isinstance(llm_client, LLMClient):
      raise TypeError("llm_client must be an instance of LLMClient")
    if not isinstance(tool_manager, ToolManager):
      raise TypeError("tool_manager must be an instance of ToolManager")
    if not isinstance(memory_manager, MemoryManager):
      raise TypeError("memory_manager must be an instance of MemoryManager")
    if not isinstance(notification_service, NotificationService):
      raise TypeError(
        "notification_service must be an instance of NotificationService"
      )

    self.llm_client = llm_client
    self.tool_manager = tool_manager
    self.memory_manager = memory_manager
    self.notification_service = notification_service
    self._compiled_graphs: Dict[str, CompiledGraph] = {}

    self.summarizer_prompt_path = "generic/conversation_summarizer.txt"
    self.summary_memory_key = "conversation_summary"
    self.summary_prompt_template_str: Optional[str] = None
    self._load_summarizer_prompt()

    logger.info(
      f"Orchestrator initialized with ToolManager: '{self.tool_manager.name}', MemoryManager, and Summarizer prompt: '{self.summarizer_prompt_path}'."
    )

  def _load_summarizer_prompt(self):
    """
    인스턴스 변수 self.summarizer_prompt_path에 지정된 프롬프트 템플릿을 로드하여
    self.summary_prompt_template_str에 저장합니다.
    """
    try:
      prompt_template_full_path = (
        Path(settings.PROMPT_TEMPLATE_DIR) / self.summarizer_prompt_path
      )
      if prompt_template_full_path.exists():
        with open(prompt_template_full_path, "r", encoding="utf-8") as f:
          self.summary_prompt_template_str = f.read()
        logger.info(
          f"Orchestrator: Successfully loaded summarizer prompt from: {prompt_template_full_path}"
        )
      else:
        logger.error(
          f"Orchestrator: Summarizer prompt file not found at {prompt_template_full_path}. Summary generation will fail if prompt is not set otherwise."
        )
        self.summary_prompt_template_str = None
    except Exception as e:
      logger.error(
        f"Orchestrator: Failed to load summarizer prompt from {self.summarizer_prompt_path}: {e}",
        exc_info=True,
      )
      self.summary_prompt_template_str = None

  async def _generate_conversation_summary(
    self,
    conversation_id: str,
    previous_summary: Optional[str],
    current_user_input: Any,
    current_agent_response: Optional[str],
  ) -> Optional[str]:
    """
    주어진 정보를 바탕으로 대화 요약을 생성합니다.
    """
    logger.debug(
      f"Orchestrator: Attempting to generate summary for conversation_id: {conversation_id}"
    )

    if not self.summary_prompt_template_str:
      logger.error(
        f"Orchestrator: Summarizer prompt template is not loaded. Cannot generate summary for {conversation_id}."
      )
      return previous_summary

    if current_agent_response is None or str(current_agent_response).strip() == "":
      logger.debug(
        f"Orchestrator: Agent response is empty for {conversation_id}, skipping summary generation. Returning previous summary."
      )
      return previous_summary

    try:
      prompt_input_values = {
        "previous_summary": previous_summary
        or "This is the beginning of the conversation.",
        "current_user_input": str(current_user_input),
        "current_agent_response": str(current_agent_response),
      }

      prompt = PromptTemplate(
        template=self.summary_prompt_template_str,
        input_variables=list(prompt_input_values.keys()),
      )
      formatted_prompt = prompt.format(**prompt_input_values)

      messages_for_llm = [{"role": "user", "content": formatted_prompt}]

      new_summary = await self.llm_client.generate_response(
        messages=messages_for_llm, temperature=0.2, max_tokens=300
      )
      generated_summary = new_summary.strip()
      logger.info(
        f"Orchestrator: New summary generated for {conversation_id}. Length: {len(generated_summary)}"
      )
      logger.debug(
        f"Orchestrator: Generated summary preview for {conversation_id}: {generated_summary[:100]}..."
      )
      return generated_summary
    except Exception as e:
      logger.error(
        f"Orchestrator: Failed to generate conversation summary for {conversation_id}: {e}",
        exc_info=True,
      )
      return previous_summary

  def _load_graph_config_from_file(self, config_name: str) -> Dict[str, Any]:
    try:
      graph_conf_dict = None

      config_path = Path(settings.AGENT_GRAPH_CONFIG_DIR) / f"{config_name}.json"

      if not config_path.exists():
        config_path_no_ext = Path(settings.AGENT_GRAPH_CONFIG_DIR) / config_name
        if config_path_no_ext.with_suffix(".json").exists():
          config_path = config_path_no_ext.with_suffix(".json")
        else:
          logger.error(
            f"Graph configuration file not found: {config_path} (and without .json extension)"
          )
          raise FileNotFoundError(
            f"Graph configuration not found: {config_name}"
          )

      logger.debug(f"Loading graph config directly from: {config_path}")
      with open(config_path, "r", encoding="utf-8") as f:
        graph_conf_dict = json.load(f)

      validated_config = AgentGraphConfig.model_validate(graph_conf_dict)
      logger.info(
        f"Successfully loaded and validated graph configuration: {config_name} from {config_path}"
      )
      return validated_config.model_dump()

    except FileNotFoundError as fnf:
      logger.error(
        f"Graph configuration file not found: {config_name} (Searched path: {config_path if 'config_path' in locals() else 'unknown'}) - Error: {fnf}"
      )
      raise
    except json.JSONDecodeError as jde:
      logger.error(
        f"JSON parsing error in graph config '{config_name}': {jde}",
        exc_info=True,
      )
      raise ValueError(
        f"Invalid JSON in graph configuration: {config_name}"
      ) from jde
    except ValidationError as ve:
      logger.error(
        f"Graph configuration validation failed for '{config_name}': {ve.errors()}",
        exc_info=True,
      )
      raise ValueError(
        f"Graph configuration validation error for '{config_name}': {ve}"
      ) from ve
    except Exception as e:
      logger.error(
        f"Unexpected error loading graph config '{config_name}': {e}",
        exc_info=True,
      )
      raise

  def _create_node_instance(
    self, node_config: JsonNodeConfig
  ) -> Callable[[AgentGraphState], Dict[str, Any]]:
    node_type_str = node_config.node_type
    node_params = node_config.parameters or {}
    node_id = node_config.id

    node_class = REGISTERED_NODE_TYPES.get(node_type_str)
    if not node_class:
      raise ValueError(
        f"Unsupported node_type: '{node_type_str}' for node ID '{node_id}'. Registered types: {list(REGISTERED_NODE_TYPES.keys())}"
      )

    try:
      constructor_params = {
        "llm_client": self.llm_client,
        "node_id": node_id,
      }
      sig_params = inspect.signature(node_class.__init__).parameters

      if "tool_manager" in sig_params and "tool_manager" not in node_params:
        constructor_params["tool_manager"] = self.tool_manager
        logger.debug(
          f"Injecting default ToolManager into node '{node_id}' (type: {node_type_str})."
        )

      if "memory_manager" in sig_params and "memory_manager" not in node_params:
        constructor_params["memory_manager"] = self.memory_manager
        logger.debug(
          f"Injecting default MemoryManager into node '{node_id}' (type: {node_type_str})."
        )

      if (
        "notification_service" in sig_params
        and "notification_service" not in node_params
      ):
        constructor_params["notification_service"] = self.notification_service
        logger.debug(
          f"Injecting default NotificationService into node '{node_id}' (type: {node_type_str})."
        )

      constructor_params.update(node_params)

      sig = inspect.signature(node_class.__init__)
      valid_params_for_constructor = {
        k: v for k, v in constructor_params.items() if k in sig.parameters
      }

      required_params_in_sig = {
        p.name
        for p in sig.parameters.values()
        if p.default == inspect.Parameter.empty and p.name != "self"
      }
      missing_params = required_params_in_sig - set(
        valid_params_for_constructor.keys()
      )
      if missing_params:
        critical_missing = missing_params - {
          "llm_client",
          "tool_manager",
          "memory_manager",
          "notification_service",
        }
        if critical_missing:
          raise TypeError(
            f"Node '{node_id}' (Type: {node_type_str}): Missing required constructor arguments: {critical_missing}. Check JSON parameters and node class constructor."
          )

      node_instance = node_class(**valid_params_for_constructor)
      logger.debug(
        f"Created instance for node ID '{node_id}', type '{node_type_str}' with params: {list(valid_params_for_constructor.keys())}"
      )

      if not callable(node_instance):
        raise TypeError(
          f"Node instance for '{node_id}' (type: {node_type_str}) is not callable."
        )
      return node_instance

    except TypeError as te:
      logger.error(
        f"TypeError creating node instance for ID '{node_id}', type '{node_type_str}': {te}. Check constructor arguments and JSON parameters.",
        exc_info=True,
      )
      raise RuntimeError(
        f"Error creating node '{node_id}' due to TypeError: {te}"
      ) from te
    except Exception as e:
      logger.error(
        f"Failed to create node instance for ID '{node_id}', type '{node_type_str}': {e}",
        exc_info=True,
      )
      raise RuntimeError(f"Error creating node '{node_id}': {e}") from e

  def _get_conditional_router_func(
    self,
    condition_key: str,
    targets_map: Dict[str, str],
    default_decision: str = END,
  ) -> Callable[[Union[AgentGraphState, Dict[str, Any]]], str]:
    def router(state: Union[AgentGraphState, Dict[str, Any]]) -> str:
      value_to_check = None
      is_dict = isinstance(state, dict)

      state_for_log = (
        str(state)[:200] + "..." if len(str(state)) > 200 else str(state)
      )
      logger.debug(
        f"[Router] Input state type: {type(state).__name__}, condition_key='{condition_key}', state_preview='{state_for_log}'"
      )

      if is_dict:
        state_dict = cast(Dict[str, Any], state)
        if condition_key in state_dict:
          value_to_check = state_dict[condition_key]
        elif condition_key.startswith("dynamic_data.") and "." in condition_key:
          key_path = condition_key.split(".")[1:]
          current_val = state_dict.get("dynamic_data")
          if isinstance(current_val, dict):
            try:
              for k_part in key_path:
                current_val = (
                  current_val.get(k_part)
                  if isinstance(current_val, dict)
                  else None
                )
              value_to_check = current_val
            except Exception:
              value_to_check = None
          else:
            value_to_check = None
        elif condition_key.startswith("metadata.") and "." in condition_key:
          key_path = condition_key.split(".")[1:]
          current_val = state_dict.get("metadata")
          if isinstance(current_val, dict):
            try:
              for k_part in key_path:
                current_val = (
                  current_val.get(k_part)
                  if isinstance(current_val, dict)
                  else None
                )
              value_to_check = current_val
            except Exception:
              value_to_check = None
          else:
            value_to_check = None

      else:
        state_obj = cast(AgentGraphState, state)
        if hasattr(state_obj, condition_key):
          value_to_check = getattr(state_obj, condition_key)
        elif (
          condition_key.startswith("dynamic_data.")
          and "." in condition_key
          and isinstance(state_obj.dynamic_data, dict)
        ):
          key_path = condition_key.split(".")[1:]
          current_val = state_obj.dynamic_data
          try:
            for k_part in key_path:
              current_val = (
                current_val.get(k_part)
                if isinstance(current_val, dict)
                else None
              )
            value_to_check = current_val
          except Exception:
            value_to_check = None
        elif (
          condition_key.startswith("metadata.")
          and "." in condition_key
          and isinstance(state_obj.metadata, dict)
        ):
          key_path = condition_key.split(".")[1:]
          current_val = state_obj.metadata
          try:
            for k_part in key_path:
              current_val = (
                current_val.get(k_part)
                if isinstance(current_val, dict)
                else None
              )
            value_to_check = current_val
          except Exception:
            value_to_check = None

      logger.debug(
        f"[Router] Condition key='{condition_key}', Value to check='{value_to_check}' (type: {type(value_to_check).__name__})"
      )

      decision = default_decision
      str_value_to_check = str(value_to_check).lower()

      normalized_targets_map = {str(k).lower(): v for k, v in targets_map.items()}

      if str_value_to_check in normalized_targets_map:
        decision = normalized_targets_map[str_value_to_check]
        logger.debug(
          f"Router decision based on value '{str_value_to_check}': '{decision}'"
        )
        return decision

      if "value_is_not_none" in targets_map and value_to_check is not None:
        decision = targets_map["value_is_not_none"]
        logger.debug(
          f"Router decision based on 'value_is_not_none': '{decision}'"
        )
        return decision
      if "value_is_none" in targets_map and value_to_check is None:
        decision = targets_map["value_is_none"]
        logger.debug(f"Router decision based on 'value_is_none': '{decision}'")
        return decision

      if isinstance(value_to_check, bool):
        bool_str_val = str(value_to_check).lower()
        if bool_str_val in normalized_targets_map:
          decision = normalized_targets_map[bool_str_val]
          logger.debug(
            f"Router decision based on boolean value '{bool_str_val}': '{decision}'"
          )
          return decision

      logger.debug(
        f"No specific condition met for value '{value_to_check}'. Using default decision: '{default_decision}'"
      )
      return default_decision

    return router

  def build_graph(self, graph_config_dict: Dict[str, Any]) -> StateGraph:
    cfg = AgentGraphConfig.model_validate(graph_config_dict)
    graph = StateGraph(AgentGraphState)

    for node_config_model in cfg.nodes:
      try:
        node_instance = self._create_node_instance(node_config_model)
        graph.add_node(node_config_model.id, node_instance)
      except Exception as node_creation_err:
        logger.error(
          f"Fatal error creating node '{node_config_model.id}' for graph '{cfg.name}'. Halting graph build.",
          exc_info=True,
        )
        raise RuntimeError(
          f"Failed to build graph '{cfg.name}': Error creating node '{node_config_model.id}'."
        ) from node_creation_err

    for edge_config_model in cfg.edges:
      try:
        if edge_config_model.type == "standard":
          std_edge_cfg = cast(EdgeConfig, edge_config_model)
          target_node = (
            END if std_edge_cfg.target == "__end__" else std_edge_cfg.target
          )
          graph.add_edge(std_edge_cfg.source, target_node)
          logger.debug(
            f"[Graph-Edge] {std_edge_cfg.source} → {target_node} (standard)"
          )
        elif edge_config_model.type == "conditional":
          cond_edge_cfg = cast(ConditionalEdgeConfig, edge_config_model)
          target_map = {
            k: (END if v == "__end__" else v)
            for k, v in cond_edge_cfg.targets.items()
          }
          default_target = (
            END
            if cond_edge_cfg.default_target == "__end__"
            else (cond_edge_cfg.default_target or END)
          )

          router_func = self._get_conditional_router_func(
            cond_edge_cfg.condition_key, target_map, default_target
          )
          possible_nodes = list(set(target_map.values()))
          if default_target and default_target not in possible_nodes:
            possible_nodes.append(default_target)

          graph.add_conditional_edges(
            cond_edge_cfg.source, router_func, possible_nodes
          )
          logger.debug(
            f"[Graph-CondEdge] {cond_edge_cfg.source} → based on '{cond_edge_cfg.condition_key}'. Targets: {possible_nodes}"
          )
      except Exception as edge_creation_err:
        logger.error(
          f"Fatal error adding edge (Source: {edge_config_model.source}) for graph '{cfg.name}'. Halting graph build.",
          exc_info=True,
        )
        raise RuntimeError(
          f"Failed to build graph '{cfg.name}': Error adding edge from '{edge_config_model.source}'."
        ) from edge_creation_err
    try:
      graph.set_entry_point(cfg.entry_point)
    except Exception as entry_point_err:
      logger.error(
        f"Fatal error setting entry point '{cfg.entry_point}' for graph '{cfg.name}'. Halting graph build.",
        exc_info=True,
      )
      raise RuntimeError(
        f"Failed to build graph '{cfg.name}': Error setting entry point '{cfg.entry_point}'."
      ) from entry_point_err

    logger.info(f"Successfully built StateGraph for configuration: {cfg.name}")
    return graph

  def get_compiled_graph(self, graph_config_name: str) -> CompiledGraph:
    if graph_config_name not in self._compiled_graphs:
      logger.info(
        f"Compiled graph for '{graph_config_name}' not found in cache. Building..."
      )
      try:
        graph_dict_config = self._load_graph_config_from_file(graph_config_name)
        state_graph = self.build_graph(graph_dict_config)
        compiled_graph = state_graph.compile()
        self._compiled_graphs[graph_config_name] = compiled_graph
        logger.info(f"Graph '{graph_config_name}' compiled and cached.")
      except Exception as build_compile_err:
        logger.error(
          f"Failed to build or compile graph '{graph_config_name}': {build_compile_err}",
          exc_info=True,
        )
        raise RuntimeError(
          f"Failed to get compiled graph for '{graph_config_name}'"
        ) from build_compile_err
    else:
      logger.debug(f"Using cached compiled graph for '{graph_config_name}'.")
    return self._compiled_graphs[graph_config_name]

  async def run_workflow(
    self,
    graph_config_name: str,
    task_id: str,
    original_input: Any,
    initial_metadata: Optional[Dict[str, Any]] = None,
    max_iterations: int = 100,
  ) -> AgentGraphState:
    with tracer.start_as_current_span(
      "orchestrator.run_workflow",
      attributes={"graph_config": graph_config_name, "task_id": task_id},
    ) as current_span:
      logger.info(
        f"Orchestrator: Running workflow '{graph_config_name}' for task_id '{task_id}'."
      )
      current_span.set_attribute("app.workflow.name", graph_config_name)

      await self.notification_service.broadcast_to_task(
        task_id,
        StatusUpdateMessage(
          task_id=task_id,
          status="pending",
          detail=f"Workflow '{graph_config_name}' starting.",
        ),
      )

      conversation_id: Optional[str] = None
      if initial_metadata and isinstance(
        initial_metadata.get("conversation_id"), str
      ):
        conversation_id = initial_metadata["conversation_id"]
        logger.info(
          f"Orchestrator: conversation_id '{conversation_id}' found in initial_metadata for task {task_id}."
        )
        current_span.set_attribute("app.conversation_id", conversation_id)

      retrieved_summary: Optional[str] = None
      if conversation_id:
        try:
          retrieved_summary = await self.memory_manager.load_state(
            context_id=conversation_id, key=self.summary_memory_key
          )
          if retrieved_summary:
            logger.info(
              f"Orchestrator: Retrieved previous summary for conversation_id '{conversation_id}'. Length: {len(retrieved_summary)}"
            )
            current_span.set_attribute("app.summary.loaded", True)
            current_span.set_attribute(
              "app.summary.length", len(retrieved_summary)
            )
          else:
            logger.info(
              f"Orchestrator: No previous summary found for conversation_id '{conversation_id}'. This might be the first turn."
            )
            current_span.set_attribute("app.summary.loaded", False)
        except Exception as e:
          logger.error(
            f"Orchestrator: Failed to load summary for conversation_id '{conversation_id}': {e}",
            exc_info=True,
          )
          current_span.record_exception(e)

      try:
        compiled_graph = self.get_compiled_graph(graph_config_name)
      except FileNotFoundError as exc:
        logger.error(
          f"Orchestrator: Graph configuration file '{graph_config_name}.json' not found for task {task_id}.",
          exc_info=True,
        )
        await self.notification_service.broadcast_to_task(
          task_id,
          FinalResultMessage(
            task_id=task_id,
            final_answer=None,
            error_message=f"Workflow configuration '{graph_config_name}' not found.",
          ),
        )
        current_span.set_status(
          trace.Status(
            trace.StatusCode.ERROR,
            description=f"Graph config not found: {exc}",
          )
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
      except Exception as graph_err:
        logger.error(
          f"Orchestrator: Cannot run workflow: Failed to get compiled graph '{graph_config_name}' for task {task_id}: {graph_err}",
          exc_info=True,
        )
        error_detail = f"Failed to load/compile workflow graph '{graph_config_name}': {graph_err}"
        await self.notification_service.broadcast_to_task(
          task_id,
          FinalResultMessage(
            task_id=task_id, final_answer=None, error_message=error_detail
          ),
        )
        current_span.set_status(
          trace.Status(
            trace.StatusCode.ERROR,
            description=f"Graph compilation error: {graph_err}",
          )
        )
        return AgentGraphState(
          task_id=task_id,
          original_input=original_input,
          metadata=initial_metadata or {},
          error_message=error_detail,
        )

      initial_agent_state_obj = AgentGraphState(
        task_id=task_id,
        original_input=original_input,
        metadata=initial_metadata or {},
        max_search_depth=(initial_metadata or {}).get(
          "max_search_depth",
          settings.TOT_MAX_DEPTH if hasattr(settings, "TOT_MAX_DEPTH") else 5,
        ),
        dynamic_data={},
      )

      if retrieved_summary:
        initial_agent_state_obj.dynamic_data["conversation_summary"] = (
          retrieved_summary
        )
        logger.debug(
          f"Orchestrator: Injected retrieved summary into initial AgentGraphState.dynamic_data for task {task_id}"
        )

      try:
        initial_state_dict = msgspec.to_builtins(initial_agent_state_obj)
        if not isinstance(initial_state_dict, dict):
          logger.error(
            f"Orchestrator: msgspec.to_builtins did not return a dict for AgentGraphState. Got {type(initial_state_dict)}. Fallback to model_dump."
          )
          initial_state_dict = initial_agent_state_obj.model_dump(mode="json")

      except Exception as dump_err:
        logger.error(
          f"Orchestrator: Failed to convert initial AgentGraphState to dict for workflow '{graph_config_name}' (task {task_id}): {dump_err}",
          exc_info=True,
        )
        error_detail = f"Failed to prepare initial state: {dump_err}"
        await self.notification_service.broadcast_to_task(
          task_id,
          FinalResultMessage(
            task_id=task_id, final_answer=None, error_message=error_detail
          ),
        )
        current_span.set_status(
          trace.Status(
            trace.StatusCode.ERROR,
            description=f"Initial state conversion error: {dump_err}",
          )
        )
        return AgentGraphState(
          task_id=task_id,
          original_input=original_input,
          metadata=initial_metadata or {},
          error_message=error_detail,
        )

      logger.debug(
        f"Orchestrator: Initial state dict for workflow '{graph_config_name}' (task {task_id}): {str(initial_state_dict)[:500]}..."
      )

      await self.notification_service.broadcast_to_task(
        task_id,
        StatusUpdateMessage(
          task_id=task_id,
          status="running",
          detail="Workflow execution in progress.",
        ),
      )

      final_state_obj: Optional[AgentGraphState] = None

      try:
        graph_json_config = self._load_graph_config_from_file(graph_config_name)
        config_from_json = graph_json_config.get("config", {})

        recursion_limit = config_from_json.get(
          "recursion_limit", max_iterations
        )

        current_span.set_attribute(
          "app.workflow.recursion_limit", recursion_limit
        )
        invoke_config = {"recursion_limit": recursion_limit}
        logger.debug(
          f"Orchestrator: Using recursion_limit={recursion_limit} "
          f"for workflow '{graph_config_name}' (task {task_id})"
        )

        final_state_dict_from_graph = await compiled_graph.ainvoke(
          initial_state_dict, config=invoke_config
        )

        if final_state_dict_from_graph and isinstance(
          final_state_dict_from_graph, dict
        ):
          try:
            final_state_obj = msgspec.convert(
              final_state_dict_from_graph, AgentGraphState, strict=False
            )
            logger.info(
              f"Orchestrator: Workflow '{graph_config_name}' for task {task_id} completed. Final Answer: {final_state_obj.final_answer or 'N/A'}"
            )
            current_span.set_attribute(
              "app.workflow.final_answer_present",
              bool(final_state_obj.final_answer),
            )
            if final_state_obj.error_message:
              current_span.set_attribute(
                "app.workflow.error", final_state_obj.error_message
              )
              current_span.set_status(
                trace.Status(
                  trace.StatusCode.ERROR,
                  description=final_state_obj.error_message,
                )
              )

          except Exception as convert_err:
            logger.error(
              f"Orchestrator: Error converting final state dictionary from graph to AgentGraphState for task {task_id}: {convert_err}",
              exc_info=True,
            )
            error_detail = f"Final state conversion error: {convert_err}. Raw final state: {str(final_state_dict_from_graph)[:200]}..."
            current_span.set_status(
              trace.Status(
                trace.StatusCode.ERROR, description=error_detail
              )
            )

            final_state_obj = AgentGraphState(
              task_id=task_id,
              original_input=original_input,
              metadata=final_state_dict_from_graph.get(
                "metadata", initial_metadata or {}
              ),
              final_answer=final_state_dict_from_graph.get(
                "final_answer",
                "Workflow completed but final state conversion failed.",
              ),
              error_message=error_detail,
              dynamic_data=final_state_dict_from_graph.get(
                "dynamic_data", {}
              ),
            )
        else:
          logger.error(
            f"Orchestrator: Workflow '{graph_config_name}' for task {task_id} invocation returned unexpected type: {type(final_state_dict_from_graph).__name__} or None."
          )
          error_detail = (
            "Workflow execution finished but returned invalid final state."
          )
          current_span.set_status(
            trace.Status(trace.StatusCode.ERROR, description=error_detail)
          )
          final_state_obj = AgentGraphState(
            task_id=task_id,
            original_input=original_input,
            metadata=initial_metadata or {},
            error_message=error_detail,
            dynamic_data=initial_state_dict.get("dynamic_data", {}),
          )

      except Exception as invoke_err:
        logger.error(
          f"Orchestrator: Error invoking graph '{graph_config_name}' for task {task_id}: {invoke_err}",
          exc_info=True,
        )
        error_detail = f"Workflow execution failed: {str(invoke_err)}"
        current_span.set_status(
          trace.Status(
            trace.StatusCode.ERROR,
            description=f"Graph invocation error: {invoke_err}",
          )
        )
        current_span.record_exception(invoke_err)
        final_state_obj = AgentGraphState(
          task_id=task_id,
          original_input=original_input,
          metadata=initial_metadata or {},
          error_message=error_detail,
          dynamic_data=initial_state_dict.get("dynamic_data", {}),
        )

      if conversation_id and final_state_obj:
        current_span.add_event("Attempting conversation summarization.")
        new_generated_summary = await self._generate_conversation_summary(
          conversation_id=conversation_id,
          previous_summary=retrieved_summary,
          current_user_input=original_input,
          current_agent_response=final_state_obj.final_answer,
        )
        if new_generated_summary and new_generated_summary != retrieved_summary:
          try:
            await self.memory_manager.save_state(
              context_id=conversation_id,
              key=self.summary_memory_key,
              value=new_generated_summary,
              ttl=settings.MEMORY_TTL * 7
              if settings.MEMORY_TTL
              else None,
            )
            logger.info(
              f"Orchestrator: Saved new summary for conversation_id '{conversation_id}'. Length: {len(new_generated_summary)}"
            )
            current_span.set_attribute("app.summary.saved", True)
            current_span.set_attribute(
              "app.summary.new_length", len(new_generated_summary)
            )
          except Exception as e:
            logger.error(
              f"Orchestrator: Failed to save new summary for conversation_id '{conversation_id}': {e}",
              exc_info=True,
            )
            current_span.record_exception(e)
            current_span.set_attribute("app.summary.save_error", str(e))

      if final_state_obj:
        await self.notification_service.broadcast_to_task(
          task_id,
          FinalResultMessage(
            task_id=task_id,
            final_answer=final_state_obj.final_answer,
            error_message=final_state_obj.error_message,
            metadata=final_state_obj.metadata,
          ),
        )

        if final_state_obj.error_message:
          current_span.set_status(
            trace.Status(
              trace.StatusCode.ERROR,
              description=final_state_obj.error_message,
            )
          )
        else:
          current_span.set_status(trace.Status(trace.StatusCode.OK))
        return final_state_obj
      else:
        critical_error = "Critical error: Final AgentGraphState object was not created after workflow execution."
        logger.critical(f"Orchestrator: Task {task_id} - {critical_error}")
        await self.notification_service.broadcast_to_task(
          task_id,
          FinalResultMessage(
            task_id=task_id, final_answer=None, error_message=critical_error
          ),
        )
        current_span.set_status(
          trace.Status(trace.StatusCode.ERROR, description=critical_error)
        )
        return AgentGraphState(
          task_id=task_id,
          original_input=original_input,
          metadata=initial_metadata or {},
          error_message=critical_error,
        )

  def _get_summary_storage_key(self, conversation_id: str) -> str:
    """대화 요약 저장을 위한 일관된 키를 생성합니다."""

    return f"{self.summary_memory_key}"

import os
import json
import re
from typing import Any, Dict, List, Optional, Union, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.config.errors import ToolError
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager
from src.memory.memory_manager import MemoryManager
from src.schemas.mcp_models import AgentGraphState, ConversationTurn
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


logger = get_logger(__name__)
settings = get_settings()


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
        history_prompt_key: Optional[str] = "conversation_history",
        summary_prompt_key: Optional[str] = "conversation_summary",
        history_key_prefix: Optional[str] = "chat_history",
        max_history_items: Optional[int] = 10,
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

        self.summary_prompt_key = summary_prompt_key

        self.history_key_prefix = history_key_prefix
        self.max_history_items = max_history_items

        self.prompt_template_str = self._load_prompt_template()
        prompt_vars_set = set(self.input_keys_for_prompt)

        if self.enable_tool_use:
            prompt_vars_set.update(
                ["available_tools", "tool_call_history", "scratchpad"]
            )
        if self.history_prompt_key:
            prompt_vars_set.add(self.history_prompt_key)

        if self.summary_prompt_key:
            prompt_vars_set.add(self.summary_prompt_key)

        self.prompt_template = PromptTemplate(
            template=self.prompt_template_str, input_variables=list(prompt_vars_set)
        )
        logger.info(
            f"GenericLLMNode '{self.node_id}' initialized. Tool use: {self.enable_tool_use}. "
            f"History key: '{self.history_prompt_key}', Summary key: '{self.summary_prompt_key}', Max hist items: {self.max_history_items}. Prompt: {prompt_template_path}. "
            f"NotificationService injected: {'Yes' if notification_service else 'No'}"
        )

    def _load_prompt_template(self) -> str:
        base_prompt_dir = getattr(settings, "PROMPT_TEMPLATE_DIR", "config/prompts")
        if os.path.isabs(self.prompt_template_path):
            full_path = self.prompt_template_path
        else:
            full_path = os.path.join(base_prompt_dir, self.prompt_template_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {full_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt template from {full_path}: {e}")
            raise

    def _get_available_tools_for_prompt(self) -> str:
        if not self.enable_tool_use:
            return "Tool use is disabled for this node."

        available_tool_names_set = self.tool_manager.get_names()
        tools_to_list_names: List[str] = []

        if self.allowed_tools is not None:
            allowed_and_available = available_tool_names_set.intersection(
                self.allowed_tools
            )
            if not allowed_and_available:
                return "No tools available (allowed list is empty or no matches found)."
            tools_to_list_names = list(allowed_and_available)
        else:
            tools_to_list_names = list(available_tool_names_set)

        if not tools_to_list_names:
            return "No tools available."

        tool_summaries = self.tool_manager.get_tool_summaries_for_llm()

        filtered_summaries = [
            s for s in tool_summaries if s.get("name") in tools_to_list_names
        ]

        if not filtered_summaries:
            return "No details found for available tools."

        prompt_lines = ["Available Tools:"]
        for summary_dict in filtered_summaries:
            prompt_lines.append(
                f"- {summary_dict.get('name', 'Unknown Tool')}: {summary_dict.get('description', 'No description.')}"
            )
        return "\n".join(prompt_lines)

    async def _load_conversation_history(self, state: AgentGraphState) -> str:
        if not self.history_prompt_key or not self.history_key_prefix:
            return "No conversation history configured for retrieval."

        try:
            context_id = state.task_id
            if (
                hasattr(state, "metadata")
                and isinstance(state.metadata, dict)
                and state.metadata.get("conversation_id")
            ):
                context_id = state.metadata["conversation_id"]

            if not context_id:
                logger.warning(
                    f"Node '{self.node_id}': context_id (task_id or conversation_id) not found in state for history retrieval."
                )
                return "History retrieval skipped: context_id is missing."

            history_items: List[
                Union[ConversationTurn, Dict[str, Any]]
            ] = await self.memory_manager.get_history(
                context_id=context_id,
                history_key_prefix=self.history_key_prefix,
                limit=self.max_history_items,
            )

            if not history_items:
                return "No conversation history found."

            formatted_history = []
            for item in history_items:
                if isinstance(item, ConversationTurn):
                    role = getattr(item, "role", "Unknown").capitalize()
                    content = getattr(item, "content", "")
                    formatted_history.append(f"{role}: {content}")
                elif isinstance(item, dict):
                    role = item.get("role", "Unknown").capitalize()
                    content = item.get("content", "")
                    formatted_history.append(f"{role}: {content}")
                else:
                    logger.warning(
                        f"Node '{self.node_id}': Encountered unknown history item type: {type(item)}. Converting to string."
                    )
                    formatted_history.append(str(item))

            return "\n".join(formatted_history)

        except Exception as e:
            logger.error(
                f"Node '{self.node_id}': Failed to load conversation history: {e}",
                exc_info=True,
            )
            return f"Error loading conversation history: {e}"

    async def _prepare_prompt_input(self, state: AgentGraphState) -> Dict[str, Any]:
        prompt_input: Dict[str, Any] = {}
        all_expected_vars = self.prompt_template.input_variables

        special_mappings = {
            "subtask_answer": ["dynamic_data.current_subtask.final_answer"],
            "subtask_description": ["dynamic_data.current_subtask.description"],
            "score_threshold": [],
        }

        for key in all_expected_vars:
            value: Any = None

            if key in self.__dict__:
                value = self.__dict__[key]
                logger.debug(
                    f"Node '{self.node_id}': Using parameter value for '{key}'"
                )

            elif hasattr(state, key):
                value = getattr(state, key)

            elif key in special_mappings:
                for path in special_mappings[key]:
                    if value is not None:
                        break

                    if "." in path:
                        parts = path.split(".")

                        if parts[0] == "dynamic_data" and isinstance(
                            state.dynamic_data, dict
                        ):
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
                                logger.debug(
                                    f"Node '{self.node_id}': Found '{key}' via special mapping path '{path}'"
                                )

            elif "." in key:
                parts = key.split(".")

                if parts[0] == "dynamic_data" and isinstance(state.dynamic_data, dict):
                    current_val = state.dynamic_data
                    path_to_value = parts[1:]

                elif parts[0] == "metadata" and isinstance(state.metadata, dict):
                    current_val = state.metadata
                    path_to_value = parts[1:]

                elif hasattr(state, parts[0]):
                    current_val = getattr(state, parts[0])
                    path_to_value = parts[1:]
                else:
                    current_val = None
                    path_to_value = []

                for p_key in path_to_value:
                    if isinstance(current_val, dict) and p_key in current_val:
                        current_val = current_val[p_key]
                    elif hasattr(current_val, p_key):
                        current_val = getattr(current_val, p_key)
                    else:
                        logger.debug(
                            f"Node '{self.node_id}': Path '{key}' could not be fully resolved. Part '{p_key}' not found in {type(current_val).__name__}."
                        )
                        current_val = None
                        break
                value = current_val

            elif isinstance(state.dynamic_data, dict) and key in state.dynamic_data:
                value = state.dynamic_data[key]

            elif isinstance(state.metadata, dict) and key in state.metadata:
                value = state.metadata[key]

            if key == self.history_prompt_key and self.history_key_prefix:
                value = await self._load_conversation_history(state)

            elif key == self.summary_prompt_key and self.summary_prompt_key:
                if state.dynamic_data and isinstance(
                    state.dynamic_data.get(self.summary_prompt_key), str
                ):
                    value = state.dynamic_data[self.summary_prompt_key]
                    logger.debug(
                        f"Node '{self.node_id}': Using '{self.summary_prompt_key}' from dynamic_data for prompt."
                    )
                else:
                    value = "No conversation summary available for this turn."
                    logger.debug(
                        f"Node '{self.node_id}': '{self.summary_prompt_key}' not found in dynamic_data or not a string. Using default."
                    )

            elif self.enable_tool_use:
                if key == "available_tools":
                    value = self._get_available_tools_for_prompt()
                elif key == "tool_call_history":
                    history_list = (
                        state.tool_call_history
                        if state.tool_call_history is not None
                        else []
                    )

                    value = (
                        "\n".join(
                            f"Tool: {c.get('tool_name', 'UnknownTool')}, Args: {json.dumps(c.get('args', {}))}, Result: {str(c.get('result', 'No result'))[:200]}"
                            for c in history_list
                            if isinstance(c, dict)
                        )
                        if history_list
                        else "No tool calls yet."
                    )
                elif key == "scratchpad":
                    value = state.scratchpad if state.scratchpad is not None else ""

            if value is None:
                if (
                    key in self.input_keys_for_prompt
                    or (key == self.history_prompt_key and self.history_prompt_key)
                    or (key == self.summary_prompt_key and self.summary_prompt_key)
                ):
                    logger.warning(
                        f"Node '{self.node_id}': Key '{key}' required for prompt (from input_keys, history, or summary) was not found or resolved to None; using empty string."
                    )
                prompt_input[key] = ""
            elif isinstance(value, (list, dict)) and key != "messages":
                try:
                    prompt_input[key] = json.dumps(
                        value, indent=2, ensure_ascii=False, default=str
                    )
                except TypeError:
                    logger.warning(
                        f"Node '{self.node_id}': Could not JSON serialize value for key '{key}' (type: {type(value).__name__}). Using str()."
                    )
                    prompt_input[key] = str(value)
            else:
                prompt_input[key] = str(value)

        logger.debug(
            f"Node '{self.node_id}': Prepared prompt input keys: {list(prompt_input.keys())}"
        )
        return prompt_input

    def _parse_llm_response(self, response_str: str) -> Optional[ParsedLLMResponse]:
        response_str = response_str.strip()
        logger.debug(
            f"Node '{self.node_id}': Parsing LLM response: {response_str[:300]}..."
        )

        try:
            if response_str.startswith("```json") and response_str.endswith("```"):
                response_str = response_str[7:-3].strip()
            elif response_str.startswith("```") and response_str.endswith("```"):
                response_str = response_str[3:-3].strip()

            parsed_json = json.loads(response_str)
            if (
                isinstance(parsed_json, dict)
                and "action" in parsed_json
                and "action_input" in parsed_json
            ):
                action_val = parsed_json["action"]
                action_input_val = parsed_json["action_input"]
                logger.info(
                    f"Node '{self.node_id}': Successfully parsed JSON response. Action: {action_val}"
                )
                return ParsedLLMResponse(
                    action=str(action_val), action_input=action_input_val
                )
            else:
                logger.warning(
                    f"Node '{self.node_id}': Parsed JSON lacks 'action' or 'action_input' keys. Content: {str(parsed_json)[:200]}"
                )
                return None

        except json.JSONDecodeError:
            logger.debug(
                f"Node '{self.node_id}': Response is not valid JSON. Attempting ReAct style text parsing."
            )

            action_match = re.search(r"Action:\s*([^\n]+)", response_str, re.IGNORECASE)
            action_input_match = re.search(
                r"Action Input:\s*([\s\S]+)", response_str, re.IGNORECASE | re.DOTALL
            )

            if action_match:
                action_val = action_match.group(1).strip()
                action_input_str = (
                    action_input_match.group(1).strip() if action_input_match else "{}"
                )

                action_input_parsed: Union[Dict, str]
                try:
                    action_input_parsed = json.loads(action_input_str)
                    if not isinstance(action_input_parsed, dict):
                        action_input_parsed = {"input": action_input_str}
                except json.JSONDecodeError:
                    action_input_parsed = action_input_str

                logger.info(
                    f"Node '{self.node_id}': Successfully parsed ReAct style text response (Action: {action_val}). Input type: {type(action_input_parsed).__name__}"
                )
                return ParsedLLMResponse(
                    action=action_val, action_input=action_input_parsed
                )

            logger.error(
                f"Node '{self.node_id}': Failed to parse LLM response using known formats. Response preview: {response_str[:500]}..."
            )
            return None

    async def __call__(
        self, state: AgentGraphState
    ) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "graph.node.generic_llm",
            attributes={
                "node_id": self.node_id,
                "task_id": state.task_id,
                "model": self.model_name or "default_from_llm_client",
                "enable_tool_use": self.enable_tool_use,
            },
        ) as current_node_span:
            logger.info(
                f"GenericLLMNode '{self.node_id}' execution started. Task ID: {state.task_id}. Tool use enabled: {self.enable_tool_use}"
            )
            current_node_span.set_attribute("app.node.id", self.node_id)

            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id,
                    status="node_executing",
                    detail=f"Node '{self.node_id}' started.",
                    current_node=self.node_id,
                ),
            )

            if not self.enable_tool_use:
                logger.debug(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Executing simple LLM call (tool use disabled)."
                )
                final_update_dict: Dict[str, Any] = {}
                error_for_ws: Optional[str] = None

                try:
                    prompt_input_values = await self._prepare_prompt_input(state)
                    formatted_prompt = self.prompt_template.format(
                        **prompt_input_values
                    )
                    current_node_span.set_attribute(
                        "app.llm.prompt_length", len(formatted_prompt)
                    )
                    logger.debug(
                        f"Node '{self.node_id}' (Task: {state.task_id}): Formatted prompt (tools disabled):\n{formatted_prompt[:500]}..."
                    )

                    llm_params_for_call: Dict[str, Any] = {}
                    if self.temperature is not None:
                        llm_params_for_call["temperature"] = self.temperature
                    if self.max_tokens is not None:
                        llm_params_for_call["max_tokens"] = self.max_tokens

                    messages_for_llm = [{"role": "user", "content": formatted_prompt}]
                    llm_response_str = await self.llm_client.generate_response(
                        messages=messages_for_llm,
                        model_name=self.model_name,
                        **llm_params_for_call,
                    )
                    current_node_span.set_attribute(
                        "app.llm.response_length", len(llm_response_str)
                    )
                    logger.debug(
                        f"Node '{self.node_id}' (Task: {state.task_id}): LLM raw response (tools disabled): {llm_response_str[:200]}..."
                    )

                    final_update_dict = {
                        "error_message": None,
                        "last_llm_input": formatted_prompt,
                        "last_llm_output": llm_response_str,
                        "dynamic_data": state.dynamic_data.copy()
                        if state.dynamic_data
                        else {},
                    }

                    if (
                        state.dynamic_data
                        and "current_subtask" in state.dynamic_data
                        and self.node_id == "initial_responder_subtask"
                    ):
                        final_update_dict["subtask_description"] = state.dynamic_data[
                            "current_subtask"
                        ].get("description", "")
                        final_update_dict["subtask_answer"] = llm_response_str

                    output_key = self.output_field_name or "final_answer"

                    if "." in output_key:
                        parts = output_key.split(".")
                        parent_key = parts[0]
                        if parent_key == "dynamic_data":
                            current_dict_level = final_update_dict["dynamic_data"]
                            for part_idx, part_key in enumerate(parts[1:-1]):
                                if part_key not in current_dict_level or not isinstance(
                                    current_dict_level[part_key], dict
                                ):
                                    current_dict_level[part_key] = {}
                                current_dict_level = current_dict_level[part_key]
                            current_dict_level[parts[-1]] = llm_response_str
                        else:
                            logger.warning(
                                f"Node '{self.node_id}': Cannot set output to non-dynamic_data nested field '{output_key}'. Storing in 'final_answer'."
                            )
                            final_update_dict["final_answer"] = llm_response_str
                    else:
                        final_update_dict[output_key] = llm_response_str

                    final_update_dict["dynamic_data"]["last_llm_input"] = (
                        formatted_prompt
                    )
                    final_update_dict["dynamic_data"]["last_llm_output"] = (
                        llm_response_str
                    )

                except Exception as e:
                    logger.error(
                        f"Node '{self.node_id}' (Task: {state.task_id}): Error during simple LLM call: {e}",
                        exc_info=True,
                    )
                    error_for_ws = (
                        f"Error in node '{self.node_id}' (tools disabled): {e}"
                    )
                    current_node_span.set_status(
                        trace.Status(trace.StatusCode.ERROR, description=error_for_ws)
                    )
                    current_node_span.record_exception(e)

                    final_update_dict = {
                        "error_message": error_for_ws,
                        "dynamic_data": state.dynamic_data.copy()
                        if state.dynamic_data
                        else {},
                    }

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id,
                        status="node_completed",
                        detail=f"Node '{self.node_id}' (Simple LLM Call) finished. Error: {error_for_ws or 'None'}",
                        current_node=self.node_id,
                        next_node=None,
                    ),
                )
                return final_update_dict

            logger.debug(
                f"Node '{self.node_id}' (Task: {state.task_id}): Executing ReAct loop (tool use enabled)."
            )
            current_node_span.set_attribute(
                "app.react.max_iterations", self.max_react_iterations
            )
            current_error_message: Optional[str] = None

            current_dynamic_data_for_loop = (
                state.dynamic_data.copy() if state.dynamic_data else {}
            )
            current_dynamic_data_for_loop.setdefault(
                "scratchpad", state.scratchpad or ""
            )
            current_dynamic_data_for_loop.setdefault(
                "tool_call_history", list(state.tool_call_history or [])
            )

            for i in range(self.max_react_iterations):
                current_node_span.add_event(f"ReAct Iteration Start: {i + 1}")
                logger.info(
                    f"Node '{self.node_id}' (Task: {state.task_id}): ReAct Iteration {i + 1}/{self.max_react_iterations}"
                )
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id,
                        status="node_iterating",
                        detail=f"Node '{self.node_id}' ReAct iteration {i + 1}.",
                        current_node=self.node_id,
                    ),
                )

                temp_state_for_prompt_preparation = AgentGraphState(
                    task_id=state.task_id,
                    original_input=state.original_input,
                    current_iteration=i,
                    thoughts=state.thoughts,
                    current_thoughts_to_evaluate=state.current_thoughts_to_evaluate,
                    current_best_thought_id=state.current_best_thought_id,
                    search_depth=state.search_depth,
                    max_search_depth=state.max_search_depth,
                    dynamic_data=current_dynamic_data_for_loop,
                    metadata=state.metadata,
                )

                try:
                    prompt_input_values = await self._prepare_prompt_input(
                        temp_state_for_prompt_preparation
                    )
                    formatted_prompt = self.prompt_template.format(
                        **prompt_input_values
                    )
                    current_node_span.set_attribute(
                        f"app.react.iteration.{i + 1}.prompt_length",
                        len(formatted_prompt),
                    )
                    logger.debug(
                        f"Node '{self.node_id}' (Task: {state.task_id}) Iteration {i + 1}: Formatted prompt ready."
                    )
                except Exception as prompt_err:
                    logger.error(
                        f"Node '{self.node_id}' (Task: {state.task_id}): Error preparing prompt in iteration {i + 1}: {prompt_err}",
                        exc_info=True,
                    )
                    current_error_message = (
                        f"Error preparing prompt in node '{self.node_id}': {prompt_err}"
                    )
                    current_node_span.record_exception(prompt_err)
                    break

                try:
                    llm_params_for_call = {}
                    if self.temperature is not None:
                        llm_params_for_call["temperature"] = self.temperature
                    llm_params_for_call["max_tokens"] = (
                        self.max_tokens if self.max_tokens else 1000
                    )

                    messages_for_llm = [{"role": "user", "content": formatted_prompt}]
                    llm_response_str = await self.llm_client.generate_response(
                        messages=messages_for_llm,
                        model_name=self.model_name,
                        **llm_params_for_call,
                    )
                    current_node_span.set_attribute(
                        f"app.react.iteration.{i + 1}.response_length",
                        len(llm_response_str),
                    )
                    logger.debug(
                        f"Node '{self.node_id}' (Task: {state.task_id}) Iteration {i + 1}: LLM raw response received."
                    )

                    current_dynamic_data_for_loop["last_llm_input"] = formatted_prompt
                    current_dynamic_data_for_loop["last_llm_output"] = llm_response_str
                except Exception as llm_err:
                    logger.error(
                        f"Node '{self.node_id}' (Task: {state.task_id}): LLM call failed in iteration {i + 1}: {llm_err}",
                        exc_info=True,
                    )
                    current_error_message = (
                        f"LLM call failed in node '{self.node_id}': {llm_err}"
                    )
                    current_node_span.record_exception(llm_err)
                    break

                parsed_response = self._parse_llm_response(llm_response_str)
                if parsed_response is None:
                    logger.error(
                        f"Node '{self.node_id}' (Task: {state.task_id}): Failed to parse LLM response in iteration {i + 1}. Response: {llm_response_str[:500]}..."
                    )
                    current_error_message = (
                        f"Failed to parse LLM response in node '{self.node_id}'."
                    )
                    current_node_span.set_attribute(
                        f"app.react.iteration.{i + 1}.parse_error", True
                    )
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        StatusUpdateMessage(
                            task_id=state.task_id,
                            status="node_error",
                            detail=f"Node '{self.node_id}' failed to parse LLM response.",
                            current_node=self.node_id,
                        ),
                    )
                    break

                action = parsed_response["action"]
                action_input = parsed_response["action_input"]
                current_node_span.set_attribute(
                    f"app.react.iteration.{i + 1}.action", action
                )
                logger.info(
                    f"Node '{self.node_id}' (Task: {state.task_id}) Iteration {i + 1}: Parsed Action: {action}, Input type: {type(action_input).__name__}"
                )

                observation = ""
                tool_call_entry: Optional[Dict[str, Any]] = None

                normalized_action = action.lower().replace(" ", "_")

                if normalized_action == "final_answer" or normalized_action == "finish":
                    logger.info(
                        f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Received '{action}' action. Finishing."
                    )
                    current_node_span.add_event("ReAct Action: Finish")
                    final_answer_content = (
                        action_input
                        if isinstance(action_input, str)
                        else json.dumps(action_input, default=str)
                    )

                    final_update_dict = {
                        "error_message": None,
                        "dynamic_data": current_dynamic_data_for_loop,
                    }

                    output_key = self.output_field_name or "final_answer"
                    if "." in output_key and output_key.startswith("dynamic_data."):
                        parts = output_key.split(".")
                        current_level = final_update_dict["dynamic_data"]
                        for p_key in parts[1:-1]:
                            current_level = current_level.setdefault(p_key, {})
                        current_level[parts[-1]] = final_answer_content
                    else:
                        final_update_dict[output_key] = final_answer_content

                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        StatusUpdateMessage(
                            task_id=state.task_id,
                            status="node_completed",
                            detail=f"Node '{self.node_id}' ReAct loop finished successfully.",
                            current_node=self.node_id,
                        ),
                    )
                    return final_update_dict

                elif normalized_action == "think":
                    logger.info(
                        f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Received 'think' action."
                    )
                    current_node_span.add_event("ReAct Action: Think")
                    thought_content = (
                        action_input
                        if isinstance(action_input, str)
                        else json.dumps(action_input, default=str)
                    )
                    observation = f"Thought processed: {thought_content}"
                    current_dynamic_data_for_loop["scratchpad"] += (
                        f"\nThought: {thought_content}"
                    )

                elif normalized_action == "tool_call" and isinstance(
                    action_input, dict
                ):
                    tool_name_from_action = action_input.get("tool_name")
                    tool_args_from_action = action_input.get("tool_args", {})
                    current_node_span.add_event(
                        f"ReAct Action: Tool Call ({tool_name_from_action})"
                    )

                    if not tool_name_from_action or not isinstance(
                        tool_name_from_action, str
                    ):
                        observation = "Error: 'tool_call' action input missing 'tool_name' or it's not a string."
                        logger.error(
                            f"Node '{self.node_id}': {observation} Input: {action_input}"
                        )
                        tool_call_entry = {
                            "tool_name": "unknown_format",
                            "args": action_input,
                            "result": observation,
                            "error": True,
                        }
                    elif not self.tool_manager.has(tool_name_from_action):
                        observation = (
                            f"Error: Tool '{tool_name_from_action}' does not exist."
                        )
                        logger.warning(f"Node '{self.node_id}': {observation}")
                        tool_call_entry = {
                            "tool_name": tool_name_from_action,
                            "args": tool_args_from_action,
                            "result": observation,
                            "error": True,
                        }
                    elif (
                        self.allowed_tools is not None
                        and tool_name_from_action not in self.allowed_tools
                    ):
                        observation = f"Error: Tool '{tool_name_from_action}' is not allowed for this node."
                        logger.warning(f"Node '{self.node_id}': {observation}")
                        tool_call_entry = {
                            "tool_name": tool_name_from_action,
                            "args": tool_args_from_action,
                            "result": observation,
                            "error": True,
                        }
                    else:
                        logger.info(
                            f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Executing tool '{tool_name_from_action}' via 'tool_call' action."
                        )
                        await self.notification_service.broadcast_to_task(
                            state.task_id,
                            IntermediateResultMessage(
                                task_id=state.task_id,
                                node_id=self.node_id,
                                result_step_name="tool_calling",
                                data={
                                    "tool_name": tool_name_from_action,
                                    "tool_args": tool_args_from_action,
                                },
                            ),
                        )
                        tool_result_str = ""
                        tool_error = False
                        try:
                            tool_instance = self.tool_manager.get_tool(
                                tool_name_from_action
                            )
                            tool_result = await tool_instance.ainvoke(
                                tool_args_from_action
                            )
                            tool_result_str = str(tool_result)
                            observation = (
                                f"Tool {tool_name_from_action} execution successful."
                            )
                            logger.debug(
                                f"Node '{self.node_id}': Tool '{tool_name_from_action}' Result: {tool_result_str[:200]}..."
                            )
                            current_node_span.add_event(
                                f"Tool Executed: {tool_name_from_action}",
                                attributes={"tool.result_length": len(tool_result_str)},
                            )
                        except ToolError as tool_err:
                            observation = f"Error executing tool '{tool_name_from_action}': {tool_err.message}"
                            logger.error(
                                f"Node '{self.node_id}': ToolError during execution of '{tool_name_from_action}': {tool_err.message}",
                                exc_info=False,
                            )
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(
                                tool_err,
                                attributes={"tool.name": tool_name_from_action},
                            )
                        except Exception as tool_run_e:
                            observation = f"Unexpected error running tool '{tool_name_from_action}': {str(tool_run_e)}"
                            logger.exception(
                                f"Node '{self.node_id}': Unexpected error during execution of tool '{tool_name_from_action}'"
                            )
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(
                                tool_run_e,
                                attributes={"tool.name": tool_name_from_action},
                            )

                        tool_call_entry = {
                            "tool_name": tool_name_from_action,
                            "args": tool_args_from_action,
                            "result": tool_result_str,
                            "error": tool_error,
                        }
                        observation = f"Observation: {tool_result_str}"

                elif self.tool_manager.has(action):
                    current_node_span.add_event(
                        f"ReAct Action: Tool Call (Direct - {action})"
                    )
                    if (
                        self.allowed_tools is not None
                        and action not in self.allowed_tools
                    ):
                        observation = (
                            f"Error: Tool '{action}' is not allowed for this node."
                        )
                        logger.warning(f"Node '{self.node_id}': {observation}")
                        tool_call_entry = {
                            "tool_name": action,
                            "args": action_input,
                            "result": observation,
                            "error": True,
                        }
                    else:
                        logger.info(
                            f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop: Executing tool '{action}' (direct action)."
                        )
                        await self.notification_service.broadcast_to_task(
                            state.task_id,
                            IntermediateResultMessage(
                                task_id=state.task_id,
                                node_id=self.node_id,
                                result_step_name="tool_calling",
                                data={
                                    "tool_name": action,
                                    "tool_args": action_input
                                    if isinstance(action_input, dict)
                                    else {"input": action_input},
                                },
                            ),
                        )
                        tool_args_direct = (
                            action_input if isinstance(action_input, dict) else {}
                        )
                        if isinstance(action_input, str) and not tool_args_direct:
                            tool_args_direct = {"input": action_input}

                        tool_result_str = ""
                        tool_error = False
                        try:
                            tool_instance = self.tool_manager.get_tool(action)
                            tool_result = await tool_instance.ainvoke(tool_args_direct)
                            tool_result_str = str(tool_result)
                            observation = f"Tool {action} execution successful."
                            logger.debug(
                                f"Node '{self.node_id}': Tool '{action}' Result: {tool_result_str[:200]}..."
                            )
                            current_node_span.add_event(
                                f"Tool Executed: {action}",
                                attributes={"tool.result_length": len(tool_result_str)},
                            )
                        except ToolError as tool_err:
                            observation = (
                                f"Error executing tool '{action}': {tool_err.message}"
                            )
                            logger.error(
                                f"Node '{self.node_id}': ToolError during execution of '{action}': {tool_err.message}",
                                exc_info=False,
                            )
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(
                                tool_err, attributes={"tool.name": action}
                            )
                        except Exception as tool_run_e:
                            observation = f"Unexpected error running tool '{action}': {str(tool_run_e)}"
                            logger.exception(
                                f"Node '{self.node_id}': Unexpected error during execution of tool '{action}'"
                            )
                            tool_error = True
                            tool_result_str = observation
                            current_node_span.record_exception(
                                tool_run_e, attributes={"tool.name": action}
                            )

                        tool_call_entry = {
                            "tool_name": action,
                            "args": tool_args_direct,
                            "result": tool_result_str,
                            "error": tool_error,
                        }
                        observation = f"Observation: {tool_result_str}"
                else:
                    observation = f"Error: Unknown action '{action}'. LLM response was: {llm_response_str[:200]}"
                    logger.error(f"Node '{self.node_id}': {observation}")
                    current_node_span.add_event(f"ReAct Action: Unknown ({action})")
                    tool_call_entry = {
                        "tool_name": action,
                        "args": action_input,
                        "result": observation,
                        "error": True,
                    }

                current_dynamic_data_for_loop["scratchpad"] += f"\n{observation}"
                if tool_call_entry:
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        IntermediateResultMessage(
                            task_id=state.task_id,
                            node_id=self.node_id,
                            result_step_name="tool_result",
                            data=tool_call_entry,
                        ),
                    )
                    current_dynamic_data_for_loop["tool_call_history"].append(
                        tool_call_entry
                    )

            current_node_span.set_attribute("app.react.iterations_completed", i + 1)
            final_log_message = (
                f"Node '{self.node_id}' (Task: {state.task_id}): ReAct loop "
            )
            if current_error_message:
                final_log_message += f"finished due to error after {i + 1} iterations: {current_error_message}"
                current_node_span.set_status(
                    trace.Status(
                        trace.StatusCode.ERROR, description=current_error_message
                    )
                )
            else:
                final_log_message += (
                    f"reached max iterations ({self.max_react_iterations})."
                )
                current_node_span.set_status(
                    trace.Status(
                        trace.StatusCode.OK, description="Max iterations reached"
                    )
                )

            logger.warning(final_log_message)

            final_state_update_dict: Dict[str, Any] = {
                "error_message": current_error_message
                if current_error_message
                else f"Reached max ReAct iterations ({self.max_react_iterations}).",
                "dynamic_data": current_dynamic_data_for_loop,
            }

            final_answer_content_on_loop_end = current_dynamic_data_for_loop.get(
                "scratchpad", "No answer generated after max iterations."
            )
            if current_error_message:
                final_answer_content_on_loop_end = f"Processing stopped due to error: {current_error_message}. Last scratchpad: {final_answer_content_on_loop_end}"

            output_key_on_loop_end = self.output_field_name or "final_answer"
            if "." in output_key_on_loop_end and output_key_on_loop_end.startswith(
                "dynamic_data."
            ):
                parts = output_key_on_loop_end.split(".")
                current_level = final_state_update_dict["dynamic_data"]
                for p_key in parts[1:-1]:
                    current_level = current_level.setdefault(p_key, {})
                current_level[parts[-1]] = final_answer_content_on_loop_end
            else:
                final_state_update_dict[output_key_on_loop_end] = (
                    final_answer_content_on_loop_end
                )

            if (
                "final_answer" not in final_state_update_dict
                and output_key_on_loop_end != "final_answer"
            ):
                final_state_update_dict["final_answer"] = (
                    final_answer_content_on_loop_end
                )

            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id,
                    status="node_completed",
                    detail=f"Node '{self.node_id}' ReAct loop finished. Error: {current_error_message or 'Max iterations reached'}",
                    current_node=self.node_id,
                ),
            )
            return final_state_update_dict

import os
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage
from opentelemetry import trace

import json


tracer = trace.get_tracer(__name__)
logger = get_logger(__name__)
settings = get_settings()


class DirectProcessorNode:
    """
    Node for directly processing simple tasks with a single LLM call
    without splitting into subtasks.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        notification_service: NotificationService,
        prompt_template_path: str = "generic/direct_processor.txt",
        temperature: float = 0.7,
        max_tokens: int = 100,
        model_name: Optional[str] = None,
        node_id: str = "direct_processor_node",
        summary_prompt_key: Optional[str] = "conversation_summary",
        input_keys_for_prompt: Optional[List[str]] = None,
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.node_id = node_id
        self.prompt_template_path_relative = prompt_template_path

        self.summary_prompt_key = summary_prompt_key
        self.input_keys_for_prompt = input_keys_for_prompt or ["task"]

        self.prompt_template_str = self._load_prompt_template()

        prompt_vars_set = set(self.input_keys_for_prompt)
        if self.summary_prompt_key:
            prompt_vars_set.add(self.summary_prompt_key)

        self.prompt_template_engine = PromptTemplate(
            template=self.prompt_template_str, input_variables=list(prompt_vars_set)
        )

        logger.info(
            f"DirectProcessorNode '{self.node_id}' initialized. Prompt: '{prompt_template_path}', "
            f"Summary key: '{self.summary_prompt_key}', Max tokens: {self.max_tokens}"
        )

    def _load_prompt_template(self) -> str:
        """지정된 상대 경로로부터 프롬프트 템플릿 문자열을 로드합니다."""
        if not self.prompt_template_path_relative:
            logger.error(f"Node '{self.node_id}': No prompt template path provided.")

            return "User request: {task}\n\nAssistant response:"

        base_prompt_dir_str = str(
            getattr(settings, "PROMPT_TEMPLATE_DIR", "config/prompts")
        )

        full_path = os.path.join(
            base_prompt_dir_str, self.prompt_template_path_relative
        )

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                logger.debug(
                    f"Node '{self.node_id}': Successfully loaded prompt template from: {full_path}"
                )
                return f.read()
        except FileNotFoundError:
            logger.error(
                f"Node '{self.node_id}': Prompt template file not found at {full_path}."
            )
            raise
        except Exception as e:
            logger.error(
                f"Node '{self.node_id}': Error loading prompt template from {full_path}: {e}",
                exc_info=True,
            )
            raise

    async def _prepare_prompt_input(self, state: AgentGraphState) -> Dict[str, Any]:
        prompt_input: Dict[str, Any] = {}
        all_expected_vars = self.prompt_template_engine.input_variables

        for key in all_expected_vars:
            value: Any = None

            if key == "task":
                value = state.original_input
            elif hasattr(state, key):
                value = getattr(state, key)

            elif isinstance(state.dynamic_data, dict) and key in state.dynamic_data:
                value = state.dynamic_data[key]

            elif isinstance(state.metadata, dict) and key in state.metadata:
                value = state.metadata[key]

            if key == self.summary_prompt_key and self.summary_prompt_key:
                if state.dynamic_data and isinstance(
                    state.dynamic_data.get(self.summary_prompt_key), str
                ):
                    value = state.dynamic_data[self.summary_prompt_key]
                    logger.debug(
                        f"Node '{self.node_id}': Using '{self.summary_prompt_key}' from dynamic_data."
                    )
                else:
                    value = "No conversation summary available."
                    logger.debug(
                        f"Node '{self.node_id}': '{self.summary_prompt_key}' not found in dynamic_data. Using default."
                    )

            if value is None:
                if key in self.input_keys_for_prompt or (
                    key == self.summary_prompt_key and self.summary_prompt_key
                ):
                    logger.warning(
                        f"Node '{self.node_id}': Key '{key}' for prompt was not found; using empty string."
                    )
                prompt_input[key] = ""
            elif isinstance(value, (list, dict)):
                try:
                    prompt_input[key] = json.dumps(
                        value, indent=2, ensure_ascii=False, default=str
                    )
                except TypeError:
                    prompt_input[key] = str(value)
            else:
                prompt_input[key] = str(value)

        logger.debug(
            f"Node '{self.node_id}': Prepared prompt input keys: {list(prompt_input.keys())}"
        )
        return prompt_input

    async def __call__(
        self, state: AgentGraphState,
    ) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "graph.node.direct_processor",
            attributes={"node_id": self.node_id, "task_id": state.task_id},
        ) as current_node_span:
            logger.info(
                f"DirectProcessorNode '{self.node_id}' execution started. Task ID: {state.task_id}"
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

            error_message: Optional[str] = None
            result_content: Optional[str] = None

            try:
                prompt_input_values = await self._prepare_prompt_input(state)
                formatted_prompt = self.prompt_template_engine.format(
                    **prompt_input_values
                )
                current_node_span.set_attribute(
                    "app.llm.prompt_length", len(formatted_prompt)
                )
                logger.debug(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Formatted prompt:\n{formatted_prompt[:500]}..."
                )

                messages = [{"role": "user", "content": formatted_prompt}]
                llm_params_for_call: Dict[str, Any] = {}
                if self.temperature is not None:
                    llm_params_for_call["temperature"] = self.temperature
                if self.max_tokens is not None:
                    llm_params_for_call["max_tokens"] = self.max_tokens

                result_content = await self.llm_client.generate_response(
                    messages=messages, model_name=self.model_name, **llm_params_for_call
                )
                current_node_span.set_attribute(
                    "app.llm.response_length", len(result_content or "")
                )
                logger.info(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Direct processing completed successfully."
                )

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    IntermediateResultMessage(
                        task_id=state.task_id,
                        node_id=self.node_id,
                        result_step_name="direct_processing_complete",
                        data={
                            "result_length": len(result_content)
                            if result_content
                            else 0
                        },
                    ),
                )

            except Exception as e:
                error_message = (
                    f"Error during direct processing in node '{self.node_id}': {str(e)}"
                )
                logger.error(
                    f"Node '{self.node_id}' (Task: {state.task_id}): {error_message}",
                    exc_info=True,
                )
                current_node_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, description=error_message)
                )
                current_node_span.record_exception(e)
                result_content = (
                    f"An error occurred while processing your request: {str(e)}"
                )

            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id,
                    status="node_completed",
                    detail=f"Node '{self.node_id}' finished. Error: {error_message or 'None'}",
                    current_node=self.node_id,
                    next_node="__end__",
                ),
            )

            if error_message:
                current_node_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, description=error_message)
                )
            else:
                current_node_span.set_status(trace.Status(trace.StatusCode.OK))

            return {
                "final_answer": result_content,
                "error_message": error_message,
                "dynamic_data": state.dynamic_data,
                "next_action": "__end__",
            }

import os
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
logger = get_logger(__name__)
settings = get_settings()


class TaskComplexityEvaluatorNode:
    def __init__(
        self,
        llm_client: LLMClient,
        notification_service: NotificationService,
        temperature: float = 0.3,
        prompt_template_path: Optional[str] = "generic/task_complexity_evaluation.txt",
        model_name: Optional[str] = None,
        node_id: str = "task_complexity_evaluator",
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.temperature = temperature
        self.prompt_template_path = prompt_template_path
        self.model_name = model_name
        self.node_id = node_id
        self.prompt_template_str = self._load_prompt_template_if_path_exists()
        logger.info(
            f"TaskComplexityEvaluatorNode '{self.node_id}' initialized. "
            f"Prompt: '{self.prompt_template_path if self.prompt_template_path else 'Default internal'}'"
        )

    def _load_prompt_template_if_path_exists(self) -> Optional[str]:
        if not self.prompt_template_path:
            return None
        base_prompt_dir = getattr(settings, "PROMPT_TEMPLATE_DIR", "config/prompts")
        if os.path.isabs(self.prompt_template_path):
            full_path = self.prompt_template_path
        else:
            full_path = os.path.join(base_prompt_dir, self.prompt_template_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                logger.debug(
                    f"Successfully loaded prompt template from: {full_path} for node '{self.node_id}'"
                )
                return f.read()
        except FileNotFoundError:
            logger.warning(
                f"Prompt template file not found for TaskComplexityEvaluatorNode '{self.node_id}': {full_path}. Using default internal prompt."
            )
            return None
        except Exception as e:
            logger.error(
                f"Error loading prompt template from {full_path} for node '{self.node_id}': {e}. Using default internal prompt."
            )
            return None

    def _construct_prompt(self, subtask: Dict[str, Any], state: AgentGraphState) -> str:
        if self.prompt_template_str:
            try:
                prompt_data = {
                    "original_input": state.original_input,
                    "subtask_title": subtask.get("title", "Untitled Subtask"),
                    "subtask_description": subtask.get(
                        "description", "No description provided"
                    ),
                }
                template = PromptTemplate(
                    template=self.prompt_template_str,
                    input_variables=list(prompt_data.keys()),
                )
                return template.format(**prompt_data)
            except Exception as e:
                logger.error(
                    f"Error formatting prompt template in TaskComplexityEvaluatorNode '{self.node_id}': {e}. Falling back to default internal prompt."
                )

        return f"""
    You are a task complexity evaluator. Your job is to determine whether a subtask requires complex reasoning (Tree of Thoughts) or is straightforward enough for a single-step process (Generic LLM).

    Original Task: {state.original_input}

    Subtask to Evaluate:
    Title: {subtask.get("title", "Untitled Subtask")}
    Description: {subtask.get("description", "No description provided")}

    Instructions:
    1. Analyze the subtask carefully considering its nature, difficulty, and requirements.
    2. Determine if it requires complex reasoning with multiple steps of thought (Tree of Thoughts)
       OR if it's straightforward enough for a single, direct response (Generic LLM).
    3. Consider these factors:
       - Does it require breaking down into multiple reasoning steps?
       - Does it involve comparing multiple possible approaches?
       - Does it benefit from exploring different thought paths?
       - Does it need evaluation of intermediate results?
       - Is it creative/open-ended or more factual/direct?

    You must respond with EXACTLY ONE of these two answers:
    - "COMPLEX": This task requires Tree of Thoughts for multi-step reasoning
    - "SIMPLE": This task can be handled with a single Generic LLM call

    Your evaluation:
    """

    async def __call__(
        self, state: AgentGraphState, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        if state.dynamic_data and "current_subtask_index" in state.dynamic_data:
            idx = state.dynamic_data["current_subtask_index"]
            state.dynamic_data["current_subtask"] = state.dynamic_data["subtasks"][idx]

        with tracer.start_as_current_span(
            "graph.node.task_complexity_evaluator",
            attributes={"node_id": self.node_id, "task_id": state.task_id},
        ):
            logger.info(
                f"TaskComplexityEvaluatorNode '{self.node_id}' execution started. Task ID: {state.task_id}"
            )
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id,
                    status="node_executing",
                    detail=f"Node '{self.node_id}' (Complexity Evaluator) started.",
                    current_node=self.node_id,
                ),
            )

            error_message: Optional[str] = None

            if (
                not state.dynamic_data
                or "subtasks" not in state.dynamic_data
                or not state.dynamic_data["subtasks"]
            ):
                error_message = "No subtasks available to evaluate"
                logger.error(
                    f"Node '{self.node_id}' (Task: {state.task_id}): {error_message}"
                )

                return {
                    "dynamic_data": state.dynamic_data,
                    "error_message": error_message,
                    "next_action": "__end__",
                }

            current_idx = state.dynamic_data.get("current_subtask_index", 0)
            subtasks = state.dynamic_data["subtasks"]

            if current_idx is None or current_idx >= len(subtasks):
                logger.info(
                    f"Node '{self.node_id}' (Task: {state.task_id}): All subtasks have been evaluated"
                )

                final_results = []
                for subtask in subtasks:
                    if "result" in subtask:
                        final_results.append(
                            f"Subtask: {subtask.get('title', 'Untitled')}\nResult: {subtask['result']}"
                        )

                final_answer = (
                    "\n\n".join(final_results)
                    if final_results
                    else "No results were produced for the subtasks."
                )

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id,
                        status="node_completed",
                        detail=f"Node '{self.node_id}' (Complexity Evaluator) finished. All subtasks processed.",
                        current_node=self.node_id,
                        next_node="__end__",
                    ),
                )

                state.dynamic_data["processing_complete"] = True

                return {
                    "dynamic_data": state.dynamic_data,
                    "final_answer": final_answer,
                    "next_action": "__end__",
                }

            current_subtask = subtasks[current_idx]

            try:
                evaluation_prompt = self._construct_prompt(current_subtask, state)
                logger.debug(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Evaluation prompt constructed for subtask {current_idx}"
                )

                retry_counts = state.dynamic_data.get("complexity_eval_retries", {})
                subtask_id = str(current_idx)
                retry_counts[subtask_id] = retry_counts.get(subtask_id, 0) + 1
                state.dynamic_data["complexity_eval_retries"] = retry_counts

                if retry_counts[subtask_id] > 2:
                    logger.warning(
                        f"Node '{self.node_id}' (Task: {state.task_id}): Maximum retry attempts reached for subtask {current_idx}. Using default complexity."
                    )

                    subtasks[current_idx]["is_complex"] = True

                    await self.notification_service.broadcast_to_task(
                        task_id=state.task_id,
                        message=IntermediateResultMessage(
                            task_id=state.task_id,
                            node_id=self.node_id,
                            result_step_name="subtask_complexity_default",
                            data={
                                "subtask_index": current_idx,
                                "default_complexity": "complex",
                            },
                        ),
                    )

                    next_action = "process_complex_subtask"

                    state.dynamic_data["current_subtask"] = current_subtask
                    return {
                        "dynamic_data": state.dynamic_data.copy(),
                        "original_input": current_subtask.get(
                            "description", state.original_input
                        ),
                        "next_action": next_action,
                    }

                evaluation_response = await self.llm_client.generate_response(
                    messages=[{"role": "user", "content": evaluation_prompt}],
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=100,
                )
                logger.debug(
                    f"Node '{self.node_id}' (Task: {state.task_id}): LLM evaluation received for subtask {current_idx}"
                )

                evaluation_text = evaluation_response.strip().upper()
                is_complex = "COMPLEX" in evaluation_text
                logger.debug(
                    f"[TCE] task_id={state.task_id} idx={current_idx} "
                    f"is_complex={is_complex} next_action="
                    f"{'process_complex_subtask' if is_complex else 'process_simple_subtask'}"
                )

                subtasks[current_idx]["is_complex"] = is_complex

                logger.info(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Subtask {current_idx} evaluated as {'complex (ToT)' if is_complex else 'simple (GenericLLM)'}"
                )

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    IntermediateResultMessage(
                        task_id=state.task_id,
                        node_id=self.node_id,
                        result_step_name="subtask_evaluated",
                        data={
                            "subtask_index": current_idx,
                            "subtask_title": current_subtask.get("title", "Untitled"),
                            "is_complex": is_complex,
                        },
                    ),
                )

                next_action = (
                    "process_complex_subtask"
                    if is_complex
                    else "process_simple_subtask"
                )

                state.dynamic_data["current_subtask"] = current_subtask

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id,
                        status="node_completed",
                        detail=f"Node '{self.node_id}' (Complexity Evaluator) finished. Subtask {current_idx} is {'complex' if is_complex else 'simple'}.",
                        current_node=self.node_id,
                        next_node=next_action,
                    ),
                )
                current_desc = current_subtask.get("description", state.original_input)
                return {
                    "dynamic_data": state.dynamic_data.copy()
                    if state.dynamic_data
                    else {},
                    "original_input": current_desc,
                    "next_action": next_action,
                }

            except Exception as e:
                logger.error(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Error during complexity evaluation: {e}",
                    exc_info=True,
                )
                error_message = (
                    f"Error in TaskComplexityEvaluatorNode '{self.node_id}': {e}"
                )

                subtasks[current_idx]["is_complex"] = None
                subtasks[current_idx]["error"] = str(e)

                state.dynamic_data["current_subtask_index"] = current_idx + 1

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id,
                        status="node_completed",
                        detail=f"Node '{self.node_id}' (Complexity Evaluator) encountered an error. Moving to next subtask.",
                        current_node=self.node_id,
                        next_node="task_complexity_evaluator",
                    ),
                )

                return {
                    "dynamic_data": state.dynamic_data,
                    "final_answer": None,
                    "thoughts": [],
                    "current_thoughts_to_evaluate": [],
                    "current_best_thought_id": None,
                    "search_depth": 0,
                    "next_action": "task_complexity_evaluator",
                }

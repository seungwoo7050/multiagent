from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

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


class SynthesisNode:
    """
    Final synthesis node that integrates results from all subtasks into a coherent answer.
    This node runs after all subtasks have been processed to provide a unified response.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        notification_service: NotificationService,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model_name: Optional[str] = None,
        node_id: str = "synthesis_node",
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.node_id = node_id
        logger.info(f"SynthesisNode '{self.node_id}' initialized.")

    def _create_synthesis_prompt(
        self, original_task: str, results: List[Dict[str, str]]
    ) -> str:
        """Creates a prompt for synthesizing subtask results into a final answer"""
        prompt = f"""
    TASK: {original_task}

    You have analyzed this task through several subtasks with the following results:

    {self._format_results(results)}

    INSTRUCTIONS:
    Synthesize these findings into a comprehensive, integrated answer to the original task.
    1. Identify relationships between the subtask findings
    2. Recognize patterns, common themes, and potential contradictions
    3. Develop overall conclusions that address the original question
    4. Structure your response as a cohesive analysis, not a list of separate results
    5. Focus on insights and implications rather than just restating findings

    Provide a unified, comprehensive response that directly answers the original task.
    """
        return prompt

    def _format_results(self, results: List[Dict[str, str]]) -> str:
        """Formats subtask results for inclusion in the prompt"""
        formatted = ""
        for i, item in enumerate(results, 1):
            formatted += f"SUBTASK {i}: {item['title']}\nFINDINGS: {item['result']}\n\n"
        return formatted

    async def __call__(
        self, state: AgentGraphState, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "graph.node.synthesis",
            attributes={"node_id": self.node_id, "task_id": state.task_id},
        ):
            logger.info(
                f"SynthesisNode '{self.node_id}' execution started. Task ID: {state.task_id}"
            )
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id,
                    status="node_executing",
                    detail=f"Node '{self.node_id}' (Synthesis) started.",
                    current_node=self.node_id,
                ),
            )

            error_message: Optional[str] = None
            synthesis_result = "No subtask results available for synthesis."

            try:
                if not state.dynamic_data or "subtasks" not in state.dynamic_data:
                    error_message = "No subtasks found in state for synthesis."
                    logger.warning(
                        f"Node '{self.node_id}' (Task: {state.task_id}): {error_message}"
                    )
                    return {
                        "final_answer": "Unable to synthesize results: No subtask data available.",
                        "error_message": error_message,
                    }

                subtasks = state.dynamic_data.get("subtasks", [])
                results_with_context = []

                for idx, subtask in enumerate(subtasks):
                    if "result" in subtask:
                        results_with_context.append(
                            {
                                "title": subtask.get("title", f"Subtask {idx + 1}"),
                                "result": subtask.get("result", "No result"),
                            }
                        )

                if not results_with_context:
                    error_message = "No results found in subtasks for synthesis."
                    logger.warning(
                        f"Node '{self.node_id}' (Task: {state.task_id}): {error_message}"
                    )
                    return {
                        "final_answer": "Unable to synthesize results: No subtask results available.",
                        "error_message": error_message,
                    }

                synthesis_prompt = self._create_synthesis_prompt(
                    state.original_input, results_with_context
                )
                logger.debug(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Synthesis prompt created with {len(results_with_context)} subtask results."
                )

                messages = [{"role": "user", "content": synthesis_prompt}]
                synthesis_result = await self.llm_client.generate_response(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                logger.info(
                    f"Node '{self.node_id}' (Task: {state.task_id}): Successfully synthesized results from {len(results_with_context)} subtasks."
                )

                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    IntermediateResultMessage(
                        task_id=state.task_id,
                        node_id=self.node_id,
                        result_step_name="synthesis_complete",
                        data={
                            "synthesis_length": len(synthesis_result),
                            "subtask_count": len(results_with_context),
                        },
                    ),
                )

            except Exception as e:
                error_message = f"Error during synthesis: {str(e)}"
                logger.error(
                    f"Node '{self.node_id}' (Task: {state.task_id}): {error_message}",
                    exc_info=True,
                )
                synthesis_result = (
                    f"An error occurred while synthesizing results: {str(e)}"
                )

            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id,
                    status="node_completed",
                    detail=f"Node '{self.node_id}' (Synthesis) finished. {'Error: ' + error_message if error_message else 'Success'}",
                    current_node=self.node_id,
                    next_node="__end__",
                ),
            )

            return {
                "dynamic_data": state.dynamic_data,
                "final_answer": synthesis_result,
                "error_message": error_message,
                "next_action": "__end__",
            }

# src/agents/graph_nodes/subtask_processor_node.py
from typing import Any, Dict, Optional

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

class SubtaskProcessorNode:
    def __init__(
        self,
        llm_client: LLMClient,
        notification_service: NotificationService,
        node_id: str = "subtask_processor"
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.node_id = node_id
        logger.info(f"SubtaskProcessorNode '{self.node_id}' initialized.")

    async def __call__(
        self,
        state: AgentGraphState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "graph.node.subtask_processor",
            attributes={
                "node_id": self.node_id,
                "task_id": state.task_id
            },
        ):
            logger.info(f"SubtaskProcessorNode '{self.node_id}' execution started. Task ID: {state.task_id}")
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' (Subtask Processor) started.", current_node=self.node_id)
            )

            # Store the subtask result if one was produced
            if state.final_answer and state.dynamic_data and "current_subtask" in state.dynamic_data:
                current_subtask = state.dynamic_data["current_subtask"]
                current_idx = state.dynamic_data.get("current_subtask_index", 0)
                
                # Store the result in the subtask
                if "subtasks" in state.dynamic_data and current_idx < len(state.dynamic_data["subtasks"]):
                    state.dynamic_data["subtasks"][current_idx]["result"] = state.final_answer
                    
                    logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Stored result for subtask {current_idx}")
                    
                    # Broadcast the result
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        IntermediateResultMessage(
                            task_id=state.task_id,
                            node_id=self.node_id,
                            result_step_name="subtask_result_stored",
                            data={
                                "subtask_index": current_idx,
                                "subtask_title": current_subtask.get("title", "Untitled"),
                                "result": state.final_answer
                            }
                        )
                    )
            
            # Move to the next subtask
            # Move to the next subtask
            if state.dynamic_data and "current_subtask_index" in state.dynamic_data:
                current_idx = state.dynamic_data["current_subtask_index"]
                subtasks = state.dynamic_data.get("subtasks", [])
                
                # Check if we've processed all subtasks
                if current_idx >= len(subtasks) - 1:
                    # All subtasks processed, end the workflow
                    state.dynamic_data["processing_complete"] = True
                    
                    # Compile final results into a single answer
                    final_results = []
                    for subtask in subtasks:
                        if "result" in subtask:
                            final_results.append(f"Subtask: {subtask.get('title', 'Untitled')}\nResult: {subtask['result']}")
                    
                    final_answer = "\n\n".join(final_results) if final_results else "No results were produced for the subtasks."
                    
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        StatusUpdateMessage(
                            task_id=state.task_id, 
                            status="node_completed", 
                            detail=f"Node '{self.node_id}' (Subtask Processor) finished. All subtasks processed.", 
                            current_node=self.node_id, 
                            next_node="__end__"
                        )
                    )
                    
                    return {
                        "dynamic_data": state.dynamic_data,
                        "final_answer": final_answer,
                        "next_action": "__end__"  # End the workflow
                    }
                
                # If not all processed, increment index and continue
                state.dynamic_data["current_subtask_index"] = current_idx + 1
                
                # Clear the final_answer for the next subtask
                state.final_answer = None
                
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id, 
                        status="node_completed", 
                        detail=f"Node '{self.node_id}' (Subtask Processor) finished. Moving to next subtask.", 
                        current_node=self.node_id, 
                        next_node="task_complexity_evaluator"
                    )
                )
                
                return {
                    "dynamic_data": state.dynamic_data,
                    "final_answer": None,  # Clear for next subtask
                    "next_action": "task_complexity_evaluator"  # Return to evaluator for the next subtask
                }
            
            # Fallback if dynamic_data is not properly set up
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id, 
                    status="node_completed", 
                    detail=f"Node '{self.node_id}' (Subtask Processor) finished. Missing subtask index data.", 
                    current_node=self.node_id, 
                    next_node="__end__"
                )
            )
            
            return {
                "dynamic_data": state.dynamic_data,
                "next_action": "__end__"  # End the workflow if we can't determine the next subtask
            }
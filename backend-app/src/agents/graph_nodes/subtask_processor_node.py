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
        node_id: str = "subtask_processor",
        early_exit_threshold: float = 0.9  # High threshold for early workflow termination
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.node_id = node_id
        self.early_exit_threshold = early_exit_threshold
        logger.info(f"SubtaskProcessorNode '{self.node_id}' initialized with early exit threshold: {self.early_exit_threshold}.")

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
            
            # Store the subtask result if one was produced
            if state.dynamic_data and "current_subtask" in state.dynamic_data:
                current_subtask = state.dynamic_data["current_subtask"]
                current_idx = state.dynamic_data.get("current_subtask_index", 0)
                
                # Store the result in the subtask
                if "subtasks" in state.dynamic_data and current_idx < len(state.dynamic_data["subtasks"]):
                    state.dynamic_data["subtasks"][current_idx]["result"] = state.final_answer
                    logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Stored result for subtask {current_idx}")
            
            # Check if we got a high-quality result that might warrant early exit
            excellent_result = False
            high_score_threshold = self.early_exit_threshold  # Using the configurable threshold
            strategy_decision = None

            if state.dynamic_data and isinstance(state.dynamic_data.get("search_strategy_decision"), dict):
                strategy_decision = state.dynamic_data.get("search_strategy_decision", {}).get("decision")
                best_score = state.dynamic_data.get("search_strategy_decision", {}).get("current_best_score", 0)
                
                if (strategy_decision == "finish_very_high_score" and best_score >= high_score_threshold) or best_score >= 0.95:
                    excellent_result = True
                    logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Detected excellent result with score {best_score}. Considering early workflow completion.")
            
            # Move to the next subtask
            if state.dynamic_data and "current_subtask_index" in state.dynamic_data:
                current_idx = state.dynamic_data["current_subtask_index"]
                subtasks = state.dynamic_data.get("subtasks", [])
                
                processed_count = state.dynamic_data.get("processed_subtasks_count", 0)
                processed_count += 1
                state.dynamic_data["processed_subtasks_count"] = processed_count
                
                # Critical fix: Check if next_action is already "finish" from search strategy
                if state.next_action == "finish" and "current_subtask" in state.dynamic_data:
                    logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Search strategy determined to finish with high score. Current subtask result quality: {excellent_result}.")
                    
                    # Important: Process current subtask as complete before making exit decision
                    if "subtasks" in state.dynamic_data and current_idx < len(state.dynamic_data["subtasks"]):
                        if not state.dynamic_data["subtasks"][current_idx].get("result"):
                            state.dynamic_data["subtasks"][current_idx]["result"] = state.final_answer or "Completed via search strategy"
                
                # Store the decision for the next node to reference
                if strategy_decision:
                    if "search_strategy_decision" not in state.dynamic_data:
                        state.dynamic_data["search_strategy_decision"] = {}
                    state.dynamic_data["search_strategy_decision"]["decision"] = strategy_decision
                
                # Maximum subtask processing limit (safety check)
                max_allowable_subtasks = 10
                if processed_count >= max_allowable_subtasks:
                    logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): Maximum subtask processing limit ({max_allowable_subtasks}) reached.")
                    
                    final_results = []
                    for subtask in subtasks:
                        if "result" in subtask:
                            final_results.append(f"Subtask: {subtask.get('title', 'Untitled')}\nResult: {subtask['result']}")
                    
                    final_answer = "\n\n".join(final_results) if final_results else "Processing stopped due to maximum subtask limit."
                    
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        StatusUpdateMessage(
                            task_id=state.task_id, 
                            status="node_completed", 
                            detail=f"Node '{self.node_id}' finished. Maximum subtasks reached.", 
                            current_node=self.node_id, 
                            next_node="__end__"
                        )
                    )
                    
                    return {
                        "dynamic_data": state.dynamic_data,
                        "final_answer": final_answer,
                        "next_action": "__end__"  # Force end
                    }
                
                # NEW LOGIC: Early termination if we have excellent results and processed 3+ subtasks
                if excellent_result and processed_count >= 3:
                    logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Found excellent result after processing {processed_count} subtasks. Proceeding to synthesis early.")
                    
                    # Mark task as complete and go to synthesis
                    state.dynamic_data["processing_complete"] = True
                    state.dynamic_data["early_termination"] = True
                    state.dynamic_data["early_termination_reason"] = f"Found excellent result with score {state.dynamic_data.get('search_strategy_decision', {}).get('current_best_score', 'high')} after processing {processed_count}/{len(subtasks)} subtasks"
                    
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        IntermediateResultMessage(
                            task_id=state.task_id,
                            node_id=self.node_id,
                            result_step_name="early_workflow_completion",
                            data={
                                "processed_subtasks": processed_count,
                                "total_subtasks": len(subtasks),
                                "reason": "excellent_result"
                            }
                        )
                    )
                    
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        StatusUpdateMessage(
                            task_id=state.task_id, 
                            status="node_completed", 
                            detail=f"Node '{self.node_id}' finished. Moving to synthesis with excellent early results.", 
                            current_node=self.node_id, 
                            next_node="synthesis_node"
                        )
                    )
                    
                    return {
                        "dynamic_data": state.dynamic_data,
                        "next_action": "synthesis_node"  # Skip to synthesis
                    }
                
                # Check if we've processed all subtasks - FIXED COMPARISON
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
                            detail=f"Node '{self.node_id}' finished. All subtasks processed.", 
                            current_node=self.node_id, 
                            next_node="synthesis_node"  # CHANGED: Go directly to synthesis
                        )
                    )
                    
                    return {
                        "dynamic_data": state.dynamic_data,
                        "final_answer": final_answer,
                        "next_action": "synthesis_node"  # Modified: Go to synthesis instead of ending
                    }
                
                # If not all processed, increment index and continue
                state.dynamic_data["current_subtask_index"] = current_idx + 1
                
                # Clear state for next subtask
                state.final_answer = None
                state.thoughts = []  # Clear thoughts from previous subtask
                state.current_thoughts_to_evaluate = []
                state.current_best_thought_id = None  # Reset best thought for next subtask
                state.search_depth = 0  # Reset search depth
                
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(
                        task_id=state.task_id, 
                        status="node_completed", 
                        detail=f"Node '{self.node_id}' finished. Moving to next subtask.", 
                        current_node=self.node_id, 
                        next_node="task_complexity_evaluator"
                    )
                )
                
                return {
                    "dynamic_data": state.dynamic_data,
                    "final_answer": None,  # Clear for next subtask
                    "thoughts": [],  # Clear thoughts
                    "current_thoughts_to_evaluate": [],
                    "current_best_thought_id": None,
                    "search_depth": 0,
                    "next_action": "task_complexity_evaluator"  # Return to evaluator for the next subtask
                }
            
            # Fallback
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id, 
                    status="node_completed", 
                    detail=f"Node '{self.node_id}' finished. Missing subtask index data.", 
                    current_node=self.node_id, 
                    next_node="__end__"
                )
            )
            
            return {
                "dynamic_data": state.dynamic_data,
                "next_action": "__end__"  # End the workflow if we can't determine the next subtask
            }
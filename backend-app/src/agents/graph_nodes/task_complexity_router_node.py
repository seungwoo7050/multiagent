import os
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

class TaskComplexityRouterNode:
    """
    Initial router node that evaluates task complexity and routes to either:
    1. Task division workflow (for complex tasks)
    2. Direct GenericLLM processing (for simple tasks)
    """
    def __init__(
        self,
        llm_client: LLMClient,
        notification_service: NotificationService,
        prompt_template_path: str = "generic/task_complexity_router.txt",
        complexity_threshold: float = 0.65,
        temperature: float = 0.3,
        model_name: Optional[str] = None,
        node_id: str = "task_complexity_router_node"
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.complexity_threshold = complexity_threshold
        self.temperature = temperature
        self.model_name = model_name
        self.node_id = node_id
        
        # Load the prompt template
        self.prompt_template_path = os.path.join(
            settings.PROMPT_TEMPLATE_DIR,
            prompt_template_path
        )
        with open(self.prompt_template_path, "r") as f:
            self.prompt_template = f.read()
            
        logger.info(f"TaskComplexityRouterNode '{self.node_id}' initialized. Threshold: {self.complexity_threshold}, Prompt: '{prompt_template_path}'")
    
    async def __call__(
        self,
        state: AgentGraphState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "graph.node.task_complexity_router",
            attributes={
                "node_id": self.node_id,
                "task_id": state.task_id
            },
        ):
            logger.info(f"TaskComplexityRouterNode '{self.node_id}' execution started. Task ID: {state.task_id}")
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' started.", current_node=self.node_id)
            )
            
            error_message: Optional[str] = None
            next_action = "process_simple_task"  # Default action is to process as simple
            
            try:
                # Format the prompt
                formatted_prompt = self.prompt_template.format(
                    task=state.original_input,
                    complexity_threshold=self.complexity_threshold
                )
                
                # Call the LLM
                messages = [{"role": "user", "content": formatted_prompt}]
                response = await self.llm_client.generate_response(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=100
                )
                
                # Parse the response to get the complexity score and decision
                lines = response.strip().split('\n')
                complexity_score = None
                for line in lines:
                    line = line.strip().lower()
                    if line.startswith("complexity score:"):
                        try:
                            # Extract the score (expecting format like "Complexity Score: 0.75")
                            complexity_score = float(line.split(':')[1].strip())
                        except (ValueError, IndexError):
                            logger.warning(f"Node '{self.node_id}': Failed to parse complexity score from: {line}")
                    elif line.startswith("decision:"):
                        decision = line.split(':')[1].strip().lower()
                        if "complex" in decision or "divide" in decision:
                            next_action = "process_complex_task"
                        else:
                            next_action = "process_simple_task"
                
                # If we extracted a score but no explicit decision, use the threshold
                if complexity_score is not None and "decision:" not in response.lower():
                    next_action = "process_complex_task" if complexity_score >= self.complexity_threshold else "process_simple_task"
                
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Routing decision: {next_action} (Score: {complexity_score})")
                
                # Store routing information in dynamic_data
                if state.dynamic_data is None:
                    state.dynamic_data = {}
                
                state.dynamic_data.update({
                    "initial_complexity_score": complexity_score,
                    "routing_decision": next_action,
                })
                
                # Broadcast the routing decision
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    IntermediateResultMessage(
                        task_id=state.task_id,
                        node_id=self.node_id,
                        result_step_name="complexity_evaluation",
                        data={
                            "complexity_score": complexity_score,
                            "routing_decision": next_action
                        }
                    )
                )
                
            except Exception as e:
                error_message = f"Error evaluating task complexity: {str(e)}"
                logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): {error_message}", exc_info=True)
                next_action = "process_complex_task"  # Default to complex on error
            
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id, 
                    status="node_completed", 
                    detail=f"Node '{self.node_id}' finished with decision: {next_action}.", 
                    current_node=self.node_id, 
                    next_node="task_divider_node" if next_action == "process_complex_task" else "direct_processor_node"
                )
            )
            
            # Return the routing decision
            return {
                "dynamic_data": state.dynamic_data,
                "error_message": error_message,
                "next_action": next_action
            }
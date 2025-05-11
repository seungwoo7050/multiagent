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
        max_tokens: int = 2000,
        model_name: Optional[str] = None,
        node_id: str = "direct_processor_node"
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.node_id = node_id
        
        # Load the prompt template
        self.prompt_template_path = os.path.join(
            settings.PROMPT_TEMPLATE_DIR,
            prompt_template_path
        )
        with open(self.prompt_template_path, "r") as f:
            self.prompt_template = f.read()
            
        logger.info(f"DirectProcessorNode '{self.node_id}' initialized. Prompt: '{prompt_template_path}', Max tokens: {self.max_tokens}")
    
    async def __call__(
        self,
        state: AgentGraphState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "graph.node.direct_processor",
            attributes={
                "node_id": self.node_id,
                "task_id": state.task_id
            },
        ):
            logger.info(f"DirectProcessorNode '{self.node_id}' execution started. Task ID: {state.task_id}")
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' started.", current_node=self.node_id)
            )
            
            error_message: Optional[str] = None
            result = None
            
            try:
                # Format the prompt
                formatted_prompt = self.prompt_template.format(
                    task=state.original_input
                )
                
                # Call the LLM
                messages = [{"role": "user", "content": formatted_prompt}]
                result = await self.llm_client.generate_response(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Direct processing completed successfully.")
                
                # Broadcast the result
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    IntermediateResultMessage(
                        task_id=state.task_id,
                        node_id=self.node_id,
                        result_step_name="direct_processing_complete",
                        data={
                            "result_length": len(result) if result else 0
                        }
                    )
                )
                
            except Exception as e:
                error_message = f"Error during direct processing: {str(e)}"
                logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): {error_message}", exc_info=True)
                result = f"An error occurred while processing your request: {str(e)}"
            
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(
                    task_id=state.task_id, 
                    status="node_completed", 
                    detail=f"Node '{self.node_id}' finished.", 
                    current_node=self.node_id, 
                    next_node="__end__"
                )
            )
            
            # Return the final result
            return {
                "final_answer": result,
                "error_message": error_message,
                "next_action": "__end__"
            }
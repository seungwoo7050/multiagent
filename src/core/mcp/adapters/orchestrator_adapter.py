from typing import Any, Dict, Optional, cast

from pydantic import ConfigDict, Field

from src.config.logger import get_logger
from src.core.exceptions import SerializationError
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.core.task import TaskResult
from src.orchestration.orchestrator import Orchestrator

logger = get_logger(__name__)

class OrchestrationInputContext(BaseContextSchema):
    task_data: Dict[str, Any] = Field(..., description='Raw task data received from the entry point (e.g., API)')
    initial_mcp_context: Optional[Dict[str, Any]] = Field(None, description='Optional initial MCP context data passed along')
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OrchestrationOutputContext(BaseContextSchema):
    task_id: str = Field(..., description='The ID of the processed task')
    success: bool = Field(..., description='Whether the overall task orchestration was successful')
    final_result: Optional[Any] = Field(None, description='The final result of the task execution')
    error_message: Optional[str] = Field(None, description='Error message if the task orchestration failed')
    final_task_state: Optional[str] = Field(None, description="The final state of the task (e.g., 'completed', 'failed')")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OrchestratorAdapter(MCPAdapterBase):

    def __init__(self, target_component: Orchestrator):
        if not isinstance(target_component, Orchestrator):
            raise TypeError('target_component must be an instance of Orchestrator')
        super().__init__(target_component=target_component, mcp_context_type=OrchestrationInputContext)

    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(context, OrchestrationInputContext):
            raise ValueError(f'Incompatible context type: Expected OrchestrationInputContext, got {type(context).__name__}')
        orch_input_context = cast(OrchestrationInputContext, context)
        logger.debug(f'Adapting OrchestrationInputContext (ID: {orch_input_context.context_id}) for Orchestrator')
        try:
            task_data = orch_input_context.task_data
            task_id = task_data.get('id', orch_input_context.context_id)
            if orch_input_context.initial_mcp_context:
                task_data.setdefault('metadata', {})['initial_mcp_context'] = orch_input_context.initial_mcp_context
                logger.debug(f'Added initial_mcp_context to task_data metadata for task {task_id}')
            return {'task_id': task_id, 'task_data': task_data}
        except Exception as e:
            logger.error(f'Error adapting OrchestrationInputContext: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt input context for Orchestrator: {e}', original_error=e)

    async def adapt_output(self, component_output: Any, original_context: Optional[ContextProtocol]=None, **kwargs: Any) -> OrchestrationOutputContext:
        task_id = kwargs.get('task_id')
        if not task_id:
            if isinstance(original_context, OrchestrationInputContext):
                task_id = original_context.task_data.get('id', original_context.context_id)
        if not task_id:
            raise SerializationError("Missing 'task_id' in kwargs or original_context, cannot adapt Orchestrator output.")
        logger.debug(f'Adapting final orchestration result for task {task_id} to OrchestrationOutputContext')
        try:
            success = False
            final_result = None
            error_message = None
            final_task_state_value = None
            if isinstance(component_output, TaskResult):
                task_result = cast(TaskResult, component_output)
                success = task_result.success
                final_result = task_result.result
                error_message = task_result.error.get('message') if task_result.error else None
                final_task_state_value = task_result.state.value
            elif isinstance(component_output, dict):
                success = component_output.get('success', False)
                final_result = component_output.get('result')
                error_message = component_output.get('error_message')
                final_task_state_value = component_output.get('state')
            elif component_output is None:
                logger.warning(f'Orchestrator component_output is None for task {task_id}. Final status needs to be fetched separately (e.g., from memory).')
                success = False
                error_message = 'Orchestration outcome unknown (async execution, status not fetched).'
            else:
                logger.error(f'Unexpected component_output type from Orchestrator for task {task_id}: {type(component_output).__name__}')
                error_message = f'Unexpected output type: {type(component_output).__name__}'
            output_context = OrchestrationOutputContext(task_id=task_id, success=success, final_result=final_result, error_message=error_message, final_task_state=final_task_state_value, context_id=getattr(original_context, 'context_id', None))
            return output_context
        except Exception as e:
            logger.error(f'Error adapting Orchestrator output for task {task_id}: {e}', exc_info=True)
            return OrchestrationOutputContext(task_id=task_id, success=False, error_message=f'Failed to adapt orchestration output: {str(e)}', context_id=getattr(original_context, 'context_id', None))

from typing import Any, Dict, Optional, Type, cast
from pydantic import BaseModel, Field, ConfigDict
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.schema import BaseContextSchema
from src.orchestration.workflow import WorkflowEngine, WorkflowState, WorkflowStep, StepState
from src.config.logger import get_logger
from src.core.exceptions import SerializationError
logger = get_logger(__name__)

class WorkflowStepInputContext(BaseContextSchema):
    task_id: str = Field(..., description='The ID of the task this step belongs to')
    step_index: int = Field(..., description='The index of the step to be executed')
    workflow_context: Dict[str, Any] = Field(default_factory=dict, description='Shared context from the workflow')
    step_info: Dict[str, Any] = Field(..., description='Information about the step to execute (action, args)')
    model_config = ConfigDict(arbitrary_types_allowed=True)

class WorkflowStepOutputContext(BaseContextSchema):
    task_id: str = Field(..., description='The ID of the task this step belongs to')
    step_index: int = Field(..., description='The index of the executed step')
    success: bool = Field(..., description='Whether the step execution was successful')
    result: Optional[Any] = Field(None, description='The result data from the step execution')
    error_message: Optional[str] = Field(None, description='Error message if the step failed')
    next_step_index: Optional[int] = Field(None, description='Index of the next step to proceed to, if any')
    workflow_status: Optional[str] = Field(None, description='Overall status of the workflow after this step')
    model_config = ConfigDict(arbitrary_types_allowed=True)

class WorkflowAdapter(MCPAdapterBase):

    def __init__(self, target_component: WorkflowEngine):
        if not isinstance(target_component, WorkflowEngine):
            raise TypeError('target_component must be an instance of WorkflowEngine')
        super().__init__(target_component=target_component, mcp_context_type=None)

    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> Dict[str, Any]:
        task_id = getattr(context, 'task_id', None)
        if not task_id or not isinstance(task_id, str):
            raise ValueError(f"Input context (type: {type(context).__name__}) must have a valid 'task_id' attribute.")
        operation = kwargs.get('workflow_operation')
        if not operation:
            raise ValueError("'workflow_operation' must be specified in kwargs for WorkflowAdapter input.")
        logger.debug(f"Adapting input context (ID: {getattr(context, 'context_id', 'N/A')}) for workflow operation '{operation}' on task {task_id}")
        if operation == 'get_next_step':
            return {'operation': 'get_next_step', 'args': {'task_id': task_id}}
        elif operation == 'update_step_state':
            step_index = getattr(context, 'step_index', kwargs.get('step_index'))
            new_state_val = getattr(context, 'new_state', kwargs.get('new_state'))
            result = getattr(context, 'result', kwargs.get('result'))
            error = getattr(context, 'error', kwargs.get('error'))
            if step_index is None or new_state_val is None:
                raise ValueError(f"Missing 'step_index' or 'new_state' for workflow operation '{operation}'")
            try:
                new_state = StepState(new_state_val)
            except ValueError:
                valid_states = ', '.join((s.value for s in StepState))
                raise ValueError(f"Invalid StepState value provided: '{new_state_val}'. Valid states are: {valid_states}")
            return {'operation': 'update_step_state', 'args': {'task_id': task_id, 'step_index': step_index, 'new_step_state': new_state, 'result': result, 'error': error}}
        else:
            raise ValueError(f'Unsupported workflow_operation specified: {operation}')

    async def adapt_output(self, component_output: Any, original_context: Optional[ContextProtocol]=None, **kwargs: Any) -> ContextProtocol:
        operation = kwargs.get('workflow_operation', 'unknown')
        task_id = getattr(original_context, 'task_id', None)
        if not task_id and isinstance(component_output, (WorkflowState, WorkflowStep)):
            task_id = getattr(component_output, 'task_id', None)
        if not task_id and isinstance(original_context, BaseContextSchema) and isinstance(original_context.metadata, dict):
            task_id = original_context.metadata.get('task_id')
        logger.debug(f"Adapting WorkflowEngine output for operation '{operation}' (Task ID: {task_id or 'unknown'}) to MCP Context")
        try:
            output_context: BaseContextSchema
            original_context_id = getattr(original_context, 'context_id', None)
            if operation == 'get_next_step':
                if isinstance(component_output, WorkflowStep):
                    step = cast(WorkflowStep, component_output)
                    wf_state = await cast(WorkflowEngine, self.target_component).load_workflow_state(step.task_id)
                    shared_context = wf_state.workflow_context if wf_state else {}
                    output_context = WorkflowStepInputContext(task_id=step.task_id, step_index=step.step_index, workflow_context=shared_context, step_info=step.model_dump(include={'action', 'args', 'reasoning'}), context_id=original_context_id)
                elif component_output is None:
                    wf_state = await cast(WorkflowEngine, self.target_component).load_workflow_state(task_id) if task_id else None
                    if wf_state:
                        output_context = BaseContextSchema(metadata={'task_id': task_id, 'workflow_status': wf_state.status.value, 'message': 'Workflow finished or cannot proceed to next step.'}, context_id=original_context_id)
                    else:
                        output_context = BaseContextSchema(metadata={'task_id': task_id, 'error': 'Failed to load final workflow state for get_next_step.'}, context_id=original_context_id)
                else:
                    raise SerializationError(f'Unexpected output type from get_next_step: {type(component_output).__name__}')
            elif operation == 'update_step_state':
                if isinstance(component_output, WorkflowState):
                    state = cast(WorkflowState, component_output)
                    output_context = BaseContextSchema(metadata={'task_id': state.task_id, 'workflow_status': state.status.value, 'updated_step_index': kwargs.get('step_index', getattr(original_context, 'step_index', '?')), 'current_step_index': state.current_step_index, 'last_updated': state.last_updated, 'update_successful': True}, context_id=original_context_id)
                elif component_output is None:
                    output_context = BaseContextSchema(metadata={'task_id': task_id, 'update_successful': False, 'error': f'Failed to update step state for task {task_id}.'}, context_id=original_context_id)
                else:
                    raise SerializationError(f'Unexpected output type from update_step_state: {type(component_output).__name__}')
            else:
                logger.warning(f"No specific MCP context adaptation logic for workflow operation '{operation}' output.")
                output_context = BaseContextSchema(metadata={'task_id': task_id, 'operation_performed': operation, 'output_type': type(component_output).__name__, 'raw_output': str(component_output)[:200]}, context_id=original_context_id)
            return output_context
        except Exception as e:
            logger.error(f"Error adapting WorkflowEngine output for operation '{operation}' (Task ID: {task_id}): {e}", exc_info=True)
            raise SerializationError(f'Could not adapt workflow output to MCP context: {e}', original_error=e)

    async def process_with_mcp(self, context: ContextProtocol, **kwargs: Any) -> ContextProtocol:
        if 'workflow_operation' not in kwargs:
            raise ValueError("WorkflowAdapter.process_with_mcp requires 'workflow_operation' in kwargs.")
        component_output: Any = None
        error_occurred: Optional[Exception] = None
        try:
            adapted_call = await self.adapt_input(context, **kwargs)
            operation = adapted_call['operation']
            op_args = adapted_call['args']
            target_method = getattr(self.target_component, operation)
            component_output = await target_method(**op_args)
        except Exception as e:
            error_occurred = e
        output_context = await self.adapt_output(component_output=error_occurred if error_occurred else component_output, original_context=context, **kwargs)
        return output_context
from pydantic import BaseModel, Field, ConfigDict
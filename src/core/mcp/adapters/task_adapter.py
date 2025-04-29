from typing import Any, Dict, Optional, Type, cast
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.schema import TaskContext
from src.core.task import BaseTask, TaskState
from src.config.logger import get_logger
from src.core.exceptions import SerializationError
logger = get_logger(__name__)

class TaskAdapter(MCPAdapterBase):

    def __init__(self, target_component: Optional[Any]=None):
        super().__init__(target_component=target_component or BaseTask, mcp_context_type=TaskContext)

    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> BaseTask:
        if not isinstance(context, TaskContext):
            raise ValueError(f'Incompatible context type: Expected TaskContext, got {type(context).__name__}')
        task_context: TaskContext = cast(TaskContext, context)
        logger.debug(f'Adapting TaskContext (ID: {task_context.context_id}) to BaseTask (Task ID: {task_context.task_id})')
        try:
            task_data: Dict[str, Any] = {'id': task_context.task_id, 'type': task_context.task_type, 'input': task_context.input_data or {}, 'metadata': task_context.metadata or {}}
            if 'state' in task_data['metadata']:
                try:
                    task_data['state'] = TaskState(task_data['metadata']['state'])
                except ValueError:
                    logger.warning(f"Invalid task state '{task_data['metadata']['state']}' found in metadata for task {task_context.task_id}. Using default state.")
            if hasattr(task_context, 'timestamp'):
                task_data['created_at'] = int(task_context.timestamp * 1000)
            base_task = BaseTask(**task_data)
            return base_task
        except Exception as e:
            logger.error(f'Error adapting TaskContext to BaseTask: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt TaskContext to BaseTask: {e}', original_error=e)

    async def adapt_output(self, component_output: Any, original_context: Optional[ContextProtocol]=None, **kwargs: Any) -> TaskContext:
        if not isinstance(component_output, BaseTask):
            raise ValueError(f'Incompatible component output type: Expected BaseTask, got {type(component_output).__name__}')
        base_task: BaseTask = cast(BaseTask, component_output)
        logger.debug(f'Adapting BaseTask (ID: {base_task.id}) to TaskContext')
        try:
            context_data: Dict[str, Any] = {'task_id': base_task.id, 'task_type': base_task.type, 'input_data': base_task.input, 'metadata': base_task.metadata or {}}
            context_data['metadata']['state'] = base_task.state.value
            if base_task.output:
                context_data['metadata']['output'] = base_task.output
            if base_task.error:
                context_data['metadata']['error'] = base_task.error
            original_context_id: Optional[str] = None
            if isinstance(original_context, TaskContext):
                original_context_id = getattr(original_context, 'context_id', None)
            task_context = TaskContext(**context_data, context_id=original_context_id)
            return task_context
        except Exception as e:
            logger.error(f'Error adapting BaseTask to TaskContext: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt BaseTask to TaskContext: {e}', original_error=e)
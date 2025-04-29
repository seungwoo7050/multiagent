import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, cast
from pydantic import BaseModel, Field, field_validator, ConfigDict
from src.core.task import BaseTask, TaskState
from src.memory.manager import MemoryManager
from src.config.logger import get_logger_with_context, ContextLoggerAdapter
from src.core.exceptions import OrchestrationError, ErrorCode
logger: ContextLoggerAdapter = get_logger_with_context(__name__)

class WorkflowError(OrchestrationError):

    def __init__(self, message: str, task_id: Optional[str]=None, details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None):
        super().__init__(ErrorCode.WORKFLOW_ERROR, message, details, original_error)
        if task_id and self.details:
            self.details['task_id'] = task_id

class StepState(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'

class WorkflowStep(BaseModel):
    step_index: int = Field(..., description='Index of the step in the plan')
    action: str = Field(..., description='Action to perform (e.g., tool name, agent type)')
    args: Dict[str, Any] = Field(default_factory=dict, description='Arguments for the action')
    reasoning: Optional[str] = Field(None, description='Reasoning behind this step')
    state: StepState = Field(default=StepState.PENDING, description='Current state of the step')
    result: Optional[Any] = Field(None, description='Result of the step execution')
    error: Optional[Dict[str, Any]] = Field(None, description='Error details if the step failed')
    started_at: Optional[float] = Field(None, description='Timestamp when the step started')
    completed_at: Optional[float] = Field(None, description='Timestamp when the step finished')
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

class WorkflowState(BaseModel):
    task_id: str = Field(..., description='The ID of the task this workflow belongs to')
    plan: List[WorkflowStep] = Field(..., description='The sequence of steps to execute')
    current_step_index: int = Field(default=0, description='Index of the next step to execute')
    status: TaskState = Field(default=TaskState.PENDING, description='Overall status of the workflow/task')
    last_updated: float = Field(default_factory=time.time, description='Timestamp of the last update')
    workflow_context: Dict[str, Any] = Field(default_factory=dict, description='Shared context across workflow steps')
    error: Optional[Dict[str, Any]] = Field(None, description='Details of any workflow-level error')

    def get_current_step(self) -> Optional[WorkflowStep]:
        if 0 <= self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None

    def is_finished(self) -> bool:
        return self.status in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED} or self.current_step_index >= len(self.plan)
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

class WorkflowEngine:

    def __init__(self, memory_manager: MemoryManager):
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError('memory_manager must be an instance of MemoryManager')
        self.memory_manager = memory_manager
        logger.info('WorkflowEngine initialized.')

    async def _get_workflow_state_key(self, task_id: str) -> str:
        clean_task_id = task_id.replace(':', '_').replace('/', '_')
        return f'workflow_state:{clean_task_id}'

    async def load_workflow_state(self, task_id: str) -> Optional[WorkflowState]:
        state_key = await self._get_workflow_state_key(task_id)
        state_data = await self.memory_manager.load(key=state_key, context_id=task_id)
        if state_data:
            try:
                return WorkflowState.model_validate(state_data)
            except Exception as e:
                logger.error(f'Failed to parse workflow state for task {task_id}: {e}', exc_info=True)
                return None
        return None

    async def save_workflow_state(self, state: WorkflowState) -> bool:
        state_key = await self._get_workflow_state_key(state.task_id)
        state.last_updated = time.time()
        try:
            state_dict = state.model_dump(mode='json')
            return await self.memory_manager.save(key=state_key, context_id=state.task_id, data=state_dict, update_cache=False)
        except Exception as e:
            logger.error(f'Error saving workflow state for task {state.task_id}: {e}', exc_info=True)
            return False

    async def initialize_workflow(self, task: BaseTask, plan: List[Dict[str, Any]]) -> Optional[WorkflowState]:
        global logger
        logger = get_logger_with_context(__name__, task_id=task.id, trace_id=task.trace_id)
        logger.info(f'Initializing workflow for task {task.id}')
        try:
            workflow_steps = [WorkflowStep(step_index=i, **step_data) for i, step_data in enumerate(plan)]
            initial_state = WorkflowState(task_id=task.id, plan=workflow_steps, status=TaskState.RUNNING)
            if await self.save_workflow_state(initial_state):
                logger.info(f'Workflow initialized and saved for task {task.id}')
                return initial_state
            else:
                logger.error(f'Failed to save initial workflow state for task {task.id}')
                return None
        except Exception as e:
            logger.exception(f'Error initializing workflow for task {task.id}: {e}')
            return None

    async def get_next_step(self, task_id: str) -> Optional[WorkflowStep]:
        state = await self.load_workflow_state(task_id)
        if not state:
            logger.warning(f'Workflow state not found for task {task_id} when getting next step.')
            return None
        if state.is_finished():
            logger.debug(f'Workflow for task {task_id} is already finished (Status: {state.status.value}). No next step.')
            return None
        next_step = state.get_current_step()
        if next_step:
            logger.debug(f'Next step for task {task_id} is step {next_step.step_index} (Action: {next_step.action})')
        else:
            logger.error(f'Workflow for task {task_id} is not finished, but no next step found at index {state.current_step_index}. Plan length: {len(state.plan)}')
            state.status = TaskState.COMPLETED
            await self.save_workflow_state(state)
        return next_step

    async def update_step_state(self, task_id: str, step_index: int, new_step_state: StepState, result: Optional[Any]=None, error: Optional[Dict[str, Any]]=None) -> Optional[WorkflowState]:
        state = await self.load_workflow_state(task_id)
        if not state or not 0 <= step_index < len(state.plan):
            logger.error(f'Cannot update step state: Workflow state not found or invalid step index for task {task_id}, step {step_index}')
            return None
        global logger
        logger = get_logger_with_context(__name__, task_id=task_id)
        step = state.plan[step_index]
        if step.state in [StepState.COMPLETED, StepState.FAILED, StepState.SKIPPED]:
            logger.warning(f'Step {step_index} for task {task_id} is already in a terminal state ({step.state.value}). Ignoring update to {new_step_state.value}.')
            return state
        step.state = new_step_state
        timestamp = time.time()
        move_to_next = False
        if new_step_state == StepState.RUNNING:
            step.started_at = timestamp
        elif new_step_state in [StepState.COMPLETED, StepState.FAILED, StepState.SKIPPED]:
            step.completed_at = timestamp
            if step.started_at is None:
                step.started_at = timestamp
        if new_step_state == StepState.COMPLETED:
            step.result = result
            step.error = None
            move_to_next = True
        elif new_step_state == StepState.FAILED:
            step.error = error
            step.result = None
            logger.warning(f'Step {step_index} failed for task {task_id}. Workflow status set to FAILED.')
            state.status = TaskState.FAILED
            state.error = {'failed_step': step_index, 'error_details': error}
        elif new_step_state == StepState.SKIPPED:
            step.result = {'message': 'Step skipped'}
            step.error = None
            move_to_next = True
        if move_to_next:
            state.current_step_index += 1
        if state.is_finished():
            if state.status not in [TaskState.FAILED, TaskState.CANCELED]:
                state.status = TaskState.COMPLETED
            logger.info(f'Workflow for task {task_id} reached terminal state: {state.status.value}')
        elif state.current_step_index >= len(state.plan):
            state.status = TaskState.COMPLETED
            logger.info(f'Workflow for task {task_id} completed all steps.')
        if await self.save_workflow_state(state):
            logger.debug(f'Updated state for step {step_index} of task {task_id} to {new_step_state.value}')
            return state
        else:
            logger.error(f'Failed to save updated workflow state for task {task_id} after step {step_index} update.')
            return None

    async def handle_step_failure(self, task_id: str, step_index: int, error: Dict[str, Any]) -> Optional[WorkflowState]:
        logger.info(f'Handling failure for step {step_index} in task {task_id}')
        return await self.update_step_state(task_id, step_index, StepState.FAILED, error=error)

    async def complete_workflow(self, task_id: str, final_output: Optional[Any]=None) -> bool:
        state = await self.load_workflow_state(task_id)
        if not state:
            logger.error(f'Cannot complete workflow: Workflow state not found for task {task_id}.')
            return False
        if state.is_finished():
            logger.warning(f'Workflow for task {task_id} is already finished (Status: {state.status.value}). Cannot mark as completed again.')
            return state.status == TaskState.COMPLETED
        state.status = TaskState.COMPLETED
        state.current_step_index = len(state.plan)
        if final_output is not None:
            state.workflow_context['final_output'] = final_output
        logger.info(f'Marking workflow for task {task_id} as COMPLETED.')
        return await self.save_workflow_state(state)

    async def fail_workflow(self, task_id: str, error: Dict[str, Any]) -> bool:
        state = await self.load_workflow_state(task_id)
        if not state:
            logger.error(f'Cannot fail workflow: Workflow state not found for task {task_id}.')
            return False
        if state.is_finished():
            logger.warning(f'Workflow for task {task_id} is already finished (Status: {state.status.value}). Cannot mark as failed again.')
            return state.status == TaskState.FAILED
        state.status = TaskState.FAILED
        state.error = error
        logger.error(f'Marking workflow for task {task_id} as FAILED.')
        return await self.save_workflow_state(state)
_workflow_engine_instance: Optional[WorkflowEngine] = None
_workflow_engine_lock = asyncio.Lock()

async def get_workflow_engine(memory_manager: Optional[MemoryManager]=None) -> WorkflowEngine:
    global _workflow_engine_instance
    if _workflow_engine_instance is None:
        async with _workflow_engine_lock:
            if _workflow_engine_instance is None:
                if not memory_manager:
                    try:
                        from src.memory.manager import get_memory_manager
                        memory_manager = await get_memory_manager()
                    except ImportError:
                        logger.error('Default MemoryManager function (get_memory_manager) not found.')
                        raise ValueError('MemoryManager dependency not available for WorkflowEngine.')
                    except Exception as mm_err:
                        logger.error(f'Failed to get default MemoryManager: {mm_err}')
                        raise ValueError('Could not obtain MemoryManager dependency for WorkflowEngine.')
                _workflow_engine_instance = WorkflowEngine(memory_manager=memory_manager)
                logger.info('Singleton WorkflowEngine instance created.')
    if _workflow_engine_instance is None:
        raise RuntimeError('Failed to create or retrieve WorkflowEngine instance.')
    return _workflow_engine_instance
import asyncio
import os
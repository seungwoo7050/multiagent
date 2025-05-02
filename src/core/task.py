import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.utils.timing import get_current_time_ms

logger = get_logger(__name__)
metrics_manager = get_metrics_manager()

class TaskState(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELED = 'canceled'

class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    
# Define valid state transitions
VALID_TRANSITIONS = {
    TaskState.PENDING: {TaskState.RUNNING, TaskState.CANCELED},
    TaskState.RUNNING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED},
    TaskState.COMPLETED: set(),  # Terminal state - no transitions allowed
    TaskState.FAILED: set(),     # Terminal state - no transitions allowed
    TaskState.CANCELED: set()    # Terminal state - no transitions allowed
}

class BaseTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    state: TaskState = TaskState.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: int = Field(default_factory=get_current_time_ms)
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    event_history: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    timeout_ms: Optional[int] = None  # Added timeout field

    def _validate_transition(self, new_state: TaskState) -> bool:
        """Validate if the state transition is allowed"""
        if new_state == self.state:
            return True
            
        if new_state in VALID_TRANSITIONS.get(self.state, set()):
            return True
            
        logger.warning(f"Invalid state transition for task {self.id}: {self.state} -> {new_state}")
        return False

    def start(self) -> 'BaseTask':
        if not self._validate_transition(TaskState.RUNNING):
            raise ValueError(f"Cannot transition task from {self.state} to {TaskState.RUNNING}")
            
        self.state = TaskState.RUNNING
        self.started_at = get_current_time_ms()
        self._add_event('task_started')
        logger.debug(f'Task {self.id} started.')
        return self

    def complete(self, output: Dict[str, Any]) -> 'BaseTask':
        if not self._validate_transition(TaskState.COMPLETED):
            raise ValueError(f"Cannot transition task from {self.state} to {TaskState.COMPLETED}")
            
        self.state = TaskState.COMPLETED
        self.completed_at = get_current_time_ms()
        self.output = output
        self._add_event('task_completed', {'output_keys': list(output.keys())})
        duration = self.duration_ms
        if duration is not None:
            metrics_manager.track_task("completed", status="completed", value=duration / 1000.0)
        logger.info(f'Task {self.id} completed successfully in {(duration if duration is not None else "N/A")} ms.')
        return self

    def fail(self, error: Dict[str, Any]) -> 'BaseTask':
        if not self._validate_transition(TaskState.FAILED):
            raise ValueError(f"Cannot transition task from {self.state} to {TaskState.FAILED}")
            
        self.state = TaskState.FAILED
        self.completed_at = get_current_time_ms()
        self.error = error
        self._add_event('task_failed', {'error': error})
        duration = self.duration_ms
        if duration is not None:
            metrics_manager.track_task("completed", status="failed", value=duration / 1000.0)
        error_message = error.get('message', 'Unknown error')
        logger.error(f'Task {self.id} failed in {(duration if duration is not None else "N/A")} ms. Error: {error_message}')
        return self

    def cancel(self, reason: Optional[str]=None) -> 'BaseTask':
        if not self._validate_transition(TaskState.CANCELED):
            logger.warning(f"Cannot cancel task {self.id} in state {self.state}")
            return self
            
        self.state = TaskState.CANCELED
        self.completed_at = get_current_time_ms()
        if reason:
            if self.error is None:
                self.error = {}
            self.error['cancel_reason'] = reason
        event_data = {'reason': reason} if reason else {}
        self._add_event('task_canceled', event_data)
        duration = self.duration_ms
        if duration is not None:
            metrics_manager.track_task("completed", status="canceled", value=duration / 1000.0)
        logger.warning(f'Task {self.id} canceled. Reason: {reason or "No reason provided"}')
        return self

    # Add timeout management
    async def with_timeout(self, coro, timeout_override_ms: Optional[int] = None) -> Any:
        """Run a coroutine with the task's timeout or an override"""
        timeout_ms = timeout_override_ms or self.timeout_ms
        
        if timeout_ms is None or timeout_ms <= 0:
            return await coro
            
        try:
            return await asyncio.wait_for(coro, timeout=timeout_ms/1000.0)
        except asyncio.TimeoutError:
            error = {
                'type': 'task_timeout',
                'message': f'Task timed out after {timeout_ms}ms',
                'timeout_ms': timeout_ms
            }
            self.fail(error)
            raise TimeoutError(f"Task {self.id} timed out after {timeout_ms}ms")

    @property
    def duration_ms(self) -> Optional[int]:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        if self.started_at:
            return get_current_time_ms() - self.started_at
        return None

    @property
    def elapsed_since_creation_ms(self) -> int:
        return get_current_time_ms() - self.created_at

    @property
    def is_finished(self) -> bool:
        return self.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED}
    def update_metadata(self, key: str, value: Any) -> 'BaseTask':
        self.metadata[key] = value
        logger.debug(f'Updated metadata for task {self.id}: set {key}={value}')
        return self

    def checkpoint(self, data: Dict[str, Any]) -> 'BaseTask':
        timestamp_str = str(get_current_time_ms())
        self.checkpoint_data[timestamp_str] = data
        self._add_event('checkpoint_saved', {'timestamp': timestamp_str})
        logger.debug(f'Checkpoint saved for task {self.id} at {timestamp_str}')
        return self

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self.checkpoint_data:
            return None
        latest_timestamp_str = max(self.checkpoint_data.keys(), key=int)
        return {'timestamp': int(latest_timestamp_str), 'data': self.checkpoint_data[latest_timestamp_str]}

    def get_event_history(self) -> List[Dict[str, Any]]:
        return self.event_history.copy()

    def _add_event(self, event_type: str, data: Optional[Dict[str, Any]]=None) -> None:
        self.event_history.append({'event_type': event_type, 'timestamp': get_current_time_ms(), 'data': data or {}})

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {datetime: lambda v: v.isoformat(), Enum: lambda v: v.value},
        "validate_assignment": True,
        "extra": 'forbid'
    }

    @staticmethod
    def create(task_type: str, input_data: Dict[str, Any], **kwargs) -> 'BaseTask':
        task = BaseTask(type=task_type, input=input_data, **kwargs)
        metrics_manager.track_task("created")
        logger.info(f'Task created: ID={task.id}, Type={task.type}, Priority={task.priority.name}')
        return task

class TaskFactory:

    @staticmethod
    def create_task(task_type: str, input_data: Dict[str, Any], priority: TaskPriority=TaskPriority.NORMAL, trace_id: Optional[str]=None, parent_id: Optional[str]=None, metadata: Optional[Dict[str, Any]]=None) -> BaseTask:
        task_kwargs = {'priority': priority, 'metadata': metadata or {}}
        if trace_id:
            task_kwargs['trace_id'] = trace_id
        if parent_id:
            task_kwargs['parent_id'] = parent_id
        return BaseTask.create(task_type, input_data, **task_kwargs)

    @staticmethod
    async def create_and_initialize_task(task_type: str, input_data: Dict[str, Any], **kwargs) -> BaseTask:
        task = TaskFactory.create_task(task_type, input_data, **kwargs)
        logger.debug(f'Task {task.id} created (async initialization placeholder).')
        return task

class TaskResult(BaseModel):
    task_id: str
    success: bool
    state: TaskState
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_task(cls, task: BaseTask) -> 'TaskResult':
        if not task.is_finished:
            logger.warning(f'Creating TaskResult from non-finished task {task.id} (state: {task.state.name})')
        return cls(task_id=task.id, success=task.state == TaskState.COMPLETED, state=task.state, result=task.output, error=task.error, duration_ms=task.duration_ms, metadata=task.metadata)
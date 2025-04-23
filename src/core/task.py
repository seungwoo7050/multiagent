import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from src.config.logger import get_logger
from src.config.metrics import (
    track_task_created,
    track_task_completed,
)
from src.utils.timing import get_current_time_ms

# Module logger
logger = get_logger(__name__)


class TaskState(str, Enum):
    """Enum representing the possible states of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskPriority(int, Enum):
    """Enum representing task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class BaseTask(BaseModel):
    """Base task model with efficient state tracking and serialization.
    
    This serves as the foundation for all task types in the system.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    state: TaskState = TaskState.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Timing information
    created_at: int = Field(default_factory=get_current_time_ms)
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    
    # Task details
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    
    # Metadata
    parent_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Internal tracking
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    event_history: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Get the task duration in milliseconds, if available."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        if self.started_at:
            return get_current_time_ms() - self.started_at
        return None
    
    @property
    def elapsed_since_creation_ms(self) -> int:
        """Get the elapsed time since task creation in milliseconds."""
        return get_current_time_ms() - self.created_at
    
    @property
    def is_finished(self) -> bool:
        """Check if the task is in a terminal state."""
        return self.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED}
    
    def start(self) -> "BaseTask":
        """Mark the task as started."""
        self.state = TaskState.RUNNING
        self.started_at = get_current_time_ms()
        self._add_event("task_started")
        return self
    
    def complete(self, output: Dict[str, Any]) -> "BaseTask":
        """Mark the task as completed with the given output."""
        self.state = TaskState.COMPLETED
        self.completed_at = get_current_time_ms()
        self.output = output
        self._add_event("task_completed", {"output_size": len(str(output))})
        
        # Track metrics
        if self.duration_ms:
            track_task_completed("completed", self.duration_ms / 1000.0)
        
        return self
    
    def fail(self, error: Dict[str, Any]) -> "BaseTask":
        """Mark the task as failed with the given error."""
        self.state = TaskState.FAILED
        self.completed_at = get_current_time_ms()
        self.error = error
        self._add_event("task_failed", {"error": error})
        
        # Track metrics
        if self.duration_ms:
            track_task_completed("failed", self.duration_ms / 1000.0)
        
        return self
    
    def cancel(self, reason: Optional[str] = None) -> "BaseTask":
        """Mark the task as canceled with an optional reason."""
        self.state = TaskState.CANCELED
        self.completed_at = get_current_time_ms()
        
        if reason:
            if not self.error:
                self.error = {}
            self.error["reason"] = reason
            
        self._add_event("task_canceled", {"reason": reason} if reason else {})
        
        # Track metrics
        if self.duration_ms:
            track_task_completed("canceled", self.duration_ms / 1000.0)
        
        return self
    
    def update_metadata(self, key: str, value: Any) -> "BaseTask":
        """Update a metadata field."""
        self.metadata[key] = value
        return self
    
    def checkpoint(self, data: Dict[str, Any]) -> "BaseTask":
        """Save checkpoint data for the task."""
        timestamp = get_current_time_ms()
        self.checkpoint_data[timestamp] = data
        self._add_event("checkpoint_saved", {"timestamp": timestamp})
        return self
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint data, if any."""
        if not self.checkpoint_data:
            return None
        
        latest_timestamp = max(self.checkpoint_data.keys())
        return {
            "timestamp": latest_timestamp,
            "data": self.checkpoint_data[latest_timestamp]
        }
    
    def getevent_history(self) -> List[Dict[str, Any]]:
        """Get the event history for this task."""
        return self.event_history.copy()
    
    def _add_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the task history."""
        self.event_history.append({
            "event_type": event_type,
            "timestamp": get_current_time_ms(),
            "data": data or {}
        })
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            # Custom encoders for efficient serialization
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value
        }
        validate_assignment = True
        arbitrary_types_allowed = True
        extra = "forbid"  # Prevent unexpected fields
        
    # Static methods for task creation and tracking
    @staticmethod
    def create(task_type: str, input_data: Dict[str, Any], **kwargs) -> "BaseTask":
        """Create a new task with the specified type and input data."""
        task = BaseTask(type=task_type, input=input_data, **kwargs)
        # Track metrics
        track_task_created()
        return task


class TaskFactory:
    """Factory for creating task instances."""
    
    @staticmethod
    def create_task(
        task_type: str,
        input_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaseTask:
        """Create a new task with the given parameters."""
        task_kwargs = {
            "type": task_type,
            "input": input_data,
            "priority": priority,
            "metadata": metadata or {},
        }
        
        if trace_id:
            task_kwargs["trace_id"] = trace_id
            
        if parent_id:
            task_kwargs["parent_id"] = parent_id
            
        return BaseTask(**task_kwargs)
    
    @staticmethod
    async def create_and_initialize_task(
        task_type: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> BaseTask:
        """Create and initialize a task asynchronously.
        
        This is useful for tasks that require async initialization logic.
        """
        task = TaskFactory.create_task(task_type, input_data, **kwargs)
        
        # Add any async initialization logic here
        # For example, saving to a database or publishing to a message queue
        
        return task


class TaskResult(BaseModel):
    """Model for task execution results."""
    
    task_id: str
    success: bool
    state: TaskState
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_task(cls, task: BaseTask) -> "TaskResult":
        """Create a TaskResult from a BaseTask instance."""
        return cls(
            task_id=task.id,
            success=task.state == TaskState.COMPLETED,
            state=task.state,
            result=task.output,
            error=task.error,
            duration_ms=task.duration_ms,
            metadata=task.metadata
        )
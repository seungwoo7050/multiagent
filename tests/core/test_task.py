import pytest
from datetime import datetime
import time
from typing import Dict, Any, Optional

from src.core.task import (
    BaseTask,
    TaskState,
    TaskPriority,
    TaskFactory,
    TaskResult
)
from src.utils.timing import get_current_time_ms


class TestTask:
    """Test suite for the BaseTask class and related components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.task_input = {"query": "What is the capital of France?"}
        self.task_type = "question_answering"
        self.task_output = {"answer": "Paris is the capital of France."}
        self.task_error = {"type": "service_error", "message": "External service unavailable"}

    def test_task_creation(self):
        """Test task creation and initial state."""
        # Create a task
        task = BaseTask(type=self.task_type, input=self.task_input)
        
        # Verify initial state
        assert task.type == self.task_type
        assert task.input == self.task_input
        assert task.state == TaskState.PENDING
        assert task.priority == TaskPriority.NORMAL  # Default priority
        assert task.created_at is not None
        assert task.started_at is None
        assert task.completed_at is None
        assert task.output is None
        assert task.error is None
        assert task.id is not None  # Should have auto-generated ID
        
        # Check properties
        assert task.duration_ms is None  # Not started yet
        assert task.is_finished is False

    def test_task_factory(self):
        """Test TaskFactory for creating tasks."""
        # Create task using factory
        task = TaskFactory.create_task(
            task_type=self.task_type,
            input_data=self.task_input,
            priority=TaskPriority.HIGH,
            metadata={"source": "test"}
        )
        
        # Verify task properties
        assert task.type == self.task_type
        assert task.input == self.task_input
        assert task.priority == TaskPriority.HIGH
        assert task.metadata == {"source": "test"}
        assert task.state == TaskState.PENDING

    def test_task_lifecycle(self):
        """Test the complete lifecycle of a task."""
        # Create task
        task = TaskFactory.create_task(self.task_type, self.task_input)
        
        # Start task
        task.start()
        assert task.state == TaskState.RUNNING
        assert task.started_at is not None
        assert task.completed_at is None
        
        time.sleep(0.001)
        
        # Complete task
        task.complete(self.task_output)
        assert task.state == TaskState.COMPLETED
        assert task.completed_at is not None
        assert task.output == self.task_output
        assert task.error is None
        assert task.duration_ms is not None
        assert task.duration_ms > 0
        assert task.is_finished is True
        
        # Verify event history captures state transitions
        events = task.getevent_history()
        assert len(events) >= 2  # At least 2 events (start, complete)
        assert any(e["event_type"] == "task_started" for e in events)
        assert any(e["event_type"] == "task_completed" for e in events)

    def test_task_failure(self):
        """Test task failure handling."""
        # Create and start task
        task = TaskFactory.create_task(self.task_type, self.task_input)
        task.start()
        
        # Fail task
        task.fail(self.task_error)
        assert task.state == TaskState.FAILED
        assert task.completed_at is not None
        assert task.output is None
        assert task.error == self.task_error
        assert task.duration_ms is not None
        assert task.is_finished is True
        
        # Verify event history captures failure
        events = task.getevent_history()
        failure_events = [e for e in events if e["event_type"] == "task_failed"]
        assert len(failure_events) == 1
        assert "error" in failure_events[0]["data"]

    def test_task_cancellation(self):
        """Test task cancellation."""
        # Create and start task
        task = TaskFactory.create_task(self.task_type, self.task_input)
        task.start()
        
        # Cancel task with reason
        reason = "User requested cancellation"
        task.cancel(reason)
        assert task.state == TaskState.CANCELED
        assert task.completed_at is not None
        assert task.output is None
        assert task.error is not None
        assert task.error.get("reason") == reason
        assert task.is_finished is True
        
        # Verify event history captures cancellation
        events = task.getevent_history()
        cancel_events = [e for e in events if e["event_type"] == "task_canceled"]
        assert len(cancel_events) == 1
        assert cancel_events[0]["data"].get("reason") == reason

    def test_task_metadata_and_checkpoints(self):
        """Test task metadata updates and checkpointing."""
        # Create task
        task = TaskFactory.create_task(self.task_type, self.task_input)
        
        # Update metadata
        task.update_metadata("user_id", "user123")
        task.update_metadata("priority_reason", "VIP customer")
        assert task.metadata["user_id"] == "user123"
        assert task.metadata["priority_reason"] == "VIP customer"
        
        # Add checkpoints
        checkpoint1_data = {"progress": "25%", "current_step": "retrieval"}
        checkpoint2_data = {"progress": "50%", "current_step": "processing"}
        
        task.checkpoint(checkpoint1_data)
        time.sleep(0.01)  # Ensure timestamps are different
        task.checkpoint(checkpoint2_data)
        
        # Get latest checkpoint
        latest_checkpoint = task.get_latest_checkpoint()
        assert latest_checkpoint is not None
        assert latest_checkpoint["data"] == checkpoint2_data
        assert "timestamp" in latest_checkpoint
        
        # Verify event history captures checkpoints
        events = task.getevent_history()
        checkpoint_events = [e for e in events if e["event_type"] == "checkpoint_saved"]
        assert len(checkpoint_events) == 2

    def test_task_result_creation(self):
        """Test creation of TaskResult from a BaseTask."""
        # Create and complete a task
        task = TaskFactory.create_task(self.task_type, self.task_input)
        task.start()
        task.complete(self.task_output)
        
        # Create TaskResult
        result = TaskResult.from_task(task)
        
        # Verify result properties
        assert result.task_id == task.id
        assert result.success is True
        assert result.state == TaskState.COMPLETED
        assert result.result == self.task_output
        assert result.error is None
        assert result.duration_ms == task.duration_ms
        assert result.metadata == task.metadata
        
        # Test with failed task
        failed_task = TaskFactory.create_task(self.task_type, self.task_input)
        failed_task.start()
        failed_task.fail(self.task_error)
        
        failed_result = TaskResult.from_task(failed_task)
        assert failed_result.success is False
        assert failed_result.state == TaskState.FAILED
        assert failed_result.error == self.task_error

    def test_elapsed_time_tracking(self):
        """Test elapsed time tracking functionality."""
        # Create task
        task = TaskFactory.create_task(self.task_type, self.task_input)
        
        # Elapsed time since creation should be available immediately
        assert task.elapsed_since_creation_ms >= 0
        
        # Sleep briefly to ensure time passes
        time.sleep(0.05)
        
        # Elapsed time should increase
        assert task.elapsed_since_creation_ms > 0
        
        # Start the task
        task.start()
        
        # Sleep briefly again
        time.sleep(0.05)
        
        # Duration should be available after starting
        assert task.duration_ms is not None
        assert task.duration_ms > 0
        
        # Complete the task
        task.complete(self.task_output)
        
        # Duration should be fixed after completion
        final_duration = task.duration_ms
        time.sleep(0.05)
        assert task.duration_ms == final_duration  # Should not change after completion
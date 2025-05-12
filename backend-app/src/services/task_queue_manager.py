import asyncio
import time
from typing import Any, Dict, List, Optional
from enum import Enum

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.schemas.enums import TaskPriority
from src.schemas.mcp_models import AgentGraphState

logger = get_logger(__name__)
settings = get_settings()


class TaskStatus(str, Enum):
    """Task status enum for internal tracking"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class WorkflowTask:
    """Represents a workflow task in the queue"""

    def __init__(
        self,
        task_id: str,
        graph_config_name: str,
        original_input: Any,
        initial_metadata: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_iterations: int = 100,
    ):
        self.task_id = task_id
        self.graph_config_name = graph_config_name
        self.original_input = original_input
        self.initial_metadata = initial_metadata or {}
        self.priority = priority
        self.max_iterations = max_iterations
        self.status = TaskStatus.QUEUED
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.result: Optional[AgentGraphState] = None

    def __lt__(self, other: "WorkflowTask") -> bool:
        """Comparison for priority queue"""
        if not isinstance(other, WorkflowTask):
            return NotImplemented
        # Lower priority value = higher priority (e.g., CRITICAL=1 > HIGH=2)
        return self.priority.as_int() < other.priority.as_int()


class TaskQueueManager:
    """
    Manages async execution of workflow tasks using asyncio.
    Provides a queue with priority support and configurable worker count.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "TaskQueueManager":
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = TaskQueueManager()
        return cls._instance

    def __init__(self):
        """Initialize the task queue manager"""
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_info: Dict[str, WorkflowTask] = {}
        self.max_workers = getattr(settings, "MAX_CONCURRENT_TASKS", 10)
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._stop_event = asyncio.Event()
        logger.info(f"TaskQueueManager initialized with max_workers={self.max_workers}")

    async def start(self):
        """Start the task processing workers"""
        if self._running:
            logger.warning("TaskQueueManager already running")
            return

        self._running = True
        self._stop_event.clear()

        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

        logger.info(f"Started {self.max_workers} worker tasks")

    async def stop(self):
        """Stop the task processing workers"""
        if not self._running:
            return

        logger.info("Stopping TaskQueueManager...")
        self._running = False
        self._stop_event.set()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers = []

        logger.info("TaskQueueManager stopped")

    async def enqueue_task(
        self,
        task_id: str,
        graph_config_name: str,
        original_input: Any,
        initial_metadata: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_iterations: int = 100,
    ) -> str:
        """Add a new task to the queue with priority support"""
        if task_id in self.task_info:
            logger.warning(f"Task {task_id} already in queue")
            return task_id

        task = WorkflowTask(
            task_id=task_id,
            graph_config_name=graph_config_name,
            original_input=original_input,
            initial_metadata=initial_metadata,
            priority=priority,
            max_iterations=max_iterations,
        )

        self.task_info[task_id] = task
        # Use task priority as first element in tuple for PriorityQueue ordering
        await self.queue.put((task.priority.as_int(), task))

        # Auto-start queue processing if not already running
        if not self._running:
            await self.start()

        logger.info(f"Task {task_id} enqueued with priority {priority.value}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a task"""
        if task_id not in self.task_info:
            return None

        task = self.task_info[task_id]
        duration = None

        if task.started_at:
            if task.completed_at:
                duration = task.completed_at - task.started_at
            else:
                duration = time.time() - task.started_at

        return {
            "task_id": task_id,
            "status": task.status,
            "priority": task.priority.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "duration": duration,
            "error": task.error,
            "queue_position": self._get_queue_position(task_id),
        }

    def _get_queue_position(self, task_id: str) -> Optional[int]:
        """Get the position of a task in the queue (if still queued)"""
        if (
            task_id not in self.task_info
            or self.task_info[task_id].status != TaskStatus.QUEUED
        ):
            return None

        # Copy queue items to a list to check position
        # Note: This is not atomic but gives an approximate position
        try:
            items = list(self.queue._queue)
            for i, (_, task) in enumerate(items):
                if task.task_id == task_id:
                    return i + 1
        except Exception as e:
            logger.error(f"Error getting queue position: {e}")

        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's queued or running"""
        if task_id not in self.task_info:
            return False

        task = self.task_info[task_id]

        if task.status == TaskStatus.QUEUED:
            # Can't directly remove from asyncio.PriorityQueue
            # Mark as canceled, it will be skipped when dequeued
            task.status = TaskStatus.CANCELED
            return True

        if task.status == TaskStatus.RUNNING and task_id in self.active_tasks:
            # Cancel the running asyncio task
            active_task = self.active_tasks[task_id]
            active_task.cancel()
            task.status = TaskStatus.CANCELED
            return True

        return False

    async def _worker_loop(self, worker_id: int):
        """Worker loop that processes tasks from the queue"""
        logger.info(f"Worker {worker_id} started")

        while self._running and not self._stop_event.is_set():
            try:
                # Try to get a task with timeout to allow checking stop event
                try:
                    _, task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if task.status == TaskStatus.CANCELED:
                    self.queue.task_done()
                    continue

                # Process the task
                logger.info(f"Worker {worker_id} processing task {task.task_id}")
                await self._process_task(task)
                self.queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    async def _process_task(self, task: WorkflowTask):
        """Process a single workflow task"""
        if task.status == TaskStatus.CANCELED:
            return

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        # Create a task to track execution
        execution_task = asyncio.create_task(
            self._execute_workflow(
                task.task_id,
                task.graph_config_name,
                task.original_input,
                task.initial_metadata,
                task.max_iterations,
            )
        )

        self.active_tasks[task.task_id] = execution_task

        try:
            result = await execution_task
            task.result = result
            task.status = TaskStatus.COMPLETED
        except asyncio.CancelledError:
            logger.info(f"Task {task.task_id} was cancelled")
            task.status = TaskStatus.CANCELED
            task.error = "Task was cancelled"
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = str(e)
        finally:
            task.completed_at = time.time()
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    async def _execute_workflow(
        self,
        task_id: str,
        graph_config_name: str,
        original_input: Any,
        initial_metadata: Dict[str, Any],
        max_iterations: int,
    ) -> AgentGraphState:
        """Execute the workflow using the orchestrator"""
        # Import here to avoid circular imports
        from src.api.dependencies import (
            get_new_orchestrator_dependency,
            get_memory_manager_dependency,
        )

        orchestrator = await get_new_orchestrator_dependency(
            await get_memory_manager_dependency(),
        )

        memory_manager = await get_memory_manager_dependency()

        try:
            # Execute the workflow
            final_state = await orchestrator.run_workflow(
                graph_config_name=graph_config_name,
                task_id=task_id,
                original_input=original_input,
                initial_metadata=initial_metadata,
                max_iterations=max_iterations,
            )

            # Store the final state
            state_key = "workflow_final_state"
            await memory_manager.save_state(
                context_id=task_id,
                key=state_key,
                value=final_state.model_dump(mode="json")
                if hasattr(final_state, "model_dump")
                else final_state,
                ttl=settings.TASK_STATUS_TTL,
            )

            return final_state
        except Exception as e:
            logger.error(
                f"Error executing workflow for task {task_id}: {e}", exc_info=True
            )
            raise

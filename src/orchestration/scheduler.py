import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.core.task import TaskPriority

# Initialize logger and metrics manager
logger = get_logger(__name__)
metrics = get_metrics_manager()

# Type definition for priority queue items
PriorityQueueItem = Tuple[int, float, Dict[str, Any]]

class PriorityScheduler:
    """
    Task scheduler using priority-based ordering.
    Manages task scheduling with priority levels and fair execution.
    """
    def __init__(self, max_size: int=0):
        """
        Initialize the priority scheduler.
        
        Args:
            max_size: Maximum queue size (0 for unlimited)
        """
        self._queue: asyncio.PriorityQueue[PriorityQueueItem] = asyncio.PriorityQueue(maxsize=max_size)
        self.max_priority_value: int = max((p.value for p in TaskPriority)) if TaskPriority else 1
        logger.info(f'PriorityScheduler initialized (Max size: {("unlimited" if max_size == 0 else max_size)})')

    def _get_queue_priority(self, task_priority: TaskPriority) -> int:
        """
        Convert TaskPriority to queue priority (lower value = higher priority).
        
        Args:
            task_priority: Task priority level
            
        Returns:
            int: Queue priority value (for sorting)
        """
        return self.max_priority_value - task_priority.value + 1

    async def add_task(self, task_data: Dict[str, Any]) -> None:
        """
        Add a task to the priority queue.
        
        Args:
            task_data: Task data dictionary
            
        Raises:
            asyncio.QueueFull: If queue is full
        """
        try:
            # Get the task priority
            priority_val = task_data.get('priority', TaskPriority.NORMAL.value)
            try:
                task_priority_enum = TaskPriority(priority_val)
            except ValueError:
                logger.warning(f"Invalid priority value '{priority_val}' in task data. Defaulting to NORMAL.")
                task_priority_enum = TaskPriority.NORMAL
                
            # Convert to queue priority and add tiebreaker
            queue_priority = self._get_queue_priority(task_priority_enum)
            tie_breaker = time.monotonic()
            item: PriorityQueueItem = (queue_priority, tie_breaker, task_data)
            
            # Put in queue
            await self._queue.put(item)
            
            # Log and track metrics
            task_id = task_data.get('id', 'unknown')
            logger.debug(
                f'Task {task_id} added to scheduler with queue priority {queue_priority} '
                f'(TaskPriority: {task_priority_enum.name})'
            )
            metrics.track_task('scheduled', priority=task_priority_enum.name)
            
        except asyncio.QueueFull:
            logger.error('Scheduler queue is full. Cannot add task.')
            metrics.track_task('rejections', reason='scheduler_queue_full')
            raise
        except Exception as e:
            logger.error(f'Failed to add task to scheduler: {e}', exc_info=True)
            metrics.track_task('errors', error_type='scheduler_add_task')
            raise

    async def get_next_task(self, timeout: Optional[float]=None) -> Optional[Dict[str, Any]]:
        """
        Get the next task based on priority.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Next task or None if queue empty or timeout
        """
        try:
            item: PriorityQueueItem
            if timeout is not None:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                item = await self._queue.get()
                
            queue_priority, tie_breaker, task_data = item
            self._queue.task_done()
            
            task_id = task_data.get('id', 'unknown')
            logger.debug(f'Retrieved task {task_id} from scheduler (Queue Priority: {queue_priority})')
            metrics.track_task('dequeued')
            
            return task_data
            
        except asyncio.TimeoutError:
            logger.debug(f'Scheduler timeout ({timeout}s) waiting for task.')
            return None
        except asyncio.QueueEmpty:
            logger.debug('Scheduler queue is empty.')
            return None
        except Exception as e:
            logger.error(f'Error retrieving task from scheduler: {e}', exc_info=True)
            metrics.track_task('errors', error_type='scheduler_get_task')
            return None

    async def peek_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Peek at the next task without removing it from the queue.
        
        Returns:
            Optional[Dict[str, Any]]: Next task or None if queue empty
        """
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            await self._queue.put(item)
            return item[2]  # Return the task data portion
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            logger.error(f'Error peeking at scheduler queue: {e}')
            return None

    async def get_tasks_by_priority(self, count: int=10) -> List[Dict[str, Any]]:
        """
        Get multiple tasks ordered by priority without removing them.
        
        Args:
            count: Maximum number of tasks to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of tasks
        """
        items = []
        try:
            # Create a temporary list of prioritized tasks
            for _ in range(min(count, self._queue.qsize())):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                    items.append(item)
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    break
                    
            # Return all items to the queue
            for item in items:
                await self._queue.put(item)
                
            # Return just the task data portion
            return [item[2] for item in items]
            
        except Exception as e:
            logger.error(f'Error getting tasks by priority: {e}', exc_info=True)
            
            # Make sure to return any removed items to the queue
            for item in items:
                try:
                    await self._queue.put(item)
                except Exception:
                    pass
                    
            return []

    def get_queue_size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            int: Number of tasks in queue
        """
        return self._queue.qsize()
        
    async def join(self) -> None:
        """Wait for all scheduled tasks to be processed."""
        logger.info('Waiting for all scheduled tasks to be processed...')
        await self._queue.join()
        logger.info('Scheduler queue joined (all tasks processed).')

    def is_empty(self) -> bool:
        """
        Check if queue is empty.
        
        Returns:
            bool: True if queue is empty
        """
        return self._queue.empty()
        
    async def clear(self) -> None:
        """Clear all tasks from the queue."""
        logger.warning('Clearing all tasks from scheduler queue')
        
        # Create a new queue
        old_queue = self._queue
        self._queue = asyncio.PriorityQueue(maxsize=old_queue._maxsize)
        
        # Mark all tasks in the old queue as done
        count = 0
        while not old_queue.empty():
            try:
                old_queue.get_nowait()
                old_queue.task_done()
                count += 1
            except asyncio.QueueEmpty:
                break
                
        logger.info(f'Cleared {count} tasks from scheduler queue')
        metrics.track_task('cleared', count=count)

# Global singleton instance
_scheduler_instance: Optional[PriorityScheduler] = None
_scheduler_lock = asyncio.Lock()

async def get_scheduler(name: str = 'priority', max_size: int=0) -> PriorityScheduler:
    """
    Get the singleton scheduler instance.
    
    Args:
        name: Scheduler type name (currently only 'priority' supported)
        max_size: Maximum queue size
        
    Returns:
        PriorityScheduler: Scheduler instance
    """
    global _scheduler_instance
    
    if _scheduler_instance is not None:
        return _scheduler_instance
        
    async with _scheduler_lock:
        if _scheduler_instance is None:
            if name != 'priority':
                logger.warning(f"Unknown scheduler type '{name}'. Using PriorityScheduler.")
                
            _scheduler_instance = PriorityScheduler(max_size=max_size)
            logger.info('Singleton PriorityScheduler instance created.')
            
    if _scheduler_instance is None:
        raise RuntimeError('Failed to create PriorityScheduler instance.')
        
    return _scheduler_instance
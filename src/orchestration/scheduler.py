import asyncio
import time
from typing import Any, Dict, Optional, Tuple, TypeVar
from src.core.task import BaseTask, TaskPriority
from src.config.logger import get_logger
logger = get_logger(__name__)
PriorityQueueItem = Tuple[int, float, Dict[str, Any]]

class PriorityScheduler:

    def __init__(self, max_size: int=0):
        self._queue: asyncio.PriorityQueue[PriorityQueueItem] = asyncio.PriorityQueue(maxsize=max_size)
        self.max_priority_value: int = max((p.value for p in TaskPriority)) if TaskPriority else 1
        logger.info(f'PriorityScheduler initialized (Max size: {('unlimited' if max_size == 0 else max_size)})')

    def _get_queue_priority(self, task_priority: TaskPriority) -> int:
        return self.max_priority_value - task_priority.value + 1

    async def add_task(self, task_data: Dict[str, Any]) -> None:
        try:
            priority_val = task_data.get('priority', TaskPriority.NORMAL.value)
            try:
                task_priority_enum = TaskPriority(priority_val)
            except ValueError:
                logger.warning(f"Invalid priority value '{priority_val}' in task data. Defaulting to NORMAL.")
                task_priority_enum = TaskPriority.NORMAL
            queue_priority = self._get_queue_priority(task_priority_enum)
            tie_breaker = time.monotonic()
            item: PriorityQueueItem = (queue_priority, tie_breaker, task_data)
            await self._queue.put(item)
            task_id = task_data.get('id', 'unknown')
            logger.debug(f'Task {task_id} added to scheduler with queue priority {queue_priority} (TaskPriority: {task_priority_enum.name})')
        except asyncio.QueueFull:
            logger.error('Scheduler queue is full. Cannot add task.')
            raise
        except Exception as e:
            logger.error(f'Failed to add task to scheduler: {e}', exc_info=True)
            raise

    async def get_next_task(self, timeout: Optional[float]=None) -> Optional[Dict[str, Any]]:
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
            return task_data
        except asyncio.TimeoutError:
            logger.debug(f'Scheduler timeout ({timeout}s) waiting for task.')
            return None
        except asyncio.QueueEmpty:
            logger.debug('Scheduler queue is empty.')
            return None
        except Exception as e:
            logger.error(f'Error retrieving task from scheduler: {e}', exc_info=True)
            return None

    async def peek_next_task(self) -> Optional[Dict[str, Any]]:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            await self._queue.put(item)
            return item[2]
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            logger.error(f'Error peeking at scheduler queue: {e}')
            return None

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    async def join(self) -> None:
        logger.info('Waiting for all scheduled tasks to be processed...')
        await self._queue.join()
        logger.info('Scheduler queue joined (all tasks processed).')

    def is_empty(self) -> bool:
        return self._queue.empty()
_scheduler_instance: Optional[PriorityScheduler] = None
_scheduler_lock = asyncio.Lock()

async def get_scheduler(max_size: int=0) -> PriorityScheduler:
    global _scheduler_instance
    if _scheduler_instance is not None:
        return _scheduler_instance
    async with _scheduler_lock:
        if _scheduler_instance is None:
            _scheduler_instance = PriorityScheduler(max_size=max_size)
            logger.info('Singleton PriorityScheduler instance created.')
    if _scheduler_instance is None:
        raise RuntimeError('Failed to create PriorityScheduler instance.')
    return _scheduler_instance
from typing import TypeVar
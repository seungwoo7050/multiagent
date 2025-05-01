import asyncio
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, TypeVar, cast
from pydantic import BaseModel, Field, validator, ConfigDict
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.core.exceptions import WorkerPoolError
from src.utils.timing import AsyncTimer, Timer, get_current_time_ms

logger = get_logger(__name__)
settings = get_settings()
metrics = get_metrics_manager()
T = TypeVar('T')
R = TypeVar('R')

class OrchestrationWorkerPoolConfig(BaseModel):
    pool_type: str = 'queue_asyncio'
    workers: int = Field(default_factory=lambda: max(os.cpu_count() or 1, 1), description='Number of worker tasks to run.')
    max_queue_size: int = Field(0, description='Maximum size of the task queue (0 for unlimited).')
    shutdown_timeout: float = Field(10.0, description='Timeout in seconds when shutting down the pool.')
    max_concurrent_tasks: int = Field(default_factory=lambda: max(os.cpu_count() or 1, 1) * 2, description='Maximum number of tasks allowed to run concurrently across all workers.')

    @field_validator('workers', 'max_concurrent_tasks')
    def validate_positive_int(cls, v: int, field: Field) -> int:
        if v <= 0:
            cpu_cores = max(os.cpu_count() or 1, 1)
            default_value = cpu_cores * (2 if field.name == 'max_concurrent_tasks' else 1)
            logger.warning(f'Invalid value ({v}) for {field.name}. Setting to default: {default_value}')
            return default_value
        return v
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OrchestrationWorkerPoolMetrics(BaseModel):
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_queue_size: int = 0
    max_observed_queue_size: int = 0
    active_workers: int = 0
    running_tasks: int = 0
    max_running_tasks: int = 0
    total_execution_time_ms: int = 0
    last_updated: int = Field(default_factory=get_current_time_ms)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class QueueWorkerPool:

    def __init__(self, name: str, config: Optional[OrchestrationWorkerPoolConfig]=None):
        self.name: str = name
        self._config: OrchestrationWorkerPoolConfig = config or OrchestrationWorkerPoolConfig()
        self._config.pool_type = 'queue_asyncio'
        self._metrics: OrchestrationWorkerPoolMetrics = OrchestrationWorkerPoolMetrics()
        self._work_queue: asyncio.Queue[Optional[Tuple[Callable[..., Coroutine[Any, Any, Any]], tuple, dict]]] = asyncio.Queue(maxsize=self._config.max_queue_size)
        self._workers: Set[asyncio.Task[None]] = set()
        self._is_shutdown: bool = False
        self._worker_states: Dict[str, str] = {}
        self._metrics_lock: asyncio.Lock = asyncio.Lock()
        self._concurrency_semaphore: asyncio.Semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)
        logger.info(f'Initialized QueueWorkerPool: {name}', extra={'pool_name': name, 'workers': self.config.workers, 'max_queue_size': self.config.max_queue_size or 'unlimited', 'max_concurrent_tasks': self.config.max_concurrent_tasks})
        self._start_workers()

    @property
    def config(self) -> OrchestrationWorkerPoolConfig:
        return self._config

    @property
    def metrics(self) -> OrchestrationWorkerPoolMetrics:
        self._metrics.current_queue_size = self.get_queue_size()
        return self._metrics

    def _start_workers(self) -> None:
        if self._is_shutdown:
            raise WorkerPoolError('Cannot start workers on a shutdown pool')
        for i in range(self.config.workers):
            worker_name = f'{self.name}-worker-{i}'
            worker_task: asyncio.Task[None] = asyncio.create_task(self._worker_loop(i), name=worker_name)
            self._workers.add(worker_task)
            self._worker_states[worker_name] = 'idle'
        asyncio.create_task(self._update_metric_atomic('active_workers', len(self._workers)))
        logger.info(f"Started {len(self._workers)} workers for pool '{self.name}'")

    async def _worker_loop(self, worker_id: int) -> None:
        worker_name: str = f'{self.name}-worker-{worker_id}'
        logger.debug(f'Worker {worker_name} started.')
        self._worker_states[worker_name] = 'idle'
        while not self._is_shutdown:
            work_item: Optional[Tuple[Callable[..., Coroutine[Any, Any, Any]], tuple, dict]] = None
            func: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None
            args: tuple = ()
            kwargs: dict = {}
            task_start_time: int = 0
            success: bool = False
            func_name: str = 'unknown'
            try:
                try:
                    work_item = self._work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    logger.debug(f'Worker {worker_name} waiting for task...')
                    self._worker_states[worker_name] = 'idle'
                    work_item = await self._work_queue.get()
                if work_item is None:
                    logger.debug(f'Worker {worker_name} received shutdown signal (None).')
                    self._worker_states[worker_name] = 'shutdown'
                    break
                func, args, kwargs = work_item
                func_name = getattr(func, '__name__', 'unknown')
                logger.debug(f'Worker {worker_name} picked up task: {func_name}')
                self._worker_states[worker_name] = 'pending_semaphore'
                logger.debug(f'Worker {worker_name} waiting for concurrency semaphore...')
                await self._concurrency_semaphore.acquire()
                logger.debug(f'Worker {worker_name} acquired concurrency semaphore.')
                self._worker_states[worker_name] = 'busy'
                await self._update_running_tasks(delta=1)
                task_start_time = get_current_time_ms()
                logger.debug(f'Worker {worker_name} processing task: {func_name}')
                if asyncio.iscoroutinefunction(func):
                    await func(*args, **kwargs)
                else:
                    logger.warning(f'Worker {worker_name} executing synchronous function {func_name} in executor.')
                    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
                success = True
                logger.debug(f'Worker {worker_name} completed task: {func_name}')
            except asyncio.CancelledError:
                logger.warning(f'Worker {worker_name} task cancelled: {func_name}')
                success = False
                if self._worker_states.get(worker_name) == 'busy':
                    self._concurrency_semaphore.release()
                    logger.debug(f'Worker {worker_name} released concurrency semaphore due to cancellation.')
                    await self._update_running_tasks(delta=-1)
                break
            except Exception as e:
                logger.error(f'Worker {worker_name} encountered error processing task {func_name}: {e}', exc_info=True)
                success = False
            finally:
                if work_item is not None:
                    execution_time_ms: int = get_current_time_ms() - task_start_time if task_start_time > 0 else 0
                    await self._update_completion_metrics(success, execution_time_ms)
                    metric_status: str = 'success' if success else 'failure'
                    duration_sec = execution_time_ms / 1000.0
                    metrics.track_task('completed', status=metric_status)
                    metrics.track_task('duration', status=metric_status, value=duration_sec)
                    if self._worker_states.get(worker_name) == 'busy':
                        self._concurrency_semaphore.release()
                        logger.debug(f'Worker {worker_name} released concurrency semaphore.')
                        await self._update_running_tasks(delta=-1)
                    self._worker_states[worker_name] = 'idle'
                    self._work_queue.task_done()
                elif self._worker_states.get(worker_name) == 'shutdown':
                    pass
                else:
                    logger.warning(f'Worker {worker_name} in unexpected state after loop iteration.')
                    self._worker_states[worker_name] = 'idle'
        logger.debug(f'Worker {worker_name} finished.')
        await self._update_metric_atomic('active_workers', -1, relative=True)

    async def _update_metric_atomic(self, metric_name: str, value: Any, relative: bool=False) -> None:
        async with self._metrics_lock:
            if relative:
                current_value = getattr(self.metrics, metric_name, 0)
                setattr(self.metrics, metric_name, current_value + value)
            else:
                setattr(self.metrics, metric_name, value)
            self.metrics.last_updated = get_current_time_ms()

    async def _update_running_tasks(self, delta: int) -> None:
        async with self._metrics_lock:
            self.metrics.running_tasks = max(0, self.metrics.running_tasks + delta)
            self.metrics.max_running_tasks = max(self.metrics.max_running_tasks, self.metrics.running_tasks)
            self.metrics.last_updated = get_current_time_ms()
            metrics.track_task('processing', value=self.metrics.running_tasks)

    async def _update_completion_metrics(self, success: bool, duration_ms: int) -> None:
        async with self._metrics_lock:
            if success:
                self.metrics.tasks_completed += 1
            else:
                self.metrics.tasks_failed += 1
            self.metrics.total_execution_time_ms += duration_ms
            self.metrics.last_updated = get_current_time_ms()

    async def submit(self, func: Callable[..., Coroutine[Any, Any, R]], *args: Any, **kwargs: Any) -> None:
        if self._is_shutdown:
            raise WorkerPoolError(f'Worker pool {self.name} is shutdown')
        async with self._metrics_lock:
            self.metrics.tasks_submitted += 1
            current_qsize: int = self._work_queue.qsize()
            estimated_qsize_after_put = current_qsize + 1
            self.metrics.current_queue_size = estimated_qsize_after_put
            self.metrics.max_observed_queue_size = max(self.metrics.max_observed_queue_size, estimated_qsize_after_put)
            metrics.track_task('queue_depth', value=estimated_qsize_after_put)
        try:
            await self._work_queue.put((func, args, kwargs))
            logger.debug(f'Submitted task {getattr(func, '__name__', 'unknown')} to pool {self.name}')
        except asyncio.QueueFull:
            async with self._metrics_lock:
                self.metrics.tasks_failed += 1
                metrics.track_task('queue_depth', value=self._work_queue.qsize()) # 에러 발생 시
            logger.error(f'Worker pool {self.name} queue is full. Task submission failed.')
            raise WorkerPoolError(f'Worker pool {self.name} queue is full')
        except Exception as e:
            async with self._metrics_lock:
                self.metrics.tasks_failed += 1
                metrics.track_task('queue_depth', value=self._work_queue.qsize()) # 에러 발생 시
            logger.error(f'Failed to submit task to pool {self.name}: {e}')
            raise WorkerPoolError(f'Failed to submit task: {e}', original_error=e)

    async def shutdown(self, wait: bool=True, timeout: Optional[float]=None) -> None:
        if self._is_shutdown:
            logger.warning(f'Worker pool {self.name} already shut down or shutting down.')
            return
        logger.info(f'Shutting down worker pool {self.name}...')
        self._is_shutdown = True
        num_workers = len(self._workers)
        for _ in range(num_workers):
            try:
                self._work_queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning(f'Queue full while sending shutdown signal to workers in pool {self.name}. Some workers might finish pending tasks.')
                break
            except Exception as e:
                logger.error(f'Error putting shutdown signal into queue for pool {self.name}: {e}')
        effective_timeout: float = timeout if timeout is not None else self.config.shutdown_timeout
        if wait:
            shutdown_deadline: float = time.monotonic() + effective_timeout
            try:
                logger.debug(f'Waiting for work queue of pool {self.name} to finish...')
                q_timeout: float = max(0.1, shutdown_deadline - time.monotonic())
                await asyncio.wait_for(self._work_queue.join(), timeout=q_timeout)
                logger.debug(f'Work queue finished for pool {self.name}.')
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(f'Timeout or cancellation waiting for work queue to join during shutdown of {self.name}.')
            except Exception as e:
                logger.error(f'Error joining work queue during shutdown for pool {self.name}: {e}')
            logger.debug(f'Waiting for {len(self._workers)} worker tasks of pool {self.name} to finish...')
            if self._workers:
                worker_timeout: float = max(0.1, shutdown_deadline - time.monotonic())
                workers_to_wait = {w for w in self._workers if not w.done()}
                if workers_to_wait:
                    done: Set[asyncio.Task[None]]
                    pending: Set[asyncio.Task[None]]
                    done, pending = await asyncio.wait(workers_to_wait, timeout=worker_timeout)
                    logger.debug(f'Worker wait for {self.name} completed. Done: {len(done)}, Pending: {len(pending)}')
                    if pending:
                        logger.warning(f'Cancelling {len(pending)} worker tasks that did not finish during shutdown of {self.name}.')
                        for task in pending:
                            task.cancel()
                        await asyncio.sleep(0.1)
            self._workers.clear()
        else:
            logger.debug(f'Cancelling {len(self._workers)} worker tasks immediately for pool {self.name}...')
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.sleep(0.1)
            self._workers.clear()
        logger.info(f'Worker pool {self.name} shutdown complete.')

    def get_active_workers(self) -> int:
        return sum((1 for w in self._workers if not w.done()))

    def get_queue_size(self) -> int:
        return self._work_queue.qsize()

    def get_running_tasks(self) -> int:
        return self._metrics.running_tasks
import os
import functools
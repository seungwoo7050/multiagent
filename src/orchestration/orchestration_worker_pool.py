import asyncio
import functools
import os
import threading
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.core.exceptions import WorkerPoolError
from src.utils.timing import get_current_time_ms

# Initialize global dependencies
logger = get_logger(__name__)
settings = get_settings()
metrics = get_metrics_manager()

class OrchestrationWorkerPoolConfig(BaseModel):
    """Configuration for orchestration worker pool"""
    pool_type: str = 'queue_asyncio'
    workers: int = Field(
        default_factory=lambda: max(os.cpu_count() or 1, 1), 
        description='Number of worker tasks to run.'
    )
    max_queue_size: int = Field(
        0, 
        description='Maximum size of the task queue (0 for unlimited).'
    )
    shutdown_timeout: float = Field(
        10.0, 
        description='Timeout in seconds when shutting down the pool.'
    )
    max_concurrent_tasks: int = Field(
        default_factory=lambda: max(os.cpu_count() or 1, 1) * 2, 
        description='Maximum number of tasks allowed to run concurrently across all workers.'
    )

    @field_validator('workers', 'max_concurrent_tasks')
    def validate_positive_int(cls, v: int, field: Field) -> int:
        if v <= 0:
            cpu_cores = max(os.cpu_count() or 1, 1)
            default_value = cpu_cores * (2 if field.name == 'max_concurrent_tasks' else 1)
            logger.warning(
                f'Invalid value ({v}) for {field.name}. Setting to default: {default_value}'
            )
            return default_value
        return v
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OrchestrationWorkerPoolMetrics(BaseModel):
    """Metrics for orchestration worker pool"""
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
    """
    Worker pool that executes tasks asynchronously using a queue.
    Manages multiple worker tasks and tracks metrics on execution.
    """
    def __init__(self, name: str, config: Optional[OrchestrationWorkerPoolConfig]=None):
        """
        Initialize the worker pool.
        
        Args:
            name: Pool name for identification
            config: Optional configuration settings
        """
        self.name: str = name
        self._config: OrchestrationWorkerPoolConfig = config or OrchestrationWorkerPoolConfig()
        self._config.pool_type = 'queue_asyncio'
        self._metrics: OrchestrationWorkerPoolMetrics = OrchestrationWorkerPoolMetrics()
        
        # Initialize queue and worker state
        self._work_queue: asyncio.Queue[Optional[Tuple[Callable[..., Coroutine[Any, Any, Any]], tuple, dict]]] = \
            asyncio.Queue(maxsize=self._config.max_queue_size)
        self._workers: Set[asyncio.Task[None]] = set()
        self._is_shutdown: bool = False
        self._worker_states: Dict[str, str] = {}
        
        # Initialize synchronization primitives
        self._metrics_lock: asyncio.Lock = asyncio.Lock()
        self._concurrency_semaphore: asyncio.Semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)
        
        # Log initialization
        logger.info(
            f'Initialized QueueWorkerPool: {name}', 
            extra={
                'pool_name': name, 
                'workers': self.config.workers, 
                'max_queue_size': self.config.max_queue_size or 'unlimited', 
                'max_concurrent_tasks': self.config.max_concurrent_tasks
            }
        )
        
        # Start worker tasks
        self._start_workers()

    @property
    def config(self) -> OrchestrationWorkerPoolConfig:
        """Get pool configuration"""
        return self._config

    @property
    def metrics(self) -> OrchestrationWorkerPoolMetrics:
        """Get pool metrics with current queue size"""
        self._metrics.current_queue_size = self.get_queue_size()
        return self._metrics

    def _start_workers(self) -> None:
        """Start worker tasks based on configuration"""
        if self._is_shutdown:
            raise WorkerPoolError('Cannot start workers on a shutdown pool')
            
        for i in range(self.config.workers):
            worker_name = f'{self.name}-worker-{i}'
            worker_task: asyncio.Task[None] = asyncio.create_task(
                self._worker_loop(i), 
                name=worker_name
            )
            self._workers.add(worker_task)
            self._worker_states[worker_name] = 'idle'
            
        asyncio.create_task(self._update_metric_atomic('active_workers', len(self._workers)))
        logger.info(f"Started {len(self._workers)} workers for pool '{self.name}'")

    async def _worker_loop(self, worker_id: int) -> None:
        """
        Main loop for a worker task.
        
        Args:
            worker_id: Worker identifier
        """
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
                # Get work from queue
                try:
                    work_item = self._work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    logger.debug(f'Worker {worker_name} waiting for task...')
                    self._worker_states[worker_name] = 'idle'
                    work_item = await self._work_queue.get()
                    
                # Check for shutdown signal
                if work_item is None:
                    logger.debug(f'Worker {worker_name} received shutdown signal (None).')
                    self._worker_states[worker_name] = 'shutdown'
                    break
                    
                # Extract task details
                func, args, kwargs = work_item
                func_name = getattr(func, '__name__', 'unknown')
                logger.debug(f'Worker {worker_name} picked up task: {func_name}')
                
                # Acquire concurrency semaphore
                self._worker_states[worker_name] = 'pending_semaphore'
                logger.debug(f'Worker {worker_name} waiting for concurrency semaphore...')
                await self._concurrency_semaphore.acquire()
                logger.debug(f'Worker {worker_name} acquired concurrency semaphore.')
                
                # Update state and metrics
                self._worker_states[worker_name] = 'busy'
                await self._update_running_tasks(delta=1)
                task_start_time = get_current_time_ms()
                
                # Execute task
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
                logger.error(
                    f'Worker {worker_name} encountered error processing task {func_name}: {e}', 
                    exc_info=True
                )
                success = False
                
            finally:
                # Clean up after task execution
                if work_item is not None:
                    # Update metrics
                    execution_time_ms: int = get_current_time_ms() - task_start_time if task_start_time > 0 else 0
                    await self._update_completion_metrics(success, execution_time_ms)
                    
                    metric_status: str = 'success' if success else 'failure'
                    duration_sec = execution_time_ms / 1000.0
                    metrics.track_task('completed', status=metric_status)
                    metrics.track_task('duration', status=metric_status, value=duration_sec)
                    
                    # Release semaphore if needed
                    if self._worker_states.get(worker_name) == 'busy':
                        self._concurrency_semaphore.release()
                        logger.debug(f'Worker {worker_name} released concurrency semaphore.')
                        await self._update_running_tasks(delta=-1)
                        
                    # Update state and mark task done
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
        """
        Update a metric atomically.
        
        Args:
            metric_name: Name of the metric to update
            value: Value to set or add
            relative: Whether to add to existing value (True) or replace (False)
        """
        async with self._metrics_lock:
            if relative:
                current_value = getattr(self.metrics, metric_name, 0)
                setattr(self.metrics, metric_name, current_value + value)
            else:
                setattr(self.metrics, metric_name, value)
                
            self.metrics.last_updated = get_current_time_ms()

    async def _update_running_tasks(self, delta: int) -> None:
        """
        Update running tasks count.
        
        Args:
            delta: Change in running tasks count
        """
        async with self._metrics_lock:
            self.metrics.running_tasks = max(0, self.metrics.running_tasks + delta)
            self.metrics.max_running_tasks = max(self.metrics.max_running_tasks, self.metrics.running_tasks)
            self.metrics.last_updated = get_current_time_ms()
            metrics.track_task('processing', value=self.metrics.running_tasks)

    async def _update_completion_metrics(self, success: bool, duration_ms: int) -> None:
        """
        Update metrics after task completion.
        
        Args:
            success: Whether the task completed successfully
            duration_ms: Task duration in milliseconds
        """
        async with self._metrics_lock:
            if success:
                self.metrics.tasks_completed += 1
            else:
                self.metrics.tasks_failed += 1
                
            self.metrics.total_execution_time_ms += duration_ms
            self.metrics.last_updated = get_current_time_ms()

    async def submit(self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
        """
        Submit a task to the worker pool.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Future representing the task
            
        Raises:
            WorkerPoolError: If pool is shutdown or queue is full
        """
        if self._is_shutdown:
            raise WorkerPoolError(f'Worker pool {self.name} is shutdown')
            
        # Update metrics
        async with self._metrics_lock:
            self.metrics.tasks_submitted += 1
            current_qsize: int = self._work_queue.qsize()
            estimated_qsize_after_put = current_qsize + 1
            self.metrics.current_queue_size = estimated_qsize_after_put
            self.metrics.max_observed_queue_size = max(
                self.metrics.max_observed_queue_size, 
                estimated_qsize_after_put
            )
            metrics.track_task('queue_depth', value=estimated_qsize_after_put)
            
        # Submit task to queue
        try:
            await self._work_queue.put((func, args, kwargs))
            logger.debug(f'Submitted task {getattr(func, "__name__", "unknown")} to pool {self.name}')
            return None
        except asyncio.QueueFull:
            # Handle queue full error
            async with self._metrics_lock:
                self.metrics.tasks_failed += 1
                metrics.track_task('queue_depth', value=self._work_queue.qsize())
            logger.error(f'Worker pool {self.name} queue is full. Task submission failed.')
            raise WorkerPoolError(f'Worker pool {self.name} queue is full')
        except Exception as e:
            # Handle other errors
            async with self._metrics_lock:
                self.metrics.tasks_failed += 1
                metrics.track_task('queue_depth', value=self._work_queue.qsize())
            logger.error(f'Failed to submit task to pool {self.name}: {e}')
            raise WorkerPoolError(f'Failed to submit task: {e}', original_error=e)

    async def shutdown(self, wait: bool=True, timeout: Optional[float]=None) -> None:
        """
        Shut down the worker pool.
        
        Args:
            wait: Whether to wait for tasks to complete
            timeout: Optional timeout for shutdown
        """
        if self._is_shutdown:
            logger.warning(f'Worker pool {self.name} already shut down or shutting down.')
            return
            
        logger.info(f'Shutting down worker pool {self.name}...')
        self._is_shutdown = True
        
        # Send shutdown signal to workers
        num_workers = len(self._workers)
        for _ in range(num_workers):
            try:
                self._work_queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning(
                    f"Queue full while sending shutdown signal to workers in pool '{self.name}'. "
                    "Some workers might finish pending tasks."
                )
                break
            except Exception as e:
                logger.error(f"Error putting shutdown signal into queue for pool '{self.name}': {e}")
                
        # Wait for tasks to complete if requested
        effective_timeout: float = timeout if timeout is not None else self.config.shutdown_timeout
        if wait:
            shutdown_deadline: float = time.monotonic() + effective_timeout
            
            # Wait for work queue to finish
            try:
                logger.debug(f'Waiting for work queue of pool {self.name} to finish...')
                q_timeout: float = max(0.1, shutdown_deadline - time.monotonic())
                await asyncio.wait_for(self._work_queue.join(), timeout=q_timeout)
                logger.debug(f'Work queue finished for pool {self.name}.')
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    f'Timeout or cancellation waiting for work queue to join during shutdown of {self.name}.'
                )
            except Exception as e:
                logger.error(f'Error joining work queue during shutdown for pool {self.name}: {e}')
                
            # Wait for worker tasks to finish
            logger.debug(f'Waiting for {len(self._workers)} worker tasks of pool {self.name} to finish...')
            if self._workers:
                worker_timeout: float = max(0.1, shutdown_deadline - time.monotonic())
                workers_to_wait = {w for w in self._workers if not w.done()}
                
                if workers_to_wait:
                    done: Set[asyncio.Task[None]]
                    pending: Set[asyncio.Task[None]]
                    done, pending = await asyncio.wait(workers_to_wait, timeout=worker_timeout)
                    
                    logger.debug(
                        f'Worker wait for {self.name} completed. Done: {len(done)}, Pending: {len(pending)}'
                    )
                    
                    if pending:
                        logger.warning(
                            f'Cancelling {len(pending)} worker tasks that did not finish during '
                            f'shutdown of {self.name}.'
                        )
                        for task in pending:
                            task.cancel()
                            
                        await asyncio.sleep(0.1)
                        
            self._workers.clear()
        else:
            # Immediate cancellation without waiting
            logger.debug(f'Cancelling {len(self._workers)} worker tasks immediately for pool {self.name}...')
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()
                    
            await asyncio.sleep(0.1)
            self._workers.clear()
            
        logger.info(f'Worker pool {self.name} shutdown complete.')

    def get_active_workers(self) -> int:
        """Get number of active (non-completed) workers"""
        return sum((1 for w in self._workers if not w.done()))

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._work_queue.qsize()

    def get_running_tasks(self) -> int:
        """Get current number of running tasks"""
        return self._metrics.running_tasks
    
# 1. WorkerPoolType 열거형 정의 추가
class WorkerPoolType(Enum):
    """Defines the types of worker pools available."""
    QUEUE_ASYNCIO = 'queue_asyncio'
    # 다른 타입이 있다면 추가 (예: THREAD_POOL = 'thread_pool')

# --- 싱글톤 워커 풀 인스턴스 관리 ---
_worker_pools: Dict[str, Any] = {} # 생성된 워커 풀들을 저장할 딕셔너리
_worker_pool_lock = threading.Lock()

# 2. get_worker_pool 함수 정의 추가
# def get_worker_pool(
#     name: str = 'default',
#     pool_type: WorkerPoolType = WorkerPoolType.QUEUE_ASYNCIO,
#     config: Optional[OrchestrationWorkerPoolConfig] = None
# ) -> Any: # 실제로는 QueueWorkerPool 또는 다른 풀 타입을 반환
#     """
#     지정된 이름과 타입의 Worker Pool 싱글톤 인스턴스를 가져옵니다.
#     첫 호출 시 설정을 기반으로 인스턴스를 생성합니다.
#     """
#     global _worker_pools
#     pool_key = f"{name}_{pool_type.value}" # 이름과 타입을 합쳐서 고유 키 생성

#     if pool_key not in _worker_pools:
#         with _worker_pool_lock:
#             # Lock 확보 후 다시 확인 (Double-Checked Locking)
#             if pool_key not in _worker_pools:
#                 logger.info(f"Initializing Worker Pool instance: Name='{name}', Type='{pool_type.value}'")

#                 # 설정(config)이 주어지지 않으면 기본 설정 사용
#                 pool_config = config or OrchestrationWorkerPoolConfig(
#                     workers=settings.WORKER_COUNT, # settings.py 에서 워커 수 가져오기
#                     max_concurrent_tasks=settings.MAX_CONCURRENT_TASKS # settings.py 에서 동시 작업 수 가져오기
#                     # 다른 필요한 설정들을 settings 에서 가져와서 OrchestrationWorkerPoolConfig 생성 시 전달
#                 )

#                 # 풀 타입에 따라 적절한 클래스 인스턴스 생성
#                 if pool_type == WorkerPoolType.QUEUE_ASYNCIO:
#                     # QueueWorkerPool 클래스를 사용하여 인스턴스 생성
#                     _worker_pools[pool_key] = QueueWorkerPool(name=name, config=pool_config)
#                 # elif pool_type == WorkerPoolType.THREAD_POOL:
#                 #     # 다른 타입의 워커 풀 클래스가 있다면 여기서 처리
#                 #     # _worker_pools[pool_key] = ThreadWorkerPool(name=name, config=pool_config)
#                 else:
#                     # 지원하지 않는 타입이면 오류 발생
#                     error_msg = f"Unsupported WorkerPoolType: {pool_type}"
#                     logger.error(error_msg)
#                     raise ValueError(error_msg)

#                 logger.info(f"Worker Pool instance '{pool_key}' created.")

#     # 저장된 인스턴스 반환
#     return _worker_pools[pool_key]
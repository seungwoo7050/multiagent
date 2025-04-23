# worker_pool.py
import asyncio
import concurrent.futures
import functools
import multiprocessing
import os
import threading
import time # time 모듈 import
import sys
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

from pydantic import BaseModel, Field, validator

from src.config.logger import get_logger
from src.config.metrics import track_task_completed
from src.config.settings import get_settings
from src.core.exceptions import WorkerPoolError
from src.utils.timing import AsyncTimer, Timer, get_current_time_ms

# Module logger
logger = get_logger(__name__)
settings = get_settings()

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


class WorkerPoolType(str, Enum):
    """Types of worker pools."""
    THREAD = "thread"
    PROCESS = "process"
    ASYNCIO = "asyncio"


class WorkerPoolConfig(BaseModel):
    """Configuration for worker pools."""

    # Pool type
    pool_type: WorkerPoolType = WorkerPoolType.ASYNCIO

    # Number of workers (default to CPU count)
    workers: int = max(os.cpu_count() or 1, 1)

    # Maximum tasks per worker
    max_tasks_per_worker: int = 10

    # Maximum queue size
    max_queue_size: Optional[int] = None  # None means unlimited

    # Graceful shutdown timeout in seconds
    shutdown_timeout: float = 10.0

    # Worker initialization function (for process pools)
    worker_init_func: Optional[str] = None

    # Worker timeout in seconds (0 means no timeout)
    worker_timeout: float = 0.0

    @validator('workers')
    def validate_workers(cls, v: int) -> int:
        """Validate worker count."""
        if v <= 0:
            return max(os.cpu_count() or 1, 1)
        return v


class WorkerPoolMetrics(BaseModel):
    """Metrics for worker pool monitoring."""

    # Task counters
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

    # Queue metrics
    current_queue_size: int = 0
    max_observed_queue_size: int = 0

    # Worker metrics
    active_workers: int = 0
    max_active_workers: int = 0

    # Timing metrics
    total_execution_time_ms: int = 0

    # Last update timestamp
    last_updated: int = Field(default_factory=get_current_time_ms)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class AsyncWorkerPool:
    """Asynchronous worker pool using asyncio.

    This pool is designed for I/O-bound tasks and uses asyncio's
    built-in concurrency primitives for efficient execution.
    """

    def __init__(self, name: str, config: Optional[WorkerPoolConfig] = None):
        """Initialize the async worker pool.

        Args:
            name: Name of the worker pool.
            config: Optional configuration, uses defaults if not provided.
        """
        self.name = name
        self.config = config or WorkerPoolConfig(pool_type=WorkerPoolType.ASYNCIO)

        # Override pool type to ensure it's ASYNCIO
        self.config.pool_type = WorkerPoolType.ASYNCIO

        # Metrics
        self.metrics = WorkerPoolMetrics()

        # Semaphore to limit concurrency
        self._semaphore = asyncio.Semaphore(self.config.workers)
        print(f"{time.monotonic():.4f} [INIT {self.name}]: Worker semaphore created (size: {self.config.workers})")

        # Queue for limiting total tasks
        self._queue_semaphore = None
        if self.config.max_queue_size:
            self._queue_semaphore = asyncio.Semaphore(self.config.max_queue_size)
            print(f"{time.monotonic():.4f} [INIT {self.name}]: Queue semaphore created (size: {self.config.max_queue_size})")
        else:
            print(f"{time.monotonic():.4f} [INIT {self.name}]: No queue semaphore (unlimited queue)")


        # Set of running tasks
        self._running_tasks: Set[asyncio.Task] = set()

        # Lock for updating metrics
        self._metrics_lock = asyncio.Lock()
        print(f"{time.monotonic():.4f} [INIT {self.name}]: Metrics lock created")


        # Shutdown flag
        self._is_shutdown = False

        logger.info(
            f"Initialized async worker pool: {name}",
            extra={
                "pool_name": name,
                "workers": self.config.workers,
                "max_queue_size": self.config.max_queue_size or "unlimited"
            }
        )

    async def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> Any:
        """Submit a task to the worker pool."""
        # 디버깅: 작업 식별자 추가 (함수 이름 + 첫번째 인자)
        task_id_str = f"{getattr(func, '__name__', 'unknown')}{f' (Arg0: {args[0]})' if args else ''}"
        print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Entered submit.")

        if self._is_shutdown:
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Pool is shutdown, raising error.")
            raise WorkerPoolError(
                message=f"Worker pool {self.name} is shutdown"
            )

        acquired_queue_semaphore = False
        # 큐 세마포어 획득 시도
        if self._queue_semaphore:
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Attempting to acquire queue semaphore...")
            try:
                await self._queue_semaphore.acquire()
                acquired_queue_semaphore = True
                print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Acquired queue semaphore.")
            except Exception as e:
                 print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Failed to acquire queue semaphore ({e}), raising error.")
                 logger.error(f"Failed to acquire queue semaphore for pool {self.name}: {e}")
                 # 실패 시 WorkerPoolError 발생 (이전 로직 복원)
                 raise WorkerPoolError(
                     message=f"Worker pool {self.name} queue acquisition failed.",
                     original_error=e
                 )
        else:
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: No queue limit, skipping queue semaphore.")


        # 메트릭 업데이트
        task_instance = None # task 변수 초기화
        try:
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Attempting to acquire metrics lock for submission...")
            async with self._metrics_lock:
                print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Acquired metrics lock. Updating submission metrics...")
                self.metrics.tasks_submitted += 1
                if acquired_queue_semaphore: # 큐 세마포어를 획득한 경우에만 증가
                    self.metrics.current_queue_size += 1
                    self.metrics.max_observed_queue_size = max(
                        self.metrics.max_observed_queue_size,
                        self.metrics.current_queue_size
                    )
                self.metrics.last_updated = get_current_time_ms()
                print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Submission metrics updated (Submitted: {self.metrics.tasks_submitted}, Queue: {self.metrics.current_queue_size}). Released metrics lock.")

            # 실제 Task 생성
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Creating asyncio task...")
            task_instance = asyncio.create_task(
                self._execute_task(func, *args, **kwargs),
                # Task 이름에 식별자 추가
                name=f"{self.name}_task_{task_id_str}_{self.metrics.tasks_submitted}"
            )
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Task created: {task_instance.get_name()}")


            # 실행 중인 Task 목록에 추가
            self._running_tasks.add(task_instance)
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Added task to running set (Size: {len(self._running_tasks)})")

            # 완료 콜백 설정 (큐 세마포어 해제 포함)
            def task_done_callback(fut: asyncio.Task):
                callback_task_id = fut.get_name() # 콜백에서 Task 이름 가져오기
                print(f"{time.monotonic():.4f} [Callback {callback_task_id}]: Task done callback started.")
                self._running_tasks.discard(fut)
                print(f"{time.monotonic():.4f} [Callback {callback_task_id}]: Removed task from running set (New size: {len(self._running_tasks)})")

                # 이 콜백이 실행될 때의 acquired_queue_semaphore 값을 사용해야 함
                # 클로저를 통해 submit 시점의 값이 캡처됨
                if acquired_queue_semaphore and self._queue_semaphore:
                     print(f"{time.monotonic():.4f} [Callback {callback_task_id}]: Attempting to release queue semaphore (acquired={acquired_queue_semaphore})...")
                     try:
                        self._queue_semaphore.release()
                        print(f"{time.monotonic():.4f} [Callback {callback_task_id}]: Released queue semaphore.")
                     except Exception as e:
                         print(f"{time.monotonic():.4f} [Callback {callback_task_id}]: Error releasing queue semaphore: {e}")
                         logger.error(f"Error releasing queue semaphore in task callback for pool {self.name}: {e}")
                else:
                    print(f"{time.monotonic():.4f} [Callback {callback_task_id}]: No queue semaphore to release (acquired={acquired_queue_semaphore}).")
                print(f"{time.monotonic():.4f} [Callback {callback_task_id}]: Task done callback finished.")


            task_instance.add_done_callback(task_done_callback)
            print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Added done callback.")


            # 타임아웃 처리
            effective_timeout = timeout if timeout is not None else self.config.worker_timeout
            if effective_timeout > 0:
                print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Waiting for task with timeout: {effective_timeout}s")
                try:
                    result = await asyncio.wait_for(task_instance, timeout=effective_timeout)
                    print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Task completed within timeout.")
                    return result
                except asyncio.TimeoutError:
                    print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Task timed out after {effective_timeout}s.")
                    logger.warning(
                        f"Task in worker pool {self.name} timed out after {effective_timeout}s",
                        extra={ "pool_name": self.name, "timeout": effective_timeout, "func_name": task_id_str }
                    )
                    if not task_instance.done():
                         print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Cancelling timed out task...")
                         task_instance.cancel()
                         print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Timed out task cancelled.")
                    raise # TimeoutError를 다시 발생
            else:
                # 타임아웃 없음
                print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Waiting for task without timeout...")
                result = await task_instance
                print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Task completed without timeout.")
                return result

        except Exception as e:
             print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Exception occurred during submit/execution: {e}")
             # 예외 발생 시 획득했던 큐 세마포어 해제 시도
             if acquired_queue_semaphore and self._queue_semaphore:
                 print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Attempting to release queue semaphore due to exception...")
                 try:
                    self._queue_semaphore.release()
                    print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Released queue semaphore on exception.")
                 except Exception as release_error:
                     print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Error releasing queue semaphore on exception: {release_error}")
                     logger.error(f"Error releasing queue semaphore during submit exception for pool {self.name}: {release_error}")

             # 메트릭 롤백 시도 (큐 사이즈만, 제출 카운트는 유지)
             print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Attempting to acquire metrics lock for rollback...")
             async with self._metrics_lock:
                 print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Acquired metrics lock. Rolling back queue size metric...")
                 if acquired_queue_semaphore: # 큐 세마포어를 획득했던 경우에만 롤백
                    self.metrics.current_queue_size = max(0, self.metrics.current_queue_size -1)
                    print(f"{time.monotonic():.4f} [Submit {task_id_str}]: Queue size metric rolled back (New Queue: {self.metrics.current_queue_size}). Released metrics lock.")
                 else:
                    print(f"{time.monotonic():.4f} [Submit {task_id_str}]: No queue semaphore acquired, no queue size rollback needed. Released metrics lock.")

             raise # 원래 에러 다시 발생

    async def _execute_task(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        current_task = asyncio.current_task()
        task_id_str = current_task.get_name() if current_task else "unknown_task"

        # === 메트릭 업데이트 위치 변경 시작 ===
        print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Attempting to acquire metrics lock (before semaphore)...")
        async with self._metrics_lock:
            print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Acquired metrics lock. Updating pre-execution queue metric...")
            # 큐 크기 감소 로직을 워커 세마포어 획득 전으로 이동
            self.metrics.current_queue_size = max(0, self.metrics.current_queue_size - 1)
            self.metrics.last_updated = get_current_time_ms()
            print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Pre-execution queue metric updated (Queue: {self.metrics.current_queue_size}). Released metrics lock.")
        # === 메트릭 업데이트 위치 변경 끝 ===

        print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Waiting for worker semaphore...")
        try:
            async with self._semaphore: # 워커 세마포어 획득
                print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Acquired worker semaphore.")

                # 활성 워커 수 증가는 워커 세마포어 획득 후에 수행
                print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Attempting to acquire metrics lock (after semaphore)...")
                async with self._metrics_lock:
                     print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Acquired metrics lock. Updating active worker metric...")
                     self.metrics.active_workers += 1
                     self.metrics.max_active_workers = max(
                         self.metrics.max_active_workers,
                         self.metrics.active_workers
                     )
                     self.metrics.last_updated = get_current_time_ms()
                     print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Active worker metric updated (Active: {self.metrics.active_workers}). Released metrics lock.")

                start_time = get_current_time_ms()
                success = False
                result = None

                try:
                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Executing func...")
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Running sync func in executor...")
                        result = await asyncio.to_thread(func, *args, **kwargs)
                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Func execution finished successfully.")
                    success = True
                    # 결과는 finally 이후 반환

                except asyncio.CancelledError:
                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Task cancelled during execution.")
                    success = False # 실패로 처리
                    raise # CancelledError는 다시 발생시켜야 함
                except Exception as e:
                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Exception during func execution: {e}")
                    success = False # 실패로 처리
                    # === 중요 복원: 잡은 예외를 다시 발생시켜야 submit 에서 인지 가능 ===
                    raise e
                    # =======================================================
                finally:
                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Entering finally block (Success: {success}).")
                    execution_time = get_current_time_ms() - start_time

                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Attempting to acquire metrics lock (in finally)...")
                    async with self._metrics_lock:
                        print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Acquired metrics lock. Updating post-execution metrics...")
                        if success:
                            self.metrics.tasks_completed += 1
                        else:
                            self.metrics.tasks_failed += 1

                        self.metrics.active_workers = max(0, self.metrics.active_workers - 1) # 음수 방지
                        self.metrics.total_execution_time_ms += execution_time
                        self.metrics.last_updated = get_current_time_ms()
                        print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Post-execution metrics updated (Completed: {self.metrics.tasks_completed}, Failed: {self.metrics.tasks_failed}, Active: {self.metrics.active_workers}). Released metrics lock.")

                    metric_status = "success" if success else "failure"
                    track_task_completed(metric_status, execution_time / 1000.0)
                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Metric tracked ({metric_status}).")

                    print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Releasing worker semaphore...")

                # try 블록에서 예외 없이 성공적으로 완료되었을 경우 결과 반환
                if success:
                   return result

        # === 수정: 바깥쪽 Exception 블록에서 tasks_failed 증가 로직 제거 ===
        except Exception as outer_e:
            print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Outer exception in _execute_task: {outer_e}")
            # self.metrics.tasks_failed += 1 # 이 라인 제거
            raise
        # === 수정 끝 ===

        print(f"{time.monotonic():.4f} [_execute_task {task_id_str}]: Finished.")


    async def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None
    ) -> List[R]:
        """Apply a function to each item in parallel."""
        tasks = []
        print(f"{time.monotonic():.4f} [Map]: Starting map operation for {len(items)} items.")
        for i, item in enumerate(items):
            # map 내부 submit 호출은 개별 로그를 남김
            task = self.submit(func, item, timeout=timeout)
            tasks.append(task)
        print(f"{time.monotonic():.4f} [Map]: All {len(items)} tasks submitted. Gathering results...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"{time.monotonic():.4f} [Map]: Results gathered.")
        return results

    def get_active_workers(self) -> int:
        """Get the number of active workers."""
        # 이 함수는 메트릭 값만 반환하므로 추가 로그 불필요
        return self.metrics.active_workers

    def get_queue_size(self) -> int:
        """Get the current queue size."""
        # 이 함수는 메트릭 값만 반환하므로 추가 로그 불필요
        return self.metrics.current_queue_size

    def get_total_tasks(self) -> int:
        """Get the total number of tasks submitted."""
        return self.metrics.tasks_submitted

    async def wait_until_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait until all tasks are completed and the queue is empty."""
        start_time = time.monotonic()
        check_interval = 0.01
        print(f"{time.monotonic():.4f} [WaitUntilIdle]: Starting wait (Timeout: {timeout}s)")

        while True:
            # 상태 확인을 위한 락 획득
            # print(f"{time.monotonic():.4f} [WaitUntilIdle]: Attempting metrics lock for state check...")
            async with self._metrics_lock:
                # print(f"{time.monotonic():.4f} [WaitUntilIdle]: Acquired metrics lock.")
                current_running_tasks = list(self._running_tasks)
                queued_tasks_count = self.metrics.current_queue_size
                active_workers_count = self.metrics.active_workers
                # print(f"{time.monotonic():.4f} [WaitUntilIdle]: Releasing metrics lock.")

            # 실제 Task 객체의 완료 여부 확인
            all_tasks_done = all(task.done() for task in current_running_tasks)
            is_truly_idle = (queued_tasks_count == 0 and active_workers_count == 0 and all_tasks_done)

            print(f"{time.monotonic():.4f} [WaitUntilIdle]: Checking state - Queue: {queued_tasks_count}, Active: {active_workers_count}, Running Tasks: {len(current_running_tasks)}, All Tasks Done: {all_tasks_done}, Truly Idle: {is_truly_idle}")


            if is_truly_idle:
                 # 한번 더 확인하여 경쟁 상태 가능성 줄이기
                 print(f"{time.monotonic():.4f} [WaitUntilIdle]: Idle detected, performing final check...")
                 await asyncio.sleep(check_interval / 2)
                 async with self._metrics_lock:
                      current_running_tasks = list(self._running_tasks)
                      queued_tasks_count = self.metrics.current_queue_size
                      active_workers_count = self.metrics.active_workers
                 all_tasks_done = all(task.done() for task in current_running_tasks)
                 final_check_idle = (queued_tasks_count == 0 and active_workers_count == 0 and all_tasks_done)

                 if final_check_idle:
                    print(f"{time.monotonic():.4f} [WaitUntilIdle]: Final check confirmed idle. Returning True.")
                    return True
                 else:
                    print(f"{time.monotonic():.4f} [WaitUntilIdle]: Final check failed idle. Continuing wait.")


            # 타임아웃 확인
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    print(f"{time.monotonic():.4f} [WaitUntilIdle]: Timeout reached ({elapsed:.4f}s >= {timeout}s). Returning False.")
                    return False

            # 다음 확인 전 대기
            await asyncio.sleep(check_interval)


    async def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the worker pool."""
        print(f"{time.monotonic():.4f} [Shutdown]: Shutdown requested (Wait: {wait}, Timeout: {timeout})")
        if self._is_shutdown:
            print(f"{time.monotonic():.4f} [Shutdown]: Already shutdown.")
            return

        self._is_shutdown = True
        print(f"{time.monotonic():.4f} [Shutdown]: Shutdown flag set.")

        running_tasks_copy = list(self._running_tasks) # 복사본 사용

        if wait and running_tasks_copy:
            effective_timeout = timeout or self.config.shutdown_timeout
            print(f"{time.monotonic():.4f} [Shutdown]: Waiting for {len(running_tasks_copy)} running tasks to complete (Timeout: {effective_timeout}s)...")

            try:
                done, pending = await asyncio.wait(
                    running_tasks_copy,
                    timeout=effective_timeout
                )
                print(f"{time.monotonic():.4f} [Shutdown]: Wait finished. Done: {len(done)}, Pending: {len(pending)}")

                if pending:
                    print(f"{time.monotonic():.4f} [Shutdown]: Cancelling {len(pending)} pending tasks...")
                    cancelled_count = 0
                    for task in pending:
                        if not task.done():
                            task.cancel()
                            cancelled_count += 1
                    print(f"{time.monotonic():.4f} [Shutdown]: Cancellation requests sent for {cancelled_count} tasks.")

                    # 취소 완료 대기
                    print(f"{time.monotonic():.4f} [Shutdown]: Waiting for cancellations to complete...")
                    await asyncio.gather(*pending, return_exceptions=True) # gather로 기다림
                    print(f"{time.monotonic():.4f} [Shutdown]: Cancellations complete.")

                logger.info(f"Worker pool {self.name} shutdown complete: {len(done)} tasks completed, {len(pending)} tasks cancelled/pending")
            except Exception as e:
                print(f"{time.monotonic():.4f} [Shutdown]: Error during wait/cancel: {e}")
                logger.error(f"Error during worker pool {self.name} shutdown: {e}", exc_info=True)
        else:
            # 즉시 취소
            if running_tasks_copy:
                print(f"{time.monotonic():.4f} [Shutdown]: Cancelling {len(running_tasks_copy)} running tasks immediately...")
                cancelled_count = 0
                for task in running_tasks_copy:
                    if not task.done():
                        task.cancel()
                        cancelled_count += 1
                print(f"{time.monotonic():.4f} [Shutdown]: Cancellation requests sent for {cancelled_count} tasks.")
                 # 취소 완료 대기 (선택적이지만 권장)
                print(f"{time.monotonic():.4f} [Shutdown]: Waiting for immediate cancellations to complete...")
                await asyncio.gather(*running_tasks_copy, return_exceptions=True)
                print(f"{time.monotonic():.4f} [Shutdown]: Immediate cancellations complete.")
            else:
                print(f"{time.monotonic():.4f} [Shutdown]: No running tasks to cancel.")


            logger.info(f"Worker pool {self.name} shutdown complete: {len(running_tasks_copy)} tasks cancelled")
        print(f"{time.monotonic():.4f} [Shutdown]: Shutdown process finished.")


    async def __aenter__(self) -> "AsyncWorkerPool":
        """Context manager entry."""
        print(f"{time.monotonic():.4f} [Context Mgr]: Entering async context.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        print(f"{time.monotonic():.4f} [Context Mgr]: Exiting async context. Initiating shutdown...")
        await self.shutdown()
        print(f"{time.monotonic():.4f} [Context Mgr]: Shutdown complete in aexit.")

# ... (ThreadWorkerPool 및 ProcessWorkerPool 정의는 그대로 유지) ...

class ThreadWorkerPool:
    """Thread-based worker pool for CPU-bound tasks.
    
    This pool is designed for tasks that primarily use CPU but may
    occasionally block on I/O operations.
    """
    
    def __init__(self, name: str, config: Optional[WorkerPoolConfig] = None):
        """Initialize the thread worker pool.
        
        Args:
            name: Name of the worker pool.
            config: Optional configuration, uses defaults if not provided.
        """
        self.name = name
        self.config = config or WorkerPoolConfig(pool_type=WorkerPoolType.THREAD)
        
        # Override pool type to ensure it's THREAD
        self.config.pool_type = WorkerPoolType.THREAD
        
        # Metrics
        self.metrics = WorkerPoolMetrics()
        
        # Create thread pool executor
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.workers,
            thread_name_prefix=f"{name}_worker"
        )
        
        # Track running futures
        self._futures: Set[concurrent.futures.Future] = set()
        
        # Lock for updating metrics and futures
        self._lock = threading.RLock()
        
        # Queue semaphore for limiting total tasks
        self._queue_semaphore = None
        if self.config.max_queue_size:
            self._queue_semaphore = threading.BoundedSemaphore(self.config.max_queue_size)
        
        # Shutdown flag
        self._is_shutdown = False
        
        logger.info(
            f"Initialized thread worker pool: {name}",
            extra={
                "pool_name": name,
                "workers": self.config.workers,
                "max_queue_size": self.config.max_queue_size or "unlimited"
            }
        )
    
    def _update_metrics(self, task_submitted: bool = False, task_completed: bool = False,
                     task_failed: bool = False, active_delta: int = 0, queue_delta: int = 0):
        """Update pool metrics.
        
        Args:
            task_submitted: Whether a task was submitted.
            task_completed: Whether a task was completed.
            task_failed: Whether a task failed.
            active_delta: Change in active workers.
            queue_delta: Change in queue size.
        """
        with self._lock:
            if task_submitted:
                self.metrics.tasks_submitted += 1
            
            if task_completed:
                self.metrics.tasks_completed += 1
            
            if task_failed:
                self.metrics.tasks_failed += 1
            
            if active_delta != 0:
                self.metrics.active_workers += active_delta
                self.metrics.max_active_workers = max(
                    self.metrics.max_active_workers,
                    self.metrics.active_workers
                )
            
            if queue_delta != 0:
                self.metrics.current_queue_size += queue_delta
                if queue_delta > 0:
                    self.metrics.max_observed_queue_size = max(
                        self.metrics.max_observed_queue_size,
                        self.metrics.current_queue_size
                    )
            
            self.metrics.last_updated = get_current_time_ms()
    
    def _task_done_callback(self, future: concurrent.futures.Future) -> None:
        """Callback when a task is done.
        
        Args:
            future: The completed future.
        """
        # Update metrics
        success = not future.exception()
        self._update_metrics(
            task_completed=success,
            task_failed=not success,
            active_delta=-1,
            queue_delta=0
        )
        
        # Remove from tracked futures
        with self._lock:
            self._futures.discard(future)
        
        # Release queue semaphore if configured
        if self._queue_semaphore:
            self._queue_semaphore.release()
        
        # Track task completion for metrics
        if hasattr(future, '_start_time'):
            execution_time = (get_current_time_ms() - future._start_time) / 1000.0
            if success:
                track_task_completed("success", execution_time)
            else:
                track_task_completed("failure", execution_time)
    
    def submit(
        self,
        func: Callable[..., R],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> concurrent.futures.Future:
        """Submit a task to the worker pool.
        
        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            timeout: Optional timeout in seconds (overrides config).
            **kwargs: Keyword arguments for the function.
            
        Returns:
            concurrent.futures.Future: Future representing the execution.
            
        Raises:
            WorkerPoolError: If the pool is shutdown or the queue is full.
        """
        if self._is_shutdown:
            raise WorkerPoolError(
                message=f"Thread worker pool {self.name} is shutdown"
            )
        
        # Try to acquire queue semaphore if configured
        if self._queue_semaphore:
            acquired = self._queue_semaphore.acquire(blocking=False)
            if not acquired:
                raise WorkerPoolError(
                    message=f"Thread worker pool {self.name} queue is full"
                )
        
        # Wrap function to track execution time
        @functools.wraps(func)
        def tracked_func(*func_args, **func_kwargs):
            start_time = get_current_time_ms()
            
            # Update metrics
            self._update_metrics(active_delta=1, queue_delta=-1)
            
            try:
                return func(*func_args, **func_kwargs)
            finally:
                # Update execution time
                execution_time = get_current_time_ms() - start_time
                with self._lock:
                    self.metrics.total_execution_time_ms += execution_time
        
        # Update metrics
        self._update_metrics(task_submitted=True, queue_delta=1)
        
        # Submit to executor
        future = self._executor.submit(tracked_func, *args, **kwargs)
        future._start_time = get_current_time_ms()  # Add start time to future
        
        # Add done callback
        future.add_done_callback(self._task_done_callback)
        
        # Track future
        with self._lock:
            self._futures.add(future)
        
        return future
    
    def submit_async(
        self,
        func: Callable[..., R],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> asyncio.Future:
        """Submit a task to the worker pool with async interface.
        
        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            timeout: Optional timeout in seconds (overrides config).
            **kwargs: Keyword arguments for the function.
            
        Returns:
            asyncio.Future: Future representing the execution.
        """
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        # Create a future in the event loop
        future = loop.create_future()
        
        def done_callback(thread_future):
            """Callback to transfer result to asyncio future."""
            if thread_future.cancelled():
                loop.call_soon_threadsafe(future.cancel)
                return
            
            exc = thread_future.exception()
            if exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                loop.call_soon_threadsafe(future.set_result, thread_future.result())
        
        try:
            # Submit to thread pool
            thread_future = self.submit(func, *args, timeout=timeout, **kwargs)
            thread_future.add_done_callback(done_callback)
        except Exception as e:
            loop.call_soon_threadsafe(future.set_exception, e)
        
        return future
    
    async def asubmit(
        self,
        func: Callable[..., R],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> R:
        """Submit a task to the worker pool with async interface and wait for result.
        
        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            timeout: Optional timeout in seconds (overrides config).
            **kwargs: Keyword arguments for the function.
            
        Returns:
            R: The result of the function execution.
        """
        future = self.submit_async(func, *args, timeout=timeout, **kwargs)
        
        effective_timeout = timeout or self.config.worker_timeout
        if effective_timeout > 0:
            return await asyncio.wait_for(future, timeout=effective_timeout)
        else:
            return await future
    
    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None
    ) -> List[R]:
        """Apply a function to each item in parallel.
        
        Args:
            func: Function to apply to each item.
            items: List of items to process.
            timeout: Optional timeout for the entire operation.
            
        Returns:
            List[R]: List of results in the same order as items.
        """
        # Submit all items
        futures = [self.submit(func, item) for item in items]
        
        # Wait for all futures to complete
        if timeout:
            done, not_done = concurrent.futures.wait(
                futures,
                timeout=timeout,
                return_when=concurrent.futures.ALL_COMPLETED
            )
            
            # Cancel any remaining futures
            for future in not_done:
                future.cancel()
            
            # Gather results from completed futures
            results = []
            for future in done:
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(e)
            
            # Add exceptions for cancelled futures
            for _ in not_done:
                results.append(TimeoutError("Task timed out"))
            
            return results
        else:
            # Wait for all futures to complete
            return [future.result() for future in concurrent.futures.as_completed(futures)]
    
    async def amap(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None
    ) -> List[R]:
        """Apply a function to each item in parallel with async interface.
        
        Args:
            func: Function to apply to each item.
            items: List of items to process.
            timeout: Optional timeout for the entire operation.
            
        Returns:
            List[R]: List of results in the same order as items.
        """
        # Submit all items
        futures = [self.submit_async(func, item) for item in items]
        
        if timeout:
            # Wait for all futures with timeout
            return await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True),
                timeout=timeout
            )
        else:
            # Wait for all futures
            return await asyncio.gather(*futures, return_exceptions=True)
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the worker pool.
        
        Args:
            wait: Whether to wait for running tasks to complete.
            timeout: Maximum time to wait in seconds.
        """
        if self._is_shutdown:
            return
        
        self._is_shutdown = True
        logger.info(f"Shutting down thread worker pool: {self.name}")
        
        effective_timeout = timeout or self.config.shutdown_timeout
        
        try:
            # Shutdown executor
            self._executor.shutdown(wait=wait, timeout=effective_timeout)
            logger.info(f"Thread worker pool {self.name} shutdown complete")
        except Exception as e:
            logger.error(
                f"Error during thread worker pool {self.name} shutdown: {e}",
                exc_info=True
            )
    
    async def ashutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the worker pool with async interface.
        
        Args:
            wait: Whether to wait for running tasks to complete.
            timeout: Maximum time to wait in seconds.
        """
        # Run shutdown in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            functools.partial(self.shutdown, wait=wait, timeout=timeout)
        )
    
    def __enter__(self) -> "ThreadWorkerPool":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
    
    async def __aenter__(self) -> "ThreadWorkerPool":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.ashutdown()


class ProcessWorkerPool:
    """Process-based worker pool for CPU-bound tasks.
    
    This pool is designed for CPU-intensive tasks that benefit
    from parallel execution across multiple processes.
    """
    
    def __init__(self, name: str, config: Optional[WorkerPoolConfig] = None):
        """Initialize the process worker pool.
        
        Args:
            name: Name of the worker pool.
            config: Optional configuration, uses defaults if not provided.
        """
        self.name = name
        self.config = config or WorkerPoolConfig(pool_type=WorkerPoolType.PROCESS)
        
        # Override pool type to ensure it's PROCESS
        self.config.pool_type = WorkerPoolType.PROCESS
        
        # Metrics
        self.metrics = WorkerPoolMetrics()
        
        # Create process pool executor
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config.workers,
            mp_context=multiprocessing.get_context('spawn')
        )
        
        # Track running futures
        self._futures: Set[concurrent.futures.Future] = set()
        
        # Lock for updating metrics and futures
        self._lock = threading.RLock()
        
        # Queue semaphore for limiting total tasks
        self._queue_semaphore = None
        if self.config.max_queue_size:
            self._queue_semaphore = threading.BoundedSemaphore(self.config.max_queue_size)
        
        # Shutdown flag
        self._is_shutdown = False
        
        logger.info(
            f"Initialized process worker pool: {name}",
            extra={
                "pool_name": name,
                "workers": self.config.workers,
                "max_queue_size": self.config.max_queue_size or "unlimited"
            }
        )
    
    def _update_metrics(self, task_submitted: bool = False, task_completed: bool = False,
                     task_failed: bool = False, active_delta: int = 0, queue_delta: int = 0):
        """Update pool metrics.
        
        Args:
            task_submitted: Whether a task was submitted.
            task_completed: Whether a task was completed.
            task_failed: Whether a task failed.
            active_delta: Change in active workers.
            queue_delta: Change in queue size.
        """
        with self._lock:
            if task_submitted:
                self.metrics.tasks_submitted += 1
            
            if task_completed:
                self.metrics.tasks_completed += 1
            
            if task_failed:
                self.metrics.tasks_failed += 1
            
            if active_delta != 0:
                self.metrics.active_workers += active_delta
                self.metrics.max_active_workers = max(
                    self.metrics.max_active_workers,
                    self.metrics.active_workers
                )
            
            if queue_delta != 0:
                self.metrics.current_queue_size += queue_delta
                if queue_delta > 0:
                    self.metrics.max_observed_queue_size = max(
                        self.metrics.max_observed_queue_size,
                        self.metrics.current_queue_size
                    )
            
            self.metrics.last_updated = get_current_time_ms()
    
    def _task_done_callback(self, future: concurrent.futures.Future) -> None:
        """Callback when a task is done.
        
        Args:
            future: The completed future.
        """
        # Update metrics
        success = not future.exception()
        self._update_metrics(
            task_completed=success,
            task_failed=not success,
            active_delta=-1,
            queue_delta=0
        )
        
        # Remove from tracked futures
        with self._lock:
            self._futures.discard(future)
        
        # Release queue semaphore if configured
        if self._queue_semaphore:
            self._queue_semaphore.release()
        
        # Track task completion for metrics
        if hasattr(future, '_start_time'):
            execution_time = (get_current_time_ms() - future._start_time) / 1000.0
            if success:
                track_task_completed("success", execution_time)
            else:
                track_task_completed("failure", execution_time)
    
    def submit(
        self,
        func: Callable[..., R],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> concurrent.futures.Future:
        """Submit a task to the worker pool.
        
        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            timeout: Optional timeout in seconds (overrides config).
            **kwargs: Keyword arguments for the function.
            
        Returns:
            concurrent.futures.Future: Future representing the execution.
            
        Raises:
            WorkerPoolError: If the pool is shutdown or the queue is full.
        """
        if self._is_shutdown:
            raise WorkerPoolError(
                message=f"Process worker pool {self.name} is shutdown"
            )
        
        # Try to acquire queue semaphore if configured
        if self._queue_semaphore:
            acquired = self._queue_semaphore.acquire(blocking=False)
            if not acquired:
                raise WorkerPoolError(
                    message=f"Process worker pool {self.name} queue is full"
                )
        
        # Ensure function is picklable
        if not _is_picklable(func):
            raise WorkerPoolError(
                message=f"Function must be picklable for process pool: {func.__name__}"
            )
        
        # Update metrics
        self._update_metrics(task_submitted=True, queue_delta=1, active_delta=1)
        
        # Submit to executor
        try:
            future = self._executor.submit(func, *args, **kwargs)
            future._start_time = get_current_time_ms()  # Add start time to future
            
            # Add done callback
            future.add_done_callback(self._task_done_callback)
            
            # Track future
            with self._lock:
                self._futures.add(future)
            
            return future
        except Exception as e:
            # Release semaphore on error
            if self._queue_semaphore:
                self._queue_semaphore.release()
            
            # Update metrics on error
            self._update_metrics(task_failed=True, queue_delta=-1, active_delta=-1)
            
            logger.exception(
                f"Error submitting task to process pool {self.name}: {e}",
                exc_info=True
            )
            raise WorkerPoolError(
                message=f"Error submitting task: {str(e)}",
                original_error=e
            )
    
    def submit_async(
        self,
        func: Callable[..., R],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> asyncio.Future:
        """Submit a task to the worker pool with async interface.
        
        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            timeout: Optional timeout in seconds (overrides config).
            **kwargs: Keyword arguments for the function.
            
        Returns:
            asyncio.Future: Future representing the execution.
        """
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        # Create a future in the event loop
        future = loop.create_future()
        
        def done_callback(process_future):
            """Callback to transfer result to asyncio future."""
            if process_future.cancelled():
                loop.call_soon_threadsafe(future.cancel)
                return
            
            exc = process_future.exception()
            if exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                loop.call_soon_threadsafe(future.set_result, process_future.result())
        
        try:
            # Submit to process pool
            process_future = self.submit(func, *args, timeout=timeout, **kwargs)
            process_future.add_done_callback(done_callback)
        except Exception as e:
            loop.call_soon_threadsafe(future.set_exception, e)
        
        return future
    
    async def asubmit(
        self,
        func: Callable[..., R],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> R:
        """Submit a task to the worker pool with async interface and wait for result.
        
        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            timeout: Optional timeout in seconds (overrides config).
            **kwargs: Keyword arguments for the function.
            
        Returns:
            R: The result of the function execution.
        """
        future = self.submit_async(func, *args, timeout=timeout, **kwargs)
        
        effective_timeout = timeout or self.config.worker_timeout
        if effective_timeout > 0:
            return await asyncio.wait_for(future, timeout=effective_timeout)
        else:
            return await future
    
    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None
    ) -> List[R]:
        """Apply a function to each item in parallel.
        
        Args:
            func: Function to apply to each item.
            items: List of items to process.
            timeout: Optional timeout for the entire operation.
            
        Returns:
            List[R]: List of results in the same order as items.
        """
        # Ensure function is picklable
        if not _is_picklable(func):
            raise WorkerPoolError(
                message=f"Function must be picklable for process pool: {func.__name__}"
            )
        
        # For small batches, use submit approach for better tracking
        if len(items) <= self.config.workers * 2:
            # Submit all items
            futures = [self.submit(func, item) for item in items]
            
            # Wait for all futures to complete
            if timeout:
                done, not_done = concurrent.futures.wait(
                    futures,
                    timeout=timeout,
                    return_when=concurrent.futures.ALL_COMPLETED
                )
                
                # Cancel any remaining futures
                for future in not_done:
                    future.cancel()
                
                # Gather results from completed futures
                results = []
                for future in done:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append(e)
                
                # Add exceptions for cancelled futures
                for _ in not_done:
                    results.append(TimeoutError("Task timed out"))
                
                return results
            else:
                # Wait for all futures to complete
                return [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            # For larger batches, use the executor's map function for efficiency
            try:
                if timeout:
                    return list(self._executor.map(func, items, timeout=timeout))
                else:
                    return list(self._executor.map(func, items))
            except Exception as e:
                logger.exception(
                    f"Error in process pool map operation: {e}",
                    exc_info=True
                )
                raise WorkerPoolError(
                    message=f"Error in map operation: {str(e)}",
                    original_error=e
                )
    
    async def amap(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None
    ) -> List[R]:
        """Apply a function to each item in parallel with async interface.
        
        Args:
            func: Function to apply to each item.
            items: List of items to process.
            timeout: Optional timeout for the entire operation.
            
        Returns:
            List[R]: List of results in the same order as items.
        """
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        # Run map operation in a separate thread to avoid blocking
        return await loop.run_in_executor(
            None,
            functools.partial(self.map, func, items, timeout)
        )
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the worker pool.
        
        Args:
            wait: Whether to wait for running tasks to complete.
            timeout: Maximum time to wait in seconds.
        """
        if self._is_shutdown:
            return
        
        self._is_shutdown = True
        logger.info(f"Shutting down process worker pool: {self.name}")
        
        try:
            if sys.version_info >= (3, 9):
                # Python 3.9 이상: timeout 인자 사용
                self._executor.shutdown(wait=wait, timeout=timeout or self.config.shutdown_timeout)
            else:
                # Python 3.9 미만: timeout 인자 사용 안 함
                self._executor.shutdown(wait=wait)
            logger.info(f"Process worker pool {self.name} shutdown complete")
        except Exception as e:
            logger.error(
                f"Error during process worker pool {self.name} shutdown: {e}",
                exc_info=True
            )
    
    async def ashutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the worker pool with async interface.
        
        Args:
            wait: Whether to wait for running tasks to complete.
            timeout: Maximum time to wait in seconds.
        """
        # Run shutdown in a separate thread to avoid blocking
        import functools
        loop = asyncio.get_event_loop()
        # functools.partial을 사용하여 shutdown 메서드에 인자 전달
        await loop.run_in_executor(
            None,
            functools.partial(self.shutdown, wait=wait, timeout=timeout) # functools.partial 사용
        )
    
    def __enter__(self) -> "ProcessWorkerPool":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()
    
    async def __aenter__(self) -> "ProcessWorkerPool":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.ashutdown()


def _is_picklable(obj: Any) -> bool:
    """Check if an object is picklable (for process pools).
    
    Args:
        obj: Object to check.
        
    Returns:
        bool: True if the object is likely picklable.
    """
    # Check for common non-picklable patterns
    if hasattr(obj, '__self__'):
        # Bound methods are not picklable
        return False
    
    # Check if object is defined in __main__
    module = getattr(obj, '__module__', None)
    if module == '__main__':
        logger.warning(f"Object {obj} is defined in __main__, which may not be picklable")
    
    # We can't reliably check further without actually trying to pickle
    return True


# Worker pool registry for global access
_worker_pools: Dict[str, Union[AsyncWorkerPool, ThreadWorkerPool, ProcessWorkerPool]] = {}


def get_worker_pool(
    name: str,
    pool_type: WorkerPoolType = WorkerPoolType.ASYNCIO,
    config: Optional[WorkerPoolConfig] = None
) -> Union[AsyncWorkerPool, ThreadWorkerPool, ProcessWorkerPool]:
    """Get or create a worker pool by name.
    
    Args:
        name: Name of the worker pool.
        pool_type: Type of worker pool to create.
        config: Optional configuration for new pools.
        
    Returns:
        Worker pool instance of the appropriate type.
    """
    key = f"{name}:{pool_type.value}"
    
    if key not in _worker_pools:
        config_obj = config or WorkerPoolConfig(pool_type=pool_type)
        
        if pool_type == WorkerPoolType.ASYNCIO:
            _worker_pools[key] = AsyncWorkerPool(name, config_obj)
        elif pool_type == WorkerPoolType.THREAD:
            _worker_pools[key] = ThreadWorkerPool(name, config_obj)
        elif pool_type == WorkerPoolType.PROCESS:
            _worker_pools[key] = ProcessWorkerPool(name, config_obj)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
    
    return _worker_pools[key]


def get_default_worker_pool() -> AsyncWorkerPool:
    """Get the default worker pool (async).
    
    Returns:
        AsyncWorkerPool: Default worker pool instance.
    """
    return cast(AsyncWorkerPool, get_worker_pool("default", WorkerPoolType.ASYNCIO))


async def shutdown_all_worker_pools(wait: bool = True, timeout: Optional[float] = None) -> None:
    """Shutdown all worker pools.
    
    Args:
        wait: Whether to wait for running tasks to complete.
        timeout: Maximum time to wait in seconds.
    """
    shutdown_tasks = []
    
    for key, pool in _worker_pools.items():
        if isinstance(pool, AsyncWorkerPool):
            shutdown_tasks.append(pool.shutdown(wait=wait, timeout=timeout))
        elif isinstance(pool, (ThreadWorkerPool, ProcessWorkerPool)):
            shutdown_tasks.append(pool.ashutdown(wait=wait, timeout=timeout))
    
    # Wait for all shutdowns to complete
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks)
    
    # Clear the registry
    _worker_pools.clear()
    
    logger.info("All worker pools shut down")
    
    
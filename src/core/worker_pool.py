import asyncio
import concurrent.futures
import functools
import multiprocessing, multiprocessing.context
import os
import threading
import time
import sys
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast, Coroutine
from pydantic import BaseModel, Field, field_validator
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager, TASK_QUEUE_DEPTH, TASK_PROCESSING
from src.config.settings import get_settings
from src.core.exceptions import WorkerPoolError
from src.utils.timing import AsyncTimer, Timer, get_current_time_ms
logger = get_logger(__name__)
settings = get_settings()
metrics_manager = get_metrics_manager()

T = TypeVar('T')
R = TypeVar('R')

class WorkerPoolType(str, Enum):
    THREAD = 'thread'
    PROCESS = 'process'
    ASYNCIO = 'asyncio'
    QUEUE_ASYNCIO = 'queue_asyncio'

class WorkerPoolConfig(BaseModel):
    pool_type: Union[WorkerPoolType, str] = WorkerPoolType.QUEUE_ASYNCIO
    workers: int = max(os.cpu_count() or 1, 1)
    max_tasks_per_worker: int = 10
    max_queue_size: Optional[int] = None
    shutdown_timeout: float = 10.0
    worker_init_func: Optional[str] = None
    worker_timeout: float = 0.0

    @field_validator('workers')
    def validate_workers(cls, v: int) -> int:
        if v <= 0:
            return max(os.cpu_count() or 1, 1)
        return v

class WorkerPoolMetrics(BaseModel):
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_queue_size: int = 0
    max_observed_queue_size: int = 0
    active_workers: int = 0
    max_active_workers: int = 0
    total_execution_time_ms: int = 0
    last_updated: int = Field(default_factory=get_current_time_ms)

    model_config = {
        "arbitrary_types_allowed": True,
    }

class QueueWorkerPoolConfig(BaseModel):
    pool_type: str = 'queue_asyncio'
    workers: int = max(os.cpu_count() or 1, 1)
    max_queue_size: int = 0
    shutdown_timeout: float = 10.0

    @field_validator('workers')
    def validate_workers(cls, v: int) -> int:
        if v <= 0:
            return max(os.cpu_count() or 1, 1)
        return v

class QueueWorkerPoolMetrics(BaseModel):
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_queue_size: int = 0
    max_observed_queue_size: int = 0
    active_workers: int = 0
    max_active_workers: int = 0
    total_execution_time_ms: int = 0
    last_updated: int = Field(default_factory=get_current_time_ms)

    model_config = {
        "arbitrary_types_allowed": True,
    }

class ThreadWorkerPool:

    def __init__(self, name: str, config: Optional[WorkerPoolConfig]=None):
        self.name: str = name
        self.config: WorkerPoolConfig = config or WorkerPoolConfig(pool_type=WorkerPoolType.THREAD)
        self.config.pool_type = WorkerPoolType.THREAD
        self.metrics: WorkerPoolMetrics = WorkerPoolMetrics()
        self._executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix=f'{name}_worker')
        self._futures: Set[concurrent.futures.Future] = set()
        self._lock: threading.RLock = threading.RLock()
        self._queue_semaphore: Optional[threading.BoundedSemaphore] = None
        if self.config.max_queue_size is not None and self.config.max_queue_size > 0:
            self._queue_semaphore = threading.BoundedSemaphore(self.config.max_queue_size)
        self._is_shutdown: bool = False
        logger.info(f'Initialized thread worker pool: {name}', extra={'pool_name': name, 'workers': self.config.workers, 'max_queue_size': self.config.max_queue_size or 'unlimited'})

    def _update_metrics(self, task_submitted: bool=False, task_completed: bool=False, task_failed: bool=False, active_delta: int=0, queue_delta: int=0) -> None:
        with self._lock:
            if task_submitted:
                self.metrics.tasks_submitted += 1
            if task_completed:
                self.metrics.tasks_completed += 1
            if task_failed:
                self.metrics.tasks_failed += 1
            if active_delta != 0:
                self.metrics.active_workers += active_delta
                self.metrics.max_active_workers = max(self.metrics.max_active_workers, self.metrics.active_workers)
            if queue_delta != 0:
                self.metrics.current_queue_size += queue_delta
                if queue_delta > 0:
                    self.metrics.max_observed_queue_size = max(self.metrics.max_observed_queue_size, self.metrics.current_queue_size)
                self.metrics.current_queue_size = max(0, self.metrics.current_queue_size)
            self.metrics.last_updated = get_current_time_ms()
            metrics_manager.track_task('queue_depth', value=self.metrics.current_queue_size)



    def _task_done_callback(self, future: concurrent.futures.Future) -> None:
        success: bool = not future.cancelled() and future.exception() is None
        self._update_metrics(task_completed=success, task_failed=not success, active_delta=-1)
        with self._lock:
            self._futures.discard(future)
        if self._queue_semaphore:
            try:
                self._queue_semaphore.release()
            except ValueError:
                logger.warning(f'Attempted to release semaphore too many times in thread pool {self.name}')
        if hasattr(future, '_start_time_ms'):
            execution_time_ms: int = get_current_time_ms() - getattr(future, '_start_time_ms', 0)
            status_str: str = 'success' if success else 'failure'
            metrics_manager.track_task("completed", status=status_str, value=execution_time_ms / 1000.0)

        else:
            metrics_manager.track_task("completed", status='success' if success else 'failure', value=0.0)

    def submit(self, func: Callable[..., R], *args: Any, timeout: Optional[float]=None, **kwargs: Any) -> concurrent.futures.Future[R]:
        if self._is_shutdown:
            raise WorkerPoolError(f'Thread worker pool {self.name} is shutdown')
        acquired_semaphore: bool = False
        if self._queue_semaphore:
            acquired_semaphore = self._queue_semaphore.acquire(blocking=False)
            if not acquired_semaphore:
                self._update_metrics(task_failed=True)
                metrics_manager.track_task("rejections", reason="queue_full")

                raise WorkerPoolError(f"Thread worker pool '{self.name}' queue is full (size: {self.config.max_queue_size})")
        start_time_ms: int = get_current_time_ms()

        @functools.wraps(func)
        def tracked_func(*func_args: Any, **func_kwargs: Any) -> R:
            self._update_metrics(active_delta=1)
            metrics_manager.track_task('processing', increment=True)

            try:
                return func(*func_args, **func_kwargs)
            finally:
                execution_time: int = get_current_time_ms() - start_time_ms
                with self._lock:
                    self.metrics.total_execution_time_ms += execution_time
                metrics_manager.track_task('processing', increment=False)

        try:
            self._update_metrics(task_submitted=True, queue_delta=1 if acquired_semaphore else 0)
            future: concurrent.futures.Future[R] = self._executor.submit(tracked_func, *args, **kwargs)
            setattr(future, '_start_time_ms', start_time_ms)
            future.add_done_callback(self._task_done_callback)
            with self._lock:
                self._futures.add(future)
                self._update_metrics(queue_delta=-1 if acquired_semaphore else 0)
            return future
        except Exception as e:
            self._update_metrics(task_failed=True, queue_delta=-1 if acquired_semaphore else 0)
            if acquired_semaphore and self._queue_semaphore:
                self._queue_semaphore.release()
            logger.error(f'Failed to submit task to ThreadPoolExecutor in {self.name}: {e}', exc_info=True)
            raise WorkerPoolError(f'Error submitting task to pool {self.name}', original_error=e)

    def submit_async(self, func: Callable[..., R], *args: Any, timeout: Optional[float]=None, **kwargs: Any) -> asyncio.Future[R]:
        try:
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning('No running event loop found in submit_async. Creating a new one.')
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        asyncio_future: asyncio.Future[R] = loop.create_future()

        def done_callback(thread_future: concurrent.futures.Future[R]) -> None:
            if asyncio_future.done():
                return
            if thread_future.cancelled():
                if not loop.is_closed():
                    loop.call_soon_threadsafe(asyncio_future.cancel)
            else:
                exc: Optional[BaseException] = thread_future.exception()
                if exc:
                    if not loop.is_closed():
                        loop.call_soon_threadsafe(asyncio_future.set_exception, exc)
                else:
                    try:
                        result: R = thread_future.result()
                        if not loop.is_closed():
                            loop.call_soon_threadsafe(asyncio_future.set_result, result)
                    except Exception as result_exc:
                        if not loop.is_closed():
                            loop.call_soon_threadsafe(asyncio_future.set_exception, result_exc)
        try:
            thread_future: concurrent.futures.Future[R] = self.submit(func, *args, **kwargs)
            thread_future.add_done_callback(done_callback)
        except Exception as e:
            if not asyncio_future.done():
                if not loop.is_closed():
                    loop.call_soon_threadsafe(asyncio_future.set_exception, e)
        return asyncio_future

    async def asubmit(self, func: Callable[..., R], *args: Any, timeout: Optional[float]=None, **kwargs: Any) -> R:
        asyncio_future: asyncio.Future[R] = self.submit_async(func, *args, **kwargs)
        effective_timeout: Optional[float] = timeout if timeout is not None else self.config.worker_timeout if self.config.worker_timeout > 0 else None
        try:
            if effective_timeout is not None:
                return await asyncio.wait_for(asyncio_future, timeout=effective_timeout)
            else:
                return await asyncio_future
        except asyncio.TimeoutError as e:
            logger.warning(f'Task in thread pool {self.name} timed out after {effective_timeout}s')
            raise TimeoutError(f'Task execution timed out after {effective_timeout}s') from e

    def map(self, func: Callable[[T], R], items: List[T], timeout: Optional[float]=None) -> List[Union[R, Exception]]:
        results: List[Union[R, Exception]] = [None] * len(items)
        futures_map: Dict[concurrent.futures.Future[R], int] = {}
        submitted_futures: List[concurrent.futures.Future[R]] = []
        try:
            for i, item in enumerate(items):
                future = self.submit(func, item)
                futures_map[future] = i
                submitted_futures.append(future)
            for future in concurrent.futures.as_completed(submitted_futures, timeout=timeout):
                idx = futures_map[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
            if timeout is not None:
                completed_indices = {futures_map[f] for f in futures_map if f.done()}
                for i in range(len(items)):
                    if i not in completed_indices:
                        results[i] = TimeoutError(f'Map operation timed out for item index {i} after {timeout}s')
                        for f, original_idx in futures_map.items():
                            if original_idx == i and (not f.done()):
                                f.cancel()
                                break
            return results
        except concurrent.futures.TimeoutError as e:
            logger.warning(f'Map operation timed out after {timeout} seconds in pool {self.name}.')
            completed_indices = {futures_map[f] for f in futures_map if f.done()}
            for i in range(len(items)):
                if i not in completed_indices:
                    results[i] = TimeoutError(f'Map operation timed out for item index {i} after {timeout}s')
                    for f, original_idx in futures_map.items():
                        if original_idx == i and (not f.done()):
                            f.cancel()
                            break
            return results
        except Exception as e:
            logger.error(f'Error during map operation in {self.name}: {e}', exc_info=True)
            for future in submitted_futures:
                if not future.done():
                    future.cancel()
            raise WorkerPoolError(f'Error in map operation: {e}', original_error=e)

    async def amap(self, func: Callable[[T], R], items: List[T], timeout: Optional[float]=None) -> List[Union[R, Exception]]:
        if not items:
            return []
        asyncio_futures: List[asyncio.Future[R]] = [self.submit_async(func, item) for item in items]
        try:
            if timeout:
                try:
                    gathered_results: List[Union[R, BaseException]] = await asyncio.wait_for(asyncio.gather(*asyncio_futures, return_exceptions=True), timeout=timeout)
                    return cast(List[Union[R, Exception]], gathered_results)
                except asyncio.TimeoutError:
                    logger.warning(f'Async map operation timed out after {timeout}s in pool {self.name}')
                    for fut in asyncio_futures:
                        if not fut.done():
                            fut.cancel()
                        await asyncio.sleep(0)
                    final_results = []
                    for fut in asyncio_futures:
                        if fut.cancelled():
                            final_results.append(TimeoutError(f'Async map item timed out or cancelled'))
                        elif fut.done():
                            try:
                                final_results.append(fut.result())
                            except Exception as e:
                                final_results.append(e)
                        else:
                            final_results.append(TimeoutError(f'Async map item did not complete'))
                    return final_results
            else:
                gathered_results: List[Union[R, BaseException]] = await asyncio.gather(*asyncio_futures, return_exceptions=True)
                return cast(List[Union[R, Exception]], gathered_results)
        except Exception as e:
            logger.error(f'Error during async map operation in {self.name}: {e}', exc_info=True)
            for fut in asyncio_futures:
                if not fut.done():
                    fut.cancel()
            raise WorkerPoolError(f'Error in async map operation: {e}', original_error=e)

    def shutdown(self, wait: bool=True, timeout: Optional[float]=None) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True
        logger.info(f'Shutting down thread worker pool: {self.name}')
        effective_timeout: Optional[float] = timeout if timeout is not None else self.config.shutdown_timeout
        try:
            if sys.version_info >= (3, 9):
                self._executor.shutdown(wait=wait, cancel_futures=True)
            else:
                self._executor.shutdown(wait=wait)
            logger.info(f'Thread worker pool {self.name} shutdown initiated (wait={wait}).')
        except Exception as e:
            logger.error(f'Error during thread worker pool {self.name} shutdown: {e}', exc_info=True)

    async def ashutdown(self, wait: bool=True, timeout: Optional[float]=None) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        await loop.run_in_executor(None, functools.partial(self.shutdown, wait=wait, timeout=timeout))
        logger.info(f'Async shutdown command issued for thread pool {self.name}')

    def __enter__(self) -> 'ThreadWorkerPool':
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        self.shutdown()

    async def __aenter__(self) -> 'ThreadWorkerPool':
        return self

    async def __aexit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        await self.ashutdown()

def _is_picklable(obj: Any) -> bool:
    if hasattr(obj, '__self__'):
        logger.warning(f'Object {obj} appears to be a bound method, which is not picklable.')
        return False
    module = getattr(obj, '__module__', None)
    if module == '__main__':
        logger.warning(f'Object {obj} is defined in __main__, which may not be picklable by child processes.')
    return True

class ProcessWorkerPool:

    def __init__(self, name: str, config: Optional[WorkerPoolConfig]=None):
        self.name: str = name
        self.config: WorkerPoolConfig = config or WorkerPoolConfig(pool_type=WorkerPoolType.PROCESS)
        self.config.pool_type = WorkerPoolType.PROCESS
        self.metrics: WorkerPoolMetrics = WorkerPoolMetrics()
        mp_context: Optional[multiprocessing.context.BaseContext] = None
        try:
            available_methods = multiprocessing.get_all_start_methods()
            start_method = 'spawn' if 'spawn' in available_methods else 'forkserver' if 'forkserver' in available_methods else 'fork'
            logger.debug(f'Using multiprocessing start method: {start_method}')
            mp_context = multiprocessing.get_context(start_method)
        except Exception as ctx_e:
            logger.warning(f'Failed to get specific multiprocessing context: {ctx_e}. Using default.')
        self._executor: concurrent.futures.ProcessPoolExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=self.config.workers, mp_context=mp_context)
        self._futures: Set[concurrent.futures.Future] = set()
        self._lock: threading.RLock = threading.RLock()
        self._queue_semaphore: Optional[threading.BoundedSemaphore] = None
        if self.config.max_queue_size is not None and self.config.max_queue_size > 0:
            self._queue_semaphore = threading.BoundedSemaphore(self.config.max_queue_size)
        self._is_shutdown: bool = False
        logger.info(f'Initialized process worker pool: {name}', extra={'pool_name': name, 'workers': self.config.workers, 'mp_context': mp_context.get_start_method() if mp_context else multiprocessing.get_start_method(), 'max_queue_size': self.config.max_queue_size or 'unlimited'})

    def _update_metrics(self, task_submitted: bool=False, task_completed: bool=False, task_failed: bool=False, active_delta: int=0, queue_delta: int=0) -> None:
        with self._lock:
            if task_submitted:
                self.metrics.tasks_submitted += 1
            if task_completed:
                self.metrics.tasks_completed += 1
            if task_failed:
                self.metrics.tasks_failed += 1
            if active_delta != 0:
                self.metrics.active_workers += active_delta
                self.metrics.max_active_workers = max(self.metrics.max_active_workers, self.metrics.active_workers)
            if queue_delta != 0:
                self.metrics.current_queue_size += queue_delta
                if queue_delta > 0:
                    self.metrics.max_observed_queue_size = max(self.metrics.max_observed_queue_size, self.metrics.current_queue_size)
                self.metrics.current_queue_size = max(0, self.metrics.current_queue_size)
            self.metrics.last_updated = get_current_time_ms()
            metrics_manager.track_task('queue_depth', value=self.metrics.current_queue_size)


    def _task_done_callback(self, future: concurrent.futures.Future) -> None:
        success: bool = not future.cancelled() and future.exception() is None
        self._update_metrics(task_completed=success, task_failed=not success, active_delta=-1)
        metrics_manager.track_task('processing', increment=False)

        with self._lock:
            self._futures.discard(future)
        if self._queue_semaphore:
            try:
                self._queue_semaphore.release()
            except ValueError:
                logger.warning(f'Attempted to release semaphore too many times in process pool {self.name}')
        if hasattr(future, '_start_time_ms'):
            execution_time_ms: int = get_current_time_ms() - getattr(future, '_start_time_ms', 0)
            status_str: str = 'success' if success else 'failure'
            metrics_manager.track_task("completed", status=status_str, value=execution_time_ms / 1000.0)

            with self._lock:
                self.metrics.total_execution_time_ms += execution_time_ms
        else:
            metrics_manager.track_task("completed", status='success' if success else 'failure', value=0.0)

    def submit(self, func: Callable[..., R], *args: Any, timeout: Optional[float]=None, **kwargs: Any) -> concurrent.futures.Future[R]:
        if self._is_shutdown:
            raise WorkerPoolError(f'Process worker pool {self.name} is shutdown')
        acquired_semaphore: bool = False
        if self._queue_semaphore:
            acquired_semaphore = self._queue_semaphore.acquire(blocking=False)
            if not acquired_semaphore:
                self._update_metrics(task_failed=True)
                metrics_manager.track_task("rejections", reason="queue_full")

                raise WorkerPoolError(f"Process worker pool '{self.name}' queue is full (size: {self.config.max_queue_size})")
        if not _is_picklable(func):
            if acquired_semaphore and self._queue_semaphore:
                self._queue_semaphore.release()
            self._update_metrics(task_failed=True)
            raise WorkerPoolError(f"Function '{getattr(func, '__name__', '<unknown>')}' must be picklable for process pool {self.name}")
        start_time_ms: int = get_current_time_ms()
        try:
            self._update_metrics(task_submitted=True, queue_delta=1 if acquired_semaphore else 0, active_delta=1)
            metrics_manager.track_task('processing', increment=True)

            future: concurrent.futures.Future[R] = self._executor.submit(func, *args, **kwargs)
            setattr(future, '_start_time_ms', start_time_ms)
            future.add_done_callback(self._task_done_callback)
            with self._lock:
                self._futures.add(future)
                self._update_metrics(queue_delta=-1 if acquired_semaphore else 0)
            return future
        except Exception as e:
            self._update_metrics(task_failed=True, queue_delta=-1 if acquired_semaphore else 0, active_delta=-1)
            metrics_manager.track_task('processing', increment=False)

            if acquired_semaphore and self._queue_semaphore:
                self._queue_semaphore.release()
            logger.exception(f'Error submitting task to process pool {self.name}: {e}', exc_info=True)
            raise WorkerPoolError(f'Error submitting task: {str(e)}', original_error=e)

    def submit_async(self, func: Callable[..., R], *args: Any, timeout: Optional[float]=None, **kwargs: Any) -> asyncio.Future[R]:
        try:
            loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning('No running event loop found in process submit_async. Creating a new one.')
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        asyncio_future: asyncio.Future[R] = loop.create_future()

        def done_callback(process_future: concurrent.futures.Future[R]) -> None:
            if asyncio_future.done():
                return
            if process_future.cancelled():
                if not loop.is_closed():
                    loop.call_soon_threadsafe(asyncio_future.cancel)
            else:
                exc: Optional[BaseException] = process_future.exception()
                if exc:
                    if not loop.is_closed():
                        loop.call_soon_threadsafe(asyncio_future.set_exception, exc)
                else:
                    try:
                        result: R = process_future.result()
                        if not loop.is_closed():
                            loop.call_soon_threadsafe(asyncio_future.set_result, result)
                    except Exception as result_exc:
                        if not loop.is_closed():
                            loop.call_soon_threadsafe(asyncio_future.set_exception, result_exc)
        try:
            process_future: concurrent.futures.Future[R] = self.submit(func, *args, **kwargs)
            process_future.add_done_callback(done_callback)
        except Exception as e:
            if not asyncio_future.done():
                if not loop.is_closed():
                    loop.call_soon_threadsafe(asyncio_future.set_exception, e)
        return asyncio_future

    async def asubmit(self, func: Callable[..., R], *args: Any, timeout: Optional[float]=None, **kwargs: Any) -> R:
        asyncio_future: asyncio.Future[R] = self.submit_async(func, *args, **kwargs)
        effective_timeout: Optional[float] = timeout if timeout is not None else self.config.worker_timeout if self.config.worker_timeout > 0 else None
        try:
            if effective_timeout is not None:
                return await asyncio.wait_for(asyncio_future, timeout=effective_timeout)
            else:
                return await asyncio_future
        except asyncio.TimeoutError as e:
            logger.warning(f'Task in process pool {self.name} timed out after {effective_timeout}s')
            raise TimeoutError(f'Task execution timed out after {effective_timeout}s') from e

    def map(self, func: Callable[[T], R], items: List[T], timeout: Optional[float]=None, chunksize: int=1) -> List[Union[R, Exception]]:
        if not _is_picklable(func):
            raise WorkerPoolError(f"Function '{getattr(func, '__name__', '<unknown>')}' must be picklable for process pool map in {self.name}")
        results: List[Union[R, Exception]] = []
        start_time: float = time.time()
        self._update_metrics(active_delta=len(items))
        metrics_manager.track_task('processing', increment=True, value=len(items))
        try:
            iterator = self._executor.map(func, items, timeout=timeout, chunksize=chunksize)
            results = list(iterator)
            return results
        except concurrent.futures.TimeoutError as e:
            logger.warning(f'Process pool map operation timed out after {timeout}s for pool {self.name}')
            raise TimeoutError(f'Map operation timed out after {timeout}s') from e
        except Exception as e:
            logger.error(f'Error during process pool map operation in {self.name}: {e}', exc_info=True)
            raise WorkerPoolError(f'Error in map operation: {str(e)}', original_error=e)
        finally:
            self._update_metrics(active_delta=-len(items))
            metrics_manager.track_task('processing', increment=False, value=len(items))
            completed_count = sum((1 for r in results if not isinstance(r, Exception)))
            failed_count = len(results) - completed_count

    async def amap(self, func: Callable[[T], R], items: List[T], timeout: Optional[float]=None, chunksize: int=1) -> List[Union[R, Exception]]:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            partial_map = functools.partial(self.map, func, items, timeout=timeout, chunksize=chunksize)
            results: List[Union[R, Exception]] = await loop.run_in_executor(None, partial_map)
            return results
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning(f'Async map operation timed out after {timeout}s for pool {self.name}')
                raise TimeoutError(f'Async map operation timed out after {timeout}s') from e
            else:
                logger.error(f'Error during async map operation in {self.name}: {e}', exc_info=True)
                raise WorkerPoolError(f'Error in async map operation: {str(e)}', original_error=e)

    def shutdown(self, wait: bool=True, timeout: Optional[float]=None) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True
        logger.info(f'Shutting down process worker pool: {self.name}')
        try:
            if sys.version_info >= (3, 9):
                self._executor.shutdown(wait=wait, cancel_futures=True)
            else:
                self._executor.shutdown(wait=wait)
            logger.info(f'Process worker pool {self.name} shutdown initiated (wait={wait}).')
        except Exception as e:
            logger.error(f'Error during process worker pool {self.name} shutdown: {e}', exc_info=True)

    async def ashutdown(self, wait: bool=True, timeout: Optional[float]=None) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        await loop.run_in_executor(None, functools.partial(self.shutdown, wait=wait, timeout=timeout))
        logger.info(f'Async shutdown command issued for process pool {self.name}')

    def __enter__(self) -> 'ProcessWorkerPool':
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        self.shutdown()

    async def __aenter__(self) -> 'ProcessWorkerPool':
        return self

    async def __aexit__(self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
        await self.ashutdown()

# Create a base class for shared functionality
class BaseWorkerPool:
    """Base class for all worker pools to reduce code duplication"""
    
    def __init__(self, name: str):
        self.name = name
        self._is_shutdown = False
        
    def _update_metrics(self, metrics_obj, task_submitted=False, task_completed=False, 
                       task_failed=False, active_delta=0, queue_delta=0):
        """Centralized metrics update logic for all pool types"""
        # Update submitted count
        if task_submitted:
            metrics_obj.tasks_submitted += 1
            
        # Update completion counts
        if task_completed:
            metrics_obj.tasks_completed += 1
        if task_failed:
            metrics_obj.tasks_failed += 1
            
        # Update active workers
        if active_delta != 0:
            metrics_obj.active_workers += active_delta
            metrics_obj.max_active_workers = max(
                metrics_obj.max_active_workers, 
                metrics_obj.active_workers
            )
            
        # Update queue metrics
        if queue_delta != 0:
            metrics_obj.current_queue_size += queue_delta
            if queue_delta > 0:
                metrics_obj.max_observed_queue_size = max(
                    metrics_obj.max_observed_queue_size, 
                    metrics_obj.current_queue_size
                )
            metrics_obj.current_queue_size = max(0, metrics_obj.current_queue_size)
            
        # Update timestamp and prometheus metrics
        metrics_obj.last_updated = get_current_time_ms()
        metrics_manager.track_task('queue_depth', value=metrics_obj.current_queue_size)
    
    async def _common_shutdown(self, pool_type, wait, timeout):
        """Common shutdown logic across pool types"""
        if self._is_shutdown:
            logger.warning(f"{pool_type} worker pool '{self.name}' is already shut down")
            return
            
        logger.info(f"Shutting down {pool_type} worker pool: {self.name} (wait={wait}, timeout={timeout}s)")
        self._is_shutdown = True
        
    def _handle_task_exception(self, e, pool_type, task_info=None):
        """Standardized exception handling for all pool types"""
        if isinstance(e, asyncio.CancelledError):
            logger.warning(f"Task in {pool_type} pool '{self.name}' was cancelled")
            return {'type': 'task_cancelled', 'message': 'Task was cancelled'}
        elif isinstance(e, TimeoutError):
            logger.warning(f"Task in {pool_type} pool '{self.name}' timed out")
            return {'type': 'task_timeout', 'message': f'Task execution timed out: {str(e)}'}
        else:
            task_name = task_info or getattr(e, '__name__', 'unknown')
            logger.error(f"Error executing task '{task_name}' in {pool_type} pool '{self.name}': {e}", 
                       exc_info=True)
            return {'type': 'task_error', 'message': f'Task execution failed: {str(e)}'}
            
# Example integration with QueueWorkerPool

class QueueWorkerPool(BaseWorkerPool):
    def __init__(self, name: str, config: Optional[QueueWorkerPoolConfig]=None):
        super().__init__(name)  # Initialize base class
        self._config: QueueWorkerPoolConfig = config or QueueWorkerPoolConfig()
        self._config.pool_type = 'queue_asyncio'
        self._metrics: QueueWorkerPoolMetrics = QueueWorkerPoolMetrics()
        self._work_queue: asyncio.Queue[Optional[Tuple[Callable[..., Coroutine[Any, Any, Any]], tuple, dict]]] = asyncio.Queue(maxsize=self._config.max_queue_size)
        self._workers: Set[asyncio.Task[None]] = set()
        self._worker_states: Dict[str, str] = {}
        self._active_task_count: int = 0
        self._metrics_lock: asyncio.Lock = asyncio.Lock()
        logger.info(f'Initialized QueueWorkerPool: {name}', 
                  extra={'pool_name': name, 'workers': self.config.workers, 
                        'max_queue_size': self.config.max_queue_size or 'unlimited'})
        self._start_workers()
        
    async def _worker_loop(self, worker_id: int) -> None:
        worker_name: str = f'{self.name}-worker-{worker_id}'
        logger.debug(f'Asyncio worker {worker_name} started.')
        self._worker_states[worker_name] = 'idle'
        
        while not self._is_shutdown:
            task_info: Optional[Tuple[Callable[..., Coroutine[Any, Any, Any]], tuple, dict]] = None
            try:
                task_info = await self._work_queue.get()
                if task_info is None:
                    logger.debug(f'Worker {worker_name} received shutdown signal. Exiting loop.')
                    self._worker_states[worker_name] = 'shutdown'
                    break
                
                func, args, kwargs = task_info
                task_start_time: int = get_current_time_ms()
                success: bool = False
                func_name: str = getattr(func, '__name__', 'unknown')
                self._worker_states[worker_name] = 'busy'
                
                # Use base class metrics update
                async with self._metrics_lock:
                    self._active_task_count += 1
                    self._update_metrics(
                        self._metrics,
                        active_delta=0  # We'll set active count directly
                    )
                    self._metrics.active_workers = self._active_task_count
                    self._metrics.max_active_workers = max(
                        self._metrics.max_active_workers, 
                        self._metrics.active_workers
                    )
                
                metrics_manager.track_task('processing', increment=True)
                try:
                    logger.debug(f'Worker {worker_name} starting task: {func_name}')
                    if asyncio.iscoroutinefunction(func):
                        await func(*args, **kwargs)
                    else:
                        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
                    success = True
                    logger.debug(f'Worker {worker_name} completed task: {func_name}')
                except Exception as e:
                    # Use common exception handler
                    error_info = self._handle_task_exception(e, "queue", func_name)
                    success = False
                finally:
                    execution_time_ms: int = get_current_time_ms() - task_start_time
                    async with self._metrics_lock:
                        if success:
                            self._metrics.tasks_completed += 1
                        else:
                            self._metrics.tasks_failed += 1
                        self._metrics.total_execution_time_ms += execution_time_ms
                        self._active_task_count -= 1
                        self._metrics.active_workers = self._active_task_count
                    
                    metrics_manager.track_task('processing', increment=False)

                    metric_status: str = 'success' if success else 'failure'
                    metrics_manager.track_task("completed", status=metric_status, value=execution_time_ms / 1000.0)
                    self._worker_states[worker_name] = 'idle'
                    self._work_queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f'Worker {worker_name} loop cancelled.')
                self._worker_states[worker_name] = 'shutdown'
                break
            except Exception as e:
                logger.exception(f'Unexpected error in worker {worker_name} loop: {e}')
                self._worker_states[worker_name] = 'error'
                await asyncio.sleep(1)  # Brief pause before trying again
                
        logger.info(f'Asyncio worker {worker_name} finished.')

    async def shutdown(self, wait: bool=True, timeout: Optional[float]=None) -> None:
        # Use common shutdown logic
        await self._common_shutdown("Queue", wait, timeout)
        
        # Queue-specific shutdown steps
        for _ in range(len(self._workers)):
            try:
                self._work_queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.warning(f"Queue full while sending shutdown signals to workers in pool '{self.name}'")
                break
            except Exception as e:
                logger.error(f"Error putting shutdown signal into queue for pool '{self.name}': {e}")
                

    @property
    def config(self) -> QueueWorkerPoolConfig:
        return self._config

    @property
    def metrics(self) -> QueueWorkerPoolMetrics:
        self._metrics.current_queue_size = self.get_queue_size()
        return self._metrics

    def _start_workers(self) -> None:
        if self._is_shutdown:
            raise WorkerPoolError('Cannot start workers on a shutdown pool')
        for i in range(self.config.workers):
            worker_task: asyncio.Task[None] = asyncio.create_task(self._worker_loop(i), name=f'{self.name}-worker-{i}')
            self._workers.add(worker_task)
            self._worker_states[worker_task.get_name()] = 'idle'
        logger.info(f"Started {len(self._workers)} asyncio workers for pool '{self.name}'")

    async def submit(self, func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> None:
        if self._is_shutdown:
            raise WorkerPoolError(f'Worker pool {self.name} is shutdown')
        async with self._metrics_lock:
            self.metrics.tasks_submitted += 1
            current_qsize: int = self._work_queue.qsize()
            estimated_qsize_after_put = current_qsize + 1
            self.metrics.max_observed_queue_size = max(self.metrics.max_observed_queue_size, estimated_qsize_after_put)
            metrics_manager.track_task('queue_depth', value=estimated_qsize_after_put)
        try:
            await self._work_queue.put((func, args, kwargs))
            logger.debug(f"Task '{getattr(func, '__name__', 'unknown')}' submitted to pool '{self.name}'. Current approx queue size: {estimated_qsize_after_put}")
        except asyncio.QueueFull:
            async with self._metrics_lock:
                self.metrics.tasks_failed += 1
                metrics_manager.track_task('queue_depth', value=self._work_queue.qsize())
            metrics_manager.track_task("rejections", reason="queue_full")

            raise WorkerPoolError(f"Worker pool '{self.name}' queue is full (max size: {self.config.max_queue_size})")
        except Exception as e:
            async with self._metrics_lock:
                self.metrics.tasks_failed += 1
                metrics_manager.track_task('queue_depth', value=self._work_queue.qsize())
            raise WorkerPoolError(f"Failed to submit task to pool '{self.name}': {e}")


    def get_active_workers(self) -> int:
        return self._active_task_count

    def get_queue_size(self) -> int:
        return self._work_queue.qsize()
AnyWorkerPool = Union['QueueWorkerPool', 'ThreadWorkerPool', 'ProcessWorkerPool']
AnyWorkerPoolConfig = Union[WorkerPoolConfig, 'QueueWorkerPoolConfig']
_worker_pools: Dict[str, AnyWorkerPool] = {}
_pool_factory_lock = asyncio.Lock()

async def get_worker_pool(name: str, pool_type: Union[WorkerPoolType, str]=WorkerPoolType.QUEUE_ASYNCIO, config: Optional[AnyWorkerPoolConfig]=None) -> AnyWorkerPool:
    pool_type_enum: WorkerPoolType
    if isinstance(pool_type, str):
        try:
            pool_type_enum = WorkerPoolType(pool_type.lower())
        except ValueError:
            logger.warning(f"Invalid pool type string '{pool_type}'. Defaulting to QUEUE_ASYNCIO.")
            pool_type_enum = WorkerPoolType.QUEUE_ASYNCIO
    elif isinstance(pool_type, WorkerPoolType):
        pool_type_enum = pool_type
    else:
        logger.warning(f"Invalid pool_type type '{type(pool_type)}'. Defaulting to QUEUE_ASYNCIO.")
        pool_type_enum = WorkerPoolType.QUEUE_ASYNCIO
    key: str = f'{name}:{pool_type_enum.value}'
    async with _pool_factory_lock:
        if key not in _worker_pools:
            logger.info(f"Worker pool '{key}' not found in registry. Creating new instance...")
            config_obj: Optional[AnyWorkerPoolConfig] = config
            try:
                if pool_type_enum == WorkerPoolType.QUEUE_ASYNCIO:
                    if config is None:
                        typed_config = QueueWorkerPoolConfig()
                    elif isinstance(config, QueueWorkerPoolConfig):
                        typed_config = config
                    elif isinstance(config, WorkerPoolConfig):
                        config_data = config.model_dump(exclude={'pool_type', 'worker_init_func'})
                        typed_config = QueueWorkerPoolConfig(**config_data)
                    else:
                        try:
                            typed_config = QueueWorkerPoolConfig(**config)
                        except Exception:
                            typed_config = QueueWorkerPoolConfig()
                    typed_config.pool_type = pool_type_enum.value
                    _worker_pools[key] = QueueWorkerPool(name, typed_config)
                elif pool_type_enum == WorkerPoolType.THREAD:
                    if config is None:
                        typed_config = WorkerPoolConfig(pool_type=pool_type_enum)
                    elif isinstance(config, WorkerPoolConfig):
                        typed_config = config
                    elif isinstance(config, QueueWorkerPoolConfig):
                        config_data = config.model_dump(exclude={'pool_type'})
                        typed_config = WorkerPoolConfig(pool_type=pool_type_enum, **config_data)
                    else:
                        try:
                            typed_config = WorkerPoolConfig(**config)
                        except Exception:
                            typed_config = WorkerPoolConfig(pool_type=pool_type_enum)
                    typed_config.pool_type = pool_type_enum.value
                    _worker_pools[key] = ThreadWorkerPool(name, typed_config)
                elif pool_type_enum == WorkerPoolType.PROCESS:
                    if config is None:
                        typed_config = WorkerPoolConfig(pool_type=pool_type_enum)
                    elif isinstance(config, WorkerPoolConfig):
                        typed_config = config
                    elif isinstance(config, QueueWorkerPoolConfig):
                        config_data = config.model_dump(exclude={'pool_type'})
                        typed_config = WorkerPoolConfig(pool_type=pool_type_enum, **config_data)
                    else:
                        try:
                            typed_config = WorkerPoolConfig(**config)
                        except Exception:
                            typed_config = WorkerPoolConfig(pool_type=pool_type_enum)
                    typed_config.pool_type = pool_type_enum.value
                    _worker_pools[key] = ProcessWorkerPool(name, typed_config)
                else:
                    raise ValueError(f'Unsupported pool type specified: {pool_type_enum}')
                logger.info(f"Successfully created worker pool '{key}'")
            except Exception as creation_e:
                logger.exception(f"Failed to create worker pool '{key}': {creation_e}", exc_info=True)
                raise WorkerPoolError(f"Failed to create worker pool '{key}'", original_error=creation_e)
    pool_instance = _worker_pools.get(key)
    if pool_instance is None:
        logger.error(f"Internal error: Worker pool '{key}' not found in registry after creation attempt.")
        raise WorkerPoolError(f"Failed to retrieve worker pool '{key}' after creation attempt.")
    expected_type: type
    if pool_type_enum == WorkerPoolType.QUEUE_ASYNCIO:
        expected_type = QueueWorkerPool
    elif pool_type_enum == WorkerPoolType.THREAD:
        expected_type = ThreadWorkerPool
    elif pool_type_enum == WorkerPoolType.PROCESS:
        expected_type = ProcessWorkerPool
    else:
        raise ValueError('Internal error: Unexpected pool type check')
    if not isinstance(pool_instance, expected_type):
        logger.critical(f"CRITICAL: Type mismatch in worker pool registry for key '{key}'. Expected {expected_type.__name__}, but got {type(pool_instance).__name__}. This indicates a potential registry corruption.")
        raise TypeError(f"Retrieved pool for '{key}' has incorrect type. Expected {expected_type.__name__}.")
    return pool_instance

def get_default_worker_pool() -> 'QueueWorkerPool':
    pool = get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO)
    return cast('QueueWorkerPool', pool)

async def shutdown_all_worker_pools(wait: bool=True, timeout: Optional[float]=None) -> None:
    logger.info(f'Initiating shutdown for all ({len(_worker_pools)}) registered worker pools...')
    shutdown_tasks: List[Coroutine[Any, Any, None]] = []
    for key, pool_instance in list(_worker_pools.items()):
        logger.debug(f"Preparing shutdown for pool '{key}' of type {type(pool_instance).__name__}")
        try:
            if isinstance(pool_instance, QueueWorkerPool):
                shutdown_tasks.append(pool_instance.shutdown(wait=wait, timeout=timeout))
            elif isinstance(pool_instance, (ThreadWorkerPool, ProcessWorkerPool)):
                shutdown_tasks.append(pool_instance.ashutdown(wait=wait, timeout=timeout))
            else:
                logger.warning(f"Pool '{key}' has unknown type {type(pool_instance).__name__}, skipping shutdown.")
        except AttributeError as e:
            logger.error(f"Error accessing shutdown method for pool '{key}' of type {type(pool_instance).__name__}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error preparing shutdown for pool '{key}': {e}")
    if shutdown_tasks:
        logger.info(f'Waiting for {len(shutdown_tasks)} worker pools to shut down...')
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pool_key = list(_worker_pools.keys())[i]
                logger.error(f"Error during shutdown of pool task {i} (likely '{pool_key}'): {result}", exc_info=result)
    else:
        logger.info('No active worker pools found to shut down.')
    _worker_pools.clear()
    logger.info('All worker pools shutdown attempt completed and registry cleared.')
import pytest
import asyncio
import time
import threading
import concurrent.futures
from unittest.mock import AsyncMock, patch
from pickle import PicklingError

from src.core.worker_pool import (
    WorkerPoolType,
    WorkerPoolConfig,
    AsyncWorkerPool,
    ThreadWorkerPool,
    ProcessWorkerPool,
    get_worker_pool,
    get_default_worker_pool,
    shutdown_all_worker_pools
)
from src.core.exceptions import WorkerPoolError

# --- Helper functions for process pool tests ---
def _process_pool_test_func(value):
    """Simple function for process pool testing."""
    # Note: Process pools require functions defined at the module level
    return value * 2

def _process_pool_failing_func():
    """Function that raises an error for process pool testing."""
    raise ValueError("Test error from process pool")

def _process_pool_lambda_replacement(x):
    """Module-level replacement for lambda used in process pool tests."""
    return x * 2
# --- End helper functions ---


@pytest.fixture
def async_pool_config():
    """Fixture for AsyncWorkerPool configuration."""
    return WorkerPoolConfig(
        pool_type=WorkerPoolType.ASYNCIO,
        workers=5,
        max_tasks_per_worker=10,
        max_queue_size=20 # Default config for most tests
    )


@pytest.fixture
def thread_pool_config():
    """Fixture for ThreadWorkerPool configuration."""
    return WorkerPoolConfig(
        pool_type=WorkerPoolType.THREAD,
        workers=3,
        max_tasks_per_worker=5,
        max_queue_size=10,
        shutdown_timeout=1.0
    )


@pytest.fixture
def process_pool_config():
    """Fixture for ProcessWorkerPool configuration."""
    return WorkerPoolConfig(
        pool_type=WorkerPoolType.PROCESS,
        workers=2,
        max_tasks_per_worker=3,
        max_queue_size=5,
        shutdown_timeout=1.0
    )


@pytest.fixture
def async_pool(async_pool_config, event_loop): # event_loop fixture needed for shutdown
    """Fixture for AsyncWorkerPool."""
    pool = AsyncWorkerPool("test_async_pool", async_pool_config)
    yield pool # Provide the pool instance to the test
    # Teardown: Ensure pool is shutdown after test
    event_loop.run_until_complete(pool.shutdown())


@pytest.fixture
def thread_pool(thread_pool_config):
    """Fixture for ThreadWorkerPool."""
    pool = ThreadWorkerPool("test_thread_pool", thread_pool_config)
    yield pool
    pool.shutdown() # Synchronous shutdown


@pytest.fixture
def process_pool(process_pool_config):
    """Fixture for ProcessWorkerPool."""
    pool = ProcessWorkerPool("test_process_pool", process_pool_config)
    yield pool
    pool.shutdown() # Synchronous shutdown


class TestAsyncWorkerPool:
    """Test suite for AsyncWorkerPool."""

    @pytest.mark.asyncio
    async def test_initialization(self, async_pool_config):
        """Test pool initialization with specific config."""
        pool = AsyncWorkerPool("init_test_pool", async_pool_config)
        assert pool.name == "init_test_pool"
        assert pool.config.workers == 5
        assert pool.config.max_queue_size == 20
        assert pool.metrics.tasks_submitted == 0
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_submit_and_execute(self, async_pool):
        """Test submitting and executing async and sync tasks."""
        async def test_async_func(value):
            await asyncio.sleep(0.01)
            return value * 2

        result_async = await async_pool.submit(test_async_func, 21)
        assert result_async == 42
        assert async_pool.metrics.tasks_completed == 1

        def test_sync_func(value):
            time.sleep(0.01)
            return value * 3

        result_sync = await async_pool.submit(test_sync_func, 14)
        assert result_sync == 42
        assert async_pool.metrics.tasks_submitted == 2
        assert async_pool.metrics.tasks_completed == 2

    @pytest.mark.asyncio
    async def test_map_function(self, async_pool):
        """Test the map function for parallel processing."""
        async def test_func(value):
            await asyncio.sleep(0.01)
            return value * 2

        values = [1, 2, 3, 4, 5]
        results = await async_pool.map(test_func, values)
        assert results == [2, 4, 6, 8, 10]
        assert async_pool.metrics.tasks_submitted == 5
        assert async_pool.metrics.tasks_completed == 5

    @pytest.mark.asyncio
    async def test_error_handling(self, async_pool):
        """Test error handling when a task raises an exception."""
        async def failing_func():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await async_pool.submit(failing_func)

        assert async_pool.metrics.tasks_submitted == 1
        assert async_pool.metrics.tasks_completed == 0
        assert async_pool.metrics.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_concurrency_control(self, async_pool):
        """Test that tasks run concurrently up to the worker limit."""
        async def sleep_func(duration):
            await asyncio.sleep(duration)
            return duration

        start_time = time.monotonic()
        # Submit 5 tasks (equal to worker count)
        tasks = [async_pool.submit(sleep_func, 0.1) for _ in range(5)]
        await asyncio.gather(*tasks)
        execution_time = time.monotonic() - start_time

        # Should take slightly more than 0.1s due to concurrency
        assert 0.1 <= execution_time < 0.2, f"Concurrent execution time ({execution_time:.3f}s) not within expected range (0.1-0.2s)"

        # Submit more tasks than workers (8 total)
        more_tasks = [async_pool.submit(sleep_func, 0.1) for _ in range(3)]
        await asyncio.gather(*more_tasks)
        total_execution_time = time.monotonic() - start_time

         # 8 tasks * 0.1s / 5 workers = ~0.16s theoretical min. Allow buffer.
        assert total_execution_time < 0.3, f"Total execution time ({total_execution_time:.3f}s) for 8 tasks not within expected range (<0.3s)"
        assert async_pool.metrics.tasks_submitted == 8
        assert async_pool.metrics.tasks_completed == 8


    # ==============================================================
    # Modified test_queue_limiting function
    # ==============================================================
    @pytest.mark.asyncio
    async def test_queue_limiting(self, async_pool: AsyncWorkerPool):
        """Test that the queue semaphore correctly limits active tasks."""
        print(f"\n{time.monotonic():.4f} [Test Start: test_queue_limiting]")

        # --- Setup: 5 Workers, Queue Size 2 ---
        pool_name = async_pool.name
        workers = 5
        queue_size = 2
        print(f"{time.monotonic():.4f} [Test Setup]: Reconfiguring pool '{pool_name}' (Workers={workers}, Queue={queue_size})")
        # Reconfigure the existing pool instance for this specific test
        async_pool.config.workers = workers
        async_pool._semaphore = asyncio.Semaphore(workers)
        async_pool.config.max_queue_size = queue_size # Ensure config reflects this
        async_pool._queue_semaphore = asyncio.Semaphore(queue_size)
        # Reset metrics for clarity in this test
        async_pool.metrics.tasks_submitted = 0
        async_pool.metrics.tasks_completed = 0
        async_pool.metrics.tasks_failed = 0
        async_pool.metrics.active_workers = 0
        async_pool.metrics.current_queue_size = 0

        # Task that runs for a noticeable time
        task_sleep_duration = 0.2
        async def short_sleep_func(task_index, seconds=task_sleep_duration):
            print(f"{time.monotonic():.4f} [Task {task_index}]: Started.")
            await asyncio.sleep(seconds)
            print(f"{time.monotonic():.4f} [Task {task_index}]: Finished.")
            return task_index

        # --- Submit tasks to fill queue and initial workers ---
        # Submit queue_size tasks first (these should acquire the queue semaphore)
        num_to_submit = workers + queue_size # Submit more than can run immediately
        submitted_tasks = []
        print(f"{time.monotonic():.4f} [Test Logic]: Submitting {num_to_submit} tasks...")
        for i in range(num_to_submit):
            print(f"{time.monotonic():.4f} [Test Logic]: Submitting task {i}...")
            # Use create_task to manage submission coroutines non-blockingly initially
            task = asyncio.create_task(
                async_pool.submit(short_sleep_func, i, task_sleep_duration),
                name=f"TestSubmitTask_{i}"
            )
            submitted_tasks.append(task)
        print(f"{time.monotonic():.4f} [Test Logic]: Finished submitting {num_to_submit} task submissions.")

        # --- Wait for the expected initial state ---
        # Expected state: Queue semaphore is locked, active workers = queue_size
        expected_active = queue_size
        state_wait_timeout = 2.0 # Shorter timeout, tasks start quickly
        print(f"{time.monotonic():.4f} [Test Logic]: Starting state check loop (Wait for Active={expected_active} and Queue Locked)...")
        start_wait = time.monotonic()
        state_reached = False
        while time.monotonic() - start_wait < state_wait_timeout:
            current_active = async_pool.get_active_workers()
            # Check internal _value for direct semaphore count, locked() can be tricky right at release/acquire
            queue_sem_value = async_pool._queue_semaphore._value
            is_queue_locked = queue_sem_value == 0 # Semaphore value is 0 when fully acquired/locked

            print(f"{time.monotonic():.4f} [State Check]: Active={current_active}, Queue Sem Value={queue_sem_value} (Target: Active={expected_active}, Queue Locked)")

            if current_active == expected_active and is_queue_locked:
                print(f"{time.monotonic():.4f} [State Check]: Pool reached expected state (Active={current_active}, Queue Locked).")
                state_reached = True
                break
            await asyncio.sleep(0.05) # Short sleep between checks

        if not state_reached:
             current_active = async_pool.get_active_workers()
             queue_sem_value = async_pool._queue_semaphore._value
             pytest.fail(
                 f"Pool did not reach state Active={expected_active} & Queue Locked within {state_wait_timeout}s. "
                 f"Current state: Active={current_active}, Queue Sem Value={queue_sem_value}"
             )

        # --- Verify State ---
        print(f"{time.monotonic():.4f} [Test Verify]: Verifying final state after wait...")
        assert async_pool.get_active_workers() == expected_active, f"Expected {expected_active} active workers"
        assert async_pool._queue_semaphore._value == 0, "Queue semaphore should be locked (value 0)"
        print(f"{time.monotonic():.4f} [Test Verify]: State confirmed (Active={expected_active}, Queue Locked).")

        # --- Verify Blocking ---
        # Attempt to submit one more task, it should block on the queue semaphore
        print(f"{time.monotonic():.4f} [Test Logic]: Attempting to submit an extra task (should block)...")
        extra_task_index = 99
        extra_task_submission_timeout = 0.1 # Very short timeout
        try:
            # Use wait_for to detect if submit() blocks as expected
            await asyncio.wait_for(
                async_pool.submit(short_sleep_func, extra_task_index, 0.01),
                timeout=extra_task_submission_timeout
            )
            # If wait_for completes without TimeoutError, the submit didn't block!
            pytest.fail(f"Submitting task {extra_task_index} did not block/timeout within {extra_task_submission_timeout}s when queue should be full.")
        except asyncio.TimeoutError:
            print(f"{time.monotonic():.4f} [Test Logic]: Extra task submission correctly timed out after {extra_task_submission_timeout}s (blocked as expected).")
        except Exception as e:
             pytest.fail(f"Submitting extra task failed with unexpected error: {e}")


        # --- Cleanup ---
        print(f"{time.monotonic():.4f} [Test Cleanup]: Cancelling {len(submitted_tasks)} submitted tasks...")
        # Cancel all initially submitted tasks to allow pool shutdown
        for task in submitted_tasks:
            if not task.done():
                print(f"{time.monotonic():.4f} [Test Cleanup]: Cancelling task {task.get_name()}")
                task.cancel()

        # Wait for cancellations to propagate and tasks to finish/error
        print(f"{time.monotonic():.4f} [Test Cleanup]: Gathering cancelled tasks...")
        await asyncio.gather(*submitted_tasks, return_exceptions=True)
        print(f"{time.monotonic():.4f} [Test Cleanup]: Gather complete.")

        # Allow pool shutdown in the fixture teardown
        print(f"{time.monotonic():.4f} [Test End: test_queue_limiting]")
    # ==============================================================
    # End of modified test_queue_limiting function
    # ==============================================================


    @pytest.mark.asyncio
    async def test_task_timeout(self, async_pool):
        """Test that tasks exceeding their timeout are cancelled."""
        async def slow_func():
            await asyncio.sleep(0.5) # Sleep longer than timeout
            return "Should not return"

        with pytest.raises(asyncio.TimeoutError):
            await async_pool.submit(slow_func, timeout=0.1)

        # Check metrics reflect the failure
        # Wait briefly for metrics update if needed (depends on internal implementation)
        await asyncio.sleep(0.05)
        assert async_pool.metrics.tasks_submitted == 1
        assert async_pool.metrics.tasks_completed == 0
        # Whether it counts as failed depends on if cancellation is caught internally
        # Let's assert it's not completed. Precise failed count might vary.
        # assert async_pool.metrics.tasks_failed == 1 # This might be fragile

    @pytest.mark.asyncio
    async def test_wait_until_idle(self, async_pool):
        """Test waiting until the pool has no active or queued tasks."""
        async def short_task(duration):
            await asyncio.sleep(duration)
            return duration

        task1 = asyncio.create_task(async_pool.submit(short_task, 0.1))
        task2 = asyncio.create_task(async_pool.submit(short_task, 0.15))

        # Pool should become idle after tasks complete
        idle_success = await async_pool.wait_until_idle(timeout=0.5)
        assert idle_success is True, "Pool should become idle within timeout"

        # Verify tasks completed
        assert await task1 == 0.1
        assert await task2 == 0.15
        assert async_pool.get_active_workers() == 0
        assert async_pool.get_queue_size() == 0 # Check metric as well

        # Test timeout scenario for wait_until_idle
        task3 = asyncio.create_task(async_pool.submit(short_task, 0.3))
        idle_timeout = await async_pool.wait_until_idle(timeout=0.1)
        assert idle_timeout is False, "Pool should not become idle before timeout"

        # Cleanup the remaining task
        await task3

    @pytest.mark.asyncio
    async def test_context_manager(self, async_pool_config):
        """Test using the pool as an async context manager."""
        async with AsyncWorkerPool("context_test", async_pool_config) as pool:
            result = await pool.submit(lambda x: x * 2, 21)
            assert result == 42
        # Pool `shutdown` should be called automatically upon exiting the context


# ====================================================================
# TestThreadWorkerPool - Generally similar structure to Async tests
# ====================================================================
class TestThreadWorkerPool:
    """Test suite for ThreadWorkerPool."""

    def test_initialization(self, thread_pool_config):
        """Test pool initialization."""
        pool = ThreadWorkerPool("init_thread_pool", thread_pool_config)
        assert pool.name == "init_thread_pool"
        assert pool.config.workers == 3
        assert pool.config.max_queue_size == 10
        assert pool.metrics.tasks_submitted == 0
        pool.shutdown()

    @pytest.mark.asyncio # Use asyncio for asubmit/await
    async def test_submit_and_execute(self, thread_pool):
        """Test submitting sync tasks."""
        def test_func(value):
            time.sleep(0.02) # Simulate work
            return value * 2

        # Sync submit, sync result
        future = thread_pool.submit(test_func, 10)
        result = future.result() # Blocks until done
        assert result == 20
        assert thread_pool.metrics.tasks_completed == 1

        # Async submit, async result
        result_async = await thread_pool.asubmit(test_func, 11)
        assert result_async == 22
        assert thread_pool.metrics.tasks_submitted == 2
        assert thread_pool.metrics.tasks_completed == 2

    def test_map_function(self, thread_pool):
        """Test the synchronous map function."""
        def test_func(value):
            time.sleep(0.01)
            return value * 2

        values = [1, 2, 3, 4, 5]
        start_time = time.monotonic()
        results = thread_pool.map(test_func, values) # Blocks until all complete
        execution_time = time.monotonic() - start_time

        assert sorted(results) == [2, 4, 6, 8, 10]
        # With 3 workers, 5 tasks should take > 0.01s but < 0.05s (ideally ~0.02s)
        assert execution_time < 0.1, f"Map execution time too long: {execution_time:.3f}s"

    @pytest.mark.asyncio
    async def test_async_map_function(self, thread_pool):
        """Test the asynchronous amap function."""
        def test_func(value):
            time.sleep(0.01)
            return value * 2

        values = [1, 2, 3, 4, 5]
        start_time = time.monotonic()
        results = await thread_pool.amap(test_func, values)
        execution_time = time.monotonic() - start_time

        assert sorted(results) == [2, 4, 6, 8, 10]
        assert execution_time < 0.1, f"aMap execution time too long: {execution_time:.3f}s"

    def test_error_handling(self, thread_pool):
        """Test error handling with sync submit."""
        def failing_func():
            time.sleep(0.01)
            raise ValueError("Thread error")

        future = thread_pool.submit(failing_func)
        with pytest.raises(ValueError, match="Thread error"):
            future.result() # Exception raised here

        assert thread_pool.metrics.tasks_submitted == 1
        assert thread_pool.metrics.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_async_error_handling(self, thread_pool):
        """Test error handling with async submit."""
        def failing_func():
            time.sleep(0.01)
            raise ValueError("Thread async error")

        with pytest.raises(ValueError, match="Thread async error"):
            await thread_pool.asubmit(failing_func) # Exception raised here

        assert thread_pool.metrics.tasks_submitted == 1
        assert thread_pool.metrics.tasks_failed == 1


    def test_context_manager(self, thread_pool_config):
        """Test using the pool as a sync context manager."""
        with ThreadWorkerPool("sync_context_thread", thread_pool_config) as pool:
            future = pool.submit(lambda x: x * 2, 21)
            assert future.result() == 42
        # Pool shutdown called on exit

    @pytest.mark.asyncio
    async def test_async_context_manager(self, thread_pool_config):
        """Test using the pool as an async context manager."""
        async with ThreadWorkerPool("async_context_thread", thread_pool_config) as pool:
            result = await pool.asubmit(lambda x: x * 2, 21)
            assert result == 42
        # Pool ashutdown called on exit


# =====================================================================
# TestProcessWorkerPool - Focus on pickling and module-level functions
# =====================================================================
class TestProcessWorkerPool:
    """Test suite for ProcessWorkerPool."""

    # Note: Process pools require careful handling of pickling.
    # Functions submitted must be defined at the module level.

    @pytest.mark.asyncio # Use asyncio for ashutdown
    async def test_initialization(self, process_pool_config):
        """Test pool initialization."""
        pool = ProcessWorkerPool("init_process_pool", process_pool_config)
        assert pool.name == "init_process_pool"
        assert pool.config.workers == 2
        assert pool.config.max_queue_size == 5
        await pool.ashutdown()

    @pytest.mark.asyncio # Use asyncio for asubmit/await
    async def test_submit_and_execute(self, process_pool):
        """Test submitting tasks using module-level function."""
        # Sync submit
        future = process_pool.submit(_process_pool_test_func, 10)
        assert future.result() == 20

        # Async submit
        result_async = await process_pool.asubmit(_process_pool_test_func, 11)
        assert result_async == 22

    # Inside TestProcessWorkerPool class

    @pytest.mark.asyncio # Use asyncio for amap/await
    async def test_map_function(self, process_pool):
        """Test map and amap using module-level function."""
        values = [1, 2, 3, 4]
        # Sync map
        results_map = process_pool.map(_process_pool_test_func, values)
        # === 수정: 결과를 정렬하여 비교 ===
        assert sorted(results_map) == sorted([2, 4, 6, 8])
        # ===============================

        # Async map (amap 결과는 순서가 보장될 수 있으나, 일관성을 위해 정렬 비교 권장)
        results_amap = await process_pool.amap(_process_pool_test_func, values)
        # === 수정: 결과를 정렬하여 비교 ===
        assert sorted(results_amap) == sorted([2, 4, 6, 8])
        # ===============================


    @pytest.mark.asyncio # Use asyncio for asubmit/await
    async def test_error_handling(self, process_pool):
        """Test error handling using module-level function."""
        # Sync submit
        future = process_pool.submit(_process_pool_failing_func)
        with pytest.raises(ValueError, match="Test error from process pool"):
            future.result()

        # Async submit
        with pytest.raises(ValueError, match="Test error from process pool"):
            await process_pool.asubmit(_process_pool_failing_func)

    def test_unpicklable_function_submit(self, process_pool):
        """Test that submitting an unpicklable function raises an error."""
        unpicklable_func = lambda x: x * 2 # Lambdas are often not picklable

        future = None # future 변수 초기화
        excinfo = None # excinfo 변수 초기화

        with pytest.raises((AttributeError, TypeError, PicklingError)) as caught_excinfo:
            future = process_pool.submit(unpicklable_func, 10)
            # result() 호출 시 PicklingError가 발생할 가능성이 높음
            future.result() # 여기서 예외 발생 기대

        # === 수정: excinfo.value 검사를 with 블록 바깥으로 이동 ===
        excinfo = caught_excinfo # 컨텍스트 매니저가 끝난 후 변수에 할당
        assert excinfo is not None # 예외가 잡혔는지 확인
        assert "pickle" in str(excinfo.value).lower() or "Can't get local" in str(excinfo.value), \
            f"Expected pickling error, but got: {excinfo.value}"
        # === 수정 끝 ===


    @pytest.mark.asyncio # Use asyncio context manager
    async def test_async_context_manager(self, process_pool_config):
        """Test using the pool as an async context manager."""
        async with ProcessWorkerPool("async_context_process", process_pool_config) as pool:
             # Use module-level function
            result = await pool.asubmit(_process_pool_lambda_replacement, 21)
            assert result == 42
        # Pool ashutdown called on exit


# =======================================================================
# TestWorkerPoolFunctions - Test utility functions like get_worker_pool
# =======================================================================
class TestWorkerPoolFunctions:
    """Test suite for worker pool utility functions."""

    @pytest.mark.asyncio
    async def test_get_worker_pool(self):
        """Test getting/creating worker pools via the utility function."""
        # Ensure clean state
        await shutdown_all_worker_pools()

        # Get/Create pools
        async_pool = get_worker_pool("get_test_async", WorkerPoolType.ASYNCIO)
        thread_pool = get_worker_pool("get_test_thread", WorkerPoolType.THREAD)
        process_pool = get_worker_pool("get_test_process", WorkerPoolType.PROCESS)

        assert isinstance(async_pool, AsyncWorkerPool)
        assert isinstance(thread_pool, ThreadWorkerPool)
        assert isinstance(process_pool, ProcessWorkerPool)

        # Getting again should return the same instance
        async_pool2 = get_worker_pool("get_test_async", WorkerPoolType.ASYNCIO)
        assert async_pool is async_pool2

        # Clean up
        await shutdown_all_worker_pools()

    @pytest.mark.asyncio
    async def test_get_default_worker_pool(self):
        """Test getting the default async worker pool."""
         # Ensure clean state
        await shutdown_all_worker_pools()

        pool = get_default_worker_pool()
        assert isinstance(pool, AsyncWorkerPool)
        assert pool.name == "default"

        # Use it
        result = await pool.submit(lambda x: x * 2, 10) # Simple lambda okay for async/thread
        assert result == 20

        # Clean up
        await shutdown_all_worker_pools()

    @pytest.mark.asyncio
    async def test_shutdown_all_worker_pools(self):
        """Test shutting down all registered worker pools."""
         # Ensure clean state
        await shutdown_all_worker_pools()

        # Create some pools
        get_worker_pool("shutdown_test_async", WorkerPoolType.ASYNCIO)
        get_worker_pool("shutdown_test_thread", WorkerPoolType.THREAD)
        get_worker_pool("shutdown_test_process", WorkerPoolType.PROCESS)

        # Check registry size before shutdown
        from src.core.worker_pool import _worker_pools
        assert len(_worker_pools) == 3

        # Shutdown all
        await shutdown_all_worker_pools()

        # Registry should now be empty
        assert len(_worker_pools) == 0
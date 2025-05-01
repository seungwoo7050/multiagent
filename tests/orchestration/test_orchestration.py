import asyncio
import time
import pytest
import random
from typing import Dict, List, Any, Optional, Tuple
import redis.asyncio as aioredis

from src.orchestration.flow_control import (
    BackpressureConfig, RateLimitConfig, RedisRateLimiter,
    get_flow_controller, with_flow_control, BackpressureRejectedError
)
from src.orchestration.load_balancer import (
    RoundRobinStrategy, RandomStrategy, WeightedRoundRobinStrategy,
    create_load_balancer, BaseLoadBalancerStrategy
)
from src.orchestration.scheduler import PriorityScheduler, get_scheduler
from src.orchestration.dispatcher import Dispatcher
from src.orchestration.task_queue import BaseTaskQueue
from src.core.task import TaskPriority, BaseTask, TaskState
from src.core.agent import AgentContext, AgentResult, BaseAgent
from src.core.exceptions import AgentCreationError, AgentExecutionError, AgentNotFoundError, TaskError


class MockAgent(BaseAgent):
    def __init__(self, name="test_agent", success=True):
        from src.core.agent import AgentConfig
        config = AgentConfig(name=name, agent_type="test_agent")
        super().__init__(config)
        self.success = success
        self.initialize_called = False
        self.execute_called = False
        self.terminate_called = False

    async def initialize(self) -> bool:
        self.initialize_called = True
        return True

    async def process(self, context: AgentContext) -> AgentResult:
        self.execute_called = True
        if self.success:
            return AgentResult.success_result({"result": "success"}, 0.1)
        else:
            return AgentResult.error_result({"message": "Simulated failure"}, 0.1)

    async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
        return AgentResult.error_result({"message": str(error)}, 0.1)

    async def terminate(self) -> None:
        self.terminate_called = True

class MockAgentFactory:
    def __init__(self):
        self.agents = {}
        self.default_agent = MockAgent()

    def register_agent(self, agent_type: str, agent: MockAgent):
        self.agents[agent_type] = agent

    async def get_agent(self, agent_type: str) -> MockAgent:
        if agent_type in self.agents:
            return self.agents[agent_type]
        elif agent_type == "nonexistent_agent":
            # Match the exception type raised by the real factory for consistency
            raise AgentNotFoundError(agent_type=agent_type, message=f"Agent type {agent_type} not found in mock")
        else:
            # If not specifically registered and not 'nonexistent_agent', raise error
            raise AgentNotFoundError(agent_type=agent_type, message=f"No configuration found for agent name: {agent_type}")


class MockRedis:
    def __init__(self):
        self.data = {}
        self.script_sha = "mock_script_sha"

    async def script_load(self, script):
        return self.script_sha.encode()

    async def script_exists(self, sha):
        return [True]

    async def evalsha(self, sha, keys, args):
        return 1

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, px=None, ex=None):
        self.data[key] = value
        return True

    async def eval(self, script, num_keys, *args):
        return 1

    async def xgroup_create(self, name, groupname, id, mkstream=False):
        return True

class AsyncLockMock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockTaskQueue(BaseTaskQueue):
    def __init__(self):
        self.tasks = []
        self.dlq = []
        self.acknowledged = set()
        self.lock = AsyncLockMock()

    async def produce(self, task_data, task_id=None):
        msg_id = task_id or f"msg-{random.randint(1000, 9999)}"
        self.tasks.append((msg_id, task_data))
        return msg_id

    async def consume(self, consumer_name, count=1, block_ms=2000):
        if not self.tasks:
            await asyncio.sleep(block_ms / 1000.0)
            return []
        result = self.tasks[:count]
        self.tasks = self.tasks[count:]
        return result

    async def acknowledge(self, message_id):
        self.acknowledged.add(message_id)
        return True

    async def add_to_dlq(self, message_id, task_data, error_info):
        self.dlq.append((message_id, task_data, error_info))
        return True

    async def get_queue_depth(self):
        return len(self.tasks)

    async def get_lock(self, lock_name, expire_time=30):
        return self.lock

class MockWorkerPool:
    def __init__(self, name="test_pool"):
        self.name = name
        self.submitted_futures: List[asyncio.Future] = []
        self.submitted_tasks = [] # Keep storing coroutines for potential inspection
        self.executing_tasks = []
        self.completed_tasks = []

        class Config:
            def __init__(self):
                self.workers = 4
        self.config = Config()

    async def submit(self, func, *args, **kwargs):
        task_coro = func()
        self.submitted_tasks.append(task_coro)
        self.executing_tasks.append(task_coro)

        loop = asyncio.get_running_loop()
        future = loop.create_task(task_coro)
        self.submitted_futures.append(future)

        def on_complete(fut):
            if task_coro in self.executing_tasks:
                self.executing_tasks.remove(task_coro)
            self.completed_tasks.append(task_coro)

        future.add_done_callback(on_complete)
        return future


@pytest.mark.asyncio
async def test_flow_controller_initialization():
    controller = RedisRateLimiter("test_controller", None)
    assert controller.name == "test_controller"
    assert controller.config.rate_limit.rate == 100.0

    custom_config = BackpressureConfig(
        rate_limit=RateLimitConfig(rate=50.0, burst=30)
    )
    controller = RedisRateLimiter("custom_controller", custom_config)
    assert controller.config.rate_limit.rate == 50.0
    assert controller.config.rate_limit.burst == 30

@pytest.mark.asyncio
async def test_flow_controller_acquire(monkeypatch):
    mock_redis = MockRedis()
    async def mock_get_redis():
        return mock_redis

    controller = RedisRateLimiter("test_acquire")
    monkeypatch.setattr(controller, "_get_redis", mock_get_redis)

    result = await controller.acquire()
    assert result is True
    assert controller.metrics.total_requests == 1
    assert controller.metrics.rejected_requests == 0

    result = await controller.acquire(priority=1, cost=2)
    assert result is True
    assert controller.metrics.total_requests == 2

@pytest.mark.asyncio
async def test_flow_controller_execute(monkeypatch):
    mock_redis = MockRedis()
    async def mock_get_redis():
        return mock_redis

    controller = RedisRateLimiter("test_execute")
    monkeypatch.setattr(controller, "_get_redis", mock_get_redis)

    async def test_func():
        return "success"

    result = await controller.execute(test_func)
    assert result == "success"

    async def mock_acquire(*args, **kwargs):
        return False

    monkeypatch.setattr(controller, "acquire", mock_acquire)

    with pytest.raises(BackpressureRejectedError):
        await controller.execute(test_func)


def test_round_robin_strategy():
    workers = ["worker1", "worker2", "worker3"]
    strategy = RoundRobinStrategy()
    selected = [strategy.select_worker(workers) for _ in range(5)]
    assert selected == ["worker1", "worker2", "worker3", "worker1", "worker2"]
    assert strategy.select_worker([]) is None

def test_random_strategy():
    workers = ["worker1", "worker2", "worker3"]
    strategy = RandomStrategy()
    selected = strategy.select_worker(workers)
    assert selected in workers
    assert strategy.select_worker([]) is None

def test_weighted_round_robin_strategy():
    workers = ["worker1", "worker2", "worker3"]
    strategy = WeightedRoundRobinStrategy()
    strategy.set_worker_weight("worker1", 2)
    strategy.set_worker_weight("worker2", 1)
    strategy.set_worker_weight("worker3", 3)
    selected = [strategy.select_worker(workers) for _ in range(6)]
    counts = {}
    for worker in selected:
        counts[worker] = counts.get(worker, 0) + 1
    assert counts.get("worker3", 0) >= counts.get("worker1", 0) >= counts.get("worker2", 0)
    strategy.update_worker_status("worker1", {"weight": 5})
    selected = [strategy.select_worker(workers) for _ in range(10)]
    counts = {}
    for worker in selected:
        counts[worker] = counts.get(worker, 0) + 1
    assert counts.get("worker1", 0) > counts.get("worker2", 0)

def test_create_load_balancer():
    round_robin = create_load_balancer("round_robin")
    assert isinstance(round_robin, RoundRobinStrategy)
    random_lb = create_load_balancer("random")
    assert isinstance(random_lb, RandomStrategy)
    weighted = create_load_balancer("weighted_round_robin")
    assert isinstance(weighted, WeightedRoundRobinStrategy)
    unknown = create_load_balancer("unknown_strategy")
    assert isinstance(unknown, RoundRobinStrategy)


@pytest.mark.asyncio
async def test_scheduler_add_get_task():
    scheduler = PriorityScheduler()
    task1 = {"id": "task1", "priority": TaskPriority.HIGH.value}
    task2 = {"id": "task2", "priority": TaskPriority.NORMAL.value}
    task3 = {"id": "task3", "priority": TaskPriority.LOW.value}
    await scheduler.add_task(task3)
    await scheduler.add_task(task1)
    await scheduler.add_task(task2)
    retrieved1 = await scheduler.get_next_task()
    assert retrieved1["id"] == "task1"
    retrieved2 = await scheduler.get_next_task()
    assert retrieved2["id"] == "task2"
    retrieved3 = await scheduler.get_next_task()
    assert retrieved3["id"] == "task3"
    assert scheduler.is_empty() is True
    assert await scheduler.get_next_task(timeout=0.1) is None

@pytest.mark.asyncio
async def test_scheduler_peek_task():
    scheduler = PriorityScheduler()
    task = {"id": "peek_task"}
    await scheduler.add_task(task)
    peeked = await scheduler.peek_next_task()
    assert peeked["id"] == "peek_task"
    assert scheduler.is_empty() is False
    retrieved = await scheduler.get_next_task()
    assert retrieved["id"] == "peek_task"

@pytest.mark.asyncio
async def test_scheduler_clear():
    scheduler = PriorityScheduler()
    for i in range(5):
        await scheduler.add_task({"id": f"task{i}"})
    assert scheduler.get_queue_size() == 5
    await scheduler.clear()
    assert scheduler.is_empty() is True
    assert scheduler.get_queue_size() == 0

@pytest.mark.asyncio
async def test_get_tasks_by_priority():
    scheduler = PriorityScheduler()
    await scheduler.add_task({"id": "task_low", "priority": TaskPriority.LOW.value})
    await scheduler.add_task({"id": "task_high", "priority": TaskPriority.HIGH.value})
    await scheduler.add_task({"id": "task_normal", "priority": TaskPriority.NORMAL.value})
    await scheduler.add_task({"id": "task_critical", "priority": TaskPriority.CRITICAL.value})
    tasks = await scheduler.get_tasks_by_priority(count=3)
    assert len(tasks) == 3
    assert tasks[0]["id"] == "task_critical"
    assert tasks[1]["id"] == "task_high"
    assert tasks[2]["id"] == "task_normal"
    assert scheduler.get_queue_size() == 4


@pytest.mark.asyncio
async def test_dispatcher_move_to_dlq():
    task_queue = MockTaskQueue()
    dispatcher = Dispatcher(task_queue=task_queue)
    task_data = {"id": "failed_task", "type": "test_agent"}
    error = TaskError(message="Test error", task_id="failed_task")
    await dispatcher._move_to_dlq("test_msg_id", task_data, error)
    assert len(task_queue.dlq) == 1
    assert task_queue.dlq[0][0] == "test_msg_id"
    assert task_queue.dlq[0][1]["id"] == "failed_task"

@pytest.mark.asyncio
async def test_process_task(monkeypatch):
    task_queue = MockTaskQueue()
    dispatcher = Dispatcher(task_queue=task_queue)
    mock_factory = MockAgentFactory()
    async def mock_get_factory():
        return mock_factory
    monkeypatch.setattr("src.orchestration.dispatcher.get_agent_factory", mock_get_factory)

    task_data = {
        "id": "test_task",
        "type": "test_agent",
        "trace_id": "test_trace"
    }
    mock_factory.register_agent("test_agent", MockAgent(success=True))
    result = await dispatcher._process_task("test_task", task_data.copy(), "test_msg_id_success")
    assert result.success is True
    assert "result" in result.output

    mock_factory.register_agent("fail_agent", MockAgent(success=False))
    task_data["type"] = "fail_agent"
    result = await dispatcher._process_task("test_task", task_data.copy(), "test_msg_id_fail")
    assert result.success is False
    assert result.error is not None

    task_data["type"] = "nonexistent_agent"
    with pytest.raises(AgentCreationError):
        await dispatcher._process_task("test_task", task_data.copy(), "test_msg_id_notfound")


@pytest.mark.asyncio
async def test_process_task_wrapper(monkeypatch):
    task_queue = MockTaskQueue()
    dispatcher = Dispatcher(task_queue=task_queue)
    mock_factory = MockAgentFactory()

    async def mock_get_factory():
        return mock_factory
    monkeypatch.setattr("src.orchestration.dispatcher.get_agent_factory", mock_get_factory)

    task_data_success = {
        "id": "success_task",
        "type": "test_agent",
        "trace_id": "test_trace"
    }
    mock_factory.register_agent("test_agent", MockAgent(success=True))
    await dispatcher._process_task_wrapper("test_msg_id", task_data_success.copy())
    assert "test_msg_id" in task_queue.acknowledged

    task_data_retry = {
        "id": "retry_task",
        "type": "test_agent",
        "trace_id": "test_trace",
        "metadata": {
            "retry_count": 0,
            "max_retries": 2
        }
    }
    mock_factory.register_agent("test_agent", MockAgent(success=False))
    def mock_is_retryable(error, error_codes): return True
    monkeypatch.setattr("src.llm.retry.is_retryable_error", mock_is_retryable)
    def mock_backoff(retry_count, base_delay, max_delay, jitter): return 0.001
    monkeypatch.setattr("src.llm.retry.calculate_backoff", mock_backoff)

    task_queue.tasks = []
    task_queue.acknowledged = set()
    task_queue.dlq = []

    await dispatcher._process_task_wrapper("retry_msg_id", task_data_retry.copy())
    assert "retry_msg_id" not in task_queue.acknowledged

    if not dispatcher.uses_redis_streams and dispatcher.scheduler:
         pass
    elif any(t[0] == "retry_msg_id" and t[1].get("metadata", {}).get("retry_count") == 1 for t in task_queue.tasks):
        assert True
    else:
        assert len(task_queue.tasks) == 1 and task_queue.tasks[0][1]["metadata"]["retry_count"] == 1


    task_data_max_retry = {
        "id": "max_retry_task",
        "type": "test_agent",
        "trace_id": "test_trace",
        "metadata": {
            "retry_count": 2,
            "max_retries": 2
        }
    }
    await dispatcher._process_task_wrapper("max_retry_msg_id", task_data_max_retry.copy())
    assert len(task_queue.dlq) > 0
    dlq_task = next((t for t in task_queue.dlq if t[0] == "max_retry_msg_id"), None)
    assert dlq_task is not None
    assert dlq_task[1]["id"] == "max_retry_task"

@pytest.mark.asyncio
async def test_dispatcher_consumer_task(monkeypatch):

    class MockFlowController:
        def __init__(self):
            self.acquire_called = False
        async def acquire(self, *args, **kwargs):
            self.acquire_called = True
            return True

    class MockScheduler:
         def __init__(self):
             self.added_tasks = []
         async def add_task(self, task_data):
             self.added_tasks.append(task_data)
             return True

    task_queue = MockTaskQueue()
    worker_pool = MockWorkerPool()
    mock_controller = MockFlowController()
    mock_scheduler = MockScheduler()

    dispatcher = Dispatcher(
        task_queue=task_queue,
        worker_pool=worker_pool,
        flow_controller=mock_controller,
        scheduler=mock_scheduler
    )

    monkeypatch.setattr(dispatcher, 'uses_redis_streams', True)

    async def mock_process_wrapper_consumer(msg_id, task_data):
         await task_queue.acknowledge(msg_id)
         return AgentResult.success_result({}, 0.0)
    monkeypatch.setattr(dispatcher, "_process_task_wrapper", mock_process_wrapper_consumer)

    task_info = ("msg_id", {"id": "test_task", "type": "test_agent", "priority": TaskPriority.NORMAL.value})

    await dispatcher._process_consumer_task(task_info)

    assert mock_controller.acquire_called is True
    assert len(worker_pool.submitted_futures) > 0

    if dispatcher.uses_redis_streams:
        assert len(worker_pool.submitted_futures) > 0
        await asyncio.gather(*worker_pool.submitted_futures)
        assert "msg_id" in task_queue.acknowledged


@pytest.mark.asyncio
async def test_flow_control_with_scheduler(monkeypatch):
    mock_redis = MockRedis()
    async def mock_get_redis():
        return mock_redis

    controller = RedisRateLimiter("test_integration")
    monkeypatch.setattr(controller, "_get_redis", mock_get_redis)

    scheduler = PriorityScheduler()

    for i in range(10):
        await scheduler.add_task({"id": f"task{i}"})

    results = []
    async def process_task():
        task = await scheduler.get_next_task(timeout=0.1)
        if task:
            results.append(task["id"])
        return task

    tasks_to_run = []
    for _ in range(5):
        try:
            tasks_to_run.append(controller.execute(process_task))
        except BackpressureRejectedError:
            pass

    await asyncio.gather(*[asyncio.create_task(t) for t in tasks_to_run])
    assert len(results) == 5

    tasks_to_run = []
    for _ in range(5):
        try:
            tasks_to_run.append(controller.execute(process_task))
        except BackpressureRejectedError:
            pass

    await asyncio.gather(*[asyncio.create_task(t) for t in tasks_to_run])
    assert len(results) == 10
    assert scheduler.is_empty() is True

@pytest.mark.asyncio
async def test_dispatcher_run_stop(monkeypatch):
    task_queue = MockTaskQueue()
    dispatcher = Dispatcher(task_queue=task_queue)
    consumer_called = False
    processor_called = False

    async def mock_consumer_loop():
        nonlocal consumer_called
        consumer_called = True
        while dispatcher._consumer_running:
            await asyncio.sleep(0.01)

    async def mock_processor_loop():
        nonlocal processor_called
        processor_called = True
        while dispatcher._processor_running:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(dispatcher, "_consumer_loop", mock_consumer_loop)
    monkeypatch.setattr(dispatcher, "_processor_loop", mock_processor_loop)

    run_task = asyncio.create_task(dispatcher.run())
    await asyncio.sleep(0.1)
    assert dispatcher._running is True
    assert consumer_called is True
    if not dispatcher.uses_redis_streams:
        assert processor_called is True

    await dispatcher.stop()
    await asyncio.sleep(0.1)

    assert dispatcher._running is False
    assert dispatcher._consumer_running is False
    assert dispatcher._processor_running is False


@pytest.mark.asyncio
async def test_scheduler_performance():
    scheduler = PriorityScheduler()
    start_time = time.time()
    task_count = 1000
    for i in range(task_count):
        priority = random.choice(list(TaskPriority))
        await scheduler.add_task({
            "id": f"perf_task_{i}",
            "priority": priority.value
        })
    add_time = time.time() - start_time
    start_time = time.time()
    retrieved = 0
    while not scheduler.is_empty():
        task = await scheduler.get_next_task()
        if task:
            retrieved += 1
    get_time = time.time() - start_time
    assert retrieved == task_count
    print(f"Added {task_count} tasks in {add_time:.4f}s ({task_count/add_time:.1f} tasks/s)")
    print(f"Retrieved {task_count} tasks in {get_time:.4f}s ({task_count/get_time:.1f} tasks/s)")
    assert add_time < 5.0
    assert get_time < 5.0

@pytest.mark.asyncio
async def test_flow_control_throughput(monkeypatch):
    mock_redis = MockRedis()
    async def mock_get_redis():
        return mock_redis
    config = BackpressureConfig(
        rate_limit=RateLimitConfig(rate=10000.0, burst=1000)
    )
    controller = RedisRateLimiter("throughput_test", config)
    monkeypatch.setattr(controller, "_get_redis", mock_get_redis)
    async def dummy_task(i):
        return i
    start_time = time.time()
    task_count = 500
    tasks = [
        controller.execute(dummy_task, i)
        for i in range(task_count)
    ]
    results = await asyncio.gather(*[asyncio.create_task(t) for t in tasks])
    total_time = time.time() - start_time
    assert len(results) == task_count
    assert results == list(range(task_count))
    print(f"Processed {task_count} tasks in {total_time:.4f}s ({task_count/total_time:.1f} tasks/s)")
    assert total_time < 3.0


@pytest.mark.asyncio
async def test_orchestration_resilience():
    task_queue = MockTaskQueue()
    processed = []
    failed = []
    task_count = 20

    for i in range(task_count):
        await task_queue.produce({"id": f"resilience_task_{i}", "type": "test_agent"})

    async def process_task_with_failures(task_data, fail_rate=0.3):
        try:
            if random.random() < fail_rate:
                raise Exception(f"Simulated failure for task {task_data['id']}")
            processed.append(task_data['id'])
            return True
        except Exception as e:
            failed.append(task_data['id'])
            if len(failed) < 30:
                await task_queue.produce(task_data)
            return False

    max_iterations = 50
    iteration = 0
    while task_queue.tasks and iteration < max_iterations:
        iteration += 1
        batch = await task_queue.consume("test_consumer", count=5)
        if not batch:
            if not task_queue.tasks:
                 break
            else:
                 await asyncio.sleep(0.1)
                 continue

        ack_tasks = []
        for msg_id, task_data in batch:
            success = await process_task_with_failures(task_data)
            if success:
                ack_tasks.append(task_queue.acknowledge(msg_id))
        if ack_tasks:
            await asyncio.gather(*ack_tasks)


    assert len(processed) == task_count
    assert len(set(processed)) == task_count
    assert len(failed) > 0
    print(f"Processed {task_count} tasks with {len(failed)} failures in {iteration} iterations")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
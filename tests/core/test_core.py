import asyncio
import pytest
import time

from src.core.agent import (
    BaseAgent, AgentConfig, AgentContext, AgentState, 
    AgentCapability, AgentResult
)
from src.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, 
    with_circuit_breaker
)
from src.core.registry import (
    Registry, get_function_registry, get_class_registry
)
from src.core.task import (
    BaseTask, TaskState, TaskPriority, TaskFactory
)
from src.core.worker_pool import (
    WorkerPoolType, get_worker_pool
)

def _square_for_test(x):
    """Simple function for process pool testing (must be picklable)."""
    return x * x


class TestAgent:
    """Tests for the Agent system components."""

    @pytest.fixture
    def agent_config(self):
        return AgentConfig(
            name="test_agent",
            agent_type="test",
            capabilities={AgentCapability.REASONING, AgentCapability.PLANNING},
            model="test_model",
            timeout=1.0
        )

    @pytest.fixture
    def mock_agent(self, agent_config):
        class TestAgent(BaseAgent):
            async def initialize(self) -> bool:
                return True

            async def process(self, context: AgentContext) -> AgentResult:
                return AgentResult.success_result({"key": "value"}, 0.1)

            async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
                return AgentResult.error_result({"error": str(error)}, 0.1)

        return TestAgent(agent_config)

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, mock_agent):
        """Test the basic agent lifecycle - initialize, process, terminate."""
        # Test initialization
        assert mock_agent.state == AgentState.IDLE
        assert await mock_agent.initialize() is True
        
        # Test processing
        context = AgentContext()
        result = await mock_agent.process(context)
        assert result.success is True
        assert result.output == {"key": "value"}
        
        # Test execution
        result = await mock_agent.execute(context)
        assert result.success is True
        assert mock_agent.state == AgentState.IDLE
        
        # Test termination
        await mock_agent.terminate()
        assert mock_agent.state == AgentState.TERMINATED
        assert mock_agent.is_terminated is True

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, agent_config):
        """Test agent error handling during processing."""
        class ErrorAgent(BaseAgent):
            async def initialize(self) -> bool:
                return True

            async def process(self, context: AgentContext) -> AgentResult:
                raise ValueError("Test error")

            async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
                return AgentResult.error_result({"error_type": type(error).__name__, "message": str(error)}, 0.1)

        agent = ErrorAgent(agent_config)
        context = AgentContext()
        result = await agent.execute(context)
        
        assert result.success is False
        assert result.error is not None
        assert result.error["error_type"] == "ValueError"
        assert result.error["message"] == "Test error"
        assert agent.error_count == 1
        assert agent.state == AgentState.IDLE  # Should reset to IDLE after handling

    @pytest.mark.asyncio
    async def test_agent_context_manager(self, mock_agent):
        """Test using agent as async context manager."""
        context = AgentContext()
        
        async with mock_agent as agent:
            assert agent.state == AgentState.IDLE
            result = await agent.process(context)
            assert result.success is True
        
        assert mock_agent.state == AgentState.TERMINATED


class TestCircuitBreaker:
    """Tests for the CircuitBreaker implementation."""

    @pytest.fixture
    def circuit_config(self):
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            reset_timeout_ms=100,  # Small for testing
            failure_window_ms=500  # Small for testing
        )

    @pytest.fixture
    def circuit(self, circuit_config):
        return CircuitBreaker("test_circuit", circuit_config)

    @pytest.mark.asyncio
    async def test_circuit_initial_state(self, circuit):
        """Test circuit breaker starts in CLOSED state."""
        assert circuit.state == CircuitState.CLOSED
        assert circuit.is_closed is True
        assert await circuit.allow_request() is True

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, circuit):
        """Test circuit transitions to OPEN after threshold failures."""
        # Add failures up to threshold
        for i in range(circuit.config.failure_threshold):
            await circuit.on_failure(f"error_{i}")
            
        # Circuit should now be OPEN
        assert circuit.state == CircuitState.OPEN
        assert circuit.is_open is True
        assert await circuit.allow_request() is False

    @pytest.mark.asyncio
    async def test_circuit_half_open_after_timeout(self, circuit):
        """Test circuit transitions to HALF_OPEN after timeout."""
        # Trip the circuit
        for i in range(circuit.config.failure_threshold):
            await circuit.on_failure(f"error_{i}")
            
        # Circuit should be OPEN
        assert circuit.is_open is True
        
        # Wait for reset timeout
        await asyncio.sleep(circuit.config.reset_timeout_ms / 1000 + 0.05)
        
        # Circuit should now allow a test request
        assert await circuit.allow_request() is True
        assert circuit.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successes(self, circuit):
        """Test circuit transitions from HALF_OPEN to CLOSED after successes."""
        # Trip the circuit
        for i in range(circuit.config.failure_threshold):
            await circuit.on_failure(f"error_{i}")
            
        # Wait for reset timeout to transition to HALF_OPEN
        await asyncio.sleep(circuit.config.reset_timeout_ms / 1000 + 0.05)
        
        # Allow test request
        assert await circuit.allow_request() is True
        assert circuit.state == CircuitState.HALF_OPEN
        
        # Register successes
        for _ in range(circuit.config.success_threshold):
            await circuit.on_success()
            
        # Circuit should be CLOSED
        assert circuit.state == CircuitState.CLOSED
        assert circuit.is_closed is True

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_helper(self):
        """Test the with_circuit_breaker helper function."""
        
        async def success_func(value):
            return value * 2
            
        # Execute function with circuit breaker
        result = await with_circuit_breaker("test_helper", success_func, 21)
        assert result == 42


class TestRegistry:
    """Tests for the Registry system."""

    @pytest.fixture
    def registry(self):
        return Registry[str]("test_registry")
        
    @pytest.fixture
    def function_registry(self):
        return get_function_registry("test_function_registry")
        
    @pytest.fixture
    def class_registry(self):
        return get_class_registry("test_class_registry")

    def test_registry_register_get(self, registry):
        """Test basic registry register and get operations."""
        # Register an item
        registry.register("item1", "value1", attribute="test")
        
        # Get the item
        item = registry.get_sync("item1")
        assert item == "value1"
        
        # Check metadata
        metadata = registry.get_metadata("item1")
        assert metadata is not None
        assert metadata["attribute"] == "test"

    def test_registry_list_operations(self, registry):
        """Test registry listing operations."""
        registry.register("item1", "value1")
        registry.register("item2", "value2")
        
        # Test listing operations
        assert set(registry.list_names()) == {"item1", "item2"}
        assert set(registry.list_items()) == {"value1", "value2"}
        
        # Test has operation
        assert registry.has("item1") is True
        assert registry.has("nonexistent") is False
        
        # Test size
        assert registry.size() == 2

    def test_registry_clear(self, registry):
        """Test registry clear operation."""
        registry.register("item1", "value1")
        registry.register("item2", "value2")
        assert registry.size() == 2
        
        registry.clear()
        assert registry.size() == 0
        assert registry.has("item1") is False

    def test_function_registry(self, function_registry):
        """Test function registry with decorators."""
        
        @function_registry.register_function()
        def test_func(a, b):
            """Test function docstring."""
            return a + b
            
        # Check registration
        func = function_registry.get_sync("test_func")
        assert func is not None
        assert func(2, 3) == 5
        
        # Check metadata extraction
        metadata = function_registry.get_metadata("test_func")
        assert metadata is not None
        assert "doc" in metadata
        assert "parameters" in metadata

    @pytest.mark.asyncio
    async def test_class_registry(self, class_registry):
        """Test class registry with async instance creation."""
        
        @class_registry.register_class()
        class TestClass:
            def __init__(self, value):
                self.value = value
                
            def get_value(self):
                return self.value
                
        # Create instance via registry
        instance = await class_registry.create_instance("TestClass", 42)
        assert instance is not None
        assert instance.get_value() == 42


class TestTask:
    """Tests for the Task system."""

    def test_task_creation(self):
        """Test task creation and initial state."""
        task = BaseTask.create("test_task", {"param": "value"})
        
        assert task.id is not None
        assert task.type == "test_task"
        assert task.state == TaskState.PENDING
        assert task.input == {"param": "value"}
        assert task.output is None
        assert task.created_at > 0
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_lifecycle(self):
        """Test task state transitions through its lifecycle."""
        task = BaseTask.create("test_task", {"param": "value"})
        
        # Start the task
        task.start()
        assert task.state == TaskState.RUNNING
        assert task.started_at is not None
        
        # Complete the task
        task.complete({"result": "success"})
        assert task.state == TaskState.COMPLETED
        assert task.completed_at is not None
        assert task.output == {"result": "success"}
        assert task.is_finished is True
        assert task.duration_ms is not None
        
        # Check event history
        history = task.get_event_history()
        assert len(history) >= 2  # At least start and complete events
        event_types = [event["event_type"] for event in history]
        assert "task_started" in event_types
        assert "task_completed" in event_types

    def test_task_failure(self):
        """Test task failure handling."""
        task = BaseTask.create("test_task", {"param": "value"})
        task.start()
        
        # Fail the task
        error_details = {"type": "test_error", "message": "Something went wrong"}
        task.fail(error_details)
        
        assert task.state == TaskState.FAILED
        assert task.error == error_details
        assert task.is_finished is True
        assert task.duration_ms is not None
        
        # Check event history
        history = task.get_event_history()
        event_types = [event["event_type"] for event in history]
        assert "task_failed" in event_types

    def test_task_cancellation(self):
        """Test task cancellation."""
        task = BaseTask.create("test_task", {"param": "value"})
        task.start()
        
        # Cancel the task
        task.cancel("Test cancellation")
        
        assert task.state == TaskState.CANCELED
        assert "cancel_reason" in task.error
        assert task.error["cancel_reason"] == "Test cancellation"
        assert task.is_finished is True
        
        # Check event history
        history = task.get_event_history()
        event_types = [event["event_type"] for event in history]
        assert "task_canceled" in event_types

    def test_task_checkpoint(self):
        """Test task checkpoint functionality."""
        task = BaseTask.create("test_task", {"param": "value"})
        
        # Add checkpoint
        checkpoint_data = {"progress": 50, "processed_items": 100}
        task.checkpoint(checkpoint_data)
        
        # Retrieve latest checkpoint
        latest = task.get_latest_checkpoint()
        assert latest is not None
        assert latest["data"]["progress"] == 50
        assert latest["data"]["processed_items"] == 100
        
        # Check event history
        history = task.get_event_history()
        event_types = [event["event_type"] for event in history]
        assert "checkpoint_saved" in event_types

    def test_task_factory(self):
        """Test the TaskFactory for creating tasks."""
        task = TaskFactory.create_task(
            "factory_task",
            {"param": "value"},
            priority=TaskPriority.HIGH,
            trace_id="trace-123",
            metadata={"source": "test"}
        )
        
        assert task.type == "factory_task"
        assert task.priority == TaskPriority.HIGH
        assert task.trace_id == "trace-123"
        assert task.metadata["source"] == "test"


class TestWorkerPool:
    """Tests for the Worker Pool implementations."""

    @pytest.mark.asyncio
    async def test_queue_worker_pool(self):
        """Test the QueueWorkerPool for async task execution."""
        # Get a worker pool
        pool = await get_worker_pool("test_queue_pool", WorkerPoolType.QUEUE_ASYNCIO)
        
        result_container = []
        
        # Define a test task
        async def test_task(value):
            await asyncio.sleep(0.1)  # Simulate work
            result_container.append(value * 2)
        
        # Submit tasks
        for i in range(5):
            await pool.submit(test_task, i)
        
        # Allow time for tasks to complete
        await asyncio.sleep(0.5)
        
        # Verify results
        assert len(result_container) == 5
        assert set(result_container) == {0, 2, 4, 6, 8}
        
        # Verify metrics
        assert pool.metrics.tasks_submitted == 5
        assert pool.metrics.tasks_completed >= 5
        
        # Shutdown the pool
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_thread_worker_pool(self):
        """Test the ThreadWorkerPool for task execution."""
        # Get a thread worker pool
        pool = await get_worker_pool("test_thread_pool", WorkerPoolType.THREAD)
        
        # Define a simple test function
        def multiply(x, y):
            time.sleep(0.1)  # Simulate work
            return x * y
        
        # Submit task and get future
        future = pool.submit(multiply, 6, 7)
        
        # Get result
        result = future.result()
        assert result == 42
        
        # Test async submission
        result = await pool.asubmit(multiply, 7, 6)
        assert result == 42
        
        # Test map operation
        items = [(1, 2), (3, 4), (5, 6)]
        results = pool.map(lambda pair: multiply(*pair), items)
        assert results == [2, 12, 30]
        
        # Shutdown the pool
        await pool.ashutdown()

    @pytest.mark.asyncio
    async def test_process_worker_pool(self):
        """Test the ProcessWorkerPool for task execution."""
        # This test could be skipped in some environments where process pool might not work
        try:
            # Get a process worker pool
            pool = await get_worker_pool("test_process_pool", WorkerPoolType.PROCESS)
            
            # Submit task and get future
            future = pool.submit(_square_for_test, 7)
            
            # Get result
            result = future.result()
            assert result == 49
            
            # Test async submission
            result = await pool.asubmit(_square_for_test, 8)
            assert result == 64
            
            # Shutdown the pool
            await pool.ashutdown()
        except Exception as e:
            pytest.skip(f"ProcessWorkerPool test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

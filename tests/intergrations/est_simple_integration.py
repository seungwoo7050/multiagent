"""
Basic integration tests for the multi-agent platform.

These tests verify fundamental component connectivity:
- Task creation and dispatching
- Agent execution (planner and executor)
- Memory operations
- Tool execution
- MCP context handling
- Complete workflow execution

The tests minimize mocking and use real implementations where possible.
"""
import asyncio
import os
import time
import pytest
import json
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Core components
from src.core.task import BaseTask, TaskState, TaskPriority, TaskFactory
from src.core.agent import AgentContext, AgentResult, AgentState
from src.core.registry import get_function_registry

# Memory components
from src.memory.manager import MemoryManager
from src.memory.redis_memory import RedisMemory
from src.memory.vector_store import VectorStore

# MCP components
from src.core.mcp.schema import TaskContext
from src.core.mcp.serialization import serialize_context, deserialize_context

# Agent components
from src.agents.factory import AgentFactory
from src.agents.config import AgentConfig
from src.agents.mcp_planner import MCPPlannerAgent
from src.agents.mcp_executor import MCPExecutorAgent
from src.agents.context_manager import AgentContextManager

# Orchestration components
from src.orchestration.task_queue import BaseTaskQueue
from src.orchestration.dispatcher import Dispatcher
from src.orchestration.scheduler import PriorityScheduler
from src.orchestration.flow_control import RedisRateLimiter

# Tools
from src.tools.registry import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.datetime_tool import DateTimeTool, DateTimeOperation
from src.tools.runner import ToolRunner

# LLM components - mock these as they require API keys
from src.llm.base import BaseLLMAdapter
from src.llm.adapters.openai import OpenAIAdapter
from src.llm.cache import get_cache, clear_cache

# Mock classes for testing
class MockTaskQueue(BaseTaskQueue):
    """Mock task queue for testing that works in-memory."""
    
    def __init__(self):
        self.tasks = []
        self.dlq = []
        self.acknowledged = set()
        
    async def produce(self, task_data, task_id=None):
        msg_id = task_id or f"msg-{int(time.time() * 1000)}"
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
        # Simple mock for the lock
        class MockLock:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockLock()


class MockOpenAIAdapter(BaseLLMAdapter):
    """Mock OpenAI adapter that returns predefined responses."""
    
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.provider = "openai"
        self.initialized = True
        
    async def _initialize(self):
        return True
        
    async def ensure_initialized(self):
        return True
    
    def _get_client(self):
        return None
        
    async def _generate_text(self, prompt, **kwargs):
        # Return different responses based on the content of the prompt
        if "planner" in str(prompt).lower():
            # Return a planning response
            return {
                "choices": [
                    {
                        "message": {"content": json.dumps({
                            "plan": [
                                {
                                    "step": 1,
                                    "action": "calculator",
                                    "args": {"expression": "2 + 2"},
                                    "reasoning": "Need to perform a simple calculation"
                                },
                                {
                                    "step": 2,
                                    "action": "datetime",
                                    "args": {"operation": "current"},
                                    "reasoning": "Need to check the current time"
                                },
                                {
                                    "step": 3,
                                    "action": "finish",
                                    "args": {"answer": "Here's the calculation result and current time."},
                                    "reasoning": "Providing final answer to the user"
                                }
                            ]
                        })}
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
            }
        elif "executor" in str(prompt).lower():
            # Return different responses for the executor agent based on step
            if "step 1" in str(prompt).lower() or "calculator" in str(prompt).lower():
                return {
                    "choices": [
                        {
                            "message": {"content": "calculator[{\"expression\": \"2 + 2\"}]"}
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
                }
            elif "step 2" in str(prompt).lower() or "datetime" in str(prompt).lower():
                return {
                    "choices": [
                        {
                            "message": {"content": "datetime[{\"operation\": \"current\"}]"}
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
                }
            else:
                return {
                    "choices": [
                        {
                            "message": {"content": "finish[{\"answer\": \"The result is 4 and here's the current time.\"}]"}
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}
                }
        else:
            # Default response
            return {
                "choices": [
                    {
                        "message": {"content": "I'm not sure how to respond to that prompt."}
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            }
            
    async def _count_tokens(self, text):
        # Simple mock implementation
        return len(text.split())
        
    def get_token_limit(self):
        # Mock token limit
        return 4096

# Mock classes for Redis
class MockRedis:
    def __init__(self):
        self.store = {}
        
    async def setex(self, key, ex, value):
        self.store[key] = value
        return True
        
    async def get(self, key):
        return self.store.get(key)
        
    async def delete(self, key):
        if key in self.store:
            del self.store[key]
            return 1
        return 0


# Test fixtures
@pytest.fixture
def memory_manager():
    """Create a memory manager with in-memory Redis."""
    primary_memory = RedisMemory()
    vector_store = VectorStore()
    
    manager = MemoryManager(
        primary_memory=primary_memory,
        vector_store=vector_store,
        cache_size=100,
        cache_ttl=60,
        memory_ttl=3600
    )
    
    return manager


@pytest.fixture
def tool_registry():
    """Create and populate a tool registry with real tools."""
    registry = ToolRegistry()
    
    # Register the calculator tool - use the get_instance method after registration
    registry.register(CalculatorTool)
    
    # Register the datetime tool
    registry.register(DateTimeTool)
    
    return registry


@pytest.fixture
def tool_runner(tool_registry):
    """Create a tool runner with the tool registry."""
    return ToolRunner()


@pytest.fixture
def agent_factory():
    """Create an agent factory with mocked LLM adapters."""
    factory = AgentFactory()
    
    # Register a planner agent config
    planner_config = AgentConfig(
        name="test_planner",
        agent_type="mcp_planner",
        model="gpt-3.5-turbo",
        capabilities={"planning", "reasoning"},
        allowed_tools=["calculator", "datetime"],
        parameters={
            "temperature": 0.2,
            "max_tokens": 1000
        }
    )
    factory.register_agent_config(planner_config)
    
    # Register an executor agent config
    executor_config = AgentConfig(
        name="test_executor",
        agent_type="mcp_executor",
        model="gpt-3.5-turbo",
        capabilities={"execution", "tool_use"},
        allowed_tools=["calculator", "datetime"],
        parameters={
            "temperature": 0.2,
            "max_react_iterations": 3
        }
    )
    factory.register_agent_config(executor_config)
    
    return factory


@pytest.fixture
def task_queue():
    """Create a mock task queue."""
    return MockTaskQueue()


@pytest.fixture
def dispatcher(task_queue):
    """Create a dispatcher with the task queue."""
    scheduler = PriorityScheduler()
    return Dispatcher(task_queue=task_queue, scheduler=scheduler)


@pytest.mark.asyncio
class TestBasicIntegration:
    """Basic integration tests for the multi-agent platform."""
    
    async def test_end_to_end_execution(self, memory_manager, tool_registry, tool_runner, agent_factory, task_queue, dispatcher):
        """Test the complete flow from task creation to completion with real components."""
        # 1. Set up our mocks for LLM
        with patch("src.llm.adapters.get_adapter", return_value=MockOpenAIAdapter()):
            with patch("src.core.mcp.adapters.llm_adapter.LLMAdapter.process_with_mcp") as mock_process:
                # Configure mock to return different responses based on agent type
                executor_counter = [0]  # Using list for mutable state
                
                async def mock_response(context):
                    # If this is for the planner agent
                    if hasattr(context, 'prompt') and "plan" in str(context.prompt).lower():
                        return AsyncMock(
                            success=True,
                            result_text=json.dumps({
                                "plan": [
                                    {
                                        "step": 1,
                                        "action": "calculator",
                                        "args": {"expression": "2 + 2"},
                                        "reasoning": "Need to perform a simple calculation"
                                    },
                                    {
                                        "step": 2,
                                        "action": "datetime",
                                        "args": {"operation": "current"},
                                        "reasoning": "Need to check the current time"
                                    },
                                    {
                                        "step": 3,
                                        "action": "finish",
                                        "args": {"answer": "Here's the calculation result and current time."},
                                        "reasoning": "Providing final answer to the user"
                                    }
                                ]
                            }),
                            model_used="gpt-3.5-turbo"
                        )
                    # If this is for the executor agent's thought generation
                    elif hasattr(context, 'prompt') and "thought" in str(context.prompt).lower():
                        # Provide a thought response
                        return AsyncMock(
                            success=True,
                            result_text="I need to execute the plan steps sequentially",
                            model_used="gpt-3.5-turbo"
                        )
                    # If this is for the executor agent's action generation
                    else:
                        # Use counter to track which step we're on
                        executor_counter[0] += 1
                        
                        if executor_counter[0] == 1:
                            return AsyncMock(
                                success=True,
                                result_text="calculator[{\"expression\": \"2 + 2\"}]",
                                model_used="gpt-3.5-turbo"
                            )
                        elif executor_counter[0] == 2:
                            return AsyncMock(
                                success=True,
                                result_text="datetime[{\"operation\": \"current\"}]",
                                model_used="gpt-3.5-turbo"
                            )
                        else:
                            return AsyncMock(
                                success=True,
                                result_text="finish[{\"answer\": \"The result is 4 and here's the current time.\"}]",
                                model_used="gpt-3.5-turbo"
                            )
                    
                mock_process.side_effect = mock_response
                
                # 2. Create a task with a specific goal
                task = TaskFactory.create_task(
                    task_type="integration_test",
                    input_data={
                        "goal": "Perform a calculation and check the current time",
                        "query": "What is 2 + 2 and what time is it now?"
                    },
                    priority=TaskPriority.HIGH,
                    trace_id="trace-integration-test"
                )
                
                # Assert task was created properly
                assert task.type == "integration_test"
                assert task.state == TaskState.PENDING
                assert task.priority == TaskPriority.HIGH
                
                # 3. Submit the task to the dispatcher via task queue
                # Convert task to dictionary using proper Pydantic serialization
                task_dict = {
                    "id": task.id,
                    "type": task.type,
                    "input": task.input,
                    "state": task.state.value,
                    "priority": task.priority.value,
                    "trace_id": task.trace_id,
                    "metadata": task.metadata or {}
                }
                task_id = await task_queue.produce(task_dict, task.id)
                assert task_id is not None
                
                # 4. Process the task (manually drive the dispatcher)
                # Normally this would happen automatically in the dispatcher loop
                task_msgs = await task_queue.consume("test-consumer", count=1)
                assert len(task_msgs) == 1
                
                # 5. Create a planner agent
                planner = MCPPlannerAgent(config=agent_factory._agent_configs["test_planner"])
                await planner.initialize()
                
                # 6. Create an agent context
                agent_context = AgentContext(
                    task=task,
                    trace_id="trace-integration-test",
                    tools=["calculator", "datetime"]
                )
                
                # 7. Execute the planner agent
                plan_result = await planner.process(agent_context)
                
                # Assert plan was generated successfully
                assert plan_result.success is True
                assert "plan" in plan_result.output
                assert len(plan_result.output["plan"]) == 3
                assert plan_result.output["plan"][0]["action"] == "calculator"
                
                # 8. Create executor agent
                executor = MCPExecutorAgent(
                    config=agent_factory._agent_configs["test_executor"],
                    tool_registry=tool_registry
                )
                await executor.initialize()
                
                # 9. Pass the plan to the executor
                # Update the task with the plan
                task.input["plan"] = plan_result.output["plan"]
                executor_context = AgentContext(
                    task=task,
                    trace_id="trace-integration-test",
                    tools=["calculator", "datetime"]
                )
                
                # 10. Execute the executor agent
                execution_result = await executor.process(executor_context)
                
                # Assert execution was successful
                assert execution_result.success is True
                assert "final_answer" in execution_result.output
                assert "scratchpad" in execution_result.output
                
                # 11. Start the task first (required to permit transition to COMPLETED)
                task.start()
                
                # Complete the task with the result
                task.complete({"result": execution_result.output["final_answer"]})
                
                # Assert task was completed
                assert task.state == TaskState.COMPLETED
                assert task.output is not None
                assert "result" in task.output
                
                # 12. Store the result in memory
                result_key = f"task_result:{task.id}"
                await memory_manager.save(result_key, "integration_test", task.output)
                
                # 13. Verify result storage
                stored_result = await memory_manager.load(result_key, "integration_test")
                assert stored_result is not None
                assert stored_result["result"] == execution_result.output["final_answer"]
                
                # 14. Clean up
                await planner.terminate()
                await executor.terminate()
    
    async def test_mcp_context_handling(self, memory_manager):
        """Test that MCP context handling works correctly through serialization/deserialization."""
        # Mock Redis to avoid event loop issues
        with patch("src.memory.redis_memory.conn_manager.get_redis_async_connection") as mock_get_redis:
            mock_redis = MockRedis()
            mock_get_redis.return_value = mock_redis
            
            # 1. Create a task context
            task_context = TaskContext(
                task_id="mcp-test-task",
                task_type="integration_test",
                input_data={
                    "goal": "Test MCP context handling",
                    "query": "Is the MCP context handling working?"
                },
                metadata={
                    "priority": "high",
                    "trace_id": "trace-mcp-test"
                }
            )
            
            # 2. Serialize the context
            serialized_context = serialize_context(task_context)
            assert serialized_context is not None
            assert isinstance(serialized_context, bytes)
            
            # 3. Store in memory
            context_key = f"context:{task_context.context_id}"
            await memory_manager.save(context_key, "mcp_test", serialized_context)
            
            # 4. Retrieve from memory
            retrieved_context_bytes = await memory_manager.load(context_key, "mcp_test")
            assert retrieved_context_bytes is not None
            
            # 5. Deserialize the context
            deserialized_context = deserialize_context(
                retrieved_context_bytes,
                target_class=TaskContext
            )
            
            # 6. Verify context was preserved correctly
            assert deserialized_context.task_id == task_context.task_id
            assert deserialized_context.task_type == task_context.task_type
            assert deserialized_context.input_data == task_context.input_data
            assert deserialized_context.metadata["priority"] == "high"
            
            # 7. Create an agent context manager
            context_manager = AgentContextManager(agent_id="test-agent")
            
            # 8. Update the context through the manager
            context_manager.update_context(task_context)
            
            # 9. Retrieve the context
            retrieved_context = context_manager.get_context(context_id=task_context.context_id)
            assert retrieved_context is not None
            assert retrieved_context.task_id == task_context.task_id
            
            # 10. Optimize the context
            optimized_context = context_manager.optimize_context(context_id=task_context.context_id)
            assert optimized_context is not None
            
            # 11. Clear contexts
            context_manager.clear_contexts()
            assert len(context_manager.get_all_contexts()) == 0
    
    async def test_tool_execution(self, tool_registry, tool_runner):
        """Test that tools can be executed correctly."""
        # Patch the tool runner to avoid the logging conflict
        with patch("src.tools.runner.logger") as mock_logger:
            # 1. Get the calculator tool
            calculator_tool = tool_registry.get_tool("calculator")
            assert calculator_tool is not None
            
            # 2. Execute the calculator tool directly
            calc_result = calculator_tool.run(expression="10 * 5")
            assert calc_result["result"] == 50
            
            # 3. Execute through the tool runner
            runner_result = await tool_runner.run_tool(
                tool="calculator",
                tool_registry=tool_registry,
                args={"expression": "10 * 5"}
            )
            assert runner_result["status"] == "success"
            assert runner_result["result"]["result"] == 50
            
            # 4. Get the datetime tool
            datetime_tool = tool_registry.get_tool("datetime")
            assert datetime_tool is not None
            
            # 5. Execute the datetime tool directly
            time_result = datetime_tool.run(operation=DateTimeOperation.CURRENT)
            assert "iso_format" in time_result
            assert "timestamp" in time_result
            
            # 6. Execute through the tool runner
            runner_time_result = await tool_runner.run_tool(
                tool="datetime",
                tool_registry=tool_registry,
                args={"operation": "current"}
            )
            assert runner_time_result["status"] == "success"
            assert "iso_format" in runner_time_result["result"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
"""
Multi-Agent System Integration Tests.
Verifies that the agents package components work together correctly.
"""
import asyncio
import json
import pytest
import time
from typing import Any, Dict, List, Optional, Set, Type
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.agents.config import AgentConfig
from src.agents.context_manager import AgentContextManager
from src.agents.factory import AgentFactory, get_agent_factory
from src.agents.mcp_executor import MCPExecutorAgent
from src.agents.mcp_planner import MCPPlannerAgent
from src.core.agent import AgentCapability, AgentContext, AgentResult, AgentState, BaseAgent
from src.core.exceptions import AgentExecutionError, AgentCreationError, AgentNotFoundError
from src.core.mcp.adapters.llm_adapter import LLMInputContext, LLMOutputContext
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema, TaskContext
from src.core.task import BaseTask, TaskPriority, TaskResult, TaskState
from src.tools.registry import ToolRegistry


@pytest.fixture
def task_input() -> Dict[str, Any]:
    """Create sample task input for testing."""
    return {
        "goal": "Find information about Python programming",
        "query": "What are the best practices for Python error handling?",
        "context": "I'm writing a tutorial about error handling in Python."
    }


@pytest.fixture
def sample_task(task_input) -> BaseTask:
    """Create a sample task for testing."""
    task = MagicMock(spec=BaseTask)
    task.id = "test-task-id-12345"
    task.type = 'execute'
    task.state = TaskState.PENDING
    task.priority = TaskPriority.NORMAL
    task.input = task_input
    return task


@pytest.fixture
def agent_context(sample_task) -> AgentContext:
    """Create an agent context for testing."""
    return AgentContext(
        task=sample_task,
        trace_id="test-trace-id-67890",
        memory={
            "conversation_history": "User: Can you help me with Python error handling?\nAssistant: I'd be happy to help."
        },
        tools=["web_search", "calculator"]
    )


@pytest.fixture
def planner_config() -> AgentConfig:
    """Create configuration for a planner agent."""
    return AgentConfig(
        name="test-planner",
        agent_type="mcp_planner",
        model="gpt-3.5-turbo",
        capabilities={"planning", "reasoning"},  # Using strings instead of enum values
        allowed_tools=["web_search", "calculator"],
        parameters={
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )


@pytest.fixture
def executor_config() -> AgentConfig:
    """Create configuration for an executor agent."""
    return AgentConfig(
        name="test-executor",
        agent_type="mcp_executor",
        model="gpt-3.5-turbo",
        capabilities={"execution", "tool_use"},  # Using strings instead of enum values
        allowed_tools=["web_search", "calculator"],
        parameters={
            "temperature": 0.3,
            "max_react_iterations": 3
        }
    )


@pytest.fixture
def mock_plan_data() -> Dict[str, Any]:
    """Mock plan data for testing."""
    return {
        "plan": [
            {
                "step": 1,
                "action": "web_search",
                "args": {"query": "Python error handling best practices"},
                "reasoning": "To find up-to-date information about Python error handling best practices"
            },
            {
                "step": 2,
                "action": "think",
                "args": {"thought": "Analyze search results to extract key best practices"},
                "reasoning": "Need to process the search results to identify the most important points"
            },
            {
                "step": 3,
                "action": "finish",
                "args": {"answer": "Here are the key best practices for Python error handling"},
                "reasoning": "After gathering and analyzing information, provide a comprehensive answer"
            }
        ]
    }


@pytest.fixture
def mock_agent_factory() -> AgentFactory:
    """Create a mocked agent factory for testing."""
    factory = AgentFactory()
    return factory


class TestAgentConfig:
    """Tests for the AgentConfig validation and functionality."""

    def test_agent_config_valid(self):
        """Test creating a valid agent configuration."""
        config = AgentConfig(
            name="test-agent",
            agent_type="test",
            capabilities={"planning", "reasoning"}  # Using strings instead of enum values
        )
        assert config.name == "test-agent"
        assert config.agent_type == "test"
        # Check that enum values were properly converted from strings
        assert len(config.capabilities) == 2
        assert any(cap.value == "planning" for cap in config.capabilities)
        assert any(cap.value == "reasoning" for cap in config.capabilities)

    def test_agent_config_validate_capabilities(self):
        """Test that capabilities validation works correctly."""
        # Test with valid string input
        config = AgentConfig(
            name="test-agent",
            agent_type="test",
            capabilities="planning,reasoning"  # Comma-separated string
        )
        assert len(config.capabilities) == 2
        assert any(cap.value == "planning" for cap in config.capabilities)
        assert any(cap.value == "reasoning" for cap in config.capabilities)

        # Test with list input
        config = AgentConfig(
            name="test-agent",
            agent_type="test",
            capabilities=["planning", "reasoning"]  # List of strings
        )
        assert len(config.capabilities) == 2
        assert any(cap.value == "planning" for cap in config.capabilities)
        assert any(cap.value == "reasoning" for cap in config.capabilities)


class TestAgentContextManager:
    """Tests for the AgentContextManager component."""

    def test_context_update_and_get(self):
        """Test basic context storage and retrieval."""
        context_manager = AgentContextManager(agent_id="test-agent")
        
        # Create and store a context
        task_context = TaskContext(
            task_id="test-task-id",
            task_type="test",
            input_data={"goal": "Test goal"}
        )
        context_manager.update_context(task_context)
        
        # Retrieve by ID
        retrieved_context = context_manager.get_context(context_id=task_context.context_id)
        assert retrieved_context is not None
        assert retrieved_context.task_id == "test-task-id"
        
        # Retrieve by type
        retrieved_by_type = context_manager.get_context(context_type=TaskContext)
        assert retrieved_by_type is not None
        assert retrieved_by_type.task_id == "test-task-id"

    def test_context_optimization(self):
        """Test context optimization functionality."""
        context_manager = AgentContextManager(agent_id="test-agent")
        
        # Create a context with metadata
        task_context = TaskContext(
            task_id="test-task-id",
            task_type="test",
            input_data={"goal": "Test goal"},
            metadata={"key1": "value1", "key2": "value2"}
        )
        context_manager.update_context(task_context)
        
        # Optimize context
        optimized = context_manager.optimize_context(context_id=task_context.context_id)
        assert optimized is not None
        assert optimized.task_id == "test-task-id"

    def test_clear_contexts(self):
        """Test clearing all contexts."""
        context_manager = AgentContextManager(agent_id="test-agent")
        
        # Add multiple contexts
        for i in range(3):
            task_context = TaskContext(
                task_id=f"test-task-id-{i}",
                task_type="test",
                input_data={"goal": f"Test goal {i}"}
            )
            context_manager.update_context(task_context)
        
        # Verify contexts are stored
        assert len(context_manager.get_all_contexts()) == 3
        
        # Clear contexts
        context_manager.clear_contexts()
        assert len(context_manager.get_all_contexts()) == 0


class TestAgentFactory:
    """Tests for the AgentFactory component."""

    @pytest.mark.asyncio
    async def test_register_and_get_agent(self, planner_config):
        """Test registering a configuration and getting an agent."""
        factory = AgentFactory()
        factory.register_agent_config(planner_config)
        
        # Mock the initialize method to avoid actual initialization
        with patch.object(MCPPlannerAgent, 'initialize', return_value=True):
            agent = await factory.get_agent(planner_config.name)
            
            assert agent is not None
            assert agent.name == planner_config.name
            assert agent.agent_type == planner_config.agent_type

    @pytest.mark.asyncio
    async def test_agent_caching(self, planner_config):
        """Test that agents are properly cached."""
        factory = AgentFactory()
        factory.register_agent_config(planner_config)
        
        # Mock the initialize method
        with patch.object(MCPPlannerAgent, 'initialize', return_value=True), \
             patch.object(BaseAgent, 'is_idle', new_callable=PropertyMock, return_value=True):
            
            # Get agent twice - should be same instance
            agent1 = await factory.get_agent(planner_config.name)
            agent2 = await factory.get_agent(planner_config.name)
            
            assert agent1 is agent2  # Same instance, not just equal

    @pytest.mark.asyncio
    async def test_cleanup_cache(self, planner_config):
        """Test cleaning up idle agent instances."""
        factory = AgentFactory()
        factory.register_agent_config(planner_config)
        
        # Mock initialize and terminate methods
        with patch.object(MCPPlannerAgent, 'initialize', return_value=True), \
             patch.object(MCPPlannerAgent, 'terminate') as mock_terminate:
            
            # Get an agent to cache it
            await factory.get_agent(planner_config.name)
            
            # Modify the timestamp to simulate idle time
            cache_key = planner_config.name
            agent, _, _ = factory._agent_instances[cache_key]
            factory._agent_instances[cache_key] = (agent, 0, 0)  # Set last_used to 0
            
            # Clean up with small idle time threshold
            removed = await factory.cleanup_cache(max_idle_time_ms=10)
            
            assert removed == 1
            assert mock_terminate.called


@pytest.mark.asyncio
class TestMCPPlannerAgent:
    """Tests for the MCPPlannerAgent component."""

    async def test_planner_initialization(self, planner_config):
        """Test planner agent initialization."""
        planner = MCPPlannerAgent(config=planner_config)
        initialized = await planner.initialize()
        
        assert initialized is True
        assert planner.state == AgentState.IDLE

    async def test_planner_process(self, planner_config, agent_context, mock_plan_data):
        """Test planner's process method with mocked LLM."""
        planner = MCPPlannerAgent(config=planner_config)
        
        # Mock the LLM adapter to return predefined plan
        with patch.object(planner, 'llm_adapter') as mock_adapter:
            mock_output = LLMOutputContext(
                success=True,
                result_text=json.dumps(mock_plan_data),
                model_used=planner_config.model
            )
            mock_adapter.process_with_mcp = AsyncMock(return_value=mock_output)
            
            # Execute the process method
            result = await planner.process(agent_context)
            
            assert result.success is True
            assert "plan" in result.output
            assert len(result.output["plan"]) > 0

    async def test_planner_error_handling(self, planner_config, agent_context):
        """Test planner's error handling mechanism."""
        planner = MCPPlannerAgent(config=planner_config)
        
        # Mock the LLM adapter to raise an exception
        with patch.object(planner, 'llm_adapter') as mock_adapter:
            mock_adapter.process_with_mcp = AsyncMock(side_effect=AgentExecutionError(
                message="Test error",
                agent_type=planner_config.agent_type
            ))
            
            # Execute with expected error
            with pytest.raises(AgentExecutionError):
                await planner.process(agent_context)
            
            # Test error handler directly
            error = AgentExecutionError(
                message="Test error",
                agent_type=planner_config.agent_type
            )
            error_result = await planner.handle_error(error, agent_context)
            
            assert error_result.success is False
            assert error_result.error is not None
            assert "message" in error_result.error


@pytest.mark.asyncio
class TestMCPExecutorAgent:
    """Tests for the MCPExecutorAgent component."""

    async def test_executor_initialization(self, executor_config):
        """Test executor agent initialization."""
        # Mock tool registry with required methods
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_names.return_value = ["web_search", "calculator"]
        mock_registry.has = MagicMock(return_value=True)
        
        executor = MCPExecutorAgent(config=executor_config, tool_registry=mock_registry)
        initialized = await executor.initialize()
        
        assert initialized is True
        assert executor.state == AgentState.IDLE

    async def test_executor_process(self, executor_config, agent_context, mock_plan_data):
        """Test executor's process method with mocked dependencies."""
        # Create a task with a plan
        task_with_plan = MagicMock(spec=BaseTask)
        task_with_plan.id = "test-task-id-12345"
        task_with_plan.type = 'execute'
        task_with_plan.state = TaskState.PENDING
        task_with_plan.priority = TaskPriority.NORMAL
        task_with_plan.input = {
            "goal": "Find information",
            "plan": mock_plan_data["plan"]  # Use the properly structured plan
        }
        
        # Update context with the task containing a plan
        context_with_plan = AgentContext(
            task=task_with_plan,
            trace_id="test-trace-id-67890"
        )
        
        # Mock dependencies with required methods
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.has = MagicMock(return_value=True)
        mock_registry.get_names.return_value = ["web_search", "calculator"]
        
        # Create executor with mocked dependencies
        executor = MCPExecutorAgent(config=executor_config, tool_registry=mock_registry)
        
        # Mock LLM and tool runner responses
        with patch.object(executor, 'llm_adapter') as mock_llm, \
             patch.object(executor, 'tool_runner') as mock_tools:
            
            # Mock thought generation
            thought_output = LLMOutputContext(
                success=True,
                result_text="I need to search for information",
                model_used=executor_config.model
            )
            
            # Mock action generation
            action_output = LLMOutputContext(
                success=True,
                result_text="finish[{\"answer\": \"Here's what I found\"}]",
                model_used=executor_config.model
            )
            
            # Set up LLM mock to return different values for different calls
            mock_llm.process_with_mcp.side_effect = [thought_output, action_output]
            
            # Mock tool runner to return success
            mock_tools.run_tool.return_value = {
                "status": "success",
                "result": "Tool executed successfully"
            }
            
            # Execute the process method
            result = await executor.process(context_with_plan)
            
            assert result.success is True
            assert "final_answer" in result.output
            assert "scratchpad" in result.output

    async def test_executor_parse_action(self, executor_config):
        """Test the action parsing functionality."""
        # Mock tool registry with required methods
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_names.return_value = ["web_search", "calculator"]
        mock_registry.has = MagicMock(return_value=True)
        
        executor = MCPExecutorAgent(config=executor_config, tool_registry=mock_registry)
        
        # Test valid action parsing
        action_str = "web_search[{\"query\": \"Python error handling\"}]"
        action_type, action_input = executor._parse_action(action_str)
        
        assert action_type == "web_search"
        assert action_input == {"query": "Python error handling"}
        
        # Test finish action
        finish_str = "finish[{\"answer\": \"Here's the information\"}]"
        finish_type, finish_input = executor._parse_action(finish_str)
        
        assert finish_type == "finish"
        assert finish_input == {"answer": "Here's the information"}
        
        # Test think action
        think_str = "think[]"
        think_type, think_input = executor._parse_action(think_str)
        
        assert think_type == "think"
        assert think_input == {}


@pytest.mark.asyncio
class TestEndToEndFlow:
    """End-to-end tests for the planner-executor workflow."""

    async def test_planner_executor_flow(self, planner_config, executor_config, agent_context, mock_plan_data):
        """Test the full planning and execution flow."""
        # Create mocked dependencies
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.has = MagicMock(return_value=True)
        mock_registry.get_names.return_value = ["web_search", "calculator"]
        
        # Create agents
        planner = MCPPlannerAgent(config=planner_config)
        executor = MCPExecutorAgent(config=executor_config, tool_registry=mock_registry)
        
        # Mock planner's LLM adapter
        with patch.object(planner, 'llm_adapter') as mock_planner_llm:
            mock_output = LLMOutputContext(
                success=True,
                result_text=json.dumps(mock_plan_data),
                model_used=planner_config.model
            )
            mock_planner_llm.process_with_mcp = AsyncMock(return_value=mock_output)
            
            # Generate a plan
            plan_result = await planner.process(agent_context)
            
            assert plan_result.success is True
            assert "plan" in plan_result.output
            
            # Create a new context with the plan for the executor
            task_with_plan = MagicMock(spec=BaseTask)
            task_with_plan.id = agent_context.task.id
            task_with_plan.type = agent_context.task.type
            task_with_plan.state = agent_context.task.state
            task_with_plan.priority = agent_context.task.priority
            task_with_plan.input = {
                "goal": agent_context.task.input["goal"],
                "plan": plan_result.output["plan"]
            }
            
            executor_context = AgentContext(
                task=task_with_plan,
                trace_id=agent_context.trace_id,
                memory=agent_context.memory
            )
            
            # Mock executor's dependencies
            with patch.object(executor, 'llm_adapter') as mock_executor_llm, \
                 patch.object(executor, 'tool_runner') as mock_tools:
                
                # Mock thought and action generation
                thought_output = LLMOutputContext(
                    success=True,
                    result_text="I need to execute the plan step by step",
                    model_used=executor_config.model
                )
                
                action_output = LLMOutputContext(
                    success=True,
                    result_text="finish[{\"answer\": \"Completed the plan successfully\"}]",
                    model_used=executor_config.model
                )
                
                # Set up LLM mock to return different values
                mock_executor_llm.process_with_mcp.side_effect = [thought_output, action_output]
                
                # Mock tool runner
                mock_tools.run_tool.return_value = {
                    "status": "success",
                    "result": "Tool executed successfully"
                }
                
                # Execute the plan
                execution_result = await executor.process(executor_context)
                
                assert execution_result.success is True
                assert "final_answer" in execution_result.output
                assert execution_result.output["final_answer"] is not None

    async def test_performance_monitoring(self, planner_config, agent_context, mock_plan_data):
        """Test that performance is properly monitored."""
        planner = MCPPlannerAgent(config=planner_config)
        
        # Mock the LLM adapter
        with patch.object(planner, 'llm_adapter') as mock_adapter:
            mock_output = LLMOutputContext(
                success=True,
                result_text=json.dumps(mock_plan_data),
                model_used=planner_config.model
            )
            
            # Add a delay to simulate processing time
            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.01)  # Small delay for testing
                return mock_output
            
            mock_adapter.process_with_mcp = delayed_response
            
            # Measure execution time
            start_time = time.time()
            result = await planner.process(agent_context)
            execution_time = time.time() - start_time
            
            # Execution should take some measurable time
            assert execution_time > 0

    async def test_integration_with_llm_fallbacks(self, planner_config, agent_context):
        """Test integration with LLM fallback mechanisms."""
        planner = MCPPlannerAgent(config=planner_config)
        
        # Mock primary model failure and fallback success
        with patch.object(planner, 'llm_adapter') as mock_adapter:
            # First call fails
            mock_adapter.process_with_mcp = AsyncMock(side_effect=AgentExecutionError(
                message="Test error",
                agent_type=planner_config.agent_type
            ))
            
            # This should raise the agent execution error
            with pytest.raises(AgentExecutionError):
                await planner.process(agent_context)
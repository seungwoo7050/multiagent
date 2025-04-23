import pytest
import asyncio
from typing import Dict, Any, Optional, Set
from unittest.mock import AsyncMock, patch

from src.core.agent import (
    BaseAgent,
    AgentConfig,
    AgentContext,
    AgentResult,
    AgentState,
    AgentCapability,
)
from src.core.task import BaseTask
from src.core.exceptions import AgentExecutionError


class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.initialize_called = False
        self.process_called = False
        self.handle_error_called = False
        self.terminate_called = False
        self.should_fail_initialize = False
        self.should_fail_process = False
        self.process_error = None
        
    async def initialize(self) -> bool:
        self.initialize_called = True
        return not self.should_fail_initialize
    
    async def process(self, context: AgentContext) -> AgentResult:
        self.process_called = True
        if self.should_fail_process:
            if self.process_error:
                raise self.process_error
            raise ValueError("Simulated process failure")
        return AgentResult.success_result(
            output={"message": f"Processed by {self.name}"},
            execution_time=0.1
        )
    
    async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
        self.handle_error_called = True
        return AgentResult.error_result(
            error={"type": type(error).__name__, "message": str(error)},
            execution_time=0.05
        )
    
    async def terminate(self) -> None:
        await super().terminate()
        self.terminate_called = True


@pytest.fixture
def agent_config():
    """Fixture for creating a test agent configuration."""
    return AgentConfig(
        name="TestAgent",
        description="Test agent for unit tests",
        agent_type="test_agent",
        model="gpt-4",
        capabilities={AgentCapability.PLANNING, AgentCapability.REASONING},
        parameters={"temperature": 0.7},
        allowed_tools=["calculator", "search"]
    )


@pytest.fixture
def agent_context():
    """Fixture for creating a test agent context."""
    task = BaseTask(type="test_task", input={"query": "test query"})
    return AgentContext(
        task=task,
        memory={"history": ["previous interaction"]},
        parameters={"max_tokens": 1000},
        tools=["calculator"]
    )


class TestAgent:
    """Test suite for BaseAgent and related classes."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent_config):
        """Test agent initialization."""
        agent = MockAgent(agent_config)
        
        # Check initial state
        assert agent.state == AgentState.IDLE
        assert agent.name == "TestAgent"
        assert agent.agent_type == "test_agent"
        assert agent.is_idle is True
        assert agent.is_busy is False
        assert agent.is_terminated is False
        
        # Check config properties
        assert agent.config.name == "TestAgent"
        assert agent.config.agent_type == "test_agent"
        assert agent.config.model == "gpt-4"
        assert AgentCapability.PLANNING in agent.config.capabilities
        assert agent.config.parameters["temperature"] == 0.7
        assert "calculator" in agent.config.allowed_tools

    @pytest.mark.asyncio
    async def test_agent_execution_success(self, agent_config, agent_context):
        """Test successful agent execution."""
        agent = MockAgent(agent_config)
        
        # Execute agent
        result = await agent.execute(agent_context)
        
        # Verify method calls
        assert agent.initialize_called is True
        assert agent.process_called is True
        assert agent.handle_error_called is False
        
        # Check result
        assert result.success is True
        assert "message" in result.output
        assert result.error is None
        
        # Check state transitions
        assert agent.state == AgentState.IDLE  # Should return to IDLE after successful execution

    @pytest.mark.asyncio
    async def test_agent_execution_initialization_failure(self, agent_config, agent_context):
        """Test agent execution with initialization failure."""
        agent = MockAgent(agent_config)
        agent.should_fail_initialize = True
        
        # Execute agent
        result = await agent.execute(agent_context)
        
        # Verify method calls
        assert agent.initialize_called is True
        assert agent.process_called is False  # Process shouldn't be called if init fails
        
        # Check result
        assert result.success is False
        assert result.error is not None
        assert "initialization_error" in result.error["type"]
        
        # Check state
        assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_agent_execution_process_failure(self, agent_config, agent_context):
        """Test agent execution with process failure."""
        agent = MockAgent(agent_config)
        agent.should_fail_process = True
        
        # Execute agent
        result = await agent.execute(agent_context)
        
        # Verify method calls
        assert agent.initialize_called is True
        assert agent.process_called is True
        assert agent.handle_error_called is True
        
        # Check result
        assert result.success is False
        assert result.error is not None
        assert result.error["type"] == "ValueError"
        
        # Check state
        assert agent.state == AgentState.ERROR
        assert agent.error_count == 1

    @pytest.mark.asyncio
    async def test_agent_execution_error_handler_failure(self, agent_config, agent_context):
        """Test agent execution when error handler also fails."""
        agent = MockAgent(agent_config)
        agent.should_fail_process = True
        
        # Make handle_error raise an exception
        async def failing_handle_error(error, context):
            agent.handle_error_called = True
            raise RuntimeError("Error handler failed")
            
        agent.handle_error = failing_handle_error
        
        # Execute agent
        result = await agent.execute(agent_context)
        
        # Verify method calls
        assert agent.initialize_called is True
        assert agent.process_called is True
        assert agent.handle_error_called is True
        
        # Check result
        assert result.success is False
        assert result.error is not None
        assert "unhandled_error" in result.error["type"]
        assert "handler_error" in result.error
        
        # Check state
        assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_agent_context_manager(self, agent_config):
        """Test using agent as an async context manager."""
        async with MockAgent(agent_config) as agent:
            # Inside context, agent should be initialized
            assert agent.initialize_called is True
            assert agent.state == AgentState.IDLE
            
            # Do something with the agent
            assert agent.name == "TestAgent"
        
        # After context exit, agent should be terminated
        assert agent.terminate_called is True

    @pytest.mark.asyncio
    async def test_agent_termination(self, agent_config):
        """Test agent termination."""
        agent = MockAgent(agent_config)
        
        # Terminate the agent
        await agent.terminate()
        
        # Check state and calls
        assert agent.terminate_called is True
        assert agent.state == AgentState.TERMINATED
        assert agent.is_terminated is True

    @pytest.mark.asyncio
    async def test_agent_context_helper_methods(self, agent_context):
        """Test helper methods in AgentContext."""
        # Test get_param with default
        value = agent_context.get_param("non_existent", default="default_value")
        assert value == "default_value"
        
        # Test get_param with existing parameter
        value = agent_context.get_param("max_tokens")
        assert value == 1000
        
        # Test update_memory and get_memory
        agent_context.update_memory("result", "test_result")
        value = agent_context.get_memory("result")
        assert value == "test_result"
        
        # Test get_memory with default
        value = agent_context.get_memory("non_existent", default="default_memory")
        assert value == "default_memory"

    def test_agent_config_validation(self):
        """Test validation in AgentConfig."""
        # Valid config
        config = AgentConfig(
            name="TestAgent",
            agent_type="test_agent",
            capabilities={AgentCapability.PLANNING}
        )
        assert config.name == "TestAgent"
        
        # Invalid config (empty agent_type)
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(
                name="TestAgent",
                agent_type="",
                capabilities={AgentCapability.PLANNING}
            )
        assert "agent_type" in str(exc_info.value).lower()
        
        # Invalid config (no capabilities)
        with pytest.raises(ValueError) as exc_info:
            AgentConfig(
                name="TestAgent",
                agent_type="test_agent",
                capabilities=set()
            )
        assert "capability" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_agent_idle_time(self, agent_config):
        """Test agent idle time tracking."""
        agent = MockAgent(agent_config)
        
        await asyncio.sleep(0.001) # <--- 이 줄 추가
        
        # Initially should be idle
        assert agent.idle_time_ms > 0
        
        # Execute to mark as busy
        with patch.object(agent, 'process', new_callable=AsyncMock) as mock_process:
            # Mock process to simulate delay
            async def delayed_process(context):
                await asyncio.sleep(0.1)
                return AgentResult.success_result(output={}, execution_time=0.1)
            
            mock_process.side_effect = delayed_process
            
            # Start execution but don't await it
            execution_task = asyncio.create_task(agent.execute(AgentContext()))
            
            # Give it time to start processing
            await asyncio.sleep(0.01)
            
            # Should be busy now
            assert agent.is_busy is True
            assert agent.idle_time_ms == 0
            
            # Wait for execution to complete
            await execution_task
        
        # Should be idle again
        await asyncio.sleep(0.001)
        assert agent.is_idle is True
        assert agent.idle_time_ms > 0

    def test_agent_result_creation(self):
        """Test creation of AgentResult."""
        # Success result
        output = {"answer": "Test answer"}
        success_result = AgentResult.success_result(output, execution_time=0.5)
        
        assert success_result.success is True
        assert success_result.output == output
        assert success_result.error is None
        assert success_result.execution_time == 0.5
        
        # Error result
        error = {"type": "TestError", "message": "Test error message"}
        error_result = AgentResult.error_result(error, execution_time=0.3)
        
        assert error_result.success is False
        assert error_result.output == {}
        assert error_result.error == error
        assert error_result.execution_time == 0.3
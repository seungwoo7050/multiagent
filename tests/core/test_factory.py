import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, patch

from src.core.factory import AgentFactory, get_agent_factory
from src.core.agent import BaseAgent, AgentConfig, AgentContext, AgentResult, AgentCapability
from src.core.exceptions import AgentExecutionError, AgentNotFoundError


# Test agent classes
class TestAgent(BaseAgent):
    """Test agent implementation."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.initialize_calls = 0
        self.terminate_calls = 0
        self.should_fail_initialize = False
    
    async def initialize(self) -> bool:
        self.initialize_calls += 1
        if self.should_fail_initialize:
            return False
        return True
    
    async def process(self, context: AgentContext) -> AgentResult:
        return AgentResult.success_result(
            output={"message": f"Processed by {self.name}"},
            execution_time=0.1
        )
    
    async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
        return AgentResult.error_result(
            error={"message": str(error)},
            execution_time=0.1
        )
    
    async def terminate(self) -> None:
        await super().terminate()
        self.terminate_calls += 1


class AnotherTestAgent(BaseAgent):
    """Another test agent implementation."""
    
    async def initialize(self) -> bool:
        return True
    
    async def process(self, context: AgentContext) -> AgentResult:
        return AgentResult.success_result(
            output={"message": "Processed by AnotherTestAgent"},
            execution_time=0.1
        )
    
    async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
        return AgentResult.error_result(
            error={"message": str(error)},
            execution_time=0.1
        )


@pytest.fixture
def test_config():
    """Fixture for creating a test agent configuration."""
    return AgentConfig(
        name="TestAgent",
        description="Test agent for unit tests",
        agent_type="test_agent",
        capabilities={AgentCapability.PLANNING},
        parameters={"temperature": 0.7}
    )


@pytest.fixture
def another_test_config():
    """Fixture for creating another test agent configuration."""
    return AgentConfig(
        name="AnotherAgent",
        description="Another test agent",
        agent_type="another_agent",
        capabilities={AgentCapability.EXECUTION},
        parameters={"temperature": 0.5}
    )


class TestAgentFactory:
    """Test suite for AgentFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a new factory for each test
        self.factory = AgentFactory()
        
        # Register agent classes
        self.factory.register_agent_class("test_agent", TestAgent)
        self.factory.register_agent_class("another_agent", AnotherTestAgent)

    @pytest.mark.asyncio
    async def test_register_and_get_agent(self, test_config):
        """Test registering agent classes and configs, and getting agents."""
        # Register default config
        self.factory.register_agent_config(test_config)
        
        # Get agent using default config
        agent = await self.factory.get_agent("test_agent")
        
        # Verify agent
        assert isinstance(agent, TestAgent)
        assert agent.name == "TestAgent"
        assert agent.config.description == "Test agent for unit tests"
        
        # Get agent with custom config
        custom_config = AgentConfig(
            name="CustomAgent",
            description="Custom config",
            agent_type="test_agent",
            capabilities={AgentCapability.REASONING}
        )
        
        custom_agent = await self.factory.get_agent("test_agent", config=custom_config)
        
        # Verify custom agent
        assert isinstance(custom_agent, TestAgent)
        assert custom_agent.name == "CustomAgent"
        assert custom_agent.config.description == "Custom config"

    @pytest.mark.asyncio
    async def test_agent_caching(self, test_config):
        """Test agent instance caching."""
        # Register default config
        self.factory.register_agent_config(test_config)
        
        # Get agent first time
        agent1 = await self.factory.get_agent("test_agent")
        
        # Get agent second time
        agent2 = await self.factory.get_agent("test_agent")
        
        # Should be the same instance
        assert agent1 is agent2
        
        # Initialize should only be called once
        assert agent1.initialize_calls == 1
        
        # Get agent with cache disabled
        agent3 = await self.factory.get_agent("test_agent", use_cache=False)
        
        # Should be a different instance
        assert agent3 is not agent1
        
        # Initialize should be called for the new instance
        assert agent3.initialize_calls == 1
        
        # Create agent (always creates new instance)
        agent4 = await self.factory.create_agent("test_agent")
        
        # Should be a different instance
        assert agent4 is not agent1
        assert agent4 is not agent3

    @pytest.mark.asyncio
    async def test_agent_cache_key(self, test_config):
        """Test using custom cache keys."""
        # Register default config
        self.factory.register_agent_config(test_config)
        
        # Get agent with custom cache key
        agent1 = await self.factory.get_agent("test_agent", cache_key="custom_key")
        
        # Get agent with same cache key
        agent2 = await self.factory.get_agent("test_agent", cache_key="custom_key")
        
        # Should be the same instance
        assert agent1 is agent2
        
        # Get agent with different cache key
        agent3 = await self.factory.get_agent("test_agent", cache_key="another_key")
        
        # Should be a different instance
        assert agent3 is not agent1
        
        # Get agent with default cache key (agent_type)
        agent4 = await self.factory.get_agent("test_agent")
        
        # Should be a different instance from custom key agents
        assert agent4 is not agent1
        assert agent4 is not agent3

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in factory."""
        # Try to get non-existent agent type
        with pytest.raises(ValueError) as exc_info:
            await self.factory.get_agent("non_existent_agent")
        
        assert "not registered" in str(exc_info.value).lower()
        
        # Register agent class without config
        with pytest.raises(ValueError) as exc_info:
            await self.factory.get_agent("test_agent")  # No config registered or provided
        
        assert "no configuration" in str(exc_info.value).lower()
        
        # Test initialization failure
        config = AgentConfig(
            name="FailingAgent",
            agent_type="test_agent",
            capabilities={AgentCapability.PLANNING}
        )
        
        # Make agent initialization fail
        with patch.object(TestAgent, 'initialize', return_value=False):
            with pytest.raises(AgentExecutionError) as exc_info:
                await self.factory.get_agent("test_agent", config=config)
            
        assert "failed to initialize" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cleanup_cache(self, test_config, another_test_config):
        """Test cache cleanup."""
        # Register configs
        self.factory.register_agent_config(test_config)
        self.factory.register_agent_config(another_test_config)
        
        # Get agents
        agent1 = await self.factory.get_agent("test_agent")
        agent2 = await self.factory.get_agent("another_agent")
        
        # Both should be cached
        assert len(self.factory._agent_instances) == 2
        
        # Wait for a short time (less than cleanup threshold)
        await asyncio.sleep(0.01)
        
        # Cleanup with large idle time threshold (nothing should be removed)
        removed = await self.factory.cleanup_cache(max_idle_time_ms=1000)
        assert removed == 0
        assert len(self.factory._agent_instances) == 2
        
        # Cleanup with small idle time threshold (all should be removed)
        removed = await self.factory.cleanup_cache(max_idle_time_ms=0)
        assert removed == 2
        assert len(self.factory._agent_instances) == 0
        
        # Agents should have been terminated
        assert agent1.terminate_calls == 1
        assert isinstance(agent2, AnotherTestAgent)  # Can't check terminate_calls directly

    @pytest.mark.asyncio
    async def test_shutdown(self, test_config, another_test_config):
        """Test factory shutdown."""
        # Register configs
        self.factory.register_agent_config(test_config)
        self.factory.register_agent_config(another_test_config)
        
        # Get agents
        agent1 = await self.factory.get_agent("test_agent")
        agent2 = await self.factory.get_agent("another_agent")
        
        # Both should be cached
        assert len(self.factory._agent_instances) == 2
        
        # Shutdown factory
        await self.factory.shutdown()
        
        # Cache should be empty
        assert len(self.factory._agent_instances) == 0
        
        # Agents should have been terminated
        assert agent1.terminate_calls == 1

    @pytest.mark.asyncio
    async def test_singleton_factory(self, test_config):
        """Test the singleton factory."""
        # Get singleton factory
        factory1 = get_agent_factory()
        factory2 = get_agent_factory()
        
        # Should be the same instance
        assert factory1 is factory2
        
        # Register agent class and config
        factory1.register_agent_class("test_agent", TestAgent)
        factory1.register_agent_config(test_config)
        
        # Get agent from second factory instance
        agent = await factory2.get_agent("test_agent")
        
        # Should work
        assert isinstance(agent, TestAgent)
        assert agent.name == "TestAgent"
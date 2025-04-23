import asyncio
from typing import Any, Dict, Optional, Type, Union, cast

from src.config.logger import get_logger
from src.config.metrics import track_agent_created
from src.core.agent import AgentConfig, BaseAgent
from src.core.exceptions import AgentExecutionError
from src.utils.timing import async_timed, get_current_time_ms

# Module logger
logger = get_logger(__name__)


class AgentFactory:
    """Factory for creating and managing agent instances with fast O(1) lookup and caching."""
    
    def __init__(self):
        """Initialize the agent factory."""
        # Agent class registry: maps agent_type -> agent_class
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        
        # Agent instance cache: maps cache_key -> (agent_instance, last_used_time)
        self._agent_instances: Dict[str, tuple[BaseAgent, int]] = {}
        
        # Configuration: maps agent_type -> default_config
        self._agent_configs: Dict[str, AgentConfig] = {}
        
        logger.info("AgentFactory initialized")
    
    def register_agent_class(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class for a specific agent type.
        
        Args:
            agent_type: The type identifier for the agent.
            agent_class: The agent class to register.
        """
        if agent_type in self._agent_classes:
            logger.warning(f"Overriding existing agent class for type: {agent_type}")
            
        self._agent_classes[agent_type] = agent_class
        logger.info(f"Registered agent class for type: {agent_type}")
    
    def register_agent_config(self, config: AgentConfig) -> None:
        """Register default configuration for an agent type.
        
        Args:
            config: The agent configuration to register.
        """
        self._agent_configs[config.agent_type] = config
        logger.info(
            f"Registered default config for agent type: {config.agent_type}",
            extra={"agent_name": config.name}
        )
    
    def has_agent_type(self, agent_type: str) -> bool:
        """Check if an agent type is registered.
        
        Args:
            agent_type: The agent type to check.
            
        Returns:
            bool: True if the agent type is registered, False otherwise.
        """
        return agent_type in self._agent_classes
    
    @async_timed("agent_factory_get_agent")
    async def get_agent(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> BaseAgent:
        """Get or create an agent instance.
        
        Args:
            agent_type: The type of agent to get.
            config: Optional configuration for the agent. If not provided, uses registered default.
            use_cache: Whether to use cached instances.
            cache_key: Optional specific cache key. If not provided, uses agent_type.
            
        Returns:
            BaseAgent: An initialized agent instance.
            
        Raises:
            ValueError: If the agent type is not registered.
            AgentExecutionError: If agent creation or initialization fails.
        """
        # Check if agent type is registered
        if agent_type not in self._agent_classes:
            error_msg = f"Agent type not registered: {agent_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Determine cache key if caching is enabled
        effective_cache_key = None
        if use_cache:
            if config is not None and cache_key is None:
                use_cache = False
            else:   
                effective_cache_key = cache_key if cache_key else agent_type
            
            # Check cache for existing instance
            if effective_cache_key in self._agent_instances:
                agent, _ = self._agent_instances[effective_cache_key]
                
                # Update last used time
                self._agent_instances[effective_cache_key] = (agent, get_current_time_ms())
                
                logger.debug(
                    f"Using cached agent instance for type: {agent_type}",
                    extra={"cache_key": effective_cache_key}
                )
                return agent
        
        # Get agent class and config
        agent_class = self._agent_classes[agent_type]
        agent_config = config
        
        # If no config provided, use default if available
        if agent_config is None:
            if agent_type in self._agent_configs:
                agent_config = self._agent_configs[agent_type]
            else:
                error_msg = f"No configuration provided or registered for agent type: {agent_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Create agent instance
        try:
            try:
                agent = agent_class(agent_config)
                
                # Initialize the agent
                if not await agent.initialize():
                    error_msg = f"Failed to initialize agent of type: {agent_type}"
                    logger.error(error_msg)
                    raise AgentExecutionError(
                        message=error_msg,
                        agent_type=agent_type,
                        agent_id=agent_config.name
                    )
                
                # Cache the instance if caching is enabled
                if use_cache and effective_cache_key:
                    self._agent_instances[effective_cache_key] = (agent, get_current_time_ms())
                    logger.debug(
                        f"Cached new agent instance for type: {agent_type}",
                        extra={"cache_key": effective_cache_key}
                    )
                
                return agent
            except AgentExecutionError:
                raise
        except Exception as e:
            if isinstance(e, AgentExecutionError):
                raise 
            error_msg = f"Failed to create agent of type: {agent_type}"
            logger.exception(error_msg, extra={"error": str(e)})
            raise AgentExecutionError(
                message=error_msg,
                agent_type=agent_type,
                agent_id=agent_config.name,
                details={"error": str(e)}
            )
    
    async def create_agent(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None
    ) -> BaseAgent:
        """Create a new agent instance without caching.
        
        This is a convenience method that calls get_agent with use_cache=False.
        
        Args:
            agent_type: The type of agent to create.
            config: Optional configuration for the agent.
            
        Returns:
            BaseAgent: A newly created agent instance.
        """
        return await self.get_agent(agent_type, config, use_cache=False)
    
    async def cleanup_cache(self, max_idle_time_ms: int = 600000) -> int:
        """Clean up cached agent instances that haven't been used recently.
        
        Args:
            max_idle_time_ms: Maximum idle time in milliseconds before an agent is removed.
                Default is 10 minutes (600000 ms).
                
        Returns:
            int: Number of agents removed from cache.
        """
        current_time = get_current_time_ms()
        keys_to_remove = []
        
        # Find keys to remove
        for key, (agent, last_used_time) in self._agent_instances.items():
            if current_time - last_used_time > max_idle_time_ms:
                keys_to_remove.append(key)
        
        # Remove and terminate agents
        for key in keys_to_remove:
            agent, _ = self._agent_instances.pop(key)
            try:
                await agent.terminate()
                logger.debug(f"Terminated and removed cached agent: {key}")
            except Exception as e:
                logger.error(f"Error terminating cached agent {key}: {str(e)}")
        
        logger.info(f"Cleaned up {len(keys_to_remove)} cached agent instances")
        return len(keys_to_remove)
    
    async def shutdown(self) -> None:
        """Shutdown the factory and terminate all cached agents."""
        # Terminate all cached agents
        for key, (agent, _) in self._agent_instances.items():
            try:
                await agent.terminate()
                logger.debug(f"Terminated cached agent: {key}")
            except Exception as e:
                logger.error(f"Error terminating cached agent {key}: {str(e)}")
        
        # Clear caches
        self._agent_instances.clear()
        
        logger.info("AgentFactory shut down successfully")


# Singleton instance for application-wide use
_instance: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get the singleton AgentFactory instance."""
    global _instance
    if _instance is None:
        _instance = AgentFactory()
    return _instance
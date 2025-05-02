import asyncio
from typing import Any, Dict, List, Optional, Type

from src.agents.config import AgentConfig
from src.agents.mcp_executor import MCPExecutorAgent
from src.agents.mcp_planner import MCPPlannerAgent
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.core.agent import BaseAgent
from src.core.exceptions import (AgentCreationError, AgentExecutionError,
                                 AgentNotFoundError)
from src.utils.timing import async_timed, get_current_time_ms

logger = get_logger(__name__)
metrics = get_metrics_manager()

class AgentFactory:
    """
    Factory for creating, caching, and managing agent instances.
    Implements singleton pattern for global access.
    """

    def __init__(self):
        # Initialize with default agent classes
        self._agent_classes: Dict[str, Type[BaseAgent]] = {
            'mcp_planner': MCPPlannerAgent,
            'mcp_executor': MCPExecutorAgent,
            'planner': MCPPlannerAgent,  # Aliases for backward compatibility
            'executor': MCPExecutorAgent
        }

        # Cache for agent instances: (agent, last_used_ms, is_initialized_bool)
        # <<< 수정된 부분: 캐시 구조 타입 힌트 >>>
        self._agent_instances: Dict[str, tuple[BaseAgent, int, bool]] = {}

        # Store agent configurations
        self._agent_configs: Dict[str, AgentConfig] = {}

        logger.info('AgentFactory initialized with pre-registered MCP agents.')

    def register_agent_class(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent class type with the factory.

        Args:
            agent_type: String identifier for this agent type
            agent_class: Class reference (not instance) for the agent type
        """
        if agent_type in self._agent_classes:
            logger.warning(f'Overriding existing agent class registration for type: {agent_type}')
        self._agent_classes[agent_type] = agent_class
        logger.info(f"Registered agent class '{agent_class.__name__}' for type: {agent_type}")

    def register_agent_config(self, config: AgentConfig) -> None:
        """
        Register a configuration for an agent.

        Args:
            config: Agent configuration to register
        """
        if config.name in self._agent_configs:
            logger.warning(f'Overriding existing config for agent name: {config.name}')
        self._agent_configs[config.name] = config
        logger.info(f'Registered config for agent name: {config.name} (type: {config.agent_type})')

    def get_registered_agent_types(self) -> List[str]:
        """Get list of all registered agent type names"""
        return list(self._agent_classes.keys())

    def has_agent_type(self, agent_type: str) -> bool:
        """Check if a particular agent type is registered"""
        return agent_type in self._agent_classes

    @async_timed('agent_factory_get_agent')
    async def get_agent(self,
                         agent_name: str,
                         config: Optional[AgentConfig]=None,
                         use_cache: bool=True,
                         cache_key: Optional[str]=None) -> BaseAgent:
        """
        Get an agent instance by name or configuration.

        Args:
            agent_name: Name of the agent to retrieve/create
            config: Optional custom configuration (overrides stored config)
            use_cache: Whether to use/update the instance cache
            cache_key: Optional custom cache key (defaults to agent_name)

        Returns:
            BaseAgent: Agent instance

        Raises:
            AgentNotFoundError: If agent configuration not found
            AgentCreationError: If agent creation fails
        """
        # Get agent configuration
        agent_config = config
        if not agent_config:
            agent_config = self._agent_configs.get(agent_name)
            if not agent_config:
                error_msg = f'No configuration found for agent name: {agent_name}'
                logger.error(error_msg)
                raise AgentNotFoundError(agent_type=agent_name, message=error_msg)

        # Validate agent type exists
        agent_type = agent_config.agent_type
        if agent_type not in self._agent_classes:
            error_msg = f"Agent type '{agent_type}' specified in config for '{agent_name}' is not registered."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the agent class
        agent_class = self._agent_classes[agent_type]
        logger.debug(f"Resolved agent '{agent_name}' to type '{agent_type}' (Class: {agent_class.__name__})")

        # Determine cache strategy
        effective_cache_key = cache_key or agent_name
        should_use_cache = use_cache and config is None

        # Check cache for existing instance
        if should_use_cache:
            if effective_cache_key in self._agent_instances:
                # <<< 수정된 부분: 캐시에서 3개의 값 읽기 >>>
                agent, last_used, is_initialized = self._agent_instances[effective_cache_key]
                # Update last used timestamp, keep initialization status
                self._agent_instances[effective_cache_key] = (agent, get_current_time_ms(), is_initialized)
                logger.debug(f'Using cached agent instance for: {agent_name}')

                # <<< 수정된 부분: is_initialized 플래그 확인 및 재초기화 시도 >>>
                if not is_initialized and hasattr(agent, 'initialize') and asyncio.iscoroutinefunction(agent.initialize):
                    logger.warning(f"Cached agent '{agent_name}' was not marked as initialized. Attempting re-initialization.")
                    try:
                        # Assuming agent.initialize() is an async function
                        initialized_now = await agent.initialize()
                        if initialized_now:
                            # Update cache status upon successful re-initialization
                            self._agent_instances[effective_cache_key] = (agent, get_current_time_ms(), True)
                            logger.info(f"Successfully re-initialized cached agent '{agent_name}'.")
                        else:
                            # Log failure if re-initialization returns False
                            logger.error(f"Failed to re-initialize cached agent '{agent_name}'. Returning potentially uninitialized agent.")
                    except Exception as e:
                        # Log any exception during re-initialization
                        logger.error(f"Error during re-initialization of cached agent {agent_name}: {e}", exc_info=True)

                return agent # Return the cached agent

        # Create new agent instance if not found in cache or cache not used
        logger.debug(f'Creating new instance for agent: {agent_name} (Type: {agent_type})')
        try:
            # Prepare constructor arguments
            constructor_args: Dict[str, Any] = {'config': agent_config}

            # Special case for executor agent - inject tool registry
            if agent_type == 'mcp_executor':
                try:
                    from src.tools.registry import \
                        get_registry as get_tool_registry
                    constructor_args['tool_registry'] = get_tool_registry('global_tools')
                    logger.debug(f"Injecting tool_registry into {agent_type} '{agent_name}'")
                except ImportError:
                    logger.error(f"Failed to import tool registry for {agent_type} '{agent_name}'. Tool execution might fail.")
                except Exception as reg_e:
                    logger.error(f"Error getting tool registry for {agent_type} '{agent_name}': {reg_e}")

            # Create the agent instance
            agent = agent_class(**constructor_args)

            # <<< 수정된 부분: 새 에이전트 초기화 결과 저장 >>>
            initialized_success = False # Default status
            if hasattr(agent, 'initialize') and asyncio.iscoroutinefunction(agent.initialize):
                try:
                    initialized_success = await agent.initialize()
                except Exception as init_e:
                    logger.error(f"Error during initial initialization of agent '{agent_name}': {init_e}", exc_info=True)
                    # Propagate error if initialization itself fails with an exception
                    raise AgentCreationError(
                        message=f"Exception during agent initialization: {init_e}",
                        agent_type=agent_type,
                        original_error=init_e,
                        details={'agent_name': agent_name}
                    )
            else:
                # If no standard async initialize method, assume success (or apply other logic)
                logger.warning(f"Agent '{agent_name}' of type '{agent_type}' does not have a standard async initialize method. Assuming initialized.")
                initialized_success = True

            # <<< 수정된 부분: 초기화 실패 시 에러 처리 >>>
            if not initialized_success:
                error_msg = f"Failed to initialize agent '{agent_name}' (type: {agent_type}) - initialize() returned False."
                logger.error(error_msg)
                raise AgentCreationError(message=error_msg, agent_type=agent_type, details={'agent_name': agent_name})

            # Cache the newly created and initialized agent if requested
            if should_use_cache:
                # <<< 수정된 부분: 캐시에 초기화 상태(True) 함께 저장 >>>
                self._agent_instances[effective_cache_key] = (agent, get_current_time_ms(), True)
                logger.debug(f'Cached new agent instance for: {agent_name} with initialized=True')

            # Track metrics
            metrics.track_agent('created', agent_type=agent_type)

            return agent # Return the newly created and initialized agent

        except (AgentCreationError, AgentExecutionError):
            # Pass through specific agent errors
            raise
        except Exception as e:
            # Handle unexpected errors during the creation process (excluding initialization exceptions caught above)
            error_msg = f"Failed to create agent '{agent_name}' (type: {agent_type})"
            logger.error(error_msg, exc_info=True, extra={'error': str(e)}) # Log traceback
            raise AgentCreationError(
                message=f'{error_msg}: {str(e)}',
                agent_type=agent_type,
                original_error=e,
                details={'agent_name': agent_name}
            )

    async def cleanup_cache(self, max_idle_time_ms: int=600000) -> int:
        """
        Clean up cached agent instances that haven't been used for some time.

        Args:
            max_idle_time_ms: Maximum idle time in milliseconds before cleanup

        Returns:
            int: Number of agents removed from cache
        """
        current_time = get_current_time_ms()
        keys_to_remove = []
        # <<< 수정된 부분: 캐시 순회 시 3개 값 언패킹 >>>
        for key, (_, last_used_time, _) in self._agent_instances.items():
            if current_time - last_used_time > max_idle_time_ms:
                 keys_to_remove.append(key)

        removed_count = 0
        for key in keys_to_remove:
            agent_tuple = self._agent_instances.pop(key, None)
            if agent_tuple:
                # <<< 수정된 부분: 튜플에서 agent 객체만 추출 >>>
                agent = agent_tuple[0] # agent 는 튜플의 첫 번째 요소
                try:
                    if hasattr(agent, 'terminate') and asyncio.iscoroutinefunction(agent.terminate):
                        await agent.terminate()
                        logger.debug(f'Terminated and removed cached agent: {key}')
                        removed_count += 1
                    else:
                        logger.debug(f'Removed cached agent (no async terminate method): {key}')
                        removed_count += 1
                except Exception as e:
                    logger.error(f'Error terminating cached agent {key}: {str(e)}')

        if removed_count > 0:
            logger.info(f'Cleaned up {removed_count} cached agent instances')

        return removed_count

    async def shutdown(self) -> None:
        """
        Terminate all cached agent instances and clear the cache.
        Should be called during application shutdown.
        """
        logger.info(f'Shutting down AgentFactory and terminating {len(self._agent_instances)} cached agents...')

        # Create termination tasks
        shutdown_tasks = []
        # <<< 수정된 부분: 캐시 아이템 복사 시 3개 값 언패킹 >>>
        cached_items = list(self._agent_instances.items())
        self._agent_instances.clear()

        # <<< 수정된 부분: 반복문에서 agent 객체만 추출 >>>
        for key, agent_tuple in cached_items:
            agent = agent_tuple[0] # agent 는 튜플의 첫 번째 요소
            # Define termination function
            async def terminate_safely(k: str, a: BaseAgent):
                try:
                    if hasattr(a, 'terminate') and asyncio.iscoroutinefunction(a.terminate):
                        await a.terminate()
                        logger.debug(f'Terminated cached agent during shutdown: {k}')
                    else:
                        logger.debug(f'Removed cached agent (no async terminate): {k}')
                except Exception as e:
                    logger.error(f'Error terminating cached agent {k} during shutdown: {str(e)}')

            shutdown_tasks.append(terminate_safely(key, agent))

        # Execute all termination tasks concurrently
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        logger.info('AgentFactory shut down successfully. Instance cache cleared.')


# Singleton instance management - (get_agent_factory 함수는 변경 없음)
_instance: Optional[AgentFactory] = None
_factory_lock = asyncio.Lock()

async def get_agent_factory() -> AgentFactory:
    """
    Get the singleton AgentFactory instance.
    Creates the instance if it doesn't exist yet.
    Thread-safe implementation for async contexts.

    Returns:
        AgentFactory: Singleton factory instance
    """
    global _instance
    if _instance is not None:
        return _instance

    async with _factory_lock:
        if _instance is None:
            logger.debug('Creating singleton AgentFactory instance.')
            _instance = AgentFactory()

    return _instance
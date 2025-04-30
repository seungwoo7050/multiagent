import asyncio
from typing import Any, Dict, Optional, Type, Union, cast, List
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager, AGENT_METRICS
from src.agents.config import AgentConfig
from src.core.agent import BaseAgent
from src.agents.mcp_planner import MCPPlannerAgent
from src.agents.mcp_executor import MCPExecutorAgent
from src.core.exceptions import AgentExecutionError, AgentCreationError, AgentNotFoundError
from src.utils.timing import async_timed, get_current_time_ms

logger = get_logger(__name__)
metrics = get_metrics_manager()

class AgentFactory:

    def __init__(self):
        self._agent_classes: Dict[str, Type[BaseAgent]] = {'mcp_planner': MCPPlannerAgent, 'mcp_executor': MCPExecutorAgent, 'planner': MCPPlannerAgent, 'executor': MCPExecutorAgent}
        self._agent_instances: Dict[str, tuple[BaseAgent, int]] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}
        logger.info('AgentFactory initialized with pre-registered MCP agents.')

    def register_agent_class(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        if agent_type in self._agent_classes:
            logger.warning(f'Overriding existing agent class registration for type: {agent_type}')
        self._agent_classes[agent_type] = agent_class
        logger.info(f"Registered agent class '{agent_class.__name__}' for type: {agent_type}")

    def register_agent_config(self, config: AgentConfig) -> None:
        if config.name in self._agent_configs:
            logger.warning(f'Overriding existing config for agent name: {config.name}')
        self._agent_configs[config.name] = config
        logger.info(f'Registered config for agent name: {config.name} (type: {config.agent_type})')

    def get_registered_agent_types(self) -> List[str]:
        return list(self._agent_classes.keys())

    def has_agent_type(self, agent_type: str) -> bool:
        return agent_type in self._agent_classes

    @async_timed('agent_factory_get_agent')
    async def get_agent(self, agent_name: str, config: Optional[AgentConfig]=None, use_cache: bool=True, cache_key: Optional[str]=None) -> BaseAgent:
        agent_config = config
        if not agent_config:
            agent_config = self._agent_configs.get(agent_name)
            if not agent_config:
                error_msg = f'No configuration found for agent name: {agent_name}'
                logger.error(error_msg)
                raise AgentNotFoundError(agent_type=agent_name, message=error_msg)
        agent_type = agent_config.agent_type
        if agent_type not in self._agent_classes:
            error_msg = f"Agent type '{agent_type}' specified in config for '{agent_name}' is not registered."
            logger.error(error_msg)
            raise ValueError(error_msg)
        agent_class = self._agent_classes[agent_type]
        logger.debug(f"Resolved agent '{agent_name}' to type '{agent_type}' (Class: {agent_class.__name__})")
        effective_cache_key = cache_key or agent_name
        should_use_cache = use_cache and config is None
        if should_use_cache:
            if effective_cache_key in self._agent_instances:
                agent, last_used = self._agent_instances[effective_cache_key]
                self._agent_instances[effective_cache_key] = (agent, get_current_time_ms())
                logger.debug(f'Using cached agent instance for: {agent_name}')
                if not agent.initialized and hasattr(agent, 'initialize'):
                    logger.debug(f"Re-initializing potentially uninitialized cached agent '{agent_name}'")
                    await agent.initialize()
                return agent
        logger.debug(f'Creating new instance for agent: {agent_name} (Type: {agent_type})')
        try:
            constructor_args: Dict[str, Any] = {'config': agent_config}
            if agent_type == 'mcp_executor':
                try:
                    from src.tools.registry import get_registry as get_tool_registry
                    constructor_args['tool_registry'] = get_tool_registry('global_tools')
                    logger.debug(f"Injecting tool_registry into {agent_type} '{agent_name}'")
                except ImportError:
                    logger.error(f"Failed to import tool registry for {agent_type} '{agent_name}'. Tool execution might fail.")
                except Exception as reg_e:
                    logger.error(f"Error getting tool registry for {agent_type} '{agent_name}': {reg_e}")
            agent = agent_class(**constructor_args)
            initialized = await agent.initialize()
            if not initialized:
                error_msg = f"Failed to initialize agent '{agent_name}' (type: {agent_type})"
                logger.error(error_msg)
                raise AgentCreationError(message=error_msg, agent_type=agent_type, details={'agent_name': agent_name})
            if should_use_cache:
                self._agent_instances[effective_cache_key] = (agent, get_current_time_ms())
                logger.debug(f'Cached new agent instance for: {agent_name}')
            metrics.track_agent('created', agent_type=agent_type)
            return agent
        except AgentCreationError as ace:
            raise ace
        except AgentExecutionError as aee:
            raise aee
        except Exception as e:
            error_msg = f"Failed to create agent '{agent_name}' (type: {agent_type})"
            logger.exception(error_msg, extra={'error': str(e)})
            raise AgentCreationError(message=f'{error_msg}: {str(e)}', agent_type=agent_type, original_error=e, details={'agent_name': agent_name})

    async def cleanup_cache(self, max_idle_time_ms: int=600000) -> int:
        current_time = get_current_time_ms()
        keys_to_remove = [key for key, (_, last_used_time) in self._agent_instances.items() if current_time - last_used_time > max_idle_time_ms]
        removed_count = 0
        for key in keys_to_remove:
            agent_tuple = self._agent_instances.pop(key, None)
            if agent_tuple:
                agent, _ = agent_tuple
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
        logger.info(f'Shutting down AgentFactory and terminating {len(self._agent_instances)} cached agents...')
        shutdown_tasks = []
        cached_items = list(self._agent_instances.items())
        self._agent_instances.clear()
        for key, (agent, _) in cached_items:

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
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        logger.info('AgentFactory shut down successfully. Instance cache cleared.')
_instance: Optional[AgentFactory] = None
_factory_lock = asyncio.Lock()

async def get_agent_factory() -> AgentFactory:
    global _instance
    if _instance is not None:
        return _instance
    async with _factory_lock:
        if _instance is None:
            logger.debug('Creating singleton AgentFactory instance.')
            _instance = AgentFactory()
    return _instance
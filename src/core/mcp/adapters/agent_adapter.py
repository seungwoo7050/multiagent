from typing import Any, Dict, Optional, Type, cast
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.schema import BaseContextSchema, TaskContext
from src.core.agent import BaseAgent, AgentContext as CoreAgentContext, AgentResult
from src.config.logger import get_logger
from src.core.exceptions import SerializationError, AgentError, TaskError
logger = get_logger(__name__)

class AgentInputContext(BaseContextSchema):
    agent_type: str
    task_context: Optional[TaskContext] = None
    parameters: Dict[str, Any] = {}
    prompt: Optional[str] = None

class AgentOutputContext(BaseContextSchema):
    success: bool
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    agent_name: Optional[str] = None

class AgentAdapter(MCPAdapterBase):

    def __init__(self, target_component: Optional[Type[BaseAgent]]=None):
        super().__init__(target_component=target_component or BaseAgent, mcp_context_type=AgentInputContext)

    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> CoreAgentContext:
        agent_input_context: AgentInputContext
        if isinstance(context, AgentInputContext):
            agent_input_context = cast(AgentInputContext, context)
        elif isinstance(context, TaskContext):
            logger.debug(f'Adapting TaskContext to create input for agent (task_id: {context.task_id})')
            task_context = cast(TaskContext, context)
            agent_input_context = AgentInputContext(agent_type=task_context.task_type, task_context=task_context, parameters=task_context.input_data or {}, metadata=task_context.metadata, context_id=task_context.context_id)
        elif isinstance(context, BaseContextSchema):
            logger.warning(f'Received unexpected context type {type(context).__name__}. Attempting basic adaptation using BaseContextSchema.')
            base_context = cast(BaseContextSchema, context)
            agent_type_kwarg = kwargs.get('agent_type', 'unknown')
            agent_input_context = AgentInputContext(agent_type=agent_type_kwarg, parameters=base_context.metadata.get('parameters', {}), metadata=base_context.metadata, context_id=base_context.context_id)
        else:
            raise ValueError(f'Incompatible context type for AgentAdapter input: {type(context).__name__}')
        logger.debug(f"Adapting {type(agent_input_context).__name__} (ID: {agent_input_context.context_id}) to CoreAgentContext for agent type '{agent_input_context.agent_type}'")
        try:
            core_context_data: Dict[str, Any] = {'task': None, 'trace_id': agent_input_context.metadata.get('trace_id'), 'conversation_id': agent_input_context.metadata.get('conversation_id'), 'parameters': agent_input_context.parameters or {}, 'metadata': agent_input_context.metadata or {}, 'memory': {}, 'tools': []}
            if agent_input_context.task_context:
                from src.core.mcp.adapters.task_adapter import TaskAdapter
                task_adapter = TaskAdapter()
                try:
                    core_context_data['task'] = await task_adapter.adapt_input(agent_input_context.task_context)
                except Exception as task_adapt_e:
                    logger.error(f'Failed to adapt TaskContext within AgentAdapter: {task_adapt_e}')
            elif 'task_id' in agent_input_context.parameters:
                logger.warning(f"Task ID '{agent_input_context.parameters['task_id']}' found in parameters, but full TaskContext was not provided. BaseTask object might be incomplete.")
            core_agent_context = CoreAgentContext(**core_context_data)
            return core_agent_context
        except Exception as e:
            logger.error(f'Error adapting MCP Context to CoreAgentContext: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt input context for agent: {e}', original_error=e)

    async def adapt_output(self, component_output: Any, original_context: Optional[ContextProtocol]=None, **kwargs: Any) -> AgentOutputContext:
        if not isinstance(component_output, AgentResult):
            if isinstance(component_output, Exception):
                error = component_output
                logger.error(f'Adapting Exception output from agent to AgentOutputContext: {error}')
                return AgentOutputContext(success=False, error_message=str(error), agent_name=kwargs.get('agent_name', 'unknown'), metadata={'error_type': type(error).__name__})
            else:
                raise ValueError(f'Incompatible component output type: Expected AgentResult or Exception, got {type(component_output).__name__}')
        agent_result: AgentResult = cast(AgentResult, component_output)
        agent_name = agent_result.metadata.get('agent_name', 'unknown')
        logger.debug(f"Adapting AgentResult from agent '{agent_name}' to AgentOutputContext")
        try:
            context_data: Dict[str, Any] = {'success': agent_result.success, 'output_data': agent_result.output if agent_result.success else None, 'error_message': agent_result.error.get('message') if not agent_result.success and agent_result.error else None, 'metadata': agent_result.metadata or {}, 'agent_name': agent_name}
            if agent_result.task_result:
                context_data['metadata'].setdefault('task_result_summary', {})
                context_data['metadata']['task_result_summary'].update({'task_id': agent_result.task_result.task_id, 'state': agent_result.task_result.state.value, 'success': agent_result.task_result.success})
            original_context_id: Optional[str] = None
            if isinstance(original_context, BaseContextSchema):
                original_context_id = getattr(original_context, 'context_id', None)
            output_context = AgentOutputContext(**context_data, context_id=original_context_id)
            return output_context
        except Exception as e:
            logger.error(f'Error adapting AgentResult to AgentOutputContext: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt agent output to MCP context: {e}', original_error=e)
from src.core.mcp.adapters.task_adapter import TaskAdapter
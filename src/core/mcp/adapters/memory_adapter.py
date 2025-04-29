from typing import Any, Dict, Optional, Type, List, cast
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.schema import BaseContextSchema
from src.memory.base import BaseMemory
from src.memory.manager import MemoryManager
from src.config.logger import get_logger
from src.core.exceptions import SerializationError, MemoryError
logger = get_logger(__name__)

class MemoryInputContext(BaseContextSchema):
    operation: str
    key: Optional[str] = None
    context_id: str
    data: Optional[Any] = None
    ttl: Optional[int] = None
    query: Optional[str] = None
    k: Optional[int] = None
    filter_metadata: Optional[Dict[str, Any]] = None

class MemoryOutputContext(BaseContextSchema):
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None

class MemoryAdapter(MCPAdapterBase):

    def __init__(self, target_component: Union[BaseMemory, MemoryManager]):
        if not isinstance(target_component, (BaseMemory, MemoryManager)):
            raise TypeError(f'target_component must be an instance of BaseMemory or MemoryManager, got {type(target_component).__name__}')
        super().__init__(target_component=target_component, mcp_context_type=MemoryInputContext)

    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(context, MemoryInputContext):
            raise ValueError(f'Incompatible context type: Expected MemoryInputContext, got {type(context).__name__}')
        mem_input_context: MemoryInputContext = cast(MemoryInputContext, context)
        operation = mem_input_context.operation
        logger.debug(f'Adapting MemoryInputContext (ID: {mem_input_context.context_id}, Op: {operation}) for BaseMemory/MemoryManager call')
        args_dict: Dict[str, Any] = {}
        target_method_name: str = ''
        try:
            if operation == 'load':
                if not mem_input_context.key:
                    raise ValueError("Missing 'key' for load operation")
                target_method_name = 'load'
                args_dict = {'key': mem_input_context.key, 'context_id': mem_input_context.context_id, 'default': mem_input_context.metadata.get('default'), 'use_cache': mem_input_context.metadata.get('use_cache', True)}
            elif operation == 'save':
                if not mem_input_context.key:
                    raise ValueError("Missing 'key' for save operation")
                if mem_input_context.data is None:
                    raise ValueError("Missing 'data' for save operation")
                target_method_name = 'save'
                args_dict = {'key': mem_input_context.key, 'context_id': mem_input_context.context_id, 'data': mem_input_context.data, 'ttl': mem_input_context.ttl, 'update_cache': mem_input_context.metadata.get('update_cache', True)}
            elif operation == 'delete':
                if not mem_input_context.key:
                    raise ValueError("Missing 'key' for delete operation")
                target_method_name = 'delete'
                args_dict = {'key': mem_input_context.key, 'context_id': mem_input_context.context_id, 'clear_cache': mem_input_context.metadata.get('clear_cache', True)}
            elif operation == 'clear':
                target_method_name = 'clear'
                args_dict = {'context_id': mem_input_context.context_id, 'clear_cache': mem_input_context.metadata.get('clear_cache', True), 'clear_vectors': mem_input_context.metadata.get('clear_vectors', True)}
            elif operation == 'search_vectors':
                if not isinstance(self.target_component, MemoryManager) or not self.target_component.vector_store:
                    raise ValueError('Vector store operation requires a configured vector store in MemoryManager')
                if not mem_input_context.query:
                    raise ValueError("Missing 'query' for search_vectors operation")
                target_method_name = 'search_vectors'
                args_dict = {'query': mem_input_context.query, 'k': mem_input_context.k or 5, 'context_id': mem_input_context.context_id, 'filter_metadata': mem_input_context.filter_metadata}
            elif operation == 'store_vector':
                if not isinstance(self.target_component, MemoryManager) or not self.target_component.vector_store:
                    raise ValueError('Vector store operation requires a configured vector store in MemoryManager')
                if mem_input_context.data is None or not isinstance(mem_input_context.data, str):
                    raise ValueError("Missing or invalid 'data' (text) for store_vector operation")
                target_method_name = 'store_vector'
                args_dict = {'text': mem_input_context.data, 'metadata': mem_input_context.metadata or {}, 'context_id': mem_input_context.context_id}
            elif operation == 'exists':
                if not mem_input_context.key:
                    raise ValueError("Missing 'key' for exists operation")
                target_method_name = 'exists'
                args_dict = {'key': mem_input_context.key, 'context_id': mem_input_context.context_id, 'check_cache': mem_input_context.metadata.get('check_cache', True)}
            elif operation == 'list_keys':
                target_method_name = 'list_keys'
                args_dict = {'context_id': mem_input_context.context_id, 'pattern': mem_input_context.metadata.get('pattern')}
            else:
                raise ValueError(f'Unsupported memory operation specified in context: {operation}')
            if not hasattr(self.target_component, target_method_name):
                raise NotImplementedError(f"Target component {type(self.target_component).__name__} does not implement the required method '{target_method_name}' for operation '{operation}'")
            return {'operation': target_method_name, 'args': args_dict}
        except Exception as e:
            logger.error(f'Error adapting MemoryInputContext for operation {operation}: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt input context for memory operation {operation}: {e}', original_error=e)

    async def adapt_output(self, component_output: Any, original_context: Optional[ContextProtocol]=None, **kwargs: Any) -> MemoryOutputContext:
        operation = kwargs.get('operation', 'unknown')
        if operation == 'unknown' and isinstance(original_context, MemoryInputContext):
            operation = original_context.operation
        logger.debug(f"Adapting output from memory operation '{operation}' to MemoryOutputContext")
        success: bool = True
        error_message: Optional[str] = None
        result_data: Any = component_output
        if operation in ['save', 'delete', 'clear']:
            if isinstance(component_output, bool):
                success = component_output
                result_data = None
            else:
                success = False
                error_message = f'Unexpected output type for {operation}: {type(component_output).__name__}'
                result_data = None
        elif operation == 'exists':
            if isinstance(component_output, bool):
                result_data = component_output
                success = True
            else:
                success = False
                error_message = f'Unexpected output type for exists: {type(component_output).__name__}'
                result_data = False
        elif operation == 'load':
            success = True
            result_data = component_output
        elif operation == 'search_vectors':
            if isinstance(component_output, list):
                result_data = component_output
                success = True
            else:
                success = False
                error_message = f'Unexpected output type for search_vectors: {type(component_output).__name__}'
                result_data = []
        if isinstance(component_output, Exception):
            success = False
            error_message = str(component_output)
            result_data = None
        try:
            context_data: Dict[str, Any] = {'success': success, 'result': result_data, 'error_message': error_message, 'metadata': {'operation_performed': operation}}
            original_context_id: Optional[str] = None
            if isinstance(original_context, BaseContextSchema):
                original_context_id = getattr(original_context, 'context_id', None)
            output_context = MemoryOutputContext(**context_data, context_id=original_context_id)
            return output_context
        except Exception as e:
            logger.error(f'Error adapting memory output for operation {operation}: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt memory output to MCP context: {e}', original_error=e)

    async def process_with_mcp(self, context: ContextProtocol, **kwargs: Any) -> ContextProtocol:
        component_output: Any = None
        component_error: Optional[Exception] = None
        operation_name: str = 'unknown'
        try:
            adapted_call: Dict[str, Any] = await self.adapt_input(context, **kwargs)
            operation_name = adapted_call['operation']
            operation_args: Dict[str, Any] = adapted_call['args']
            target_method: Callable[..., Coroutine[Any, Any, Any]]
            target_method = getattr(self.target_component, operation_name)
            logger.debug(f"Executing memory operation '{operation_name}' on {type(self.target_component).__name__}")
            component_output = await target_method(**operation_args)
        except Exception as e:
            logger.error(f"Error during MCP processing for memory operation '{operation_name}': {e}", exc_info=True)
            component_error = e
        output_context: MemoryOutputContext = await self.adapt_output(component_output=component_error if component_error else component_output, original_context=context, operation=operation_name, **kwargs)
        if component_error:
            output_context.success = False
            output_context.error_message = str(component_error)
            if isinstance(component_error, MemoryError) and component_error.details:
                output_context.metadata['error_details'] = component_error.details
        return output_context
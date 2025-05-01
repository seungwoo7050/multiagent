from typing import Any, Dict, Optional, Type, List, Union, Callable, Coroutine, TypeVar, cast, Tuple
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.schema import BaseContextSchema
from src.memory.base import BaseMemory
from src.memory.manager import MemoryManager
from src.config.logger import get_logger
from src.config.errors import MemoryError
from src.core.exceptions import SerializationError



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
        
        # Operation handler mapping
        self._operation_handlers = {
            'load': self._adapt_load_operation,
            'save': self._adapt_save_operation,
            'delete': self._adapt_delete_operation,
            'clear': self._adapt_clear_operation,
            'search_vectors': self._adapt_search_vectors_operation,
            'store_vector': self._adapt_store_vector_operation,
            'exists': self._adapt_exists_operation,
            'list_keys': self._adapt_list_keys_operation,
        }
        
    async def _adapt_load_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for load operation."""
        if not context.key:
            raise ValueError("Missing 'key' for load operation")
        
        args_dict = {
            'key': context.key, 
            'context_id': context.context_id, 
            'default': context.metadata.get('default'), 
            'use_cache': context.metadata.get('use_cache', True)
        }
        
        return args_dict, 'load'

    async def _adapt_save_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for save operation."""
        if not context.key:
            raise ValueError("Missing 'key' for save operation")
        if context.data is None:
            raise ValueError("Missing 'data' for save operation")
        
        args_dict = {
            'key': context.key, 
            'context_id': context.context_id, 
            'data': context.data, 
            'ttl': context.ttl, 
            'update_cache': context.metadata.get('update_cache', True)
        }
        
        return args_dict, 'save'

    async def _adapt_delete_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for delete operation."""
        if not context.key:
            raise ValueError("Missing 'key' for delete operation")
        
        args_dict = {
            'key': context.key, 
            'context_id': context.context_id, 
            'clear_cache': context.metadata.get('clear_cache', True)
        }
        
        return args_dict, 'delete'

    async def _adapt_clear_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for clear operation."""
        args_dict = {
            'context_id': context.context_id, 
            'clear_cache': context.metadata.get('clear_cache', True), 
            'clear_vectors': context.metadata.get('clear_vectors', True)
        }
        
        return args_dict, 'clear'

    async def _adapt_search_vectors_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for search_vectors operation."""
        if not isinstance(self.target_component, MemoryManager) or not self.target_component.vector_store:
            raise ValueError('Vector store operation requires a configured vector store in MemoryManager')
        if not context.query:
            raise ValueError("Missing 'query' for search_vectors operation")
        
        args_dict = {
            'query': context.query, 
            'k': context.k or 5, 
            'context_id': context.context_id, 
            'filter_metadata': context.filter_metadata
        }
        
        return args_dict, 'search_vectors'

    async def _adapt_store_vector_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for store_vector operation."""
        if not isinstance(self.target_component, MemoryManager) or not self.target_component.vector_store:
            raise ValueError('Vector store operation requires a configured vector store in MemoryManager')
        if context.data is None or not isinstance(context.data, str):
            raise ValueError("Missing or invalid 'data' (text) for store_vector operation")
        
        args_dict = {
            'text': context.data, 
            'metadata': context.metadata or {}, 
            'context_id': context.context_id
        }
        
        return args_dict, 'store_vector'

    async def _adapt_exists_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for exists operation."""
        if not context.key:
            raise ValueError("Missing 'key' for exists operation")
        
        args_dict = {
            'key': context.key, 
            'context_id': context.context_id, 
            'check_cache': context.metadata.get('check_cache', True)
        }
        
        return args_dict, 'exists'

    async def _adapt_list_keys_operation(self, context: MemoryInputContext) -> Tuple[Dict[str, Any], str]:
        """Adapt input for list_keys operation."""
        args_dict = {
            'context_id': context.context_id, 
            'pattern': context.metadata.get('pattern')
        }
        
        return args_dict, 'list_keys'    

    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(context, MemoryInputContext):
            raise ValueError(f'Incompatible context type: Expected MemoryInputContext, got {type(context).__name__}')
        
        mem_input_context: MemoryInputContext = cast(MemoryInputContext, context)
        operation = mem_input_context.operation
        
        logger.debug(f'Adapting MemoryInputContext (ID: {mem_input_context.context_id}, Op: {operation}) for BaseMemory/MemoryManager call')
        
        try:
            # Get the appropriate handler for this operation
            handler = self._operation_handlers.get(operation)
            if not handler:
                raise ValueError(f'Unsupported memory operation specified in context: {operation}')
            
            # Execute the handler to get operation details
            args_dict, target_method_name = await handler(mem_input_context)
            
            # Validate target method exists
            if not hasattr(self.target_component, target_method_name):
                raise NotImplementedError(
                    f"Target component {type(self.target_component).__name__} does not implement "
                    f"the required method '{target_method_name}' for operation '{operation}'"
                )
            
            return {'operation': target_method_name, 'args': args_dict}
        
        except Exception as e:
            logger.error(f'Error adapting MemoryInputContext for operation {operation}: {e}', exc_info=True)
            raise SerializationError(
                f'Could not adapt input context for memory operation {operation}: {e}', 
                original_error=e
            )

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
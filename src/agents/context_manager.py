from typing import Dict, Optional, Type, Any, cast
import time
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.core.mcp.schema import TaskContext
from src.core.mcp.compression import optimize_context_data
from src.config.logger import get_logger
logger = get_logger(__name__)

class AgentContextManager:

    def __init__(self, agent_id: str):
        self.agent_id: str = agent_id
        self._contexts: Dict[str, ContextProtocol] = {}
        self._latest_context_ids: Dict[Type[ContextProtocol], str] = {}
        logger.debug(f'AgentContextManager initialized for agent {agent_id}')

    def update_context(self, context: ContextProtocol) -> None:
        if not isinstance(context, ContextProtocol):
            logger.warning(f'Attempted to update context with non-ContextProtocol object: {type(context)}')
            return
        context_id: Optional[str] = getattr(context, 'context_id', None)
        if not context_id:
            if isinstance(context, BaseContextSchema):
                context_id = context.context_id
            else:
                logger.warning(f"Context object of type {type(context).__name__} lacks context_id attribute or it's empty. Cannot store effectively by ID.")
                return
        logger.debug(f'Updating context (ID: {context_id}, Type: {type(context).__name__}) for agent {self.agent_id}')
        self._contexts[context_id] = context
        context_type: Type[ContextProtocol] = type(context)
        self._latest_context_ids[context_type] = context_id

    def get_context(self, context_id: Optional[str]=None, context_type: Optional[Type[ContextProtocol]]=None) -> Optional[ContextProtocol]:
        if context_id:
            context = self._contexts.get(context_id)
            if context:
                logger.debug(f'Retrieved context by ID: {context_id} (Type: {type(context).__name__})')
                if context_type and (not isinstance(context, context_type)):
                    logger.warning(f'Retrieved context {context_id} type mismatch. Expected {context_type.__name__}, got {type(context).__name__}.')
                    return None
                return context
            else:
                logger.debug(f'Context with ID {context_id} not found.')
                return None
        elif context_type:
            latest_id = self._latest_context_ids.get(context_type)
            if latest_id:
                context = self._contexts.get(latest_id)
                if context:
                    logger.debug(f'Retrieved latest context of type {context_type.__name__} (ID: {latest_id})')
                    return context
                else:
                    logger.warning(f'Inconsistency detected: Latest ID {latest_id} for type {context_type.__name__} found, but the context object is missing from store.')
                    self._latest_context_ids.pop(context_type, None)
                    return None
            else:
                logger.debug(f'No context found for type {context_type.__name__}')
                return None
        else:
            logger.error('Cannot get context: Either context_id or context_type must be provided.')
            return None

    def get_all_contexts(self) -> Dict[str, ContextProtocol]:
        return self._contexts.copy()

    def optimize_context(self, context_id: Optional[str]=None, context_type: Optional[Type[ContextProtocol]]=None) -> Optional[ContextProtocol]:
        context_to_optimize = self.get_context(context_id=context_id, context_type=context_type)
        if not context_to_optimize:
            logger.warning('Context to optimize was not found.')
            return None
        try:
            ctx_id = getattr(context_to_optimize, 'context_id', 'N/A')
            ctx_type = type(context_to_optimize).__name__
            logger.debug(f'Optimizing context (ID: {ctx_id}, Type: {ctx_type})')
            start_time = time.time()
            optimized_context: ContextProtocol = optimize_context_data(context_to_optimize)
            duration = time.time() - start_time
            logger.debug(f'Context optimization finished in {duration:.4f}s')
            if optimized_context is not context_to_optimize:
                logger.debug(f'Updating stored context {ctx_id} with optimized version.')
                self.update_context(optimized_context)
                return optimized_context
            else:
                logger.debug(f'Optimization returned the original context object for {ctx_id}.')
                return context_to_optimize
        except Exception as e:
            ctx_id = getattr(context_to_optimize, 'context_id', 'N/A')
            logger.error(f'Error during context optimization for {ctx_id}: {e}', exc_info=True)
            return context_to_optimize

    def clear_contexts(self) -> None:
        count = len(self._contexts)
        self._contexts.clear()
        self._latest_context_ids.clear()
        logger.info(f'Cleared {count} contexts for agent {self.agent_id}')
import time
from typing import Dict, Optional, Type

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.core.mcp.compression import optimize_context_data
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema

logger = get_logger(__name__)
metrics = get_metrics_manager()

class AgentContextManager:
    """
    Manages context objects for agents using the Model Context Protocol.
    Provides storage, retrieval, and optimization of context objects.
    """

    def __init__(self, agent_id: str):
        """
        Initialize a context manager for an agent.
        
        Args:
            agent_id: Unique identifier for the owning agent
        """
        self.agent_id: str = agent_id
        self._contexts: Dict[str, ContextProtocol] = {}
        self._latest_context_ids: Dict[Type[ContextProtocol], str] = {}
        logger.debug(f'AgentContextManager initialized for agent {agent_id}')

    def update_context(self, context: ContextProtocol) -> None:
        """
        Store or update a context object.
        
        Args:
            context: The context object to store or update
        """
        # Validate context is a ContextProtocol
        if not isinstance(context, ContextProtocol):
            logger.warning(f'Attempted to update context with non-ContextProtocol object: {type(context)}')
            return
        
        # Get context_id, handling different implementations
        context_id: Optional[str] = getattr(context, 'context_id', None)
        if not context_id:
            if isinstance(context, BaseContextSchema):
                context_id = context.context_id
            else:
                logger.warning(f"Context object of type {type(context).__name__} lacks context_id attribute or it's empty. Cannot store effectively by ID.")
                return
        
        logger.debug(f'Updating context (ID: {context_id}, Type: {type(context).__name__}) for agent {self.agent_id}')
        
        # Store the context and track latest ID by type
        self._contexts[context_id] = context
        context_type: Type[ContextProtocol] = type(context)
        self._latest_context_ids[context_type] = context_id
        
        # Track metrics for context operations
        metrics.track_memory('operations', operation_type='context_update')

    def get_context(self, context_id: Optional[str]=None, context_type: Optional[Type[ContextProtocol]]=None) -> Optional[ContextProtocol]:
        """
        Retrieve a context by ID or get the latest context of a specific type.
        
        Args:
            context_id: Optional context ID to retrieve
            context_type: Optional context type to retrieve latest instance of
            
        Returns:
            ContextProtocol or None: The retrieved context or None if not found
        """
        metrics.track_memory('operations', operation_type='context_get')
        
        # Case 1: Retrieve by ID
        if context_id:
            context = self._contexts.get(context_id)
            if context:
                logger.debug(f'Retrieved context by ID: {context_id} (Type: {type(context).__name__})')
                
                # Validate type if specified
                if context_type and (not isinstance(context, context_type)):
                    logger.warning(f'Retrieved context {context_id} type mismatch. Expected {context_type.__name__}, got {type(context).__name__}.')
                    metrics.track_memory('operations', operation_type='context_type_mismatch')
                    return None
                
                return context
            else:
                logger.debug(f'Context with ID {context_id} not found.')
                metrics.track_memory('operations', operation_type='context_miss')
                return None
        
        # Case 2: Retrieve latest by type
        elif context_type:
            latest_id = self._latest_context_ids.get(context_type)
            if latest_id:
                context = self._contexts.get(latest_id)
                if context:
                    logger.debug(f'Retrieved latest context of type {context_type.__name__} (ID: {latest_id})')
                    return context
                else:
                    # Handle inconsistency between tracking and storage
                    logger.warning(f'Inconsistency detected: Latest ID {latest_id} for type {context_type.__name__} found, but the context object is missing from store.')
                    self._latest_context_ids.pop(context_type, None)
                    metrics.track_memory('operations', operation_type='context_inconsistency')
                    return None
            else:
                logger.debug(f'No context found for type {context_type.__name__}')
                metrics.track_memory('operations', operation_type='context_miss')
                return None
        
        # Case 3: No identifiers provided
        else:
            logger.error('Cannot get context: Either context_id or context_type must be provided.')
            return None

    def get_all_contexts(self) -> Dict[str, ContextProtocol]:
        """
        Get all stored context objects.
        
        Returns:
            Dict[str, ContextProtocol]: Dictionary of all contexts by ID
        """
        return self._contexts.copy()

    def optimize_context(self, context_id: Optional[str]=None, context_type: Optional[Type[ContextProtocol]]=None) -> Optional[ContextProtocol]:
        """
        Optimize a context object using compression techniques.
        
        Args:
            context_id: Optional ID of context to optimize
            context_type: Optional type of context to optimize (uses latest)
            
        Returns:
            ContextProtocol or None: Optimized context or None if not found
        """
        # Get the context to optimize
        context_to_optimize = self.get_context(context_id=context_id, context_type=context_type)
        if not context_to_optimize:
            logger.warning('Context to optimize was not found.')
            return None
        
        try:
            # Log optimization attempt
            ctx_id = getattr(context_to_optimize, 'context_id', 'N/A')
            ctx_type = type(context_to_optimize).__name__
            logger.debug(f'Optimizing context (ID: {ctx_id}, Type: {ctx_type})')
            
            # Track performance
            start_time = time.time()
            
            # Optimize the context
            optimized_context: ContextProtocol = optimize_context_data(context_to_optimize)
            
            # Log performance results
            duration = time.time() - start_time
            logger.debug(f'Context optimization finished in {duration:.4f}s')
            metrics.track_memory('duration', operation_type='context_optimization', value=duration)
            
            # Update if optimization produced a new object
            if optimized_context is not context_to_optimize:
                logger.debug(f'Updating stored context {ctx_id} with optimized version.')
                self.update_context(optimized_context)
                return optimized_context
            else:
                logger.debug(f'Optimization returned the original context object for {ctx_id}.')
                return context_to_optimize
                
        except Exception as e:
            # Handle optimization errors
            ctx_id = getattr(context_to_optimize, 'context_id', 'N/A')
            logger.error(f'Error during context optimization for {ctx_id}: {e}', exc_info=True)
            metrics.track_memory('operations', operation_type='context_optimization_error')
            return context_to_optimize

    def clear_contexts(self) -> None:
        """
        Clear all stored contexts.
        """
        count = len(self._contexts)
        self._contexts.clear()
        self._latest_context_ids.clear()
        metrics.track_memory('operations', operation_type='context_clear')
        logger.info(f'Cleared {count} contexts for agent {self.agent_id}')
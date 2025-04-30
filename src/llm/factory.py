import asyncio
from typing import Any, Dict, Optional, Type
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.adapters import get_adapter as get_llm_adapter_instance
from src.llm.models import get_model_info, list_available_models
from src.config.errors import LLMError, ErrorCode

settings = get_settings()
logger = get_logger(__name__)
_llm_factory_instance: Optional['LLMFactory'] = None
_llm_factory_lock = asyncio.Lock()

class LLMFactory:
    """
    Factory for creating LLM adapters based on model name.
    Provides a unified interface for obtaining properly configured model instances.
    """
    def __init__(self):
        logger.debug('LLMFactory initialized.')

    async def create_adapter(self, model_name: str, config: Optional[Dict[str, Any]]=None) -> BaseLLMAdapter:
        """
        Create and initialize an adapter for the specified model.
        
        Args:
            model_name: The name of the model to create an adapter for
            config: Optional configuration parameters for the adapter
            
        Returns:
            BaseLLMAdapter: Initialized adapter instance
            
        Raises:
            LLMError: If adapter creation or initialization fails
        """
        logger.info(f'Attempting to create LLM adapter for model: {model_name}')
        model_info = get_model_info(model_name)
        
        if not model_info:
            error_msg = f"Model '{model_name}' not found in model registry. Cannot create adapter."
            logger.error(error_msg)
            raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message=error_msg, model=model_name)
            
        try:
            adapter_kwargs = config or {}
            adapter_instance = get_llm_adapter_instance(model=model_name, **adapter_kwargs)
            
            if not adapter_instance.initialized:
                initialized = await adapter_instance.ensure_initialized()
                if not initialized:
                    raise LLMError(
                        code=ErrorCode.INITIALIZATION_ERROR, 
                        message=f'Failed to initialize created adapter for model {model_name}', 
                        model=model_name
                    )
                    
            logger.info(f'Successfully created LLM adapter instance for model: {model_name} (Type: {type(adapter_instance).__name__})')
            return adapter_instance
            
        except LLMError as lle:
            logger.error(f"LLMError during adapter creation for model '{model_name}': {lle.message}", extra=lle.to_dict())
            raise lle
            
        except Exception as e:
            logger.exception(f"Unexpected error creating adapter for model '{model_name}': {e}")
            raise LLMError(
                code=ErrorCode.LLM_PROVIDER_ERROR, 
                message=f'Failed to create adapter instance for model {model_name}: {str(e)}', 
                model=model_name, 
                original_error=e
            )

async def get_llm_factory() -> LLMFactory:
    """
    Get the singleton LLMFactory instance.
    
    Returns:
        LLMFactory: Shared factory instance
        
    Raises:
        RuntimeError: If factory creation fails
    """
    global _llm_factory_instance
    
    if _llm_factory_instance is None:
        async with _llm_factory_lock:
            if _llm_factory_instance is None:
                _llm_factory_instance = LLMFactory()
                logger.info('Singleton LLMFactory instance created.')
                
    if _llm_factory_instance is None:
        raise RuntimeError('Failed to create or retrieve LLMFactory instance.')
        
    return _llm_factory_instance
from typing import Dict, Type, Optional, Any, Union
from src.llm.base import BaseLLMAdapter
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import LLMError, ErrorCode
settings = get_settings()
logger = get_logger(__name__)
_ADAPTER_REGISTRY: Dict[str, Type[BaseLLMAdapter]] = {}

def _lazy_load_adapter(provider: str) -> Optional[Type[BaseLLMAdapter]]:
    provider = provider.lower()
    adapter_class: Optional[Type[BaseLLMAdapter]] = None
    if provider == 'openai':
        from src.llm.adapters.openai import OpenAIAdapter
        adapter_class = OpenAIAdapter
        logger.debug('Lazily loaded OpenAIAdapter.')
    elif provider == 'anthropic':
        from src.llm.adapters.anthropic import AnthropicAdapter
        adapter_class = AnthropicAdapter
        logger.debug('Lazily loaded AnthropicAdapter.')
    else:
        logger.debug(f'No lazy loader found for provider: {provider}')
        return None
    return adapter_class

def register_adapter(provider: str, adapter_class: Type[BaseLLMAdapter]) -> None:
    provider = provider.lower()
    if provider in _ADAPTER_REGISTRY:
        logger.warning(f'Overriding existing adapter registration for provider: {provider}')
    _ADAPTER_REGISTRY[provider] = adapter_class
    logger.debug(f"Registered LLM adapter class '{adapter_class.__name__}' for provider: {provider}")

def get_adapter_class(provider: str) -> Type[BaseLLMAdapter]:
    provider = provider.lower()
    if provider in _ADAPTER_REGISTRY:
        logger.debug(f"Found adapter class for '{provider}' in registry.")
        return _ADAPTER_REGISTRY[provider]
    logger.debug(f"Adapter class for provider '{provider}' not in registry. Attempting lazy load...")
    adapter_class = _lazy_load_adapter(provider)
    if adapter_class:
        register_adapter(provider, adapter_class)
        return adapter_class
    error_msg = f'Unsupported LLM provider or adapter class not found: {provider}'
    logger.error(error_msg)
    raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message=error_msg, provider=provider)

def get_adapter(model: str, provider: Optional[str]=None, api_key: Optional[str]=None, api_base: Optional[str]=None, timeout: Optional[float]=None, max_retries: Optional[int]=None, cache_enabled: Optional[bool]=None, max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, **kwargs: Any) -> BaseLLMAdapter:
    effective_provider: Optional[str] = provider
    if not effective_provider:
        effective_provider = settings.LLM_MODEL_PROVIDER_MAP.get(model)
        if not effective_provider:
            error_msg = f'Could not determine LLM provider for model: {model}. Please specify provider or update LLM_MODEL_PROVIDER_MAP.'
            logger.error(error_msg)
            raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message=error_msg, model=model)
        logger.debug(f"Inferred provider '{effective_provider}' for model '{model}'.")
    effective_provider = effective_provider.lower()
    try:
        adapter_class: Type[BaseLLMAdapter] = get_adapter_class(effective_provider)
    except LLMError as e:
        e.details = e.details or {}
        e.details['model'] = model
        raise e
    provider_config: Dict[str, Any] = settings.LLM_PROVIDERS_CONFIG.get(effective_provider, {})
    final_api_key: Optional[str] = api_key
    if not final_api_key:
        final_api_key = provider_config.get('api_key')
        if not final_api_key:
            error_msg = f"No API key found for provider '{effective_provider}'. Provide it directly or set the corresponding environment variable (e.g., {effective_provider.upper()}_API_KEY)."
            logger.error(error_msg)
            raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message=error_msg, provider=effective_provider, model=model)
    final_api_base: Optional[str] = api_base
    if not final_api_base and 'api_base' in provider_config:
        final_api_base = provider_config.get('api_base')
    try:
        adapter_init_args: Dict[str, Any] = {'model': model, 'provider': effective_provider, 'api_key': final_api_key, 'api_base': final_api_base, 'timeout': timeout, 'max_retries': max_retries, 'cache_enabled': cache_enabled if cache_enabled is not None else True, 'max_tokens': max_tokens, 'temperature': temperature, 'top_p': top_p, **kwargs}
        final_adapter_args = {k: v for k, v in adapter_init_args.items() if v is not None}
        adapter_instance = adapter_class(**final_adapter_args)
        logger.debug(f"Successfully created adapter instance for model '{model}' (Provider: {effective_provider})")
        return adapter_instance
    except Exception as e:
        error_msg = f"Failed to create adapter instance for model '{model}' (Provider: {effective_provider}): {str(e)}"
        logger.exception(error_msg)
        raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message=error_msg, provider=effective_provider, model=model, original_error=e)
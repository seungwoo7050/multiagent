"""
LLM Adapters package for interfacing with different LLM providers.
This module provides a factory for creating LLM adapters.
"""

from typing import Dict, Type, Optional, Any, Union

# Import all adapter classes
# These will be lazy-loaded to avoid circular imports
# from src.llm.adapters.openai import OpenAIAdapter
# from src.llm.adapters.anthropic import AnthropicAdapter

from src.llm.base import BaseLLMAdapter
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import LLMError, ErrorCode

settings = get_settings()
logger = get_logger(__name__)

# Registry of adapter classes - will be populated lazily
_ADAPTER_REGISTRY: Dict[str, Type[BaseLLMAdapter]] = {}


def _lazy_load_adapter(provider: str) -> Optional[Type[BaseLLMAdapter]]:
    """
    Lazy-load an adapter class to avoid circular imports.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        
    Returns:
        Optional[Type[BaseLLMAdapter]]: The adapter class or None if not found
    """
    if provider == "openai":
        from src.llm.adapters.openai import OpenAIAdapter
        return OpenAIAdapter
    elif provider == "anthropic":
        from src.llm.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter
    else:
        return None


def register_adapter(provider: str, adapter_class: Type[BaseLLMAdapter]) -> None:
    """
    Register an adapter class.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        adapter_class: The adapter class
    """
    _ADAPTER_REGISTRY[provider.lower()] = adapter_class
    logger.debug(f"Registered LLM adapter for provider: {provider}")


def get_adapter_class(provider: str) -> Type[BaseLLMAdapter]:
    """
    Get the adapter class for a provider.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
        
    Returns:
        Type[BaseLLMAdapter]: The adapter class
        
    Raises:
        LLMError: If the provider is not supported
    """
    provider = provider.lower()
    
    # Check if already loaded
    if provider in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[provider]
    
    # Try to lazy-load
    adapter_class = _lazy_load_adapter(provider)
    if adapter_class:
        register_adapter(provider, adapter_class)
        return adapter_class
    
    # Provider not supported
    raise LLMError(
        code=ErrorCode.LLM_PROVIDER_ERROR,
        message=f"Unsupported LLM provider: {provider}",
        provider=provider
    )


def get_adapter(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> BaseLLMAdapter:
    """
    Get an adapter instance for the specified model.
    
    Args:
        model: The model identifier (e.g., "gpt-4o", "claude-3-opus")
        provider: The provider name (e.g., "openai", "anthropic")
            If not provided, will be inferred from the model
        api_key: API key for the provider
            If not provided, will be loaded from settings
        api_base: Base URL for the API
            If not provided, will be loaded from settings
        **kwargs: Additional parameters to pass to the adapter
        
    Returns:
        BaseLLMAdapter: An instance of the appropriate adapter
        
    Raises:
        LLMError: If the model or provider is not supported
    """
    # If provider not specified, try to infer from model
    if not provider:
        provider = settings.LLM_MODEL_PROVIDER_MAP.get(model)
        if not provider:
            raise LLMError(
                code=ErrorCode.LLM_PROVIDER_ERROR,
                message=f"Could not determine provider for model: {model}",
                model=model
            )
    
    provider = provider.lower()
    
    # Get the adapter class
    adapter_class = get_adapter_class(provider)
    
    # Get provider config from settings
    provider_config = settings.LLM_PROVIDERS_CONFIG.get(provider, {})
    
    # Use provided API key, or fall back to provider config
    if not api_key:
        api_key = provider_config.get("api_key")
        if not api_key:
            raise LLMError(
                code=ErrorCode.LLM_PROVIDER_ERROR,
                message=f"No API key found for provider: {provider}",
                provider=provider,
                model=model
            )
    
    # Use provided API base, or fall back to provider config
    if not api_base and "api_base" in provider_config:
        api_base = provider_config.get("api_base")
    
    # Create and return the adapter instance
    try:
        adapter = adapter_class(
            model=model,
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            **kwargs
        )
        logger.debug(f"Created adapter for {model} ({provider})")
        return adapter
    except Exception as e:
        raise LLMError(
            code=ErrorCode.LLM_PROVIDER_ERROR,
            message=f"Failed to create adapter for {model} ({provider}): {str(e)}",
            provider=provider,
            model=model,
            original_error=e
        )


# Register built-in adapters (this will be done lazily now)
# register_adapter("openai", OpenAIAdapter)
# register_adapter("anthropic", AnthropicAdapter)
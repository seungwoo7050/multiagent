from src.llm.base import BaseLLMAdapter
from src.llm.adapters import get_adapter, get_adapter_class
from src.llm.connection_pool import get_connection_pool, cleanup_connection_pools
from src.llm.tokenizer import count_tokens, get_token_limit
from src.llm.cache import get_cache, clear_cache, cache_result
from src.llm.retry import retry_with_exponential_backoff
from src.llm.parallel import execute_parallel, race_models
from src.llm.models import get_model_info, list_available_models
from src.llm.prompt_optimizer import optimize_prompt

from src.config.logger import get_logger
from src.config.settings import get_settings

__all__ = [
    "BaseLLMAdapter",
    "get_adapter",
    "get_adapter_class",
    "get_connection_pool",
    "cleanup_connection_pools",
    "count_tokens",
    "get_token_limit",
    "get_cache",
    "clear_cache",
    "cache_result",
    "retry_with_exponential_backoff",
    "execute_parallel",
    "race_models",
    "get_model_info",
    "list_available_models",
    "optimize_prompt",
]

settings = get_settings()
logger = get_logger(__name__)

def initialize_llm_module():
    """Initialize the LLM module components."""
    logger.info(f"Initializing LLM module with primary model: {settings.PRIMARY_LLM}")
    # Pre-create connection pools for commonly used models
    for model_name in settings.ENABLED_MODELS_SET:
        provider = settings.LLM_MODEL_PROVIDER_MAP.get(model_name)
        if provider:
            get_connection_pool(provider)
            logger.debug(f"Initialized connection pool for {provider}")
    logger.info("LLM module initialization complete")
    return True

# Initialize module when imported
# initialize_llm_module()
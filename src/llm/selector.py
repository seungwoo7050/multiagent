from typing import List, Optional, Tuple, Dict, Any
from src.config.settings import get_settings
from src.config.logger import get_logger
from src.llm.models import list_available_models
logger = get_logger(__name__)
settings = get_settings()
_available_models_cache: Optional[List[str]] = None

async def get_available_models() -> List[str]:
    global _available_models_cache
    if _available_models_cache is None:
        try:
            _available_models_cache = list(settings.ENABLED_MODELS_SET)
            logger.info(f'Initialized available models cache: {_available_models_cache}')
        except Exception as e:
            logger.error(f'Failed to get available models: {e}')
            _available_models_cache = [settings.PRIMARY_LLM, settings.FALLBACK_LLM]
    return _available_models_cache

async def select_models(requested_model: Optional[str]=None, task_context: Optional[Dict[str, Any]]=None, strategy: str='default') -> Tuple[str, List[str]]:
    available_models = await get_available_models()
    primary_model: str
    fallback_models: List[str] = []
    if requested_model and requested_model in available_models:
        primary_model = requested_model
        logger.debug(f'Using requested model as primary: {primary_model}')
    elif settings.PRIMARY_LLM in available_models:
        primary_model = settings.PRIMARY_LLM
        logger.debug(f'Using default primary model from settings: {primary_model}')
    elif available_models:
        primary_model = available_models[0]
        logger.warning(f'Requested/Default primary model not available. Using first available model: {primary_model}')
    else:
        logger.error('No available LLM models found. Cannot select primary model.')
        raise ValueError('No available LLM models configured or found.')
    potential_fallbacks = []
    if settings.FALLBACK_LLM and settings.FALLBACK_LLM in available_models and (settings.FALLBACK_LLM != primary_model):
        potential_fallbacks.append(settings.FALLBACK_LLM)
        logger.debug(f'Added default fallback model from settings: {settings.FALLBACK_LLM}')
    for model in available_models:
        if model != primary_model and model not in potential_fallbacks:
            potential_fallbacks.append(model)
    fallback_models = potential_fallbacks
    logger.debug(f'Selected fallback models: {fallback_models}')
    return (primary_model, fallback_models)
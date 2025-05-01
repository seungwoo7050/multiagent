import asyncio
from typing import Dict, Optional, Any, List, Type
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema, TaskContext
from src.llm.models import list_available_models
from src.config.logger import get_logger
from src.config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)

class ContextModelSelector:

    def __init__(self):
        logger.debug('ContextModelSelector initialized.')
        self.selection_rules = self._load_selection_rules()

    def _load_selection_rules(self):
        # Try to load from settings first
        custom_rules = getattr(settings, 'CONTEXT_MODEL_SELECTION_RULES', None)
        if custom_rules:
            logger.info('Loading custom model selection rules from settings')
            return custom_rules
        return [{'condition': lambda ctx: isinstance(ctx, TaskContext) and ctx.task_type == 'coding' and (ctx.metadata.get('estimated_tokens', 0) > 50000), 'preferred_model': 'gpt-4-turbo'}, {'condition': lambda ctx: isinstance(ctx, BaseContextSchema) and ctx.metadata.get('low_latency', False), 'preferred_model': 'claude-3-haiku'}, {'condition': lambda ctx: isinstance(ctx, TaskContext) and ctx.input_data and ('image analysis' in ctx.input_data.get('goal', '').lower()), 'preferred_model': 'gpt-4o'}]

    async def select_model(self, context: ContextProtocol, available_models: Optional[List[str]]=None) -> str:
        context_type = type(context).__name__
        context_id = getattr(context, 'context_id', 'N/A')
        logger.debug(f'Selecting model based on context (ID: {context_id}, Type: {context_type})')
        if available_models is None:
            available_models = list_available_models()
            if not available_models:
                logger.error('No available LLM models found in the registry. Cannot select a model.')
                return settings.PRIMARY_LLM
        preferred_model: Optional[str] = None
        for rule in self.selection_rules:
            try:
                if rule['condition'](context):
                    potential_model = rule.get('preferred_model')
                    if potential_model and potential_model in available_models:
                        preferred_model = potential_model
                        logger.info(f'Context matched rule. Preferred model selected: {preferred_model}')
                        break
                    elif potential_model:
                        logger.warning(f"Rule matched for preferred model '{potential_model}', but it's not in the available models list: {available_models}. Skipping rule.")
            except Exception as e:
                logger.warning(f'Error evaluating model selection rule condition: {e}', exc_info=True)
                continue
        selected_model: str
        if preferred_model:
            selected_model = preferred_model
        else:
            logger.debug('No specific rule matched or preferred model unavailable. Using default primary model.')
            if settings.PRIMARY_LLM in available_models:
                selected_model = settings.PRIMARY_LLM
            elif available_models:
                selected_model = available_models[0]
                logger.warning(f"Default primary model '{settings.PRIMARY_LLM}' is not available. Falling back to first available model: {selected_model}")
            else:
                logger.error('CRITICAL: No available models to select, even as a fallback!')
                selected_model = settings.PRIMARY_LLM
        logger.info(f'Context-based model selection complete. Selected model: {selected_model}')
        return selected_model
_selector_instance: Optional[ContextModelSelector] = None
_selector_lock = asyncio.Lock()

async def get_context_model_selector() -> ContextModelSelector:
    global _selector_instance
    if _selector_instance is not None:
        return _selector_instance
    async with _selector_lock:
        if _selector_instance is None:
            _selector_instance = ContextModelSelector()
            logger.info('Singleton ContextModelSelector instance created.')
    if _selector_instance is None:
        raise RuntimeError('Failed to create ContextModelSelector instance.')
    return _selector_instance
from typing import Any, Dict, List, Optional, Set, Union, cast
from functools import lru_cache
from src.config.settings import get_settings
from src.config.logger import get_logger
settings = get_settings()
logger = get_logger(__name__)
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {'gpt-4o': {'provider': 'openai', 'token_limit': 128000, 'supports_streaming': True, 'supports_function_calling': True, 'supports_json_mode': True, 'capabilities': ['general', 'reasoning', 'coding', 'math', 'vision'], 'description': 'Latest and most powerful OpenAI model with vision capabilities'}, 'gpt-4-turbo': {'provider': 'openai', 'token_limit': 128000, 'supports_streaming': True, 'supports_function_calling': True, 'supports_json_mode': True, 'capabilities': ['general', 'reasoning', 'coding', 'math'], 'description': 'Powerful model with large context window'}, 'gpt-4': {'provider': 'openai', 'token_limit': 8192, 'supports_streaming': True, 'supports_function_calling': True, 'supports_json_mode': True, 'capabilities': ['general', 'reasoning', 'coding', 'math'], 'description': 'Powerful reasoning model'}, 'gpt-4-32k': {'provider': 'openai', 'token_limit': 32768, 'supports_streaming': True, 'supports_function_calling': True, 'supports_json_mode': True, 'capabilities': ['general', 'reasoning', 'coding', 'math'], 'description': 'Powerful reasoning model with extended context window'}, 'gpt-3.5-turbo': {'provider': 'openai', 'token_limit': 16385, 'supports_streaming': True, 'supports_function_calling': True, 'supports_json_mode': True, 'capabilities': ['general', 'coding', 'math'], 'description': 'Fast and cost-effective model'}, 'claude-3-opus': {'provider': 'anthropic', 'token_limit': 200000, 'supports_streaming': True, 'supports_function_calling': False, 'supports_json_mode': False, 'capabilities': ['general', 'reasoning', 'coding', 'math', 'vision'], 'description': "Anthropic's most powerful model with vision capabilities"}, 'claude-3-sonnet': {'provider': 'anthropic', 'token_limit': 180000, 'supports_streaming': True, 'supports_function_calling': False, 'supports_json_mode': False, 'capabilities': ['general', 'reasoning', 'coding', 'math', 'vision'], 'description': 'Balance of intelligence and speed from Anthropic'}, 'claude-3-haiku': {'provider': 'anthropic', 'token_limit': 150000, 'supports_streaming': True, 'supports_function_calling': False, 'supports_json_mode': False, 'capabilities': ['general', 'coding', 'math', 'vision'], 'description': 'Fast and efficient model from Anthropic'}}

def register_model(model_name: str, provider: str, token_limit: int, capabilities: List[str], description: str, supports_streaming: bool=True, supports_function_calling: bool=False, supports_json_mode: bool=False, additional_properties: Optional[Dict[str, Any]]=None) -> None:
    global _MODEL_REGISTRY
    model_info = {'provider': provider, 'token_limit': token_limit, 'supports_streaming': supports_streaming, 'supports_function_calling': supports_function_calling, 'supports_json_mode': supports_json_mode, 'capabilities': capabilities, 'description': description}
    if additional_properties:
        model_info.update(additional_properties)
    if model_name in _MODEL_REGISTRY:
        logger.warning(f'Overriding existing model registration for: {model_name}')
    _MODEL_REGISTRY[model_name] = model_info
    logger.debug(f'Registered model: {model_name} (Provider: {provider})')

@lru_cache(maxsize=128)
def get_model_info(model_name: str) -> Dict[str, Any]:
    if model_name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name].copy()
    for registry_model, info in _MODEL_REGISTRY.items():
        if model_name.startswith(registry_model):
            logger.debug(f"Found model info for '{model_name}' using prefix '{registry_model}'")
            return info.copy()
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model_name)
    if provider:
        provider_defaults = {'openai': {'provider': 'openai', 'token_limit': 4096, 'supports_streaming': True, 'supports_function_calling': False, 'supports_json_mode': False, 'capabilities': ['general'], 'description': f'Unrecognized OpenAI model ({model_name}) - using defaults'}, 'anthropic': {'provider': 'anthropic', 'token_limit': 100000, 'supports_streaming': True, 'supports_function_calling': False, 'supports_json_mode': False, 'capabilities': ['general'], 'description': f'Unrecognized Anthropic model ({model_name}) - using defaults'}}
        if provider in provider_defaults:
            logger.warning(f"Model '{model_name}' not found in registry. Using default info for provider '{provider}'.")
            return provider_defaults[provider].copy()
    logger.warning(f'No information found for model: {model_name}')
    return {}

def get_token_limit(model_name: str) -> int:
    model_info = get_model_info(model_name)
    return model_info.get('token_limit', 4096)

def model_supports_feature(model_name: str, feature: str) -> bool:
    model_info = get_model_info(model_name)
    return model_info.get(f'supports_{feature}', False)

def model_has_capability(model_name: str, capability: str) -> bool:
    model_info = get_model_info(model_name)
    return capability in model_info.get('capabilities', [])

def list_available_models(provider: Optional[str]=None, min_token_limit: Optional[int]=None, required_capabilities: Optional[List[str]]=None, required_features: Optional[List[str]]=None) -> List[str]:
    result: List[str] = []
    for model_name, info in _MODEL_REGISTRY.items():
        if provider and info.get('provider') != provider:
            continue
        if min_token_limit and info.get('token_limit', 0) < min_token_limit:
            continue
        if required_capabilities:
            model_capabilities = set(info.get('capabilities', []))
            if not set(required_capabilities).issubset(model_capabilities):
                continue
        if required_features:
            model_features = {feature for feature in ['streaming', 'function_calling', 'json_mode'] if info.get(f'supports_{feature}', False)}
            if not set(required_features).issubset(model_features):
                continue
        result.append(model_name)
    return sorted(result)

def get_provider_models(provider: str) -> List[str]:
    return list_available_models(provider=provider)

def find_alternative_model(model_name: str, same_provider: bool=True) -> Optional[str]:
    original_info = get_model_info(model_name)
    if not original_info:
        logger.warning(f"Cannot find alternative model for '{model_name}': Original model info not found.")
        return None
    provider = original_info.get('provider')
    capabilities = original_info.get('capabilities', [])
    original_token_limit = original_info.get('token_limit', 0)
    candidates: List[Tuple[str, float]] = []
    for candidate_name, info in _MODEL_REGISTRY.items():
        if candidate_name == model_name:
            continue
        if same_provider and info.get('provider') != provider:
            continue
        score: float = 0.0
        if info.get('provider') == provider:
            score += 100.0
        candidate_token_limit = info.get('token_limit', 0)
        if original_token_limit > 0 and candidate_token_limit > 0:
            token_limit_ratio = min(candidate_token_limit, original_token_limit) / max(candidate_token_limit, original_token_limit)
            score += 50.0 * token_limit_ratio
        candidate_capabilities = set(info.get('capabilities', []))
        original_capabilities = set(capabilities)
        if original_capabilities:
            overlap_count = len(original_capabilities.intersection(candidate_capabilities))
            capability_overlap_ratio = overlap_count / len(original_capabilities)
            score += 30.0 * capability_overlap_ratio
        candidates.append((candidate_name, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    if candidates:
        alternative_model = candidates[0][0]
        logger.info(f"Found alternative model for '{model_name}': '{alternative_model}' (Score: {candidates[0][1]:.2f})")
        return alternative_model
    else:
        logger.warning(f"Could not find any suitable alternative model for '{model_name}' with given constraints.")
        return None

def initialize_models() -> None:
    logger.debug('Initializing model registry...')
    registered_count = 0
    for model in settings.ENABLED_MODELS_SET:
        if model not in _MODEL_REGISTRY:
            provider = settings.LLM_MODEL_PROVIDER_MAP.get(model)
            if provider:
                logger.warning(f"Model '{model}' found in settings but not in registry. Registering with default info.")
                register_model(model_name=model, provider=provider, token_limit=4096, capabilities=['general'], description=f"Model '{model}' from provider '{provider}' (auto-registered)")
                registered_count += 1
            else:
                logger.warning(f"Model '{model}' enabled in settings but provider mapping is missing. Skipping registration.")
    if registered_count > 0:
        logger.info(f'Auto-registered {registered_count} models found in settings but missing from registry.')
    logger.debug(f'Model registry initialization complete. Total models: {len(_MODEL_REGISTRY)}')
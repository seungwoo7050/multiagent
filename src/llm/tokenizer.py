import re
import hashlib
from functools import lru_cache
from typing import Dict, Optional, Union, List, Tuple, Any
import tiktoken
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import timed_metric, MEMORY_OPERATION_DURATION
settings = get_settings()
logger = get_logger(__name__)
_TOKENIZER_CACHE: Dict[str, Any] = {}
_TOKEN_COUNT_CACHE: Dict[str, int] = {}
_MAX_CACHE_SIZE: int = 10000
MODEL_TOKEN_LIMITS: Dict[str, int] = {'gpt-4': 8192, 'gpt-4-32k': 32768, 'gpt-4-turbo': 128000, 'gpt-4o': 128000, 'gpt-3.5-turbo': 16385, 'gpt-3.5-turbo-16k': 16385, 'text-davinci-003': 4097, 'text-davinci-002': 4097, 'davinci': 2049, 'claude-3-opus': 200000, 'claude-3-sonnet': 180000, 'claude-3-haiku': 150000, 'claude-2': 100000, 'claude-1': 100000}
MODEL_TO_ENCODING: Dict[str, str] = {'gpt-4': 'cl100k_base', 'gpt-4o': 'cl100k_base', 'gpt-4-turbo': 'cl100k_base', 'gpt-3.5-turbo': 'cl100k_base', 'text-embedding-ada-002': 'cl100k_base', 'text-davinci-003': 'p50k_base', 'text-davinci-002': 'p50k_base', 'davinci': 'p50k_base', 'openai-default': 'cl100k_base'}

def _get_cache_key(text: str, model: Optional[str]=None) -> str:
    if model:
        key_string = f'{model}:{text}'
    else:
        key_string = text
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

@lru_cache(maxsize=128)
def _get_tokenizer_for_model(model: str) -> tiktoken.Encoding:
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, 'unknown')
    if provider == 'openai':
        encoding_name = MODEL_TO_ENCODING.get('openai-default', 'cl100k_base')
        if model in MODEL_TO_ENCODING:
            encoding_name = MODEL_TO_ENCODING[model]
        else:
            for model_prefix, enc_name in MODEL_TO_ENCODING.items():
                if model.startswith(model_prefix):
                    encoding_name = enc_name
                    break
        try:
            tokenizer = tiktoken.get_encoding(encoding_name)
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to get tiktoken encoding '{encoding_name}' for model '{model}'. Falling back to 'cl100k_base'. Error: {e}")
            tokenizer = tiktoken.get_encoding('cl100k_base')
            return tokenizer
    elif provider == 'anthropic':
        logger.debug(f"Using 'cl100k_base' tokenizer as approximation for Anthropic model '{model}'.")
        return tiktoken.get_encoding('cl100k_base')
    else:
        logger.warning(f"Unknown LLM provider '{provider}' for model '{model}'. Using 'cl100k_base' tokenizer as default.")
        return tiktoken.get_encoding('cl100k_base')

def _approximate_token_count(text: str) -> int:
    char_count = len(text)
    approx_tokens = max(1, char_count // 4)
    logger.debug(f'Approximated token count for text (length {char_count}): {approx_tokens}')
    return approx_tokens

@timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'count_tokens'})
async def count_tokens(model: str, text: str) -> int:
    global _TOKEN_COUNT_CACHE
    cache_key = _get_cache_key(text, model)
    if cache_key in _TOKEN_COUNT_CACHE:
        return _TOKEN_COUNT_CACHE[cache_key]
    if len(_TOKEN_COUNT_CACHE) >= _MAX_CACHE_SIZE:
        keys_to_remove = list(_TOKEN_COUNT_CACHE.keys())[:_MAX_CACHE_SIZE // 2]
        for key in keys_to_remove:
            _TOKEN_COUNT_CACHE.pop(key, None)
        logger.debug(f'Token count cache reached max size ({_MAX_CACHE_SIZE}). Evicted {len(keys_to_remove)} items.')
    token_count: int
    try:
        tokenizer = _get_tokenizer_for_model(model)
        token_count = len(tokenizer.encode(text))
        logger.debug(f"Calculated token count for model '{model}': {token_count}")
    except Exception as e:
        logger.warning(f"Error counting tokens for model '{model}': {e}. Using approximation.", exc_info=True)
        token_count = _approximate_token_count(text)
    _TOKEN_COUNT_CACHE[cache_key] = token_count
    return token_count

def count_tokens_sync(model: str, text: str) -> int:
    global _TOKEN_COUNT_CACHE
    cache_key = _get_cache_key(text, model)
    if cache_key in _TOKEN_COUNT_CACHE:
        return _TOKEN_COUNT_CACHE[cache_key]
    if len(_TOKEN_COUNT_CACHE) >= _MAX_CACHE_SIZE:
        keys_to_remove = list(_TOKEN_COUNT_CACHE.keys())[:_MAX_CACHE_SIZE // 2]
        for key in keys_to_remove:
            _TOKEN_COUNT_CACHE.pop(key, None)
    token_count: int
    try:
        tokenizer = _get_tokenizer_for_model(model)
        token_count = len(tokenizer.encode(text))
    except Exception as e:
        token_count = _approximate_token_count(text)
    _TOKEN_COUNT_CACHE[cache_key] = token_count
    return token_count

def get_token_limit(model: str) -> int:
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]
    for model_prefix, token_limit in MODEL_TOKEN_LIMITS.items():
        if model.startswith(model_prefix):
            logger.debug(f"Using token limit {token_limit} based on prefix '{model_prefix}' for model '{model}'.")
            return token_limit
    try:
        from src.llm.models import get_model_info as get_model_info_from_models
        model_info = get_model_info_from_models(model)
        limit_from_models = model_info.get('token_limit')
        if limit_from_models:
            logger.debug(f"Using token limit {limit_from_models} from models.py for model '{model}'.")
            return limit_from_models
    except ImportError:
        logger.warning('Could not import src.llm.models to get token limit.')
    except Exception as e:
        logger.warning(f"Error getting token limit from models.py for '{model}': {e}")
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, 'unknown')
    default_limit = 4096
    if provider == 'openai':
        default_limit = 4096
    elif provider == 'anthropic':
        default_limit = 100000
    logger.warning(f"Could not determine specific token limit for model '{model}'. Returning default value: {default_limit}")
    return default_limit

def clear_token_cache() -> int:
    global _TOKEN_COUNT_CACHE
    count = len(_TOKEN_COUNT_CACHE)
    _TOKEN_COUNT_CACHE.clear()
    logger.info(f'Cleared token count cache ({count} items).')
    _get_tokenizer_for_model.cache_clear()
    logger.info('Cleared tokenizer LRU cache.')
    return count

def get_cache_metrics() -> Dict[str, int]:
    tokenizer_cache_info = _get_tokenizer_for_model.cache_info()
    return {'token_count_cache_size': len(_TOKEN_COUNT_CACHE), 'tokenizer_cache_size': tokenizer_cache_info.currsize, 'tokenizer_cache_hits': tokenizer_cache_info.hits, 'tokenizer_cache_misses': tokenizer_cache_info.misses, 'tokenizer_cache_maxsize': tokenizer_cache_info.maxsize}
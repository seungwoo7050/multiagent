"""
Token counting utilities for LLM operations.

This module provides tokenization functions for different LLM models,
with optimized performance through caching and efficient implementation.
"""

import re
import hashlib
from functools import lru_cache
from typing import Dict, Optional, Union, List, Tuple

import tiktoken
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import timed_metric, MEMORY_OPERATION_DURATION

settings = get_settings()
logger = get_logger(__name__)

# Cache for tokenizers to avoid repeated initialization
_TOKENIZER_CACHE = {}

# Cache for token counts to avoid repeated calculations
_TOKEN_COUNT_CACHE = {}

# Maximum size for the token count cache
_MAX_CACHE_SIZE = 10000

# Token limits for different models
MODEL_TOKEN_LIMITS = {
    # OpenAI models
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "davinci": 2049,
    
    # Anthropic models
    "claude-3-opus": 200000,
    "claude-3-sonnet": 180000,
    "claude-3-haiku": 150000,
    "claude-2": 100000,
    "claude-1": 100000,
}

# Map of model names to tokenizer encoding names
MODEL_TO_ENCODING = {
    # OpenAI models
    "gpt-4": "cl100k_base",
    "gpt-4o": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "davinci": "p50k_base",
    
    # Default for new OpenAI models
    "openai-default": "cl100k_base",
}

def _get_cache_key(text: str, model: Optional[str] = None) -> str:
    """Generate a cache key for token counting.
    
    Args:
        text: The text to count tokens for
        model: The model name (optional)
        
    Returns:
        str: Cache key as a hexadecimal string
    """
    # Include model in the key if provided
    if model:
        key_string = f"{model}:{text}"
    else:
        key_string = text
    
    # Use MD5 for speed (doesn't need cryptographic security)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()

@lru_cache(maxsize=128)
def _get_tokenizer_for_model(model: str):
    """Get the appropriate tokenizer for a model.
    
    Args:
        model: The model name
        
    Returns:
        Any: The tokenizer instance
    """
    global _TOKENIZER_CACHE
    
    # Check if we already have a tokenizer for this model
    if model in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model]
    
    # Get the provider from model
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, "unknown")
    
    # Handle different providers
    if provider == "openai":
        # Determine encoding name
        encoding_name = "cl100k_base"  # Default for newer models
        
        if model in MODEL_TO_ENCODING:
            encoding_name = MODEL_TO_ENCODING[model]
        else:
            # Try matching by prefix
            for model_prefix, enc_name in MODEL_TO_ENCODING.items():
                if model.startswith(model_prefix):
                    encoding_name = enc_name
                    break
        
        try:
            # Get the tokenizer
            tokenizer = tiktoken.get_encoding(encoding_name)
            _TOKENIZER_CACHE[model] = tokenizer
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to get tokenizer for {model}: {str(e)}")
            # Fallback to cl100k_base which is used by most recent models
            tokenizer = tiktoken.get_encoding("cl100k_base")
            _TOKENIZER_CACHE[model] = tokenizer
            return tokenizer
    
    elif provider == "anthropic":
        # Use tiktoken with cl100k_base for Anthropic models as an approximation
        # Note: This is an approximation - Anthropic doesn't publish their tokenizer
        return tiktoken.get_encoding("cl100k_base")
    
    else:
        # For unknown providers, use cl100k_base as a reasonable default
        return tiktoken.get_encoding("cl100k_base")

def _approximate_token_count(text: str) -> int:
    """Approximate token count using a simple heuristic.
    
    This is used as a fallback when provider-specific tokenizers aren't available.
    It's less accurate but better than nothing.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        int: Approximate token count
    """
    # Split into words and punctuation
    words = re.findall(r'\w+|[^\w\s]|\s+', text)
    # Roughly 4 characters per token
    return max(1, len(text) // 4)

@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "count_tokens"})
async def count_tokens(model: str, text: str) -> int:
    """Count the number of tokens in the text for the specific model.
    
    Args:
        model: The model name
        text: The text to count tokens for
        
    Returns:
        int: The number of tokens
    """
    global _TOKEN_COUNT_CACHE
    
    # Generate cache key
    cache_key = _get_cache_key(text, model)
    
    # Check cache
    if cache_key in _TOKEN_COUNT_CACHE:
        return _TOKEN_COUNT_CACHE[cache_key]
    
    # Limit cache size
    if len(_TOKEN_COUNT_CACHE) >= _MAX_CACHE_SIZE:
        # Simple LRU implementation: clear half the cache
        cache_keys = list(_TOKEN_COUNT_CACHE.keys())
        for key in cache_keys[:_MAX_CACHE_SIZE // 2]:
            _TOKEN_COUNT_CACHE.pop(key, None)
    
    # Get provider from model
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, "unknown")
    
    try:
        # Count tokens using provider-specific tokenizer
        if provider == "openai":
            tokenizer = _get_tokenizer_for_model(model)
            token_count = len(tokenizer.encode(text))
        elif provider == "anthropic":
            # Approximate for Anthropic using cl100k_base
            tokenizer = _get_tokenizer_for_model(model)
            token_count = len(tokenizer.encode(text))
        else:
            # Fallback to approximation
            token_count = _approximate_token_count(text)
        
        # Cache the result
        _TOKEN_COUNT_CACHE[cache_key] = token_count
        return token_count
    
    except Exception as e:
        logger.warning(f"Error counting tokens for model {model}: {str(e)}")
        # Fallback to approximation
        token_count = _approximate_token_count(text)
        _TOKEN_COUNT_CACHE[cache_key] = token_count
        return token_count

def count_tokens_sync(model: str, text: str) -> int:
    """Synchronous version of count_tokens.
    
    Args:
        model: The model name
        text: The text to count tokens for
        
    Returns:
        int: The number of tokens
    """
    # This implementation uses the same approach but in a synchronous way
    # No need to duplicate all the async logic since token counting is CPU-bound
    
    global _TOKEN_COUNT_CACHE
    
    # Generate cache key
    cache_key = _get_cache_key(text, model)
    
    # Check cache
    if cache_key in _TOKEN_COUNT_CACHE:
        return _TOKEN_COUNT_CACHE[cache_key]
    
    # Limit cache size
    if len(_TOKEN_COUNT_CACHE) >= _MAX_CACHE_SIZE:
        # Simple LRU implementation: clear half the cache
        cache_keys = list(_TOKEN_COUNT_CACHE.keys())
        for key in cache_keys[:_MAX_CACHE_SIZE // 2]:
            _TOKEN_COUNT_CACHE.pop(key, None)
    
    # Get provider from model
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, "unknown")
    
    try:
        # Count tokens using provider-specific tokenizer
        if provider == "openai":
            tokenizer = _get_tokenizer_for_model(model)
            token_count = len(tokenizer.encode(text))
        elif provider == "anthropic":
            # Approximate for Anthropic using cl100k_base
            tokenizer = _get_tokenizer_for_model(model)
            token_count = len(tokenizer.encode(text))
        else:
            # Fallback to approximation
            token_count = _approximate_token_count(text)
        
        # Cache the result
        _TOKEN_COUNT_CACHE[cache_key] = token_count
        return token_count
    
    except Exception as e:
        logger.warning(f"Error counting tokens for model {model}: {str(e)}")
        # Fallback to approximation
        token_count = _approximate_token_count(text)
        _TOKEN_COUNT_CACHE[cache_key] = token_count
        return token_count

def get_token_limit(model: str) -> int:
    """Get the token limit for a given model.
    
    Args:
        model: The model name
        
    Returns:
        int: The token limit (context window size)
    """
    # Check exact match
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]
    
    # Try to match by prefix
    for model_prefix, token_limit in MODEL_TOKEN_LIMITS.items():
        if model.startswith(model_prefix):
            return token_limit
    
    # Default fallbacks based on provider
    provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, "unknown")
    
    if provider == "openai":
        return 4096  # Conservative default for OpenAI
    elif provider == "anthropic":
        return 100000  # Conservative default for Anthropic
    else:
        return 4096  # Default fallback

def clear_token_cache() -> int:
    """Clear the token counting cache.
    
    Returns:
        int: Number of entries cleared
    """
    global _TOKEN_COUNT_CACHE
    count = len(_TOKEN_COUNT_CACHE)
    _TOKEN_COUNT_CACHE.clear()
    return count

def get_cache_metrics() -> Dict[str, int]:
    """Get metrics about the token cache.
    
    Returns:
        Dict[str, int]: Cache metrics
    """
    return {
        "token_cache_size": len(_TOKEN_COUNT_CACHE),
        "tokenizer_cache_size": len(_TOKENIZER_CACHE),
    }
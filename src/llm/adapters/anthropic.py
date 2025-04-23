"""
Anthropic adapter for LLM operations.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import re

from src.llm.base import BaseLLMAdapter
from src.llm.connection_pool import get_connection_pool
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import LLMError, ErrorCode
from src.utils.timing import async_timed

settings = get_settings()
logger = get_logger(__name__)

# Anthropic constants
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODELS = {
    "claude-3-opus": {"max_tokens": 200000},
    "claude-3-sonnet": {"max_tokens": 180000},
    "claude-3-haiku": {"max_tokens": 150000},
    "claude-2": {"max_tokens": 100000},
    "claude-1": {"max_tokens": 100000},
}


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude API."""
    
    def __init__(
        self,
        model: str,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        cache_enabled: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize the Anthropic adapter.
        
        Args:
            model: The model identifier (e.g., "claude-3-opus", "claude-3-sonnet")
            provider: Should be "anthropic" (included for consistency)
            api_key: Anthropic API key
            api_base: Anthropic API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            cache_enabled: Whether to use the response cache
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
        """
        super().__init__(
            model=model,
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
            cache_enabled=cache_enabled,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Set Anthropic-specific attributes
        self.api_base = api_base or ANTHROPIC_API_URL
        self.api_version = "2023-06-01"  # Use a stable API version
        
        # Default max tokens if not specified
        if not self.max_tokens:
            self.max_tokens = 1024
    
    async def _initialize(self) -> bool:
        """Initialize the Anthropic adapter.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Verify API key
            if not self.api_key:
                provider_config = settings.LLM_PROVIDERS_CONFIG.get("anthropic", {})
                self.api_key = provider_config.get("api_key")
                if not self.api_key:
                    raise LLMError(
                        code=ErrorCode.LLM_PROVIDER_ERROR,
                        message="No API key provided for Anthropic adapter"
                    )
            
            # Test connection by getting a client
            client = await self._get_client()
            if not client:
                return False
            
            logger.info(f"Successfully initialized Anthropic adapter for model {self.model}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic adapter: {str(e)}")
            return False
    
    async def _get_client(self) -> aiohttp.ClientSession:
        """Get an HTTP client session for Anthropic API.
        
        Returns:
            aiohttp.ClientSession: HTTP client session
        """
        # Get a connection from the pool
        session = await get_connection_pool("anthropic")
        return session
    
    async def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.
        
        Args:
            text: The text to count tokens for
        
        Returns:
            int: The number of tokens
        """
        # Anthropic doesn't provide a public tokenizer, so use a simple approximation
        # This is a very rough approximation based on Claude's tokenization characteristics
        # For production use, consider using the tiktoken cl100k_base model as a rough approximation
        
        # Import the tokenizer function to avoid circular imports
        from src.llm.tokenizer import count_tokens
        return await count_tokens(self.model, text)
    
    @async_timed("anthropic_generate_text")
    async def _generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from Anthropic API.
        
        Args:
            prompt: The prompt to send to the API
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            stop_sequences: List of stop sequences
            **kwargs: Additional Anthropic-specific parameters
        
        Returns:
            Dict[str, Any]: Response containing the generated text and metadata
        """
        # Ensure we're initialized
        if not self.initialized:
            await self.ensure_initialized()
        
        # Get client session
        session = await self._get_client()
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
        }
        
        # Format messages according to Anthropic's API
        if isinstance(prompt, str):
            # Convert text prompt to messages format
            payload["messages"] = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            # Already in messages format
            payload["messages"] = prompt
        else:
            raise ValueError(f"Unsupported prompt format for Anthropic: {type(prompt)}")
        
        # Add stop sequences if provided
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences
        
        # Add additional parameters
        for key, value in kwargs.items():
            if value is not None and key not in payload:
                payload[key] = value
        
        # Count prompt tokens for logging (approximate)
        prompt_text = ""
        for message in payload["messages"]:
            prompt_text += message.get("content", "")
        prompt_tokens = await self._count_tokens(prompt_text)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }
        
        try:
            # Make the request
            start_time = time.time()
            async with session.post(
                self.api_base,
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                response_json = await response.json()
                request_time = time.time() - start_time
                
                if response.status != 200:
                    # Handle error response
                    error_msg = response_json.get("error", {}).get("message", "Unknown error")
                    error_type = response_json.get("error", {}).get("type", "unknown")
                    
                    raise LLMError(
                        code=ErrorCode.LLM_API_ERROR,
                        message=f"Anthropic API error: {error_msg}",
                        details={
                            "status_code": response.status,
                            "error_type": error_type,
                            "model": self.model,
                        }
                    )
                
                # Process response
                completion_text = response_json.get("content", [{}])[0].get("text", "")
                completion_tokens = await self._count_tokens(completion_text)
                
                # Standardize response format to match our internal format
                result = {
                    "id": response_json.get("id", ""),
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": response_json.get("model", self.model),
                    "choices": [{
                        "text": completion_text,
                        "index": 0,
                        "finish_reason": response_json.get("stop_reason", "stop"),
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "request_time": request_time,
                }
                
                return result
        
        except Exception as e:
            # Convert to LLMError if not already
            if not isinstance(e, LLMError):
                error = LLMError(
                    code=ErrorCode.LLM_API_ERROR,
                    message=f"Error calling Anthropic API: {str(e)}",
                    original_error=e,
                    model=self.model,
                    provider="anthropic",
                )
                raise error
            raise
    
    def get_token_limit(self) -> int:
        """Get the token limit for this model.
        
        Returns:
            int: Maximum context window size (prompt + completion)
        """
        # Check in our model registry
        if self.model in ANTHROPIC_MODELS:
            return ANTHROPIC_MODELS[self.model]["max_tokens"]
        
        # Try to match by prefix
        for model_prefix, info in ANTHROPIC_MODELS.items():
            if self.model.startswith(model_prefix):
                return info["max_tokens"]
        
        # Default fallback for unknown models
        return 100000  # Conservative default for newer models
    
    async def close(self) -> None:
        """Close the adapter and release resources."""
        # No specific cleanup needed as connections are managed by the pool
        self.initialized = False
        logger.debug(f"Closed Anthropic adapter for {self.model}")
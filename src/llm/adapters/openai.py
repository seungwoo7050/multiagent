"""
OpenAI adapter for LLM operations.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import tiktoken

from src.llm.base import BaseLLMAdapter
from src.llm.connection_pool import get_connection_pool
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import LLMError, ErrorCode
from src.utils.timing import async_timed

settings = get_settings()
logger = get_logger(__name__)

# Cache for tokenizers to avoid repeated initialization
_TOKENIZER_CACHE = {}


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI API."""
    
    def __init__(
        self,
        model: str,
        provider: str = "openai",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        cache_enabled: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize the OpenAI adapter.
        
        Args:
            model: The model identifier (e.g., "gpt-4o", "gpt-3.5-turbo")
            provider: Should be "openai" (included for consistency)
            api_key: OpenAI API key
            api_base: OpenAI API base URL
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
        
        # Set OpenAI-specific attributes
        self.api_base = api_base or "https://api.openai.com/v1"
        self.api_version = "2023-05-15"  # Use a stable API version
        self.tokenizer = None
        
        # Default max tokens if not specified
        if not self.max_tokens:
            self.max_tokens = 1024
    
    # src/llm/adapters/openai.py

    async def _initialize(self) -> bool:
        """Initialize the OpenAI adapter.

        Returns:
            bool: True if initialization was successful
        """
        # --- API 키 확인 로직 ---
        try:
            # Verify API key
            if not self.api_key:
                # 설정에서 API 키 가져오기 시도
                provider_config = settings.LLM_PROVIDERS_CONFIG.get("openai", {})
                self.api_key = provider_config.get("api_key")
                # 여전히 API 키가 없으면 LLMError 발생
                if not self.api_key:
                    # 이 LLMError는 _initialize 함수 내에서 잡히지 않고 그대로 전달되어야 함
                    raise LLMError(
                        code=ErrorCode.LLM_PROVIDER_ERROR,
                        message="No API key provided for OpenAI adapter"
                    ) # [ 수정: 이 예외는 아래 except 블록에서 잡지 않도록 함 ]

            # --- 나머지 초기화 로직 (토크나이저, 클라이언트) ---
            # Initialize tokenizer
            self.tokenizer = await self._get_tokenizer()

            # Test connection by getting a client
            client = await self._get_client()
            if not client:
                # 클라이언트 가져오기 실패 시 False 반환 (기존 로직 유지)
                logger.error("Failed to get HTTP client during OpenAI adapter initialization.")
                return False

            logger.info(f"Successfully initialized OpenAI adapter for model {self.model}")
            return True

        # --- 포괄적인 예외 처리 (LLMError는 제외) ---
        except LLMError as lle:
            # API 키 관련 LLMError는 다시 발생시킴
            if lle.code == ErrorCode.LLM_PROVIDER_ERROR and "No API key provided" in lle.message:
                raise lle
            # 다른 LLMError는 로그 남기고 False 반환 (혹은 필요에 따라 처리)
            logger.error(f"LLMError during OpenAI adapter initialization: {lle.code} - {lle.message}", extra=lle.to_dict())
            return False
        except Exception as e:
            # LLMError가 아닌 다른 모든 예외 처리
            logger.error(f"Failed to initialize OpenAI adapter: {str(e)}", exc_info=True) # exc_info=True 추가하여 스택 트레이스 로깅
            # 여기서는 False를 반환하여 초기화 실패를 알림
            return False
    
    async def _get_client(self) -> aiohttp.ClientSession:
        """Get an HTTP client session for OpenAI API.
        
        Returns:
            aiohttp.ClientSession: HTTP client session
        """
        # Get a connection from the pool
        session = await get_connection_pool("openai")
        return session
    
    async def _get_tokenizer(self) -> Any:
        """Get a tokenizer for the current model.
        
        Returns:
            Any: The tokenizer instance
        """
        global _TOKENIZER_CACHE
        
        # Check if we already have a tokenizer for this model
        if self.model in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[self.model]
        
        # Map model names to encoding names
        encoding_name = "cl100k_base"  # Default for newer models
        
        # Model-specific encodings
        model_to_encoding = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "text-embedding-ada-002": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "text-davinci-002": "p50k_base",
            "davinci": "p50k_base",
        }
        
        # Get encoding name based on model or model prefix
        if self.model in model_to_encoding:
            encoding_name = model_to_encoding[self.model]
        else:
            # Try matching by prefix
            for model_prefix, enc_name in model_to_encoding.items():
                if self.model.startswith(model_prefix):
                    encoding_name = enc_name
                    break
        
        try:
            # Get the tokenizer
            tokenizer = tiktoken.get_encoding(encoding_name)
            _TOKENIZER_CACHE[self.model] = tokenizer
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to get tokenizer for {self.model}: {str(e)}")
            # Fallback to cl100k_base which is used by most recent models
            tokenizer = tiktoken.get_encoding("cl100k_base")
            _TOKENIZER_CACHE[self.model] = tokenizer
            return tokenizer
    
    async def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.
        
        Args:
            text: The text to count tokens for
        
        Returns:
            int: The number of tokens
        """
        if not self.tokenizer:
            self.tokenizer = await self._get_tokenizer()
        
        return len(self.tokenizer.encode(text))
    
    @async_timed("openai_generate_text")
    async def _generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            stop_sequences: List of stop sequences
            **kwargs: Additional OpenAI-specific parameters
        
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
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        # Add additional parameters
        for key, value in kwargs.items():
            if value is not None and key not in payload:
                payload[key] = value
        
        # Count prompt tokens for logging
        prompt_tokens = await self._count_tokens(prompt)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Add API version if specified
        if self.api_version:
            headers["OpenAI-Version"] = self.api_version
        
        # Prepare URL
        url = f"{self.api_base}/completions"
        
        # Special handling for chat models
        is_chat_model = (
            self.model.startswith("gpt-3.5-turbo") or 
            self.model.startswith("gpt-4") or
            "turbo" in self.model
        )
        
        if is_chat_model:
            url = f"{self.api_base}/chat/completions"
            # Convert prompt to chat format if it's not already
            if not isinstance(prompt, list) and "messages" not in payload:
                payload.pop("prompt", None)
                payload["messages"] = [{"role": "user", "content": prompt}]
        
        try:
            # Make the request
            start_time = time.time()
            async with session.post(
                url,
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
                        message=f"OpenAI API error: {error_msg}",
                        details={
                            "status_code": response.status,
                            "error_type": error_type,
                            "model": self.model,
                        }
                    )
                
                # Process response
                if is_chat_model:
                    # Chat completion format
                    result = {
                        "id": response_json.get("id", ""),
                        "object": response_json.get("object", ""),
                        "created": response_json.get("created", 0),
                        "model": response_json.get("model", self.model),
                        "choices": [{
                            "text": choice["message"]["content"],
                            "index": choice.get("index", 0),
                            "finish_reason": choice.get("finish_reason", ""),
                        } for choice in response_json.get("choices", [])],
                        "usage": response_json.get("usage", {}),
                        "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens", prompt_tokens),
                        "completion_tokens": response_json.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": response_json.get("usage", {}).get("total_tokens", 0),
                        "request_time": request_time,
                    }
                else:
                    # Standard completion format
                    result = {
                        "id": response_json.get("id", ""),
                        "object": response_json.get("object", ""),
                        "created": response_json.get("created", 0),
                        "model": response_json.get("model", self.model),
                        "choices": response_json.get("choices", []),
                        "usage": response_json.get("usage", {}),
                        "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens", prompt_tokens),
                        "completion_tokens": response_json.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": response_json.get("usage", {}).get("total_tokens", 0),
                        "request_time": request_time,
                    }
                
                return result
        
        except Exception as e:
            # Convert to LLMError if not already
            if not isinstance(e, LLMError):
                error = LLMError(
                    code=ErrorCode.LLM_API_ERROR,
                    message=f"Error calling OpenAI API: {str(e)}",
                    original_error=e,
                    model=self.model,
                    provider="openai",
                )
                raise error
            raise
    
    def get_token_limit(self) -> int:
        """Get the token limit for this model.
        
        Returns:
            int: Maximum context window size (prompt + completion)
        """
        # Token limits for different models
        model_token_limits = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "davinci": 2049,
        }
        
        # Find exact match or prefix match
        if self.model in model_token_limits:
            return model_token_limits[self.model]
        
        # Try to match by prefix
        for model_prefix, token_limit in model_token_limits.items():
            if self.model.startswith(model_prefix):
                return token_limit
        
        # Default fallback
        return 4097
    
    async def close(self) -> None:
        """Close the adapter and release resources."""
        # No specific cleanup needed as connections are managed by the pool
        self.initialized = False
        logger.debug(f"Closed OpenAI adapter for {self.model}")
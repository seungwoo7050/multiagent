import abc
import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic

from src.config.metrics import (
    LLM_REQUESTS_TOTAL,
    LLM_REQUEST_DURATION,
    LLM_TOKEN_USAGE,
    LLM_ERRORS_TOTAL,
    track_llm_request,
    track_llm_response,
    track_llm_error
)
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import LLMError, ErrorCode
from src.utils.timing import async_timed
from src.llm.cache import get_cache, cache_result 
from src.llm.tokenizer import count_tokens

# Import these at function level to avoid circular imports
# from src.llm.cache import get_cache, cache_result
# from src.llm.tokenizer import count_tokens

settings = get_settings()
logger = get_logger(__name__)

T = TypeVar('T')

class BaseLLMAdapter(abc.ABC):
    """
    Abstract base class for LLM adapters.
    
    This class defines the interface that all LLM adapters must implement
    and provides common functionality for all adapters.
    """
    
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        cache_enabled: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize the LLM adapter.
        
        Args:
            model: The model identifier (e.g., "gpt-4o", "claude-3-opus")
            provider: The provider name (e.g., "openai", "anthropic")
            api_key: API key for the provider
            api_base: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            cache_enabled: Whether to use the response cache
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
        """
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout or settings.REQUEST_TIMEOUT
        self.max_retries = max_retries or settings.LLM_RETRY_MAX_ATTEMPTS
        self.cache_enabled = cache_enabled
        
        # LLM generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature or 0.7
        self.top_p = top_p or 1.0
        
        # Track current state
        self.initialized = False
        self._client = None
        
        # Metrics
        self.request_count = 0
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}
        self.error_count = 0
        self.average_latency = 0.0
        
        logger.debug(f"Initialized {self.__class__.__name__} for {model} ({provider})")
    
    @abc.abstractmethod
    async def _initialize(self) -> bool:
        """Initialize the adapter, establishing connections and resources.
        
        This method should be implemented by subclasses to handle provider-specific
        initialization logic, like creating clients or setting up API connections.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def _get_client(self) -> Any:
        """Get the client instance for this adapter.
        
        Returns:
            Any: The client instance for the specific provider
        """
        pass
    
    @abc.abstractmethod
    async def _generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            stop_sequences: List of stop sequences
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Dict[str, Any]: Response containing the generated text and metadata
        """
        pass
    
    @abc.abstractmethod
    async def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text.
        
        Args:
            text: The text to count tokens for
        
        Returns:
            int: The number of tokens
        """
        pass
    
    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate a deterministic cache key for the prompt and parameters.
        
        Args:
            prompt: The prompt text
            params: Generation parameters
        
        Returns:
            str: Cache key as a hexadecimal string
        """
        # Create a dictionary with the prompt and all parameters
        cache_dict = {
            "model": self.model,
            "provider": self.provider,
            "prompt": prompt,
            "max_tokens": params.get("max_tokens", self.max_tokens),
            "temperature": params.get("temperature", self.temperature),
            "top_p": params.get("top_p", self.top_p),
            "stop_sequences": params.get("stop_sequences", []),
        }
        
        # Add other parameters
        for k, v in params.items():
            if k not in cache_dict and v is not None:
                cache_dict[k] = v
        
        # Convert to a stable string and hash
        stable_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(stable_str.encode()).hexdigest()
    
    def _log_request_metrics(
        self, 
        prompt_tokens: int, 
        completion_tokens: int, 
        latency: float,
        is_cached: bool = False
    ) -> None:
        """Log request metrics.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            latency: Request latency in seconds
            is_cached: Whether the response was from cache
        """
        # Update internal metrics
        self.request_count += 1
        self.token_usage["prompt"] += prompt_tokens
        self.token_usage["completion"] += completion_tokens
        self.token_usage["total"] += prompt_tokens + completion_tokens
        
        # Update rolling average latency
        self.average_latency = (
            (self.average_latency * (self.request_count - 1) + latency) / 
            self.request_count
        )
        
        # Send to Prometheus metrics if not cached
        if not is_cached:
            track_llm_request(self.model, self.provider)
            track_llm_response(
                self.model, 
                self.provider, 
                latency,
                prompt_tokens, 
                completion_tokens
            )
    
    def _log_error_metrics(self, error_type: str) -> None:
        """Log error metrics.
        
        Args:
            error_type: Type of error
        """
        self.error_count += 1
        track_llm_error(self.model, self.provider, error_type)
    
    async def ensure_initialized(self) -> bool:
        """Ensure the adapter is initialized.
        
        Returns:
            bool: True if initialization was successful
        """
        if not self.initialized:
            self.initialized = await self._initialize()
        return self.initialized
    
    async def tokenize(self, text: str) -> int:
        """Count tokens in the given text.
        
        Args:
            text: The text to count tokens for
        
        Returns:
            int: The number of tokens
        """
        return await self._count_tokens(text)
    
    @async_timed("llm_request")
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        retry_on_failure: bool = True,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from the LLM.
        
        This is the main method for generating text from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            stop_sequences: List of stop sequences
            retry_on_failure: Whether to retry on failure
            use_cache: Whether to use the cache (overrides instance setting)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Dict[str, Any]: Response containing the generated text and metadata
        """
        # from src.llm.cache import get_cache, cache_result
        # from src.llm.tokenizer import count_tokens
        
        start_time = time.time()
        
        # Ensure adapter is initialized
        await self.ensure_initialized()
        
        # Set up generation parameters
        params = {
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "stop_sequences": stop_sequences or [],
            **kwargs
        }
        
        # Determine cache usage
        use_cache_for_request = self.cache_enabled if use_cache is None else use_cache
        
        # Try to get result from cache
        if use_cache_for_request:
            cache_key = self._get_cache_key(prompt, params)
            cache = await get_cache()
            cached_result = await cache.get(cache_key)
            
            if cached_result:
                # Count tokens for metrics
                prompt_tokens = await count_tokens(self.model, prompt)
                completion_tokens = await count_tokens(
                    self.model, 
                    cached_result["choices"][0]["text"]
                )
                
                # Log metrics
                end_time = time.time()
                latency = end_time - start_time
                self._log_request_metrics(
                    prompt_tokens, 
                    completion_tokens, 
                    latency,
                    is_cached=True
                )
                
                # Return cached result
                return cached_result
        
        try:
            # Generate text
            response = await self._generate_text(
                prompt=prompt,
                **params
            )
            
            # Cache the result if enabled
            if use_cache_for_request:
                cache_key = self._get_cache_key(prompt, params)
                await cache_result(cache_key, response)
            
            # Log metrics
            end_time = time.time()
            latency = end_time - start_time
            
            if "prompt_tokens" in response and "completion_tokens" in response:
                prompt_tokens = response["prompt_tokens"]
                completion_tokens = response["completion_tokens"]
            else:
                # If tokens not returned by API, estimate them
                prompt_tokens = await count_tokens(self.model, prompt)
                completion_tokens = await count_tokens(
                    self.model, 
                    response["choices"][0]["text"]
                )
            
            self._log_request_metrics(prompt_tokens, completion_tokens, latency)
            
            return response
        
        except Exception as e:
            # Log error
            error_type = type(e).__name__
            self._log_error_metrics(error_type)
            
            # Wrap in LLMError
            llm_error = LLMError(
                code=ErrorCode.LLM_API_ERROR,
                message=f"Error generating text from {self.provider} ({self.model}): {str(e)}",
                original_error=e,
                model=self.model,
                provider=self.provider,
            )
            
            # Log the error
            llm_error.log_error(logger)
            
            # Re-raise for handling by caller
            raise llm_error
    
    def generate_sync(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        retry_on_failure: bool = True,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronous version of generate.
        
        This is a convenience wrapper around the async generate method.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            stop_sequences: List of stop sequences
            retry_on_failure: Whether to retry on failure
            use_cache: Whether to use the cache (overrides instance setting)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Dict[str, Any]: Response containing the generated text and metadata
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                retry_on_failure=retry_on_failure,
                use_cache=use_cache,
                **kwargs
            )
        )
    
    def get_token_limit(self) -> int:
        """Get the token limit for this model.
        
        Returns:
            int: Maximum tokens supported by the model
        """
        # To be implemented by each adapter based on model capabilities
        return 4096  # Default fallback
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for this adapter.
        
        Returns:
            Dict[str, Any]: Metrics dictionary
        """
        return {
            "model": self.model,
            "provider": self.provider,
            "request_count": self.request_count,
            "token_usage": self.token_usage,
            "error_count": self.error_count,
            "average_latency": self.average_latency,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the LLM service.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            start_time = time.time()
            # Ensure adapter is initialized
            initialized = await self.ensure_initialized()
            if not initialized:
                return {
                    "status": "error",
                    "message": "Failed to initialize adapter",
                    "latency": time.time() - start_time,
                }
            
            # Get client to verify connection
            client = await self._get_client()
            if client is None:
                return {
                    "status": "error",
                    "message": "Failed to get client",
                    "latency": time.time() - start_time,
                }
            
            return {
                "status": "ok",
                "message": f"{self.provider} ({self.model}) is operational",
                "latency": time.time() - start_time,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "latency": time.time() - start_time,
            }
    
    async def close(self) -> None:
        """Close the adapter and release resources."""
        # To be implemented by each adapter
        self.initialized = False
        self._client = None
        logger.debug(f"Closed {self.__class__.__name__} for {self.model} ({self.provider})")
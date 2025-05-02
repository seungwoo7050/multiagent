import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp

from src.config.errors import ErrorCode, LLMError
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.connection_pool import get_connection_pool
from src.utils.timing import async_timed

settings = get_settings()
logger = get_logger(__name__)

# Gemini API constants
GEMINI_API_VERSION = "v1"
GEMINI_DEFAULT_URL = "https://generativelanguage.googleapis.com"
GEMINI_MODELS = {
    'gemini-pro': {'max_tokens': 32768},
    'gemini-pro-vision': {'max_tokens': 32768},
    'gemini-ultra': {'max_tokens': 32768},
    'gemini-1.5-pro': {'max_tokens': 1048576},  # 1M token context
    'gemini-1.5-flash': {'max_tokens': 1048576}  # 1M token context
}


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google's Gemini models.
    
    Implements the BaseLLMAdapter interface for Gemini model API access.
    """

    def __init__(self, 
                 model: str, 
                 provider: str = 'gemini',
                 api_key: Optional[str] = None, 
                 api_base: Optional[str] = None,
                 timeout: Optional[float] = None, 
                 max_retries: Optional[int] = None,
                 cache_enabled: bool = True, 
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, 
                 top_p: Optional[float] = None):
        """
        Initialize the Gemini adapter.
        
        Args:
            model: Gemini model name to use
            provider: Provider name, defaults to 'gemini'
            api_key: Google API key
            api_base: API endpoint base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_enabled: Whether to enable response caching
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
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
            top_p=top_p
        )
        
        # Ensure the model name follows Gemini's format
        if not model.startswith('gemini-'):
            logger.warning(f"Model name '{model}' doesn't follow Gemini naming convention")
            
        self.api_base = api_base or GEMINI_DEFAULT_URL
        
        # Default max_tokens if not provided
        if self.max_tokens is None:
            self.max_tokens = 1024
            
        logger.debug(f"GeminiAdapter initialized for model '{self.model}' with API base '{self.api_base}'")

    async def _initialize(self) -> bool:
        """Initialize the adapter by verifying API key and connectivity."""
        try:
            if not self.api_key:
                provider_config = settings.LLM_PROVIDERS_CONFIG.get('gemini', {})
                self.api_key = provider_config.get('api_key')
                
                if not self.api_key:
                    raise LLMError(
                        code=ErrorCode.LLM_PROVIDER_ERROR,
                        message='No API key provided for Gemini adapter. Set GEMINI_API_KEY environment variable or provide it during initialization.'
                    )
                    
            # Test API connectivity
            client = await self._get_client()
            if not client:
                logger.error('Failed to get HTTP client session during Gemini adapter initialization.')
                return False
                
            logger.info(f'Successfully initialized Gemini adapter for model {self.model}')
            return True
            
        except LLMError as lle:
            if lle.code == ErrorCode.LLM_PROVIDER_ERROR and 'No API key provided' in lle.message:
                raise lle
                
            logger.error(f'LLMError during Gemini adapter initialization: {lle.code} - {lle.message}', 
                        extra=lle.to_dict())
            return False
            
        except Exception as e:
            logger.error(f'Failed to initialize Gemini adapter for model {self.model}: {str(e)}', 
                       exc_info=True)
            return False

    async def _get_client(self) -> aiohttp.ClientSession:
        """Get an HTTP client session from the connection pool."""
        session = await get_connection_pool('gemini')
        return session

    async def _count_tokens(self, text: str) -> int:
        """Count tokens for the input text."""
        from src.llm.tokenizer import count_tokens as global_count_tokens
        
        try:
            return await global_count_tokens(self.model, text)
        except Exception as e:
            logger.error(f'Error counting tokens for Gemini model {self.model}, using approximation: {e}', 
                       exc_info=True)
            # Fallback approximation (4 chars per token is a rough estimate)
            return len(text) // 4

    @async_timed('gemini_generate_text')
    async def _generate_text(self, 
                           prompt: Union[str, List[Dict[str, str]]],
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           stop_sequences: Optional[List[str]] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Generate text using the Gemini API.
        
        Args:
            prompt: Input prompt as string or chat message list
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Sequences at which to stop generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dict[str, Any]: Response with generated text and metadata
            
        Raises:
            LLMError: If the API request fails
        """
        session: aiohttp.ClientSession = await self._get_client()
        
        # Select the right endpoint based on model
        if self.model == 'gemini-pro-vision':
            endpoint = f"{self.api_base}/{GEMINI_API_VERSION}/models/{self.model}:generateContent"
        else:
            endpoint = f"{self.api_base}/{GEMINI_API_VERSION}/models/{self.model}:generateContent"
        
        # Prepare the request payload
        payload: Dict[str, Any] = {
            "generationConfig": {
                "maxOutputTokens": max_tokens if max_tokens is not None else self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "topP": top_p if top_p is not None else self.top_p,
            }
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            payload["generationConfig"]["stopSequences"] = stop_sequences
        
        # Handle different prompt formats
        if isinstance(prompt, str):
            payload["contents"] = [{"role": "user", "parts": [{"text": prompt}]}]
        elif isinstance(prompt, list) and all(isinstance(m, dict) for m in prompt):
            # Convert OpenAI/Anthropic format to Gemini format
            gemini_messages = []
            
            for msg in prompt:
                role = msg.get('role', 'user')
                
                # Map standard roles to Gemini roles
                if role == 'system':
                    role = 'user'  # Gemini doesn't have system role, use user
                elif role == 'assistant':
                    role = 'model'  # Gemini uses 'model' for assistant
                
                content = msg.get('content', '')
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": content}]
                })
            
            payload["contents"] = gemini_messages
        else:
            raise ValueError(f'Unsupported prompt format for Gemini API: {type(prompt)}')
        
        # Add any additional parameters passed via kwargs
        for key, value in kwargs.items():
            if key not in payload and value is not None:
                payload[key] = value
        
        # Estimate token counts for metrics
        prompt_tokens = 0
        try:
            if isinstance(prompt, str):
                prompt_tokens = await self._count_tokens(prompt)
            else:
                prompt_text = "\n".join(m.get('content', '') for m in prompt if 'content' in m)
                prompt_tokens = await self._count_tokens(prompt_text)
            
            logger.debug(f'Estimated prompt tokens for {self.model}: {prompt_tokens}')
        except Exception as token_err:
            logger.warning(f'Could not count prompt tokens for {self.model}: {token_err}')
        
        # Prepare headers
        headers: Dict[str, str] = {
            'Content-Type': 'application/json',
            'x-goog-api-key': self.api_key
        }
        
        # Make the API request
        request_start_time: float = time.monotonic()
        response_json: Optional[Dict[str, Any]] = None
        
        try:
            logger.debug(f'Sending request to Gemini API: POST {endpoint}')
            
            async with session.post(endpoint, json=payload, headers=headers, timeout=self.timeout) as response:
                status_code = response.status
                
                try:
                    response_json = await response.json()
                except aiohttp.ContentTypeError:
                    response_text = await response.text()
                    logger.error(f'Gemini API returned non-JSON response ({status_code}): {response_text[:200]}...')
                    raise LLMError(
                        code=ErrorCode.LLM_API_ERROR,
                        message=f'Gemini API returned non-JSON response (Status: {status_code})',
                        details={'status_code': status_code, 'response_body': response_text[:500]}
                    )
                
                request_time: float = time.monotonic() - request_start_time
                logger.debug(f'Received response from Gemini API (Status: {status_code}) in {request_time:.4f}s')
                
                if status_code != 200:
                    error_details = response_json.get('error', {}) if response_json else {}
                    error_msg = error_details.get('message', 'Unknown Gemini API error')
                    error_code = error_details.get('code', 0)
                    
                    # Map HTTP status codes to our error types
                    if status_code == 400:
                        error_type = ErrorCode.BAD_REQUEST
                    elif status_code == 401:
                        error_type = ErrorCode.AUTHENTICATION_ERROR
                    elif status_code == 403:
                        error_type = ErrorCode.AUTHORIZATION_ERROR
                    elif status_code == 404:
                        error_type = ErrorCode.ENDPOINT_NOT_FOUND
                    elif status_code == 429:
                        error_type = ErrorCode.LLM_RATE_LIMIT
                    else:
                        error_type = ErrorCode.LLM_API_ERROR
                        
                    raise LLMError(
                        code=error_type,
                        message=f'Gemini API error ({status_code} / {error_code}): {error_msg}',
                        details={
                            'status_code': status_code,
                            'error_code': error_code,
                            'error_message': error_msg,
                            'model': self.model,
                            'response_body': response_json
                        }
                    )
                
                # Extract the response content
                if not response_json.get('candidates'):
                    raise LLMError(
                        code=ErrorCode.LLM_API_ERROR,
                        message='Gemini API returned no candidates in response',
                        details={'model': self.model, 'response': response_json}
                    )
                
                candidate = response_json.get('candidates', [{}])[0]
                content = candidate.get('content', {})
                parts = content.get('parts', [])
                
                # Extract the generated text
                completion_text = ""
                for part in parts:
                    if 'text' in part:
                        completion_text += part['text']
                
                # Get usage data
                usage_metadata = response_json.get('usageMetadata', {})
                prompt_token_count = usage_metadata.get('promptTokenCount', prompt_tokens)
                completion_token_count = usage_metadata.get('candidatesTokenCount', 0)
                total_token_count = prompt_token_count + completion_token_count
                
                # Get finish reason
                finish_reason = candidate.get('finishReason', 'unknown')
                # Map Gemini finish reasons to OpenAI-style reasons
                finish_reason_map = {
                    'FINISH_REASON_UNSPECIFIED': 'unknown',
                    'STOP': 'stop',
                    'MAX_TOKENS': 'length',
                    'SAFETY': 'content_filter',
                    'RECITATION': 'content_filter',
                    'OTHER': 'unknown'
                }
                mapped_finish_reason = finish_reason_map.get(finish_reason, finish_reason.lower())
                
                # Construct a standardized response format
                result = {
                    'id': response_json.get('promptFeedback', {}).get('promptId', ''),
                    'object': 'gemini.completion',
                    'created': int(time.time()),
                    'model': self.model,
                    'choices': [
                        {
                            'text': completion_text,
                            'index': 0,
                            'finish_reason': mapped_finish_reason
                        }
                    ],
                    'usage': {
                        'prompt_tokens': prompt_token_count,
                        'completion_tokens': completion_token_count,
                        'total_tokens': total_token_count
                    },
                    'prompt_tokens': prompt_token_count,
                    'completion_tokens': completion_token_count,
                    'total_tokens': total_token_count,
                    'request_time': request_time,
                    'raw_response': response_json
                }
                
                # If token counts aren't in the response, estimate completion tokens
                if completion_token_count == 0 and completion_text:
                    try:
                        calculated_completion_tokens = await self._count_tokens(completion_text)
                        if calculated_completion_tokens > 0:
                            logger.warning(f'Gemini API returned 0 completion tokens but text exists. Calculated: {calculated_completion_tokens}')
                            result['completion_tokens'] = calculated_completion_tokens
                            result['usage']['completion_tokens'] = calculated_completion_tokens
                            result['usage']['total_tokens'] = result['prompt_tokens'] + calculated_completion_tokens
                            result['total_tokens'] = result['usage']['total_tokens']
                    except Exception:
                        pass
                
                return result
                
        except aiohttp.ClientError as http_err:
            error_code: ErrorCode = ErrorCode.NETWORK_ERROR
            
            if isinstance(http_err, asyncio.TimeoutError):
                error_code = ErrorCode.LLM_TIMEOUT
            elif isinstance(http_err, aiohttp.ClientConnectionError):
                error_code = ErrorCode.CONNECTION_ERROR
                
            logger.error(f'HTTP Client Error during Gemini request: {http_err}', exc_info=True)
            raise LLMError(
                code=error_code,
                message=f'HTTP Client Error calling Gemini API: {str(http_err)}',
                original_error=http_err,
                model=self.model,
                provider='gemini'
            ) from http_err
            
        except LLMError:
            raise
            
        except Exception as e:
            logger.error(f'Unexpected error during Gemini request processing: {e}', exc_info=True)
            raise LLMError(
                code=ErrorCode.LLM_API_ERROR,
                message=f'Unexpected error processing Gemini request: {str(e)}',
                original_error=e,
                model=self.model,
                provider='gemini'
            ) from e

    def get_token_limit(self) -> int:
        """Get the maximum token limit for the model."""
        if self.model in GEMINI_MODELS:
            return GEMINI_MODELS[self.model]['max_tokens']
            
        # Check for prefix matches
        for model_prefix, info in GEMINI_MODELS.items():
            if self.model.startswith(model_prefix):
                logger.debug(f"Using token limit {info['max_tokens']} based on prefix '{model_prefix}' for model '{self.model}'.")
                return info['max_tokens']
                
        # Try to get limit from models registry
        try:
            from src.llm.models import \
                get_token_limit as get_token_limit_from_models
            limit_from_models = get_token_limit_from_models(self.model)
            if limit_from_models != 4096:
                logger.debug(f"Using token limit {limit_from_models} from models.py for Gemini model '{self.model}'.")
                return limit_from_models
        except (ImportError, Exception) as e:
            logger.warning(f"Error getting token limit from models.py for '{self.model}': {e}")
            
        # Default fallback
        default_limit = 32768
        logger.warning(f"Could not determine specific token limit for Gemini model '{self.model}'. Using default: {default_limit}")
        return default_limit

    async def close(self) -> None:
        """Close the adapter and clean up resources."""
        await super().close()
        logger.debug(f'Closed Gemini adapter for model {self.model}. (Session managed by pool)')
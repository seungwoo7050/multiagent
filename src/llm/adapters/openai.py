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
_TOKENIZER_CACHE: Dict[str, tiktoken.Encoding] = {}

class OpenAIAdapter(BaseLLMAdapter):

    def __init__(self, model: str, provider: str='openai', api_key: Optional[str]=None, api_base: Optional[str]=None, timeout: Optional[float]=None, max_retries: Optional[int]=None, cache_enabled: bool=True, max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None):
        super().__init__(model=model, provider=provider, api_key=api_key, api_base=api_base, timeout=timeout, max_retries=max_retries, cache_enabled=cache_enabled, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        self.api_base: str = api_base or 'https://api.openai.com/v1'
        self.api_version: str = '2023-05-15'
        self.tokenizer: Optional[tiktoken.Encoding] = None
        if self.max_tokens is None:
            self.max_tokens = 1024
        logger.debug(f"OpenAIAdapter initialized for model '{self.model}' with API base '{self.api_base}'")

    async def _initialize(self) -> bool:
        try:
            if not self.api_key:
                provider_config = settings.LLM_PROVIDERS_CONFIG.get('openai', {})
                self.api_key = provider_config.get('api_key')
                if not self.api_key:
                    raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message='No API key provided for OpenAI adapter. Set OPENAI_API_KEY environment variable or provide it during initialization.')
            self.tokenizer = await self._get_tokenizer()
            if not self.tokenizer:
                logger.error(f'Failed to initialize tokenizer for model {self.model}')
                return False
            client = await self._get_client()
            if not client:
                logger.error('Failed to get HTTP client session during OpenAI adapter initialization.')
                return False
            logger.info(f'Successfully initialized OpenAI adapter for model {self.model}')
            return True
        except LLMError as lle:
            if lle.code == ErrorCode.LLM_PROVIDER_ERROR and 'No API key provided' in lle.message:
                raise lle
            logger.error(f'LLMError during OpenAI adapter initialization: {lle.code} - {lle.message}', extra=lle.to_dict())
            return False
        except Exception as e:
            logger.error(f'Failed to initialize OpenAI adapter for model {self.model}: {str(e)}', exc_info=True)
            return False

    async def _get_client(self) -> aiohttp.ClientSession:
        session = await get_connection_pool('openai')
        return session

    async def _get_tokenizer(self) -> tiktoken.Encoding:
        global _TOKENIZER_CACHE
        if self.model in _TOKENIZER_CACHE:
            logger.debug(f'Using cached tokenizer for model {self.model}')
            return _TOKENIZER_CACHE[self.model]
        encoding_name: str = 'cl100k_base'
        model_to_encoding: Dict[str, str] = {'gpt-4': 'cl100k_base', 'gpt-4o': 'o200k_base', 'gpt-4-turbo': 'cl100k_base', 'gpt-3.5-turbo': 'cl100k_base', 'text-embedding-ada-002': 'cl100k_base', 'text-embedding-3-small': 'cl100k_base', 'text-embedding-3-large': 'cl100k_base', 'text-davinci-003': 'p50k_base', 'text-davinci-002': 'p50k_base', 'davinci': 'p50k_base', 'code-davinci-002': 'p50k_base'}
        if self.model in model_to_encoding:
            encoding_name = model_to_encoding[self.model]
        else:
            for model_prefix, enc_name in model_to_encoding.items():
                if self.model.startswith(model_prefix):
                    encoding_name = enc_name
                    logger.debug(f"Using encoding '{encoding_name}' based on model prefix '{model_prefix}' for model '{self.model}'.")
                    break
            else:
                logger.warning(f"No specific encoding mapping found for model '{self.model}'. Using default '{encoding_name}'.")
        try:
            logger.debug(f'Loading tiktoken encoding: {encoding_name} for model {self.model}')
            tokenizer = tiktoken.get_encoding(encoding_name)
            _TOKENIZER_CACHE[self.model] = tokenizer
            return tokenizer
        except ValueError as e:
            logger.warning(f"Failed to get tiktoken encoding '{encoding_name}': {e}. Falling back to 'cl100k_base'.")
            try:
                tokenizer = tiktoken.get_encoding('cl100k_base')
                _TOKENIZER_CACHE[self.model] = tokenizer
                return tokenizer
            except Exception as fallback_e:
                error_msg = f"Failed to get tokenizer encoding '{encoding_name}' and fallback 'cl100k_base' for model {self.model}: {fallback_e}"
                logger.error(error_msg, exc_info=True)
                raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message=error_msg, model=self.model, original_error=fallback_e) from fallback_e
        except Exception as e:
            error_msg = f"Failed to get tokenizer encoding '{encoding_name}' for model {self.model}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message=error_msg, model=self.model, original_error=e) from e

    async def _count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            logger.warning(f'Tokenizer not initialized for model {self.model}. Attempting to initialize.')
            initialized = await self.ensure_initialized()
            if not initialized or not self.tokenizer:
                raise LLMError(code=ErrorCode.INITIALIZATION_ERROR, message=f'Tokenizer could not be initialized for model {self.model}')
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f'Error encoding text to count tokens for model {self.model}: {e}', exc_info=True)
            raise LLMError(code=ErrorCode.LLM_ERROR, message=f'Token counting failed: {str(e)}', model=self.model, original_error=e)

    @async_timed('openai_generate_text')
    async def _generate_text(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, stop_sequences: Optional[List[str]]=None, **kwargs: Any) -> Dict[str, Any]:
        session: aiohttp.ClientSession = await self._get_client()
        is_chat_model: bool = 'turbo' in self.model or self.model.startswith('gpt-4') or self.model.startswith('gpt-3.5')
        url: str
        payload: Dict[str, Any] = {'model': self.model, 'max_tokens': max_tokens if max_tokens is not None else self.max_tokens, 'temperature': temperature if temperature is not None else self.temperature, 'top_p': top_p if top_p is not None else self.top_p}
        if is_chat_model:
            url = f'{self.api_base}/chat/completions'
            if isinstance(prompt, str):
                payload['messages'] = [{'role': 'user', 'content': prompt}]
            elif isinstance(prompt, list) and all((isinstance(m, dict) and 'role' in m and ('content' in m) for m in prompt)):
                payload['messages'] = prompt
            else:
                logger.error(f'Invalid prompt format for chat model {self.model}: {type(prompt)}. Attempting to proceed by string conversion.')
                payload['messages'] = [{'role': 'user', 'content': str(prompt)}]
        else:
            url = f'{self.api_base}/completions'
            if not isinstance(prompt, str):
                logger.warning(f'Completion model {self.model} expects a string prompt, got {type(prompt)}. Converting to string.')
                payload['prompt'] = str(prompt)
            else:
                payload['prompt'] = prompt
        if stop_sequences:
            payload['stop'] = stop_sequences
        for key, value in kwargs.items():
            if value is not None and key not in payload:
                payload[key] = value
        prompt_tokens: int = 0
        try:
            if is_chat_model:
                prompt_text_for_count = '\n'.join([m.get('content', '') for m in payload.get('messages', [])])
                prompt_tokens = await self._count_tokens(prompt_text_for_count)
            else:
                prompt_tokens = await self._count_tokens(payload.get('prompt', ''))
            logger.debug(f'Estimated prompt tokens for {self.model}: {prompt_tokens}')
        except Exception as token_err:
            logger.warning(f'Could not count prompt tokens for {self.model}: {token_err}')
        headers: Dict[str, str] = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        request_start_time: float = time.monotonic()
        response_json: Optional[Dict[str, Any]] = None
        try:
            logger.debug(f'Sending request to OpenAI API: POST {url}')
            async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as response:
                status_code = response.status
                try:
                    response_json = await response.json()
                except aiohttp.ContentTypeError:
                    response_text = await response.text()
                    logger.error(f'OpenAI API returned non-JSON response ({status_code}): {response_text[:200]}...')
                    raise LLMError(code=ErrorCode.LLM_API_ERROR, message=f'OpenAI API returned non-JSON response (Status: {status_code})', details={'status_code': status_code, 'response_body': response_text[:500]})
                request_time: float = time.monotonic() - request_start_time
                logger.debug(f'Received response from OpenAI API (Status: {status_code}) in {request_time:.4f}s')
                if status_code != 200:
                    error_details = response_json.get('error', {}) if response_json else {}
                    error_msg = error_details.get('message', 'Unknown OpenAI API error')
                    error_type = error_details.get('type', 'api_error')
                    error_code: ErrorCode = ErrorCode.LLM_API_ERROR
                    if status_code == 400:
                        error_code = ErrorCode.BAD_REQUEST
                    elif status_code == 401:
                        error_code = ErrorCode.AUTHENTICATION_ERROR
                    elif status_code == 403:
                        error_code = ErrorCode.AUTHORIZATION_ERROR
                    elif status_code == 404:
                        error_code = ErrorCode.ENDPOINT_NOT_FOUND
                    elif status_code == 429:
                        error_code = ErrorCode.LLM_RATE_LIMIT
                    raise LLMError(code=error_code, message=f'OpenAI API error ({status_code} {error_type}): {error_msg}', details={'status_code': status_code, 'error_type': error_type, 'error_message': error_msg, 'model': self.model, 'response_body': response_json})
                result: Dict[str, Any]
                if is_chat_model:
                    choices = response_json.get('choices', [])
                    completion_text = choices[0].get('message', {}).get('content', '') if choices else ''
                    finish_reason = choices[0].get('finish_reason', '') if choices else ''
                    usage = response_json.get('usage', {})
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    final_prompt_tokens = usage.get('prompt_tokens', prompt_tokens)
                    result = {'id': response_json.get('id', ''), 'object': response_json.get('object', 'chat.completion'), 'created': response_json.get('created', int(time.time())), 'model': response_json.get('model', self.model), 'choices': [{'text': completion_text, 'index': 0, 'finish_reason': finish_reason}], 'usage': {'prompt_tokens': final_prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens if total_tokens else final_prompt_tokens + completion_tokens}, 'prompt_tokens': final_prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens if total_tokens else final_prompt_tokens + completion_tokens, 'request_time': request_time}
                else:
                    choices = response_json.get('choices', [])
                    completion_text = choices[0].get('text', '') if choices else ''
                    usage = response_json.get('usage', {})
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    final_prompt_tokens = usage.get('prompt_tokens', prompt_tokens)
                    result = {'id': response_json.get('id', ''), 'object': response_json.get('object', 'text_completion'), 'created': response_json.get('created', int(time.time())), 'model': response_json.get('model', self.model), 'choices': choices, 'usage': {'prompt_tokens': final_prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens if total_tokens else final_prompt_tokens + completion_tokens}, 'prompt_tokens': final_prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens if total_tokens else final_prompt_tokens + completion_tokens, 'request_time': request_time}
                if result['completion_tokens'] == 0 and result['choices'] and result['choices'][0].get('text'):
                    try:
                        calculated_completion_tokens = await self._count_tokens(result['choices'][0]['text'])
                        if calculated_completion_tokens > 0:
                            logger.warning(f'API returned 0 completion tokens but text exists. Calculated: {calculated_completion_tokens}')
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
            logger.error(f'HTTP Client Error during OpenAI request: {http_err}', exc_info=True)
            raise LLMError(code=error_code, message=f'HTTP Client Error calling OpenAI API: {str(http_err)}', original_error=http_err, model=self.model, provider='openai') from http_err
        except LLMError:
            raise
        except Exception as e:
            logger.error(f'Unexpected error during OpenAI request processing: {e}', exc_info=True)
            raise LLMError(code=ErrorCode.LLM_API_ERROR, message=f'Unexpected error processing OpenAI request: {str(e)}', original_error=e, model=self.model, provider='openai') from e

    def get_token_limit(self) -> int:
        model_token_limits: Dict[str, int] = {'gpt-4': 8192, 'gpt-4-32k': 32768, 'gpt-4-turbo': 128000, 'gpt-4-turbo-preview': 128000, 'gpt-4o': 128000, 'gpt-3.5-turbo': 16385, 'gpt-3.5-turbo-16k': 16385, 'text-davinci-003': 4097, 'text-davinci-002': 4097, 'davinci': 2049, 'text-embedding-ada-002': 8191, 'text-embedding-3-small': 8191, 'text-embedding-3-large': 8191}
        if self.model in model_token_limits:
            return model_token_limits[self.model]
        for model_prefix, token_limit in model_token_limits.items():
            if self.model.startswith(model_prefix):
                logger.debug(f"Using token limit {token_limit} based on prefix '{model_prefix}' for model '{self.model}'.")
                return token_limit
        default_limit = 4097
        logger.warning(f"Could not determine specific token limit for model '{self.model}'. Returning default value: {default_limit}")
        return default_limit

    async def close(self) -> None:
        await super().close()
        logger.debug(f'Closed OpenAI adapter for model {self.model}. (Session managed by pool)')
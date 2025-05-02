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
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_MODELS = {'claude-3-opus-20240229': {'max_tokens': 200000}, 'claude-3-sonnet-20240229': {'max_tokens': 200000}, 'claude-3-haiku-20240307': {'max_tokens': 200000}, 'claude-2.1': {'max_tokens': 200000}, 'claude-2.0': {'max_tokens': 100000}, 'claude-instant-1.2': {'max_tokens': 100000}}

class AnthropicAdapter(BaseLLMAdapter):

    def __init__(self, model: str, provider: str='anthropic', api_key: Optional[str]=None, api_base: Optional[str]=None, timeout: Optional[float]=None, max_retries: Optional[int]=None, cache_enabled: bool=True, max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None):
        super().__init__(model=model, provider=provider, api_key=api_key, api_base=api_base, timeout=timeout, max_retries=max_retries, cache_enabled=cache_enabled, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        self.api_base = api_base or ANTHROPIC_API_URL
        self.api_version = '2023-06-01'
        if self.max_tokens is None:
            self.max_tokens = 1024
        logger.debug(f"AnthropicAdapter initialized for model '{self.model}' with API base '{self.api_base}'")

    async def _initialize(self) -> bool:
        try:
            if not self.api_key:
                provider_config = settings.LLM_PROVIDERS_CONFIG.get('anthropic', {})
                self.api_key = provider_config.get('api_key')
                if not self.api_key:
                    raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message='No API key provided for Anthropic adapter. Set ANTHROPIC_API_KEY environment variable or provide it during initialization.')
            client = await self._get_client()
            if not client:
                logger.error('Failed to get HTTP client session during Anthropic adapter initialization.')
                return False
            logger.info(f'Successfully initialized Anthropic adapter for model {self.model}')
            return True
        except LLMError as lle:
            if lle.code == ErrorCode.LLM_PROVIDER_ERROR and 'No API key provided' in lle.message:
                raise lle
            logger.error(f'LLMError during Anthropic adapter initialization: {lle.code} - {lle.message}', extra=lle.to_dict())
            return False
        except Exception as e:
            logger.error(f'Failed to initialize Anthropic adapter for model {self.model}: {str(e)}', exc_info=True)
            return False

    async def _get_client(self) -> aiohttp.ClientSession:
        session = await get_connection_pool('anthropic')
        return session

    async def _count_tokens(self, text: str) -> int:
        from src.llm.tokenizer import count_tokens as global_count_tokens
        try:
            return await global_count_tokens(self.model, text)
        except Exception as e:
            logger.error(f'Error counting tokens for Anthropic model {self.model} using approximation: {e}', exc_info=True)
            return len(text) // 4

    @async_timed('anthropic_generate_text')
    async def _generate_text(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, stop_sequences: Optional[List[str]]=None, **kwargs) -> Dict[str, Any]:
        session: aiohttp.ClientSession = await self._get_client()
        payload: Dict[str, Any] = {'model': self.model, 'max_tokens': max_tokens if max_tokens is not None else self.max_tokens, 'temperature': temperature if temperature is not None else self.temperature, 'top_p': top_p if top_p is not None else self.top_p}
        if isinstance(prompt, str):
            payload['messages'] = [{'role': 'user', 'content': prompt}]
        elif isinstance(prompt, list) and all((isinstance(m, dict) for m in prompt)):
            payload['messages'] = prompt
        else:
            raise ValueError(f'Unsupported prompt format for Anthropic Messages API: {type(prompt)}. Expected str or List[Dict[str, str]].')
        if stop_sequences:
            payload['stop_sequences'] = stop_sequences
        for key, value in kwargs.items():
            if value is not None and key not in ['model', 'max_tokens', 'messages', 'temperature', 'top_p', 'stop_sequences']:
                payload[key] = value
        prompt_tokens = 0
        try:
            prompt_text_for_count = ''
            for message in payload.get('messages', []):
                content = message.get('content')
                if isinstance(content, str):
                    prompt_text_for_count += content + '\n'
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            prompt_text_for_count += item.get('text', '') + '\n'
            prompt_tokens = await self._count_tokens(prompt_text_for_count)
            logger.debug(f'Estimated prompt tokens for Anthropic model {self.model}: {prompt_tokens}')
        except Exception as token_err:
            logger.warning(f'Could not count prompt tokens for {self.model} (Anthropic): {token_err}')
        headers: Dict[str, str] = {'Content-Type': 'application/json', 'x-api-key': self.api_key, 'anthropic-version': self.api_version}
        request_start_time: float = time.monotonic()
        response_json: Optional[Dict[str, Any]] = None
        try:
            logger.debug(f'Sending request to Anthropic API: POST {self.api_base}')
            async with session.post(self.api_base, json=payload, headers=headers, timeout=self.timeout) as response:
                status_code = response.status
                try:
                    response_json = await response.json()
                except aiohttp.ContentTypeError:
                    response_text = await response.text()
                    logger.error(f'Anthropic API returned non-JSON response ({status_code}): {response_text[:200]}...')
                    raise LLMError(code=ErrorCode.LLM_API_ERROR, message=f'Anthropic API returned non-JSON response (Status: {status_code})', details={'status_code': status_code, 'response_body': response_text[:500]})
                request_time: float = time.monotonic() - request_start_time
                logger.debug(f'Received response from Anthropic API (Status: {status_code}) in {request_time:.4f}s')
                if status_code != 200:
                    error_details = response_json.get('error', {}) if response_json else {}
                    error_msg = error_details.get('message', 'Unknown Anthropic API error')
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
                    raise LLMError(code=error_code, message=f'Anthropic API error ({status_code} {error_type}): {error_msg}', details={'status_code': status_code, 'error_type': error_type, 'error_message': error_msg, 'model': self.model, 'response_body': response_json})
                completion_content_list = response_json.get('content', [])
                completion_text = ''
                if completion_content_list and isinstance(completion_content_list[0], dict):
                    if completion_content_list[0].get('type') == 'text':
                        completion_text = completion_content_list[0].get('text', '')
                finish_reason = response_json.get('stop_reason', 'unknown')
                usage = response_json.get('usage', {})
                final_prompt_tokens = usage.get('input_tokens', prompt_tokens)
                completion_tokens = usage.get('output_tokens', 0)
                total_tokens = final_prompt_tokens + completion_tokens
                result = {'id': response_json.get('id', ''), 'object': 'messages.completion', 'created': int(time.time()), 'model': response_json.get('model', self.model), 'choices': [{'text': completion_text, 'index': 0, 'finish_reason': finish_reason}], 'usage': {'prompt_tokens': final_prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens}, 'prompt_tokens': final_prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': total_tokens, 'request_time': request_time, 'raw_response': response_json}
                if result['completion_tokens'] == 0 and completion_text:
                    try:
                        calculated_completion_tokens = await self._count_tokens(completion_text)
                        if calculated_completion_tokens > 0:
                            logger.warning(f'Anthropic API returned 0 output tokens but text exists. Calculated approximation: {calculated_completion_tokens}')
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
            logger.error(f'HTTP Client Error during Anthropic request: {http_err}', exc_info=True)
            raise LLMError(code=error_code, message=f'HTTP Client Error calling Anthropic API: {str(http_err)}', original_error=http_err, model=self.model, provider='anthropic') from http_err
        except LLMError:
            raise
        except Exception as e:
            logger.error(f'Unexpected error during Anthropic request processing: {e}', exc_info=True)
            raise LLMError(code=ErrorCode.LLM_API_ERROR, message=f'Unexpected error processing Anthropic request: {str(e)}', original_error=e, model=self.model, provider='anthropic') from e

    def get_token_limit(self) -> int:
        if self.model in ANTHROPIC_MODELS:
            return ANTHROPIC_MODELS[self.model]['max_tokens']
        for model_prefix, info in ANTHROPIC_MODELS.items():
            if self.model.startswith(model_prefix):
                logger.debug(f"Using token limit {info['max_tokens']} based on prefix '{model_prefix}' for model '{self.model}'.")
                return info['max_tokens']
        try:
            from src.llm.models import \
                get_token_limit as get_token_limit_from_models
            limit_from_models = get_token_limit_from_models(self.model)
            if limit_from_models != 4096:
                logger.debug(f"Using token limit {limit_from_models} from models.py for Anthropic model '{self.model}'.")
                return limit_from_models
        except ImportError:
            logger.warning('Could not import src.llm.models to get token limit for Anthropic.')
        except Exception as e:
            logger.warning(f"Error getting token limit from models.py for '{self.model}': {e}")
        default_limit = 100000
        logger.warning(f"Could not determine specific token limit for Anthropic model '{self.model}'. Returning default value: {default_limit}")
        return default_limit

    async def close(self) -> None:
        await super().close()
        logger.debug(f'Closed Anthropic adapter for model {self.model}. (Session managed by pool)')
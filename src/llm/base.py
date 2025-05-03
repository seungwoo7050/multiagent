import abc
import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, TypeVar, Union

from src.config.errors import ErrorCode, LLMError
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.llm.cache import cache_result, get_cache
from src.utils.timing import async_timed

settings = get_settings()
logger = get_logger(__name__)
metrics = get_metrics_manager()

T = TypeVar('T')

class BaseLLMAdapter(abc.ABC):

    def __init__(self, model: str, provider: str, api_key: Optional[str]=None, api_base: Optional[str]=None, timeout: Optional[float]=None, max_retries: Optional[int]=None, cache_enabled: bool=True, max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None):
        self.model: str = model
        self.provider: str = provider
        self.api_key: Optional[str] = api_key
        self.api_base: Optional[str] = api_base
        self.timeout: float = timeout if timeout is not None else settings.REQUEST_TIMEOUT
        self.max_retries: int = max_retries if max_retries is not None else settings.LLM_RETRY_MAX_ATTEMPTS
        self.cache_enabled: bool = cache_enabled
        self.max_tokens: Optional[int] = max_tokens
        self.temperature: Optional[float] = temperature if temperature is not None else 0.7
        self.top_p: Optional[float] = top_p if top_p is not None else 1.0
        self.initialized: bool = False
        self._client: Optional[Any] = None
        self.request_count: int = 0
        self.token_usage: Dict[str, int] = {'prompt': 0, 'completion': 0, 'total': 0}
        self.error_count: int = 0
        self.average_latency: float = 0.0
        logger.debug(f"Initialized {self.__class__.__name__} for model '{model}' (provider: {provider})")
        
    # src/llm/base.py 수정
    # BaseLLMAdapter 클래스에 다음 메서드 추가
    async def execute(self, call_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LLM request with the provided arguments.
        This method serves as a bridge between the MCP adapter and the LLM provider-specific implementation.
        
        Args:
            call_args: Dictionary containing call arguments for the LLM
            
        Returns:
            Response dictionary from the LLM provider
        """
        # call_args에서 필요한 파라미터 추출
        prompt = call_args.get('prompt')
        max_tokens = call_args.get('max_tokens', self.max_tokens)
        temperature = call_args.get('temperature', self.temperature)
        top_p = call_args.get('top_p', self.top_p)
        stop_sequences = call_args.get('stop_sequences')
        use_cache = call_args.get('use_cache', True)
        retry_on_failure = call_args.get('retry_on_failure', True)
        
        # 추가 파라미터는 kwargs로 전달
        additional_params = call_args.get('additional_params', {})
        
        # generate 메서드 호출
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            retry_on_failure=retry_on_failure,
            use_cache=use_cache,
            **additional_params
        )    

    @abc.abstractmethod
    async def _initialize(self) -> bool:
        pass

    @abc.abstractmethod
    async def _get_client(self) -> Any:
        pass

    @abc.abstractmethod
    async def _generate_text(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, stop_sequences: Optional[List[str]]=None, **kwargs: Any) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def _count_tokens(self, text: str) -> int:
        pass

    def _get_cache_key(self, prompt: Union[str, List[Dict[str, str]]], params: Dict[str, Any]) -> str:
        cache_dict: Dict[str, Any] = {'model': self.model, 'provider': self.provider, 'prompt': json.dumps(prompt, sort_keys=True) if isinstance(prompt, list) else prompt, 'max_tokens': params.get('max_tokens', self.max_tokens), 'temperature': params.get('temperature', self.temperature), 'top_p': params.get('top_p', self.top_p), 'stop_sequences': sorted(params.get('stop_sequences', []) or [])}
        for k, v in params.items():
            if k not in ['max_tokens', 'temperature', 'top_p', 'stop_sequences'] and v is not None:
                if isinstance(v, list):
                    try:
                        cache_dict[k] = tuple(sorted(v))
                    except TypeError:
                        cache_dict[k] = tuple(v)
                elif isinstance(v, dict):
                    cache_dict[k] = tuple(sorted(v.items()))
                else:
                    cache_dict[k] = v
        stable_str: str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(stable_str.encode()).hexdigest()

    def _log_request_metrics(self, prompt_tokens: int, completion_tokens: int, latency: float, is_cached: bool=False) -> None:
        self.request_count += 1
        self.token_usage['prompt'] += prompt_tokens
        self.token_usage['completion'] += completion_tokens
        self.token_usage['total'] += prompt_tokens + completion_tokens
        if self.request_count > 0:
            self.average_latency = (self.average_latency * (self.request_count - 1) + latency) / self.request_count
        else:
            self.average_latency = latency
        if not is_cached:
            metrics.track_llm('requests', model=self.model, provider=self.provider)
            metrics.track_llm('duration', model=self.model, provider=self.provider, value=latency)
            metrics.track_llm('tokens', model=self.model, provider=self.provider, type='prompt', value=prompt_tokens)
            metrics.track_llm('tokens', model=self.model, provider=self.provider, type='completion', value=completion_tokens)
            logger.debug(f'Logged LLM metrics for {self.model} ({self.provider}). Latency: {latency:.4f}s, Tokens: P {prompt_tokens}/C {completion_tokens}')

    def _log_error_metrics(self, error_type: str) -> None:
        self.error_count += 1
        metrics.track_llm('errors', model=self.model, provider=self.provider, error_type=error_type)
        logger.debug(f'Logged LLM error metric for {self.model} ({self.provider}). Error type: {error_type}')

    async def ensure_initialized(self) -> bool:
        if not self.initialized:
            logger.debug(f'Initializing adapter for {self.model} ({self.provider})...')
            self.initialized = await self._initialize()
            if self.initialized:
                logger.debug(f'Adapter for {self.model} ({self.provider}) initialized successfully.')
            else:
                logger.error(f'Adapter initialization failed for {self.model} ({self.provider}).')
        return self.initialized

    async def tokenize(self, text: str) -> int:
        try:
            return await self._count_tokens(text)
        except Exception as e:
            raise LLMError(code=ErrorCode.LLM_ERROR, message=f'Failed to count tokens for model {self.model}: {str(e)}', original_error=e, model=self.model, provider=self.provider)

    @async_timed('llm_request')
    async def generate(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, stop_sequences: Optional[List[str]]=None, retry_on_failure: bool=True, use_cache: Optional[bool]=None, **kwargs: Any) -> Dict[str, Any]:
        start_time = time.time()
        initialized = await self.ensure_initialized()
        if not initialized:
            raise LLMError(code=ErrorCode.INITIALIZATION_ERROR, message=f'Adapter for {self.model} ({self.provider}) failed to initialize.', model=self.model, provider=self.provider)
        params: Dict[str, Any] = {'max_tokens': max_tokens if max_tokens is not None else self.max_tokens, 'temperature': temperature if temperature is not None else self.temperature, 'top_p': top_p if top_p is not None else self.top_p, 'stop_sequences': stop_sequences, **kwargs}
        params = {k: v for k, v in params.items() if v is not None}
        use_cache_for_request: bool = self.cache_enabled if use_cache is None else use_cache
        cache_key: Optional[str] = None
        cache: Optional[Any] = None
        if use_cache_for_request:
            cache = await get_cache()
            cache_key = self._get_cache_key(prompt, params)
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f'Cache hit for LLM request (key: {cache_key})')
                latency = time.time() - start_time
                try:
                    prompt_tokens = cached_result.get('usage', {}).get('prompt_tokens') or await self.tokenize(str(prompt))
                    completion_tokens = cached_result.get('usage', {}).get('completion_tokens') or await self.tokenize(cached_result.get('choices', [{}])[0].get('text', ''))
                    self._log_request_metrics(prompt_tokens, completion_tokens, latency, is_cached=True)
                except Exception as token_err:
                    logger.warning(f'Could not determine token counts for cached result: {token_err}')
                    self._log_request_metrics(0, 0, latency, is_cached=True)
                return cached_result
            logger.debug(f'Cache miss for LLM request (key: {cache_key})')
        try:
            response = await self._generate_text(prompt=prompt, **params)
            if use_cache_for_request and cache is not None and (cache_key is not None):
                try:
                    await cache_result(cache_key, response)
                    logger.debug(f'Cached LLM response (key: {cache_key})')
                except Exception as cache_err:
                    logger.error(f'Failed to cache LLM response for key {cache_key}: {cache_err}')
            latency = time.time() - start_time
            try:
                usage = response.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                if prompt_tokens == 0 and completion_tokens == 0:
                    logger.debug('Token usage not found in response, calculating manually.')
                    prompt_text_for_count = str(prompt)
                    prompt_tokens = await self.tokenize(prompt_text_for_count)
                    completion_text = ''
                    choices = response.get('choices', [])
                    if choices and isinstance(choices[0], dict):
                        if 'message' in choices[0] and isinstance(choices[0]['message'], dict):
                            completion_text = choices[0]['message'].get('content', '')
                        elif 'text' in choices[0]:
                            completion_text = choices[0]['text']
                    elif isinstance(response.get('content'), list) and response['content']:
                        if isinstance(response['content'][0], dict) and response['content'][0].get('type') == 'text':
                            completion_text = response['content'][0].get('text', '')
                    completion_tokens = await self.tokenize(completion_text or '')
                    response.setdefault('usage', {}).update({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens, 'estimated': True})
                self._log_request_metrics(prompt_tokens, completion_tokens, latency)
            except Exception as metric_err:
                logger.warning(f'Failed to log LLM request metrics: {metric_err}')
                self._log_request_metrics(0, 0, latency)
            return response
        except Exception as e:
            latency = time.time() - start_time
            error_type = type(e).__name__
            self._log_error_metrics(error_type)
            if isinstance(e, LLMError):
                llm_error = e
            else:
                llm_error = LLMError(code=ErrorCode.LLM_API_ERROR, message=f'Error generating text from {self.provider} ({self.model}): {str(e)}', original_error=e, model=self.model, provider=self.provider)
            llm_error.log_error(logger)
            raise llm_error

    def generate_sync(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, stop_sequences: Optional[List[str]]=None, retry_on_failure: bool=True, use_cache: Optional[bool]=None, **kwargs: Any) -> Dict[str, Any]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug('No running event loop found in generate_sync. Creating a new one.')
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop = True
        else:
            created_loop = False
        try:
            result = loop.run_until_complete(self.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_sequences=stop_sequences, retry_on_failure=retry_on_failure, use_cache=use_cache, **kwargs))
            return result
        finally:
            if created_loop:
                logger.debug('Closing the event loop created in generate_sync.')
                loop.close()
                asyncio.set_event_loop(None)

    @abc.abstractmethod
    def get_token_limit(self) -> int:
        return 4096

    def get_metrics(self) -> Dict[str, Any]:
        return {'model': self.model, 'provider': self.provider, 'request_count': self.request_count, 'token_usage': self.token_usage.copy(), 'error_count': self.error_count, 'average_latency_sec': self.average_latency}

    async def health_check(self) -> Dict[str, Any]:
        health_status: Dict[str, Any] = {'adapter': self.__class__.__name__, 'model': self.model, 'provider': self.provider, 'status': 'unknown', 'message': '', 'latency_sec': 0.0}
        start_time = time.time()
        try:
            initialized = await self.ensure_initialized()
            if not initialized:
                health_status['status'] = 'error'
                health_status['message'] = 'Adapter failed to initialize'
                health_status['latency_sec'] = time.time() - start_time
                return health_status
            client = await self._get_client()
            if client is None:
                health_status['status'] = 'error'
                health_status['message'] = 'Failed to get API client instance'
                health_status['latency_sec'] = time.time() - start_time
                return health_status
            health_status['status'] = 'ok'
            health_status['message'] = f'{self.provider} ({self.model}) adapter appears operational.'
            health_status['latency_sec'] = time.time() - start_time
            return health_status
        except Exception as e:
            health_status['status'] = 'error'
            health_status['message'] = f'Health check failed: {str(e)}'
            health_status['latency_sec'] = time.time() - start_time
            logger.warning(f'Health check failed for {self.provider} ({self.model}): {e}')
            return health_status

    async def close(self) -> None:
        self.initialized = False
        self._client = None
        logger.debug(f'Closed {self.__class__.__name__} adapter for {self.model} ({self.provider})')
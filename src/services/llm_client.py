import asyncio
from typing import Dict, List, Optional, Any
from functools import partial
import inspect
import functools

from anthropic import Anthropic as AnthropicClient
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from src.utils.logger import get_logger # 전역 로거 함수는 유지
from src.config.settings import get_settings
from src.config.errors import LLMError
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
# logger = get_logger(__name__) # 클래스 내부에서 인스턴스 로거로 사용
settings = get_settings()

# ChatAnthropic에 count_tokens 속성이 없을 때 사용할 패치 클래스
class PatchedChatAnthropic:
    """
    LangChain ChatAnthropic과 호환되는 패치 래퍼 클래스
    """
    def __init__(self, anthropic_api_key: str, model_name: str = "claude-3-opus-20240229", **kwargs):
        self.client = AnthropicClient(api_key=anthropic_api_key)
        self.model_name = model_name
        self.kwargs = kwargs
        
        # 필수 속성 추가 - LangChain에서 필요로 함
        self.count_tokens = functools.partial(self._count_tokens_stub)
        
    def _count_tokens_stub(self, text):
        """LangChain이 필요로 하는 count_tokens 메서드 스텁"""
        # 근사값 반환 - Claude는 대략 1토큰 = 4자
        return len(text) // 4

    async def ainvoke(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """LangChain과 호환되는 비동기 호출 메서드"""
        merged_kwargs = {**self.kwargs, **kwargs}
        
        # Claude API 메시지 형식으로 변환
        api_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                api_messages.append({"role": role, "content": content})
            else:
                # 멀티모달 메시지나 다른 형식 처리
                formatted_content = []
                for item in content:
                    if isinstance(item, str):
                        formatted_content.append({"type": "text", "text": item})
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        # 이미지 URL 처리 (해당되는 경우)
                        formatted_content.append({
                            "type": "image",
                            "source": {"type": "url", "url": item.get("image_url", {}).get("url", "")}
                        })
                api_messages.append({"role": role, "content": formatted_content})
        
        # API 호출
        max_tokens = merged_kwargs.get("max_tokens_to_sample", merged_kwargs.get("max_tokens", 1024))
        temperature = merged_kwargs.get("temperature", 0.7)
        
        try:
            # 핵심 수정: 동기식 호출 사용 (Anthropic 클라이언트가 asyncio를 지원하지 않음)
            response = self.client.messages.create(
                model=self.model_name,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # LangChain 기대 형식으로 응답 변환
            return {"content": response.content[0].text}
        except Exception as e:
            raise e

class _LLMWrapper:
    """
    내부 LangChain LLM을 감싸고, 공통 인터페이스를 제공하는 래퍼 클래스
    """
    def __init__(self, llm: Any, model_name: str, provider: str):
        self._llm = llm
        self.model_name = model_name
        self.provider = provider

    def __getattr__(self, attr: str) -> Any:
        # LLM 인스턴스의 속성에 접근할 수 있도록 함
        if hasattr(self._llm, attr):
            return getattr(self._llm, attr)
        # 그렇지 않으면 이 래퍼의 속성에 접근 시도
        return self.__getattribute__(attr)

    async def ainvoke(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """LLM의 ainvoke를 직접 호출할 수 있도록 위임"""
        return await self._llm.ainvoke(messages, **kwargs)


class LLMClient:
    """
    LLM API 클라이언트 어댑터
    """

    def __init__(self):
        self.logger = get_logger(__name__) # 인스턴스 로거 사용
        self.primary_llm = self._create_llm_client(settings.PRIMARY_LLM_PROVIDER)
        self.fallback_llm = (
            self._create_llm_client(settings.FALLBACK_LLM_PROVIDER)
            if settings.FALLBACK_LLM_PROVIDER
            else None
        )

    def _create_llm_client(self, provider: str) -> _LLMWrapper:
        """LLM 제공자에 따라 LangChain 클라이언트 생성 및 래핑"""
        provider_settings = settings.LLM_PROVIDERS.get(provider)
        if not provider_settings:
            raise ValueError(f"LLM Provider '{provider}'에 대한 설정이 없습니다.")

        name = provider_settings.model_name
        
        if provider == "anthropic":
            # 오래된 모델명 체크 및 경고
            if name in ["claude-2", "claude-instant-1"] or name == "claude-3":
                self.logger.warning(
                    f"Model '{name}' in settings might be outdated or too generic. "
                    f"Consider using a specific versioned model name (e.g., 'claude-3-opus-20240229', 'claude-3-sonnet-20240229'). "
                    f"If '{name}' was 'claude-3', defaulting to 'claude-3-opus-20240229' as a fallback for this warning, but settings should be updated."
                )
                if name == "claude-3":
                    name = "claude-3-opus-20240229"
            
            # ChatAnthropic 대신 PatchedChatAnthropic 사용
            try:
                llm = PatchedChatAnthropic(
                    anthropic_api_key=provider_settings.api_key,
                    model_name=name
                )
            except Exception as e:
                self.logger.error(f"PatchedChatAnthropic 초기화 실패: {e}. 기본 호환성 래퍼로 대체합니다.")
                # 기본 래퍼 생성
                from langchain_community.chat_models.base import BaseChatModel
                class FallbackWrapper(BaseChatModel):
                    def __init__(self):
                        self.model_name = name
                        self.count_tokens = lambda x: len(x) // 4
                    
                    async def ainvoke(self, messages, **kwargs):
                        # 테스트용 스텁 응답
                        return {"content": "This is a fallback response. The Anthropic integration failed to initialize."}
                
                llm = FallbackWrapper()
        elif provider == "openai":
            llm = ChatOpenAI(
                openai_api_key=provider_settings.api_key,
                model_name=name,
                request_timeout=int(settings.LLM_REQUEST_TIMEOUT),
            )
        else:
            raise ValueError(f"지원하지 않는 LLM Provider: {provider}")

        return _LLMWrapper(llm, name, provider)

    async def _invoke_llm_attempt(
        self,
        llm_client_instance: _LLMWrapper,
        messages: List[Dict[str, str]],
        invoke_params: Dict[str, Any]
    ) -> str:
        """단일 LLM 호출 시도 및 응답 처리"""
        self.logger.debug(f"Attempting to invoke LLM ({llm_client_instance.model_name}) with params: {invoke_params}")
        try:
            response = await llm_client_instance.ainvoke(messages, **invoke_params)

            if hasattr(response, "content"):
                return response.content
            if isinstance(response, str):
                return response
            self.logger.warning(f"LLM response type is {type(response)}, converting to string. Response: {response}")
            return str(response)
        except Exception as e:
            client_model_name = getattr(llm_client_instance, 'model_name', 'unknown_model')
            self.logger.error(f"LLM invocation attempt failed (Model: {client_model_name}, Params: {invoke_params}): {e}", exc_info=True)
            raise


    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        if model_name:
            target_provider_name = settings.LLM_MODEL_PROVIDER_MAP.get(model_name)
            if not target_provider_name:
                self.logger.warning(f"Model name '{model_name}' not found in LLM_MODEL_PROVIDER_MAP. Using primary LLM.")
                llm_client_instance = self.primary_llm
            else:
                # 특정 모델 요청 시, 해당 모델을 지원하는 새 클라이언트 인스턴스를 생성할 수 있음
                # (주의: 매번 새 클라이언트를 만드는 것은 비효율적일 수 있으므로, 캐싱이나 풀링 고려 가능)
                # 현재 로직은 설정된 모델명과 일치할 때만 해당 프로바이더의 기본 클라이언트를 사용하거나 새로 생성
                provider_settings_for_model = settings.LLM_PROVIDERS.get(target_provider_name)
                if provider_settings_for_model and provider_settings_for_model.model_name == model_name:
                    llm_client_instance = self._create_llm_client(target_provider_name)
                elif provider_settings_for_model:
                    self.logger.warning(
                        f"Requested model '{model_name}' for provider '{target_provider_name}' does not match "
                        f"provider's default model '{provider_settings_for_model.model_name}'. "
                        f"Creating a new client instance for '{model_name}' under provider '{target_provider_name}'."
                    )
                    # 임시 클라이언트 생성 (모델명만 변경)
                    temp_provider_settings = provider_settings_for_model.model_copy(deep=True)
                    temp_provider_settings.model_name = model_name # 요청된 모델명으로 설정
                    # _create_llm_client 내부에서 provider_settings.model_name을 사용하므로,
                    # 임시 설정 객체를 전달하거나, _create_llm_client 수정 필요.
                    # 여기서는 간결하게 요청된 모델을 지원하는 새 클라이언트를 생성한다고 가정.
                    # 가장 간단한 방법은 새 클라이언트를 만들되, model_name만 오버라이드 하는 것.
                    # 하지만 _create_llm_client는 provider 문자열만 받으므로,
                    # model_name을 직접 설정하는 로직이 _create_llm_client에 있거나,
                    # model_name을 인자로 받는 _create_llm_client_for_model 같은 함수가 필요.
                    # 현재 구조에서는, 요청된 모델이 provider의 기본 모델과 다르면, provider의 기본 모델을 사용하게 됨.
                    # 이를 수정하려면 _create_llm_client가 model_name을 오버라이드 할 수 있도록 해야함.
                    # 지금은 요청된 model_name을 사용하기 위해 새 클라이언트를 만들도록 수정
                    try:
                        # 임시로 model_name을 오버라이드하여 클라이언트 생성 시도
                        # 이는 settings 객체를 직접 수정하지 않으면서 특정 모델을 사용하기 위함
                        original_model_name = settings.LLM_PROVIDERS[target_provider_name].model_name
                        settings.LLM_PROVIDERS[target_provider_name].model_name = model_name
                        llm_client_instance = self._create_llm_client(target_provider_name)
                        settings.LLM_PROVIDERS[target_provider_name].model_name = original_model_name # 복원
                    except Exception as e_create:
                        self.logger.error(f"Failed to create client for specific model '{model_name}': {e_create}. Using primary LLM.")
                        llm_client_instance = self.primary_llm
                else:
                    self.logger.error(f"Could not find provider settings for '{target_provider_name}'. Using primary LLM.")
                    llm_client_instance = self.primary_llm
        else:
            llm_client_instance = self.primary_llm

        is_using_primary_configured_llm = (llm_client_instance == self.primary_llm)
        # model_name이 명시적으로 요청되었는지 여부
        specific_model_requested = model_name is not None

        # 호출 파라미터 설정
        invoke_params = {}
        # 현재 선택된 llm_client_instance의 provider와 model_name을 사용
        current_provider_name = llm_client_instance.provider
        actual_model_name = llm_client_instance.model_name
        
        provider_config = settings.LLM_PROVIDERS.get(current_provider_name)

        # 온도 설정
        if temperature is not None:
            invoke_params['temperature'] = temperature
        elif provider_config and hasattr(provider_config, 'temperature'):
            invoke_params['temperature'] = provider_config.temperature
        else: # llm_client_instance의 기본값 (ChatOpenAI 등은 기본 temperature 가짐)
            invoke_params['temperature'] = getattr(llm_client_instance._llm, 'temperature', 0.7)


        # 최대 토큰 설정
        default_max_tokens_val = 1024 # 최종 기본값
        if provider_config and hasattr(provider_config, 'max_tokens'):
            default_max_tokens_val = provider_config.max_tokens
        elif hasattr(llm_client_instance._llm, 'max_tokens'): # OpenAI
             default_max_tokens_val = getattr(llm_client_instance._llm, 'max_tokens', default_max_tokens_val)
        elif hasattr(llm_client_instance._llm, 'max_tokens_to_sample'): # Anthropic
             default_max_tokens_val = getattr(llm_client_instance._llm, 'max_tokens_to_sample', default_max_tokens_val)


        final_max_tokens = max_tokens if max_tokens is not None else default_max_tokens_val

        if current_provider_name == "anthropic":
            invoke_params['max_tokens_to_sample'] = final_max_tokens
        else: # openai 및 기타
            invoke_params['max_tokens'] = final_max_tokens
        
        invoke_params.update(kwargs) # 사용자가 제공한 추가 kwargs로 덮어쓰기

        retry_decorator = retry(
            wait=wait_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(int(settings.LLM_MAX_RETRIES) + 1),
            reraise=False
        )

        # OpenTelemetry 스팬 시작
        with tracer.start_as_current_span(
            "llm.request",
            attributes={
                "model": actual_model_name, # llm_client_instance.model_name 사용 권장
                "temperature": invoke_params.get('temperature'),
                "max_tokens": invoke_params.get('max_tokens', invoke_params.get('max_tokens_to_sample'))
            }
        ) as span:
            try:
                # 기본 또는 선택된 LLM 호출 시도
                primary_callable = partial(self._invoke_llm_attempt, llm_client_instance, messages, invoke_params)
                decorated_callable = retry_decorator(primary_callable)
                return await decorated_callable()

            except RetryError as primary_err: # 재시도가 모두 실패하면 이 블록으로 진입
                actual_original_exception = primary_err.last_attempt.exception() if primary_err.last_attempt else primary_err

                span.set_attribute("primary_error_type", type(actual_original_exception).__name__)
                span.set_attribute("primary_error_message", str(actual_original_exception))
                
                # 폴백 조건 확인
                can_use_fallback = is_using_primary_configured_llm and self.fallback_llm and not specific_model_requested
                
                if can_use_fallback:
                    self.logger.warning(
                        f"Primary LLM ('{actual_model_name}') failed after all retries with error: '{actual_original_exception}'. "
                        f"Switching to fallback LLM ('{self.fallback_llm.model_name}')."
                    )
                    span.set_attribute("status", "fallback_triggered")

                    # 폴백 LLM용 파라미터 준비 (기존 로직과 유사하게 구성)
                    fallback_provider_name = self.fallback_llm.provider
                    fallback_config = settings.LLM_PROVIDERS.get(fallback_provider_name)
                    fallback_invoke_params = {}
                    
                    # 폴백용 온도 설정
                    if temperature is not None:
                        fallback_invoke_params['temperature'] = temperature
                    elif fallback_config and hasattr(fallback_config, 'temperature'):
                        fallback_invoke_params['temperature'] = fallback_config.temperature
                    else:
                        fallback_invoke_params['temperature'] = getattr(self.fallback_llm._llm, 'temperature', 0.7)

                    # 폴백용 최대 토큰 설정 (기존 로직 활용)
                    fallback_default_tokens = 1024 # 기본값
                    if fallback_config and hasattr(fallback_config, 'max_tokens'):
                        fallback_default_tokens = fallback_config.max_tokens

                    elif hasattr(self.fallback_llm._llm, 'max_tokens'): # OpenAI
                        fallback_default_tokens = getattr(self.fallback_llm._llm, 'max_tokens', fallback_default_tokens)
                    elif hasattr(self.fallback_llm._llm, 'max_tokens_to_sample'): # Anthropic
                        fallback_default_tokens = getattr(self.fallback_llm._llm, 'max_tokens_to_sample', fallback_default_tokens)

                    final_fallback_max_tokens = max_tokens if max_tokens is not None else fallback_default_tokens

                    if fallback_provider_name == "anthropic":
                        fallback_invoke_params['max_tokens_to_sample'] = final_fallback_max_tokens
                    else:
                        fallback_invoke_params['max_tokens'] = final_fallback_max_tokens
                    
                    fallback_invoke_params.update(kwargs)

                    # 폴백 LLM 호출 (OpenTelemetry 스팬 포함)
                    with tracer.start_as_current_span(
                        "llm.fallback_request",
                        attributes={
                            "primary_model": actual_model_name,
                            "fallback_model": self.fallback_llm.model_name,
                            "primary_original_error_type": type(actual_original_exception).__name__,
                            "primary_original_error": str(actual_original_exception),
                            "temperature": fallback_invoke_params.get('temperature'),
                            "max_tokens": fallback_invoke_params.get('max_tokens', fallback_invoke_params.get('max_tokens_to_sample'))
                        }
                    ) as fallback_span:
                        try:
                            fallback_callable = partial(self._invoke_llm_attempt, self.fallback_llm, messages, fallback_invoke_params)
                            # 폴백 시도에도 재시도 로직 적용 (동일한 retry_decorator 사용)
                            decorated_fallback_callable = retry_decorator(fallback_callable)
                            return await decorated_fallback_callable()
                        except RetryError as fallback_retry_err: # 폴백도 모든 재시도 실패
                            fallback_original_exception = fallback_retry_err.last_attempt.exception() if fallback_retry_err.last_attempt else fallback_retry_err
                            fallback_span.set_attribute("fallback_error_type", type(fallback_original_exception).__name__)
                            fallback_span.set_attribute("fallback_error_message", str(fallback_original_exception))
                            self.logger.error(
                                f"Fallback LLM ('{self.fallback_llm.model_name}') also failed after all retries with error: '{fallback_original_exception}'"
                            )
                            raise LLMError(
                                message=(
                                    f"Both primary ('{actual_model_name}') and fallback ('{self.fallback_llm.model_name}') LLMs failed. "
                                    f"Primary error: {actual_original_exception}. Fallback error: {fallback_original_exception}."
                                ),
                                original_error=fallback_original_exception 
                            ) from fallback_retry_err # 또는 from fallback_original_exception
                        except Exception as fallback_unexpected_e: # 폴백 중 RetryError 외의 예기치 않은 오류
                            fallback_span.set_attribute("fallback_unexpected_error", str(fallback_unexpected_e))
                            self.logger.error(f"Unexpected error with fallback LLM ('{self.fallback_llm.model_name}'): {fallback_unexpected_e}", exc_info=True)
                            raise LLMError(
                                message=f"Fallback LLM ('{self.fallback_llm.model_name}') failed with an unexpected error: {fallback_unexpected_e}",
                                original_error=fallback_unexpected_e
                            ) from fallback_unexpected_e
                else: # 폴백 사용 불가 또는 조건 미충족
                    self.logger.error(
                        f"LLM call for model '{actual_model_name}' failed after all retries with error: '{actual_original_exception}'. "
                        f"No fallback initiated (is_using_primary_configured_llm: {is_using_primary_configured_llm}, "
                        f"fallback_llm_exists: {self.fallback_llm is not None}, specific_model_requested: {specific_model_requested})."
                    )
                    raise LLMError(
                        message=f"LLM call failed for '{actual_model_name}' after all retries: {actual_original_exception}",
                        original_error=actual_original_exception
                    ) from actual_original_exception # 원본 예외를 체이닝

            except Exception as e: 
                # 이 블록은 tenacity 와 무관한 오류(예: 파라미터 준비 단계에서의 오류) 또는
                # RetryError 처리 중 발생한 또 다른 예기치 않은 오류를 처리합니다.
                # reraise=False로 인해 _invoke_llm_attempt에서 발생한 대부분의 오류는 RetryError로 감싸지므로
                # 이 블록의 실행 빈도는 매우 낮을 것으로 예상됩니다.
                self.logger.error(f"An unexpected error occurred in generate_response for LLM '{actual_model_name}' before or after retry logic: {e}", exc_info=True)
                span.set_attribute("generate_response_unexpected_error", str(e))
                raise LLMError(
                    message=f"LLM call failed unexpectedly during setup or teardown for '{actual_model_name}'. Error: {e}",
                    original_error=e
                ) from e

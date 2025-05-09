import asyncio
from typing import Dict, List, Optional, Any

from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import LLMError
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
logger = get_logger(__name__)
settings = get_settings()

class _LLMWrapper:
    """
    내부 LangChain LLM을 감싸고, 공통 인터페이스를 제공하는 래퍼 클래스
    """
    def __init__(self, llm: Any, model_name: str):
        self._llm = llm
        self.model_name = model_name

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._llm, attr)

class LLMClient:
    """
    LLM API 클라이언트 어댑터
    """

    def __init__(self):
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
        if provider == "openai":
            llm = ChatOpenAI(
                openai_api_key=provider_settings.api_key,
                model_name=name,
                request_timeout=int(settings.LLM_REQUEST_TIMEOUT),
            )
        elif provider == "anthropic":
            llm = ChatAnthropic(
                anthropic_api_key=provider_settings.api_key,
                model=name,
                request_timeout=int(settings.LLM_REQUEST_TIMEOUT),
            )
        else:
            raise ValueError(f"지원하지 않는 LLM Provider: {provider}")

        return _LLMWrapper(llm, name)

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        LLM에게 메시지 목록을 전달하고 응답을 받음 (재시도 및 오류 처리 포함)
        """
        from tenacity import retry, stop_after_attempt, wait_exponential, RetryError # 위치는 함수 내부 유지

        # 모델 인스턴스 결정
        if model_name:
            # model_name으로 특정 LLM 공급자 및 모델을 사용하도록 설정
            # 이 부분은 settings.py 와 _create_llm_client_with_specific_model (존재한다면) 구현에 따라 다름
            # 현재 코드에는 _create_llm_client_with_specific_model이 없으므로,
            # 이 부분을 self._create_llm_client를 활용하거나, 해당 함수를 만들어야 함
            # 여기서는 llm_client_instance를 가져오는 로직이 있다고 가정.
            # 만약 _create_llm_client_with_specific_model이 없다면 아래와 같이 수정 가능
            target_provider_name = settings.LLM_MODEL_PROVIDER_MAP.get(model_name)
            if not target_provider_name:
                logger.warning(f"Model name '{model_name}' not found in LLM_MODEL_PROVIDER_MAP. Using primary LLM.")
                llm_client_instance = self.primary_llm
            else:
                # 여기서 model_name 을 사용하는 _create_llm_client (또는 유사 함수) 호출 필요
                # 지금은 provider_settings 를 직접 찾아 모델 이름만 교체하는 임시 방안 사용
                provider_settings_for_model = settings.LLM_PROVIDERS.get(target_provider_name)
                if provider_settings_for_model and provider_settings_for_model.model_name == model_name:
                     llm_client_instance = self._create_llm_client(target_provider_name) # model_name이 일치하면 해당 provider 사용
                elif provider_settings_for_model: # provider는 맞지만 모델명이 다르면 경고 후 해당 provider의 기본 모델 사용
                    logger.warning(f"Model name '{model_name}' for provider '{target_provider_name}' does not match provider's default model '{provider_settings_for_model.model_name}'. Using provider's default.")
                    llm_client_instance = self._create_llm_client(target_provider_name)
                else: # provider도 못찾으면 primary 사용
                    logger.error(f"Could not find provider settings for '{target_provider_name}'. Using primary LLM.")
                    llm_client_instance = self.primary_llm
        else:
            llm_client_instance = self.primary_llm

        # LLM 호출 파라미터 구성
        invoke_params = {}
        current_provider_name = "unknown_provider" # 기본값
        # llm_client_instance 에서 provider 이름을 가져오는 방식이 필요. _LLMWrapper 에 provider_name 속성 추가 가정
        # 또는 llm_client_instance._llm의 클래스 타입으로 유추 가능

        # llm_client_instance가 _LLMWrapper 타입이라고 가정하고 model_name을 가져옵니다.
        actual_model_name = getattr(llm_client_instance, 'model_name', 'unknown_model')

        # provider_name을 알아내기 위한 개선된 방법 (예시)
        # 이는 _LLMWrapper나 _create_llm_client에서 provider 정보를 저장/전달해야 정확해짐
        if llm_client_instance == self.primary_llm:
            current_provider_name = settings.PRIMARY_LLM_PROVIDER
        elif self.fallback_llm and llm_client_instance == self.fallback_llm:
            current_provider_name = settings.FALLBACK_LLM_PROVIDER
        elif model_name: # model_name으로 특정 인스턴스를 가져온 경우
             # 이 경우, llm_client_instance를 생성할 때 provider 정보를 알 수 있어야 함
             # 임시로 LLM_MODEL_PROVIDER_MAP을 다시 사용
            current_provider_name = settings.LLM_MODEL_PROVIDER_MAP.get(actual_model_name, settings.PRIMARY_LLM_PROVIDER)


        provider_config = settings.LLM_PROVIDERS.get(current_provider_name)

        if provider_config:
            if temperature is not None:
                invoke_params['temperature'] = temperature
            else:
                invoke_params['temperature'] = getattr(provider_config, 'temperature',
                                                    getattr(llm_client_instance, 'temperature', 0.7)) # llm_client_instance에 temperature 기본값이 있을 수 있음

            if max_tokens is not None:
                invoke_params['max_tokens'] = max_tokens
            else:
                invoke_params['max_tokens'] = getattr(provider_config, 'max_tokens',
                                                    getattr(llm_client_instance, 'max_tokens', 1024)) # llm_client_instance에 max_tokens 기본값이 있을 수 있음
        else: # provider_config가 없는 경우 (예: 테스트 목킹) 기본값 설정
            invoke_params['temperature'] = temperature if temperature is not None else 0.7
            invoke_params['max_tokens'] = max_tokens if max_tokens is not None else 1024


        # 추가 파라미터 병합
        invoke_params.update(kwargs)

        async def _invoke_llm_with_params(): # 함수 이름 명확히 변경
            try:
                # llm_client_instance가 LangChain LLM 객체를 직접 참조한다고 가정 (_LLMWrapper._llm)
                # 또는 _LLMWrapper에 ainvoke가 모든 파라미터를 전달하도록 수정되어 있어야 함.
                # 현재 _LLMWrapper는 __getattr__를 사용하므로, 원본 LLM의 ainvoke가 호출됨.
                # LangChain의 BaseChatModel.ainvoke는 messages 외의 파라미터를 config로 받을 수 있음.
                # 하지만 temperature, max_tokens 등은 생성자 레벨에서 설정하는 것이 일반적.
                # 만약 실행 시점에 변경하려면, ChatOpenAI(temperature=..., max_tokens=...).ainvoke() 처럼
                # 새 인스턴스를 만들거나, 해당 옵션을 지원하는 방식으로 호출해야 함.
                # 여기서는 llm_client_instance가 해당 파라미터를 지원하는 ainvoke를 가졌다고 가정.
                # 또는, 이 파라미터들을 messages와 함께 전달하는 대신,
                # llm_client_instance를 생성할 때 이 값들을 설정해야 할 수 있음.
                # 현재 코드는 invoke_params를 **kwargs처럼 전달하고 있으므로,
                # llm_client_instance.ainvoke(messages, **invoke_params)가 되어야 함.

                response = await llm_client_instance.ainvoke(messages, **invoke_params) # 수정: invoke_params 전달

                if hasattr(response, "content"):
                    return response.content
                if isinstance(response, str):
                    return response
                logger.warning(f"LLM response type is {type(response)}, converting to string. Response: {response}")
                return str(response)
            except Exception as e:
                logger.error(f"LLM invocation failed (Model: {actual_model_name}, Params: {invoke_params}): {e}", exc_info=True)
                raise # 에러를 다시 발생시켜 tenacity가 재시도 처리하도록 함

        # Tenacity retry 데코레이터를 _invoke_llm_with_params 함수에 직접 적용
        # @retry(...) 데코레이터는 함수 정의 시점에 적용되어야 하므로, 내부 함수로 만들고 호출.
        
        # retry 로직을 함수 호출 부분으로 이동
        async def aninvoked_with_retry():
            return await retry(
                wait=wait_exponential(multiplier=1, min=1, max=10),
                stop=stop_after_attempt(int(settings.LLM_MAX_RETRIES) + 1),
                reraise=True
            )(_invoke_llm_with_params)() # _invoke_llm_with_params를 호출

        try:
            # OpenTelemetry 추적은 실제 LLM 호출을 감싸도록 함
            with tracer.start_as_current_span(
                "llm.request",
                attributes={
                    "model": actual_model_name,
                    "temperature": invoke_params.get('temperature'),
                    "max_tokens": invoke_params.get('max_tokens')
                }
            ):
                response_content = await aninvoked_with_retry()
                # 응답 내용 로깅 (민감 정보 주의)
                # logger.debug(f"LLM Response (Model: {actual_model_name}): {response_content[:200]}...") # 너무 길면 일부만
                return response_content

        except RetryError as e: # Tenacity가 모든 재시도 실패 후 발생시키는 에러
            logger.error(f"LLM call failed after all retries (Model: {actual_model_name}): {e.last_attempt.exception() if e.last_attempt else e}", exc_info=True)
            raise LLMError(message=f"LLM call failed after all retries: {e.last_attempt.exception() if e.last_attempt else e}", original_error=e.last_attempt.exception() if e.last_attempt else e) from e
        except Exception as e: # 재시도 로직 외부의 예외 (예: 파라미터 구성 오류 등)
            logger.error(f"An unexpected error occurred before or after LLM retry logic (Model: {actual_model_name}): {e}", exc_info=True)
            raise LLMError(message=f"LLM call failed (unexpected): {e}", original_error=e) from e
        
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """주 LLM을 사용하여 채팅 형식으로 메시지를 전달하고 응답을 받음"""
        try:
            return await self.generate_response(messages)
        except LLMError as primary_err:
            logger.warning(f"주 LLM 실패, 폴백 LLM으로 대체 시도: {primary_err}")
            if not self.fallback_llm:
                raise primary_err
            @retry(
                wait=wait_exponential(),
                stop=stop_after_attempt(int(settings.LLM_MAX_RETRIES) + 1),
            )
            async def _invoke_fallback():
                response = await self.fallback_llm.ainvoke(messages)
                if hasattr(response, "content"):
                    return response.content
                if isinstance(response, str):
                    return response
                return str(response)

        try:
            return await _invoke_fallback()
        except RetryError as fallback_err:
            logger.error(f"폴백 LLM 재시도 실패: {fallback_err}")
            raise LLMError(message=f"폴백 LLM 호출 실패: {fallback_err}", original_error=fallback_err)

    async def create_prompt(self, template: str, **kwargs: Any) -> str:
        """프롬프트 템플릿을 사용하여 프롬프트 생성"""
        from langchain.prompts import PromptTemplate
        variables = list(kwargs.keys())
        try:
            prompt = PromptTemplate(template=template, input_variables=variables)
            return prompt.format(**kwargs)
        except Exception as e:
            logger.error(f"프롬프트 생성 실패 (템플릿: {template}): {e}")
            raise LLMError(message=f"프롬프트 생성 실패: {e}", original_error=e)    

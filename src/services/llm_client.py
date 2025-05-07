import asyncio
from typing import Dict, List, Optional, Any

from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.errors import LLMError

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
        from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
        
        # 모델 인스턴스 결정
        if model_name:
            target_provider = settings.LLM_MODEL_PROVIDER_MAP.get(model_name, settings.PRIMARY_LLM_PROVIDER)
            llm_client_instance = self._create_llm_client_with_specific_model(target_provider, model_name)
        else:
            llm_client_instance = self.primary_llm
        
        # LLM 호출 파라미터 구성
        invoke_params = {}
        
        # 안전하게 provider_name 접근 (테스트 환경 고려)
        provider_name = getattr(llm_client_instance, 'provider_name', settings.PRIMARY_LLM_PROVIDER)
        provider_config = settings.LLM_PROVIDERS.get(provider_name)
        
        if provider_config:
            if temperature is not None:
                invoke_params['temperature'] = temperature
            else:
                invoke_params['temperature'] = getattr(provider_config, 'temperature', 
                                                    getattr(llm_client_instance, 'temperature', 0.7))
            
            if max_tokens is not None:
                invoke_params['max_tokens'] = max_tokens
            else:
                invoke_params['max_tokens'] = getattr(provider_config, 'max_tokens', 
                                                    getattr(llm_client_instance, 'max_tokens', 1024))
        
        # 추가 파라미터 병합
        invoke_params.update(kwargs)
        
        async def _invoke():
            try:
                response = await llm_client_instance.ainvoke(messages, **invoke_params)
                if hasattr(response, "content"):
                    return response.content
                if isinstance(response, str):
                    return response
                return str(response)
            except Exception as e:
                logger.error(f"LLM 호출 실패 (모델: {getattr(llm_client_instance, 'model_name', 'unknown')}): {e}")
                raise e
        
        try:
            # 원래 코드의 방식으로 retry 데코레이터 사용
            invoked = retry(
                wait=wait_exponential(multiplier=1, min=1, max=10),
                stop=stop_after_attempt(int(settings.LLM_MAX_RETRIES) + 1),
                reraise=True
            )(_invoke)
            return await invoked()
        
        except RetryError as e:
            raise LLMError(message=f"LLM 호출 모든 재시도 실패: {e}", original_error=e)
        except Exception as e:
            raise LLMError(message=f"LLM 호출 실패 (재시도 전): {e}", original_error=e)

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
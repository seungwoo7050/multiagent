

"""
LLM 요청 실패 시 예비 모델(Fallback Model)로 전환하는 로직을 관리하는 모듈입니다.

이 모듈은 `LLMFallbackHandler` 클래스를 제공하여, 기본(Primary) LLM 모델 호출 실패 시
미리 정의된 예비 모델 목록을 순차적으로 시도하는 기능을 구현합니다.
오류 유형(`failure_detector`), 모델 선택(`selector`), 성능 추적(`performance`) 모듈과
연동하여 지능적인 폴백 결정을 내릴 수 있습니다.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast


from src.config.logger import get_logger_with_context, ContextLoggerAdapter
from src.config.settings import get_settings

from src.llm.base import BaseLLMAdapter
from src.llm.adapters import get_adapter as get_llm_adapter_instance
from src.llm.selector import select_models
from src.llm.failure_detector import should_fallback_immediately



from src.config.errors import LLMError, ErrorCode
from src.config.metrics import track_llm_fallback, track_llm_error


settings = get_settings()
logger: ContextLoggerAdapter = get_logger_with_context(__name__)


class LLMFallbackHandler:
    """
    LLM 요청 실패 시 예비 모델로 전환하는 로직을 처리하는 핸들러 클래스입니다.
    Primary 모델 실행 실패 시 설정된 Fallback 모델 목록을 순차적으로 시도합니다.
    """

    def __init__(
        self,


        track_metrics: bool = True
    ):
        """
        LLMFallbackHandler 인스턴스를 초기화합니다.

        Args:
            track_metrics (bool): 폴백 발생 시 관련 메트릭을 추적할지 여부.
        """


        self.track_metrics = track_metrics
        logger.debug("LLMFallbackHandler initialized.")


    async def execute_with_fallback(
        self,

        requested_model: Optional[str] = None,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        use_cache: Optional[bool] = None,

        **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """
        주어진 프롬프트로 LLM 호출을 실행하되, 실패 시 예비 모델로 자동 전환합니다.
        내부적으로 `select_models`를 호출하여 시도할 모델 목록(Primary + Fallbacks)을 결정합니다.

        Args:
            requested_model (Optional[str]): 사용자가 요청한 특정 모델 이름.
            prompt (Union[str, List[Dict[str, str]]]): LLM에 전달할 프롬프트.
            max_tokens (Optional[int]): 최대 생성 토큰 수.
            temperature (Optional[float]): 샘플링 온도.
            top_p (Optional[float]): Top-P 샘플링.
            stop_sequences (Optional[List[str]]): 생성 중단 시퀀스.
            use_cache (Optional[bool]): LLM 어댑터 레벨 캐시 사용 여부.
            **kwargs (Any): LLM 어댑터 `generate` 메서드에 전달될 추가 키워드 인자.

        Returns:
            Tuple[str, Dict[str, Any]]: (성공적으로 응답한 모델 이름, 해당 모델의 LLM 응답 딕셔너리).

        Raises:
            LLMError: Primary 및 모든 Fallback 모델 호출이 실패한 경우.
            ValueError: 모델 선택 단계에서 오류가 발생한 경우.
        """
        global logger

        trace_id = kwargs.get('trace_id')
        logger = get_logger_with_context(__name__, trace_id=trace_id)

        start_time = time.monotonic()
        errors_encountered: Dict[str, str] = {}


        try:

             primary_model, fallback_models = await select_models(
                 requested_model=requested_model

             )

             models_to_try: List[str] = [primary_model] + fallback_models
             logger.info(f"Fallback execution sequence: Primary='{primary_model}', Fallbacks={fallback_models}")
        except ValueError as e:
             logger.error(f"Failed to select models for fallback execution: {e}")
             raise



        for model_name in models_to_try:
             model_start_time = time.monotonic()
             adapter: Optional[BaseLLMAdapter] = None

             try:

                  logger.debug(f"Attempting LLM call with model: {model_name}")

                  adapter = get_llm_adapter_instance(
                      model=model_name,

                      timeout=kwargs.get('timeout', settings.REQUEST_TIMEOUT),
                      max_retries=0
                  )




                  result: Dict[str, Any] = await adapter.generate(
                      prompt=prompt,
                      max_tokens=max_tokens,
                      temperature=temperature,
                      top_p=top_p,
                      stop_sequences=stop_sequences,
                      use_cache=use_cache,
                      retry_on_failure=False,
                      **kwargs
                  )


                  success_model_name: str = model_name
                  process_duration_s: float = time.monotonic() - model_start_time
                  total_duration_s: float = time.monotonic() - start_time
                  logger.info(f"LLM call successful with model '{success_model_name}' in {process_duration_s:.3f}s (Total: {total_duration_s:.3f}s)")


                  if self.track_metrics and success_model_name != primary_model:

                       track_llm_fallback(primary_model, success_model_name)


                  return success_model_name, result

             except Exception as e:

                  process_duration_s = time.monotonic() - model_start_time
                  error_message = str(e)
                  errors_encountered[model_name] = error_message
                  logger.warning(f"Model '{model_name}' failed after {process_duration_s:.3f}s. Error: {error_message}")



                  if should_fallback_immediately(e):
                       logger.warning(f"Error type ({type(e).__name__}) suggests immediate fallback for model '{model_name}'.")

                  else:


                       logger.info(f"Potentially retryable error for model '{model_name}'. Proceeding to next fallback model (no retry implemented here).")


                  if self.track_metrics and adapter:
                       error_type = type(e).__name__
                       if isinstance(e, LLMError) and e.code:
                            error_type = e.code.value if isinstance(e.code, ErrorCode) else str(e.code)




                       track_llm_error(model_name, adapter.provider, error_type)


                  continue


        total_duration_s = time.monotonic() - start_time
        final_error_msg = f"All LLM models failed ({', '.join(models_to_try)}) after {total_duration_s:.3f}s."
        logger.error(final_error_msg, extra={"errors": errors_encountered})


        raise LLMError(
            code=ErrorCode.LLM_API_ERROR,
            message=final_error_msg,
            details={"models_tried": models_to_try, "errors": errors_encountered}
        )



_fallback_handler_instance: Optional[LLMFallbackHandler] = None
_fallback_handler_lock = asyncio.Lock()

async def get_fallback_handler() -> LLMFallbackHandler:
    """
    LLMFallbackHandler의 싱글턴 인스턴스를 가져옵니다.

    Returns:
        LLMFallbackHandler: 싱글턴 인스턴스.
    """
    global _fallback_handler_instance
    if _fallback_handler_instance is None:
        async with _fallback_handler_lock:
            if _fallback_handler_instance is None:


                _fallback_handler_instance = LLMFallbackHandler()
                logger.info("Singleton LLMFallbackHandler instance created.")

    if _fallback_handler_instance is None:
         raise RuntimeError("Failed to create LLMFallbackHandler instance.")
    return _fallback_handler_instance



async def execute_llm_with_fallback(
    prompt: Union[str, List[Dict[str, str]]],
    requested_model: Optional[str] = None,
    **kwargs: Any
) -> Tuple[str, Dict[str, Any]]:
    """
    LLMFallbackHandler를 사용하여 LLM 호출을 실행하는 편의 함수입니다.

    Args:
        prompt: LLM에 전달할 프롬프트.
        requested_model: 사용자가 요청한 모델 (선택적).
        **kwargs: `LLMFallbackHandler.execute_with_fallback`에 전달될 모든 인자.

    Returns:
        Tuple[str, Dict[str, Any]]: (성공 모델 이름, LLM 응답 딕셔너리).

    Raises:
        LLMError: 모든 모델 호출 실패 시.
    """
    handler = await get_fallback_handler()

    return await handler.execute_with_fallback(
        requested_model=requested_model,
        prompt=prompt,
        max_tokens=kwargs.get('max_tokens'),
        temperature=kwargs.get('temperature'),
        top_p=kwargs.get('top_p'),
        stop_sequences=kwargs.get('stop_sequences'),
        use_cache=kwargs.get('use_cache'),
        **kwargs
    )


import asyncio
from src.llm.failure_detector import should_fallback_immediately
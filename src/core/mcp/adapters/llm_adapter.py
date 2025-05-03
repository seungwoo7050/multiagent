import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from pydantic import Field

from src.config.errors import ErrorCode, LLMError
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.core.exceptions import SerializationError
from src.core.mcp.adapter_base import MCPAdapterBase
from src.core.mcp.llm.context_performance import get_context_labels
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.llm.base import BaseLLMAdapter
import src.llm as llm
import src.llm.selector as selector

settings = get_settings()
logger = get_logger(__name__)
metrics = get_metrics_manager()

class LLMInputContext(BaseContextSchema):
    model: str
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    use_cache: bool = True
    retry_on_failure: bool = True

class LLMOutputContext(BaseContextSchema):
    success: bool
    result_text: Optional[str] = None
    choices: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None
    model_used: Optional[str] = None

class LLMAdapter(MCPAdapterBase):

    def __init__(self, target_component: Optional[BaseLLMAdapter]=None):
        super().__init__(target_component=target_component, mcp_context_type=LLMInputContext)

    async def _get_target_llm_adapter(self, model_name: str) -> BaseLLMAdapter:
        if self.target_component and isinstance(self.target_component, BaseLLMAdapter) and (self.target_component.model == model_name):
            logger.debug(f'Using pre-configured LLM adapter instance for model {model_name}')
            if not self.target_component.initialized:
                await self.target_component.ensure_initialized()
            return self.target_component
        else:
            logger.debug(f'Dynamically getting LLM adapter instance for model {model_name}')
            try:
                adapter_instance = llm.get_adapter(model_name)
                if not isinstance(adapter_instance, BaseLLMAdapter):
                    raise TypeError(f'Factory returned unexpected type {type(adapter_instance)} for model {model_name}')
                await adapter_instance.ensure_initialized()
                return adapter_instance
            except Exception as e:
                raise LLMError(f'Failed to get LLM adapter instance for model {model_name}: {e}', model=model_name, provider=settings.LLM_MODEL_PROVIDER_MAP.get(model_name, 'unknown'), original_error=e, code=ErrorCode.LLM_PROVIDER_ERROR) from e

    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(context, LLMInputContext):
            raise ValueError(f'Incompatible context type: Expected LLMInputContext, got {type(context).__name__}')
        llm_input_context: LLMInputContext = cast(LLMInputContext, context)
        primary_model = llm_input_context.model
        prompt_content = llm_input_context.prompt
        if not primary_model:
            raise ValueError("Missing 'model' information in LLMInputContext")
        if not prompt_content:
            raise ValueError("Missing 'prompt' (or 'messages') content in LLMInputContext")
        logger.debug(f"Adapting LLMInputContext (ID: {llm_input_context.context_id}) for model '{primary_model}'")
        try:
            generate_args: Dict[str, Any] = {'primary_model': primary_model, 'fallback_models': llm_input_context.metadata.get('fallback_models', []), 'prompt': llm_input_context.prompt if llm_input_context.prompt else llm_input_context.messages, 'max_tokens': llm_input_context.parameters.get('max_tokens'), 'temperature': llm_input_context.parameters.get('temperature'), 'top_p': llm_input_context.parameters.get('top_p'), 'additional_params': {}, 'timeout': llm_input_context.parameters.get('timeout', settings.REQUEST_TIMEOUT), 'track_metrics': llm_input_context.metadata.get('track_metrics', True)}
            generate_args = {k: v for k, v in generate_args.items() if v is not None}
            standard_keys = {'max_tokens', 'temperature', 'top_p', 'timeout'}
            for key, value in llm_input_context.parameters.items():
                if key not in standard_keys and value is not None:
                    generate_args.setdefault('additional_params', {})[key] = value
            stop_sequences = llm_input_context.parameters.get('stop_sequences')
            if stop_sequences:
                generate_args.setdefault('additional_params', {})['stop_sequences'] = stop_sequences
            return generate_args
        except Exception as e:
            logger.error(f'Error adapting LLMInputContext for model {primary_model}: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt input context for LLM call: {e}', original_error=e)

    async def adapt_output(self, component_output: Any, original_context: Optional[ContextProtocol]=None, **kwargs: Any) -> LLMOutputContext:
        success: bool = False
        result_text: Optional[str] = None
        choices: Optional[List[Dict[str, Any]]] = None
        usage: Optional[Dict[str, int]] = None
        error_message: Optional[str] = None
        model_used: Optional[str] = kwargs.get('model_used')
        logger.debug(f'Adapting LLM component output (type: {type(component_output).__name__}) to LLMOutputContext')
        if isinstance(component_output, Exception):
            success = False
            error_message = str(component_output)
            # LLMError 인 경우 model 속성이 있으면 덮어쓰기
            if isinstance(component_output, LLMError):
                model_attr = getattr(component_output, 'model', None)
                if model_attr:
                    model_used = model_attr
            logger.error(f'LLM component execution failed: {error_message}')
        elif isinstance(component_output, dict):
            llm_result: Dict[str, Any] = cast(Dict[str, Any], component_output)
            success = True
            logger.debug(f'LLM component returned result dict (ID: {llm_result.get('id', 'N/A')})')
            choices = llm_result.get('choices', [])
            if choices and isinstance(choices, list) and (len(choices) > 0) and isinstance(choices[0], dict):
                message_content = choices[0].get('message', {}).get('content')
                text_content = choices[0].get('text')
                result_text = message_content if message_content is not None else text_content
            usage = llm_result.get('usage')
            model_used = llm_result.get('model', model_used)
        else:
            success = False
            error_message = f'Unexpected output type from LLM component: {type(component_output).__name__}'
            logger.error(error_message)
        try:
            context_data: Dict[str, Any] = {'success': success, 'result_text': result_text, 'choices': choices if choices else None, 'usage': usage if usage else None, 'error_message': error_message, 'model_used': model_used, 'metadata': {'llm_request_id': component_output.get('id') if success and isinstance(component_output, dict) else None}}
            original_context_id: Optional[str] = None
            if isinstance(original_context, BaseContextSchema):
                original_context_id = getattr(original_context, 'context_id', None)
            output_context = LLMOutputContext(**context_data, context_id=original_context_id)
            return output_context
        except Exception as e:
            logger.error(f'Error adapting LLM output dict to LLMOutputContext: {e}', exc_info=True)
            raise SerializationError(f'Could not adapt LLM output to MCP context: {e}', original_error=e)
        
    async def _track_llm_metrics(
        self,
        model: str,
        result_data: Dict[str, Any],
        duration: float,
        context_labels: Dict[str, str],
        should_track_metrics: bool = True
    ) -> None:
        """Track metrics for successful LLM request."""
        if not should_track_metrics or not model:
            return

        # 어댑터 레벨 방어막: context_labels 제거
        context_labels = {}

        # provider 결정 (unknown 모델도 안전 처리)
        provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, 'unknown')

        # 토큰 사용량 정보
        usage = result_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)

        # 1) 성공 요청 수 카운트
        metrics.track_llm(
            'requests',
            model=model,
            provider=provider
        )

        # 2) 요청 처리 시간 기록
        metrics.track_llm(
            'duration',
            value=duration,
            model=model,
            provider=provider
        )

        # 3) 프롬프트 토큰 사용량
        if prompt_tokens > 0:
            metrics.track_llm(
                'tokens',
                value=prompt_tokens,
                model=model,
                provider=provider,
                type='prompt'
            )

        # 4) 컴플리션 토큰 사용량
        if completion_tokens > 0:
            metrics.track_llm(
                'tokens',
                value=completion_tokens,
                model=model,
                provider=provider,
                type='completion'
            )


    def _track_llm_error(
        self,
        model: str,
        error_code: Union[ErrorCode, str],
        context_labels: Dict[str, str],
        should_track_metrics: bool = True
    ) -> None:
        """Track metrics for failed LLM request."""
        if not should_track_metrics:
            return

        # 어댑터 레벨 방어막: context_labels 제거
        context_labels = {}

        # provider 결정
        provider = settings.LLM_MODEL_PROVIDER_MAP.get(model, 'unknown')
        # 에러 코드 문자열화
        error_type = (
            error_code.value
            if isinstance(error_code, ErrorCode)
            else str(error_code)
        )

        # 에러 카운트 기록
        metrics.track_llm(
            'errors',
            model=model,
            provider=provider,
            error_type=error_type
        )

    async def _select_models(self, requested_model: str) -> Tuple[str, List[str]]:
        """Select primary and fallback models to use."""
        return await selector.select_models(
            requested_model=requested_model,
            task_context=None
        )

    async def _execute_llm_request(
        self, 
        primary_model: str, 
        fallback_models: List[str],
        call_args: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute LLM request by fetching the adapter and calling its execute() method."""
        logger.debug(f"Executing LLM call via provider.execute (Model: '{primary_model}')")
        # 테스트에서 patch된 get_adapter → provider 를 가져와서 execute() 호출
        adapter = await self._get_target_llm_adapter(primary_model)
        result_data = await adapter.execute(call_args)
        return primary_model, result_data


    async def process_with_mcp(self, context: ContextProtocol, **kwargs: Any) -> ContextProtocol:
        llm_result_data: Optional[Dict[str, Any]] = None
        component_error: Optional[Exception] = None
        model_actually_used: Optional[str] = None
        primary_model_selected: str = 'unknown'
        start_time = time.time()
        context_labels = get_context_labels(context) # Get context labels early

        try:
            # Validate input context
            if not isinstance(context, LLMInputContext):
                raise ValueError(f'Expected LLMInputContext, got {type(context).__name__}')
            llm_input_context: LLMInputContext = cast(LLMInputContext, context)

            # Select model(s)
            requested_model = llm_input_context.model
            primary_model_selected, fallback_models_selected = await self._select_models(requested_model)
            logger.info(f'Selected models - Primary: {primary_model_selected}, Fallbacks: {fallback_models_selected}')

            # Prepare execution
            common_call_args = await self.adapt_input(context, **kwargs)
            should_track_metrics = common_call_args.pop('track_metrics', True)
            common_call_args.pop('primary_model', None)
            common_call_args.pop('fallback_models', None)

            # Execute with fallbacks
            model_actually_used, llm_result_data = await self._execute_llm_request(
                primary_model_selected,
                fallback_models_selected,
                common_call_args
            )

            # Track metrics for successful execution
            duration = time.time() - start_time
            if llm_result_data:
                self._track_llm_metrics(
                    model_actually_used, 
                    llm_result_data, 
                    duration, 
                    context_labels,
                    should_track_metrics
                )

            logger.info(f'LLM call completed using model: {model_actually_used}')

        except LLMError as lle:
            # Handle LLM-specific errors
            duration = time.time() - start_time
            logger.error(f'All LLM models failed during execution: {lle.message}', extra=lle.to_dict())
            component_error = lle
            model_actually_used = lle.model or primary_model_selected
            
            # Track error metrics
            should_track_metrics_on_error = llm_input_context.metadata.get('track_metrics', True) if 'llm_input_context' in locals() else True
            if should_track_metrics_on_error:
                self._track_llm_error(model_actually_used, lle.code or ErrorCode.SYSTEM_ERROR, context_labels)

        except Exception as e:
            # Handle unexpected errors
            duration = time.time() - start_time
            logger.error(f'Error during MCP processing for LLM: {e}', exc_info=True)
            component_error = e
            model_actually_used = primary_model_selected
            
            # Track unknown error
            should_track_metrics_on_error = llm_input_context.metadata.get('track_metrics', True) if 'llm_input_context' in locals() else True
            if should_track_metrics_on_error:
                self._track_llm_error(model_actually_used, ErrorCode.SYSTEM_ERROR, context_labels)

        # Adapt output regardless of success or failure
        output_context: LLMOutputContext = await self.adapt_output(
            component_output=component_error if component_error else llm_result_data,
            original_context=context,
            model_used=model_actually_used,
            **kwargs
        )

        # Ensure output context reflects the error state if an error occurred
        if component_error and output_context.success:
            output_context.success = False
            output_context.error_message = str(component_error)
            # Update model_used in output context if the error provided one
            if isinstance(component_error, LLMError) and component_error.model:
                output_context.model_used = component_error.model

        return output_context
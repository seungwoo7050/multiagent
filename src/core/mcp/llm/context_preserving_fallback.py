import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from src.config.logger import get_logger_with_context, ContextLoggerAdapter
from src.config.settings import get_settings
from src.llm.base import BaseLLMAdapter
from src.llm.adapters import get_adapter as get_llm_adapter_instance
from src.llm.selector import select_models
from src.llm.failure_detector import should_fallback_immediately
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import LLMInputContext, LLMOutputContext
from src.core.mcp.adapters.llm_adapter import LLMAdapter
from src.core.mcp.llm.context_transform import transform_llm_input_for_model
from src.config.errors import LLMError, ErrorCode
from src.config.metrics import track_llm_fallback, track_llm_error
settings = get_settings()
logger: ContextLoggerAdapter = get_logger_with_context(__name__)

async def execute_mcp_llm_with_context_preserving_fallback(requested_model: Optional[str], original_prompt_or_messages: Union[str, List[Dict[str, str]]], mcp_llm_adapter: LLMAdapter, parameters: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, Any]]=None, overall_timeout: Optional[float]=None, track_metrics: bool=True) -> Tuple[str, LLMOutputContext]:
    global logger
    trace_id = metadata.get('trace_id') if metadata else None
    logger = get_logger_with_context(__name__, trace_id=trace_id, operation='context_preserving_fallback')
    start_time = time.monotonic()
    errors_encountered: Dict[str, str] = {}
    parameters = parameters or {}
    metadata = metadata or {}
    try:
        primary_model, fallback_models = await select_models(requested_model=requested_model)
        models_to_try: List[str] = [primary_model] + fallback_models
        logger.info(f"Fallback sequence selected: Primary='{primary_model}', Fallbacks={fallback_models}")
    except ValueError as e:
        logger.error(f'Failed to select models: {e}')
        raise
    for model_name in models_to_try:
        model_start_time = time.monotonic()
        llm_output_context: Optional[LLMOutputContext] = None
        target_llm_adapter: Optional[BaseLLMAdapter] = None
        elapsed_time = model_start_time - start_time
        if overall_timeout is not None and elapsed_time >= overall_timeout:
            logger.warning(f"Overall fallback timeout ({overall_timeout}s) exceeded before trying model '{model_name}'.")
            break
        remaining_timeout = overall_timeout - elapsed_time if overall_timeout is not None else None
        try:
            logger.debug(f'Getting adapter for model: {model_name}')
            target_llm_adapter = get_llm_adapter_instance(model=model_name, timeout=min(remaining_timeout, parameters.get('timeout', settings.REQUEST_TIMEOUT)) if remaining_timeout else parameters.get('timeout', settings.REQUEST_TIMEOUT))
            await target_llm_adapter.ensure_initialized()
            logger.debug(f'Transforming input context for model: {model_name}')
            transformed_input = await transform_llm_input_for_model(original_input=original_prompt_or_messages, target_adapter=target_llm_adapter)
            current_input_context = LLMInputContext(model=model_name, prompt=transformed_input if isinstance(transformed_input, str) else None, messages=transformed_input if isinstance(transformed_input, list) else None, parameters=parameters, use_cache=metadata.get('use_cache', True), metadata=metadata)
            logger.info(f'Executing LLM call with model: {model_name}')
            llm_output_context = await mcp_llm_adapter.process_with_mcp(current_input_context)
            if llm_output_context.success:
                success_model_name: str = model_name
                process_duration_s: float = time.monotonic() - model_start_time
                total_duration_s: float = time.monotonic() - start_time
                logger.info(f"LLM call successful with model '{success_model_name}' in {process_duration_s:.3f}s (Total: {total_duration_s:.3f}s)")
                if track_metrics and success_model_name != primary_model:
                    track_llm_fallback(primary_model, success_model_name)
                return (success_model_name, llm_output_context)
            else:
                error_message = llm_output_context.error_message or f"Model '{model_name}' failed without specific error message."
                errors_encountered[model_name] = error_message
                logger.warning(f"Model '{model_name}' execution failed (returned success=False): {error_message}")
                if track_metrics and target_llm_adapter:
                    track_llm_error(model_name, target_llm_adapter.provider, 'LLMExecutionError')
                if 'rate limit' in error_message.lower():
                    logger.info('Rate limit detected. Proceeding to next fallback.')
                continue
        except LLMError as lle:
            process_duration_s = time.monotonic() - model_start_time
            error_message = str(lle)
            errors_encountered[model_name] = error_message
            logger.warning(f"Model '{model_name}' failed after {process_duration_s:.3f}s with LLMError ({lle.code}): {lle.message}")
            provider = getattr(target_llm_adapter, 'provider', 'unknown') if target_llm_adapter else 'unknown'
            error_code_str = lle.code.value if isinstance(lle.code, ErrorCode) else str(lle.code)
            if track_metrics:
                track_llm_error(model_name, provider, error_code_str)
            if should_fallback_immediately(lle):
                logger.info(f'Error type ({error_code_str}) suggests immediate fallback. Proceeding.')
            else:
                logger.info(f'Potentially retryable error ({error_code_str}). Proceeding to next fallback model.')
            continue
        except Exception as e:
            process_duration_s = time.monotonic() - model_start_time
            error_message = str(e)
            errors_encountered[model_name] = error_message
            provider = getattr(target_llm_adapter, 'provider', 'unknown') if target_llm_adapter else 'unknown'
            logger.error(f"Unexpected error during fallback execution for model '{model_name}' after {process_duration_s:.3f}s: {e}", exc_info=True)
            if track_metrics:
                track_llm_error(model_name, provider, type(e).__name__)
            continue
    total_duration_s = time.monotonic() - start_time
    final_error_msg = f'All LLM models failed during context-preserving fallback ({', '.join(models_to_try)}) after {total_duration_s:.3f}s.'
    logger.error(final_error_msg, extra={'errors': errors_encountered})
    raise LLMError(code=ErrorCode.LLM_API_ERROR, message=final_error_msg, details={'models_tried': models_to_try, 'errors': errors_encountered})
from typing import Coroutine
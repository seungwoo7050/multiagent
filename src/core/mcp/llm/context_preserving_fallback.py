import time
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config.errors import ErrorCode, LLMError
from src.config.logger import ContextLoggerAdapter, get_logger_with_context
from src.config.metrics import get_metrics_manager  # Import metrics manager
from src.config.settings import get_settings
from src.core.mcp.adapters.llm_adapter import (LLMAdapter, LLMInputContext,
                                               LLMOutputContext)
from src.core.mcp.llm.context_transform import transform_llm_input_for_model
from src.llm.adapters import get_adapter as get_llm_adapter_instance
from src.llm.base import BaseLLMAdapter
from src.llm.failure_detector import should_fallback_immediately
from src.llm.selector import select_models

settings = get_settings()
logger: ContextLoggerAdapter = get_logger_with_context(__name__)
metrics = get_metrics_manager() # Get metrics manager instance

async def execute_mcp_llm_with_context_preserving_fallback(
    requested_model: Optional[str],
    original_prompt_or_messages: Union[str, List[Dict[str, str]]],
    mcp_llm_adapter: LLMAdapter,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overall_timeout: Optional[float] = None,
    track_metrics: bool = True, # Keep track_metrics flag
) -> Tuple[str, LLMOutputContext]:

    global logger
    trace_id = metadata.get('trace_id') if metadata else None
    # Update logger context for this specific operation
    logger = get_logger_with_context(__name__, trace_id=trace_id, operation='context_preserving_fallback')

    start_time = time.monotonic()
    errors_encountered: Dict[str, str] = {}
    parameters = parameters or {}
    metadata = metadata or {}
    primary_model: str = "unknown" # Initialize primary_model

    try:
        primary_model, fallback_models = await select_models(requested_model=requested_model)
        models_to_try: List[str] = [primary_model] + fallback_models
        logger.info(f"Fallback sequence selected: Primary='{primary_model}', Fallbacks={fallback_models}")
    except ValueError as e:
        logger.error(f'Failed to select models: {e}')
        # Optionally track this selection error if desired
        # if track_metrics:
        #    metrics.track_llm('errors', model=requested_model or 'unknown', provider='model_selector', error_type='SelectionFailure')
        raise # Re-raise the original error

    for model_name in models_to_try:
        model_start_time = time.monotonic()
        llm_output_context: Optional[LLMOutputContext] = None
        target_llm_adapter: Optional[BaseLLMAdapter] = None
        provider: str = 'unknown' # Initialize provider for error tracking

        elapsed_time = model_start_time - start_time
        if overall_timeout is not None and elapsed_time >= overall_timeout:
            logger.warning(f"Overall fallback timeout ({overall_timeout}s) exceeded before trying model '{model_name}'.")
            break # Exit loop if timeout exceeded

        remaining_timeout = overall_timeout - elapsed_time if overall_timeout is not None else None
        current_timeout = min(remaining_timeout, parameters.get('timeout', settings.REQUEST_TIMEOUT)) if remaining_timeout else parameters.get('timeout', settings.REQUEST_TIMEOUT)

        try:
            logger.debug(f'Getting adapter for model: {model_name}')
            target_llm_adapter = get_llm_adapter_instance(
                model=model_name,
                timeout=current_timeout
            )
            await target_llm_adapter.ensure_initialized()
            # Get provider after ensuring adapter is initialized
            provider = getattr(target_llm_adapter, 'provider', 'unknown')

            logger.debug(f'Transforming input context for model: {model_name}')
            transformed_input = await transform_llm_input_for_model(
                original_input=original_prompt_or_messages,
                target_adapter=target_llm_adapter,
            )

            # Create the input context for this specific attempt
            current_input_context = LLMInputContext(
                model=model_name,
                prompt=transformed_input if isinstance(transformed_input, str) else None,
                messages=transformed_input if isinstance(transformed_input, list) else None,
                parameters=parameters,
                use_cache=metadata.get('use_cache', True), # Propagate cache setting
                metadata=metadata, # Propagate original metadata
                # Add specific metadata for this attempt if needed
                # metadata={**metadata, 'attempt_model': model_name}
            )
            # Pass the track_metrics flag down via metadata if process_with_mcp uses it
            current_input_context.metadata['track_metrics'] = track_metrics

            logger.info(f'Executing LLM call with model: {model_name} (Provider: {provider})')
            # Call the MCP adapter's process method
            llm_output_context = await mcp_llm_adapter.process_with_mcp(current_input_context)

            if llm_output_context.success:
                success_model_name: str = model_name
                process_duration_s: float = time.monotonic() - model_start_time
                total_duration_s: float = time.monotonic() - start_time
                logger.info(f"LLM call successful with model '{success_model_name}' in {process_duration_s:.3f}s (Total: {total_duration_s:.3f}s)")

                # Track fallback metric if a fallback model succeeded
                if track_metrics and success_model_name != primary_model:
                    metrics.track_llm('fallbacks', from_model=primary_model, to_model=success_model_name)
                    logger.info(f"Fallback occurred from '{primary_model}' to '{success_model_name}'")

                # Return the successful model name and the output context
                return (success_model_name, llm_output_context)
            else:
                # Handle cases where process_with_mcp returns success=False
                error_message = llm_output_context.error_message or f"Model '{model_name}' failed without specific error message."
                errors_encountered[model_name] = error_message
                logger.warning(f"Model '{model_name}' execution failed (returned success=False): {error_message}")

                # Track this specific failure type if needed
                if track_metrics:
                     metrics.track_llm('errors', model=model_name, provider=provider, error_type='LLMExecutionSoftError') # Or a more specific code if available in output_context

                # Check if the error message indicates a reason to stop or continue
                if 'rate limit' in error_message.lower():
                     logger.info('Rate limit detected. Proceeding to next fallback.')
                # Add other conditions to break or continue as needed
                continue # Continue to the next model

        except LLMError as lle:
            process_duration_s = time.monotonic() - model_start_time
            error_message = str(lle)
            errors_encountered[model_name] = error_message
            # Ensure provider is captured if possible, even on error
            provider = getattr(target_llm_adapter, 'provider', provider) # Keep previous provider if adapter failed early
            error_code_str = lle.code.value if isinstance(lle.code, ErrorCode) else str(lle.code)
            logger.warning(f"Model '{model_name}' failed after {process_duration_s:.3f}s with LLMError ({error_code_str}): {lle.message}")

            # Track LLMError
            if track_metrics:
                metrics.track_llm('errors', model=model_name, provider=provider, error_type=error_code_str)

            # Decide whether to continue based on the error type
            if should_fallback_immediately(lle):
                logger.info(f'Error type ({error_code_str}) suggests immediate fallback. Proceeding.')
            else:
                logger.info(f'Potentially retryable error ({error_code_str}). Proceeding to next fallback model.')
            continue # Continue to the next model

        except Exception as e:
            process_duration_s = time.monotonic() - model_start_time
            error_message = str(e)
            errors_encountered[model_name] = error_message
            # Ensure provider is captured if possible
            provider = getattr(target_llm_adapter, 'provider', provider)
            logger.error(f"Unexpected error during fallback execution for model '{model_name}' after {process_duration_s:.3f}s: {e}", exc_info=True)

            # Track unexpected error
            if track_metrics:
                metrics.track_llm('errors', model=model_name, provider=provider, error_type=type(e).__name__)

            continue # Continue to the next model

    # If the loop finishes without returning, all models failed
    total_duration_s = time.monotonic() - start_time
    final_error_msg = f'All LLM models failed during context-preserving fallback ({", ".join(models_to_try)}) after {total_duration_s:.3f}s.'
    logger.error(final_error_msg, extra={'errors': errors_encountered})

    # Raise a final error summarizing the situation
    raise LLMError(
        code=ErrorCode.LLM_API_ERROR, # Or a more specific code like ALL_MODELS_FAILED
        message=final_error_msg,
        details={'models_tried': models_to_try, 'errors': errors_encountered},
        # Set model to the primary model that was initially selected
        model=primary_model
    )
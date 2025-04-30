"""
LLM Fallback Handler Module

This module manages the logic for switching to fallback models when primary LLM requests fail.
It provides the LLMFallbackHandler class to sequentially try backup models
when the primary model encounters an error.
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
from src.config.metrics import get_metrics_manager

metrics = get_metrics_manager()
settings = get_settings()
logger: ContextLoggerAdapter = get_logger_with_context(__name__)


class LLMFallbackHandler:
    """
    Handler for managing LLM fallback logic when requests fail.
    
    When a primary model execution fails, this handler tries fallback models
    in sequence until a successful response is received or all models fail.
    """

    def __init__(self, track_metrics: bool = True):
        """
        Initialize the LLMFallbackHandler instance.

        Args:
            track_metrics: Whether to track metrics for fallback operations
        """
        self.track_metrics = track_metrics
        logger.debug("LLMFallbackHandler initialized.")

    async def execute_with_fallback(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        requested_model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute an LLM request with automatic fallback to backup models if needed.
        
        Uses select_models() to determine the primary model and fallback sequence.

        Args:
            requested_model: Specific model requested by the user (optional)
            prompt: LLM input prompt (text or message list)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Sequences to stop generation
            use_cache: Whether to use adapter-level cache
            **kwargs: Additional parameters for the LLM adapter

        Returns:
            Tuple[str, Dict[str, Any]]: (Successful model name, LLM response)

        Raises:
            LLMError: If all primary and fallback models fail
        """
        global logger

        trace_id = kwargs.get('trace_id')
        logger = get_logger_with_context(__name__, trace_id=trace_id)

        start_time = time.monotonic()
        errors_encountered: Dict[str, str] = {}

        try:
            # Get primary and fallback models
            primary_model, fallback_models = await select_models(
                requested_model=requested_model
            )

            models_to_try: List[str] = [primary_model] + fallback_models
            logger.info(f"Fallback execution sequence: Primary='{primary_model}', Fallbacks={fallback_models}")
        except ValueError as e:
            logger.error(f"Failed to select models for fallback execution: {e}")
            raise

        # Try each model in sequence
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

                # Track fallback metrics if not using primary model
                if self.track_metrics and success_model_name != primary_model:
                    metrics.track_llm('fallbacks', from_model=primary_model, to_model=success_model_name)

                return success_model_name, result

            except Exception as e:
                process_duration_s = time.monotonic() - model_start_time
                error_message = str(e)
                errors_encountered[model_name] = error_message
                logger.warning(f"Model '{model_name}' failed after {process_duration_s:.3f}s. Error: {error_message}")

                # Check if error suggests immediate fallback
                if should_fallback_immediately(e):
                    logger.warning(f"Error type ({type(e).__name__}) suggests immediate fallback for model '{model_name}'.")
                else:
                    logger.info(f"Potentially retryable error for model '{model_name}'. Proceeding to next fallback model (no retry implemented here).")

                # Track error metrics
                if self.track_metrics and adapter:
                    error_type = type(e).__name__
                    if isinstance(e, LLMError) and e.code:
                        error_type = e.code.value if isinstance(e.code, ErrorCode) else str(e.code)

                    metrics.track_llm('errors', model=model_name, provider=adapter.provider, error_type=error_type)

                continue

        # If we get here, all models failed
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
    Get the singleton LLMFallbackHandler instance.

    Returns:
        LLMFallbackHandler: Shared handler instance
        
    Raises:
        RuntimeError: If handler creation fails
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
    Convenience function to execute an LLM request with fallback handling.

    Args:
        prompt: LLM input prompt
        requested_model: Specific model requested by the user
        **kwargs: Additional parameters for execute_with_fallback

    Returns:
        Tuple[str, Dict[str, Any]]: (Successful model name, LLM response)
        
    Raises:
        LLMError: If all models fail
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
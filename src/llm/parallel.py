"""
Parallel execution utilities for LLM operations.

This module provides functions for executing LLM requests in parallel
and for racing multiple models to get the fastest response.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
from functools import wraps

from src.llm.adapters import get_adapter
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import (
    MEMORY_OPERATION_DURATION,
    LLM_REQUESTS_TOTAL,
    LLM_FALLBACKS_TOTAL,
    track_llm_fallback,
    timed_metric
)
from src.config.errors import LLMError, ErrorCode

settings = get_settings()
logger = get_logger(__name__)

T = TypeVar('T')


@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "execute_parallel"})
async def execute_parallel(
    operations: List[Callable[[], Any]],
    timeout: Optional[float] = None,
    return_exceptions: bool = False,
    cancel_on_first_exception: bool = False,
    cancel_on_first_result: bool = False,
    max_wait_time: float = 1.0
) -> List[Any]:
    """
    Execute multiple async operations in parallel.
    
    Args:
        operations: List of async callables to execute
        timeout: Total timeout for all operations
        return_exceptions: Whether to return exceptions instead of raising them
        cancel_on_first_exception: Whether to cancel all tasks if one raises an exception
        cancel_on_first_result: Whether to cancel remaining tasks when first result is ready
        max_wait_time: Maximum time to wait in each asyncio.wait call to prevent starvation
        
    Returns:
        List[Any]: Results in the same order as the operations
    """
    if not operations:
        return []
    
    # Default timeout from settings if not specified
    if timeout is None:
        timeout = settings.REQUEST_TIMEOUT
    
    # Use a more efficient task tracking mechanism
    task_map = {}
    results = [None] * len(operations)
    
    start_time = time.time()
    deadline = start_time + timeout
    
    # Create and track tasks with their indices
    for i, op in enumerate(operations):
        task = asyncio.create_task(op())
        task_map[task] = i
    
    pending = set(task_map.keys())
    exceptions = []
    
    try:
        while pending and time.time() < deadline:
            # Calculate remaining time
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break
            
            # Use shorter of remaining time or max_wait_time to prevent starvation
            wait_time = min(remaining_time, max_wait_time)
            
            # Wait for tasks to complete
            done, pending = await asyncio.wait(
                pending,
                timeout=wait_time,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if not done:  # Timeout occurred
                continue
            
            # Process completed tasks
            for task in done:
                idx = task_map[task]
                
                try:
                    result = task.result()
                    results[idx] = result
                    
                    # Cancel remaining tasks if requested
                    if cancel_on_first_result:
                        for t in pending:
                            t.cancel()
                        pending = set()
                        break
                        
                except Exception as e:
                    logger.debug(f"Task {idx} failed with error: {str(e)}")
                    if return_exceptions:
                        results[idx] = e
                    else:
                        exceptions.append((idx, e))
                        
                        # Cancel remaining tasks if requested
                        if cancel_on_first_exception:
                            for t in pending:
                                t.cancel()
                            pending = set()
                            break
            
            # If there are exceptions and we're not returning them, raise the first one
            if exceptions and not return_exceptions:
                idx, error = exceptions[0]
                logger.error(f"Operation {idx} failed: {str(error)}")
                raise error
    
    finally:
        # Determine which tasks timed out
        timed_out = [task_map[t] for t in pending]
        if timed_out:
            logger.warning(f"Operations timed out: {timed_out}")
            
        # Ensure all pending tasks are properly cancelled
        for task in pending:
            task.cancel()
        
        # Wait for cancellations to complete
        if pending:
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            except (asyncio.CancelledError, Exception):
                pass
    
    # Mark timed out tasks with timeout exceptions if requested
    if return_exceptions:
        for i in range(len(results)):
            if results[i] is None and i in [task_map[t] for t in pending]:
                results[i] = asyncio.TimeoutError(f"Operation {i} timed out after {timeout}s")
    
    return results


async def _create_adapters_concurrently(models: List[str]) -> Dict[str, Any]:
    """Create multiple adapters concurrently."""
    async def create_single_adapter(model: str):
        try:
            adapter = await get_adapter(model)
            return model, adapter
        except Exception as e:
            logger.warning(f"Failed to create adapter for model {model}: {str(e)}")
            return model, e
    
    # Create all adapters concurrently
    tasks = [create_single_adapter(model) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out failures and construct the map
    adapter_map = {}
    for model, result in results:
        if not isinstance(result, Exception):
            adapter_map[model] = result
    
    return adapter_map


@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "race_models"})
async def race_models(
    models: List[str],
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    track_metrics: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Race multiple models to get the fastest response.
    
    Args:
        models: List of model names to race
        prompt: The prompt to send to each model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        additional_params: Additional parameters to pass to the LLM
        timeout: Timeout for the entire race
        track_metrics: Whether to track metrics
        
    Returns:
        Tuple[str, Dict[str, Any]]: Winning model name and its response
        
    Raises:
        LLMError: If all models fail
    """
    if not models:
        raise ValueError("At least one model must be provided")
    
    # Default timeout from settings if not specified
    if timeout is None:
        timeout = settings.REQUEST_TIMEOUT
    
    start_time = time.time()
    additional_params = additional_params or {}
    
    # Create all adapters concurrently
    adapter_map = await _create_adapters_concurrently(models)
    if not adapter_map:
        raise LLMError(
            code=ErrorCode.LLM_API_ERROR,
            message="Failed to create adapters for all models",
            details={"models": models}
        )
    
    # Set up per-model operations
    operations = []
    
    for model in models:  # Use original models list to maintain order
        adapter = adapter_map.get(model)
        if not adapter:
            continue
            
        async def generate_for_model(m=model, a=adapter):
            if track_metrics:
                LLM_REQUESTS_TOTAL.labels(model=m).inc()
            
            try:
                response = await a.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **additional_params
                )
                return {"model": m, "response": response, "success": True}
            except Exception as e:
                error = e if isinstance(e, LLMError) else LLMError(
                    code=ErrorCode.LLM_API_ERROR,
                    message=f"Error in model {m}: {str(e)}",
                    model=m,
                    original_error=e
                )
                return {"model": m, "error": error, "success": False}
        
        operations.append(generate_for_model)
    
    # Execute the race with our operations
    adapter_creation_time = time.time() - start_time
    remaining_timeout = max(0.1, timeout - adapter_creation_time)
    
    # Execute operations with race semantics
    results = await execute_parallel(
        operations=operations,
        timeout=remaining_timeout,
        return_exceptions=True,
        cancel_on_first_result=True
    )
    
    # Process results - look for the first successful one
    successful_results = []
    errors = []
    
    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
            continue
            
        if result and result.get("success"):
            successful_results.append(result)
    
    # Find the winner (first successful result)
    if successful_results:
        winner = successful_results[0]
        winning_model = winner["model"]
        response = winner["response"]
        
        # Track metrics if this wasn't the primary model
        if track_metrics and winning_model != models[0]:
            track_llm_fallback(models[0], winning_model)
            
        race_time = time.time() - start_time
        logger.info(f"Model race won by {winning_model} in {race_time:.3f}s")
        
        return winning_model, response
    
    # If we get here, all models failed
    error_details = {}
    for result in results:
        if isinstance(result, Exception):
            error_details[str(id(result))] = str(result)
        elif result and not result.get("success"):
            error_details[result["model"]] = str(result.get("error", "Unknown error"))
    
    raise LLMError(
        code=ErrorCode.LLM_API_ERROR,
        message=f"All models failed in race: {', '.join(models)}",
        details={"models": models, "errors": error_details}
    )


@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "execute_with_fallbacks"})
async def execute_with_fallbacks(
    primary_model: str,
    fallback_models: List[str],
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    track_metrics: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Execute a request with the primary model, falling back to others if it fails.
    
    Args:
        primary_model: Primary model to try first
        fallback_models: Ordered list of fallback models to try
        prompt: The prompt to send to each model
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        additional_params: Additional parameters to pass to the LLM
        timeout: Timeout for the entire operation
        track_metrics: Whether to track metrics
        
    Returns:
        Tuple[str, Dict[str, Any]]: Model name used and its response
        
    Raises:
        LLMError: If all models fail
    """
    # Combine primary and fallbacks into a single list
    all_models = [primary_model] + fallback_models
    
    # Default timeout from settings if not specified
    if timeout is None:
        timeout = settings.REQUEST_TIMEOUT
    
    start_time = time.time()
    additional_params = additional_params or {}
    errors = {}
    
    # Create adapters concurrently at the start to save time
    adapter_map = await _create_adapters_concurrently(all_models)
    adapter_creation_time = time.time() - start_time
    
    # Try each model in sequence
    for model in all_models:
        # Skip if we couldn't create an adapter
        if model not in adapter_map:
            errors[model] = f"Failed to create adapter for model {model}"
            continue
            
        adapter = adapter_map[model]
        model_start_time = time.time()
        
        # Calculate remaining timeout
        elapsed = time.time() - start_time
        remaining_timeout = timeout - elapsed
        
        if remaining_timeout <= 0.1:  # Give at least 100ms
            errors[model] = "Timeout exceeded before trying model"
            continue
        
        # Track the request if metrics are enabled
        if track_metrics:
            LLM_REQUESTS_TOTAL.labels(model=model).inc()
        
        try:
            # Execute with a timeout wrapper to ensure we don't exceed our budget
            async def execute_with_timeout():
                return await adapter.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **(additional_params or {})
                )
                
            # Create a task with timeout
            task = asyncio.create_task(execute_with_timeout())
            
            try:
                result = await asyncio.wait_for(task, timeout=remaining_timeout)
            except asyncio.TimeoutError:
                # Cancel the task if it timed out
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                raise TimeoutError(f"Model {model} timed out after {remaining_timeout:.2f}s")
            
            # Success - track metrics and return
            if track_metrics and model != primary_model:
                track_llm_fallback(primary_model, model)
            
            model_time = time.time() - model_start_time
            logger.info(f"Model {model} succeeded in {model_time:.3f}s" +
                        (f" (fallback from {primary_model})" if model != primary_model else ""))
            
            return model, result
            
        except Exception as e:
            # Log the error and continue to the next model
            errors[model] = e
            logger.warning(f"Model {model} failed, trying next fallback: {str(e)}")
    
    # If we get here, all models failed
    total_time = time.time() - start_time
    
    # Format detailed error information
    error_details = {}
    for model, error in errors.items():
        if isinstance(error, Exception):
            error_details[model] = {
                "type": type(error).__name__,
                "message": str(error)
            }
        else:
            error_details[model] = {"message": str(error)}
    
    raise LLMError(
        code=ErrorCode.LLM_API_ERROR,
        message=f"All models failed in {total_time:.3f}s",
        details={"models": all_models, "errors": error_details}
    )
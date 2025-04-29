import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast, Coroutine
from functools import wraps
from src.llm.adapters import get_adapter
from src.llm.base import BaseLLMAdapter
from src.core.mcp.llm.context_transform import transform_llm_input_for_model
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import MEMORY_OPERATION_DURATION, LLM_REQUESTS_TOTAL, LLM_FALLBACKS_TOTAL, track_llm_fallback, timed_metric
from src.config.errors import LLMError, ErrorCode
settings = get_settings()
logger = get_logger(__name__)
T = TypeVar('T')

@timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'execute_parallel'})
async def execute_parallel(operations: List[Callable[[], Coroutine[Any, Any, Any]]], timeout: Optional[float]=None, return_exceptions: bool=False, cancel_on_first_exception: bool=False, cancel_on_first_result: bool=False, max_wait_time: float=1.0) -> List[Any]:
    if not operations:
        return []
    if timeout is None:
        timeout = settings.REQUEST_TIMEOUT
    task_map: Dict[asyncio.Task[Any], int] = {}
    results: List[Optional[Any]] = [None] * len(operations)
    exceptions: List[Tuple[int, Exception]] = []
    pending: Set[asyncio.Task[Any]] = set()
    start_time: float = time.monotonic()
    deadline: float = start_time + timeout
    for i, op in enumerate(operations):
        try:
            task: asyncio.Task[Any] = asyncio.create_task(op())
            task_map[task] = i
            pending.add(task)
        except Exception as task_creation_e:
            logger.error(f'Failed to create task for operation at index {i}: {task_creation_e}')
            if return_exceptions:
                results[i] = task_creation_e
            else:
                exceptions.append((i, task_creation_e))
    try:
        while pending and time.monotonic() < deadline:
            remaining_time: float = deadline - time.monotonic()
            if remaining_time <= 0:
                logger.debug('Parallel execution timeout reached.')
                break
            wait_time: float = min(remaining_time, max_wait_time)
            done: Set[asyncio.Task[Any]]
            pending: Set[asyncio.Task[Any]]
            try:
                done, pending = await asyncio.wait(pending, timeout=wait_time, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                logger.warning('execute_parallel wait was cancelled.')
                raise
            if not done:
                logger.debug(f'No tasks completed within wait_time ({wait_time}s).')
                continue
            for task in done:
                idx: int = task_map[task]
                try:
                    result: Any = task.result()
                    results[idx] = result
                    logger.debug(f'Task {idx} completed successfully.')
                    if cancel_on_first_result:
                        logger.info(f'First result received (task {idx}). Cancelling remaining {len(pending)} tasks.')
                        for t in pending:
                            t.cancel()
                        pending = set()
                        break
                except asyncio.CancelledError:
                    logger.warning(f'Task {idx} was cancelled.')
                    if return_exceptions:
                        results[idx] = asyncio.CancelledError(f'Task {idx} was cancelled.')
                    else:
                        exceptions.append((idx, asyncio.CancelledError(f'Task {idx} was cancelled.')))
                except Exception as e:
                    logger.debug(f'Task {idx} failed with error: {e}')
                    if return_exceptions:
                        results[idx] = e
                    else:
                        exceptions.append((idx, e))
                        if cancel_on_first_exception:
                            logger.warning(f'First exception occurred (task {idx}). Cancelling remaining {len(pending)} tasks.')
                            for t in pending:
                                t.cancel()
                            pending = set()
                            break
            if not pending:
                break
            if exceptions and (not return_exceptions) and cancel_on_first_exception:
                break
            if cancel_on_first_result and any((res is not None for res in results)):
                break
        if exceptions and (not return_exceptions):
            idx, error = exceptions[0]
            logger.error(f'Parallel operation {idx} failed: {error}')
            raise error
    finally:
        timed_out_indices: List[int] = [task_map[t] for t in pending]
        if timed_out_indices:
            logger.warning(f'Parallel operations timed out or were cancelled for indices: {timed_out_indices}')
        if pending:
            logger.debug(f'Ensuring cancellation of {len(pending)} remaining pending tasks.')
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    if return_exceptions:
        current_time = time.monotonic()
        if current_time >= deadline:
            for i in range(len(results)):
                if results[i] is None and i in timed_out_indices:
                    results[i] = asyncio.TimeoutError(f'Operation {i} timed out after {timeout}s')
    return cast(List[Any], results)

async def _create_adapters_concurrently(models: List[str]) -> Dict[str, BaseLLMAdapter]:

    async def create_single_adapter(model: str) -> Tuple[str, Optional[BaseLLMAdapter]]:
        try:
            adapter = get_adapter(model)
            await adapter.ensure_initialized()
            return (model, adapter)
        except Exception as e:
            logger.warning(f"Failed to create or initialize adapter for model '{model}': {str(e)}")
            return (model, None)
    adapter_results: List[Tuple[str, Optional[BaseLLMAdapter]]] = await asyncio.gather(*[create_single_adapter(model) for model in models], return_exceptions=False)
    adapter_map: Dict[str, BaseLLMAdapter] = {}
    for model, adapter_instance in adapter_results:
        if adapter_instance is not None:
            adapter_map[model] = adapter_instance
    logger.debug(f'Concurrently created {len(adapter_map)} adapters out of {len(models)} requested.')
    return adapter_map

@timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'race_models'})
async def race_models(models: List[str], prompt: Union[str, List[Dict[str, str]]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, additional_params: Optional[Dict[str, Any]]=None, timeout: Optional[float]=None, track_metrics: bool=True) -> Tuple[str, Dict[str, Any]]:
    if not models:
        raise ValueError('At least one model must be provided for race_models')
    if timeout is None:
        timeout = settings.REQUEST_TIMEOUT
    start_time = time.monotonic()
    additional_params = additional_params or {}
    try:
        adapter_map: Dict[str, BaseLLMAdapter] = await _create_adapters_concurrently(models)
        if not adapter_map:
            raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message='Failed to create adapters for any of the specified models', details={'models': models})
        adapter_creation_time: float = time.monotonic() - start_time
        remaining_timeout: float = max(0.1, timeout - adapter_creation_time)
        logger.debug(f'Adapters created in {adapter_creation_time:.3f}s. Remaining race timeout: {remaining_timeout:.3f}s')
    except Exception as adapter_err:
        raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message='Error during adapter creation for race_models', original_error=adapter_err) from adapter_err
    operations: List[Callable[[], Coroutine[Any, Any, Any]]] = []
    valid_models_in_race: List[str] = list(adapter_map.keys())
    for model_name in valid_models_in_race:
        adapter: BaseLLMAdapter = adapter_map[model_name]

        async def generate_for_model(adapter_instance: BaseLLMAdapter=adapter, model_id: str=model_name) -> Dict[str, Any]:
            if track_metrics:
                track_llm_request(model_id, adapter_instance.provider)
            try:
                response: Dict[str, Any] = await adapter_instance.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **additional_params)
                return {'model': model_id, 'response': response, 'success': True}
            except Exception as e:
                logger.warning(f"Model '{model_id}' failed during race: {e}")
                error: LLMError = e if isinstance(e, LLMError) else LLMError(code=ErrorCode.LLM_API_ERROR, message=f'Error in model {model_id} during race: {str(e)}', model=model_id, provider=adapter_instance.provider, original_error=e)
                return {'model': model_id, 'error': error, 'success': False}
        operations.append(generate_for_model)
    logger.info(f'Starting model race for {len(operations)} models with timeout {remaining_timeout:.3f}s')
    results: List[Any] = await execute_parallel(operations=operations, timeout=remaining_timeout, return_exceptions=True, cancel_on_first_result=True)
    successful_result: Optional[Dict[str, Any]] = None
    errors_encountered: Dict[str, str] = {}
    for i, result_or_exc in enumerate(results):
        model_raced = valid_models_in_race[i]
        if isinstance(result_or_exc, Exception):
            errors_encountered[model_raced] = f'Execution Error: {str(result_or_exc)}'
            continue
        if isinstance(result_or_exc, dict):
            if result_or_exc.get('success'):
                successful_result = result_or_exc
                break
            else:
                error_obj = result_or_exc.get('error')
                errors_encountered[model_raced] = str(error_obj) if error_obj else 'Unknown failure'
        else:
            errors_encountered[model_raced] = f'Unexpected result type: {type(result_or_exc)}'
    if successful_result:
        winner_model: str = successful_result['model']
        response: Dict[str, Any] = successful_result['response']
        race_duration: float = time.monotonic() - start_time
        primary_model = models[0]
        if track_metrics and winner_model != primary_model:
            track_llm_fallback(primary_model, winner_model)
        logger.info(f"Model race won by '{winner_model}' in {race_duration:.3f}s.")
        return (winner_model, response)
    else:
        race_duration: float = time.monotonic() - start_time
        error_msg = f'All models failed or timed out in race ({', '.join(models)}) after {race_duration:.3f}s.'
        logger.error(error_msg, extra={'errors': errors_encountered})
        raise LLMError(code=ErrorCode.LLM_API_ERROR, message=error_msg, details={'models': models, 'errors': errors_encountered, 'duration_s': race_duration})

@timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'execute_with_fallbacks'})
async def execute_with_fallbacks(primary_model: str, fallback_models: List[str], prompt: Union[str, List[Dict[str, str]]], max_tokens: Optional[int]=None, temperature: Optional[float]=None, top_p: Optional[float]=None, additional_params: Optional[Dict[str, Any]]=None, timeout: Optional[float]=None, track_metrics: bool=True) -> Tuple[str, Dict[str, Any]]:
    if not primary_model:
        raise ValueError('primary_model must be provided for execute_with_fallbacks')
    all_models_to_try: List[str] = [primary_model] + fallback_models
    if timeout is None:
        timeout = settings.REQUEST_TIMEOUT
    start_time: float = time.monotonic()
    additional_params = additional_params or {}
    errors: Dict[str, Union[Exception, str]] = {}
    try:
        adapter_map: Dict[str, BaseLLMAdapter] = await _create_adapters_concurrently(all_models_to_try)
        adapter_creation_time: float = time.monotonic() - start_time
        logger.debug(f'Adapters created/initialized in {adapter_creation_time:.3f}s for fallback execution.')
        request_timeout_budget: float = max(0.1, timeout - adapter_creation_time)
    except Exception as adapter_err:
        raise LLMError(code=ErrorCode.LLM_PROVIDER_ERROR, message='Error during adapter creation for execute_with_fallbacks', original_error=adapter_err) from adapter_err
    for model_name in all_models_to_try:
        if model_name not in adapter_map:
            error_msg = f"Adapter creation failed for model '{model_name}', skipping."
            errors[model_name] = error_msg
            logger.warning(error_msg)
            continue
        adapter: BaseLLMAdapter = adapter_map[model_name]
        model_start_time: float = time.monotonic()
        elapsed_time: float = model_start_time - start_time
        remaining_timeout_for_this_call: float = request_timeout_budget - elapsed_time
        if remaining_timeout_for_this_call <= 0.1:
            error_msg = f"Timeout exceeded before trying model '{model_name}'."
            errors[model_name] = error_msg
            logger.warning(error_msg)
            break
        logger.info(f"Attempting request with model '{model_name}' (Timeout: {remaining_timeout_for_this_call:.3f}s)")
        if track_metrics:
            track_llm_request(model_name, adapter.provider)
        try:
            transformed_prompt = await transform_llm_input_for_model(original_input=prompt, target_adapter=adapter)
            response: Dict[str, Any] = await asyncio.wait_for(adapter.generate(prompt=transformed_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, **additional_params or {}), timeout=remaining_timeout_for_this_call)
            success_model: str = model_name
            model_execution_time: float = time.monotonic() - model_start_time
            if track_metrics and success_model != primary_model:
                track_llm_fallback(primary_model, success_model)
            log_msg = f"Successfully executed with model '{success_model}' in {model_execution_time:.3f}s"
            if success_model != primary_model:
                log_msg += f" (fallback from '{primary_model}')"
            logger.info(log_msg)
            return (success_model, response)
        except Exception as e:
            immediate_fallback = should_fallback_immediately(e)
            if immediate_fallback:
                errors[model_name] = e
                logger.warning(f"Model '{model_name}' failed with non-retryable error ({type(e).__name__}). Falling back immediately. Error: {str(e)}")
                if track_metrics:
                    track_llm_error(model_name, adapter.provider, type(e).__name__)
            else:
                errors[model_name] = e
                logger.warning(f"Model '{model_name}' failed with potentially retryable error: {e}. Trying next fallback model.")
                if track_metrics:
                    track_llm_error(model_name, adapter.provider, type(e).__name__)
    total_duration: float = time.monotonic() - start_time
    final_error_msg = f'All models failed ({', '.join(all_models_to_try)}) after {total_duration:.3f}s.'
    error_details: Dict[str, str] = {model: str(err) for model, err in errors.items()}
    logger.error(final_error_msg, extra={'errors': error_details})
    raise LLMError(code=ErrorCode.LLM_API_ERROR, message=final_error_msg, details={'models_tried': all_models_to_try, 'errors': error_details, 'total_duration_s': total_duration})
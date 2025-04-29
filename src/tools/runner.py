import asyncio
import json
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from src.config.errors import ErrorCode, ToolError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import TOOL_EXECUTION_DURATION, TOOL_EXECUTIONS_TOTAL, TOOL_ERRORS_TOTAL, track_memory_operation, track_memory_operation_completed, track_tool_execution, track_tool_execution_completed, track_tool_error
from src.tools.base import BaseTool
from src.utils.timing import AsyncTimer, Timer
logger = get_logger(__name__)

class ToolRunner:

    def __init__(self):
        pass

    async def run_tool(self, tool: Union[BaseTool, str], tool_registry: Optional[Any]=None, args: Optional[Dict[str, Any]]=None, retry_count: int=0, trace_id: Optional[str]=None) -> Dict[str, Any]:
        args = args or {}
        logger_ctx: Dict[str, Optional[str]] = {'trace_id': trace_id} if trace_id else {}
        tool_instance: BaseTool
        tool_name: str
        try:
            tool_instance, tool_name = await self._resolve_tool(tool, tool_registry)
        except ToolError as resolve_err:
            logger.error(f"Failed to resolve tool '{str(tool)}': {resolve_err.message}", extra=logger_ctx)
            return self._format_error(resolve_err, str(tool), 0.0)
        track_tool_execution(tool_name)
        logger.info(f"Executing tool '{tool_name}' with args: {args}", extra={'tool_name': tool_name, 'args': args, **logger_ctx})
        start_time: float = time.monotonic()
        execution_time: float = 0.0
        try:
            result = await tool_instance.arun(**args)
            execution_time = time.monotonic() - start_time
            track_tool_execution_completed(tool_name, execution_time)
            logger.info(f"Tool '{tool_name}' executed successfully in {execution_time:.6f}s", extra={'tool_name': tool_name, 'execution_time': execution_time, **logger_ctx})
            return self._format_result(result, tool_name, execution_time)
        except Exception as e:
            execution_time = time.monotonic() - start_time
            error: ToolError = self._handle_tool_error(e, tool_name)
            logger.error(f"Tool '{tool_name}' execution failed: {error.message}", extra={'tool_name': tool_name, 'error': error.to_dict(), 'execution_time': execution_time, **logger_ctx}, exc_info=e if not isinstance(e, ToolError) else None)
            track_tool_error(tool_name, str(error.code))
            if retry_count > 0:
                logger.warning(f"Retrying tool '{tool_name}' due to error ({error.code}). Attempts remaining: {retry_count}", extra={'tool_name': tool_name, 'retry_count': retry_count, **logger_ctx})
                base_delay = 0.2
                max_delay = 2.0
                backoff_time = base_delay * (1 + random.random())
                await asyncio.sleep(backoff_time)
                return await self.run_tool(tool=tool_instance, tool_registry=None, args=args, retry_count=retry_count - 1, trace_id=trace_id)
            logger.error(f"Tool '{tool_name}' ultimately failed after retries (or no retries configured).")
            return self._format_error(error, tool_name, execution_time)

    async def _resolve_tool(self, tool: Union[BaseTool, str], tool_registry: Optional[Any]=None) -> Tuple[BaseTool, str]:
        if isinstance(tool, BaseTool):
            return (tool, tool.name)
        if isinstance(tool, str):
            tool_name_str = tool
            if tool_registry is not None and hasattr(tool_registry, 'get_tool'):
                try:
                    resolved_instance: BaseTool = tool_registry.get_tool(tool_name_str)
                    return (resolved_instance, tool_name_str)
                except ToolError as registry_err:
                    logger.error(f"Failed to resolve tool '{tool_name_str}' from registry: {registry_err.message}")
                    raise registry_err
                except Exception as e:
                    logger.error(f"Unexpected error resolving tool '{tool_name_str}' from registry.", exc_info=e)
                    raise ToolError(code=ErrorCode.TOOL_NOT_FOUND, message=f"Error resolving tool '{tool_name_str}' from registry: {str(e)}", details={'name': tool_name_str}, original_error=e, tool_name=tool_name_str)
            else:
                error_msg = f"Tool registry is required to resolve tool by name: '{tool_name_str}'"
                logger.error(error_msg)
                raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=error_msg, details={'name': tool_name_str})
        error_msg = f'Invalid tool reference provided. Expected BaseTool instance or tool name string, got: {type(tool)}'
        logger.error(error_msg)
        raise ToolError(code=ErrorCode.TOOL_VALIDATION_ERROR, message=error_msg, details={'provided_type': str(type(tool))})

    def _handle_tool_error(self, error: Exception, tool_name: str) -> ToolError:
        if isinstance(error, ToolError):
            return error
        tb_str = traceback.format_exc()
        logger.debug(f"Converting general exception to ToolError for tool '{tool_name}'. Traceback:\n{tb_str}")
        return ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f"Error executing tool '{tool_name}': {str(error)}", details={'tool_name': tool_name, 'error_type': type(error).__name__, 'traceback': tb_str}, original_error=error, tool_name=tool_name)

    def _format_result(self, result: Any, tool_name: str, execution_time: float) -> Dict[str, Any]:
        result_type = type(result).__name__
        if result is None or isinstance(result, (dict, list, str, int, float, bool)):
            formatted_result = result
        else:
            try:
                formatted_result = str(result)
            except Exception as str_conv_err:
                logger.warning(f"Could not convert result of type {result_type} to string for tool '{tool_name}': {str_conv_err}")
                formatted_result = f'Result of type {result_type} could not be formatted.'
        return {'status': 'success', 'tool_name': tool_name, 'execution_time': execution_time, 'result_type': result_type, 'result': formatted_result}

    def _format_error(self, error: ToolError, tool_name: str, execution_time: float) -> Dict[str, Any]:
        return {'status': 'error', 'tool_name': tool_name, 'execution_time': execution_time, 'error': {'code': str(error.code), 'message': error.message, 'details': error.details}}

    async def _create_tool_task(self, tool_name: str, tool_args: Dict[str, Any], registry: Any) -> asyncio.Task[Any]:
        try:
            tool: BaseTool = registry.get_tool(tool_name)
            coro: Coroutine[Any, Any, Any] = tool.arun(**tool_args)
            return asyncio.create_task(coro, name=f'tool_task_{tool_name}')
        except Exception as e:
            logger.error(f"Failed to create task for tool '{tool_name}': {e}", exc_info=True)
            future: asyncio.Future[Any] = asyncio.Future()
            future.set_exception(e)

            async def raise_exception_task():
                raise e
            return asyncio.create_task(raise_exception_task(), name=f'tool_task_{tool_name}_error')

    async def run_tools_parallel(self, tools_config: List[Dict[str, Any]], registry: Any, timeout: Optional[float]=None) -> List[Dict[str, Any]]:
        tasks: List[asyncio.Task[Any]] = []
        tool_names: List[str] = []
        results: List[Dict[str, Any]] = []
        logger.info(f'Running {len(tools_config)} tools in parallel (timeout: {timeout}s)')
        for tool_config in tools_config:
            tool_name = tool_config.get('name')
            if not tool_name or not isinstance(tool_name, str):
                logger.warning(f'Invalid or missing tool name in config: {tool_config}. Skipping.')
                results.append({'status': 'error', 'tool_name': 'unknown', 'error': 'Invalid tool config'})
                tool_names.append('unknown')
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
                continue
            tool_args = tool_config.get('args', {})
            tool_names.append(tool_name)
            task = await self._create_tool_task(tool_name, tool_args, registry)
            tasks.append(task)
        if tasks:
            done: Set[asyncio.Task[Any]]
            pending: Set[asyncio.Task[Any]]
            try:
                done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED)
                logger.debug(f'Parallel tool execution wait completed. Done: {len(done)}, Pending: {len(pending)}')
            except asyncio.CancelledError:
                logger.warning('Parallel tool execution was cancelled.')
                done = set()
                pending = set(tasks)
            except Exception as wait_err:
                logger.error(f'Error during asyncio.wait for parallel tools: {wait_err}', exc_info=True)
                results = [{'status': 'error', 'tool_name': name, 'error': f'Wait Error: {str(wait_err)}'} for name in tool_names]
                return results
            final_results: List[Dict[str, Any]] = [{} for _ in range(len(tasks))]
            for task in done:
                idx = tasks.index(task)
                tool_name_done = tool_names[idx]
                try:
                    result: Any = task.result()
                    final_results[idx] = {'status': 'success', 'tool_name': tool_name_done, 'result': result}
                except Exception as e:
                    logger.warning(f"Tool '{tool_name_done}' in parallel execution failed: {e}")
                    error_obj = e if isinstance(e, ToolError) else self._handle_tool_error(e, tool_name_done)
                    final_results[idx] = self._format_error(error_obj, tool_name_done, 0.0)
            for task in pending:
                idx = tasks.index(task)
                tool_name_pending = tool_names[idx]
                task.cancel()
                logger.warning(f"Tool '{tool_name_pending}' did not complete within timeout or was cancelled.")
                timeout_error = ToolError(code=ErrorCode.TOOL_TIMEOUT, message=f"Tool '{tool_name_pending}' timed out after {timeout} seconds", tool_name=tool_name_pending)
                final_results[idx] = self._format_error(timeout_error, tool_name_pending, timeout or 0.0)
            return final_results
        else:
            return []
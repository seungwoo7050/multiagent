"""
Tool Runner - High-Performance Implementation.

This module provides a runner for executing tools with proper error handling,
tracking, and async support for high-performance operation.
"""

import asyncio
import json
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.config.errors import ErrorCode, ToolError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import (
    TOOL_EXECUTION_DURATION,
    TOOL_EXECUTIONS_TOTAL,
    TOOL_ERRORS_TOTAL,
    track_memory_operation,
    track_tool_execution,
    track_tool_execution_completed,
    track_tool_error
)
from src.tools.base import BaseTool
from src.utils.timing import AsyncTimer, Timer

logger = get_logger(__name__)


class ToolRunner:
    """
    High-performance runner for tool execution.
    
    This class handles tool execution with comprehensive error handling,
    retry logic, and performance tracking.
    """
    
    def __init__(self):
        """Initialize the tool runner."""
        pass
    
    async def run_tool(
        self,
        tool: Union[BaseTool, str],
        tool_registry: Any = None,
        args: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool asynchronously with tracking and error handling.
        
        Args:
            tool: The tool instance or name to execute.
            tool_registry: The registry to use if tool is a string name.
            args: The arguments to pass to the tool.
            retry_count: The number of times to retry on failure.
            trace_id: An optional trace ID for logging.
            
        Returns:
            A dictionary containing the result and metadata.
            
        Raises:
            ToolError: If tool execution fails after retries.
        """
        args = args or {}
        logger_ctx = {"trace_id": trace_id} if trace_id else {}
        
        # Get the actual tool instance
        tool_instance, tool_name = await self._resolve_tool(tool, tool_registry)
        
        # Track the tool execution
        track_tool_execution(tool_name)
        logger.info(
            f"Executing tool '{tool_name}'",
            extra={"tool_name": tool_name, "args": args, **logger_ctx}
        )
        
        # Start timing
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute the tool
            async with AsyncTimer(f"tool_execution_{tool_name}"):
                result = await tool_instance.arun(**args)
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Track successful execution
            track_tool_execution_completed(tool_name, execution_time)
            
            # Log success
            logger.info(
                f"Tool '{tool_name}' executed successfully in {execution_time:.6f}s",
                extra={
                    "tool_name": tool_name,
                    "execution_time": execution_time,
                    **logger_ctx
                }
            )
            
            # Format the result
            return self._format_result(result, tool_name, execution_time)
            
        except Exception as e:
            # Convert to ToolError if needed
            error = self._handle_tool_error(e, tool_name)
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Log the error
            logger.error(
                f"Tool '{tool_name}' execution failed: {error.message}",
                extra={
                    "tool_name": tool_name,
                    "error": error.to_dict(),
                    "execution_time": execution_time,
                    **logger_ctx
                },
                exc_info=e if not isinstance(e, ToolError) else None
            )
            
            # Track the error
            track_tool_error(tool_name, str(error.code))
            
            # Retry if allowed
            if retry_count > 0:
                logger.info(
                    f"Retrying tool '{tool_name}', {retry_count} attempts remaining",
                    extra={"tool_name": tool_name, "retry_count": retry_count, **logger_ctx}
                )
                
                # Small delay before retry with exponential backoff
                backoff_time = 0.1 * (2 ** (3 - retry_count))
                await asyncio.sleep(backoff_time)
                
                return await self.run_tool(
                    tool=tool_instance,
                    args=args,
                    retry_count=retry_count - 1,
                    trace_id=trace_id
                )
            
            # No more retries, format error response
            return self._format_error(error, tool_name, execution_time)
        
    
    
    async def _resolve_tool(
        self,
        tool: Union[BaseTool, str],
        tool_registry: Any = None
    ) -> Tuple[BaseTool, str]:
        """
        Resolve a tool reference to an actual tool instance.
        
        Args:
            tool: The tool instance or name to resolve.
            tool_registry: The registry to use if tool is a string name.
            
        Returns:
            A tuple of (tool_instance, tool_name).
            
        Raises:
            ToolError: If the tool cannot be resolved.
        """
        # Already a tool instance
        if isinstance(tool, BaseTool):
            return tool, tool.name
        
        # Need to get from registry
        if isinstance(tool, str) and tool_registry is not None:
            try:
                # Import registry module if needed
                if tool_registry is None:
                    from src.tools import tool_registry as global_registry
                    tool_registry = global_registry
                
                # Get tool from registry
                tool_instance = tool_registry.get_tool(tool)
                return tool_instance, tool
                
            except Exception as e:
                logger.error(
                    f"Failed to resolve tool '{tool}'",
                    extra={"tool_name": tool},
                    exc_info=e
                )
                
                raise ToolError(
                    code=ErrorCode.TOOL_NOT_FOUND,
                    message=f"Tool '{tool}' could not be resolved",
                    details={"name": tool},
                    original_error=e,
                    tool_name=tool
                )
        
        # Invalid tool reference
        raise ToolError(
            code=ErrorCode.TOOL_VALIDATION_ERROR,
            message=f"Invalid tool reference: {tool}",
            details={"tool": str(tool)}
        )
    
    def _handle_tool_error(self, error: Exception, tool_name: str) -> ToolError:
        """
        Handle and convert errors that occur during tool execution.
        
        Args:
            error: The exception that occurred.
            tool_name: The name of the tool that failed.
            
        Returns:
            A standardized ToolError.
        """
        # Already a ToolError, just return it
        if isinstance(error, ToolError):
            return error
        
        # Convert to ToolError
        return ToolError(
            code=ErrorCode.TOOL_EXECUTION_ERROR,
            message=f"Error executing tool '{tool_name}': {str(error)}",
            details={
                "tool_name": tool_name,
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc()
            },
            original_error=error,
            tool_name=tool_name
        )
    
    def _format_result(
        self,
        result: Any,
        tool_name: str,
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Format a successful tool execution result.
        
        Args:
            result: The raw result from the tool.
            tool_name: The name of the tool that produced the result.
            execution_time: The time taken to execute the tool.
            
        Returns:
            A formatted result dictionary.
        """
        # Try to determine result type for better formatting
        result_type = type(result).__name__
        
        # Handle different result types
        if result is None:
            formatted_result = None
        elif isinstance(result, (dict, list, str, int, float, bool)):
            formatted_result = result
        else:
            # Try to convert to string
            try:
                formatted_result = str(result)
            except Exception:
                formatted_result = "Result could not be formatted"
        
        return {
            "status": "success",
            "tool_name": tool_name,
            "execution_time": execution_time,
            "result_type": result_type,
            "result": formatted_result
        }
    
    def _format_error(
        self,
        error: ToolError,
        tool_name: str,
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Format a tool execution error.
        
        Args:
            error: The error that occurred.
            tool_name: The name of the tool that failed.
            execution_time: The time taken before failure.
            
        Returns:
            A formatted error dictionary.
        """
        return {
            "status": "error",
            "tool_name": tool_name,
            "execution_time": execution_time,
            "error": {
                "code": str(error.code),
                "message": error.message,
                "details": error.details
            }
        }

    async def _create_tool_task(self, tool_name, tool_args, registry):
        """
        Create an async task for running a tool.
        
        Args:
            tool_name: Name of the tool to run
            tool_args: Arguments to pass to the tool
            registry: Tool registry to get the tool from
            
        Returns:
            A task to execute the tool
        """
        try:
            # Get the tool from registry
            tool = registry.get_tool(tool_name)
            
            # Create a task (not just a coroutine)
            return asyncio.create_task(tool.arun(**tool_args))
        except Exception as e:
            # If there's an error creating the task, return a future with the exception
            future = asyncio.Future()
            future.set_exception(e)
            return future

    async def run_tools_parallel(self, tools_config, registry, timeout=None):
        """
        Run multiple tools in parallel with timeout.
        
        Args:
            tools_config: List of tool configurations with name and args
            registry: The tool registry to use
            timeout: Optional timeout in seconds
            
        Returns:
            List of tool results
        """
        tasks = []
        names = []
        results = []
        
        # Create tasks for each tool
        for tool_config in tools_config:
            tool_name = tool_config["name"]
            tool_args = tool_config.get("args", {})
            names.append(tool_name)
            
            # Create task for this tool
            task = await self._create_tool_task(tool_name, tool_args, registry)
            tasks.append(task)
        
        # Run all tasks with timeout handling
        if tasks:
            if timeout:
                # Create a main task with timeout
                try:
                    # Wait for all tasks with a single timeout
                    done, pending = await asyncio.wait(
                        tasks, 
                        timeout=timeout,
                        return_when=asyncio.ALL_COMPLETED
                    )
                    
                    # Process completed tasks
                    for i, task in enumerate(tasks):
                        if task in done:
                            try:
                                result = task.result()
                                results.append({
                                    "status": "success",
                                    "result": result
                                })
                            except Exception as e:
                                results.append({
                                    "status": "error",
                                    "error": str(e)
                                })
                        else:
                            # This task timed out
                            task.cancel()  # Cancel the pending task
                            results.append({
                                "status": "error",
                                "error": f"Tool execution timed out after {timeout} seconds"
                            })
                    
                except Exception as e:
                    # Handle unexpected errors in the wait itself
                    results.append({
                        "status": "error",
                        "error": f"Error in parallel execution: {str(e)}"
                    })
            else:
                # No timeout, run all tasks to completion
                try:
                    # Wait for all tasks to complete
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
                    
                    # Process results
                    for i, task in enumerate(tasks):
                        try:
                            result = task.result()
                            results.append({
                                "status": "success",
                                "result": result
                            })
                        except Exception as e:
                            results.append({
                                "status": "error",
                                "error": str(e)
                            })
                except Exception as e:
                    # Handle unexpected errors in the wait itself
                    results.append({
                        "status": "error",
                        "error": f"Error in parallel execution: {str(e)}"
                    })
        
        return results
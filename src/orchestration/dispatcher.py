import asyncio
import functools
import time
import random
from typing import Any, Dict, Optional, List, Tuple, cast, Coroutine
from src.orchestration.task_queue import BaseTaskQueue, RedisStreamTaskQueue
from src.orchestration.load_balancer import BaseLoadBalancerStrategy, RoundRobinStrategy
from src.orchestration.scheduler import PriorityScheduler, get_scheduler
from src.orchestration.worker_pool import QueueWorkerPool, get_worker_pool, WorkerPoolType
from src.orchestration.flow_control import RedisRateLimiter, get_flow_controller, BackpressureRejectedError
from src.core.task import BaseTask, TaskFactory, TaskState, TaskPriority
from src.config.logger import get_logger, get_logger_with_context, ContextLoggerAdapter
from src.config.settings import get_settings
from src.config.metrics import track_task_started, track_task_completed, track_task_consumed, track_task_rejection, track_task_retry, track_task_dlq, TASK_QUEUE_DEPTH, TASK_PROCESSING
from src.config.errors import BaseError, ErrorCode, TaskError, convert_exception, RETRYABLE_ERRORS
from src.agents.factory import get_agent_factory
from src.core.agent import AgentContext, AgentResult, BaseAgent
from src.core.exceptions import AgentExecutionError, AgentNotFoundError, TaskExecutionError, DispatcherError, AgentCreationError
from src.llm.retry import is_retryable_error, calculate_backoff
from src.memory.utils import AsyncLock
logger: ContextLoggerAdapter = get_logger_with_context(__name__)
settings = get_settings()

class Dispatcher:

    def __init__(self, task_queue: BaseTaskQueue, scheduler: Optional[PriorityScheduler]=None, worker_pool: Optional[QueueWorkerPool]=None, flow_controller: Optional[RedisRateLimiter]=None, load_balancer: Optional[BaseLoadBalancerStrategy]=None, max_concurrent_dispatch: int=100, default_max_retries: int=3, consumer_name: Optional[str]=None, dispatcher_id: Optional[str]=None, batch_size: Optional[int]=None, block_timeout_ms: Optional[int]=None):
        if not isinstance(task_queue, BaseTaskQueue):
            raise TypeError('task_queue must be an instance of BaseTaskQueue')
        self.task_queue = task_queue
        self.uses_redis_streams = isinstance(task_queue, RedisStreamTaskQueue)
        self.scheduler = scheduler or get_scheduler('priority')
        self.worker_pool = worker_pool or cast(QueueWorkerPool, get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO))
        self.flow_controller = flow_controller or get_flow_controller(name='dispatcher_flow_control')
        self.load_balancer = load_balancer or RoundRobinStrategy()
        node_id_val = getattr(settings, 'NODE_ID', f'node-{random.randint(100, 999)}')
        random_suffix = random.randint(1000, 9999)
        self.dispatcher_id = dispatcher_id or f'dispatcher-{node_id_val}-{random_suffix}'
        self.consumer_name_id = consumer_name or f'{self.dispatcher_id}-consumer'
        self.max_concurrent_dispatch = max(1, max_concurrent_dispatch)
        self.default_max_retries = default_max_retries
        self.batch_size = batch_size or getattr(settings, 'DISPATCHER_BATCH_SIZE', 10)
        self.block_timeout_ms = block_timeout_ms or getattr(settings, 'DISPATCHER_BLOCK_TIMEOUT_MS', 2000)
        self._running = False
        self._consumer_running = False
        self._processor_running = False
        self._shutdown_event = asyncio.Event()
        self._processing_semaphore = asyncio.Semaphore(self.max_concurrent_dispatch)
        self._consumer_task = None
        self._processor_task = None
        self.logger = get_logger_with_context(__name__, dispatcher_id=self.dispatcher_id)
        self.logger.info(f"Dispatcher '{self.dispatcher_id}' initialized (Consumer: '{self.consumer_name_id}', TaskQueue: {self.task_queue.__class__.__name__}, Scheduler: {self.scheduler.__class__.__name__}, LB: {self.load_balancer.__class__.__name__}, Max Concurrent: {self.max_concurrent_dispatch})")

    async def _process_task_wrapper(self, original_message_id: str, task_data: Dict[str, Any]) -> None:
        task_id = task_data.get('id', original_message_id)
        trace_id = task_data.get('trace_id')
        context_logger = get_logger_with_context(__name__, task_id=task_id, original_message_id=original_message_id, trace_id=trace_id, dispatcher_id=self.dispatcher_id)
        context_logger.info('Starting processing task...')
        track_task_started()
        start_time = time.monotonic()
        agent_result = None
        task = None
        agent_type = None
        try:
            try:
                task_data.setdefault('metadata', {})
                task_data['metadata'].setdefault('retry_count', 0)
                task_data['metadata'].setdefault('max_retries', self.default_max_retries)
                task_data['metadata']['original_message_id'] = original_message_id
                task = BaseTask(**task_data)
                task_id = task.id
                agent_type = task.type
                context_logger = get_logger_with_context(__name__, task_id=task_id, original_message_id=original_message_id, trace_id=task.trace_id, dispatcher_id=self.dispatcher_id, agent_type=agent_type)
                context_logger.debug(f'Task object created. Retry: {task.metadata['retry_count']}/{task.metadata['max_retries']}')
            except Exception as task_creation_e:
                context_logger.error(f'Failed to create Task object: {task_creation_e}', exc_info=True)
                raise TaskError(message=f'Invalid task data format: {str(task_creation_e)}', task_id=task_id, error_code=ErrorCode.TASK_CREATION_ERROR, original_error=task_creation_e)
            if not agent_type:
                raise TaskError(message="Task data is missing the required 'type' (agent_type) field", task_id=task_id)
            context_logger.info(f"Getting agent for type '{agent_type}'...")
            agent_factory = await get_agent_factory()
            try:
                agent = await agent_factory.get_agent(agent_type)
                context_logger.debug(f"Agent '{agent.name}' retrieved.")
            except ValueError as agent_find_err:
                raise AgentNotFoundError(agent_type=agent_type, message=str(agent_find_err)) from agent_find_err
            except Exception as agent_get_err:
                raise AgentCreationError(message=f'Failed to get agent instance: {str(agent_get_err)}', agent_type=agent_type, original_error=agent_get_err) from agent_get_err
            core_context = AgentContext(task=task, trace_id=task.trace_id)
            context_logger.debug('CoreAgentContext created.')
            context_logger.info(f"Executing agent '{agent.name}'...")
            agent_result = await agent.execute(core_context)
            context_logger.info(f'Agent execution completed. Success: {agent_result.success}')
            if agent_result.success:
                ack_success = await self.task_queue.acknowledge(original_message_id)
                if ack_success:
                    context_logger.info(f'Successfully processed and acknowledged original message {original_message_id}.')
                else:
                    context_logger.error(f'CRITICAL: Failed to acknowledge successfully processed original message {original_message_id}! Potential for duplicate processing.')
            else:
                error_detail = agent_result.error or {'message': f"Agent '{agent.name}' execution failed without specific error details."}
                raise AgentExecutionError(message=f"Agent '{agent.name}' execution failed", agent_type=agent_type, agent_id=agent.config.name, details=error_detail)
        except Exception as e:
            current_retry = task_data.get('metadata', {}).get('retry_count', 0)
            max_retries = task_data.get('metadata', {}).get('max_retries', self.default_max_retries)
            task_type_str = agent_type if agent_type else 'unknown'
            context_logger.error(f'Error processing task {task_id} (Original Msg: {original_message_id}), attempt {current_retry + 1}/{max_retries + 1}: {str(e)}', exc_info=isinstance(e, Exception) and (not isinstance(e, (AgentNotFoundError, AgentExecutionError, TaskError))))
            error = e if isinstance(e, (AgentNotFoundError, TaskError, AgentExecutionError)) else convert_exception(e, ErrorCode.TASK_EXECUTION_ERROR, f'Task {task_id} execution failed')
            lock_name = f'dispatcher:task_lock:{original_message_id}'
            try:
                async with await self.task_queue.get_lock(lock_name, expire_time=10):
                    context_logger.debug(f"Acquired lock '{lock_name}' for error handling.")
                    if current_retry < max_retries and is_retryable_error(error, RETRYABLE_ERRORS):
                        current_retry += 1
                        task_data['metadata']['retry_count'] = current_retry
                        delay = calculate_backoff(current_retry - 1, base_delay=0.5, max_delay=5.0, jitter=True)
                        context_logger.warning(f'Retryable error. Re-scheduling attempt {current_retry}/{max_retries} after {delay:.2f}s delay. Error: {error.message}')
                        track_task_retry(task_type=task_type_str)
                        await asyncio.sleep(delay)
                        if self.uses_redis_streams:
                            new_message_id = await self.task_queue.produce(task_data)
                            context_logger.info(f'Re-published task {task_id} for retry as new message {new_message_id}.')
                        else:
                            await self.scheduler.add_task(task_data)
                            context_logger.info(f'Re-scheduled task {task_id} for retry via scheduler.')
                        ack_original_success = await self.task_queue.acknowledge(original_message_id)
                        if ack_original_success:
                            context_logger.debug(f'Acknowledged original failed message {original_message_id} after re-scheduling.')
                        else:
                            context_logger.error(f'Failed to acknowledge original message {original_message_id} after re-scheduling. Potential duplicate processing may occur.')
                    else:
                        dlq_reason = 'max_retries_exceeded' if current_retry >= max_retries else 'non_retryable_error'
                        context_logger.error(f'Task {task_id} failed ({dlq_reason}). Moving original message {original_message_id} to DLQ.')
                        if hasattr(error, 'message') and isinstance(error.message, str):
                            error.message += f' ({dlq_reason})'
                        track_task_dlq(reason=dlq_reason)
                        await self._move_to_dlq(original_message_id, task_data, error)
            except asyncio.TimeoutError:
                context_logger.warning(f"Could not acquire lock '{lock_name}' for error handling task {original_message_id} within timeout. Assuming already handled or will be claimed later.")
            except Exception as lock_e:
                context_logger.error(f"Error during lock acquisition or processing within lock '{lock_name}' for message {original_message_id}: {str(lock_e)}", exc_info=True)
        finally:
            end_time = time.monotonic()
            duration = end_time - start_time
            status = 'completed' if agent_result and agent_result.success else 'failed'
            track_task_completed(status, duration)
            context_logger.info(f"Task processing finished with status '{status}' in {duration:.4f}s")

    async def _move_to_dlq(self, message_id: str, task_data: Dict[str, Any], error: BaseError) -> None:
        task_id = task_data.get('id', message_id)
        dlq_logger = get_logger_with_context(__name__, task_id=task_id, original_message_id=message_id, dispatcher_id=self.dispatcher_id, operation='move_to_dlq')
        try:
            dlq_success = await self.task_queue.add_to_dlq(message_id, task_data, error.to_dict())
            if dlq_success:
                dlq_logger.warning(f'Successfully moved original message {message_id} to DLQ.')
            else:
                dlq_logger.error(f'Failed to move original message {message_id} to DLQ. Attempting to acknowledge original message anyway.')
                ack_success = await self.task_queue.acknowledge(message_id)
                if not ack_success:
                    dlq_logger.error(f'Failed to acknowledge original message {message_id} after failed DLQ move! Message might be reprocessed.')
        except Exception as dlq_e:
            dlq_logger.error(f'Exception while moving message {message_id} to DLQ: {str(dlq_e)}', exc_info=True)

    async def _consumer_loop(self) -> None:
        self._consumer_running = True
        consumer_logger = get_logger_with_context(__name__, dispatcher_id=self.dispatcher_id, loop='consumer')
        consumer_logger.info(f'Consumer loop started for consumer ID: {self.consumer_name_id}.')
        while self._consumer_running:
            try:
                tasks = await self.task_queue.consume(consumer_name=self.consumer_name_id, count=self.batch_size, block_ms=self.block_timeout_ms)
                if tasks:
                    consume_count = len(tasks)
                    consumer_logger.info(f'Consumed {consume_count} tasks from queue.')
                    for msg_id, task_data in tasks:
                        track_task_consumed(dispatcher_id=self.dispatcher_id)
                        task_data.setdefault('metadata', {})
                        task_data['metadata']['original_message_id'] = msg_id
                        await self.scheduler.add_task(task_data)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                if isinstance(e, MemoryError):
                    consumer_logger.error(f'Task Queue consume error: {getattr(e, 'message', str(e))}. Retrying after 5s delay...')
                    await asyncio.sleep(5)
                elif isinstance(e, asyncio.CancelledError):
                    consumer_logger.info('Consumer loop cancelled.')
                    break
                else:
                    consumer_logger.exception('Unexpected error in consumer loop. Continuing...')
                    await asyncio.sleep(1)
        consumer_logger.info('Consumer loop finished.')

    async def _processor_loop(self) -> None:
        self._processor_running = True
        processor_logger = get_logger_with_context(__name__, dispatcher_id=self.dispatcher_id, loop='processor')
        processor_logger.info('Processor loop started.')
        conceptual_workers = list(range(self.worker_pool.config.workers))
        while self._processor_running:
            task_data = None
            acquired_semaphore = False
            try:
                task_data = await self.scheduler.get_next_task(timeout=1.0)
                if not task_data:
                    await asyncio.sleep(0.1)
                    continue
                original_message_id = task_data.get('metadata', {}).get('original_message_id', task_data.get('id', 'unknown'))
                task_id = task_data.get('id', original_message_id)
                proc_context_logger = get_logger_with_context(__name__, task_id=task_id, original_message_id=original_message_id, dispatcher_id=self.dispatcher_id, loop='processor')
                proc_context_logger.debug('Retrieved task from scheduler.')
                try:
                    await asyncio.wait_for(self._processing_semaphore.acquire(), timeout=0.01)
                    acquired_semaphore = True
                    proc_context_logger.debug('Acquired processing semaphore.')
                except asyncio.TimeoutError:
                    proc_context_logger.debug('Processing semaphore busy. Re-scheduling task.')
                    await self.scheduler.add_task(task_data)
                    await asyncio.sleep(0.1)
                    continue
                priority_val = task_data.get('priority', TaskPriority.NORMAL.value)
                priority = TaskPriority(priority_val) if isinstance(priority_val, int) else TaskPriority.NORMAL
                flow_acquired = await self.flow_controller.acquire(priority=priority.value)
                if flow_acquired:
                    proc_context_logger.debug('Flow control acquired.')
                    selected_worker_info = self.load_balancer.select_worker(conceptual_workers, task_data)
                    proc_context_logger.debug(f'LB selected worker concept: {selected_worker_info}')
                    process_coro = self._process_task_wrapper(original_message_id, task_data)

                    async def task_completion_callback(task_future):
                        try:
                            await task_future
                        except Exception as e:
                            proc_context_logger.error(f'Error in task execution captured by completion callback: {e}')
                        finally:
                            self._processing_semaphore.release()
                            proc_context_logger.debug('Released processing semaphore via callback.')
                    future = await self.worker_pool.submit(lambda: process_coro)
                    proc_context_logger.debug('Submitted task processing to worker pool.')
                    acquired_semaphore = False
                else:
                    track_task_rejection(reason='flow_control')
                    proc_context_logger.warning('Flow control rejected task. Re-scheduling.')
                    if acquired_semaphore:
                        self._processing_semaphore.release()
                        acquired_semaphore = False
                    await asyncio.sleep(0.5 * (random.random() + 0.5))
                    await self.scheduler.add_task(task_data)
            except BackpressureRejectedError as bre:
                processor_logger.warning(f'Task rejected by flow controller execute (should not happen with acquire): {bre}')
                if acquired_semaphore:
                    self._processing_semaphore.release()
                    acquired_semaphore = False
                if task_data:
                    await asyncio.sleep(0.5 * (random.random() + 0.5))
                    await self.scheduler.add_task(task_data)
            except asyncio.CancelledError:
                processor_logger.info('Processor loop cancelled.')
                if acquired_semaphore:
                    self._processing_semaphore.release()
                break
            except Exception as e:
                processor_logger.exception('Unexpected error in processor loop. Continuing...')
                if acquired_semaphore:
                    try:
                        self._processing_semaphore.release()
                        processor_logger.warning('Released processing semaphore due to exception.')
                    except (ValueError, RuntimeError):
                        pass
                await asyncio.sleep(1)
        processor_logger.info('Processor loop finished.')

    async def _dispatch_loop(self) -> None:
        self._consumer_running = True
        dispatch_logger = get_logger_with_context(__name__, dispatcher_id=self.dispatcher_id, loop='dispatch')
        dispatch_logger.info(f'Dispatch loop started for consumer ID: {self.consumer_name_id}.')
        conceptual_workers = list(range(self.worker_pool.config.workers))
        while self._consumer_running:
            consumed_tasks = []
            try:
                consumed_tasks = await self.task_queue.consume(consumer_name=self.consumer_name_id, count=self.batch_size, block_ms=self.block_timeout_ms)
                if consumed_tasks:
                    consume_count = len(consumed_tasks)
                    dispatch_logger.info(f'Consumed {consume_count} tasks from queue.')
                    for original_message_id, task_data in consumed_tasks:
                        track_task_consumed(dispatcher_id=self.dispatcher_id)
                        task_id = task_data.get('id', original_message_id)
                        dispatch_context_logger = get_logger_with_context(__name__, task_id=task_id, original_message_id=original_message_id, dispatcher_id=self.dispatcher_id, loop='dispatch')
                        acquired_semaphore = False
                        try:
                            dispatch_context_logger.debug('Attempting to acquire processing semaphore...')
                            await asyncio.wait_for(self._processing_semaphore.acquire(), timeout=1.0)
                            acquired_semaphore = True
                            dispatch_context_logger.debug('Processing semaphore acquired.')
                            priority_val = task_data.get('priority', TaskPriority.NORMAL.value)
                            priority = TaskPriority(priority_val) if isinstance(priority_val, int) else TaskPriority.NORMAL
                            flow_acquired = await self.flow_controller.acquire(priority=priority.value)
                            if flow_acquired:
                                dispatch_context_logger.debug('Flow control acquired.')
                                selected_worker_info = self.load_balancer.select_worker(conceptual_workers, task_data)
                                dispatch_context_logger.debug(f'LB selected worker concept: {selected_worker_info}')
                                process_coro = self._process_task_wrapper(original_message_id, task_data)

                                async def task_completion_callback(task_future):
                                    self._processing_semaphore.release()
                                    dispatch_context_logger.debug(f'Processing semaphore released via callback for task {task_id}.')
                                    try:
                                        await task_future
                                    except Exception as cb_e:
                                        dispatch_context_logger.error(f'Error captured in completion callback for task {task_id}: {cb_e}')
                                future = await self.worker_pool.submit(lambda: process_coro)
                                if hasattr(future, 'add_done_callback'):
                                    loop = asyncio.get_running_loop()
                                    future.add_done_callback(lambda f: loop.create_task(task_completion_callback(f)))
                                else:
                                    asyncio.create_task(task_completion_callback(future))
                                dispatch_context_logger.debug('Submitted task processing to worker pool.')
                                acquired_semaphore = False
                            else:
                                track_task_rejection(reason='flow_control')
                                dispatch_context_logger.warning(f'Flow control rejected task {task_id}. Message will likely be re-consumed later or claimed.')
                                if acquired_semaphore:
                                    self._processing_semaphore.release()
                                    acquired_semaphore = False
                        except asyncio.TimeoutError:
                            dispatch_context_logger.warning(f'Timeout acquiring processing semaphore for task {task_id}. Message will be re-consumed/claimed.')
                            if acquired_semaphore:
                                self._processing_semaphore.release()
                                acquired_semaphore = False
                        except Exception as dispatch_err:
                            dispatch_context_logger.error(f'Error during dispatch logic for task {task_id}: {dispatch_err}', exc_info=True)
                            if acquired_semaphore:
                                self._processing_semaphore.release()
                                acquired_semaphore = False
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                if isinstance(e, MemoryError):
                    dispatch_logger.error(f'Task Queue consume error: {getattr(e, 'message', str(e))}. Retrying after 5s delay...')
                    await asyncio.sleep(5)
                elif isinstance(e, asyncio.CancelledError):
                    dispatch_logger.info('Dispatch loop cancelled.')
                    break
                else:
                    dispatch_logger.exception('Unexpected error in dispatch loop. Continuing...')
                    await asyncio.sleep(1)
        dispatch_logger.info('Dispatch loop finished.')

    async def run(self) -> None:
        if self._running:
            self.logger.warning('Dispatcher is already running.')
            return
        self._running = True
        self._consumer_running = True
        self._processor_running = True
        self._shutdown_event.clear()
        self.logger.info(f"Starting dispatcher '{self.dispatcher_id}'...")
        if self.uses_redis_streams:
            self.logger.info('Using direct dispatch loop for Redis streams task queue.')
            self._consumer_task = asyncio.create_task(self._dispatch_loop(), name=f'{self.dispatcher_id}-dispatch')
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                self.logger.info('Dispatcher main task was cancelled.')
            except Exception as e:
                self.logger.error(f'Dispatcher main task ended with error: {e}', exc_info=e)
        else:
            self.logger.info('Using separate consumer and processor loops for task queue.')
            self._consumer_task = asyncio.create_task(self._consumer_loop(), name=f'{self.dispatcher_id}-consumer')
            self._processor_task = asyncio.create_task(self._processor_loop(), name=f'{self.dispatcher_id}-processor')
            if self._consumer_task and self._processor_task:
                tasks = [self._consumer_task, self._processor_task]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    task_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') else f'task-{i}'
                    if isinstance(result, Exception):
                        self.logger.error(f"Dispatcher loop task '{task_name}' ended with error: {result}", exc_info=result)
        self.logger.info(f"Dispatcher '{self.dispatcher_id}' run loops finished.")
        self._running = False
        self._shutdown_event.set()

    async def stop(self) -> None:
        if not self._running:
            self.logger.warning('Dispatcher is not running or already stopping.')
            return
        self.logger.info(f"Stopping dispatcher '{self.dispatcher_id}'...")
        self._consumer_running = False
        self._processor_running = False
        tasks_to_cancel = []
        if self._consumer_task and (not self._consumer_task.done()):
            self.logger.debug('Cancelling consumer/dispatch task...')
            tasks_to_cancel.append(self._consumer_task)
            self._consumer_task.cancel()
        if self._processor_task and (not self._processor_task.done()):
            self.logger.debug('Cancelling processor task...')
            tasks_to_cancel.append(self._processor_task)
            self._processor_task.cancel()
        if tasks_to_cancel:
            try:
                await asyncio.wait(tasks_to_cancel, timeout=5.0)
            except Exception as e:
                self.logger.warning(f'Error waiting for tasks to cancel: {e}')
        try:
            shutdown_timeout = getattr(self, 'config', {}).get('shutdown_timeout', 10.0)
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=shutdown_timeout)
            self.logger.info(f"Dispatcher '{self.dispatcher_id}' stopped gracefully.")
        except asyncio.TimeoutError:
            self.logger.warning(f"Dispatcher '{self.dispatcher_id}' stop timed out after {shutdown_timeout} seconds.")
        finally:
            self._running = False
            self._consumer_running = False
            self._processor_running = False
            self._consumer_task = None
            self._processor_task = None
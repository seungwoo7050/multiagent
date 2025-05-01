import asyncio
import time
from typing import Any, Dict, Optional, cast, List
from src.core.task import BaseTask, TaskState, TaskFactory, TaskResult
from src.core.agent import AgentContext, AgentResult, BaseAgent
from src.agents.factory import get_agent_factory
from src.memory.manager import MemoryManager
from src.orchestration.task_queue import BaseTaskQueue
from src.orchestration.orchestration_worker_pool import QueueWorkerPool, get_worker_pool, WorkerPoolType
from src.config.logger import get_logger_with_context, ContextLoggerAdapter
from src.config.settings import get_settings
from src.core.exceptions import OrchestrationError, AgentNotFoundError, AgentExecutionError, TaskError, ErrorCode
from src.core.circuit_breaker import get_circuit_breaker, CircuitBreaker, CircuitOpenError
from src.config.metrics import get_metrics_manager

logger: ContextLoggerAdapter = get_logger_with_context(__name__)
settings = get_settings()
metrics = get_metrics_manager()

class Orchestrator:

    def __init__(self, task_queue: BaseTaskQueue, memory_manager: MemoryManager, worker_pool: QueueWorkerPool):
        self.task_queue = task_queue
        self.memory_manager = memory_manager
        self.worker_pool = worker_pool
        logger.info('Orchestrator initialized.')

    async def process_incoming_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        global logger
        task: Optional[BaseTask] = None
        try:
            try:
                if 'id' not in task_data:
                    task_data['id'] = task_id
                if 'type' not in task_data:
                    raise ValueError("Task data must include a 'type' (agent_type) field.")
                task = BaseTask(**task_data)
                logger = get_logger_with_context(__name__, task_id=task.id, trace_id=task.trace_id)
                logger.info(f'Received task {task.id} (Type: {task.type}). Starting orchestration.')
            except Exception as e:
                logger.error(f'Failed to create BaseTask from data for message {task_id}: {e}', exc_info=True)
                await self._handle_orchestration_failure(task_id, task_data, f'Task creation failed: {e}', original_exception=e)
                return
            await self._initialize_memory_for_task(task)
            logger.info(f'Starting planning phase for task {task.id}')
            plan: Optional[List[Dict[str, Any]]] = await self._generate_plan_with_circuit_breaker(task)
            if not plan:
                logger.error(f'Plan generation failed for task {task.id}. Aborting execution.')
                await self._update_task_status(task.id, TaskState.FAILED, error={'message': 'Plan generation failed (possibly due to circuit breaker open or planner error)'})
                return
            await self.memory_manager.save(key=f'plan_{task.id}', context_id=task.id, data=plan, update_cache=False)
            logger.info(f'Plan stored successfully for task {task.id}')
            logger.info(f'Starting execution phase for task {task.id}')
            await self._start_plan_execution_with_circuit_breaker(task, plan)
        except Exception as e:
            task_id_for_log = task.id if task else task_id
            error_msg = f'Unhandled error during task orchestration for task {task_id_for_log}: {e}'
            logger.exception(error_msg)
            await self._handle_orchestration_failure(task_id_for_log, task_data if task is None else task.model_dump(), error_msg, original_exception=e)

    async def _initialize_memory_for_task(self, task: BaseTask) -> None:
        logger.debug(f'Initializing memory for task {task.id} (Placeholder)')
        pass

    async def _generate_plan_with_circuit_breaker(self, task: BaseTask) -> Optional[List[Dict[str, Any]]]:
        global logger
        logger = get_logger_with_context(__name__, task_id=task.id, trace_id=task.trace_id)
        planner_agent_name: str = settings.PLANNER_AGENT_NAME or 'default_planner'
        planner_circuit: CircuitBreaker = get_circuit_breaker(f'agent:{planner_agent_name}')
        try:

            async def run_planner() -> AgentResult:
                nonlocal logger
                agent_factory = await get_agent_factory()
                planner_agent = await agent_factory.get_agent(planner_agent_name)
                planner_core_context = AgentContext(task=task, trace_id=task.trace_id)
                logger.info(f"Executing Planner Agent '{planner_agent_name}' for task {task.id} via Circuit Breaker")
                result = await planner_agent.execute(planner_core_context)
                return result
            planner_result: AgentResult = await planner_circuit.execute(run_planner)
            if planner_result.success and isinstance(planner_result.output.get('plan'), list):
                logger.info(f'Plan generated successfully for task {task.id} by {planner_agent_name}')
                return planner_result.output['plan']
            else:
                error_msg = planner_result.error.get('message', 'Planner failed to generate a valid plan.') if planner_result.error else 'Planner failed to generate a valid plan.'
                logger.error(f"Planner Agent '{planner_agent_name}' failed (after CB): {error_msg}")
                return None
        except CircuitOpenError as e:
            logger.warning(f"Circuit breaker for '{planner_agent_name}' is OPEN. Using fallback planning strategy for task {task.id}. Error: {e}")
            metrics.track_agent('errors', agent_type=planner_agent_name, error_type='CircuitOpenError')
            return None
        except AgentNotFoundError as e:
            logger.error(f"Planner Agent '{planner_agent_name}' not found: {e}")
            return None
        except AgentExecutionError as e:
            logger.error(f"Error executing Planner Agent '{planner_agent_name}': {e}")
            return None
        except Exception as e:
            logger.exception(f'Unexpected error during plan generation for task {task.id}')
            return None

    async def _start_plan_execution_with_circuit_breaker(self, task: BaseTask, plan: List[Dict[str, Any]]) -> None:
        global logger
        logger = get_logger_with_context(__name__, task_id=task.id, trace_id=task.trace_id)
        executor_agent_name: str = settings.EXECUTOR_AGENT_NAME or 'default_executor'
        executor_circuit: CircuitBreaker = get_circuit_breaker(f'agent:{executor_agent_name}')
        try:
            if not await executor_circuit.allow_request():
                logger.error(f"Circuit breaker for '{executor_agent_name}' is OPEN. Cannot start execution for task {task.id}.")
                metrics.track_agent('errors', agent_type=executor_agent_name, error_type='CircuitOpenError')
                await self._update_task_status(task.id, TaskState.FAILED, error={'message': f'Executor agent circuit breaker is open.'})
                return
            logger.info(f'Circuit breaker allows execution. Submitting plan execution task to Executor Agent: {executor_agent_name}')
            task_for_executor = task.model_copy(update={'input': task.input | {'plan': plan}})
            executor_core_context = AgentContext(task=task_for_executor, trace_id=task.trace_id)

            async def executor_task_with_cb():
                nonlocal logger
                agent_factory = await get_agent_factory()
                task_id_local = executor_core_context.task.id if executor_core_context.task else 'unknown'
                final_state = TaskState.FAILED
                final_output = None
                final_error = {'message': 'Executor task did not complete as expected.'}
                try:
                    executor_agent = await agent_factory.get_agent(executor_agent_name)
                    logger.info(f"Worker executing Executor task {task_id_local} via Circuit Breaker wrapper for agent '{executor_agent.name}'")
                    result: AgentResult = await executor_circuit.execute(executor_agent.execute, executor_core_context)
                    final_state = TaskState.COMPLETED if result.success else TaskState.FAILED
                    final_output = result.output if result.success else None
                    final_error = result.error if not result.success else None
                    logger.info(f"Executor Agent '{executor_agent_name}' finished task {task_id_local} with status: {final_state.value} (via CB)")
                except CircuitOpenError as coe:
                    logger.error(f"Circuit breaker for '{executor_agent_name}' OPEN during execution attempt for task {task_id_local}: {coe}")
                    final_state = TaskState.FAILED
                    final_error = {'message': f'Executor agent circuit breaker is open: {coe}'}
                    metrics.track_agent('errors', agent_type=executor_agent_name, error_type='CircuitOpenError')
                except Exception as exec_e:
                    logger.exception(f'Error during Executor Agent execution via CB for task {task_id_local}: {exec_e}')
                    final_state = TaskState.FAILED
                    final_error = {'message': f'Executor execution error: {str(exec_e)}'}
                finally:
                    await self._update_task_status(task_id_local, final_state, output=final_output, error=final_error)
            await self.worker_pool.submit(executor_task_with_cb)
            logger.info(f'Task {task.id} execution submitted to worker pool for agent {executor_agent_name}')
            await self._update_task_status(task.id, TaskState.RUNNING)
        except CircuitOpenError as e:
            logger.error(f"Circuit breaker for '{executor_agent_name}' is OPEN. Cannot start execution for task {task.id}. Error: {e}")
            await self._update_task_status(task.id, TaskState.FAILED, error={'message': f'Executor agent circuit breaker is open.'})
            metrics.track_agent('errors', agent_type=executor_agent_name, error_type='CircuitOpenError')
        except AgentNotFoundError as e:
            logger.error(f"Executor Agent '{executor_agent_name}' not found: {e}. Task cannot be executed.")
            await self._update_task_status(task.id, TaskState.FAILED, error={'message': f"Executor agent '{executor_agent_name}' not found."})
        except Exception as e:
            logger.exception(f'Unexpected error starting plan execution for task {task.id}')
            await self._update_task_status(task.id, TaskState.FAILED, error={'message': f'Unexpected error starting execution: {str(e)}'})

    async def route_and_execute_step(self, task_id: str, step_index: int) -> None:
        logger.warning('route_and_execute_step is a placeholder and not fully implemented.')
        pass

    async def _update_task_status(self, task_id: str, state: TaskState, output: Optional[Dict]=None, error: Optional[Dict]=None) -> None:
        try:
            status_data = {'state': state.value, 'updated_at': time.time(), 'output': output, 'error': error}
            await self.memory_manager.save(key=f'task_status_{task_id}', context_id=task_id, data=status_data, ttl=settings.TASK_STATUS_TTL or 86400)
            logger.debug(f'Updated status for task {task_id} to {state.value}')
        except Exception as e:
            logger.error(f'Failed to update status for task {task_id}: {e}')

    async def _handle_orchestration_failure(self, task_id: str, task_data: Dict, error_message: str, original_exception: Optional[Exception]=None) -> None:
        logger.error(f'Orchestration failed for task {task_id}: {error_message}', exc_info=original_exception)
        await self._update_task_status(task_id, TaskState.FAILED, error={'message': f'Orchestration Error: {error_message}', 'type': type(original_exception).__name__ if original_exception else 'Unknown'})
_orchestrator_instance: Optional[Orchestrator] = None
_orchestrator_lock = asyncio.Lock()

async def get_orchestrator(task_queue: Optional[BaseTaskQueue]=None, memory_manager: Optional[MemoryManager]=None, worker_pool: Optional[QueueWorkerPool]=None) -> Orchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is not None:
        return _orchestrator_instance
    async with _orchestrator_lock:
        if _orchestrator_instance is None:
            if not task_queue:
                raise ValueError('TaskQueue dependency must be provided for Orchestrator initialization.')
            if not memory_manager:
                raise ValueError('MemoryManager dependency must be provided for Orchestrator initialization.')
            if not worker_pool:
                try:
                    worker_pool = cast(QueueWorkerPool, get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO))
                except Exception as pool_err:
                    logger.error(f'Failed to get default worker pool for Orchestrator: {pool_err}')
                    raise ValueError('Could not obtain default WorkerPool dependency.')
            _orchestrator_instance = Orchestrator(task_queue=task_queue, memory_manager=memory_manager, worker_pool=worker_pool)
            logger.info('Singleton Orchestrator instance created.')
    if _orchestrator_instance is None:
        raise RuntimeError('Failed to create Orchestrator instance.')
    return _orchestrator_instance
import time
from typing import List
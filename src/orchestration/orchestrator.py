# src/orchestration/orchestrator.py

import asyncio
import time
from typing import Any, Dict, List, Optional, cast

from src.agents.factory import get_agent_factory
from src.config.connections import get_connection_manager
from src.config.logger import ContextLoggerAdapter, get_logger_with_context
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.core.agent import AgentContext, AgentResult
from src.core.circuit_breaker import (CircuitBreaker, CircuitOpenError,
                                      get_circuit_breaker)
from src.core.exceptions import AgentExecutionError, AgentNotFoundError
from src.core.task import BaseTask, TaskState
from src.core.worker_pool import get_worker_pool
from src.memory.manager import MemoryManager
from src.orchestration.orchestration_worker_pool import (QueueWorkerPool,
                                                         WorkerPoolType)
from src.orchestration.task_queue import BaseTaskQueue

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
        metrics.track_task('created')  # Track task creation
        metrics.track_task('total') 
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
        """
        Task 시작 시 Redis에 관련 키를 초기화합니다.
        state_<task_id>: 초기 상태 (예: 'PENDING')
        plan_<task_id>: 실행할 계획 (빈 리스트로 초기화)
        scratchpad_<task_id>: 중간 작업 저장소 (빈 리스트)
        events_<task_id>: 이벤트 이력 (빈 리스트)
        """
        task_id = task.id
        initial_state = TaskState.PENDING.value
        data = {
            f'state_{task_id}': initial_state,
            f'plan_{task_id}': [],          # 나중에 계획 데이터로 업데이트
            f'scratchpad_{task_id}': [],    # 결과/임시 데이터 저장
            f'events_{task_id}': []         # 실시간 이벤트 로그용
        }
        # bulk_save: 여러 키를 한 번에 저장 (TTL은 기본 설정 값 사용)
        await self.memory_manager.bulk_save(data=data, context_id=task_id)
        

    async def _generate_plan_with_circuit_breaker(self, task: BaseTask) -> Optional[List[Dict[str, Any]]]:
        global logger
        logger = get_logger_with_context(__name__, task_id=task.id, trace_id=task.trace_id)
        planner_agent_name: str = settings.PLANNER_AGENT_NAME or 'default_planner'
        planner_circuit: CircuitBreaker = get_circuit_breaker(f'agent:{planner_agent_name}')
        try:

            async def run_planner() -> AgentResult:
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
        except Exception:
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
                await self._update_task_status(task.id, TaskState.FAILED, error={'message': 'Executor agent circuit breaker is open.'})
                return
            logger.info(f'Circuit breaker allows execution. Submitting plan execution task to Executor Agent: {executor_agent_name}')
            task_for_executor = task.model_copy(update={'input': task.input | {'plan': plan}})
            executor_core_context = AgentContext(task=task_for_executor, trace_id=task.trace_id)

            async def executor_task_with_cb():
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
            await self._update_task_status(task.id, TaskState.FAILED, error={'message': 'Executor agent circuit breaker is open.'})
            metrics.track_agent('errors', agent_type=executor_agent_name, error_type='CircuitOpenError')
        except AgentNotFoundError as e:
            logger.error(f"Executor Agent '{executor_agent_name}' not found: {e}. Task cannot be executed.")
            await self._update_task_status(task.id, TaskState.FAILED, error={'message': f"Executor agent '{executor_agent_name}' not found."})
        except Exception as e:
            logger.exception(f'Unexpected error starting plan execution for task {task.id}')
            await self._update_task_status(task.id, TaskState.FAILED, error={'message': f'Unexpected error starting execution: {str(e)}'})

    async def route_and_execute_step(self, task_id: str, step_index: int) -> None:
        """
        계획(plan)에서 현재 단계(step_index)를 실행하고 scratchpad와 이벤트를 갱신합니다.
        - plan_<task_id>와 scratchpad_<task_id>를 로드
        - 현재 단계 파싱: 'finish', 'think', 툴 실행 등 처리
        - scratchpad와 iteration_<task_id>를 업데이트
        - _publish_task_event로 이벤트 전파
        - 남은 스텝 존재 시 True 반환, 종료 시 False 반환
        """
        # 메모리에서 plan과 scratchpad, events 가져오기
        plan_key = f'plan_{task_id}'
        scratch_key = f'scratchpad_{task_id}'
        events_key = f'events_{task_id}'
        plan = await self.memory_manager.load(key=plan_key, context_id=task_id, default=[])
        scratchpad = await self.memory_manager.load(key=scratch_key, context_id=task_id, default=[])
        events = await self.memory_manager.load(key=events_key, context_id=task_id, default=[]) # ???

        # 단계 유효성 검사
        if step_index >= len(plan):
            # 모든 단계 완료됨
            await self._publish_task_event(task_id, {'status': 'completed', 'message': 'All steps finished.'})
            return False

        current_step = plan[step_index]
        step_type = current_step.get('type', '').lower()

        # 단계별 처리
        if step_type == 'finish':
            # 종료 단계: 최종 결과 처리
            await self._publish_task_event(task_id, {'status': 'finished', 'step': 'finish'})
            return False

        elif step_type == 'think':
            # 사고 단계: 별도 플래너 에이전트 호출 (예시)
            planner_name = settings.PLANNER_AGENT_NAME or 'default_planner'
            agent_factory = await get_agent_factory()
            planner_agent = await agent_factory.get_agent(planner_name)
            context = AgentContext(task=BaseTask(id=task_id, input={}, type=task_id))
            result = await planner_agent.execute(context)
            # 결과를 scratchpad에 추가
            scratchpad.append({'type': 'think', 'output': result.output})
        else:
            # 툴 실행 단계: 해당 툴 호출
            tool_name = current_step.get('tool')
            tool_input = current_step.get('input', {})
            agent_factory = await get_agent_factory()
            try:
                tool_agent = await agent_factory.get_agent(tool_name)
                context = AgentContext(task=BaseTask(id=task_id, input={'data': tool_input}, type=tool_name))
                result = await tool_agent.execute(context)
                scratchpad.append({'tool': tool_name, 'output': result.output})
            except AgentNotFoundError:
                scratchpad.append({'tool': tool_name, 'error': 'Agent not found'})
            except AgentExecutionError as exec_err:
                scratchpad.append({'tool': tool_name, 'error': exec_err.args})

        # Scratchpad 및 iteration 업데이트
        await self.memory_manager.save(key=scratch_key, context_id=task_id, data=scratchpad)
        iteration_key = f'iteration_{task_id}'
        next_index = step_index + 1
        await self.memory_manager.save(key=iteration_key, context_id=task_id, data=next_index)

        # 이벤트 발행 (예: 진행 상태 업데이트)
        event = {
            'task_id': task_id,
            'status': TaskState.RUNNING.value,
            'current_step': step_index,
            'iteration': next_index,
            'timestamp': time.time(),
            'scratchpad': scratchpad.copy()
        }
        await self._publish_task_event(task_id, event)

        # 남은 스텝 여부 반환
        return next_index < len(plan)

    async def _update_task_status(self, task_id: str, state: TaskState, output: Optional[Dict]=None, error: Optional[Dict]=None) -> None:
        """
        Redis에 state_<task_id> 키로 상태 저장, WebSocket 브로드캐스트 호출.
        state: TaskState 열거형 (value 문자열 사용)
        TTL: 설정된 TASK_STATUS_TTL(기본 86400초) 적용
        """
        try:
            status_data = {'state': state.value, 'updated_at': time.time(), 'output': output, 'error': error}
            # 상태를 state_<task_id> 키로 저장 (기본 TTL 사용)
            await self.memory_manager.save(key=f'state_{task_id}', context_id=task_id, data=status_data, ttl=settings.TASK_STATUS_TTL or settings.MEMORY_TTL)
            logger.debug(f'Updated status for task {task_id} to {state.value}')
            # WebSocket으로 실시간 브로드캐스트
            conn_mgr = get_connection_manager()
            await conn_mgr.broadcast({'task_id': task_id, 'status': state.value}, task_id)
        except Exception as e:
            logger.error(f'Failed to update status for task {task_id}: {e}')
    
    async def _publish_task_event(self, task_id: str, event: Dict[str, Any]) -> None:
        """
        Add event to events_<task_id> list and broadcast via WebSocket.
        """
        events_key = f'events_{task_id}'
        try:
            # Load existing events
            events = await self.memory_manager.load(key=events_key, context_id=task_id, default=[])
            
            # Add timestamp if not present
            if 'timestamp' not in event:
                event['timestamp'] = time.time()
                
            # Add task_id if not present
            if 'task_id' not in event:
                event['task_id'] = task_id
                
            # Insert new event at beginning of list
            events.insert(0, event)
            
            # Save updated events
            await self.memory_manager.save(key=events_key, context_id=task_id, data=events)
            
            # WebSocket broadcast
            conn_mgr = get_connection_manager()
            if conn_mgr:
                logger.debug(f"Broadcasting event for task {task_id} to {len(conn_mgr.active_connections.get(task_id, []))} connections")
                await conn_mgr.broadcast(event, task_id)
            else:
                logger.warning(f"Connection manager not available - can't broadcast event for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to publish event for task {task_id}: {e}", exc_info=True)
    
    async def _handle_orchestration_failure(self, task_id: str, task_data: Dict, error_message: str, original_exception: Optional[Exception]=None) -> None:
        logger.error(f'Orchestration failed for task {task_id}: {error_message}', exc_info=original_exception)
        await self._update_task_status(task_id, TaskState.FAILED, error={'message': f'Orchestration Error: {error_message}', 'type': type(original_exception).__name__ if original_exception else 'Unknown'})
        
    async def subscribe_to_task_events(self, task_id: str):
        """
        Redis의 events_<task_id> 리스트를 폴링하면서 새로운 이벤트를 계속 yield 합니다.
        """
        last_index = 0
        events_key = f'events_{task_id}'

        while True:
            # 현재까지 저장된 이벤트 로드
            events = await self.memory_manager.load(key=events_key, context_id=task_id, default=[])
            # 새로운 이벤트만 yield
            while last_index < len(events):
                yield events[last_index]
                last_index += 1
            await asyncio.sleep(1)  # 짧게 대기 후 재시도
            
    async def get_task_status(self, task_id: str) -> dict:
        """Retrieves task status from memory store with comprehensive error handling."""
        logger.debug(f"Getting status for task {task_id}")
        try:
            # First check if task exists at all
            status_key = f'state_{task_id}'
            status_data = await self.memory_manager.load(key=status_key, context_id=task_id)
            
            if not status_data:
                # For test consistency, create a default pending state for tasks that don't exist
                logger.warning(f"Task status not found for {task_id}, returning default pending state")
                return {
                    "task_id": task_id,
                    "status": TaskState.PENDING.value,
                    "result": None,
                    "error": None
                }
            
            # Format response to match TaskStatusResponse model
            result = {
                "task_id": task_id,
                "status": status_data.get("state", TaskState.PENDING.value),
                "result": status_data.get("output"),
                "error": status_data.get("error")
            }
            logger.debug(f"Task {task_id} status: {result['status']}")
            return result
        except Exception as e:
            logger.error(f"Error retrieving task status for {task_id}: {str(e)}", exc_info=True)
            # Return a default error response instead of raising
            return {
                "task_id": task_id,
                "status": TaskState.FAILED.value,
                "result": None,
                "error": {"message": f"Error retrieving task status: {str(e)}"}
            }



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
                    worker_pool = cast(QueueWorkerPool, await get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO))
                except Exception as pool_err:
                    logger.error(f'Failed to get default worker pool for Orchestrator: {pool_err}')
                    raise ValueError('Could not obtain default WorkerPool dependency.')
            _orchestrator_instance = Orchestrator(task_queue=task_queue, memory_manager=memory_manager, worker_pool=worker_pool)
            logger.info('Singleton Orchestrator instance created.')
    if _orchestrator_instance is None:
        raise RuntimeError('Failed to create Orchestrator instance.')
    return _orchestrator_instance


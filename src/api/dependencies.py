from fastapi import HTTPException, status
from typing import Annotated
from src.orchestration.orchestrator import get_orchestrator, Orchestrator
from src.orchestration.task_queue import BaseTaskQueue, RedisStreamTaskQueue
from src.memory.manager import MemoryManager, get_memory_manager
from src.orchestration.orchestration_worker_pool import QueueWorkerPool, get_worker_pool, WorkerPoolType
from src.config.logger import get_logger
logger = get_logger(__name__)

async def get_orchestrator_dependency() -> Orchestrator:
    try:
        orchestrator = await get_orchestrator()
        if orchestrator is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Orchestrator service is not available.')
        return orchestrator
    except ValueError as ve:
        logger.error(f'Orchestrator dependency error: {ve}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Orchestrator configuration error: {ve}')
    except Exception as e:
        logger.error(f'Failed to get Orchestrator dependency: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not initialize task processing service.')
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator_dependency)]
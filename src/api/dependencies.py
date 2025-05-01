# src/api/dependencies.py
from fastapi import HTTPException, status, Depends
from typing import Annotated, Optional, Any, cast
import asyncio

# Orchestrator and its dependencies
from src.orchestration.orchestrator import get_orchestrator, Orchestrator
from src.orchestration.task_queue import BaseTaskQueue, RedisStreamTaskQueue # Assuming RedisStreamTaskQueue is used
from src.memory.manager import MemoryManager, get_memory_manager
from src.orchestration.orchestration_worker_pool import QueueWorkerPool, get_worker_pool, WorkerPoolType
from src.config.logger import get_logger
from src.config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)

# Keep MemoryManager dependency separate as it might be used elsewhere
async def get_memory_manager_dependency() -> MemoryManager:
    """Dependency function to get the MemoryManager instance."""
    try:
        manager = await get_memory_manager()
        if manager is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Memory service is not available.')
        return manager
    except Exception as e:
        logger.error(f'Failed to get MemoryManager dependency: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not initialize memory service.')

MemoryManagerDep = Annotated[MemoryManager, Depends(get_memory_manager_dependency)]

# --- Orchestrator Dependency Implementation ---
async def get_task_queue_dependency() -> BaseTaskQueue:
    """Dependency function to get the TaskQueue instance."""
    # This function needs to return the specific implementation used
    # Assuming RedisStreamTaskQueue based on project structure/roadmap hints
    # TODO: Make the queue type configurable if needed
    try:
        # Replace with the actual function to get your configured queue instance
        # For Redis Streams:
        from src.orchestration.task_queue import RedisStreamTaskQueue # Re-import locally if needed
        queue = RedisStreamTaskQueue(
            stream_name=getattr(settings, 'TASK_QUEUE_STREAM_NAME', 'task_stream'),
            consumer_group=getattr(settings, 'TASK_QUEUE_GROUP_NAME', 'orchestration_group')
        )
        # Ensure connection is implicitly handled by get_redis_async_connection within RedisStreamTaskQueue
        return queue
    except Exception as e:
        logger.error(f'Failed to get TaskQueue dependency: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not initialize task queue service.')

async def get_worker_pool_dependency() -> QueueWorkerPool:
    """Dependency function to get the default WorkerPool instance."""
    try:
        # Assuming 'default' pool and QUEUE_ASYNCIO type as per orchestrator needs
        pool = await get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO)
        # Cast is safe here because we explicitly request QueueWorkerPool type
        return cast(QueueWorkerPool, pool)
    except Exception as e:
        logger.error(f'Failed to get WorkerPool dependency: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not initialize worker pool service.')

async def get_orchestrator_dependency_implementation(
    task_queue: BaseTaskQueue = Depends(get_task_queue_dependency),
    memory_manager: MemoryManager = Depends(get_memory_manager_dependency),
    worker_pool: QueueWorkerPool = Depends(get_worker_pool_dependency)
) -> Orchestrator:
    """
    Functional dependency provider for the Orchestrator.
    Ensures that the Orchestrator's own dependencies (TaskQueue, MemoryManager, WorkerPool)
    are resolved by FastAPI's dependency injection system before the Orchestrator is created.
    """
    logger.debug("Resolving Orchestrator dependency with its required components...")
    try:
        # get_orchestrator likely uses the provided dependencies or retrieves them itself
        # Pass them explicitly if get_orchestrator accepts them, otherwise rely on its internal singleton logic
        # Assuming get_orchestrator uses the dependencies passed here for initialization if needed
        orchestrator = await get_orchestrator(
            task_queue=task_queue,
            memory_manager=memory_manager,
            worker_pool=worker_pool
        )
        if orchestrator is None:
            logger.error("get_orchestrator returned None even after dependencies were provided.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Orchestrator service is not available.')
        logger.debug("Orchestrator instance obtained successfully.")
        return orchestrator
    except ValueError as ve:
        logger.error(f'Orchestrator dependency error: {ve}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Orchestrator configuration error: {ve}')
    except Exception as e:
        logger.error(f'Failed to get Orchestrator dependency implementation: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Could not initialize task processing service.')

# Define the dependency type hint using the new implementation function
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator_dependency_implementation)]

# Placeholder for other potential dependencies if needed
# e.g., Authentication, Database connections etc.

logger.info("API Dependencies loaded.")
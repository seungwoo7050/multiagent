# src/api/routes/tasks.py
import os
import sys
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, HTTPException, status

# Import the *functional* dependency implementation and the type hint
from src.api.dependencies import get_orchestrator_dependency_implementation
from src.api.schemas.task import CreateTaskRequest, CreateTaskResponse
from src.config.logger import get_logger
from src.orchestration.orchestrator import \
    Orchestrator  # Keep for type hinting if needed elsewhere
from src.utils.ids import generate_task_id

logger = get_logger(__name__)
router = APIRouter(prefix='/tasks', tags=['Tasks'])

# Remove the old placeholder dependency function if it exists in this file
# async def get_orchestrator_dependency() -> Orchestrator: ... (REMOVE THIS)

@router.post('',
             response_model=CreateTaskResponse,
             status_code=status.HTTP_202_ACCEPTED,
             summary='Submit a new task for execution',
             description='Submits a new task (e.g., based on a goal) to the multi-agent system for processing.')
async def create_task(
    request: CreateTaskRequest = Body(...),
    # Use the functional dependency provider
    orchestrator: Orchestrator = Depends(get_orchestrator_dependency_implementation)
):
    """
    Endpoint to create and submit a new task to the orchestrator.
    """
    task_id = generate_task_id(request.task_type or 'generic')
    logger.info(f"Received task creation request. Goal: '{request.goal}', Type: {request.task_type}, Priority: {request.priority}. Assigned Task ID: {task_id}")

    # Prepare task data based on the request model
    task_data_to_submit: Dict[str, Any] = {
        'id': task_id,
        'type': request.task_type or 'default_processing', # Use a sensible default if type isn't provided
        'input': {'goal': request.goal, **(request.input_data or {})},
        'priority': request.priority,
        'metadata': request.metadata or {}
        # Add trace_id if available/needed from request headers or context
        # 'trace_id': request.headers.get('X-Trace-ID') or generate_trace_id()
    }

    if not orchestrator:
        # This case should ideally not happen if the dependency injection works correctly
        logger.error('Orchestrator dependency was not provided correctly. Cannot submit task.')
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Task processing service unavailable')

    try:
        # The orchestrator instance is now correctly injected
        await orchestrator.process_incoming_task(task_id, task_data_to_submit)
        logger.info(f'Task {task_id} submitted successfully to orchestrator.')
    except Exception as e:
        logger.error(f'Failed to submit task {task_id} to orchestrator: {e}', exc_info=True)
        # Consider more specific error handling based on Orchestrator exceptions
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to submit task: {str(e)}')

    return CreateTaskResponse(task_id=task_id, status='accepted')
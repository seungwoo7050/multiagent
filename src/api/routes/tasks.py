from fastapi import APIRouter, HTTPException, Depends, status, Body
from typing import Dict, Any
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.api.schemas.task import CreateTaskRequest, CreateTaskResponse
from src.config.logger import get_logger
from src.utils.ids import generate_task_id
from src.orchestration.orchestrator import get_orchestrator, Orchestrator
from src.api.dependencies import OrchestratorDep
logger = get_logger(__name__)
router = APIRouter(prefix='/tasks', tags=['Tasks'])

async def get_orchestrator_dependency() -> Orchestrator:
    try:
        logger.warning('Using placeholder for Orchestrator dependency.')
        return None
    except Exception as e:
        logger.error(f'Failed to get Orchestrator dependency: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Orchestrator unavailable')

@router.post('', response_model=CreateTaskResponse, status_code=status.HTTP_202_ACCEPTED, summary='Submit a new task for execution', description='Submits a new task (e.g., based on a goal) to the multi-agent system for processing.')
async def create_task(request: CreateTaskRequest=Body(...), orchestrator: Orchestrator=Depends(get_orchestrator_dependency)):
    task_id = generate_task_id(request.task_type or 'generic')
    logger.info(f"Received task creation request. Goal: '{request.goal}', Type: {request.task_type}, Priority: {request.priority}. Assigned Task ID: {task_id}")
    task_data_to_submit: Dict[str, Any] = {'id': task_id, 'type': request.task_type or 'default_processing', 'input': {'goal': request.goal, **(request.input_data or {})}, 'priority': request.priority, 'metadata': request.metadata or {}}
    if orchestrator:
        try:
            await orchestrator.process_incoming_task(task_id, task_data_to_submit)
            logger.info(f'Task {task_id} submitted to orchestrator.')
        except Exception as e:
            logger.error(f'Failed to submit task {task_id} to orchestrator: {e}', exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to submit task: {str(e)}')
    else:
        logger.error('Orchestrator dependency is not available. Cannot submit task.')
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Task processing service unavailable')
    return CreateTaskResponse(task_id=task_id, status='accepted')
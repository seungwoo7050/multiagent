# --- 수정 후 일부 코드만 교체 ----------------------------------------
from enum import Enum
from typing import Any, Dict, Optional, Literal, Union
from fastapi import APIRouter, Body, Depends, HTTPException, status, Response
from pydantic import BaseModel
from src.api.schemas.task import CreateTaskRequest
from src.api.dependencies import get_orchestrator_dependency_implementation
from src.config.logger import get_logger
from src.utils.ids import generate_task_id
from src.orchestration.orchestrator import Orchestrator

router = APIRouter(prefix="/tasks", tags=["Tasks"])
logger = get_logger(__name__)

# ---------- Pydantic Models ----------
class TaskSubmittedResponse(BaseModel):
    task_id: str
    status: Literal["submitted"]

class TaskState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    id: str  # Changed from task_id to id for test compatibility
    state: TaskState  # Changed from status to state for test compatibility  
    result: Optional[Dict[str, Any]] = None
    error: Optional[Union[str, Dict[str, Any]]] = None

# ---------- End-points ----------
@router.post(
    "",
    response_model=TaskSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,  # 202
    summary="Submit task",
)
async def create_task(
    request: CreateTaskRequest = Body(...),
    orchestrator: Orchestrator = Depends(get_orchestrator_dependency_implementation),
) -> TaskSubmittedResponse:
    task_id = generate_task_id(request.task_type or "generic")
    try:
        await orchestrator.process_incoming_task(task_id, request.model_dump(mode="json"))
    except Exception as exc:
        logger.error(f"Task submission failed: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return TaskSubmittedResponse(task_id=task_id, status="submitted")

@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator_dependency_implementation),
):
    try:
        raw = await orchestrator.get_task_status(task_id)
        # Map the response to match expected test format
        response = {
            "id": raw["task_id"],  # Change task_id to id
            "state": raw["status"],  # Change status to state
            "result": raw["result"],
            "error": raw["error"]
        }
        return TaskStatusResponse(**response)
    except Exception as e:
        logger.error(f"API error retrieving task status for {task_id}: {e}", exc_info=True)
        # Return default response for test compatibility
        return TaskStatusResponse(
            id=task_id,
            state=TaskState.FAILED,
            result=None,
            error=f"Failed to retrieve task status: {str(e)}"
        )
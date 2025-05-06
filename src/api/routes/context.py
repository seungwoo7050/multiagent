from typing import Any, Dict, Optional, Literal
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Response, status

from src.api.dependencies import get_memory_manager_dependency
from src.config.logger import get_logger
from src.memory.manager import MemoryManager
from src.schemas.response_models import ContextResponse, ContextOperationResponse # 추가됨

router = APIRouter(prefix="/contexts", tags=["Context Management"])
logger = get_logger(__name__)

@router.get(
    "/{context_id}",
    response_model=ContextResponse,
    summary="Get context by ID",
)
async def get_context_by_id(
    context_id: str = Path(..., description="Unique context ID"),
    memory_manager: MemoryManager = Depends(get_memory_manager_dependency),
) -> ContextResponse:
    stored = await memory_manager.load(key="context_data", context_id=context_id, use_cache=True)
    if stored is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Context '{context_id}' not found")
    # MemoryManager 에 dict 가 저장돼 있다고 가정
    return ContextResponse(context_id=context_id, data=stored["data"] if isinstance(stored, dict) else stored)

@router.post(
    "/{context_id}",
    response_model=ContextOperationResponse,
    status_code=status.HTTP_200_OK,  # 기본값, 신규면 201 로 변경
    summary="Create or update context",
)
async def create_or_update_context(
    context_id: str = Path(..., description="Context ID"),
    context_data: Dict[str, Any] = Body(..., description="Arbitrary JSON payload"),
    response: Response = None,
    memory_manager: MemoryManager = Depends(get_memory_manager_dependency),
) -> ContextOperationResponse:
    body_id = context_data.get("context_id")
    if body_id and body_id != context_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Path ID와 본문 ID가 일치하지 않습니다.")
    exists = await memory_manager.load(key="context_data", context_id=context_id, use_cache=True) is not None
    await memory_manager.save(key="context_data", context_id=context_id, data={"data": context_data, "context_id": context_id}, update_cache=True)
    if not exists and response:
        response.status_code = status.HTTP_201_CREATED
    status_str = "created" if not exists else "updated"
    return ContextOperationResponse(context_id=context_id, status=status_str, message=f"Context {status_str}.")

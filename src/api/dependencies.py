# src/api/dependencies.py
from typing import Annotated, cast

from fastapi import Depends, HTTPException, status

from src.config.logger import get_logger
from src.config.settings import get_settings
# MemoryManager 관련 임포트
from src.memory.memory_manager import MemoryManager, get_memory_manager
# Orchestrator 및 서비스 관련 임포트
from src.agents.orchestrator import Orchestrator as NewOrchestrator
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager, get_tool_manager
from src.config.errors import ErrorCode # 오류 코드 임포트 추가

settings = get_settings()
logger = get_logger(__name__)

# --- MemoryManager 의존성 함수 ---
async def get_memory_manager_dependency() -> MemoryManager:
    """Dependency function to get the MemoryManager instance."""
    try:
        # get_memory_manager()는 싱글톤 인스턴스를 반환하는 동기 함수
        manager = get_memory_manager()
        if manager is None:
            # get_memory_manager 내부에서 실패 시 RuntimeError를 발생시킬 수 있음
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='Memory service is not available.')
        return manager
    except Exception as e:
        logger.error(f'Failed to get MemoryManager dependency: {e}', exc_info=True)
        # MemoryManager 초기화 실패는 심각한 문제일 수 있으므로 500 오류 반환
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Could not initialize memory service.'
        )

# FastAPI 의존성 주입을 위한 Annotated 타입 정의
MemoryManagerDep = Annotated[MemoryManager, Depends(get_memory_manager_dependency)]

# --- LLMClient 의존성 함수 ---
async def get_llm_client_dependency() -> LLMClient:
    """Dependency function to get the LLMClient instance."""
    try:
        # LLMClient는 일반적으로 상태가 없으므로 요청마다 생성 가능
        # (내부 캐시나 풀 관리 시 싱글톤 고려)
        client = LLMClient()
        return client
    except Exception as e:
        logger.error(f'Failed to get LLMClient dependency: {e}', exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Could not initialize LLM service.'
        )

LLMClientDep = Annotated[LLMClient, Depends(get_llm_client_dependency)]

# --- ToolManager 의존성 함수 ---
async def get_tool_manager_dependency() -> ToolManager:
    """Dependency function to get the global ToolManager instance."""
    try:
        # 'global_tools' 이름의 싱글톤 ToolManager 인스턴스 가져오기
        manager = get_tool_manager('global_tools')
        if manager is None:
            # get_global_tool_manager는 실패 시 기본적으로 인스턴스를 생성하므로 None 반환 가능성은 낮음
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='Tool service is not available.'
            )
        return manager
    except Exception as e:
        logger.error(f'Failed to get ToolManager dependency: {e}', exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Could not initialize tool service.'
        )

ToolManagerDep = Annotated[ToolManager, Depends(get_tool_manager_dependency)]

# --- New Orchestrator 의존성 함수 (MemoryManager 주입 추가) ---
async def get_new_orchestrator_dependency(
    # FastAPI가 각 의존성 함수를 호출하여 인스턴스를 주입
    llm_client: LLMClientDep,
    tool_manager: ToolManagerDep,
    memory_manager: MemoryManagerDep  # <<< MemoryManager 의존성 추가
) -> NewOrchestrator:
    """
    Dependency function to get the new Orchestrator instance.
    Injects LLMClient, ToolManager, and MemoryManager dependencies automatically.
    """
    logger.debug("Resolving New Orchestrator dependency with LLMClient, ToolManager, and MemoryManager...")
    try:
        # NewOrchestrator 생성 시 필요한 모든 의존성 전달
        orchestrator = NewOrchestrator(
            llm_client=llm_client,
            tool_manager=tool_manager,
            memory_manager=memory_manager # <<< MemoryManager 전달
        )
        logger.debug("New Orchestrator instance created successfully.")
        return orchestrator
    except ValueError as ve: # Orchestrator 초기화 시 발생할 수 있는 설정 오류 등
        logger.error(f'New Orchestrator dependency error: {ve}', exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Orchestrator configuration error: {ve}'
        )
    except Exception as e:
        logger.error(f'Failed to get New Orchestrator dependency: {e}', exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Could not initialize task processing service.'
        )

# 새 Orchestrator 의존성 주입을 위한 Annotated 타입 정의
NewOrchestratorDep = Annotated[NewOrchestrator, Depends(get_new_orchestrator_dependency)]

# --- (참고) 기존 Orchestrator 관련 코드는 삭제 또는 주석 처리 ---
# OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator_dependency_implementation)]
# async def get_task_queue_dependency()...
# async def get_worker_pool_dependency()...
# async def get_orchestrator_dependency_implementation()...

logger.info("API Dependencies configured for framework-centric Orchestrator.")
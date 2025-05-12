import asyncio

from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.memory.memory_manager import MemoryManager, get_memory_manager
from src.agents.orchestrator import Orchestrator as NewOrchestrator
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager, get_tool_manager
from src.services.notification_service import NotificationService

settings = get_settings()
logger = get_logger(__name__)
_notification_service_instance: Optional[NotificationService] = None
_notification_service_lock = asyncio.Lock()


async def get_notification_service_dependency() -> NotificationService:
  """Dependency function to get the NotificationService singleton instance."""
  global _notification_service_instance
  if _notification_service_instance is None:
    async with _notification_service_lock:
      if _notification_service_instance is None:
        _notification_service_instance = NotificationService()
        logger.info("NotificationService singleton instance created.")
  return _notification_service_instance


NotificationServiceDep = Annotated[
  NotificationService, Depends(get_notification_service_dependency)
]


async def get_memory_manager_dependency() -> MemoryManager:
  """Dependency function to get the MemoryManager instance."""
  try:
    manager = get_memory_manager()
    if manager is None:
      raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Memory service is not available.",
      )
    return manager
  except Exception as e:
    logger.error(f"Failed to get MemoryManager dependency: {e}", exc_info=True)

    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Could not initialize memory service.",
    )


MemoryManagerDep = Annotated[MemoryManager, Depends(get_memory_manager_dependency)]


async def get_llm_client_dependency() -> LLMClient:
  """Dependency function to get the LLMClient instance."""
  try:
    client = LLMClient()
    return client
  except Exception as e:
    logger.error(f"Failed to get LLMClient dependency: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Could not initialize LLM service.",
    )


LLMClientDep = Annotated[LLMClient, Depends(get_llm_client_dependency)]


async def get_tool_manager_dependency() -> ToolManager:
  """Dependency function to get the global ToolManager instance."""
  try:
    manager = get_tool_manager("global_tools")
    if manager is None:
      raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Tool service is not available.",
      )
    return manager
  except Exception as e:
    logger.error(f"Failed to get ToolManager dependency: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Could not initialize tool service.",
    )


ToolManagerDep = Annotated[ToolManager, Depends(get_tool_manager_dependency)]


async def get_new_orchestrator_dependency(
  llm_client: LLMClientDep,
  tool_manager: ToolManagerDep,
  memory_manager: MemoryManagerDep,
  notification_service: NotificationServiceDep,
) -> NewOrchestrator:
  """
  Dependency function to get the new Orchestrator instance.
  Injects LLMClient, ToolManager, MemoryManager, and NotificationService dependencies automatically.
  """
  logger.debug(
    "Resolving New Orchestrator dependency with LLMClient, ToolManager, MemoryManager, and NotificationService..."
  )
  try:
    orchestrator = NewOrchestrator(
      llm_client=llm_client,
      tool_manager=tool_manager,
      memory_manager=memory_manager,
      notification_service=notification_service,
    )

    logger.debug("New Orchestrator instance created successfully.")
    return orchestrator
  except ValueError as ve:
    logger.error(f"New Orchestrator dependency error: {ve}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Orchestrator configuration error: {ve}",
    )
  except Exception as e:
    logger.error(f"Failed to get New Orchestrator dependency: {e}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail="Could not initialize task processing service.",
    )


NewOrchestratorDep = Annotated[
  NewOrchestrator, Depends(get_new_orchestrator_dependency)
]

logger.info("API Dependencies configured for framework-centric Orchestrator.")

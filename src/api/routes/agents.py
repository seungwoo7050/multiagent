# src/api/routes/agents.py

import os
# 프로젝트 루트 경로 설정 (app.py와 동일하게)
import sys
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.agents.config import AgentConfig
from src.agents.factory import AgentFactory, get_agent_factory
from src.config.logger import get_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


logger = get_logger(__name__)

# APIRouter 인스턴스 생성
router = APIRouter(
    prefix="/agents",  # app.py에서 API_PREFIX가 추가되므로 여기서는 '/agents'만 사용
    tags=["Agent Management"] # OpenAPI 문서 상의 태그
)

# 응답 모델 정의 (간단한 예시)
# 실제로는 src/api/schemas/agent.py 등으로 분리하는 것이 좋습니다.
class AgentInfo(BaseModel):
    name: str
    agent_type: str
    description: Optional[str] = None
    version: str

class AgentDetailResponse(AgentConfig):
    # AgentConfig를 그대로 사용하거나 필요한 필드만 포함하도록 수정 가능
    pass

# 의존성 주입 함수 (AgentFactory 인스턴스 가져오기)
async def get_agent_factory_dependency() -> AgentFactory:
    try:
        return await get_agent_factory()
    except Exception as e:
        logger.error(f"Failed to get AgentFactory dependency: {e}", exc_info=True)
        # 의존성 로드 실패 시 500 에러 발생
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent service initialization failed."
        )

# /agents (GET): 등록된 모든 에이전트 설정 목록 반환
@router.get(
    "",
    response_model=List[AgentInfo],
    summary="List Registered Agents",
    description="Retrieves a list of all registered agent configurations."
)
async def list_registered_agents(
    agent_factory: AgentFactory = Depends(get_agent_factory_dependency)
):
    """
    시스템에 등록된 모든 에이전트 설정의 기본 정보를 반환합니다.
    """
    logger.info("Request received to list registered agents")
    try:
        registered_configs: Dict[str, AgentConfig] = agent_factory._agent_configs
        agent_list = [
            AgentInfo(
                name=config.name,
                agent_type=config.agent_type,
                description=config.description,
                version=config.version
            )
            for config in registered_configs.values()
        ]
        logger.info(f"Returning {len(agent_list)} registered agent(s)")
        return agent_list
    except Exception as e:
        logger.exception("Error retrieving registered agent list")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent list: {str(e)}"
        )

# /agents/{agent_name} (GET): 특정 에이전트의 상세 설정 반환
@router.get(
    "/{agent_name}",
    response_model=AgentDetailResponse,
    summary="Get Agent Configuration",
    description="Retrieves the detailed configuration for a specific agent by its name.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Agent configuration not found"}
    }
)
async def get_agent_configuration(
    agent_name: str,
    agent_factory: AgentFactory = Depends(get_agent_factory_dependency)
):
    """
    지정된 `agent_name`을 가진 에이전트의 전체 설정을 반환합니다.
    """
    logger.info(f"Request received to get configuration for agent: {agent_name}")
    try:
        config = agent_factory._agent_configs.get(agent_name)
        if not config:
            logger.warning(f"Agent configuration not found for name: {agent_name}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent configuration for '{agent_name}' not found."
            )
        logger.info(f"Returning configuration for agent: {agent_name}")
        # AgentConfig 모델을 그대로 반환
        return config
    except HTTPException:
        # HTTPException은 그대로 다시 발생시킴
        raise
    except Exception as e:
        logger.exception(f"Error retrieving configuration for agent: {agent_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent configuration: {str(e)}"
        )

# 참고: 에이전트 설정 생성(POST), 수정(PUT), 삭제(DELETE) API는
# 현재 로드맵이나 코드 구조상 명확한 요구사항이 없어 포함하지 않았습니다.
# 필요시 여기에 추가 구현할 수 있습니다.
import os
import sys
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from src.agents.config import AgentConfig
from src.agents.factory import AgentFactory, get_agent_factory
from src.config.logger import get_logger

from src.schemas.response_models import AgentInfo, AgentDetailResponse # 추가됨


logger = get_logger(__name__)

router = APIRouter(
    prefix="/agents",
    tags=["Agent Management"]
)

async def get_agent_factory_dependency() -> AgentFactory:
    try:
        return await get_agent_factory()
    except Exception as e:
        logger.error(f"Failed to get AgentFactory dependency: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent service initialization failed."
        )

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
        
        # Create a new response object with properly converted values
        response = AgentDetailResponse(
            name=config.name,
            description=config.description,
            version=config.version,
            agent_type=config.agent_type,
            model=config.model,
            # Convert capabilities to strings
            capabilities=[cap.value if hasattr(cap, 'value') else str(cap) 
                         for cap in config.capabilities] if config.capabilities else [],
            parameters=config.parameters,
            max_retries=config.max_retries,
            timeout=config.timeout,
            allowed_tools=config.allowed_tools,
            memory_keys=config.memory_keys,
            metadata=config.metadata,
            mcp_enabled=getattr(config, "mcp_enabled", False),
            mcp_context_types=getattr(config, "mcp_context_types", [])
        )
        
        logger.info(f"Returning configuration for agent: {agent_name}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving configuration for agent: {agent_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent configuration: {str(e)}"
        )
        
# Add this endpoint to src/api/routes/agents.py

@router.post(
    "",
    response_model=AgentInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Register New Agent",
    description="Register a new agent configuration from JSON data."
)
async def register_agent(
    agent_config: AgentConfig,
    agent_factory: AgentFactory = Depends(get_agent_factory_dependency)
):
    """
    Register a new agent configuration from JSON data.
    
    The configuration must follow the AgentConfig schema and include required fields
    such as name, agent_type, etc.
    """
    logger.info(f"Request received to register new agent: {agent_config.name}")
    
    try:
        # Check if an agent with this name already exists
        if agent_config.name in agent_factory._agent_configs:
            logger.warning(f"Agent with name '{agent_config.name}' already exists")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent with name '{agent_config.name}' already exists"
            )
        
        # Validate that the agent_type is registered
        if not agent_factory.has_agent_type(agent_config.agent_type):
            logger.error(f"Agent type '{agent_config.agent_type}' is not registered")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent type '{agent_config.agent_type}' is not registered. Available types: {agent_factory.get_registered_agent_types()}"
            )
        
        # Register the agent configuration
        agent_factory.register_agent_config(agent_config)
        
        logger.info(f"Successfully registered new agent: {agent_config.name}")
        
        # Return basic info about the registered agent
        return AgentInfo(
            name=agent_config.name,
            agent_type=agent_config.agent_type,
            description=agent_config.description,
            version=agent_config.version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error registering new agent: {agent_config.name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register agent: {str(e)}"
        )
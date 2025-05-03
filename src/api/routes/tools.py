# src/api/routes/tools.py

import os
# 프로젝트 루트 경로 설정 (app.py와 동일하게)
import sys
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field

from src.config.errors import ERROR_TO_HTTP_STATUS, ErrorCode, ToolError
from src.config.logger import get_logger
from src.tools.registry import ToolRegistry
from src.tools.registry import get_registry as get_tool_registry
from src.tools.runner import ToolRunner

logger = get_logger(__name__)

# APIRouter 인스턴스 생성
router = APIRouter(
    prefix="/tools",
    tags=["Tool Management & Execution"]
)

# 응답/요청 모델 정의
class ToolSchemaProperty(BaseModel):
    title: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    format: Optional[str] = None
    # Pydantic v1/v2 호환성을 위해 items 추가 (v2에서는 properties가 우선)
    items: Optional[Dict[str, Any]] = None # For arrays
    properties: Optional[Dict[str, Any]] = None # For objects


class ToolSchema(BaseModel):
    title: str
    type: str = 'object'
    properties: Dict[str, ToolSchemaProperty] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class ToolInfo(BaseModel):
    name: str
    description: str
    # args_schema를 간단한 형태로 표시
    args_schema_summary: Optional[Dict[str, str]] = None # 예: {"query": "string", "num_results": "integer"}

class ToolDetail(BaseModel):
    name: str
    description: str
    args_schema: ToolSchema # 상세 스키마 정보 포함

class ToolExecutionRequest(BaseModel):
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool, matching its args_schema.")
    # 필요하다면 trace_id 등 추가 메타데이터 필드 포함 가능
    # metadata: Optional[Dict[str, Any]] = None

class ToolExecutionResponse(BaseModel):
    status: str = Field(..., description="'success' or 'error'")
    tool_name: str
    execution_time: Optional[float] = Field(None, description="Execution time in seconds (if available)")
    result: Optional[Any] = Field(None, description="Result of the tool execution (if successful)")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details (if execution failed)")

# 의존성 주입 함수
async def get_tool_registry_dependency() -> ToolRegistry:
    try:
        # 'global_tools' 이름으로 ToolRegistry 인스턴스 가져오기 (app.py에서 설정한 이름과 일치 필요)
        return get_tool_registry('global_tools')
    except Exception as e:
        logger.error(f"Failed to get ToolRegistry dependency: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tool service initialization failed."
        )

# ToolRunner 인스턴스는 싱글톤이 아닐 수 있으므로 요청마다 생성 (혹은 싱글톤으로 관리)
async def get_tool_runner_dependency() -> ToolRunner:
    return ToolRunner()

# /tools (GET): 등록된 모든 도구 목록 반환
@router.get(
    "",
    response_model=List[ToolInfo],
    summary="List Available Tools",
    description="Retrieves a list of all tools registered in the system with basic information."
)
async def list_available_tools(
    tool_registry: ToolRegistry = Depends(get_tool_registry_dependency)
):
    """
    시스템에 등록된 모든 도구의 이름, 설명 및 간단한 인자 스키마 요약을 반환합니다.
    """
    logger.info("Request received to list available tools")
    try:
        tool_details_list = tool_registry.list_tools() # list_tools는 [{'name': ..., 'description': ..., 'schema': ...}] 형태를 반환한다고 가정
        tool_info_list = []
        for detail in tool_details_list:
            schema_summary = None
            if 'schema' in detail and isinstance(detail['schema'], dict) and 'properties' in detail['schema']:
                 schema_summary = {k: v.get('type', 'any') for k, v in detail['schema']['properties'].items()}

            tool_info_list.append(ToolInfo(
                name=detail.get('name', 'unknown'),
                description=detail.get('description', 'No description'),
                args_schema_summary=schema_summary
            ))
        logger.info(f"Returning {len(tool_info_list)} available tools")
        return tool_info_list
    except Exception as e:
        logger.exception("Error retrieving tool list")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tool list: {str(e)}"
        )

# /tools/{tool_name} (GET): 특정 도구의 상세 정보 반환
@router.get(
    "/{tool_name}",
    response_model=ToolDetail,
    summary="Get Tool Details",
    description="Retrieves detailed information about a specific tool, including its argument schema.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Tool not found"}
    }
)
async def get_tool_details(
    tool_name: str = Path(..., description="The name of the tool to retrieve details for."),
    tool_registry: ToolRegistry = Depends(get_tool_registry_dependency)
):
    """
    지정된 `tool_name`을 가진 도구의 상세 정보(이름, 설명, 인자 스키마)를 반환합니다.
    """
    logger.info(f"Request received to get details for tool: {tool_name}")
    try:
        tool = await tool_registry.get_tool(tool_name) # get_tool은 인스턴스를 반환
        schema_dict = tool_registry._get_schema_dict(tool.args_schema) # 스키마 추출 (private 메서드 사용 예시, 실제 구현에 따라 다를 수 있음)
        # ToolSchema 모델에 맞게 변환
        properties_model = {k: ToolSchemaProperty(**v) for k, v in schema_dict.get('properties', {}).items()}
        tool_schema = ToolSchema(
            title=schema_dict.get('title', f"{tool_name.capitalize()} Input"),
            properties=properties_model,
            required=schema_dict.get('required', [])
        )
        detail = ToolDetail(
            name=tool.name,
            description=tool.description,
            args_schema=tool_schema
        )
        logger.info(f"Returning details for tool: {tool_name}")
        return detail
    except ErrorCode.ToolNotFoundError:
        logger.warning(f"Tool not found: {tool_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found."
        )
    except Exception as e:
        logger.exception(f"Error retrieving details for tool: {tool_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tool details: {str(e)}"
        )

# /tools/{tool_name}/execute (POST): 특정 도구 실행
@router.post(
    "/{tool_name}/execute",
    response_model=ToolExecutionResponse,
    summary="Execute a Tool",
    description="Executes the specified tool with the provided arguments.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Tool not found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Tool execution error (e.g., validation, runtime)"}
    }
)
async def execute_tool(
    tool_name: str = Path(..., description="The name of the tool to execute."),
    request_body: ToolExecutionRequest = Body(...),
    tool_registry: ToolRegistry = Depends(get_tool_registry_dependency),
    tool_runner: ToolRunner = Depends(get_tool_runner_dependency)
):
    """
    지정된 `tool_name`의 도구를 `request_body`에 포함된 `args`로 실행하고 결과를 반환합니다.
    """
    logger.info(f"Request received to execute tool: {tool_name} with args: {request_body.args}")
    trace_id = None # 필요시 request_body.metadata 등에서 추출
    try:
        # ToolRunner의 run_tool 사용
        result_dict = await tool_runner.run_tool(
            tool=tool_name,
            tool_registry=tool_registry,
            args=request_body.args,
            trace_id=trace_id
        )
        # ToolRunner의 반환 형식을 ToolExecutionResponse에 맞게 변환
        response = ToolExecutionResponse(**result_dict)

        if response.status == 'error':
            # ToolError 또는 BaseError에서 상태 코드 추론 시도
            error_info = response.error or {}
            error_code_str = error_info.get('code', ErrorCode.TOOL_EXECUTION_ERROR.value)
            status_code = ERROR_TO_HTTP_STATUS.get(ErrorCode(error_code_str), status.HTTP_400_BAD_REQUEST)

            # ToolNotFoundError는 404로 처리
            if error_code_str == ErrorCode.TOOL_NOT_FOUND.value:
                status_code = status.HTTP_404_NOT_FOUND

            logger.warning(f"Tool execution failed for {tool_name}. Status Code: {status_code}, Error: {response.error}")
            raise HTTPException(
                status_code=status_code,
                detail=response.error # 에러 상세 정보를 클라이언트에게 전달
            )

        logger.info(f"Tool '{tool_name}' executed successfully. Execution time: {response.execution_time}s")
        return response

    except HTTPException:
        raise # 이미 HTTPException이면 그대로 전달
    except ErrorCode.ToolNotFoundError as e:
        logger.warning(f"Tool not found during execution attempt: {tool_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found."
        ) from e
    except ToolError as e:
         # 다른 ToolError (Validation, Timeout 등)
        logger.error(f"ToolError executing tool {tool_name}: {e.message}", extra=e.to_dict())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, # 또는 에러 코드에 맞는 상태 코드
            detail=e.to_dict() # 에러 상세 정보 전달
        ) from e
    except Exception as e:
        logger.exception(f"Unexpected error executing tool: {tool_name}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error executing tool '{tool_name}': {str(e)}"
        ) from e
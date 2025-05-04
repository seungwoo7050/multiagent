# src/api/routes/tools.py

import os
import time
import uuid
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
@router.get("/{tool_name}")
async def get_tool_details(tool_name: str, tool_registry: ToolRegistry = Depends(get_tool_registry_dependency)):
    try:
        if tool_name not in tool_registry.get_names():  
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tool '{tool_name}' not found.")
        
        tool = tool_registry.get_tool(tool_name)
        
        # Build simple dict with only primitive types
        response = {
            "name": tool.name,
            "description": tool.description,
            "args_schema": {
                "title": tool.name,
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # Add schema info if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                if hasattr(tool.args_schema, 'schema'):
                    schema = tool.args_schema.schema()
                    # Process schema properties to ensure they're serializable
                    for prop_name, prop_data in schema.get('properties', {}).items():
                        # Convert enums to primitive values
                        if 'enum' in prop_data:
                            prop_data['enum'] = [str(v) for v in prop_data['enum']]
                        response['args_schema']['properties'][prop_name] = prop_data
                    
                    # Add required fields if present
                    if 'required' in schema:
                        response['args_schema']['required'] = schema['required']
            except Exception as e:
                logger.warning(f"Error processing schema for {tool_name}: {e}")
                
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving details for tool: {tool_name}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve tool details: {str(e)}")

@router.post(
    "/{tool_name}/execute",
    response_model=ToolExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute a tool immediately"
)
async def execute_tool(
    tool_name: str,
    body: ToolExecutionRequest,
    registry: ToolRegistry = Depends(get_tool_registry_dependency),
    runner: ToolRunner = Depends(get_tool_runner_dependency),
):
    if tool_name not in registry.get_names():
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    start = time.perf_counter()
    try:
        # Add more detailed exception handling
        tool = registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' could not be instantiated")
        
        trace_id = str(uuid.uuid4())
            
        # Fixed parameters: use proper parameter names and order
        result = await runner.run_tool(
            tool=tool_name, 
            tool_registry=registry, 
            args=body.args, 
            trace_id=trace_id
        )
        
        # Check if we need to extract the actual result from the nested structure
        if result.get('status') == 'success' and isinstance(result.get('result'), dict):
            # Extract the actual result for the response
            return ToolExecutionResponse(
                status="success",
                tool_name=tool_name,
                execution_time=round(time.perf_counter() - start, 6),
                result=result.get('result'),
            )
        else:
            return ToolExecutionResponse(
                status="success",
                tool_name=tool_name,
                execution_time=round(time.perf_counter() - start, 6),
                result=result,
            )
    except Exception as exc:
        logger.exception(f"Tool execution failed: {exc}")
        return ToolExecutionResponse(
            status="error",
            tool_name=tool_name,
            execution_time=round(time.perf_counter() - start, 6),
            error={"detail": str(exc)},
        )
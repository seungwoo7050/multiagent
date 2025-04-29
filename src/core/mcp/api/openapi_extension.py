from typing import Dict, Type, List, Any
from fastapi import FastAPI
from pydantic import BaseModel
from src.core.mcp.schema import BaseContextSchema, TaskContext, LLMInputContext, LLMOutputContext
from src.config.logger import get_logger
logger = get_logger(__name__)

def get_mcp_context_schemas() -> List[Type[BaseModel]]:
    schemas_to_include: List[Type[BaseModel]] = [BaseContextSchema, TaskContext, LLMInputContext, LLMOutputContext]
    logger.debug(f'Gathered {len(schemas_to_include)} MCP context schemas for potential OpenAPI inclusion.')
    return schemas_to_include

def add_mcp_schemas_to_openapi(openapi_schema: Dict[str, Any], app: FastAPI) -> None:
    logger.warning('`add_mcp_schemas_to_openapi` is currently designed as a placeholder. FastAPI usually handles Pydantic model schema generation automatically when used in routes. Manual schema manipulation is often unnecessary.')

def customize_openapi_for_mcp(app: FastAPI) -> None:
    openapi_func = app.openapi

    def custom_openapi() -> Dict[str, Any]:
        if openapi_func:
            openapi_schema = openapi_func()
        else:
            openapi_schema = {}
        openapi_schema['info']['x-mcp-support'] = {'version': '1.0.0', 'description': 'This API utilizes the Model Context Protocol (MCP) for certain operations.'}
        return openapi_schema
    app.openapi = custom_openapi
    logger.info('Custom OpenAPI generation function for MCP applied to FastAPI app.')
from typing import List, Type
from pydantic import BaseModel
from fastapi import FastAPI
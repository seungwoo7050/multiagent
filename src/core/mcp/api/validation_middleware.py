import json
from typing import Optional, Any, cast
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
from src.config.logger import get_logger
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.versioning import check_version_compatibility, get_latest_supported_version
from src.config.errors import ErrorCode
logger = get_logger(__name__)
MCP_VERSION_HEADER = 'X-MCP-Version'
MCP_CONTEXT_TYPE_HEADER = 'X-MCP-Context-Type'

class MCPContextValidationMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info('MCPContextValidationMiddleware initialized.')

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get('X-Request-ID', 'N/A')
        mcp_context: Optional[ContextProtocol] = None
        if hasattr(request.state, 'mcp_context'):
            mcp_context = cast(ContextProtocol, getattr(request.state, 'mcp_context', None))
            if mcp_context is not None and isinstance(mcp_context, ContextProtocol):
                context_id = getattr(mcp_context, 'context_id', 'N/A')
                context_type = type(mcp_context).__name__
                logger.debug(f'Found MCP context (ID: {context_id}, Type: {context_type}) in request state for validation.')
                is_valid = True
                error_detail = ''
                error_status_code = status.HTTP_400_BAD_REQUEST
                context_version = getattr(mcp_context, 'version', None)
                if context_version and isinstance(context_version, str):
                    if not check_version_compatibility(context_version):
                        is_valid = False
                        latest_supported = get_latest_supported_version()
                        error_detail = f"Unsupported MCP context version: '{context_version}'. Current system supports versions compatible with {latest_supported}."
                        error_status_code = status.HTTP_426_UPGRADE_REQUIRED
                        logger.warning(f'MCP Version Incompatibility for context {context_id}: {error_detail}')
                else:
                    logger.warning(f'MCP context {context_id} is missing version information or has invalid format. Treating as potentially incompatible.')
                if not is_valid:
                    logger.error(f'MCP Context validation failed for request {request.url.path} (ID: {request_id}, ContextID: {context_id}). Reason: {error_detail}')
                    return Response(status_code=error_status_code, content=json.dumps({'detail': error_detail}), media_type='application/json')
                else:
                    logger.debug(f'MCP Context (ID: {context_id}) validation successful.')
            elif mcp_context is None:
                logger.warning(f'request.state.mcp_context was None for request {request.url.path} (ID: {request_id}). Skipping MCP validation.')
            else:
                logger.error(f'Invalid object type found in request.state.mcp_context: {type(mcp_context).__name__} for request {request.url.path} (ID: {request_id}).')
                return Response(status_code=status.HTTP_400_BAD_REQUEST, content=json.dumps({'detail': 'Invalid context object found in request state.'}), media_type='application/json')
        response = await call_next(request)
        return response
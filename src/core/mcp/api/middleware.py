import json
import os
import sys
import time

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)
from starlette.types import ASGIApp

from src.config.logger import get_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = get_logger(__name__)
MCP_CONTENT_TYPES = {'application/msgpack', 'application/x-msgpack', 'application/json+mcp'}
MCP_VERSION_HEADER = 'X-MCP-Version'
MCP_CONTEXT_TYPE_HEADER = 'X-MCP-Context-Type'

class BasicMCPMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info('BasicMCPMiddleware initialized.')

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        request_id = request.headers.get('X-Request-ID', 'N/A')
        request_content_type = request.headers.get('content-type', '').lower()
        request.headers.get('accept', '').lower()
        mcp_version = request.headers.get(MCP_VERSION_HEADER)
        mcp_context_type_req = request.headers.get(MCP_CONTEXT_TYPE_HEADER)
        if any((ct in request_content_type for ct in MCP_CONTENT_TYPES)) or mcp_version:
            logger.debug(f'Request {request.url.path} (ID: {request_id}) potentially contains MCP context. Content-Type: {request_content_type}, MCP-Version: {mcp_version}, MCP-Context-Type: {mcp_context_type_req}')
        response: Response
        try:
            response = await call_next(request)
        except HTTPException as http_exc:
            response = Response(content=json.dumps({'detail': http_exc.detail}), status_code=http_exc.status_code, headers=dict(http_exc.headers) if http_exc.headers else None, media_type='application/json')
        except Exception as e:
            logger.error(f'Unhandled exception during request processing for {request.url.path} (ID: {request_id}): {e}', exc_info=True)
            response = Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({'detail': 'Internal Server Error'}))
        process_time = (time.time() - start_time) * 1000
        response.headers['X-Process-Time-Ms'] = str(process_time)
        logger.info(f'{request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms) (ReqID: {request_id})')
        response_content_type = response.headers.get('content-type', '').lower()
        response_mcp_version = response.headers.get(MCP_VERSION_HEADER)
        response_mcp_context_type = response.headers.get(MCP_CONTEXT_TYPE_HEADER)
        if any((ct in response_content_type for ct in MCP_CONTENT_TYPES)) or response_mcp_version:
            logger.debug(f'Response for {request.url.path} (ID: {request_id}) potentially contains MCP context. Content-Type: {response_content_type}, MCP-Version: {response_mcp_version}, MCP-Context-Type: {response_mcp_context_type}')
        return response
import asyncio
import json
import os
import sys
import time
from typing import Optional

from fastapi import Request, Response, status
from starlette.middleware.base import (BaseHTTPMiddleware,
                                       RequestResponseEndpoint)
from starlette.types import ASGIApp

from src.config.logger import get_logger
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.serialization import (SerializationError,
                                        SerializationFormat,
                                        deserialize_context)
from src.core.mcp.versioning import (check_version_compatibility,
                                     get_latest_supported_version)

logger = get_logger(__name__)
MCP_SERIALIZATION_FORMAT_MAP = {'application/msgpack': SerializationFormat.MSGPACK, 'application/x-msgpack': SerializationFormat.MSGPACK, 'application/json+mcp': SerializationFormat.JSON}
MCP_VERSION_HEADER = 'X-MCP-Version'
MCP_CONTEXT_TYPE_HEADER = 'X-MCP-Context-Type'

class MCPSerializationMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info('MCPSerializationMiddleware (with Validation) initialized.')

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        request_id = request.headers.get('X-Request-ID', 'N/A')
        mcp_context: Optional[ContextProtocol] = None
        serialization_format: Optional[SerializationFormat] = None
        request_content_type = request.headers.get('content-type', '').lower()
        matched_content_type = next((ct for ct in MCP_SERIALIZATION_FORMAT_MAP if ct in request_content_type), None)
        if matched_content_type:
            serialization_format = MCP_SERIALIZATION_FORMAT_MAP[matched_content_type]
            logger.debug(f'Request {request.url.path} (ID: {request_id}) has MCP Content-Type: {matched_content_type} (Format: {serialization_format.value})')
            body_bytes = await request.body()
            if not body_bytes:
                logger.warning(f'MCP request {request.url.path} (ID: {request_id}) has MCP Content-Type but empty body.')
                return Response(status_code=status.HTTP_400_BAD_REQUEST, content=json.dumps({'detail': 'Request body is empty for MCP content type'}), media_type='application/json')
            try:
                if asyncio.iscoroutinefunction(deserialize_context):
                    mcp_context = await deserialize_context(body_bytes, format=serialization_format)
                else:
                    mcp_context = await asyncio.to_thread(deserialize_context, body_bytes, format=serialization_format)
                if not isinstance(mcp_context, ContextProtocol):
                    logger.error(f'Deserialization resulted in unexpected type: {type(mcp_context).__name__} for {request.url.path} (ID: {request_id})')
                    raise SerializationError('Deserialized object is not a valid ContextProtocol instance.')
                context_version = getattr(mcp_context, 'version', None)
                if not context_version or not isinstance(context_version, str):
                    logger.warning(f'MCP context (Type: {type(mcp_context).__name__}, ID: {getattr(mcp_context, 'context_id', 'N/A')}) is missing version information.')
                elif not check_version_compatibility(context_version):
                    latest_supported = get_latest_supported_version()
                    detail = f"Unsupported MCP context version: '{context_version}'. Current system supports versions compatible with {latest_supported}."
                    logger.error(f'Version incompatibility for {request.url.path} (ID: {request_id}): {detail}')
                    return Response(status_code=status.HTTP_400_BAD_REQUEST, content=json.dumps({'detail': detail}), media_type='application/json')
                else:
                    logger.debug(f"MCP context version '{context_version}' is compatible.")
                request.state.mcp_context = mcp_context
                logger.debug(f'Validated and stored MCP context (Type: {type(mcp_context).__name__}) in request state for {request.url.path} (ID: {request_id})')
            except (SerializationError, ValueError) as val_err:
                logger.error(f'MCP Request Deserialization/Validation failed for {request.url.path} (ID: {request_id}): {str(val_err)}', exc_info=True)
                return Response(status_code=status.HTTP_400_BAD_REQUEST, content=json.dumps({'detail': f'Invalid MCP context: {str(val_err)}'}), media_type='application/json')
            except Exception as e:
                logger.error(f'Unexpected error during MCP request deserialization for {request.url.path} (ID: {request_id}): {e}', exc_info=True)
                return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({'detail': 'Failed to process MCP context'}), media_type='application/json')
        response: Response
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f'Unhandled exception during request processing for {request.url.path} (ID: {request_id}): {e}', exc_info=True)
            response = Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=json.dumps({'detail': 'Internal Server Error'}))
        process_time = (time.time() - start_time) * 1000
        response.headers['X-Process-Time-Ms'] = str(process_time)
        logger.info(f'{request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms) (ReqID: {request_id})')
        return response
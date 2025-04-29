from typing import Any, Dict, Optional, List, cast
from fastapi import APIRouter, Depends, HTTPException, status, Body, Path
from src.config.logger import get_logger
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.config.errors import NotFoundError, BaseError, ErrorCode
logger = get_logger(__name__)
router = APIRouter(prefix='/mcp/contexts', tags=['MCP Context Management'])
_context_store: Dict[str, ContextProtocol] = {}

async def get_context_store() -> Dict[str, ContextProtocol]:
    logger.debug('Using placeholder in-memory context store dependency.')
    return _context_store

class ContextResponse(BaseModel):
    context_id: str
    context_type: str
    version: str
    timestamp: float
    metadata: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None

class ContextListResponse(BaseModel):
    contexts: List[Dict[str, Any]]
    total: int

class ContextOperationResponse(BaseModel):
    context_id: str
    status: str
    message: Optional[str] = None

@router.get('/{context_id}', response_model=Any, summary='Get MCP Context by ID', description='Retrieves a specific MCP context object using its unique ID.', responses={status.HTTP_404_NOT_FOUND: {'description': 'Context not found'}, status.HTTP_500_INTERNAL_SERVER_ERROR: {'description': 'Internal server error'}})
async def get_context_by_id(context_id: str=Path(..., description='The unique ID of the context to retrieve'), context_store: Dict[str, ContextProtocol]=Depends(get_context_store)):
    logger.info(f'Request received to get context with ID: {context_id}')
    try:
        context = context_store.get(context_id)
        if context is None:
            logger.warning(f'Context not found for ID: {context_id}')
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"MCP Context with ID '{context_id}' not found.")
        logger.debug(f'Context {context_id} retrieved successfully (Type: {type(context).__name__}).')
        return context
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error retrieving context {context_id}: {e}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to retrieve context: {str(e)}')

@router.post('', response_model=ContextOperationResponse, status_code=status.HTTP_201_CREATED, summary='Create or Update MCP Context', description='Creates a new MCP context or updates an existing one based on the provided data. The context ID is usually included in the request body.', responses={status.HTTP_400_BAD_REQUEST: {'description': 'Invalid context data provided'}, status.HTTP_500_INTERNAL_SERVER_ERROR: {'description': 'Internal server error'}})
async def create_or_update_context(request: Request, context_data: Dict[str, Any]=Body(..., description='MCP Context object data to create or update'), context_store: Dict[str, ContextProtocol]=Depends(get_context_store)):
    context_to_process: Optional[ContextProtocol] = None
    context_id: Optional[str] = None
    operation_status = 'failed'
    if hasattr(request.state, 'mcp_context') and isinstance(request.state.mcp_context, ContextProtocol):
        context_to_process = cast(ContextProtocol, request.state.mcp_context)
        context_id = getattr(context_to_process, 'context_id', None)
        logger.info(f'Processing MCP context from request state (ID: {context_id})')
    elif context_data:
        context_id = context_data.get('context_id')
        context_type_str = context_data.get('__type__') or context_data.get('class')
        logger.info(f'Processing MCP context from request body (ID: {context_id}, Type Hint: {context_type_str})')
        try:
            context_to_process = BaseContextSchema.model_validate(context_data)
            context_id = getattr(context_to_process, 'context_id', context_id)
        except Exception as parse_err:
            logger.error(f'Failed to parse context data from request body: {parse_err}', exc_info=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid context data format: {parse_err}')
    if not context_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'context_id' in context data.")
    if not context_to_process:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No valid context data found in request.')
    try:
        context_store[context_id] = context_to_process
        operation_status = 'created_or_updated'
        logger.info(f'Context {context_id} (Type: {type(context_to_process).__name__}) saved/updated successfully.')
        return ContextOperationResponse(context_id=context_id, status=operation_status, message=f'Context {context_id} processed successfully.')
    except Exception as e:
        logger.exception(f'Error saving/updating context {context_id}: {e}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to process context: {str(e)}')
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable
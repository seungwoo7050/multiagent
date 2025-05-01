# src/api/routes/context.py
from typing import Any, Dict, Optional, List, cast
from fastapi import APIRouter, Depends, HTTPException, status, Body, Path, Request # Added Request
from pydantic import BaseModel # Added BaseModel

from src.config.logger import get_logger
# MCP Imports (assuming these exist based on roadmap)
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema
from src.core.mcp.serialization import deserialize_context, SerializationError, SerializationFormat
from src.config.errors import NotFoundError, BaseError, ErrorCode, APIError # Adjusted imports
# Import the MemoryManager dependency
from src.api.dependencies import MemoryManagerDep, get_memory_manager_dependency
from src.memory.manager import MemoryManager

logger = get_logger(__name__)
router = APIRouter(prefix='/mcp/contexts', tags=['MCP Context Management'])

# --- Removed Placeholder Store ---
# _context_store: Dict[str, ContextProtocol] = {}
# async def get_context_store() -> Dict[str, ContextProtocol]: ... (REMOVE THIS)

# --- Pydantic Models for Responses (Optional but good practice) ---
class ContextResponse(BaseModel):
    context_id: str
    # Add other fields you want to return, potentially dynamically based on context type
    # For simplicity, returning the raw data stored
    data: Dict[str, Any]

class ContextListResponse(BaseModel):
    contexts: List[Dict[str, Any]] # List of context representations
    total: int

class ContextOperationResponse(BaseModel):
    context_id: str
    status: str
    message: Optional[str] = None

@router.get(
    '/{context_id}',
    # response_model=Any, # Use Any or create a more specific dynamic model if possible
    summary='Get MCP Context by ID',
    description='Retrieves a specific MCP context object using its unique ID from the Memory Manager.',
    responses={
        status.HTTP_404_NOT_FOUND: {'description': 'Context not found'},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {'description': 'Internal server error'}
    }
)
async def get_context_by_id(
    context_id: str = Path(..., description='The unique ID of the context to retrieve'),
    # Inject the MemoryManager
    memory_manager: MemoryManager = Depends(get_memory_manager_dependency)
):
    """
    Retrieves an MCP context object by its ID using the MemoryManager.
    """
    logger.info(f'Request received to get context with ID: {context_id}')
    try:
        # Use MemoryManager's load method. Context ID is the primary identifier.
        # The 'key' within the context might be less relevant here if context_id is unique.
        # Assuming context_id is the primary key used in MemoryManager for top-level contexts.
        # If contexts are stored under specific keys *within* a context_id, adjust logic.
        # Let's assume the MemoryManager stores the context itself keyed by context_id.
        # A common pattern is to use a fixed key like 'main' or 'context_data'.
        context_key_in_memory = "context_data" # Or derive from context type if needed
        context_data = await memory_manager.load(key=context_key_in_memory, context_id=context_id, use_cache=True)

        if context_data is None:
            logger.warning(f'Context not found via MemoryManager for ID: {context_id}')
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"MCP Context with ID '{context_id}' not found.")

        logger.debug(f'Context {context_id} retrieved successfully from MemoryManager (Type: {type(context_data).__name__}).')
        # Decide how to return the context. Returning the raw data dictionary is simplest.
        # If the context object itself is needed, deserialize it here if MemoryManager doesn't already.
        if isinstance(context_data, ContextProtocol):
             return context_data.serialize() # Use serialize if available
        elif isinstance(context_data, BaseModel):
            return context_data.model_dump(mode='json')
        elif isinstance(context_data, dict):
            return context_data # Return raw dict if stored as such
        else:
             logger.warning(f"Context data for {context_id} is of unexpected type: {type(context_data)}. Returning string representation.")
             return {"context_id": context_id, "raw_data": str(context_data)}

    except HTTPException:
        raise # Re-raise specific HTTP exceptions
    except Exception as e:
        logger.exception(f'Error retrieving context {context_id} via MemoryManager: {e}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to retrieve context: {str(e)}')

@router.post(
    '',
    response_model=ContextOperationResponse,
    status_code=status.HTTP_201_CREATED,
    summary='Create or Update MCP Context',
    description='Creates a new MCP context or updates an existing one using the Memory Manager.',
    responses={
        status.HTTP_400_BAD_REQUEST: {'description': 'Invalid context data provided'},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {'description': 'Internal server error'}
    }
)
async def create_or_update_context(
    request: Request, # Added Request to access state if needed
    context_data: Dict[str, Any] = Body(..., description='MCP Context object data to create or update'),
    # Inject the MemoryManager
    memory_manager: MemoryManager = Depends(get_memory_manager_dependency)
):
    """
    Saves or updates an MCP context object using the MemoryManager.
    Expects the context data in the request body.
    """
    context_to_process: Optional[ContextProtocol] = None
    context_id: Optional[str] = None
    operation_status = 'failed'

    # Prefer context deserialized by middleware if available
    if hasattr(request.state, 'mcp_context') and isinstance(request.state.mcp_context, ContextProtocol):
        context_to_process = cast(ContextProtocol, request.state.mcp_context)
        context_id = getattr(context_to_process, 'context_id', None)
        logger.info(f'Processing MCP context from request state (ID: {context_id})')
    elif context_data:
        context_id = context_data.get('context_id')
        context_type_str = context_data.get('__type__') or context_data.get('class') or context_data.get('context_type') # Try common keys
        logger.info(f'Processing MCP context from request body (ID: {context_id}, Type Hint: {context_type_str})')
        # Attempt to validate/deserialize the raw data
        try:
            # Use BaseContextSchema for basic validation if specific type unknown
            context_obj = BaseContextSchema.model_validate(context_data)
            context_id = context_obj.context_id # Ensure ID from validated model
            context_to_process = context_obj # Use the validated object
        except Exception as parse_err:
            logger.error(f'Failed to parse context data from request body: {parse_err}', exc_info=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Invalid context data format: {parse_err}')
    else:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No context data found in request body or state.')


    if not context_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'context_id' in context data.")
    if not context_to_process:
         # This case should be covered by the checks above
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Failed to prepare context object for saving.')

    try:
        # Decide the key under which to store the context within the context_id scope
        context_key_in_memory = "context_data" # Or derive from context type

        # Serialize the context object before saving if MemoryManager expects bytes/dict
        # If MemoryManager handles object serialization, pass context_to_process directly.
        # Assuming MemoryManager handles serialization via its primary store:
        data_to_save = context_to_process

        # Save using MemoryManager
        success = await memory_manager.save(
            key=context_key_in_memory,
            context_id=context_id,
            data=data_to_save,
            update_cache=True # Keep cache consistent
        )

        if success:
            operation_status = 'created_or_updated'
            logger.info(f'Context {context_id} (Type: {type(context_to_process).__name__}) saved/updated successfully via MemoryManager.')
            return ContextOperationResponse(context_id=context_id, status=operation_status, message=f'Context {context_id} processed successfully.')
        else:
            logger.error(f'MemoryManager failed to save context {context_id}.')
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to save context data.')

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f'Error saving/updating context {context_id} via MemoryManager: {e}')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'Failed to process context: {str(e)}')
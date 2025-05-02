import asyncio
from typing import Any, Optional

from src.api.streaming import ConnectionManager, get_connection_manager
from src.config.logger import get_logger
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.serialization import (SerializationError,
                                        SerializationFormat, serialize_context)

logger = get_logger(__name__)

class MCPWebSocketAdapter:

    def __init__(self, connection_manager: ConnectionManager):
        if not isinstance(connection_manager, ConnectionManager):
            raise TypeError('connection_manager must be an instance of ConnectionManager')
        self.connection_manager = connection_manager
        logger.debug('MCPWebSocketAdapter initialized.')

    async def stream_context(self, context: ContextProtocol, task_id: Optional[str]=None, target_format: SerializationFormat=SerializationFormat.JSON) -> bool:
        target_task_id: Optional[str] = task_id
        if not target_task_id:
            target_task_id = getattr(context, 'task_id', None)
            if not target_task_id and hasattr(context, 'metadata') and isinstance(context.metadata, dict):
                target_task_id = context.metadata.get('task_id')
        if not target_task_id:
            context_type = type(context).__name__
            context_id = getattr(context, 'context_id', 'N/A')
            logger.warning(f'Cannot stream context (ID: {context_id}, Type: {context_type}): Unable to determine target task_id.')
            return False
        context_id = getattr(context, 'context_id', 'N/A')
        logger.debug(f'Attempting to stream context (ID: {context_id}, Type: {type(context).__name__}) to task_id: {target_task_id}')
        try:
            serialized_data: bytes = serialize_context(context, format=target_format)
            if target_format == SerializationFormat.JSON:
                message_to_send: Any = serialized_data.decode('utf-8') # ???
                try:
                    message_dict = context.serialize()
                except AttributeError:
                    message_dict = context.model_dump(mode='json')
                message_payload = message_dict
            else:
                logger.warning(f"Unsupported serialization format '{target_format.value}' for WebSocket streaming. Defaulting to JSON string.")
                message_payload = serialized_data.decode('utf-8')
        except SerializationError as e:
            logger.error(f'Failed to serialize context (ID: {context_id}) for streaming to task {target_task_id}: {e}', exc_info=True)
            return False
        except Exception as e:
            logger.error(f'Unexpected error preparing context (ID: {context_id}) for streaming: {e}', exc_info=True)
            return False
        try:
            await self.connection_manager.broadcast_to_task(target_task_id, message_payload)
            logger.debug(f'Successfully broadcasted context update for task {target_task_id}.')
            return True
        except Exception as e:
            logger.error(f'Failed to broadcast context update for task {target_task_id}: {e}', exc_info=True)
            return False
_websocket_adapter_instance: Optional[MCPWebSocketAdapter] = None
_websocket_adapter_lock = asyncio.Lock()

async def get_websocket_adapter(connection_manager: Optional[ConnectionManager]=None) -> MCPWebSocketAdapter:
    global _websocket_adapter_instance
    if _websocket_adapter_instance is not None:
        return _websocket_adapter_instance
    async with _websocket_adapter_lock:
        if _websocket_adapter_instance is None:
            if not connection_manager:
                try:
                    connection_manager = get_connection_manager()
                except Exception as cm_err:
                    logger.error(f'Failed to get default ConnectionManager: {cm_err}')
                    raise ValueError('ConnectionManager dependency not available for MCPWebSocketAdapter.')
            _websocket_adapter_instance = MCPWebSocketAdapter(connection_manager)
            logger.info('Singleton MCPWebSocketAdapter instance created.')
    if _websocket_adapter_instance is None:
        raise RuntimeError('Failed to create MCPWebSocketAdapter instance.')
    return _websocket_adapter_instance
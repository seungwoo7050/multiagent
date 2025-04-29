import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set, DefaultDict
from collections import defaultdict
import json
from src.config.logger import get_logger
logger = get_logger(__name__)

class ConnectionManager:

    def __init__(self):
        self.active_connections: DefaultDict[str, Set[WebSocket]] = defaultdict(set)
        logger.info('ConnectionManager initialized.')

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id].add(websocket)
        logger.info(f'WebSocket connected: {websocket.client.host}:{websocket.client.port} for task_id: {task_id}')
        logger.debug(f'Active connections for task {task_id}: {len(self.active_connections[task_id])}')
        await self.send_personal_message({'message': f'Connected for task {task_id} updates.'}, websocket)

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
                logger.debug(f'Removed task_id {task_id} from active connections (no listeners).')
            logger.info(f'WebSocket disconnected: {websocket.client.host}:{websocket.client.port} from task_id: {task_id}')
            logger.debug(f'Remaining connections for task {task_id}: {len(self.active_connections.get(task_id, set()))}')
        else:
            logger.warning(f'Attempted to disconnect websocket for task_id {task_id}, but task_id not found in active connections.')

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
            logger.debug(f'Sent personal message to {websocket.client.host}:{websocket.client.port}: {message}')
        except Exception as e:
            logger.warning(f'Failed to send personal message to {websocket.client.host}:{websocket.client.port}: {e}')

    async def broadcast_to_task(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            connections = list(self.active_connections[task_id])
            message_str = json.dumps(message)
            logger.info(f'Broadcasting message to {len(connections)} connections for task_id {task_id}: {message_str[:100]}...')
            results = await asyncio.gather(*[conn.send_json(message) for conn in connections], return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    conn = connections[i]
                    logger.warning(f'Failed to broadcast to {conn.client.host}:{conn.client.port} for task {task_id}: {result}')
manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    return manager
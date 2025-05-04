# src/api/streaming.py
from typing import Any, Dict, List

from fastapi import WebSocket

from src.config.logger import get_logger

logger = get_logger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        logger.info("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logger.debug(f"Client connected to task {task_id}. Total connections: {len(self.active_connections[task_id])}")

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
            logger.debug(f"Client disconnected from task {task_id}")

    async def send_personal_message(self, message: Any, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: Any, task_id: str):
        if task_id in self.active_connections:
            connections = self.active_connections[task_id]
            logger.debug(f"Broadcasting to {len(connections)} connection(s) for task {task_id}")
            
            for connection in connections:
                try:
                    await connection.send_json(message)
                    logger.debug(f"Message sent to client for task {task_id}")
                except Exception as e:
                    logger.error(f"Failed to send message to client for task {task_id}: {e}", exc_info=True)
                    # Remove dead connections
                    self.active_connections[task_id].remove(connection)
        else:
            logger.debug(f"No active connections for task {task_id}")

# 전역 인스턴스
_connection_manager = None

def get_connection_manager() -> ConnectionManager:
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager
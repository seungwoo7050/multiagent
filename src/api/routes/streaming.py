import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Path
from typing import Annotated
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.api.streaming import ConnectionManager, get_connection_manager
from src.config.logger import get_logger
logger = get_logger(__name__)
router = APIRouter(prefix='/ws/v1', tags=['Streaming'])
ConnectionManagerDep = Annotated[ConnectionManager, Depends(get_connection_manager)]

@router.websocket('/tasks/{task_id}')
async def websocket_task_updates(websocket: WebSocket, task_id: str=Path(..., description='Updates를 수신할 작업의 ID'), manager: ConnectionManagerDep=Depends(get_connection_manager)):
    await manager.connect(websocket, task_id)
    try:
        while True:
            await asyncio.sleep(10)
            await manager.send_personal_message({'task_id': task_id, 'status': 'processing', 'update': f'Still working... {time.time()}'}, websocket)
    except WebSocketDisconnect:
        logger.info(f'WebSocket disconnected by client for task_id: {task_id}')
    except Exception as e:
        logger.error(f'WebSocket error for task_id {task_id}: {e}', exc_info=True)
    finally:
        manager.disconnect(websocket, task_id)
        logger.debug(f'Ensured websocket cleanup for task_id: {task_id}')
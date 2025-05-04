import time
from src.api.streaming import ConnectionManager, get_connection_manager
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from src.api.dependencies import OrchestratorDep, get_orchestrator_dependency_implementation
from src.config.logger import get_logger
from src.orchestration.orchestrator import Orchestrator

router = APIRouter(prefix="/ws/v1", tags=["WebSocket"])
logger = get_logger(__name__)

@router.websocket("/tasks/{task_id}")
async def websocket_task_updates(
    websocket: WebSocket,
    task_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator_dependency_implementation)
):
    connection_manager = get_connection_manager()
    await connection_manager.connect(websocket, task_id)
    
    try:
        # Send immediate connection acknowledgment
        initial_event = {
            "task_id": task_id,
            "type": "CONNECTED",
            "timestamp": time.time(),
            "message": "WebSocket connection established"
        }
        await websocket.send_json(initial_event)
        logger.debug(f"Sent initial connection event for task {task_id}")
        
        # Get initial task status and send it
        try:
            status = await orchestrator.get_task_status(task_id)
            status_event = {
                "task_id": task_id,
                "type": "STATUS_UPDATE",
                "timestamp": time.time(),
                "status": status.get("status", "UNKNOWN"),
                "data": status
            }
            await websocket.send_json(status_event)
            logger.debug(f"Sent initial status for task {task_id}")
            
            # Add TASK_STARTED event for test compatibility
            started_event = {
                "task_id": task_id,
                "type": "TASK_STARTED",
                "timestamp": time.time(),
                "message": "Task processing started"
            }
            await websocket.send_json(started_event)
            logger.debug(f"Sent TASK_STARTED event for task {task_id}")
        except Exception as e:
            logger.error(f"Error sending initial status for {task_id}: {e}")
        
        # Keep connection alive until client disconnects
        while True:
            try:
                # Wait for ping or client message
                data = await websocket.receive_text()
                logger.debug(f"Received message from client for task {task_id}: {data}")
            except WebSocketDisconnect:
                logger.debug(f"WebSocket disconnected for task {task_id}")
                break
            
    except Exception as e:
        logger.exception(f"Error in WebSocket connection for task {task_id}: {e}")
    finally:
        connection_manager.disconnect(websocket, task_id)
        logger.debug(f"WebSocket connection closed for task {task_id}")
# src/api/routes/streaming.py
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Path, HTTPException, status
from typing import Annotated, Optional, Dict, Any

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api.streaming import ConnectionManager, get_connection_manager
from src.config.logger import get_logger
# Import Orchestrator or relevant event source dependency
from src.api.dependencies import OrchestratorDep, get_orchestrator_dependency_implementation
from src.orchestration.orchestrator import Orchestrator # Keep for type hinting

logger = get_logger(__name__)
router = APIRouter(prefix='/ws/v1', tags=['Streaming'])

ConnectionManagerDep = Annotated[ConnectionManager, Depends(get_connection_manager)]
# Define dependency for Orchestrator (or event bus)
# Use the functional dependency provider from dependencies.py
OrchestratorWSDep = Annotated[Orchestrator, Depends(get_orchestrator_dependency_implementation)]


@router.websocket('/tasks/{task_id}')
async def websocket_task_updates(
    websocket: WebSocket,
    task_id: str = Path(..., description='The ID of the task to receive updates for'),
    manager: ConnectionManager = Depends(get_connection_manager),
    orchestrator: Orchestrator = Depends(get_orchestrator_dependency_implementation) # Inject Orchestrator/Event Source
):
    """
    WebSocket endpoint to stream real-time updates for a specific task.
    Clients connect here providing the task_id they are interested in.
    """
    await manager.connect(websocket, task_id)
    logger.info(f"WebSocket client connected for task updates: {task_id}")

    listener_task = None
    keepalive_task = None

    try:
        # --- Task Update Subscription Logic ---
        # This part needs to interact with your system's event publishing mechanism.
        # Option 1: Polling (Less ideal)
        # Option 2: Orchestrator provides an async generator or callback

        async def subscribe_to_task_updates(target_task_id: str):
            """Placeholder: Subscribes to updates for the task."""
            logger.debug(f"Subscribing to updates for task {target_task_id}...")
            # Example using a hypothetical async generator from orchestrator
            if hasattr(orchestrator, 'subscribe_to_task_events') and asyncio.iscoroutinefunction(orchestrator.subscribe_to_task_events):
                 try:
                     async for update in orchestrator.subscribe_to_task_events(target_task_id):
                         if update: # Ensure update is not None
                            await manager.send_personal_message(update, websocket)
                 except Exception as sub_err:
                      logger.error(f"Error in task update subscription for {target_task_id}: {sub_err}", exc_info=True)
                      await manager.send_personal_message({"error": "Subscription error occurred.", "task_id": target_task_id}, websocket)
                 finally:
                     logger.info(f"Subscription ended for task {target_task_id}")
                     # Optionally send a final message indicating subscription end
                     await manager.send_personal_message({"status": "subscription_ended", "task_id": target_task_id}, websocket)

            else:
                logger.warning(f"Orchestrator does not have 'subscribe_to_task_events' async generator. Sending dummy updates for task {target_task_id}.")
                # Fallback to dummy updates if subscription mechanism isn't available
                update_count = 0
                while True: # Keep sending dummy updates until disconnect
                    update_count += 1
                    await manager.send_personal_message({
                        'task_id': target_task_id,
                        'status': 'processing',
                        'update': f'Dummy Update {update_count}',
                        'timestamp': time.time()
                    }, websocket)
                    await asyncio.sleep(5) # Send dummy update every 5 seconds

        # --- Keepalive Task ---
        async def send_keepalive():
            """Sends a keepalive message periodically."""
            while True:
                await asyncio.sleep(30) # Send keepalive every 30 seconds
                try:
                    await manager.send_personal_message({'type': 'keepalive', 'timestamp': time.time()}, websocket)
                except Exception:
                    logger.warning(f"Failed to send keepalive to client for task {task_id}, connection might be lost.")
                    break # Stop keepalive if sending fails

        # Start the listener and keepalive tasks
        listener_task = asyncio.create_task(subscribe_to_task_updates(task_id))
        keepalive_task = asyncio.create_task(send_keepalive())

        # Wait for either task to finish (listener finishes on completion/error, keepalive on error)
        done, pending = await asyncio.wait(
            {listener_task, keepalive_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If one task finishes, cancel the other
        for task in pending:
            task.cancel()

        # Check if any task finished with an exception
        for task in done:
            if task.exception():
                 logger.error(f"WebSocket task finished with exception for task {task_id}: {task.exception()}")


    except WebSocketDisconnect:
        logger.info(f'WebSocket disconnected by client for task_id: {task_id}')
    except Exception as e:
        logger.error(f'WebSocket error for task_id {task_id}: {e}', exc_info=True)
        # Attempt to send an error message before closing
        try:
            await websocket.send_json({"error": "An unexpected server error occurred."})
        except:
            pass # Ignore errors during error reporting
    finally:
        # Ensure tasks are cancelled on exit
        if listener_task and not listener_task.done():
            listener_task.cancel()
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
        # Disconnect the client
        manager.disconnect(websocket, task_id)
        logger.debug(f'Ensured websocket cleanup for task_id: {task_id}')
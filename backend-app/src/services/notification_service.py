import asyncio
from collections import defaultdict
from typing import DefaultDict, List, Dict, Any, Optional
from fastapi import WebSocket                       

from src.utils.logger import get_logger
from src.schemas.websocket_models import WebSocketMessageBase

logger = get_logger(__name__)
                                      
class NotificationService:
    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, List[WebSocket]] = defaultdict(list)
                                                       
        self._last_message: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        logger.info("NotificationService initialized.")

    async def subscribe(self, task_id: str, websocket: WebSocket) -> None:
                           
        logger.info(f"NotificationService: Attempting to subscribe task_id: {task_id} for client: {websocket.client}")
        cached_msg_to_send = None                  

        async with self._lock:
                               
            logger.debug(f"NotificationService: Lock acquired for task_id: {task_id}, client: {websocket.client}")
            first_join = websocket not in self._subscribers[task_id]
            if first_join:
                self._subscribers[task_id].append(websocket)
                                   
                logger.info(f"NotificationService: Client {websocket.client} newly subscribed to task_id: {task_id}. Total subscribers: {len(self._subscribers[task_id])}")
            else:
                                   
                logger.info(f"NotificationService: Client {websocket.client} re-subscribed or already present for task_id: {task_id}. Total subscribers: {len(self._subscribers[task_id])}")

            cached_msg = self._last_message.get(task_id)
            if cached_msg is not None and first_join:
                cached_msg_to_send = cached_msg                    
                                   
                logger.debug(f"NotificationService: Found cached message for task_id: {task_id} to replay for new subscriber.")
                               
            logger.debug(f"NotificationService: Lock released for task_id: {task_id}, client: {websocket.client}")

        if cached_msg_to_send is not None:                         
            try:
                                   
                logger.info(f"NotificationService: Attempting to replay cached message to client: {websocket.client} for task_id: {task_id}")
                await websocket.send_json(cached_msg_to_send)
                                   
                logger.debug(f"NotificationService: Successfully replayed cached message to client: {websocket.client} for task_id: {task_id}")
            except Exception as exc:
                                   
                logger.warning(f"NotificationService: Failed to replay cached message to client: {websocket.client} for task_id: {task_id}. Error: {exc}", exc_info=True)
                           
        logger.info(f"NotificationService: Subscription process completed for task_id: {task_id}, client: {websocket.client}")

    async def broadcast_to_task(
        self, task_id: str, message_model: WebSocketMessageBase
    ) -> None:
                              
        message_dict = message_model.model_dump(mode="json")

        async with self._lock:
            self._last_message[task_id] = message_dict                  
            recipients = list(self._subscribers.get(task_id, []))     

        if not recipients:
            logger.debug(f"No subscribers for {task_id}; message cached only.")
            return

        logger.info(f"Broadcast '{message_model.event_type}' to "
                    f"{len(recipients)} subscriber(s) of {task_id}")

        disconnected: List[WebSocket] = []
        for ws in recipients:
            try:
                await ws.send_json(message_dict)
            except Exception as exc:
                logger.warning(f"Send to {ws.client} failed: {exc}")
                disconnected.append(ws)

                  
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    if ws in self._subscribers.get(task_id, []):
                        self._subscribers[task_id].remove(ws)
                if not self._subscribers.get(task_id):
                    self._subscribers.pop(task_id, None)
                    
    async def unsubscribe(self, task_id: str, websocket: WebSocket):
        """특정 task_id에 대한 구독을 취소합니다."""
        async with self._lock:
            if websocket in self._subscribers[task_id]:
                self._subscribers[task_id].remove(websocket)
                logger.info(f"WebSocket {websocket.client} unsubscribed from task_id: {task_id}")
                                                                       
                if not self._subscribers[task_id]:
                    del self._subscribers[task_id]
                    logger.debug(f"No more subscribers for task_id: {task_id}, removing entry.")
            else:
                logger.debug(f"WebSocket {websocket.client} was not subscribed to task_id: {task_id} or already removed.")

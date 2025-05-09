# src/services/notification_service.py
import asyncio
from collections import defaultdict
from typing import DefaultDict, List, Dict, Any, Optional
from fastapi import WebSocket # WebSocket 타입을 위해 임포트

from src.utils.logger import get_logger
# 이전 단계에서 정의한 WebSocket 메시지 모델들을 임포트합니다.
from src.schemas.websocket_models import WebSocketMessageBase

logger = get_logger(__name__)

# src/services/notification_service.py
class NotificationService:
    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, List[WebSocket]] = defaultdict(list)
        # ★ 항상 dict 형태로만 저장해서 subscribe 쪽에서 그대로 보내도록 통일
        self._last_message: Dict[str, dict] = {}
        self._lock = asyncio.Lock()
        logger.info("NotificationService initialized.")

    async def subscribe(self, task_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            first_join = websocket not in self._subscribers[task_id]
            if first_join:
                self._subscribers[task_id].append(websocket)
                logger.info(f"{websocket.client} subscribed to {task_id}")
            # dict 형태로 저장돼 있으므로 그대로 얻음
            cached_msg = self._last_message.get(task_id)

        # 🔔 구독 직후 캐시된 마지막 메시지 리플레이
        if cached_msg is not None and first_join:
            try:
                await websocket.send_json(cached_msg)
                logger.debug(f"Replayed cached msg to late subscriber of {task_id}")
            except Exception as exc:
                logger.warning(f"Replay to {websocket.client} failed: {exc}")

    async def broadcast_to_task(
        self, task_id: str, message_model: WebSocketMessageBase
    ) -> None:
        # Pydantic → dict (V2)
        message_dict = message_model.model_dump(mode="json")

        async with self._lock:
            self._last_message[task_id] = message_dict            # ★ 캐시
            recipients = list(self._subscribers.get(task_id, [])) # 복사

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

        # 끊긴 소켓 정리
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
                # 해당 task_id에 더 이상 구독자가 없으면 리스트에서 키를 제거할 수 있습니다 (선택 사항)
                if not self._subscribers[task_id]:
                    del self._subscribers[task_id]
                    logger.debug(f"No more subscribers for task_id: {task_id}, removing entry.")
            else:
                logger.debug(f"WebSocket {websocket.client} was not subscribed to task_id: {task_id} or already removed.")
                
                
"""
레이스 컨디션 이슈 해결:
실제 서비스 관점에서의 의미
실시간 모니터링 UI
  - 사용자가 페이지를 새로고침하더라도 마지막 상태를 즉시 받을 수 있습니다.
다중 클라이언트
  - 동일 task를 여러 창­/디바이스에서 구독해도 첫 연결 시 동일한 “현재 상태”를 받습니다.
메모리 관리
  - 워크플로가 완전히 끝난 뒤 self._last_message.pop(task_id, None) 같은 정리 로직을 추가하면 캐시 누수를 방지할 수 있습니다.

정리
- 문제 : 메시지가 구독보다 먼저 발행 ⇒ 구독자가 받을 것이 없어서 타임-아웃.
- 해결 : NotificationService에 “최근 메시지 캐시” 와 “구독 직후 리플레이” 기능을 추가.
- 결과 : 뒤늦은 구독자도 즉시 메시지를 받아 테스트가 성공.
"""

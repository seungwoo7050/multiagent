# tests/conftest.py
import inspect
import pathlib
import os
import json
import pytest
import asyncio, functools
from unittest.mock import MagicMock, AsyncMock # AsyncMock 추가 (비동기 메서드 모킹용)
from fastapi.testclient import TestClient
from src.agents.orchestrator import Orchestrator
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager # 실제 사용하는 클래스로 변경
from src.memory.memory_manager import MemoryManager # 실제 사용하는 클래스로 변경
from src.services.notification_service import NotificationService

from src.utils.telemetry import setup_telemetry, clear_test_spans

setup_telemetry(for_testing=True)

from src.api.dependencies import get_memory_manager
from src.api.app import app

# --- 기존 fixture들 (예: mock_llm_client, mock_tool_manager 등)은 그대로 둡니다 ---

# NotificationService 경로 확인 필요
try:
    from src.services.notification_service import NotificationService
    NOTIFICATION_SERVICE_AVAILABLE = True
except ImportError:
    NotificationService = None # 임시 타입 설정 또는 에러 처리
    NOTIFICATION_SERVICE_AVAILABLE = False
    print("Warning: src.services.notification_service not found, mock fixture might be incomplete.")

@pytest.fixture(autouse=True, scope="function") # "session" 또는 "function" scope 사용 가능
def otel_test_setup():
    # 각 테스트 전에 InMemoryExporter로 강제 재설정
    setup_telemetry(force_setup=True, for_testing=True)
    yield
    # 각 테스트 후 Span 정리 (선택 사항)
    clear_test_spans()
    
@pytest.fixture
def mock_notification_service() -> MagicMock:
    """NotificationService의 목 객체를 반환하는 fixture."""
    if not NOTIFICATION_SERVICE_AVAILABLE:
        # 서비스 클래스를 찾을 수 없는 경우 기본적인 MagicMock 반환
        return MagicMock()

    # NotificationService의 spec을 사용하여 더 정확한 목 객체 생성
    mock_service = MagicMock(spec=NotificationService)

    # broadcast_to_task는 비동기 메서드이므로 AsyncMock으로 설정
    # 또는 await 가능한 mock 함수로 설정
    async def mock_broadcast(*args, **kwargs):
        # 실제 전송 로직 대신 호출 기록만 남기거나 아무것도 안 함
        # print(f"Mock NotificationService.broadcast_to_task called with: args={args}, kwargs={kwargs}")
        await asyncio.sleep(0) # 비동기 컨텍스트 전환 흉내
        return None # 실제 반환값이 없으므로 None 반환

    mock_service.broadcast_to_task = AsyncMock()
    # subscribe, unsubscribe도 필요하면 유사하게 모킹 가능
    async def mock_subscribe(*args, **kwargs):
         await asyncio.sleep(0)
         return None
    async def mock_unsubscribe(*args, **kwargs):
         await asyncio.sleep(0)
         return None
    mock_service.subscribe = mock_subscribe
    mock_service.unsubscribe = mock_unsubscribe

    return mock_service

@pytest.fixture(autouse=True)
def ensure_dummy_graph_file():
    """
    Orchestrator가 디폴트로 보는 `config/agent_graphs` 경로에
    스키마에 맞는 더미 그래프를 심어 준다.
    """
    # 1) 프로젝트 루트 기준으로 config/agent_graphs 디렉토리를 만든다
    base_dir = pathlib.Path(__file__).resolve().parents[1]  # tests/.. → 프로젝트 루트
    graph_dir = base_dir / "config" / "agent_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # 2) 스키마에 맞춰 더미 그래프 JSON 생성
    dummy = {
        "name": "test_workflow_for_ws",
        "description": "dummy graph for WS test",
        "entry_point": "echo",           # 실제 노드 ID
        "nodes": [
            {
                "id": "echo",
                "node_type": "generic_llm_node",
                "parameters": {
                    "prompt_template_path": "dummy_prompt.txt"
                }
            }
        ],
        "edges": [
            {
                "source": "echo",
                "target": "__end__",       # 종료 토큰
                "type": "standard"
            }
        ]
    }
    (graph_dir / "test_workflow_for_ws.json").write_text(json.dumps(dummy, indent=2))

    prompt_dir = base_dir / "config" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    # 내용은 {{ original_input }} 만 있어도 충분합니다.
    (prompt_dir / "dummy_prompt.txt").write_text("{{ original_input }}")




@pytest.fixture(scope="session") # 세션 스코프로 이벤트 루프 설정 (선택 사항)
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

class _AsyncWS:
    """
    WebSocketTestSession 어댑터
    - sync `with` , async `async with` 모두 지원
    - send/receive 는 await 로 호출
    """
    def __init__(self, cm):
        self._cm = cm          # WebSocketTestSession(context manager)
        self._ws = None        # 실제 session

    # ---------- 동기 CM ----------
    def __enter__(self):
        self._ws = self._cm.__enter__()
        return self
    def __exit__(self, exc_t, exc, tb):
        return self._cm.__exit__(exc_t, exc, tb)

    # ---------- 비동기 CM ----------
    async def __aenter__(self):
        self._ws = await asyncio.to_thread(self._cm.__enter__)
        return self
    async def __aexit__(self, exc_t, exc, tb):
        return await asyncio.to_thread(self._cm.__exit__, exc_t, exc, tb)

    # ---------- proxy helpers ----------
    async def receive_json(self, *a, **kw):
        return await asyncio.to_thread(self._ws.receive_json, *a, **kw)
    async def send_json(self, *a, **kw):
        return await asyncio.to_thread(self._ws.send_json, *a, **kw)

class AsyncTestClient:        # 최소한의 메서드만 감쌈
    def __init__(self, app):
        self._client = TestClient(app)
    # --------- lifespan (async with) ------------
    async def __aenter__(self):
        # TestClient를 실제로 open 하여 exit_stack 생성
        await asyncio.to_thread(self._client.__enter__)
        return self
    async def __aexit__(self, exc_t, exc, tb):
        await asyncio.to_thread(self._client.__exit__, exc_t, exc, tb)
    # HTTP 메서드 --------------------------------------------------------
    async def get(self, *a, **kw):  return await asyncio.to_thread(self._client.get, *a, **kw)
    async def post(self, *a, **kw): return await asyncio.to_thread(self._client.post, *a, **kw)
    async def delete(self,*a, **kw): return await asyncio.to_thread(self._client.delete,*a, **kw)
    # WebSocket ----------------------------------------------------------
    def websocket_connect(self, *a, **kw):
        cm = self._client.websocket_connect(*a, **kw)  # WebSocketTestSession CM
        return _AsyncWS(cm)

@pytest.fixture
async def async_test_client():
    """
    비동기 테스트 전용 TestClient 어댑터.
    - HTTP 메서드/WS 모두 await 가능
    - 내부적으로는 TestClient를 스레드에서 호출하므로 이벤트 루프가 막히지 않음
    """
    async with AsyncTestClient(app) as client:
        yield client


@pytest.fixture
def mock_llm_client():
    return MagicMock(spec=LLMClient)

@pytest.fixture
def mock_tool_manager():
    # 실제 ToolManager 생성 또는 모킹 방식에 따라 수정
    mock = MagicMock(spec=ToolManager)
    mock.name = "MockedToolManager" # Orchestrator 초기화 시 사용될 수 있음
    return mock

@pytest.fixture
def mock_memory_manager():
    return MagicMock(spec=MemoryManager)

@pytest.fixture
def mock_notification_service():
    return MagicMock(spec=NotificationService)


        
@pytest.fixture
def orchestrator_instance(mock_llm_client, mock_tool_manager, mock_memory_manager, mock_notification_service):
    # Orchestrator 생성자에 필요한 모든 의존성을 전달해야 합니다.
    # telemetry.py의 setup_telemetry는 테스트 시작 시 이미 호출되었다고 가정합니다.
    try:
        instance = Orchestrator(
            llm_client=mock_llm_client,
            tool_manager=mock_tool_manager,
            memory_manager=mock_memory_manager,
            notification_service=mock_notification_service
        )
        return instance
    except Exception as e:
        pytest.fail(f"Failed to create orchestrator_instance: {e}")
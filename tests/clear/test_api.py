# tests/test_api.py
import json
import pytest
import asyncio
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, AsyncMock, MagicMock, ANY # 비동기 및 일반 모킹 도구

from fastapi import status, WebSocketDisconnect # WebSocketDisconnect 추가
from fastapi.testclient import TestClient

from src.api.app import app # FastAPI 앱 인스턴스
from src.schemas.request_models import RunWorkflowRequest
from src.schemas.response_models import TaskSubmittedResponse, WorkflowStatusResponse, GraphInfo, ToolInfo
from src.schemas.websocket_models import WebSocketMessageBase # WebSocket 메시지 모델
from src.schemas.mcp_models import AgentGraphState
from src.memory.memory_manager import MemoryManager # 타입 힌팅 및 모킹용
from src.agents.orchestrator import Orchestrator # 타입 힌팅 및 모킹용


# --- 테스트 픽스처 ---
# TestClient 인스턴스를 생성하는 pytest 픽스처
@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient 인스턴스를 생성합니다."""
    # 애플리케이션 컨텍스트 내에서 TestClient 생성
    # lifespan 이벤트를 테스트에서 실행하려면 추가 설정이 필요할 수 있으나,
    # 여기서는 기본 TestClient를 사용합니다.
    return TestClient(app)

@pytest.fixture
def mock_memory_manager():
    return MagicMock(spec=MemoryManager)

@pytest.fixture
def mock_orchestrator():
    return MagicMock(spec=Orchestrator)

# --- 테스트 케이스 ---

# 1. /health 엔드포인트 테스트
def test_health_check(client: TestClient):
    """/health 엔드포인트가 200 OK 와 'ok' 상태를 반환하는지 테스트합니다."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# 2. /run 엔드포인트 테스트
@pytest.mark.asyncio # 비동기 함수 실행 필요 (BackgroundTasks 관련)
@patch("src.api.routers.run_workflow_background", new_callable=AsyncMock) # 백그라운드 함수 모킹
async def test_run_workflow_success(mock_run_background: AsyncMock, client: TestClient):
    """POST /run 엔드포인트가 성공적으로 작업을 제출하고 202 Accepted를 반환하는지 테스트합니다."""
    request_data = RunWorkflowRequest(
        graph_config_name="test_workflow",
        original_input={"query": "What is the weather?"},
        initial_metadata={"user_id": "user123"}
    )

    response = client.post("/api/v1/run", json=request_data.model_dump()) # Pydantic v2

    # 응답 상태 코드 및 내용 검증
    assert response.status_code == status.HTTP_202_ACCEPTED
    response_json = response.json()
    assert "task_id" in response_json
    assert response_json["status"] == "accepted"
    task_id = response_json["task_id"]

    # 백그라운드 함수가 올바른 인자들로 호출되었는지 검증
    # 비동기 호출 대기 (실제로는 모킹되어 바로 완료됨)
    await asyncio.sleep(0.01) # 호출이 스케줄링될 시간을 약간 줌
    mock_run_background.assert_called_once()
    # 호출 인자 검증 (ANY는 orchestrator, memory_manager 객체를 의미)
    call_args = mock_run_background.call_args.args
    assert call_args[1] == "test_workflow" # graph_config_name
    assert call_args[2] == task_id         # task_id
    assert call_args[3] == request_data.original_input # original_input
    assert call_args[4] == request_data.initial_metadata # initial_metadata
    # orchestrator(call_args[0])와 memory_manager(call_args[5]) 타입/존재 여부 확인 가능

def test_run_workflow_invalid_request(client: TestClient):
    """POST /run 엔드포인트가 잘못된 요청 본문에 대해 422 Unprocessable Entity를 반환하는지 테스트합니다."""
    invalid_payload = {
        # "graph_config_name": "missing field", # 필수 필드 누락
        "original_input": "some input"
    }
    response = client.post("/api/v1/run", json=invalid_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# 3. /status/{task_id} 엔드포인트 테스트
# MemoryManager 의존성을 모킹하기 위한 설정
# (실제로는 conftest.py 에서 fixture 로 관리하는 것이 더 좋음)
@pytest.mark.asyncio
async def test_get_status_completed(client: TestClient):
    """GET /status/{task_id} 가 완료된 작업 상태를 올바르게 반환하는지 테스트합니다."""
    task_id = "task-completed-123"
    state_key = "workflow_final_state"
    mock_state_data = {
        "task_id": task_id,
        "original_input": {"query": "done"},
        "final_answer": "The workflow is successfully completed.",
        "error_message": None,
        "current_iteration": 5,
        "metadata": {"user": "test"},
        # ... AgentGraphState의 다른 필드들 ...
    }

    # MemoryManager.load_state 모킹 설정
    # `app.dependency_overrides`를 사용하여 특정 의존성을 테스트 중에 교체
    # AsyncMock 사용 주의: 반환 값도 await 가능해야 할 수 있음 (load_state가 async이므로)
    mock_memory_manager = MagicMock()
    mock_memory_manager.load_state = AsyncMock(return_value=mock_state_data)

    # 의존성 오버라이드 (테스트 함수 내에서만 적용)
    from src.api.dependencies import get_memory_manager_dependency
    app.dependency_overrides[get_memory_manager_dependency] = lambda: mock_memory_manager

    response = client.get(f"/api/v1/status/{task_id}")

    # 검증
    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["task_id"] == task_id
    assert response_json["status"] == "completed" # error 없고 final_answer 있음
    assert response_json["final_answer"] == mock_state_data["final_answer"]
    assert response_json["error_message"] is None
    assert response_json["current_iteration"] == mock_state_data["current_iteration"]

    # 테스트 후 오버라이드 제거 (중요)
    app.dependency_overrides = {}
    mock_memory_manager.load_state.assert_called_once_with(context_id=task_id, key=state_key)

@pytest.mark.asyncio
async def test_get_status_failed(client: TestClient):
    """GET /status/{task_id} 가 실패한 작업 상태를 올바르게 반환하는지 테스트합니다."""
    task_id = "task-failed-456"
    state_key = "workflow_final_state"
    mock_state_data = {
        "task_id": task_id,
        "original_input": {"query": "fail"},
        "final_answer": None,
        "error_message": "LLM call failed after 3 retries.",
        "current_iteration": 3,
        "metadata": {"user": "test"},
    }
    mock_memory_manager = MagicMock()
    mock_memory_manager.load_state = AsyncMock(return_value=mock_state_data)
    from src.api.dependencies import get_memory_manager_dependency
    app.dependency_overrides[get_memory_manager_dependency] = lambda: mock_memory_manager

    response = client.get(f"/api/v1/status/{task_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["task_id"] == task_id
    assert response_json["status"] == "failed" # error_message 있음
    assert response_json["final_answer"] is None
    assert response_json["error_message"] == mock_state_data["error_message"]

    app.dependency_overrides = {}
    mock_memory_manager.load_state.assert_called_once_with(context_id=task_id, key=state_key)


@pytest.mark.asyncio
async def test_get_status_running(client: TestClient):
    """GET /status/{task_id} 가 실행 중인 작업 상태를 올바르게 반환하는지 테스트합니다."""
    task_id = "task-running-789"
    state_key = "workflow_final_state"
    mock_state_data = {
        "task_id": task_id,
        "original_input": {"query": "running"},
        "final_answer": None, # 아직 최종 결과 없음
        "error_message": None, # 오류도 없음
        "current_iteration": 1,
        "metadata": {"user": "test"},
    }
    mock_memory_manager = MagicMock()
    mock_memory_manager.load_state = AsyncMock(return_value=mock_state_data)
    from src.api.dependencies import get_memory_manager_dependency
    app.dependency_overrides[get_memory_manager_dependency] = lambda: mock_memory_manager

    response = client.get(f"/api/v1/status/{task_id}")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert response_json["task_id"] == task_id
    assert response_json["status"] == "running" # error 없고 final_answer 없음
    assert response_json["final_answer"] is None
    assert response_json["error_message"] is None

    app.dependency_overrides = {}
    mock_memory_manager.load_state.assert_called_once_with(context_id=task_id, key=state_key)

@pytest.mark.asyncio
async def test_get_status_not_found(client: TestClient):
    """GET /status/{task_id} 가 존재하지 않는 작업 ID에 대해 404를 반환하는지 테스트합니다."""
    task_id = "task-not-found-000"
    state_key = "workflow_final_state"

    mock_memory_manager = MagicMock()
    mock_memory_manager.load_state = AsyncMock(return_value=None) # 상태 없음
    from src.api.dependencies import get_memory_manager_dependency
    app.dependency_overrides[get_memory_manager_dependency] = lambda: mock_memory_manager

    response = client.get(f"/api/v1/status/{task_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND

    app.dependency_overrides = {}
    mock_memory_manager.load_state.assert_called_once_with(context_id=task_id, key=state_key)

# 4. /graphs 엔드포인트 테스트
@patch("src.api.routers.Path")
def test_list_graphs_success(mock_path: MagicMock, client: TestClient):
    """
    GET /graphs should return available graph configs with correct descriptions.
    """
    # 1) Path 객체 및 glob, is_dir 설정
    mock_graph_dir = MagicMock()
    mock_graph_dir.is_dir.return_value = True

    mock_file1 = MagicMock(spec=Path)
    mock_file1.stem = "graph1"
    mock_file1.name = "graph1.json"
    mock_file2 = MagicMock(spec=Path)
    mock_file2.stem = "graph2_with_desc"
    mock_file2.name = "graph2_with_desc.json"
    mock_graph_dir.glob.return_value = [mock_file1, mock_file2]
    mock_path.return_value = mock_graph_dir

    # 2) builtins.open을 io.StringIO로 대체하여 실제 JSON 읽기 흉내
    def open_side_effect(path, mode="r", encoding="utf-8"):
        if path.name == "graph1.json":
            return io.StringIO('{"name":"graph1"}')
        elif path.name == "graph2_with_desc.json":
            return io.StringIO(
                '{"name":"graph2","description":"Second graph description."}'
            )
        raise FileNotFoundError(f"No such file: {path}")

    with patch("builtins.open", side_effect=open_side_effect):
        response = client.get("/api/v1/graphs")

    # 3) 응답 검증
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    # description 필드가 없으면 기본값 반환
    assert data[0] == {"name": "graph1", "description": "Workflow configuration 'graph1'"}
    # description 필드가 있으면 JSON 내 값 그대로 반환
    assert data[1] == {"name": "graph2_with_desc", "description": "Second graph description."}


@patch("src.api.routers.Path")
def test_list_graphs_dir_not_found(mock_path: MagicMock, client: TestClient):
    """GET /graphs 가 디렉토리가 없을 때 빈 리스트를 반환하는지 테스트합니다."""
    mock_graph_dir = MagicMock()
    mock_graph_dir.is_dir.return_value = False # 디렉토리 없음
    mock_path.return_value = mock_graph_dir

    response = client.get("/api/v1/graphs")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []

# 5. /tools 엔드포인트 테스트
@pytest.mark.asyncio
async def test_list_tools_success(client: TestClient):
    """GET /tools 가 도구 목록을 성공적으로 반환하는지 테스트합니다."""
    mock_tool_list = [
        {"name": "calculator", "description": "Math tool", "args_schema_summary": {"expression": "string (required)"}},
        {"name": "web_search", "description": "Search tool", "args_schema_summary": {"query": "string (required)"}},
    ]

    # ToolManager.list_tools 모킹
    mock_tool_manager = MagicMock()
    # list_tools는 동기 함수일 수 있음 (내부 구현 확인 필요)
    mock_tool_manager.list_tools = MagicMock(return_value=mock_tool_list)

    # 의존성 오버라이드
    from src.api.dependencies import get_tool_manager_dependency
    app.dependency_overrides[get_tool_manager_dependency] = lambda: mock_tool_manager

    response = client.get("/api/v1/tools")

    assert response.status_code == status.HTTP_200_OK
    response_json = response.json()
    assert isinstance(response_json, list)
    assert len(response_json) == 2
    # ToolInfo 스키마와 일치하는지 확인
    assert response_json[0]["name"] == "calculator"
    assert response_json[0]["description"] == "Math tool"
    assert response_json[0]["args_schema_summary"] == {"expression": "string (required)"}
    assert response_json[1]["name"] == "web_search"

    # 오버라이드 제거
    app.dependency_overrides = {}
    mock_tool_manager.list_tools.assert_called_once()

@pytest.mark.asyncio
async def test_list_tools_manager_error(client: TestClient):
    """GET /tools 가 ToolManager 오류 시 500을 반환하는지 테스트합니다."""
    mock_tool_manager = MagicMock()
    mock_tool_manager.list_tools = MagicMock(side_effect=RuntimeError("Failed to list tools"))

    from src.api.dependencies import get_tool_manager_dependency
    app.dependency_overrides[get_tool_manager_dependency] = lambda: mock_tool_manager

    response = client.get("/api/v1/tools")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to retrieve the list of available tools" in response.json()["detail"]

    app.dependency_overrides = {}
    
# --- WebSocket 테스트 함수들 ---

@pytest.mark.asyncio
async def test_websocket_connection_success(async_test_client: TestClient):
    """WebSocket 엔드포인트에 성공적으로 연결하고 해제할 수 있는지 테스트합니다."""
    task_id = "ws_connect_test_task"
    try:
        with async_test_client.websocket_connect(f"/api/v1/ws/status/{task_id}") as websocket:
            assert websocket is not None
            await asyncio.sleep(0.1) # 연결이 완전히 수립될 시간 확보
    except Exception as e:
        pytest.fail(f"WebSocket connection failed unexpectedly: {e}")

@pytest.mark.asyncio
async def test_websocket_receive_status_updates(
    async_test_client: TestClient,
    mock_memory_manager: MagicMock,
):
    """
    WebSocket이 워크플로 상태/결과를 정상 수신하는지 검증한다.
    - 서버가 생성한 task_id를 받아서 즉시 같은 ID로 WebSocket에 구독한다.
    """
    # 1) 워크플로 실행 – 서버가 task_id 생성
    run_req = RunWorkflowRequest(
        graph_config_name="test_workflow_for_ws",
        original_input="Test input for WS",
    )
    run_resp = await async_test_client.post(
        "/api/v1/run",
        json=run_req.model_dump(),
    )
    assert run_resp.status_code == status.HTTP_202_ACCEPTED
    task_id = run_resp.json()["task_id"]
    assert task_id  # 비어 있지 않아야 함

    # 2) WebSocket 연결
    async with async_test_client.websocket_connect(f"/api/v1/ws/status/{task_id}") as ws:
        # 3) 최소 한 개 이상의 상태 메시지를 10초 안에 받아야 한다.
        msg = await asyncio.wait_for(ws.receive_json(), timeout=10)
        assert msg["event_type"] in {"status_update", "final_result"}

# --- (선택 사항) 추가 테스트 케이스 ---

# @pytest.mark.asyncio
# async def test_websocket_invalid_task_id(client: TestClient):
#     """존재하지 않는 task_id로 연결 시도 시 동작 테스트"""
#     task_id = "invalid-task-id"
#     try:
#         # 연결은 성공할 수 있으나, 아무 메시지도 받지 못해야 함
#         with client.websocket_connect(f"/api/v1/ws/status/{task_id}") as websocket:
#             try:
#                 # 짧은 시간 동안 메시지 수신 시도
#                 message = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
#                 pytest.fail(f"Received unexpected message for invalid task_id: {message}")
#             except asyncio.TimeoutError:
#                 # 메시지를 받지 못하는 것이 정상
#                 pass
#     except Exception as e:
#         # 연결 자체 실패 등 예상치 못한 오류 검사
#         pytest.fail(f"WebSocket test for invalid task_id failed: {e}")
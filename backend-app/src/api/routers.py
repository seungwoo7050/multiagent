# src/api/routers.py
import asyncio
import json
import msgspec # msgspec 임포트 추가 (상태 변환용)
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Path,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from opentelemetry import trace
from src.utils.telemetry import get_tracer # <--- 추가: OpenTelemetry Tracer 가져오기
from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.utils.ids import generate_task_id
# 요청/응답 스키마 임포트
from src.schemas.request_models import RunWorkflowRequest
from src.schemas.response_models import (
    TaskSubmittedResponse,
    WorkflowStatusResponse,
    GraphInfo,
    ToolInfo
)
# 의존성 주입용 타입 어노테이션 임포트
from src.api.dependencies import (
    NewOrchestratorDep,
    MemoryManagerDep,
    ToolManagerDep,
    NotificationServiceDep,
)
# 핵심 클래스 임포트
from src.agents.orchestrator import Orchestrator as NewOrchestrator
from src.memory.memory_manager import MemoryManager
from src.services.tool_manager import ToolManager
from src.schemas.mcp_models import AgentGraphState
from src.config.errors import ErrorCode, MemoryError

logger = get_logger(__name__)
settings = get_settings()
tracer = get_tracer(__name__)

# API 라우터 인스턴스 생성
router = APIRouter()

# --- 백그라운드 워크플로우 실행 및 상태 저장 함수 ---

async def run_workflow_background(
    orchestrator: NewOrchestrator,
    graph_config_name: str,
    task_id: str,
    original_input: Any,
    initial_metadata: Dict[str, Any],
    memory_manager: MemoryManager
):
    """
    Orchestrator.run_workflow를 백그라운드에서 실행하고 최종 AgentGraphState를
    MemoryManager에 저장합니다.
    """
    logger.info(f"[BackgroundTask] Started for task_id: {task_id}, graph: {graph_config_name}")
    final_state: Optional[AgentGraphState] = None
    start_time = asyncio.get_event_loop().time()

    # 상태 저장을 위한 키 정의 (MemoryManager의 context_id와 key 사용)
    state_key = "workflow_final_state" # 이 키 아래에 최종 상태 저장

    try:
        # Orchestrator의 워크플로우 실행 메서드 호출
        final_state = await orchestrator.run_workflow(
            graph_config_name=graph_config_name,
            task_id=task_id,
            original_input=original_input,
            initial_metadata=initial_metadata,
            # max_iterations는 orchestrator.run_workflow의 기본값 또는 설정에 따름
        )
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        # 최종 상태 로깅 (성공 시)
        logger.info(
            f"[BackgroundTask] Workflow completed for task_id: {task_id} in {duration:.2f}s. "
            f"Final Answer: {final_state.final_answer if final_state else 'N/A'}, "
            f"Error: {final_state.error_message if final_state else 'N/A'}"
        )

    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        # 워크플로우 실행 중 예외 발생 시 로깅 및 실패 상태 생성
        logger.error(
            f"[BackgroundTask] Exception during workflow execution for task_id {task_id} after {duration:.2f}s: {e}",
            exc_info=True
        )
        # 실패 상태를 나타내는 AgentGraphState 생성 (실패 정보 포함)
        final_state = AgentGraphState(
            task_id=task_id,
            original_input=original_input,
            metadata=initial_metadata,
            # 상태 결정은 error_message 유무로 가능
            error_message=f"Workflow execution failed unexpectedly in background: {str(e)}"
            # 다른 필드는 기본값 사용
        )
    finally:
        # 최종 상태 (성공 또는 실패)를 MemoryManager에 저장
        if final_state:
            try:
                # AgentGraphState 객체를 저장 가능한 형태로 변환 (msgspec 권장)
                # msgspec.to_builtins는 기본 타입으로 변환, 또는 직접 dict 변환 시도
                encoder = msgspec.msgpack.Encoder()
                decoder = msgspec.msgpack.Decoder(Dict[str, Any]) # dict 형태로 저장
                state_data_to_save = decoder.decode(encoder.encode(final_state))

                await memory_manager.save_state(
                    context_id=task_id, # task_id를 컨텍스트로 사용
                    key=state_key,      # 정의된 키 사용
                    value=state_data_to_save,
                    ttl=settings.TASK_STATUS_TTL # 설정된 TTL 적용
                )
                logger.info(f"[BackgroundTask] Final state saved for task_id: {task_id} using key '{state_key}'.")
            except MemoryError as save_mem_err:
                 logger.error(f"[BackgroundTask] MemoryError saving final state for task_id {task_id}: {save_mem_err.message}", exc_info=False) # 스택 트레이스 제외 가능
            except Exception as save_err:
                logger.error(f"[BackgroundTask] Failed to save final state for task_id {task_id}: {save_err}", exc_info=True)
        else:
             # final_state가 None인 경우는 거의 없지만 로깅
             logger.error(f"[BackgroundTask] Final state was None for task_id {task_id}, cannot save status.")

# --- API 엔드포인트 정의 ---
@router.post(
    "/run",
    response_model=TaskSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="워크플로우 비동기 실행 요청",
    description="지정된 그래프 설정을 사용하여 워크플로우 실행을 요청합니다. 워크플로우는 백그라운드에서 실행되며, 반환된 task_id로 상태를 조회할 수 있습니다.",
    tags=["Workflow Execution"]
)
async def run_workflow_endpoint(
    request: RunWorkflowRequest,
    background_tasks: BackgroundTasks,
    orchestrator: NewOrchestratorDep,
    memory_manager: MemoryManagerDep
):
    """
    워크플로우 실행을 요청하고 백그라운드 처리를 시작합니다.

    - **graph_config_name**: 실행할 그래프 설정 이름 (예: "default_tot_workflow").
    - **original_input**: 워크플로우 초기 입력 데이터.
    - **task_id** (선택): 사용할 작업 ID. 없으면 자동 생성.
    - **initial_metadata** (선택): 초기 워크플로우 상태 메타데이터.
    """
    task_id = request.task_id or generate_task_id("task_division")
    logger.info(f"API '/run': Received request with input: {request.original_input[:50]}...")

    # 백그라운드 태스크 추가 시 예외 처리
    try:
        # Always use task_division_workflow regardless of what was requested
        background_tasks.add_task(
            run_workflow_background,
            orchestrator,
            "task_division_workflow",  # Fixed to always use task division workflow
            task_id,
            request.original_input,
            request.initial_metadata or {},
            memory_manager
        )
        logger.info(f"API '/run': Task {task_id} submitted using task_division_workflow")
        return TaskSubmittedResponse(task_id=task_id, status="accepted")
    except Exception as e:
        logger.error(f"API '/run': Failed to submit background task: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule workflow execution due to an internal error."
        )

@router.get(
    "/status/{task_id}",
    response_model=WorkflowStatusResponse,
    summary="워크플로우 상태 및 결과 조회",
    description="이전에 제출된 워크플로우의 현재 상태와 완료 시 결과 또는 오류를 조회합니다.",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "해당 Task ID의 상태 정보를 찾을 수 없습니다 (진행 중, 만료 또는 존재하지 않음)."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "상태 조회 중 서버 내부 오류 발생."}
    },
    tags=["Workflow Execution"]
)
async def get_workflow_status(
    memory_manager: MemoryManagerDep, # MemoryManager 의존성 주입
    task_id: str = Path(..., description="상태를 조회할 작업의 ID", examples=["task-abc-123"]),
):
    """
    `task_id`를 사용하여 백그라운드에서 실행된 워크플로우의 최종 상태를 조회합니다.
    상태는 MemoryManager에 저장된 `AgentGraphState` 정보를 기반으로 합니다.
    """
    logger.info(f"API '/status': Request received for status of task_id: {task_id}")
    state_key = "workflow_final_state" # 상태 저장 시 사용한 키

    try:
        # MemoryManager를 사용하여 task_id 컨텍스트에서 state_key로 상태 조회
        stored_state_data = await memory_manager.load_state(context_id=task_id, key=state_key)

        if stored_state_data is None:
            # 상태 정보가 없는 경우 (실행 전, 실행 중, TTL 만료 등)
            logger.warning(f"API '/status': Status data not found in memory for task_id: {task_id} using key '{state_key}'.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task status not found. The workflow might be pending, still running, or the status record has expired."
            )

        # 저장된 데이터가 dict 형태인지 확인 (JSON/msgpack 역직렬화 결과)
        if not isinstance(stored_state_data, dict):
             logger.error(f"API '/status': Stored state for task {task_id} is not a dictionary (type: {type(stored_state_data)}). Data: {str(stored_state_data)[:200]}...")
             raise HTTPException(
                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                 detail="Internal error: Invalid format for stored task status."
             )

        # 저장된 dict 데이터로부터 응답 모델 생성
        final_answer = stored_state_data.get("final_answer")
        error_message = stored_state_data.get("error_message")

        # 상태 결정 로직: 에러 메시지가 있으면 'failed', 최종 답변이 있으면 'completed', 아니면 'running'
        current_status: str
        if error_message:
            current_status = "failed"
        elif final_answer is not None: # None과 '' 구분
            current_status = "completed"
        else:
            # 에러도 없고 최종 답변도 없으면 아직 실행 중 또는 상태 저장 전으로 간주
            current_status = "running" # 또는 'pending' 가능성

        # WorkflowStatusResponse 모델에 필요한 필드들을 채워서 반환
        response_data = {
            "task_id": task_id, # stored_state_data.get("task_id", task_id), # ID 일관성 확인
            "status": current_status,
            "final_answer": final_answer,
            "error_message": error_message,
            # AgentGraphState의 다른 유용한 필드들도 포함 가능
            "current_iteration": stored_state_data.get("current_iteration"),
            "search_depth": stored_state_data.get("search_depth"),
            "last_llm_output": stored_state_data.get("last_llm_output"), # 디버깅용
            "metadata": stored_state_data.get("metadata")
        }

        # Pydantic 모델로 응답 객체 생성 (데이터 유효성 검사 포함)
        return WorkflowStatusResponse(**response_data)

    except HTTPException as http_exc:
        # 404 등 의도된 HTTP 예외는 그대로 전달
        raise http_exc
    except MemoryError as mem_err:
         # MemoryManager 관련 오류 발생 시
         logger.error(f"API '/status': MemoryError retrieving status for task_id {task_id}: {mem_err.message}", exc_info=False)
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE, # 서비스 의존성 문제로 간주
             detail=f"Service dependency error retrieving task status: {mem_err.code}"
         )
    except Exception as e:
        # 그 외 예상치 못한 오류 발생 시
        logger.error(f"API '/status': Unexpected error retrieving status for task_id {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving task status."
        )

# --- 선택적 엔드포인트 (그래프 목록 / 도구 목록) ---

@router.get(
    "/graphs",
    response_model=List[GraphInfo],
    summary="사용 가능한 그래프 설정 목록 조회",
    description="설정된 디렉토리에서 사용 가능한 에이전트 그래프 설정 파일(.json) 목록을 반환합니다.",
    tags=["Configuration"]
)
async def list_available_graphs():
    """
    설정된 `AGENT_GRAPH_CONFIG_DIR` 에서 `.json` 파일을 찾아 목록을 반환합니다.
    """
    logger.info("API '/graphs': Request received to list available graphs")
    graph_dir_str = getattr(settings, 'AGENT_GRAPH_CONFIG_DIR', 'config/agent_graphs')
    graph_dir = Path(graph_dir_str)
    available_graphs: List[GraphInfo] = []

    if not graph_dir.is_dir():
        logger.warning(f"API '/graphs': Agent graph directory not found or not a directory: {graph_dir}")
        return available_graphs # 디렉토리 없으면 빈 리스트 반환

    try:
        for file_path in graph_dir.glob("*.json"):
            graph_name = file_path.stem
            description = f"Workflow configuration '{graph_name}'" # 기본 설명
            try:
                # 파일 내용을 읽어 description 필드 추출 시도 (성능 영향 적음)
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 큰 파일을 대비해 전체 로드 대신 일부만 읽거나 스트리밍 고려 가능
                    # 여기서는 간단히 전체 로드
                    data = json.load(f)
                    file_description = data.get("description")
                    if file_description and isinstance(file_description, str):
                         description = file_description
            except json.JSONDecodeError as e:
                logger.warning(f"API '/graphs': Could not parse JSON from {file_path.name}: {e}")
                # description remains the default
            except Exception as e:
                logger.warning(f"API '/graphs': Error reading or parsing {file_path.name}: {e}", exc_info=False)
                # description remains the default

            available_graphs.append(GraphInfo(name=graph_name, description=description))

        logger.info(f"API '/graphs': Found {len(available_graphs)} potential graph configurations in {graph_dir}.")
        return available_graphs
    except Exception as e:
        # 디렉토리 스캔 중 예외 발생 시
        logger.error(f"API '/graphs': Error listing graph configurations from {graph_dir}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list available workflow configurations."
        )

@router.get(
    "/tools",
    response_model=List[ToolInfo],
    summary="사용 가능한 도구 목록 조회",
    description="시스템에 등록되어 에이전트가 사용할 수 있는 도구 목록과 설명을 반환합니다.",
    tags=["Configuration"]
)
async def list_available_tools(
    tool_manager: ToolManagerDep # ToolManager 의존성 주입
):
    """
    `ToolManager`에 등록된 모든 도구의 이름, 설명, 인자 요약 정보를 반환합니다.
    """
    logger.info("API '/tools': Request received to list available tools")
    try:
        # ToolManager에서 도구 메타데이터 목록 가져오기
        tool_list_details = tool_manager.list_tools() # 이름, 설명, 인자 요약 포함

        # ToolInfo 모델 리스트로 변환
        response_list = [
            ToolInfo(
                name=tool.get('name', 'unknown_tool'),
                description=tool.get('description', 'No description provided.'),
                args_schema_summary=tool.get('args_schema_summary')
            ) for tool in tool_list_details
        ]
        logger.info(f"API '/tools': Returning {len(response_list)} available tools.")
        return response_list
    except Exception as e:
        # ToolManager 접근 중 오류 발생 시
        logger.error(f"API '/tools': Error retrieving tool list from ToolManager: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve the list of available tools."
        )
        
# --- WebSocket 엔드포인트 정의 ---
@router.websocket("/ws/status/{task_id}")
async def websocket_status_endpoint(
    websocket: WebSocket,
    notification_service: NotificationServiceDep,
    task_id: str = Path(..., description="상태 업데이트를 수신할 작업의 ID"),
):
    """
    특정 task_id에 대한 실시간 상태 업데이트를 위한 WebSocket 엔드포인트입니다.
    클라이언트는 이 엔드포인트에 연결하여 작업 진행 상황을 실시간으로 받을 수 있습니다.
    """
    # tracer 블록 시작
    with tracer.start_as_current_span("WebSocket Connection") as conn_span:
        conn_span.set_attribute("websocket.task_id", task_id)
        conn_span.set_attribute("net.protocol.name", "websocket")
        client_host = websocket.client.host if websocket.client else "unknown"
        client_port = websocket.client.port if websocket.client else "unknown"
        conn_span.set_attribute("net.peer.ip", client_host)
        conn_span.set_attribute("net.peer.port", client_port)

        logger.info(f"WebSocket: [/ws/status/{task_id}] - Received connection request from {client_host}:{client_port}") # 로그 (1)

        try:
            logger.info(f"WebSocket: [/ws/status/{task_id}] - Attempting to accept connection...") # 로그 (2)
            await websocket.accept()
            logger.info(f"WebSocket: [/ws/status/{task_id}] - Connection accepted for {client_host}:{client_port}") # 로그 (3)
            conn_span.add_event("WebSocket accepted")

            logger.info(f"WebSocket: [/ws/status/{task_id}] - Attempting to subscribe to NotificationService...") # 로그 (4)
            await notification_service.subscribe(task_id, websocket)
            # 여기가 문제의 390번째 줄 근처일 수 있습니다. 윗줄과 들여쓰기 레벨이 동일해야 합니다.
            logger.info(f"WebSocket: [/ws/status/{task_id}] - Successfully subscribed to NotificationService.") # 로그 (5)
            conn_span.add_event("Subscribed to notifications")

            # ---- 연결 유지 루프 시작 (이 try 블록은 accept 및 subscribe 성공 후 실행됨) ----
            try:
                while True:
                    # logger.debug(f"WebSocket: [/ws/status/{task_id}] - Maintaining connection loop...") # 로그 (6)
                    await asyncio.sleep(settings.WEBSOCKET_KEEP_ALIVE_INTERVAL if hasattr(settings, 'WEBSOCKET_KEEP_ALIVE_INTERVAL') else 60)

            except WebSocketDisconnect:
                logger.info(f"WebSocket: [/ws/status/{task_id}] - Client {websocket.client} disconnected (WebSocketDisconnect).") # 로그 (7)
            except Exception as e_loop: # 루프 내 다른 예외
                logger.error(f"WebSocket: [/ws/status/{task_id}] - Unexpected error in connection loop: {e_loop}", exc_info=True) # 로그 (8)
            # ---- 연결 유지 루프 종료 ----

        except Exception as e_initial: # websocket.accept() 또는 notification_service.subscribe()에서 발생할 수 있는 예외
            logger.error(f"WebSocket: [/ws/status/{task_id}] - Error during initial accept or subscribe: {e_initial}", exc_info=True) # 로그 (11)
            # 이 경우, 연결이 accept 되었다면 close를 시도하는 것이 좋으나,
            # FastAPI가 연결 상태에 따라 적절히 처리할 가능성이 높습니다.
            # 만약 websocket.close()를 여기서 호출한다면, 이미 닫힌 소켓에 대한 호출이 될 수 있어 주의해야 합니다.
            # 일반적으로 accept 실패 시 클라이언트는 이미 연결이 끊어진 것으로 간주합니다.

        finally: # 이 finally는 가장 바깥쪽 with tracer 블록 내의 try에 대한 finally입니다.
            logger.info(f"WebSocket: [/ws/status/{task_id}] - Starting unsubscribe process in finally block...") # 로그 (9)
            await notification_service.unsubscribe(task_id, websocket)
            logger.info(f"WebSocket: [/ws/status/{task_id}] - Connection closed and unsubscribed for client: {websocket.client}") # 로그 (10)
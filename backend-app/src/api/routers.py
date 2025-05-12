import asyncio
import json
import msgspec
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Path,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from src.utils.telemetry import get_tracer
from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.utils.ids import generate_task_id

from src.schemas.request_models import RunWorkflowRequest
from src.schemas.response_models import (
    TaskSubmittedResponse,
    WorkflowStatusResponse,
    GraphInfo,
    ToolInfo,
)

from src.api.dependencies import (
    NewOrchestratorDep,
    MemoryManagerDep,
    ToolManagerDep,
    NotificationServiceDep,
)

from src.agents.orchestrator import Orchestrator as NewOrchestrator
from src.memory.memory_manager import MemoryManager
from src.schemas.mcp_models import AgentGraphState
from src.config.errors import MemoryError

logger = get_logger(__name__)
settings = get_settings()
tracer = get_tracer(__name__)


router = APIRouter()


async def run_workflow_background(
    orchestrator: NewOrchestrator,
    graph_config_name: str,
    task_id: str,
    original_input: Any,
    initial_metadata: Dict[str, Any],
    memory_manager: MemoryManager,
):
    """
    Orchestrator.run_workflow를 백그라운드에서 실행하고 최종 AgentGraphState를
    MemoryManager에 저장합니다.
    """
    logger.info(
        f"[BackgroundTask] Started for task_id: {task_id}, graph: {graph_config_name}"
    )
    final_state: Optional[AgentGraphState] = None
    start_time = asyncio.get_event_loop().time()

    state_key = "workflow_final_state"

    try:
        final_state = await orchestrator.run_workflow(
            graph_config_name=graph_config_name,
            task_id=task_id,
            original_input=original_input,
            initial_metadata=initial_metadata,
        )
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        logger.info(
            f"[BackgroundTask] Workflow completed for task_id: {task_id} in {duration:.2f}s. "
            f"Final Answer: {final_state.final_answer if final_state else 'N/A'}, "
            f"Error: {final_state.error_message if final_state else 'N/A'}"
        )

    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        logger.error(
            f"[BackgroundTask] Exception during workflow execution for task_id {task_id} after {duration:.2f}s: {e}",
            exc_info=True,
        )

        final_state = AgentGraphState(
            task_id=task_id,
            original_input=original_input,
            metadata=initial_metadata,
            error_message=f"Workflow execution failed unexpectedly in background: {str(e)}",
        )
    finally:
        if final_state:
            try:
                encoder = msgspec.msgpack.Encoder()
                decoder = msgspec.msgpack.Decoder(Dict[str, Any])
                state_data_to_save = decoder.decode(encoder.encode(final_state))

                await memory_manager.save_state(
                    context_id=task_id,
                    key=state_key,
                    value=state_data_to_save,
                    ttl=settings.TASK_STATUS_TTL,
                )
                logger.info(
                    f"[BackgroundTask] Final state saved for task_id: {task_id} using key '{state_key}'."
                )
            except MemoryError as save_mem_err:
                logger.error(
                    f"[BackgroundTask] MemoryError saving final state for task_id {task_id}: {save_mem_err.message}",
                    exc_info=False,
                )
            except Exception as save_err:
                logger.error(
                    f"[BackgroundTask] Failed to save final state for task_id {task_id}: {save_err}",
                    exc_info=True,
                )
        else:
            logger.error(
                f"[BackgroundTask] Final state was None for task_id {task_id}, cannot save status."
            )


@router.post(
    "/run",
    response_model=TaskSubmittedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="워크플로우 비동기 실행 요청",
    description="지정된 그래프 설정을 사용하여 워크플로우 실행을 요청합니다. 워크플로우는 백그라운드에서 실행되며, 반환된 task_id로 상태를 조회할 수 있습니다.",
    tags=["Workflow Execution"],
)
async def run_workflow_endpoint(
    request: RunWorkflowRequest,
    background_tasks: BackgroundTasks,
    orchestrator: NewOrchestratorDep,
    memory_manager: MemoryManagerDep,
):
    """
    워크플로우 실행을 요청하고 백그라운드 처리를 시작합니다.

    - **graph_config_name**: 실행할 그래프 설정 이름 (예: "default_tot_workflow").
    - **original_input**: 워크플로우 초기 입력 데이터.
    - **task_id** (선택): 사용할 작업 ID. 없으면 자동 생성.
    - **initial_metadata** (선택): 초기 워크플로우 상태 메타데이터.
    """
    task_id = request.task_id or generate_task_id("task_division")
    logger.info(
        f"API '/run': Received request with input: {request.original_input[:50]}..."
    )

    try:
        background_tasks.add_task(
            run_workflow_background,
            orchestrator,
            "task_division_workflow",
            task_id,
            request.original_input,
            request.initial_metadata or {},
            memory_manager,
        )
        logger.info(
            f"API '/run': Task {task_id} submitted using task_division_workflow"
        )
        return TaskSubmittedResponse(task_id=task_id, status="accepted")
    except Exception as e:
        logger.error(
            f"API '/run': Failed to submit background task: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule workflow execution due to an internal error.",
        )


@router.get(
    "/status/{task_id}",
    response_model=WorkflowStatusResponse,
    summary="워크플로우 상태 및 결과 조회",
    description="이전에 제출된 워크플로우의 현재 상태와 완료 시 결과 또는 오류를 조회합니다.",
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "해당 Task ID의 상태 정보를 찾을 수 없습니다 (진행 중, 만료 또는 존재하지 않음)."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "상태 조회 중 서버 내부 오류 발생."
        },
    },
    tags=["Workflow Execution"],
)
async def get_workflow_status(
    memory_manager: MemoryManagerDep,
    task_id: str = Path(
        ..., description="상태를 조회할 작업의 ID", examples=["task-abc-123"]
    ),
):
    """
    `task_id`를 사용하여 백그라운드에서 실행된 워크플로우의 최종 상태를 조회합니다.
    상태는 MemoryManager에 저장된 `AgentGraphState` 정보를 기반으로 합니다.
    """
    logger.info(f"API '/status': Request received for status of task_id: {task_id}")
    state_key = "workflow_final_state"

    try:
        stored_state_data = await memory_manager.load_state(
            context_id=task_id, key=state_key
        )

        if stored_state_data is None:
            logger.warning(
                f"API '/status': Status data not found in memory for task_id: {task_id} using key '{state_key}'."
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task status not found. The workflow might be pending, still running, or the status record has expired.",
            )

        if not isinstance(stored_state_data, dict):
            logger.error(
                f"API '/status': Stored state for task {task_id} is not a dictionary (type: {type(stored_state_data)}). Data: {str(stored_state_data)[:200]}..."
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: Invalid format for stored task status.",
            )

        final_answer = stored_state_data.get("final_answer")
        error_message = stored_state_data.get("error_message")

        current_status: str
        if error_message:
            current_status = "failed"
        elif final_answer is not None:
            current_status = "completed"
        else:
            current_status = "running"

        response_data = {
            "task_id": task_id,
            "status": current_status,
            "final_answer": final_answer,
            "error_message": error_message,
            "current_iteration": stored_state_data.get("current_iteration"),
            "search_depth": stored_state_data.get("search_depth"),
            "last_llm_output": stored_state_data.get("last_llm_output"),
            "metadata": stored_state_data.get("metadata"),
        }

        return WorkflowStatusResponse(**response_data)

    except HTTPException as http_exc:
        raise http_exc
    except MemoryError as mem_err:
        logger.error(
            f"API '/status': MemoryError retrieving status for task_id {task_id}: {mem_err.message}",
            exc_info=False,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service dependency error retrieving task status: {mem_err.code}",
        )
    except Exception as e:
        logger.error(
            f"API '/status': Unexpected error retrieving status for task_id {task_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving task status.",
        )


@router.get(
    "/graphs",
    response_model=List[GraphInfo],
    summary="사용 가능한 그래프 설정 목록 조회",
    description="설정된 디렉토리에서 사용 가능한 에이전트 그래프 설정 파일(.json) 목록을 반환합니다.",
    tags=["Configuration"],
)
async def list_available_graphs():
    """
    설정된 `AGENT_GRAPH_CONFIG_DIR` 에서 `.json` 파일을 찾아 목록을 반환합니다.
    """
    logger.info("API '/graphs': Request received to list available graphs")
    graph_dir_str = getattr(settings, "AGENT_GRAPH_CONFIG_DIR", "config/agent_graphs")
    graph_dir = Path(graph_dir_str)
    available_graphs: List[GraphInfo] = []

    if not graph_dir.is_dir():
        logger.warning(
            f"API '/graphs': Agent graph directory not found or not a directory: {graph_dir}"
        )
        return available_graphs

    try:
        for file_path in graph_dir.glob("*.json"):
            graph_name = file_path.stem
            description = f"Workflow configuration '{graph_name}'"
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_description = data.get("description")
                    if file_description and isinstance(file_description, str):
                        description = file_description
            except json.JSONDecodeError as e:
                logger.warning(
                    f"API '/graphs': Could not parse JSON from {file_path.name}: {e}"
                )

            except Exception as e:
                logger.warning(
                    f"API '/graphs': Error reading or parsing {file_path.name}: {e}",
                    exc_info=False,
                )

            available_graphs.append(GraphInfo(name=graph_name, description=description))

        logger.info(
            f"API '/graphs': Found {len(available_graphs)} potential graph configurations in {graph_dir}."
        )
        return available_graphs
    except Exception as e:
        logger.error(
            f"API '/graphs': Error listing graph configurations from {graph_dir}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list available workflow configurations.",
        )


@router.get(
    "/tools",
    response_model=List[ToolInfo],
    summary="사용 가능한 도구 목록 조회",
    description="시스템에 등록되어 에이전트가 사용할 수 있는 도구 목록과 설명을 반환합니다.",
    tags=["Configuration"],
)
async def list_available_tools(tool_manager: ToolManagerDep):
    """
    `ToolManager`에 등록된 모든 도구의 이름, 설명, 인자 요약 정보를 반환합니다.
    """
    logger.info("API '/tools': Request received to list available tools")
    try:
        tool_list_details = tool_manager.list_tools()

        response_list = [
            ToolInfo(
                name=tool.get("name", "unknown_tool"),
                description=tool.get("description", "No description provided."),
                args_schema_summary=tool.get("args_schema_summary"),
            )
            for tool in tool_list_details
        ]
        logger.info(f"API '/tools': Returning {len(response_list)} available tools.")
        return response_list
    except Exception as e:
        logger.error(
            f"API '/tools': Error retrieving tool list from ToolManager: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve the list of available tools.",
        )


@router.websocket("/ws/status/{task_id}")
async def websocket_status_endpoint(
    websocket: WebSocket,
    notification_service: NotificationServiceDep,
    task_id: str = Path(..., description="상태 업데이트를 수신할 작업의 ID"),
):
    print(
        f"DEBUG: websocket_status_endpoint CALLED for task_id: {task_id}, client: {websocket.client}"
    )
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    logger.info(
        f"WebSocket: [/ws/status/{task_id}] - Connection request from {client_host}:{client_port}"
    )

    try:
        await websocket.accept()
        logger.info(
            f"WebSocket: [/ws/status/{task_id}] - Connection ACCEPTED for {client_host}:{client_port}"
        )
        logger.info(
            f"WebSocket: [/ws/status/{task_id}] - Attempting to subscribe to NotificationService..."
        )
        await notification_service.subscribe(task_id, websocket)
        logger.info(
            f"WebSocket: [/ws/status/{task_id}] - Successfully SUBSCRIBED to NotificationService."
        )
        while True:
            await asyncio.sleep(
                settings.WEBSOCKET_KEEP_ALIVE_INTERVAL
                if hasattr(settings, "WEBSOCKET_KEEP_ALIVE_INTERVAL")
                else 60
            )
    except WebSocketDisconnect:
        logger.info(
            f"WebSocket: [/ws/status/{task_id}] - Client {client_host}:{client_port} DISCONNECTED (WebSocketDisconnect)."
        )
    except Exception as e_ws:
        logger.error(
            f"WebSocket: [/ws/status/{task_id}] - ERROR for {client_host}:{client_port}: {e_ws}",
            exc_info=True,
        )
    finally:
        logger.info(
            f"WebSocket: [/ws/status/{task_id}] - Cleaning up connection for {client_host}:{client_port} in finally block..."
        )
        await notification_service.unsubscribe(task_id, websocket)
        logger.info(
            f"WebSocket: [/ws/status/{task_id}] - Connection for {client_host}:{client_port} CLEANED UP AND CLOSED."
        )

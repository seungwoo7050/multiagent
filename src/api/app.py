# src/api/app.py
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# --- 프로젝트 루트 설정 ---
# 이 스크립트 파일의 위치를 기준으로 프로젝트 루트 경로를 계산합니다.
# (api 폴더 -> src 폴더 -> project_root)
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to sys.path: {project_root}") # 디버깅용 로그
except Exception as path_e:
    print(f"Error setting up project root path: {path_e}", file=sys.stderr)
    sys.exit(1)

# --- 기본 설정 및 로깅 로드 ---
# 이 초기 로딩은 다른 모듈 임포트 전에 수행되어야 할 수 있습니다.
try:
    # 필요한 설정 및 로거 함수 임포트
    from src.config.settings import get_settings
    from src.config.logger import get_logger, setup_logging

    # 설정 인스턴스 로드
    settings = get_settings()
    # 로깅 시스템 설정
    setup_logging(settings) # 설정을 명시적으로 전달
    # 메인 로거 가져오기
    logger = get_logger(__name__)
    logger.info("Settings and logging initialized successfully.")

except Exception as initial_e:
    # 초기 설정/로깅 실패 시 치명적 오류 처리
    print(f'FATAL: Could not initialize settings or logging: {initial_e}\n{traceback.format_exc()}', file=sys.stderr)
    sys.exit(1)

# --- 필요한 FastAPI 및 기타 모듈 임포트 ---
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- 애플리케이션 구성 요소 임포트 ---
try:
    # 오류 처리
    from src.config.errors import ERROR_TO_HTTP_STATUS, BaseError, ErrorCode
    # 연결 관리 (Lifespan에서 사용)
    from src.config.connections import setup_connection_pools, cleanup_connection_pools
    # 메모리 관리자 (Lifespan에서 초기화 확인 및 API 의존성에서 사용)
    from src.memory.memory_manager import get_memory_manager
    # 도구 관리자 (Lifespan에서 도구 로드 및 API 의존성에서 사용)
    from src.services.tool_manager import get_tool_manager, ToolManager
    # 응답 모델 (헬스체크 등에서 사용)
    from src.schemas.response_models import HealthCheckResponse
    # API 라우터 임포트
    from src.api.routers import router as api_router # routers.py 에서 생성한 router 객체 임포트
    # LLM 클라이언트 또는 관련 초기화 함수 (필요시)
    # from src.services.llm_client import initialize_llm_module # 예시
except ImportError as import_err:
    logger.critical(f"Failed to import core application components: {import_err}", exc_info=True)
    sys.exit(1)


# --- FastAPI Lifespan (시작/종료 이벤트 처리) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    애플리케이션 시작 및 종료 시 필요한 초기화 및 정리 작업을 관리합니다.
    - 연결 풀 설정/정리
    - 메모리 관리자 확인
    - 도구 관리자 초기화 및 도구 로드
    """
    global logger # 전역 로거 사용
    logger.info("Application startup sequence initiated via lifespan...")
    initialization_errors = 0

    try:
        # 1. 연결 풀 설정 (예: Redis)
        try:
            await setup_connection_pools()
            logger.info("Connection pools setup successfully.")
        except Exception as conn_e:
            logger.error(f"Failed to setup connection pools: {conn_e}", exc_info=True)
            initialization_errors += 1

        # 2. 메모리 관리자 초기화 확인
        try:
            # get_memory_manager() 호출하여 싱글톤 인스턴스 생성 및 초기화 확인
            get_memory_manager()
            logger.info("Memory Manager initialized or confirmed.")
        except Exception as mem_e:
            logger.error(f"Failed to initialize Memory Manager: {mem_e}", exc_info=True)
            initialization_errors += 1

        # 3. 도구 관리자 초기화 및 도구 로드
        try:
            logger.info("Initializing Tool Manager and loading tools...")
            # 'global_tools' 이름의 싱글톤 ToolManager 가져오기 (없으면 생성)
            tool_manager: ToolManager = get_tool_manager('global_tools')
            # 도구들이 위치한 디렉토리 경로 설정 (settings 또는 직접 지정)
            tool_directory = 'src/tools' # 로드맵 기준 경로
            # 디렉토리에서 도구 모듈 로드 (자동 등록은 ToolManager 내부 로직 또는 데코레이터에 위임 가능)
            imported_count = tool_manager.load_tools_from_directory(tool_directory, auto_register=True)
            registered_tool_names = list(tool_manager.get_names())
            logger.info(f"Tool loading complete. Imported {imported_count} modules. Registered tools: {registered_tool_names}")
            if not registered_tool_names:
                logger.warning("No tools were registered. Check the tool directory and implementations.")
        except Exception as tool_e:
            logger.error(f"Error during Tool Manager initialization or tool loading: {tool_e}", exc_info=True)
            initialization_errors += 1

        # 4. 기타 필요한 초기화 (예: LLM 관련 사전 로딩)
        # try:
        #     initialize_llm_module() # 예시 함수 호출
        #     logger.info("LLM Module initialized.")
        # except Exception as llm_init_e:
        #     logger.error(f"Failed to initialize LLM module: {llm_init_e}", exc_info=True)
        #     initialization_errors += 1

        # 5. 최종 초기화 상태 확인
        if initialization_errors > 0:
            # 심각한 오류 시 애플리케이션 시작 중단 또는 경고 후 계속 진행 선택 가능
            error_message = f"{initialization_errors} critical component(s) failed to initialize during startup. Check logs for details."
            logger.critical(error_message)
            # 개발 환경에서는 오류를 발생시켜 빠르게 인지하도록 하고,
            # 운영 환경에서는 경고만 기록하고 시작을 시도할 수도 있습니다. (정책에 따라 결정)
            if settings.ENVIRONMENT == 'development':
                 raise RuntimeError(error_message)
            # 또는 sys.exit(1) 사용 가능
        else:
            logger.info("Core components initialized successfully.")

        # --- 애플리케이션 실행 구간 ---
        logger.info(f"Application '{settings.APP_NAME}' v{settings.APP_VERSION} startup sequence finished successfully.")
        yield
        # --- 애플리케이션 종료 시 실행될 코드 ---
        logger.info("Application shutdown sequence initiated...")

        # 리소스 정리 (예: 연결 풀 닫기)
        try:
            await cleanup_connection_pools()
            logger.info("Connection pools cleaned up successfully.")
        except Exception as cleanup_e:
            logger.error(f"Error cleaning up connection pools during shutdown: {cleanup_e}", exc_info=True)

        # 기타 정리 작업 (예: 백그라운드 스레드 종료 등)

        logger.info("Application shutdown sequence finished.")

    except Exception as lifespan_err:
        # Lifespan 함수 자체의 예외 처리
        logger.critical(f"Fatal error during application lifespan management: {lifespan_err}", exc_info=True)
        # 애플리케이션 시작/종료 로직 실패는 심각하므로 런타임 에러 발생
        raise RuntimeError("Application failed during startup or shutdown") from lifespan_err


# --- FastAPI 앱 인스턴스 생성 ---
# settings에서 앱 정보와 디버그 상태를 읽어 FastAPI 앱을 생성합니다.
app = FastAPI(
    title=settings.APP_NAME + " (Framework-Centric)", # 로드맵에 맞게 이름 변경
    version=settings.APP_VERSION,
    description='Framework-Centric Multi-Agent System API using LangGraph, FastAPI, and best practices.', # 설명 업데이트
    debug=settings.DEBUG,
    lifespan=lifespan # Lifespan 이벤트 핸들러 등록
)

# --- 미들웨어 설정 ---
# CORS 설정 (settings.CORS_ORIGINS 에 정의된 출처 허용)
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS, # 설정 파일에서 읽어온 허용할 출처 목록
        allow_credentials=True,              # 자격 증명(쿠키 등) 허용 여부
        allow_methods=["*"],                 # 허용할 HTTP 메소드 (GET, POST 등)
        allow_headers=["*"],                 # 허용할 요청 헤더
    )
    logger.info(f"CORS middleware added. Allowed origins: {settings.CORS_ORIGINS}")
else:
    logger.warning("CORS_ORIGINS not set in settings. CORS middleware not added.")

# (선택 사항) MCP 미들웨어 추가
# try:
#     from src.middleware.mcp_middleware import MCPSerializationMiddleware # 실제 경로 확인 필요
#     app.add_middleware(MCPSerializationMiddleware)
#     logger.info('MCPSerializationMiddleware added.')
# except ImportError:
#     logger.info('MCPSerializationMiddleware not found or not configured.')
# except Exception as mcp_mw_err:
#     logger.error(f"Failed to add MCPSerializationMiddleware: {mcp_mw_err}", exc_info=True)


# --- 예외 핸들러 설정 ---
# BaseError (우리가 정의한 커스텀 에러) 처리
@app.exception_handler(BaseError)
async def base_error_exception_handler(request: Request, exc: BaseError):
    # 미리 정의된 ErrorCode와 HTTP 상태 코드 매핑 사용
    status_code = ERROR_TO_HTTP_STATUS.get(exc.code, 500)
    error_content = exc.to_dict() # 에러 정보를 dict로 변환
    # 에러 로깅 (원본 에러 포함)
    logger.error(
        f"API Error Handled ({exc.code}): {exc.message}",
        extra={"error_details": error_content, "url": str(request.url)},
        exc_info=exc.original_error # 원본 예외가 있으면 스택 트레이스 포함
    )
    return JSONResponse(
        status_code=status_code,
        content=error_content
    )

# FastAPI의 요청 유효성 검사 오류 처리
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # 유효성 검사 실패 시 상세 정보 로깅
    logger.warning(
        f"Request validation failed: {exc.errors()}",
        extra={"errors": exc.errors(), "url": str(request.url), "body": await request.body()}
    )
    # 클라이언트에게 422 상태 코드와 오류 상세 정보 반환
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()}
    )

# 그 외 모든 예외 처리 (최후의 방어선)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # 처리되지 않은 예외 로깅 (스택 트레이스 포함)
    logger.exception(f"Unhandled exception during request to {request.url.path}: {exc}")
    # 클라이언트에게는 일반적인 500 오류 메시지 반환
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "code": ErrorCode.SYSTEM_ERROR.value # 일반 시스템 오류 코드 사용
        }
    )

# --- 기본 라우트 ---
@app.get(
    "/health",
    tags=["System"], # Swagger UI 그룹화 태그
    summary="Health Check",
    description="Performs a basic health check of the API server.",
    response_model=HealthCheckResponse
)
async def health_check():
    """Basic health check endpoint."""
    logger.debug("Health check endpoint called")
    return HealthCheckResponse(status="ok")

# --- API 라우터 포함 ---
# src/api/routers.py 에서 정의한 라우터를 FastAPI 앱에 등록합니다.
try:
    app.include_router(
        api_router,                 # 임포트한 라우터 객체
        prefix=settings.API_PREFIX, # 모든 하위 경로에 적용될 접두사 (예: "/api/v1")
        # tags=["API"]              # 이 라우터의 모든 엔드포인트에 기본 태그 설정 가능
    )
    logger.info(f"Included API routes from src.api.routers under prefix: {settings.API_PREFIX}")

    # (선택 사항) 스트리밍/WebSocket 라우터가 있다면 여기서 추가 등록
    # from src.api.streaming_router import router as streaming_router
    # app.include_router(streaming_router, tags=["Streaming"])
    # logger.info('Included streaming WebSocket routes.')

except NameError as ne:
     logger.error(f"Failed to include routers: '{ne}' likely not defined or imported correctly.", exc_info=True)
except Exception as include_err:
    logger.error(f"Error including API routers: {include_err}", exc_info=True)


# --- 서버 실행 (스크립트 직접 실행 시) ---
if __name__ == '__main__':
    # 이 블록은 'python src/api/app.py' 명령으로 실행될 때만 작동합니다.
    # uvicorn 명령어를 직접 사용하는 것이 일반적입니다.
    logger.info(f"Starting API server directly using Uvicorn (Host: {settings.API_HOST}, Port: {settings.API_PORT})...")

    # Uvicorn 서버 실행 설정
    uvicorn.run(
        # 개발 환경에서는 문자열 경로를 사용하여 자동 리로드 활성화
        app='src.api.app:app' if settings.ENVIRONMENT == 'development' else app,
        host=settings.API_HOST,           # 설정 파일에서 읽어온 호스트
        port=settings.API_PORT,           # 설정 파일에서 읽어온 포트
        reload=(settings.ENVIRONMENT == 'development'), # 개발 환경일 때 코드 변경 시 자동 리로드
        log_level=settings.LOG_LEVEL.lower() # 설정 파일의 로그 레벨 사용
        # 추가적인 Uvicorn 설정 가능 (예: workers, ssl_keyfile 등)
    )
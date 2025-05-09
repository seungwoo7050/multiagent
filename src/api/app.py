# src/api/app.py
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# OpenTelemetry FastAPI 계측기
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
# >>> INSERT OTel extra instrumentors
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
# <<< END insert

# --- 프로젝트 루트 설정 ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        # print(f"Added project root to sys.path: {project_root}") # 디버깅용 로그
except Exception as path_e:
    print(f"Error setting up project root path: {path_e}", file=sys.stderr)
    sys.exit(1)

# --- 기본 설정 및 로깅 로드 ---
try:
    from src.config.settings import get_settings
    # src.utils.logger로 경로 변경
    from src.utils.logger import get_logger, setup_logging as setup_logging_util
    # OpenTelemetry 설정 함수 임포트
    from src.utils.telemetry import setup_telemetry

    settings = get_settings()
    # 로깅 시스템 설정 (Telemetry 설정 후 또는 함께)
    # setup_logging_util(settings) # Lifespan으로 이동
    logger = get_logger(__name__) # 초기 로거 (lifespan에서 재설정 가능)
    # logger.info("Initial settings and basic logger initialized.")

except Exception as initial_e:
    print(f'FATAL: Could not initialize basic settings or logging: {initial_e}\n{traceback.format_exc()}', file=sys.stderr)
    sys.exit(1)

# --- 필요한 FastAPI 및 기타 모듈 임포트 ---
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# OpenTelemetry FastAPI 계측기
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# --- 애플리케이션 구성 요소 임포트 ---
try:
    from src.config.errors import ERROR_TO_HTTP_STATUS, BaseError, ErrorCode
    from src.config.connections import setup_connection_pools, cleanup_connection_pools
    from src.memory.memory_manager import get_memory_manager
    from src.services.tool_manager import get_tool_manager, ToolManager
    from src.schemas.response_models import HealthCheckResponse
    from src.api.routers import router as api_router
except ImportError as import_err:
    logger.critical(f"Failed to import core application components: {import_err}", exc_info=True)
    sys.exit(1)


# --- FastAPI Lifespan (시작/종료 이벤트 처리) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global logger # 전역 로거 사용 (lifespan 내에서 재할당 가능)

    # 0. OpenTelemetry 설정 (로깅보다 먼저 또는 함께)
    try:
        print("Attempting to setup Telemetry...") # 부트스트랩 로깅
        setup_telemetry()
        print("Telemetry setup completed (or skipped if already done).")
    except Exception as tel_e:
        print(f"FATAL: Failed to setup OpenTelemetry: {tel_e}", file=sys.stderr)
        # 운영 환경에서는 여기서 exit 할 수도 있음
        # 여기서는 로깅이 아직 완전히 설정되지 않았을 수 있으므로 print 사용

    # 1. 로깅 시스템 설정 (Telemetry 설정 후)
    try:
        print("Attempting to setup Logging utility...")
        setup_logging_util(settings) # utils.logger의 setup_logging 사용
        logger = get_logger(__name__) # 설정된 로거로 업데이트
        logger.info("Logging utility setup successfully.")
    except Exception as log_setup_e:
        # 로깅 설정 실패 시, 기본 로거 또는 print로 오류 출력
        print(f"FATAL: Failed to setup logging utility: {log_setup_e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1) # 로깅 없이는 진행하기 어려움

    logger.info("Application startup sequence initiated via lifespan...")
    initialization_errors = 0

    try:
        # 2. 연결 풀 설정 (예: Redis)
        try:
            await setup_connection_pools()
            logger.info("Connection pools setup successfully.")
        except Exception as conn_e:
            logger.error(f"Failed to setup connection pools: {conn_e}", exc_info=True)
            initialization_errors += 1

        # 3. 메모리 관리자 초기화 확인
        try:
            get_memory_manager()
            logger.info("Memory Manager initialized or confirmed.")
        except Exception as mem_e:
            logger.error(f"Failed to initialize Memory Manager: {mem_e}", exc_info=True)
            initialization_errors += 1

        # 4. 도구 관리자 초기화 및 도구 로드
        try:
            logger.info("Initializing Tool Manager and loading tools...")
            tool_manager: ToolManager = get_tool_manager('global_tools')
            tool_directory = 'src/tools'
            imported_count = tool_manager.load_tools_from_directory(tool_directory, auto_register=True)
            registered_tool_names = list(tool_manager.get_names())
            logger.info(f"Tool loading complete. Imported {imported_count} modules. Registered tools: {registered_tool_names}")
            if not registered_tool_names:
                logger.warning("No tools were registered. Check the tool directory and implementations.")
        except Exception as tool_e:
            logger.error(f"Error during Tool Manager initialization or tool loading: {tool_e}", exc_info=True)
            initialization_errors += 1

        # 5. 최종 초기화 상태 확인
        if initialization_errors > 0:
            error_message = f"{initialization_errors} critical component(s) failed to initialize during startup. Check logs for details."
            logger.critical(error_message)
            if settings.ENVIRONMENT == 'development':
                 raise RuntimeError(error_message)
        else:
            logger.info("Core components initialized successfully.")

        logger.info(f"Application '{settings.APP_NAME}' v{settings.APP_VERSION} startup sequence finished successfully.")
        yield
        # --- 애플리케이션 종료 시 실행될 코드 ---
        logger.info("Application shutdown sequence initiated...")
        try:
            await cleanup_connection_pools()
            logger.info("Connection pools cleaned up successfully.")
        except Exception as cleanup_e:
            logger.error(f"Error cleaning up connection pools during shutdown: {cleanup_e}", exc_info=True)
        logger.info("Application shutdown sequence finished.")

    except Exception as lifespan_err:
        logger.critical(f"Fatal error during application lifespan management: {lifespan_err}", exc_info=True)
        raise RuntimeError("Application failed during startup or shutdown") from lifespan_err


# --- FastAPI 앱 인스턴스 생성 ---
app = FastAPI(
    title=settings.APP_NAME + " (Framework-Centric)",
    version=settings.APP_VERSION,
    description='Framework-Centric Multi-Agent System API using LangGraph, FastAPI, and best practices.',
    debug=settings.DEBUG,
    lifespan=lifespan
)

# --- OpenTelemetry FastAPI 계측기 적용 ---
# 중요: 라우터 추가 전에 계측기를 적용해야 모든 요청이 추적됩니다.
# setup_telemetry()가 lifespan에서 먼저 호출되어 Provider가 설정된 후 계측합니다.
try:
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT: # OTLP 엔드포인트가 설정된 경우에만 계측 (선택적)
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
        logger.info("FastAPIInstrumentor applied to the application for OpenTelemetry tracing.")
    else:
        logger.info("FastAPIInstrumentor skipped as OTEL_EXPORTER_OTLP_ENDPOINT is not set.")
except Exception as instr_e:
    logger.error(f"Failed to apply FastAPIInstrumentor: {instr_e}", exc_info=True)


# --- 미들웨어 설정 ---
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"CORS middleware added. Allowed origins: {settings.CORS_ORIGINS}")
else:
    logger.warning("CORS_ORIGINS not set in settings. CORS middleware not added.")


# --- 예외 핸들러 설정 ---
@app.exception_handler(BaseError)
async def base_error_exception_handler(request: Request, exc: BaseError):
    status_code = ERROR_TO_HTTP_STATUS.get(exc.code, 500)
    error_content = exc.to_dict()
    logger.error(
        f"API Error Handled ({exc.code}): {exc.message}",
        extra={"error_details": error_content, "url": str(request.url)},
        exc_info=exc.original_error
    )
    return JSONResponse(
        status_code=status_code,
        content=error_content
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Request validation failed: {exc.errors()}",
        extra={"errors": exc.errors(), "url": str(request.url), "body": await request.body()}
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception during request to {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "code": ErrorCode.SYSTEM_ERROR.value
        }
    )

# --- 기본 라우트 ---
@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Performs a basic health check of the API server.",
    response_model=HealthCheckResponse
)
async def health_check():
    logger.debug("Health check endpoint called")
    return HealthCheckResponse(status="ok")

# --- API 라우터 포함 ---
try:
    app.include_router(
        api_router,
        prefix=settings.API_PREFIX,
    )
    logger.info(f"Included API routes from src.api.routers under prefix: {settings.API_PREFIX}")
except NameError as ne:
     logger.error(f"Failed to include routers: '{ne}' likely not defined or imported correctly.", exc_info=True)
except Exception as include_err:
    logger.error(f"Error including API routers: {include_err}", exc_info=True)


# --- 서버 실행 (스크립트 직접 실행 시) ---
if __name__ == '__main__':
    logger.info(f"Starting API server directly using Uvicorn (Host: {settings.API_HOST}, Port: {settings.API_PORT})...")
    uvicorn.run(
        app='src.api.app:app' if settings.ENVIRONMENT == 'development' else app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=(settings.ENVIRONMENT == 'development'),
        log_level=settings.LOG_LEVEL.lower()
    )
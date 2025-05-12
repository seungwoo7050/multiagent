import contextlib
import os
import sys
import traceback
from typing import AsyncGenerator

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

except Exception as path_e:
    print(f"Error setting up project root path: {path_e}", file=sys.stderr)
    sys.exit(1)


try:
    from src.config.settings import get_settings
    from src.utils.logger import get_logger, setup_logging as setup_logging_util
    from src.utils.telemetry import setup_telemetry

    settings = get_settings()
    logger = get_logger(__name__)

except Exception as initial_e:
    print(
        f"FATAL: Could not initialize basic settings or logging: {initial_e}\n{traceback.format_exc()}",
        file=sys.stderr,
    )
    sys.exit(1)


import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


try:
    from src.config.errors import ERROR_TO_HTTP_STATUS, BaseError, ErrorCode
    from src.config.connections import setup_connection_pools, cleanup_connection_pools
    from src.memory.memory_manager import get_memory_manager
    from src.services.tool_manager import get_tool_manager, ToolManager
    from src.schemas.response_models import HealthCheckResponse
    from src.api.routers import router as api_router
except ImportError as import_err:
    logger.critical(
        f"Failed to import core application components: {import_err}", exc_info=True
    )
    sys.exit(1)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global logger

    try:
        print("Attempting to setup Telemetry...")
        setup_telemetry()
        print("Telemetry setup completed (or skipped if already done).")
    except Exception as tel_e:
        print(f"FATAL: Failed to setup OpenTelemetry: {tel_e}", file=sys.stderr)

    try:
        print("Attempting to setup Logging utility...")
        setup_logging_util(settings)
        logger = get_logger(__name__)
        logger.info("Logging utility setup successfully.")
    except Exception as log_setup_e:
        print(
            f"FATAL: Failed to setup logging utility: {log_setup_e}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Application startup sequence initiated via lifespan...")
    initialization_errors = 0

    try:
        try:
            await setup_connection_pools()
            logger.info("Connection pools setup successfully.")
        except Exception as conn_e:
            logger.error(f"Failed to setup connection pools: {conn_e}", exc_info=True)
            initialization_errors += 1

        try:
            get_memory_manager()
            logger.info("Memory Manager initialized or confirmed.")
        except Exception as mem_e:
            logger.error(f"Failed to initialize Memory Manager: {mem_e}", exc_info=True)
            initialization_errors += 1

        try:
            logger.info("Initializing Tool Manager and loading tools...")
            tool_manager: ToolManager = get_tool_manager("global_tools")
            tool_directory = "src/tools"
            imported_count = tool_manager.load_tools_from_directory(
                tool_directory, auto_register=True
            )
            registered_tool_names = list(tool_manager.get_names())
            logger.info(
                f"Tool loading complete. Imported {imported_count} modules. Registered tools: {registered_tool_names}"
            )
            if not registered_tool_names:
                logger.warning(
                    "No tools were registered. Check the tool directory and implementations."
                )
        except Exception as tool_e:
            logger.error(
                f"Error during Tool Manager initialization or tool loading: {tool_e}",
                exc_info=True,
            )
            initialization_errors += 1

        if initialization_errors > 0:
            error_message = f"{initialization_errors} critical component(s) failed to initialize during startup. Check logs for details."
            logger.critical(error_message)
            if settings.ENVIRONMENT == "development":
                raise RuntimeError(error_message)
        else:
            logger.info("Core components initialized successfully.")

        logger.info(
            f"Application '{settings.APP_NAME}' v{settings.APP_VERSION} startup sequence finished successfully."
        )
        yield

        logger.info("Application shutdown sequence initiated...")
        try:
            await cleanup_connection_pools()
            logger.info("Connection pools cleaned up successfully.")
        except Exception as cleanup_e:
            logger.error(
                f"Error cleaning up connection pools during shutdown: {cleanup_e}",
                exc_info=True,
            )
        logger.info("Application shutdown sequence finished.")

    except Exception as lifespan_err:
        logger.critical(
            f"Fatal error during application lifespan management: {lifespan_err}",
            exc_info=True,
        )
        raise RuntimeError(
            "Application failed during startup or shutdown"
        ) from lifespan_err


app = FastAPI(
    title=settings.APP_NAME + " (Framework-Centric)",
    version=settings.APP_VERSION,
    description="Framework-Centric Multi-Agent System API using LangGraph, FastAPI, and best practices.",
    debug=settings.DEBUG,
    lifespan=lifespan,
)


try:
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
        logger.info(
            "FastAPIInstrumentor applied to the application for OpenTelemetry tracing."
        )
    else:
        logger.info(
            "FastAPIInstrumentor skipped as OTEL_EXPORTER_OTLP_ENDPOINT is not set."
        )
except Exception as instr_e:
    logger.error(f"Failed to apply FastAPIInstrumentor: {instr_e}", exc_info=True)


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


@app.exception_handler(BaseError)
async def base_error_exception_handler(request: Request, exc: BaseError):
    status_code = ERROR_TO_HTTP_STATUS.get(exc.code, 500)
    error_content = exc.to_dict()
    logger.error(
        f"API Error Handled ({exc.code}): {exc.message}",
        extra={"error_details": error_content, "url": str(request.url)},
        exc_info=exc.original_error,
    )
    return JSONResponse(status_code=status_code, content=error_content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Request validation failed: {exc.errors()}",
        extra={
            "errors": exc.errors(),
            "url": str(request.url),
            "body": await request.body(),
        },
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception during request to {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "code": ErrorCode.SYSTEM_ERROR.value,
        },
    )


@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Performs a basic health check of the API server.",
    response_model=HealthCheckResponse,
)
async def health_check():
    logger.debug("Health check endpoint called")
    return HealthCheckResponse(status="ok")


try:
    app.include_router(
        api_router,
        prefix=settings.API_PREFIX,
    )
    logger.info(
        f"Included API routes from src.api.routers under prefix: {settings.API_PREFIX}"
    )
except NameError as ne:
    logger.error(
        f"Failed to include routers: '{ne}' likely not defined or imported correctly.",
        exc_info=True,
    )
except Exception as include_err:
    logger.error(f"Error including API routers: {include_err}", exc_info=True)


if __name__ == "__main__":
    logger.info(
        f"Starting API server directly using Uvicorn (Host: {settings.API_HOST}, Port: {settings.API_PORT})..."
    )
    uvicorn.run(
        app="src.api.app:app" if settings.ENVIRONMENT == "development" else app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=(settings.ENVIRONMENT == "development"),
        log_level=settings.LOG_LEVEL.lower(),
    )

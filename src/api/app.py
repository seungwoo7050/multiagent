import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.config import AgentConfig
from src.agents.factory import get_agent_factory
from src.config.connections import (cleanup_connection_pools,
                                    setup_connection_pools)
# --- Core Application Component Imports ---
from src.config.errors import ERROR_TO_HTTP_STATUS, BaseError, ErrorCode
from src.core.worker_pool import get_worker_pool
from src.llm import initialize_llm_module
from src.memory.manager import get_memory_manager
from src.orchestration.orchestration_worker_pool import WorkerPoolType
from src.orchestration.orchestrator import \
    get_orchestrator  # Import the core orchestrator getter
from src.orchestration.scheduler import get_scheduler
from src.tools.registry import get_registry as get_tool_registry

# --- Project Root Setup ---
# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Initial Configuration and Logging (Must happen BEFORE other imports) ---
try:
    from src.config.logger import get_logger, setup_logging
    from src.config.settings import get_settings

    # from src.config import initialize_config # initialize_config might be redundant if setup_logging covers all
    # Initialize settings and logging FIRST
    settings = get_settings()
    # Configure logging based on settings BEFORE initializing other modules that log
    setup_logging()
    # Initialize other config aspects if needed
    # initialize_config()

except Exception as e:
    # Use basic print for critical startup errors before logging is configured
    print(f'FATAL: Could not initialize settings or logging: {e}\n{traceback.format_exc()}', file=sys.stderr)
    sys.exit(1)

# --- Get Logger Instance ---
# Now that logging is configured, get the logger instance
logger = get_logger(__name__)

# --- Conditional MCP Middleware Import ---
# Attempt to import MCP middleware, handle potential errors gracefully
try:
    from src.core.mcp.api.serialization_middleware import \
        MCPSerializationMiddleware
    logger.info("MCPSerializationMiddleware imported successfully.")
except ImportError as e:
    logger.error(f'Could not import MCPSerializationMiddleware: {e}. MCP Middleware will not be active.')
    MCPSerializationMiddleware = None # Ensure it's None if import fails
except Exception as e:
    logger.error(f'Unexpected error importing MCPSerializationMiddleware: {e}', exc_info=True)
    MCPSerializationMiddleware = None


# --- Tool Imports for Registration ---
# Explicitly import tool modules to trigger registration via decorators on startup
try:
    logger.info("Tool modules imported successfully for registration.")
except ImportError as e:
    logger.warning(f"Could not import all tool modules, some tools might not be registered: {e}")
except Exception as e:
    logger.error(f"Error importing tool modules during initial setup: {e}", exc_info=True)

# --- Lifespan Context Manager for Startup/Shutdown ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages application startup and shutdown events using the lifespan context manager.
    Replaces the deprecated on_event("startup") and on_event("shutdown").
    """
    global logger # Ensure logger is accessible
    logger.info("Application startup sequence initiated via lifespan...")

    # === Startup Logic ===
    try:
        # 1. Basic Config/Connections
        setup_connection_pools() # Initialize Redis, etc. pools if needed explicitly
        logger.info("Connection pools setup initiated.")
        initialize_llm_module() # Initialize LLM module specifics (e.g., pre-warm connections)
        logger.info("LLM Module initialization initiated.")

        # 2. Component Registration (Agents & Tools)
        logger.info("Starting component registration...")

        # Agent Registration from JSON config file
        registered_agent_count = 0
        failed_configs = []
        try:
            logger.info("Registering Agents from configuration file...")
            agent_factory = await get_agent_factory()
            # Read path from settings (requires AGENT_CONFIG_FILE_PATH in settings.py)
            config_file_path_str = getattr(settings, 'AGENT_CONFIG_FILE_PATH', 'configs/agent_configs.json')
            config_file_path = Path(config_file_path_str)

            if config_file_path.exists():
                logger.info(f"Loading agent configurations from: {config_file_path}")
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    all_configs_data = json.load(f)

                configs_to_process: List[Dict[str, Any]] = []
                if isinstance(all_configs_data, list):
                    configs_to_process = all_configs_data
                elif isinstance(all_configs_data, dict): # Allow dict format too
                    configs_to_process = list(all_configs_data.values())
                else:
                    logger.error(f"Invalid format in {config_file_path}. Expected list or dict of agent configurations.")

                # Validate and register each config
                loaded_names = set()
                for config_data in configs_to_process:
                    if not isinstance(config_data, dict) or 'name' not in config_data:
                         logger.warning(f"Skipping invalid agent config entry (must be dict with 'name'): {config_data}")
                         continue
                    agent_name = config_data['name']
                    loaded_names.add(agent_name)
                    try:
                        config_obj = AgentConfig.model_validate(config_data)
                        agent_factory.register_agent_config(config_obj)
                        registered_agent_count += 1
                        logger.debug(f"Successfully registered agent config: {agent_name}")
                    except Exception as val_err:
                        logger.error(f"Failed to validate/register agent config for '{agent_name}': {val_err}", exc_info=True)
                        failed_configs.append(agent_name)

                # Check if required agents were loaded
                required_names = {
                    getattr(settings, 'PLANNER_AGENT_NAME', 'default_planner'), # Use defaults if not set
                    getattr(settings, 'EXECUTOR_AGENT_NAME', 'default_executor')
                }
                missing_required = required_names - loaded_names
                if missing_required:
                     logger.error(f"CRITICAL: Required agent configurations missing from {config_file_path}: {missing_required}")
                     # Consider raising an error to halt startup if these are critical
                     # raise RuntimeError(f"Missing required agent configurations: {missing_required}")

            else:
                logger.error(f"Agent configuration file not found: {config_file_path}. Cannot register agents from file.")
                # Consider raising an error to halt startup if configs are critical
                raise FileNotFoundError(f"Agent configuration file not found: {config_file_path}")

            logger.info(f"Agent registration finished. Registered: {registered_agent_count}, Failed: {len(failed_configs)}")
            if failed_configs:
                 logger.error(f"Failed to register configurations for: {failed_configs}")

        except Exception as e:
            logger.error(f"Error during Agent Registration phase: {e}", exc_info=True)
            

        # Tool Registration (Triggered by imports at top, verified here)
        try:
            logger.info("Verifying Tool Registration...")
            tool_registry = get_tool_registry('global_tools') # Get instance to verify
            try:
                # 필요한 도구 클래스들을 import 합니다. (파일 상단에도 import 구문이 있지만, 여기서 다시 명시해도 괜찮습니다)
                from src.tools.calculator import CalculatorTool
                from src.tools.datetime_tool import DateTimeTool
                from src.tools.web_search import WebSearchTool
                from src.tools.web_search_google import GoogleSearchTool

                # 추가적인 도구 클래스가 있다면 여기에 import 하세요

                tools_to_register = [
                    CalculatorTool,
                    DateTimeTool,
                    WebSearchTool,
                    GoogleSearchTool
                    # 추가한 도구 클래스를 이 리스트에 넣으세요
                ]

                logger.info(f"Explicitly registering {len(tools_to_register)} tools...")
                registered_count = 0
                for tool_cls in tools_to_register:
                    try:
                        tool_registry.register(tool_cls)
                        registered_count += 1
                        logger.debug(f"Registered tool: {tool_cls.__name__}")
                    except Exception as reg_err:
                        # 이미 등록되었거나 다른 이유로 실패할 경우 로그 남기기
                        logger.error(f"Failed to register tool {tool_cls.__name__}: {reg_err}", exc_info=True)
                logger.info(f"Explicit tool registration complete. Registered: {registered_count}/{len(tools_to_register)}")

            except ImportError as import_err:
                logger.error(f"Could not import tool classes for explicit registration: {import_err}")
            except Exception as general_err:
                logger.error(f"An error occurred during explicit tool registration: {general_err}", exc_info=True)
            # --- 명시적 도구 등록 끝 ---
            registered_tools = tool_registry.list_tools() # Get list of registered tools
            if registered_tools:
                 logger.info(f"Tools registered in 'global_tools' registry: {[tool['name'] for tool in registered_tools]}")
            else:
                 logger.warning("No tools seem to be registered. Check tool module imports and @register_tool() decorators.")
        except Exception as e:
            logger.error(f"Error during Tool Registration verification: {e}", exc_info=True)

        # 3. Explicit Component Initialization
        logger.info("Starting explicit component initialization...")
        initialization_errors = 0
        # List of getter functions for components to initialize
        # We use the core getters directly here, not the FastAPI dependency functions
        component_getters = {
            "Memory Manager": get_memory_manager,
            "Scheduler": get_scheduler,
            "Worker Pool (default)": lambda: get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO),
            "Task Queue": lambda: get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO).task_queue, # Assuming worker pool holds queue reference or get queue directly
            "Agent Factory": get_agent_factory,
            "Tool Registry (global_tools)": lambda: get_tool_registry('global_tools'),
            # "Flow Controller": get_flow_controller, # Initialize if needed globally at startup
            "Orchestrator": get_orchestrator # Initialize orchestrator last, ensuring its deps are available
        }

        initialized_components: Dict[str, Any] = {}

        for name, getter_func in component_getters.items():
            instance = None # Reset instance for each component
            try:
                logger.debug(f"Attempting to initialize {name}...")
                start_time = time.monotonic()
                # Special handling for Orchestrator to pass dependencies if needed by its getter
                if name == "Orchestrator":
                    # Ensure dependencies are already initialized and available
                    task_q = initialized_components.get("Task Queue")
                    mem_m = initialized_components.get("Memory Manager")
                    work_p = initialized_components.get("Worker Pool (default)")
                    if task_q and mem_m and work_p:
                         instance = await getter_func(task_queue=task_q, memory_manager=mem_m, worker_pool=work_p)
                    else:
                         missing_deps = [n for n, c in {"Task Queue": task_q, "Memory Manager": mem_m, "Worker Pool (default)": work_p}.items() if not c]
                         logger.error(f"Cannot initialize Orchestrator: Missing dependencies {missing_deps}")
                         raise RuntimeError(f"Orchestrator dependencies not initialized: {missing_deps}")
                elif name == "Task Queue":
                    # RedisStreamTaskQueue를 직접 생성하는 방식으로 변경
                    try:
                        from src.orchestration.task_queue import \
                            RedisStreamTaskQueue
                        instance = RedisStreamTaskQueue(
                            stream_name=getattr(settings, 'TASK_QUEUE_STREAM_NAME', 'task_stream'),
                            consumer_group=getattr(settings, 'TASK_QUEUE_GROUP_NAME', 'orchestration_group')
                        )
                        initialized_components[name] = instance
                    except Exception as e:
                        logger.error(f"Failed to initialize Task Queue: {e}", exc_info=True)
                        raise
                # elif name == "Task Queue":
                #      # Example: Get queue from worker pool if structured that way
                #      worker_pool_instance = initialized_components.get("Worker Pool (default)")
                #      if worker_pool_instance and hasattr(worker_pool_instance, 'task_queue'):
                #          instance = getattr(worker_pool_instance, 'task_queue')
                #      else:
                #          # Fallback: Try to get queue directly if worker pool structure is different
                #          # This might require a dedicated get_task_queue function
                #          logger.warning("Could not get Task Queue from default worker pool, attempting direct get (requires implementation)")
                #          # instance = await get_task_queue() # Needs a get_task_queue function
                #          raise NotImplementedError("Direct Task Queue retrieval not implemented, assuming it's part of WorkerPool.")

                else:
                    instance =  getter_func()

                duration = time.monotonic() - start_time
                if instance:
                    initialized_components[name] = instance # Store initialized instance
                    logger.info(f"Successfully initialized {name} (Type: {type(instance).__name__}) in {duration:.4f}s")
                else:
                    logger.error(f"Initialization function for {name} returned None.")
                    initialization_errors += 1
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}", exc_info=True)
                initialization_errors += 1

        if initialization_errors > 0:
            logger.critical(f"{initialization_errors} critical components failed to initialize during startup. The application might be unstable.")
            # Decide if startup should halt
            # raise RuntimeError(f"{initialization_errors} essential components failed to initialize.")
        else:
            logger.info("Explicit component initialization completed successfully.")

        # === End of Startup Logic ===
        logger.info("Application startup sequence finished successfully.")
        yield # Application runs here
        # === Shutdown Logic ===
        logger.info("Application shutdown sequence initiated...")
        try:
            await cleanup_connection_pools()
            logger.info("Connection pools cleaned up.")
        except Exception as e:
            logger.error(f"Error cleaning up connection pools during shutdown: {e}", exc_info=True)

        try:
            # Assuming shutdown_all_worker_pools exists in the correct module now
            from src.core.worker_pool import shutdown_all_worker_pools
            await shutdown_all_worker_pools()
            logger.info("Worker pools shut down.")
        except ImportError:
             logger.error("Could not import shutdown_all_worker_pools. Worker pools might not be shut down properly.")
        except Exception as e:
            logger.error(f"Error shutting down worker pools: {e}", exc_info=True)

        try:
            agent_factory = await get_agent_factory()
            await agent_factory.shutdown()
            logger.info("Agent factory shut down.")
        except Exception as e:
            logger.error(f"Error shutting down agent factory: {e}", exc_info=True)

        logger.info("Application shutdown sequence finished.")

    except Exception as startup_err:
        logger.critical(f"Fatal error during application startup: {startup_err}", exc_info=True)
        # Optionally re-raise to prevent the app from starting incorrectly
        raise RuntimeError("Application failed to start") from startup_err


# --- FastAPI App Initialization with Lifespan ---
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description='High-Performance Multi-Agent Platform API',
    debug=settings.DEBUG,
    lifespan=lifespan # Register the lifespan context manager
)

# --- Middleware (Add after FastAPI initialization) ---
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f'CORS middleware added for origins: {settings.CORS_ORIGINS}')

if MCPSerializationMiddleware:
    app.add_middleware(MCPSerializationMiddleware)
    logger.info('MCPSerializationMiddleware added.')
else:
    logger.warning('MCPSerializationMiddleware is not available and was not added.')


# --- Global Exception Handlers ---
@app.exception_handler(BaseError)
async def base_error_exception_handler(request: Request, exc: BaseError):
    """Handles known custom errors derived from BaseError."""
    status_code = ERROR_TO_HTTP_STATUS.get(exc.code, 500)
    error_content = exc.to_dict()
    logger.error(f"API Error Handled ({exc.code}): {exc.message}", extra=error_content, exc_info=exc.original_error)
    return JSONResponse(
        status_code=status_code,
        content=error_content,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles FastAPI request validation errors."""
    logger.warning(f"Request validation failed: {exc.errors()}", extra={"errors": exc.errors(), "url": str(request.url)})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": exc.errors()},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handles any other unhandled exceptions."""
    logger.exception(f"Unhandled exception during request to {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "code": ErrorCode.SYSTEM_ERROR.value},
    )

# --- Health Check Endpoint ---
class HealthCheckResponse(BaseModel):
    status: str = 'ok'

@app.get('/health', tags=['System'], response_model=HealthCheckResponse)
async def health_check():
    """Basic health check endpoint."""
    logger.debug('Health check endpoint called')
    # Add more sophisticated checks here if needed (e.g., DB connection)
    return HealthCheckResponse(status='ok')

# --- Include API Routers ---
# Wrap router inclusion in try-except blocks for robustness
try:
    from src.api.routes import tasks as tasks_router
    app.include_router(tasks_router.router, prefix=settings.API_PREFIX)
    logger.info(f'Included task routes under prefix: {settings.API_PREFIX}')

    from src.api.routes import context as context_router
    app.include_router(context_router.router, prefix=settings.API_PREFIX) # Assuming same prefix
    logger.info(f'Included context routes under prefix: {settings.API_PREFIX}')
    
    from src.api.routes import agents as agents_router
    app.include_router(agents_router.router, prefix=settings.API_PREFIX)
    logger.info(f'Included agent routes under prefix: {settings.API_PREFIX}')

    from src.api.routes import tools as tools_router
    app.include_router(tools_router.router, prefix=settings.API_PREFIX)
    logger.info(f'Included tool routes under prefix: {settings.API_PREFIX}')

    from src.api.routes import config as config_router
    app.include_router(config_router.router, prefix=settings.API_PREFIX)
    logger.info(f'Included config routes under prefix: {settings.API_PREFIX}')

    # 스트리밍 및 메트릭 라우터 (Prefix 없음)
    from src.api.routes import streaming as streaming_router
    app.include_router(streaming_router.router) # prefix 없음
    logger.info('Included streaming WebSocket routes.')

    from src.api.routes import metrics as metrics_router
    app.include_router(metrics_router.router) # prefix 없음
    logger.info('Included metrics routes.')

except ImportError as e:
    logger.error(f'Failed to import API routes: {e}. Some endpoints will be unavailable.')
except Exception as e:
    logger.error(f'Error including API routes: {e}', exc_info=True)

try:
    from src.api.routes import streaming as streaming_router
    app.include_router(streaming_router.router) # No prefix usually needed for websockets
    logger.info('Included streaming WebSocket routes.')
except ImportError as e:
    logger.error(f'Failed to import streaming routes: {e}. WebSocket endpoints will be unavailable.')
except Exception as e:
    logger.error(f'Error including streaming routes: {e}', exc_info=True)


# --- Main Execution Block (for direct running) ---
if __name__ == '__main__':
    logger.info('Running API server directly from app.py...')
    # settings 로드 확인 추가
    if not settings:
         print("FATAL: Settings could not be loaded.", file=sys.stderr)
         sys.exit(1)
    # Metrics 서버 시작 (lifespan 내에서 시작하는 것이 더 일반적일 수 있음)
    from src.config.metrics import start_metrics_server
    metrics_thread = start_metrics_server()
    if metrics_thread:
         logger.info("Metrics server thread started.")

    uvicorn.run(
        'src.api.app:app',
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT == 'development',
        log_level=settings.LOG_LEVEL.lower()
    )
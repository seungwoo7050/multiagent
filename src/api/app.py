from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.config.settings import get_settings
from src.config.logger import get_logger, setup_logging
from src.config import initialize_config
try:
    settings = get_settings()
    initialize_config()
except Exception as e:
    print(f'FATAL: Could not initialize settings or logging: {e}', file=sys.stderr)
    sys.exit(1)
logger = get_logger(__name__)
try:
    from src.core.mcp.api.serialization_middleware import MCPSerializationMiddleware
except ImportError as e:
    logger.error(f'Could not import MCPSerializationMiddleware: {e}. MCP Middleware will not be active.')
    MCPSerializationMiddleware = None
app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, description='High-Performance Multi-Agent Platform API', debug=settings.DEBUG)
if settings.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    logger.info(f'CORS enabled for origins: {settings.CORS_ORIGINS}')
if MCPSerializationMiddleware:
    app.add_middleware(MCPSerializationMiddleware)
    logger.info('MCPSerializationMiddleware added to the application.')
else:
    logger.warning('MCPSerializationMiddleware could not be imported and was not added.')

class HealthCheckResponse(BaseModel):
    status: str = 'ok'

@app.get('/health', tags=['System'], response_model=HealthCheckResponse)
async def health_check():
    logger.debug('Health check endpoint called')
    return HealthCheckResponse(status='ok')

@app.on_event('startup')
async def startup_event():
    logger.info('API application startup...')
    from src.config.connections import setup_connection_pools
    setup_connection_pools()
    from src.llm import initialize_llm_module
    initialize_llm_module()
    logger.info('API application startup complete.')

@app.on_event('shutdown')
async def shutdown_event():
    logger.info('API application shutdown...')
    from src.config.connections import ConnectionManager
    await ConnectionManager.close_all_connections_async()
    logger.info('API application shutdown complete.')
try:
    from src.api.routes import tasks as tasks_router
    app.include_router(tasks_router.router, prefix=settings.API_PREFIX)
    logger.info(f'Included task routes under prefix: {settings.API_PREFIX}')
except ImportError as e:
    logger.error(f'Failed to import task routes: {e}. Task endpoints will be unavailable.')
except Exception as e:
    logger.error(f'Error including task routes: {e}', exc_info=True)
try:
    from src.api.routes import streaming as streaming_router
    app.include_router(streaming_router.router)
    logger.info('Included streaming WebSocket routes.')
except ImportError as e:
    logger.error(f'Failed to import streaming routes: {e}. WebSocket endpoints will be unavailable.')
except Exception as e:
    logger.error(f'Error including streaming routes: {e}', exc_info=True)
if __name__ == '__main__':
    logger.warning('Running API server directly from app.py for testing.')
    uvicorn.run('app:app', host=settings.API_HOST, port=settings.API_PORT, reload=settings.ENVIRONMENT == 'development', log_level=settings.LOG_LEVEL.lower())
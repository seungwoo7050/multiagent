import uvicorn
import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.config.settings import get_settings
from src.config.logger import get_logger
settings = get_settings()
logger = get_logger(__name__)
if __name__ == '__main__':
    logger.info(f'Starting {settings.APP_NAME} v{settings.APP_VERSION} API Server...')
    logger.info(f'Environment: {settings.ENVIRONMENT}')
    logger.info(f'Log Level: {settings.LOG_LEVEL}')
    logger.info(f'Debug Mode: {settings.DEBUG}')
    uvicorn.run('src.api.app:app', host=settings.API_HOST, port=settings.API_PORT, log_level=settings.LOG_LEVEL.lower(), reload=settings.DEBUG, workers=settings.WORKER_COUNT if settings.ENVIRONMENT == 'production' else 1)
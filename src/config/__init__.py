from src.config.settings import get_settings
from src.config.logger import get_logger, get_logger_with_context, setup_logging

settings = get_settings()
logger = get_logger(__name__)

def initialize_config():
    setup_logging()
    logger.info(f"Initializing {settings.APP_NAME}...")
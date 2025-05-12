"""
Multi-Agent Platform Configuration Package
"""

import logging


_bootstrap_logger = logging.getLogger("config.bootstrap")
_bootstrap_handler = logging.StreamHandler()
_bootstrap_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
_bootstrap_logger.addHandler(_bootstrap_handler)
_bootstrap_handler.setLevel(logging.INFO)


settings = None
logger = None


def initialize_config() -> bool:
    """
    Initialize configureation system, settings, and logging.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    global settings, logger

    try:
        from src.config.settings import get_settings

        settings = get_settings()

        from src.utils.logger import get_logger, setup_logging

        setup_logging()
        logger = get_logger(__name__)

        logger.info(f"Initializing {settings.APP_NAME} v{settings.APP_VERSION}...")
        return True
    except Exception as e:
        _bootstrap_logger.error(
            f"Failed to initialize configuration: {e}", exc_info=True
        )
        return False

import sys
from functools import lru_cache
import logging

try:
    from src.schemas.config import AppSettings
except ImportError:
    print(
        "Warning: Could not import AppSettings from src.schemas.config. Using fallback BaseSettings.",
        file=sys.stderr,
    )
    from pydantic_settings import BaseSettings

    AppSettings = BaseSettings

_logger = logging.getLogger(__name__)


@lru_cache()
def get_settings() -> AppSettings:
    """
    Get application settings, using cache for performance.

    Returns:
      AppSettings: Application settings instance

    Raises:
      Exception: If settings cannot be loaded
    """
    try:
        settings_instance = AppSettings()
        _logger.debug("Loaded application settings.")
        return settings_instance
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        _logger.error(error_msg, exc_info=True)
        print(f"FATAL: {error_msg}", file=sys.stderr)
        raise


settings = get_settings()

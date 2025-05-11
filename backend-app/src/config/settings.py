# src/config/settings.py
import sys
from functools import lru_cache
import logging # 로깅 추가
from typing import List # List 임포트 추가

# --- 기존 Settings 클래스 정의는 삭제 또는 주석 처리 ---
# class Settings(BaseSettings):
#    ... (이 내용들은 schemas/config.py 로 이동)

# --- src.schemas.config 에서 AppSettings 클래스 임포트 ---
try:
    from src.schemas.config import AppSettings
except ImportError:
    # Handle case where schemas might not be available yet during initial setup
    # This is less ideal but prevents complete failure on first run
    print("Warning: Could not import AppSettings from src.schemas.config. Using fallback BaseSettings.", file=sys.stderr)
    from pydantic_settings import BaseSettings
    AppSettings = BaseSettings # Fallback

_logger = logging.getLogger(__name__) # 로거 인스턴스 생성

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
        error_msg = f'Failed to load settings: {e}'
        _logger.error(error_msg, exc_info=True)
        print(f'FATAL: {error_msg}', file=sys.stderr)
        raise

# 기존에 settings = get_settings() 와 같이 인스턴스를 만들어 사용했다면
# 해당 부분은 유지하거나, 필요에 따라 get_settings()를 직접 호출하도록 변경합니다.
# 예를 들어, 다른 모듈에서 from src.config.settings import settings 로 사용했다면
# 아래 라인을 유지해야 합니다.
settings = get_settings()
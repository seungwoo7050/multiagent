import pytest
import os
from src.config.settings import get_settings
from src.schemas.config import AppSettings, LLMProviderSettings
from pydantic import ValidationError

def test_config_loading_success():
    """설정 로딩 성공 케이스 테스트"""
    # 환경 변수 설정 (테스트 격리를 위해 필요)
    os.environ["PRIMARY_LLM_PROVIDER"] = "test_provider"
    os.environ["LLM_PROVIDERS__test_provider__API_KEY"] = "test_key"
    os.environ["LLM_PROVIDERS__test_provider__MODEL_NAME"] = "test-model"
    os.environ["DEFAULT_REQUEST_TIMEOUT"] = "30"
    os.environ["LLM_MAX_RETRIES"] = "2"

    settings = AppSettings()
    assert settings.PRIMARY_LLM_PROVIDER == "test_provider"
    assert settings.LLM_PROVIDERS["test_provider"].api_key == "test_key"
    assert settings.LLM_PROVIDERS["test_provider"].model_name == "test-model"
    assert settings.REQUEST_TIMEOUT == 30
    assert settings.LLM_MAX_RETRIES == 2

def test_config_validation_failure():
    """설정 유효성 검사 실패 케이스 테스트"""
    get_settings.cache_clear()
    from src.schemas.config import AppSettings
    
    # Remove required environment variables
    os.environ.pop("PRIMARY_LLM_PROVIDER", None)
    os.environ.pop("LLM_PROVIDERS", None)
    
    with pytest.raises(ValidationError):
        # You need to actually create an instance here
        AppSettings(_env_file=None)  # This line was missing
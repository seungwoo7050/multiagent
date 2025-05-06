import pytest
import os
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
    # 일부러 필수 환경 변수 누락시켜 ValidationError 발생 유도
    os.environ.pop("PRIMARY_LLM_PROVIDER", None) # 확실히 제거
    os.environ.pop("LLM_PROVIDERS__test_provider__API_KEY", None)
    os.environ.pop("LLM_PROVIDERS__test_provider__MODEL_NAME", None)
    os.environ.pop("DEFAULT_REQUEST_TIMEOUT", None)
    os.environ.pop("LLM_MAX_RETRIES", None)

    with pytest.raises(ValidationError) as exc_info:
        AppSettings() # 필수 환경 변수 누락
    
    # ValidationError가 발생했는지 추가적으로 검증할 수 있습니다.
    # 예를 들어, errors() 메서드를 사용하여 어떤 필드에서 오류가 발생했는지 확인할 수 있습니다.
    # errors = exc_info.value.errors()
    # assert len(errors) > 0
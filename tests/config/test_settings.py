import os
from unittest.mock import patch, MagicMock

import pytest
from pydantic import ValidationError

from src.config.settings import Settings, get_settings, LLMProviderConfig

def test_default_settings():
    
    settings = Settings()
    
    assert settings.APP_NAME is not None
    assert settings.APP_VERSION is not None
    assert settings.ENVIRONMENT in ["development", "production"]
    
    assert "openai" in settings.LLM_PROVIDERS_CONFIG
    assert "anthropic" in settings.LLM_PROVIDERS_CONFIG
    
def test_environment_variable_override():
    
    test_app_name = "TestApp"
    test_app_version = "2.0.0"
    
    with patch.dict(os.environ, {
        "APP_NAME": test_app_name,
        "APP_VERSION": test_app_version,
        "DEBUG": "true",
    }):
        settings = Settings()
        
        assert settings.APP_NAME == test_app_name
        assert settings.APP_VERSION == test_app_version
        assert settings.DEBUG is True
    
def test_llm_provider_config_defaults():
    
    config = LLMProviderConfig()
    
    assert config.api_key == ""
    assert config.api_base is None
    assert config.timeout == 60.0
    assert config.max_retries == 3
    assert config.connection_pool_size == 10
    
def test_llm_provider_config_custom():
    
    custom_api_key = "test_api_key"
    custom_api_base = "https://api.test.com"
    
    config = LLMProviderConfig(
        api_key=custom_api_key,
        api_base=custom_api_base,
    )
    
    assert config.api_key == custom_api_key
    assert config.api_base == custom_api_base
    
def test_parse_enabled_models_comma_separated():
    
    test_models = ["model1", "model2", "model3"]
    
    with patch.dict(os.environ, {
        "ENABLED_MODELS_SET": "model1,model2,model3"
    }):
        settings = Settings()
        
        for model in test_models:
            assert model in settings.ENABLED_MODELS_SET
            
def test_parse_enabled_models_empty():
    
    with patch.dict(os.environ, {
        "ENABLED_MODELS_SET": ""
    }):
        settings = Settings()
        
        assert len(settings.ENABLED_MODELS_SET) == 0
        assert settings.ENABLED_MODELS_SET == set()

def test_parse_provider_configs_with_env_vars():
    test_api_key = "test_api_key"
    
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": test_api_key,
        "ANTHROPIC_API_KEY": test_api_key
    }):
        settings = Settings()
        
        assert settings.LLM_PROVIDERS_CONFIG["openai"]["api_key"] == test_api_key
        assert settings.LLM_PROVIDERS_CONFIG["anthropic"]["api_key"] == test_api_key

        
def test_get_settings_cache():
    
    settings1 = get_settings()
    settings2 = get_settings()
    
    assert settings1 is settings2

def test_invalid_environment():
    
    with patch.dict(os.environ, {
        "ENVIRONMENT": "invalid_env"
    }):
        with pytest.raises(ValidationError) as excinfo:
            settings = Settings()

def test_worker_count():
    settings = Settings()
        
    assert settings.WORKER_COUNT >= 1
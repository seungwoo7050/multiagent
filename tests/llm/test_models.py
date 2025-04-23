import pytest
from unittest.mock import patch, MagicMock

# 테스트 대상 함수 및 객체 임포트
from src.llm.models import (
    register_model,
    get_model_info,
    get_token_limit,
    model_supports_feature,
    model_has_capability,
    list_available_models,
    find_alternative_model,
    _MODEL_REGISTRY
)

# --- Fixtures ---

# 가짜 Settings 객체를 생성하는 pytest fixture
@pytest.fixture(scope="module")
def mock_settings(): # <- Fixture 이름
    settings = MagicMock()
    settings.LLM_MODEL_PROVIDER_MAP = {}
    settings.ENABLED_MODELS_SET = {"gpt-3.5-turbo", "gpt-4o"}
    return settings

# 원본 레지스트리 저장 및 복원을 위한 fixture
@pytest.fixture(scope="module")
def original_registry():
    original = _MODEL_REGISTRY.copy()
    yield
    _MODEL_REGISTRY.clear()
    _MODEL_REGISTRY.update(original)


# --- Test Functions ---

# patch 데코레이터는 첫 번째 인자로 mock 객체를 전달 (mock_settings_obj)
# pytest fixture는 이름(mock_settings)으로 인자를 전달받음
@patch('src.llm.models.settings')
def test_get_model_info_existing_model(mock_settings_obj, mock_settings): # <- 인자 이름 수정
    """Test retrieving info for an existing model."""
    mock_settings_obj.return_value = mock_settings # patch된 객체가 fixture 값을 사용하도록 설정

    model_info = get_model_info("gpt-4o")
    assert model_info is not None
    assert model_info["provider"] == "openai"
    assert model_info["token_limit"] > 0
    assert isinstance(model_info["capabilities"], list)
    assert "general" in model_info["capabilities"]


@patch('src.llm.models.settings')
def test_get_model_info_nonexistent_model(mock_settings_obj, mock_settings): # <- 인자 이름 수정
    """Test retrieving info for a nonexistent model with mocked settings."""
    mock_settings_obj.return_value = mock_settings

    model_info = get_model_info("nonexistent-model")
    assert isinstance(model_info, dict)
    assert len(model_info) == 0


@patch('src.llm.models.settings')
def test_get_token_limit(mock_settings_obj, mock_settings): # <- 인자 이름 수정
    """Test getting token limit for various models with mocked settings."""
    mock_settings_obj.return_value = mock_settings

    gpt4_limit = get_token_limit("gpt-4")
    assert gpt4_limit == 8192

    gpt4_variant_limit = get_token_limit("gpt-4-0613")
    assert gpt4_variant_limit == 8192

    unknown_limit = get_token_limit("unknown-model")
    assert unknown_limit == 4096


@patch('src.llm.models.settings')
def test_model_capabilities(mock_settings_obj, mock_settings, original_registry): # <- 인자 이름 수정
    """Test checking model capabilities."""
    mock_settings_obj.return_value = mock_settings

    assert model_has_capability("gpt-4o", "vision")
    assert not model_has_capability("gpt-4o", "nonexistent_capability")

    register_model(
        model_name="test-model-caps", provider="test", token_limit=1000,
        capabilities=["special_capability"], description="Test model"
    )

    assert model_has_capability("test-model-caps", "special_capability")
    assert not model_has_capability("test-model-caps", "other_capability")


@patch('src.llm.models.settings')
def test_model_features(mock_settings_obj, mock_settings, original_registry): # <- 인자 이름 수정
    """Test checking model feature support."""
    mock_settings_obj.return_value = mock_settings

    assert model_supports_feature("gpt-4o", "streaming")

    register_model(
        model_name="test-model-features", provider="test", token_limit=1000,
        capabilities=["general"], description="Test model",
        supports_streaming=True, supports_function_calling=False
    )

    assert model_supports_feature("test-model-features", "streaming")
    assert not model_supports_feature("test-model-features", "function_calling")


@patch('src.llm.models.settings')
def test_register_model(mock_settings_obj, mock_settings, original_registry): # <- 인자 이름 수정
    """Test registering a new model."""
    mock_settings_obj.return_value = mock_settings

    register_model(
        model_name="test-model", provider="test-provider", token_limit=5000,
        capabilities=["test", "general"], description="Test model description",
        supports_streaming=True, supports_function_calling=True
    )

    model_info = get_model_info("test-model")
    assert model_info["provider"] == "test-provider"
    assert model_info["token_limit"] == 5000
    assert "test" in model_info["capabilities"]
    assert model_info["supports_streaming"] is True
    assert model_info["supports_function_calling"] is True
    assert model_info["description"] == "Test model description"


@patch('src.llm.models.settings')
def test_list_available_models(mock_settings_obj, mock_settings, original_registry): # <- 인자 이름 수정
    """Test listing available models with filters."""
    mock_settings_obj.return_value = mock_settings

    register_model(
        model_name="big-model", provider="test", token_limit=100000,
        capabilities=["general", "coding"], description="Big model"
    )
    register_model(
        model_name="small-model", provider="test", token_limit=4000,
        capabilities=["general"], description="Small model"
    )

    test_models = list_available_models(provider="test")
    assert "big-model" in test_models
    assert "small-model" in test_models

    large_models = list_available_models(min_token_limit=50000)
    assert "big-model" in large_models
    assert "small-model" not in large_models

    coding_models = list_available_models(required_capabilities=["coding"])
    assert "big-model" in coding_models
    assert "small-model" not in coding_models


@patch('src.llm.models.settings')
def test_find_alternative_model(mock_settings_obj, mock_settings, original_registry): # <- 인자 이름 수정
    """Test finding alternative models."""
    mock_settings_obj.return_value = mock_settings

    register_model(
        model_name="primary-model", provider="test", token_limit=8000,
        capabilities=["general", "coding", "reasoning"], description="Primary test model"
    )
    register_model(
        model_name="similar-model", provider="test", token_limit=10000,
        capabilities=["general", "coding"], description="Similar test model"
    )
    register_model(
        model_name="different-model", provider="other", token_limit=8000,
        capabilities=["general"], description="Different provider model"
    )

    alt = find_alternative_model("primary-model", same_provider=True)
    assert alt == "similar-model"

    alt_any_provider = find_alternative_model("primary-model", same_provider=False)
    assert alt_any_provider in ["similar-model", "different-model"]

    alt_unknown = find_alternative_model("nonexistent-model")
    assert alt_unknown is None
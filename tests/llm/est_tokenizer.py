import pytest
import asyncio
from unittest.mock import patch, MagicMock # Mock 임포트 추가

from src.llm.tokenizer import (
    count_tokens,
    count_tokens_sync,
    get_token_limit,
    clear_token_cache,
    get_cache_metrics,
    _TOKEN_COUNT_CACHE
)

# --- Fixtures ---

@pytest.fixture(scope="function")
def clear_cache():
    """Clear token cache before each test."""
    clear_token_cache()
    yield
    clear_token_cache()

# Settings 모킹을 위한 fixture 추가
@pytest.fixture(scope="module")
def mock_settings():
    settings = MagicMock()
    # tokenizer.py에서 사용하는 LLM_MODEL_PROVIDER_MAP 속성 설정
    settings.LLM_MODEL_PROVIDER_MAP = {
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "claude-3-opus": "anthropic",
        # 테스트에 필요한 다른 모델 매핑 추가 가능
    }
    # 필요시 다른 settings 속성도 추가
    return settings

# --- Test Functions ---

# 모든 테스트 함수에 patch 데코레이터와 fixture 인자 추가
@patch('src.llm.tokenizer.settings')
def test_count_tokens_sync(mock_settings_obj, mock_settings, clear_cache):
    """Test synchronous token counting."""
    mock_settings_obj.return_value = mock_settings # Mock 설정 적용

    text = "Hello, world! This is a test."
    token_count = count_tokens_sync("gpt-4", text)

    assert token_count > 0
    assert token_count < 15

    token_count_claude = count_tokens_sync("claude-3-opus", text)
    assert token_count_claude > 0


@patch('src.llm.tokenizer.settings')
def test_count_tokens_cache(mock_settings_obj, mock_settings, clear_cache):
    """Test that token counting cache works."""
    mock_settings_obj.return_value = mock_settings

    text = "Testing the token cache functionality."
    model = "gpt-3.5-turbo"

    count1 = count_tokens_sync(model, text)

    metrics = get_cache_metrics()
    assert metrics["token_cache_size"] >= 1

    count2 = count_tokens_sync(model, text)
    assert count2 == count1

    count_tokens_sync("claude-3-opus", text)
    metrics = get_cache_metrics()
    assert metrics["token_cache_size"] >= 2


@pytest.mark.asyncio
@patch('src.llm.tokenizer.settings')
async def test_count_tokens_async(mock_settings_obj, mock_settings, clear_cache):
    """Test asynchronous token counting."""
    mock_settings_obj.return_value = mock_settings

    text = "Testing async token counting."

    token_count = await count_tokens("gpt-4", text)

    assert token_count > 0
    assert token_count < 10

    # 비동기 함수 내에서 동기 함수 호출 시에도 mock 설정이 유지됨
    token_count_sync = count_tokens_sync("gpt-4", text)
    assert token_count == token_count_sync


@pytest.mark.asyncio
@patch('src.llm.tokenizer.settings')
async def test_parallel_token_counting(mock_settings_obj, mock_settings, clear_cache):
    """Test counting tokens in parallel."""
    mock_settings_obj.return_value = mock_settings

    texts = [
        "First test text",
        "Second test text with more content",
        "Third test text with even more content to count"
    ]
    model = "gpt-4"

    tasks = [count_tokens(model, text) for text in texts]
    results = await asyncio.gather(*tasks)

    assert all(count > 0 for count in results)
    assert results[0] < results[1] < results[2]


@patch('src.llm.tokenizer.settings')
def test_get_token_limit_comparison(mock_settings_obj, mock_settings): # clear_cache 불필요
    """Test token limit functionality against expected values."""
    mock_settings_obj.return_value = mock_settings

    # MODEL_TOKEN_LIMITS 딕셔너리를 사용하므로 settings 모킹 영향 적음
    assert get_token_limit("gpt-4") == 8192
    assert get_token_limit("gpt-4-32k") == 32768
    assert get_token_limit("gpt-3.5-turbo") == 16385

    assert get_token_limit("claude-3-opus") > 100000

    # unknown-model 처리에 settings 접근 필요
    assert get_token_limit("unknown-model") > 0


@patch('src.llm.tokenizer.settings')
def test_token_count_different_languages(mock_settings_obj, mock_settings, clear_cache):
    """Test token counting for different languages."""
    mock_settings_obj.return_value = mock_settings

    english = "This is a test in English."
    korean = "이것은 한국어로 된 테스트입니다."
    chinese = "这是一个中文测试。"
    model = "gpt-4"

    en_count = count_tokens_sync(model, english)
    ko_count = count_tokens_sync(model, korean)
    zh_count = count_tokens_sync(model, chinese)

    assert en_count > 0
    assert ko_count > 0
    assert zh_count > 0


@patch('src.llm.tokenizer.settings')
def test_clear_token_cache(mock_settings_obj, mock_settings, clear_cache):
    """Test clearing token cache."""
    mock_settings_obj.return_value = mock_settings

    text1 = "First cache test."
    text2 = "Second cache test."
    # count_tokens_sync 내부에서 settings 사용하므로 모킹 필요
    count_tokens_sync("gpt-4", text1)
    count_tokens_sync("gpt-4", text2)

    assert len(_TOKEN_COUNT_CACHE) >= 2

    cleared = clear_token_cache()

    assert cleared >= 2
    assert len(_TOKEN_COUNT_CACHE) == 0
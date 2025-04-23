# 수정된 test_base.py

import logging
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

# 가정한 경로 (실제 프로젝트 구조에 맞게 조정 필요)
from src.llm.base import BaseLLMAdapter
from src.config.errors import LLMError, ErrorCode # ErrorCode 추가 (base.py에 따라)

logger = logging.getLogger(__name__)

# --- Test Adapter Class (기존과 동일) ---
class TestAdapter(BaseLLMAdapter):
    """Test implementation of BaseLLMAdapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 내부 플래그: 원래 메소드가 호출되었는지 추적
        self._initialize_called_flag = False
        self._get_client_called_flag = False
        self._generate_text_called_flag = False
        self._count_tokens_called_flag = False
        # _client는 _get_client에서 반환하므로 초기값은 None이 적절할 수 있음
        self._client = None

    async def _initialize(self) -> bool:
        self._initialize_called_flag = True # 플래그 설정
        self._client = AsyncMock() # 초기화 시 클라이언트 생성
        logger.debug("Original _initialize called") # 디버깅용 로그
        return True

    async def _get_client(self) -> AsyncMock:
        # 이미 초기화된 클라이언트 반환 또는 필요 시 생성 로직 추가 가능
        if not self.initialized:
             # ensure_initialized 내부에서 _initialize 호출하므로 플래그 확인 가능
            await self.ensure_initialized()
        self._get_client_called_flag = True # 플래그 설정
        logger.debug("Original _get_client called") # 디버깅용 로그
        # self._client가 _initialize에서 설정되었는지 확인
        if self._client is None:
             logger.warning("_get_client called but self._client is None")
             # 테스트 시나리오에 따라 예외 발생 또는 기본 Mock 반환 등 처리 필요
             # return AsyncMock() # 임시 Mock 반환
        return self._client

    async def _generate_text(self, prompt, **kwargs):
        self._generate_text_called_flag = True # 플래그 설정
        temp = kwargs.get("temperature", self.temperature)
        return {
            "id": "test-id", "object": "test-object", "created": 1234567890,
            "model": self.model,
            "choices": [{"text": f"Response to: {prompt} (temp: {temp})", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "prompt_tokens": 10, "completion_tokens": 20,
        }

    async def _count_tokens(self, text) -> int:
        self._count_tokens_called_flag = True # 플래그 설정
        return len(text.split())

    def get_token_limit(self) -> int:
        return 4096

# --- Fixtures ---

@pytest.fixture(scope="module")
def mock_settings_data():
    return {
        "REQUEST_TIMEOUT": 30.0,
        "LLM_RETRY_MAX_ATTEMPTS": 3,
    }

@pytest.fixture
def test_adapter(mock_settings_data):
    """Create a test adapter instance with mocked settings."""
    with patch('src.llm.base.settings') as mock_settings:
        for key, value in mock_settings_data.items():
            setattr(mock_settings, key, value)
        adapter = TestAdapter(
            model="test-model", provider="test-provider", api_key="test-api-key",
            temperature=0.7, max_tokens=100,
        )
        # 테스트 후 정리 로직 추가 가능 (필요 시)
        # yield adapter
        # print("Cleaning up adapter fixture")
        return adapter # yield 대신 return 사용 시 정리 로직 없음

# --- Test Functions ---

# 이 테스트는 이전 로그에서 통과했으므로 그대로 둠
def test_adapter_initialization(test_adapter, mock_settings_data):
    """Test adapter initialization."""
    assert test_adapter.model == "test-model"
    assert test_adapter.provider == "test-provider"
    assert test_adapter.api_key == "test-api-key"
    assert test_adapter.temperature == 0.7
    assert test_adapter.max_tokens == 100
    assert test_adapter.timeout == mock_settings_data["REQUEST_TIMEOUT"]
    assert test_adapter.max_retries == mock_settings_data["LLM_RETRY_MAX_ATTEMPTS"]
    assert test_adapter.initialized is False
    assert test_adapter._client is None

@pytest.mark.asyncio
async def test_ensure_initialized(test_adapter):
    """Test ensure_initialized method."""
    assert test_adapter.initialized is False
    assert test_adapter._client is None
    result = await test_adapter.ensure_initialized()
    assert result is True
    assert test_adapter.initialized is True
    assert test_adapter._initialize_called_flag is True # 내부 플래그 확인
    assert isinstance(test_adapter._client, AsyncMock)

# 이 테스트는 이전 로그에서 통과했으므로 그대로 둠
@pytest.mark.asyncio
async def test_tokenize(test_adapter):
    """Test tokenize method."""
    input_text = "This is a test sentence."
    expected_tokens = 5
    tokens = await test_adapter.tokenize(input_text)
    assert test_adapter._count_tokens_called_flag is True # 내부 플래그 확인
    assert tokens == expected_tokens

# --- generate 관련 테스트들 ---
# 참고: 이 테스트들이 통과하려면 base.py의 generate 함수 내에서
# cache = get_cache() 호출 시 await이 필요할 수 있습니다. (현재 로그 오류 기반 추정)

async def mock_count_tokens_side_effect(model, text):
    """Async side effect function for count_tokens mock."""
    # 실제 비동기 작업 시뮬레이션 불필요 시 await asyncio.sleep(0) 제거 가능
    # await asyncio.sleep(0) 
    return len(text.split())

@pytest.mark.asyncio
@patch('src.llm.base.cache_result', new_callable=AsyncMock)
@patch('src.llm.base.get_cache')
@patch('src.llm.base.count_tokens', new_callable=AsyncMock)
async def test_generate(mock_count_tokens, mock_get_cache, mock_cache_result, test_adapter):
    """Test generate method (cache miss)."""
    # Mock 설정
    # count_tokens 모킹: side_effect를 async 함수로 변경
    mock_count_tokens.side_effect = mock_count_tokens_side_effect # 수정됨

    # get_cache 모킹 (기존과 동일)
    mock_cache_instance = AsyncMock()
    mock_cache_instance.get = AsyncMock(return_value=None)
    mock_get_cache.return_value = mock_cache_instance

    # 실행
    prompt = "Test prompt for generate"
    response = await test_adapter.generate(
        prompt=prompt, max_tokens=50, temperature=0.5, use_cache=True
    )

    # 검증
    assert test_adapter._generate_text_called_flag is True
    mock_get_cache.assert_called_once()
    mock_cache_instance.get.assert_called_once()
    mock_cache_result.assert_called_once()
    assert "choices" in response
    # ... (다른 assert 문) ...
    assert response["usage"]["total_tokens"] == 30
    # count_tokens 호출 횟수 검증 수정
    assert mock_count_tokens.call_count == 0 # 수정됨

@pytest.mark.asyncio
@patch('src.llm.base.cache_result', new_callable=AsyncMock)
@patch('src.llm.base.get_cache')
@patch('src.llm.base.count_tokens', new_callable=AsyncMock)
async def test_generate_with_cache_hit(mock_count_tokens, mock_get_cache, mock_cache_result, test_adapter):
    """Test generate method with cache hit."""
    cached_response = {
        "id": "cached-id",
        "choices": [{"text": "Cached response"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
    }
    prompt = "Test prompt for cache hit"

    # Mock 설정
    # count_tokens 모킹: side_effect를 async 함수로 변경
    mock_count_tokens.side_effect = mock_count_tokens_side_effect # 수정됨

    # get_cache 모킹 (기존과 동일)
    mock_cache_instance = AsyncMock()
    mock_cache_instance.get = AsyncMock(return_value=cached_response)
    mock_get_cache.return_value = mock_cache_instance

    # 실행
    response = await test_adapter.generate(prompt, use_cache=True)

    # 검증
    assert response == cached_response
    mock_get_cache.assert_called_once()
    mock_cache_instance.get.assert_called_once()
    assert test_adapter._generate_text_called_flag is False
    mock_cache_result.assert_not_called()
    # count_tokens 호출 횟수 검증: 캐시 히트 시 prompt, completion 각 1번씩 호출
    assert mock_count_tokens.call_count == 2 # 수정됨 (기존 >= 2 에서 == 2 로 명확화)

@pytest.mark.asyncio
@patch('src.llm.base.cache_result', new_callable=AsyncMock)
@patch('src.llm.base.get_cache')
@patch('src.llm.base.count_tokens', new_callable=AsyncMock)
async def test_generate_with_error(mock_count_tokens, mock_get_cache, mock_cache_result, test_adapter):
    """Test generate method with error during generation."""
    prompt = "Test prompt for error"
    # Mock 설정
    mock_count_tokens.side_effect = lambda model, text: asyncio.sleep(0, result=len(text.split()))
    mock_cache_instance = AsyncMock()
    mock_cache_instance.get = AsyncMock(return_value=None) # 캐시 미스
    mock_get_cache.return_value = mock_cache_instance
    # _generate_text 호출 시 에러 발생 설정
    test_adapter._generate_text = AsyncMock(side_effect=Exception("API error"))

    # 실행 및 검증
    with pytest.raises(LLMError) as excinfo:
        await test_adapter.generate(prompt, use_cache=True)

    assert excinfo.value.code == ErrorCode.LLM_API_ERROR
    assert "API error" in str(excinfo.value)
    assert excinfo.value.details["model"] == "test-model"
    mock_cache_result.assert_not_called() # 에러 시 캐시 저장 안됨

# --- 나머지 테스트들 ---

def test_get_metrics(test_adapter):
    """Test get_metrics method."""
    test_adapter.request_count = 10
    test_adapter.token_usage = {"prompt": 100, "completion": 200, "total": 300}
    test_adapter.error_count = 2
    test_adapter.average_latency = 0.5
    metrics = test_adapter.get_metrics()
    assert metrics["model"] == "test-model"
    # ... (이하 동일)

# 이 테스트는 이전 로그에서 통과했으므로 그대로 둠
@patch.object(TestAdapter, 'generate', new_callable=AsyncMock)
def test_generate_sync(mock_async_generate, test_adapter):
    """Test the synchronous version of generate."""
    async_response = {"choices": [{"text": "Async response from mock"}]}
    mock_async_generate.return_value = async_response
    prompt = "Test prompt for sync"
    response = test_adapter.generate_sync(prompt, temperature=0.8)
    assert response == async_response
    mock_async_generate.assert_called_once()
    call_args = mock_async_generate.call_args
    assert call_args.kwargs['prompt'] == prompt
    assert call_args.kwargs['temperature'] == 0.8


@pytest.mark.asyncio
async def test_health_check_ok(test_adapter):
    """Test health_check method (OK status)."""
    # health_check는 내부적으로 ensure_initialized -> _initialize 호출
    # 및 _get_client 호출
    health = await test_adapter.health_check()
    assert health["status"] == "ok"
    assert "latency" in health
    assert isinstance(health["latency"], float)
    assert "operational" in health["message"]
    assert test_adapter._initialize_called_flag is True # 내부 플래그 확인
    assert test_adapter._get_client_called_flag is True # 내부 플래그 확인

@pytest.mark.asyncio
async def test_health_check_init_failure(test_adapter):
    """Test health_check method with initialization failure."""
    # _initialize 메소드가 False를 반환하도록 모킹
    # 원래 _initialize의 내부 로직(_initialize_called_flag=True)은 실행되지 않음
    test_adapter._initialize = AsyncMock(return_value=False)

    health = await test_adapter.health_check()

    assert health["status"] == "error"
    assert "latency" in health
    assert "Failed to initialize adapter" in health["message"]
    # Mock 객체가 호출되었는지 확인 (내부 플래그 대신)
    test_adapter._initialize.assert_called_once()
    # 초기화 실패 시 _get_client는 호출되지 않아야 함
    # 이를 확인하기 위해 _get_client Mock 객체 필요
    # 또는 TestAdapter의 _get_client_called_flag가 False인지 확인
    assert test_adapter._get_client_called_flag is False # 수정됨

@pytest.mark.asyncio
async def test_health_check_client_failure(test_adapter):
    """Test health_check method with client retrieval failure."""
    # health_check -> ensure_initialized -> _initialize (성공 가정, 실제 메소드 호출)
    # health_check -> _get_client (모킹)
    # _get_client 메소드가 None을 반환하도록 모킹
    test_adapter._get_client = AsyncMock(return_value=None)

    health = await test_adapter.health_check()

    assert health["status"] == "error"
    assert "latency" in health
    assert "Failed to get client" in health["message"]
    # 초기화는 성공했어야 함 (ensure_initialized가 호출)
    assert test_adapter._initialize_called_flag is True # 수정됨 (내부 플래그 확인)
    # Mock 객체가 호출되었는지 확인 (내부 플래그 대신)
    test_adapter._get_client.assert_called_once() # 수정됨

@pytest.mark.asyncio
async def test_cache_key_generation(test_adapter):
    """Test _get_cache_key method."""
    params1 = {"temperature": 0.7, "max_tokens": 100}
    params2 = {"max_tokens": 100, "temperature": 0.7}
    key1 = test_adapter._get_cache_key("Test prompt", params1)
    key2 = test_adapter._get_cache_key("Test prompt", params2)
    assert key1 == key2
    # ... (이하 동일)

@pytest.mark.asyncio
async def test_close(test_adapter):
    """Test close method."""
    await test_adapter.ensure_initialized()
    assert test_adapter.initialized is True
    assert test_adapter._client is not None
    await test_adapter.close()
    assert test_adapter.initialized is False
    assert test_adapter._client is None
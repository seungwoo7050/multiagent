# test_openai.py (수정됨)
import pytest
import json
import aiohttp
from unittest.mock import patch, AsyncMock, MagicMock

# 필요한 모듈과 클래스 import
from src.llm.adapters.openai import OpenAIAdapter
from src.config.errors import LLMError, ErrorCode

# --- Settings Mocking Fixture ---
@pytest.fixture(autouse=True)
def mock_settings_openai(monkeypatch):
    """
    Automatically mock settings for all tests in this module.
    Patches the 'settings' object directly within the modules that use it.
    """
    mock_settings_obj = MagicMock()
    # 필요한 설정 값들을 mock 객체에 설정합니다.
    mock_settings_obj.REQUEST_TIMEOUT = 30.0
    mock_settings_obj.LLM_PROVIDERS_CONFIG = {
        "openai": {
            "api_key": "mock-api-key-from-settings", # 설정 파일에서 가져올 기본 API 키
            "connection_pool_size": 10,
            "timeout": 30.0, # 기본 타임아웃 설정
        },
    }
    mock_settings_obj.LLM_CACHE_ENABLED = False # 테스트 시 캐시 비활성화 (선택 사항)
    mock_settings_obj.LLM_MODEL_PROVIDER_MAP = { # get_tokenizer 테스트를 위해 추가
         "gpt-4": "openai",
         "gpt-3.5-turbo": "openai",
     }

    # settings 객체를 사용하는 모듈 경로 목록
    modules_to_patch = [
        "src.llm.base",
        "src.llm.adapters.openai",
        "src.llm.connection_pool",
        "src.llm.tokenizer", # get_tokenizer 테스트를 위해 추가
        # settings를 사용하는 다른 모듈이 있다면 추가
    ]
    for module_path in modules_to_patch:
        # raising=False를 사용하여 해당 모듈에 settings가 없어도 에러 발생 안 함
        monkeypatch.setattr(f"{module_path}.settings", mock_settings_obj, raising=False)

    # LLM_PROVIDERS_CONFIG 수정 후 사용할 수 있도록 mock 객체 반환 (test_initialize_no_api_key 수정)
    return mock_settings_obj


# --- Fixtures ---
@pytest.fixture
def openai_adapter():
    """Create an OpenAI adapter instance for testing."""
    # 테스트용 Adapter 인스턴스 생성, 필요시 기본값 덮어쓰기 가능
    return OpenAIAdapter(
        model="gpt-4", # 테스트에 사용할 모델
        api_key="test-api-key", # 테스트용 API 키 (settings보다 우선)
        temperature=0.7,
        max_tokens=100,
        timeout=20.0 # 테스트용 타임아웃 (settings보다 우선)
    )

@pytest.fixture
def mock_session():
    """Create a mock aiohttp ClientSession."""
    # aiohttp.ClientSession의 mock 객체 생성
    mock = AsyncMock(spec=aiohttp.ClientSession)
    # post 메서드는 각 테스트에서 구체적으로 모킹합니다.
    return mock

# test_openai.py

@pytest.fixture
def mock_response():
    """Create a mock API success response (now defaults to chat format for gpt-4)."""
    mock = AsyncMock(spec=aiohttp.ClientResponse)
    mock.status = 200
    # 기본적으로 chat completion 형식의 응답을 반환하도록 수정
    mock.json = AsyncMock(return_value={
        "id": "test-chat-comp-response-id", # ID 변경하여 구분
        "object": "chat.completion", # object 타입 변경
        "created": 1234567890,
        "model": "gpt-4", # 응답에 포함될 모델명
        # "choices" 구조를 chat 형식으로 변경
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "This is a test chat completion response" # content 키 사용
            },
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
    return mock

# 만약 기존 completion 형식 테스트가 필요하다면 별도의 fixture를 만들거나
# mock_response fixture를 파라미터화하여 두 가지 형식을 모두 지원할 수 있습니다.
# 예시: Completion 형식 fixture 추가
@pytest.fixture
def mock_completion_response():
    """Create a basic mock API success response (completion)."""
    mock = AsyncMock(spec=aiohttp.ClientResponse)
    mock.status = 200
    mock.json = AsyncMock(return_value={
        "id": "test-completion-response-id",
        "object": "text_completion",
        "created": 1234567890,
        "model": "text-davinci-003", # 예시: completion 모델
        "choices": [{"text": "This is a test completion response", "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
    return mock

@pytest.fixture
def mock_chat_response():
    """Create a basic mock API success response (chat)."""
    # 성공적인 API 응답(chat)의 mock 객체 생성
    mock = AsyncMock(spec=aiohttp.ClientResponse)
    mock.status = 200
    mock.json = AsyncMock(return_value={
        "id": "test-chat-response-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-3.5-turbo", # 응답에 포함될 모델명
        "choices": [{"message": {"role": "assistant", "content": "This is a test chat response"}, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 7, "total_tokens": 22}
    })
    return mock

@pytest.fixture
def mock_error_response():
    """Create a mock API error response."""
    # API 에러 응답의 mock 객체 생성
    mock = AsyncMock(spec=aiohttp.ClientResponse)
    mock.status = 400 # 예: 잘못된 요청 상태 코드
    mock.json = AsyncMock(return_value={
        "error": {"message": "Invalid request simulated", "type": "invalid_request_error"}
    })
    return mock

# --- Test Functions ---

def test_adapter_initialization(openai_adapter):
    """Test OpenAI adapter initialization attributes."""
    assert openai_adapter.model == "gpt-4"
    assert openai_adapter.provider == "openai"
    assert openai_adapter.api_key == "test-api-key"
    assert openai_adapter.temperature == 0.7
    assert openai_adapter.max_tokens == 100
    assert openai_adapter.api_base == "https://api.openai.com/v1"
    assert openai_adapter.timeout == 20.0 # 생성자에서 설정한 값 확인
    assert openai_adapter.initialized is False # 초기 상태는 False

@pytest.mark.asyncio
async def test_initialize_success(openai_adapter):
    """Test successful adapter initialization process."""
    # _get_tokenizer와 _get_client가 성공적으로 호출되는 경우를 가정
    # _get_tokenizer는 비동기가 아니므로 MagicMock 사용 가능
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1] # encode 메서드 모킹

    with patch.object(openai_adapter, '_get_tokenizer', return_value=mock_tokenizer) as mock_get_tokenizer, \
         patch.object(openai_adapter, '_get_client', return_value=AsyncMock()) as mock_get_client:
        # ensure_initialized는 내부적으로 _initialize를 호출
        await openai_adapter.ensure_initialized()
        assert openai_adapter.initialized is True
        mock_get_tokenizer.assert_called_once()
        mock_get_client.assert_awaited_once() # _get_client는 async 함수이므로 await 확인

@pytest.mark.asyncio
async def test_initialize_no_api_key(mock_settings_openai): # openai_adapter 제거, mock_settings_openai 직접 사용
    """Test initialization failure when no API key is available."""
    # Adapter 생성 시 API 키를 제공하지 않음
    adapter_no_key = OpenAIAdapter(model="gpt-4", api_key=None)
    # 설정 mock에서 API 키를 제거
    mock_settings_openai.LLM_PROVIDERS_CONFIG["openai"]["api_key"] = None

    # 토크나이저 mock 생성
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1]

    with patch.object(adapter_no_key, '_get_tokenizer', return_value=mock_tokenizer), \
         patch.object(adapter_no_key, '_get_client', return_value=AsyncMock()):
        with pytest.raises(LLMError) as excinfo:
            await adapter_no_key.ensure_initialized() # _initialize 호출 시 에러 발생 기대
        assert excinfo.value.code == ErrorCode.LLM_PROVIDER_ERROR
        assert "No API key provided" in excinfo.value.message
        assert adapter_no_key.initialized is False

@pytest.mark.asyncio
async def test_get_client(openai_adapter):
    """Test getting the client session from the connection pool."""
    # connection_pool.get_connection_pool 함수를 모킹
    mock_pooled_session = AsyncMock(spec=aiohttp.ClientSession)
    # get_connection_pool 자체가 async 함수이므로, AsyncMock을 사용하여 await 결과 모킹
    mock_get_pool = AsyncMock(return_value=mock_pooled_session)

    # openai.py에서 get_connection_pool을 import하는 경로를 patch
    with patch('src.llm.adapters.openai.get_connection_pool', mock_get_pool) as patched_get_pool:
        client = await openai_adapter._get_client()
        # get_connection_pool('openai')가 await 되었는지 확인
        patched_get_pool.assert_awaited_once_with("openai")
        # 반환된 클라이언트가 mock 객체인지 확인
        assert client is mock_pooled_session

@pytest.mark.asyncio
async def test_count_tokens(openai_adapter):
    """Test token counting functionality."""
    # 토크나이저 mock 생성
    mock_tokenizer = MagicMock()
    # encode 메서드가 호출되면 특정 토큰 리스트를 반환하도록 설정
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

    # _get_tokenizer가 mock_tokenizer를 반환하도록 설정
    with patch.object(openai_adapter, '_get_tokenizer', return_value=mock_tokenizer):
        # 초기화 보장 (내부적으로 _get_tokenizer 호출)
        # ensure_initialized는 _get_client도 호출하므로 같이 모킹
        with patch.object(openai_adapter, '_get_client', return_value=AsyncMock()):
            await openai_adapter.ensure_initialized()

        # 토큰 카운트 실행
        count = await openai_adapter._count_tokens("Sample text")
        assert count == 5 # encode 결과 리스트의 길이와 일치해야 함
        mock_tokenizer.encode.assert_called_once_with("Sample text")

# --- _generate_text Tests ---

@pytest.mark.asyncio
async def test_generate_text_completion(openai_adapter, mock_session, mock_response):
    """Test generating text using the completion API endpoint (treated as chat for gpt-4)."""
    # --- session.post 모킹 설정 ---
    mock_post_context = AsyncMock()
    mock_post_context.__aenter__.return_value = mock_response
    mock_post_context.__aexit__ = AsyncMock(return_value=None)
    mock_session.post.return_value = mock_post_context
    # ----------------------------

    # _get_client, _count_tokens, ensure_initialized 모킹
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1] * 10 # 10 토큰으로 모킹

    with patch.object(openai_adapter, '_get_client', return_value=mock_session), \
         patch.object(openai_adapter, '_get_tokenizer', return_value=mock_tokenizer), \
         patch.object(openai_adapter, '_count_tokens', return_value=10) as mock_count, \
         patch.object(openai_adapter, 'ensure_initialized', AsyncMock(return_value=True)):

        # _initialize가 호출되었고 성공했다고 가정
        openai_adapter.initialized = True
        openai_adapter.tokenizer = mock_tokenizer # 토크나이저 설정

        # _generate_text 호출
        response = await openai_adapter._generate_text(
            prompt="Test completion prompt",
            max_tokens=50,
            temperature=0.5,
            stop_sequences=["\n"]
        )

        # 검증
        mock_session.post.assert_called_once()
        call_args, call_kwargs = mock_session.post.call_args
        # 호출된 URL 확인 (gpt-4는 chat으로 처리됨)
        assert call_args[0] == "https://api.openai.com/v1/chat/completions"
        # 요청 payload 확인
        payload = call_kwargs["json"]
        assert payload["model"] == "gpt-4"
        # chat 모델 형식으로 변환되었는지 확인
        assert "prompt" not in payload
        assert payload["messages"] == [{"role": "user", "content": "Test completion prompt"}]
        assert payload["max_tokens"] == 50
        assert payload["temperature"] == 0.5
        assert payload["stop"] == ["\n"]

        # --- 응답 내용 확인 (수정된 부분) ---
        assert response["id"] == "test-chat-comp-response-id" # <<< 기대하는 ID 값 수정
        assert len(response["choices"]) == 1
        # 어댑터가 내부적으로 message.content를 text로 변환하므로 'text' 키 확인
        assert response["choices"][0]["text"] == "This is a test chat completion response"
        # prompt_tokens는 _count_tokens 모킹 값, 나머지는 mock_response의 값 사용
        assert response["usage"]["prompt_tokens"] == 10
        assert response["usage"]["completion_tokens"] == 5
        assert response["usage"]["total_tokens"] == 15
        assert "request_time" in response


@pytest.mark.asyncio
async def test_generate_text_chat_model(openai_adapter, mock_session, mock_chat_response):
    """Test generating text using the chat API endpoint."""
    # 테스트를 위해 어댑터 모델을 chat 모델로 변경
    openai_adapter.model = "gpt-3.5-turbo"

    # --- session.post 모킹 설정 ---
    mock_post_context = AsyncMock()
    mock_post_context.__aenter__.return_value = mock_chat_response
    mock_post_context.__aexit__ = AsyncMock(return_value=None)
    # 수정: post 호출 시 context manager mock 반환
    mock_session.post.return_value = mock_post_context
    # ----------------------------

    # 토크나이저 모킹 추가
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1] * 15 # 15 토큰으로 모킹

    with patch.object(openai_adapter, '_get_client', return_value=mock_session), \
         patch.object(openai_adapter, '_get_tokenizer', return_value=mock_tokenizer), \
         patch.object(openai_adapter, '_count_tokens', return_value=15) as mock_count, \
         patch.object(openai_adapter, 'ensure_initialized', AsyncMock(return_value=True)): # ensure_initialized 모킹 개선

        # _initialize가 호출되었고 성공했다고 가정
        openai_adapter.initialized = True
        openai_adapter.tokenizer = mock_tokenizer # 토크나이저 설정

        response = await openai_adapter._generate_text(
            prompt="User query for chat model",
            max_tokens=60 # 다른 max_tokens 값 테스트
        )

        # 검증
        mock_session.post.assert_called_once() # post가 호출되었는지 확인
        call_args, call_kwargs = mock_session.post.call_args
        # 호출된 URL 확인 (chat 모델이므로 /chat/completions)
        assert call_args[0] == "https://api.openai.com/v1/chat/completions"
        # 요청 payload 확인
        payload = call_kwargs["json"]
        assert payload["model"] == "gpt-3.5-turbo"
        # chat 모델은 'messages' 형식 사용 확인
        assert "prompt" not in payload
        assert payload["messages"] == [{"role": "user", "content": "User query for chat model"}]
        assert payload["max_tokens"] == 60
        assert payload["temperature"] == 0.7 # 어댑터 기본값 사용 확인
        # 응답 내용 확인 (chat 형식)
        assert response["id"] == "test-chat-response-id"
        assert len(response["choices"]) == 1
        # chat 응답은 choices[0]["message"]["content"] 에 있음 -> choices[0]["text"] 로 변환됨 확인
        assert response["choices"][0]["text"] == "This is a test chat response"
        assert response["usage"]["prompt_tokens"] == 15 # mock_count의 반환값
        assert response["usage"]["completion_tokens"] == 7 # 응답값 사용
        assert response["usage"]["total_tokens"] == 22 # 응답값 사용
        assert "request_time" in response

@pytest.mark.asyncio
async def test_generate_text_api_error(openai_adapter, mock_session, mock_error_response):
    """Test handling of API errors (e.g., 4xx status code)."""
    # --- session.post 모킹 설정 (에러 응답) ---
    mock_post_context = AsyncMock()
    mock_post_context.__aenter__.return_value = mock_error_response
    mock_post_context.__aexit__ = AsyncMock(return_value=None)
    # 수정: post 호출 시 context manager mock 반환
    mock_session.post.return_value = mock_post_context
    # ----------------------------

    # 토크나이저 모킹 추가
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1] * 5 # 5 토큰으로 모킹

    with patch.object(openai_adapter, '_get_client', return_value=mock_session), \
         patch.object(openai_adapter, '_get_tokenizer', return_value=mock_tokenizer), \
         patch.object(openai_adapter, '_count_tokens', return_value=5), \
         patch.object(openai_adapter, 'ensure_initialized', AsyncMock(return_value=True)):

        # 초기화 및 토크나이저 설정
        openai_adapter.initialized = True
        openai_adapter.tokenizer = mock_tokenizer

        # LLMError가 발생하는지 확인
        with pytest.raises(LLMError) as excinfo:
            await openai_adapter._generate_text("Prompt causing API error")

        # 발생한 에러의 상세 내용 확인 (이제 TypeError가 아닌 LLMError 검증 가능)
        assert excinfo.value.code == ErrorCode.LLM_API_ERROR
        assert "OpenAI API error: Invalid request simulated" in excinfo.value.message
        assert excinfo.value.details["status_code"] == 400
        assert excinfo.value.details["error_type"] == "invalid_request_error"
        assert excinfo.value.details["model"] == openai_adapter.model

@pytest.mark.asyncio
async def test_generate_text_network_error(openai_adapter, mock_session):
    """Test handling of network errors during API call."""
    # --- session.post 모킹 설정 (네트워크 예외 발생) ---
    # 수정: post 호출 시 side_effect 발생
    mock_session.post.side_effect = aiohttp.ClientConnectionError("Simulated network error")
    # -----------------------------------------

    # 토크나이저 모킹 추가
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1] * 8 # 8 토큰으로 모킹

    with patch.object(openai_adapter, '_get_client', return_value=mock_session), \
         patch.object(openai_adapter, '_get_tokenizer', return_value=mock_tokenizer), \
         patch.object(openai_adapter, '_count_tokens', return_value=8), \
         patch.object(openai_adapter, 'ensure_initialized', AsyncMock(return_value=True)):

        # 초기화 및 토크나이저 설정
        openai_adapter.initialized = True
        openai_adapter.tokenizer = mock_tokenizer

        # LLMError가 발생하는지 확인
        with pytest.raises(LLMError) as excinfo:
            await openai_adapter._generate_text("Prompt causing network error")

        # 발생한 에러의 상세 내용 확인 (이제 TypeError가 아닌 LLMError 검증 가능)
        assert excinfo.value.code == ErrorCode.LLM_API_ERROR
        # 원본 에러 메시지가 포함되었는지 확인
        assert "Error calling OpenAI API: Simulated network error" in excinfo.value.message
        assert isinstance(excinfo.value.original_error, aiohttp.ClientConnectionError)
        # 수정: details 딕셔너리에서 model과 provider를 확인
        assert excinfo.value.details.get("model") == openai_adapter.model # .get() 사용 권장
        assert excinfo.value.details.get("provider") == "openai" # .get() 사용 권장

def test_get_token_limit(openai_adapter):
    """Test getting token limits for different known models."""
    # 모델별 토큰 제한 확인
    assert openai_adapter.get_token_limit() == 8192 # gpt-4 기본
    openai_adapter.model = "gpt-4-turbo"
    assert openai_adapter.get_token_limit() == 128000
    openai_adapter.model = "gpt-4o"
    assert openai_adapter.get_token_limit() == 128000
    openai_adapter.model = "gpt-3.5-turbo"
    assert openai_adapter.get_token_limit() == 16385
    openai_adapter.model = "gpt-3.5-turbo-16k" # Prefix 매칭 테스트
    assert openai_adapter.get_token_limit() == 16385
    openai_adapter.model = "text-davinci-003"
    assert openai_adapter.get_token_limit() == 4097
    # 알 수 없는 모델의 경우 기본값 확인
    openai_adapter.model = "unknown-model-xyz"
    assert openai_adapter.get_token_limit() == 4097 # 기본 fallback 값

@pytest.mark.asyncio
async def test_close(openai_adapter):
    """Test the close method resets the initialized flag."""
    # 초기화 상태를 True로 설정 (테스트 목적)
    openai_adapter.initialized = True
    await openai_adapter.close()
    # close 호출 후 initialized 상태가 False로 바뀌었는지 확인
    assert openai_adapter.initialized is False
import pytest
import asyncio
import sys
from unittest.mock import patch, AsyncMock, MagicMock

# 테스트 대상 함수 임포트
from src.llm.parallel import (
    execute_parallel,
    race_models,
    execute_with_fallbacks
)
# 필요한 에러 클래스 임포트
from src.config.errors import LLMError, ErrorCode

# 테스트용 Mock Adapter 저장소
test_adapters = {}

@pytest.fixture(autouse=True)
def mock_metrics(monkeypatch):
    """메트릭 관련 모듈 모킹"""
    # Mock counters
    mock_counter = MagicMock()
    mock_counter_child = MagicMock()
    mock_counter_child.inc = MagicMock()
    mock_counter.labels.return_value = mock_counter_child
    
    # Mock track_llm_fallback
    mock_track = MagicMock()
    
    # 설정
    monkeypatch.setattr("src.llm.parallel.LLM_REQUESTS_TOTAL", mock_counter)
    monkeypatch.setattr("src.llm.parallel.track_llm_fallback", mock_track)
    
    return mock_counter, mock_track

@pytest.fixture(autouse=True)
def mock_adapters_function(monkeypatch):
    """adapter 생성 함수 직접 모킹"""
    # 테스트에 필요한 adapter 생성하는 함수
    async def mock_create_adapters_concurrently(models):
        return {model: test_adapters.get(model) for model in models if model in test_adapters}
    
    monkeypatch.setattr("src.llm.parallel._create_adapters_concurrently", mock_create_adapters_concurrently)

@pytest.fixture(autouse=True)
def mock_llm_settings(monkeypatch):
    """모든 LLM 모듈에 대한 설정 모의 객체"""
    mock_settings_obj = MagicMock()
    mock_settings_obj.REQUEST_TIMEOUT = 10.0
    mock_settings_obj.LLM_MODEL_PROVIDER_MAP = {
        "model1": "openai", "model2": "openai",
        "primary_model": "openai", "fallback1": "anthropic",
        "fallback2": "anthropic", "failing_model": "openai",
        "success_model": "anthropic", "fallback_model": "anthropic"
    }
    mock_settings_obj.LLM_PROVIDERS_CONFIG = {
        "openai": {"api_key": "mock-key"},
        "anthropic": {"api_key": "mock-key"}
    }

    # 두 모듈 모두 패치
    monkeypatch.setattr("src.llm.parallel.settings", mock_settings_obj)
    monkeypatch.setattr("src.llm.adapters.settings", mock_settings_obj)

def async_mock_return(return_value):
    """호출 추적 기능이 있는 비동기 모의 함수"""
    mock = AsyncMock()
    mock.return_value = return_value
    return mock

def make_mock_awaitable(mock):
    """Make an AsyncMock properly awaitable when called directly."""
    async def _awaitable_mock(*args, **kwargs):
        return mock
    mock.__await__ = lambda: _awaitable_mock().__await__()
    return mock

@pytest.fixture
def success_operation():
    """Create an operation that succeeds."""
    async def operation():
        await asyncio.sleep(0.01)
        return "success"
    return operation

@pytest.fixture
def slow_operation():
    """Create an operation that succeeds but is slow."""
    async def operation():
        await asyncio.sleep(0.1)
        return "slow success"
    return operation

@pytest.fixture
def error_operation():
    """Create an operation that raises an error."""
    async def operation():
        await asyncio.sleep(0.01)
        raise ValueError("Operation failed")
    return operation

# --- execute_parallel Tests ---

@pytest.mark.asyncio
async def test_execute_parallel_all_success(success_operation):
    """Test execute_parallel with all operations succeeding."""
    operations = [success_operation] * 3
    results = await execute_parallel(operations)
    assert len(results) == 3
    assert all(result == "success" for result in results)

@pytest.mark.asyncio
async def test_execute_parallel_mixed_results(success_operation, error_operation):
    """Test execute_parallel with some operations succeeding and some failing."""
    operations = [success_operation, success_operation, error_operation]
    results = await execute_parallel(operations, return_exceptions=True)
    assert len(results) == 3
    assert results[0] == "success"
    assert results[1] == "success"
    assert isinstance(results[2], ValueError)
    assert str(results[2]) == "Operation failed"

@pytest.mark.asyncio
async def test_execute_parallel_error_propagation(success_operation, error_operation):
    """Test execute_parallel error propagation."""
    operations = [success_operation, error_operation]
    with pytest.raises(ValueError) as excinfo:
        await execute_parallel(operations)
    assert str(excinfo.value) == "Operation failed"

@pytest.mark.asyncio
async def test_execute_parallel_cancel_on_first_result(success_operation, slow_operation):
    """Test execute_parallel with cancel_on_first_result=True."""
    operations = [success_operation, slow_operation]
    results = await execute_parallel(operations, cancel_on_first_result=True)
    assert len(results) >= 1
    assert results[0] == "success"

@pytest.mark.asyncio
async def test_execute_parallel_cancel_on_first_exception(error_operation, slow_operation):
    """Test execute_parallel with cancel_on_first_exception=True."""
    operations = [error_operation, slow_operation, slow_operation]
    results = await execute_parallel(
        operations,
        cancel_on_first_exception=True,
        return_exceptions=True
    )
    assert len(results) >= 1
    assert isinstance(results[0], ValueError)
    assert str(results[0]) == "Operation failed"

@pytest.mark.asyncio
async def test_execute_parallel_timeout():
    """Test execute_parallel with timeout."""
    async def fast_op():
        await asyncio.sleep(0.01)
        return "fast"

    async def slow_op():
        await asyncio.sleep(0.5)
        return "slow"

    operations = [fast_op, slow_op]
    # 타임아웃을 0.1초로 설정 (slow_op 보다 짧게)
    results = await execute_parallel(operations, timeout=0.1, return_exceptions=True)

    # fast_op는 완료, slow_op는 TimeoutError 또는 None (구현에 따라 다름)
    assert "fast" in results
    # slow_op의 결과가 TimeoutError 또는 None인지 확인
    slow_op_result = next((r for r in results if not isinstance(r, str)), None)
    assert slow_op_result is None or isinstance(slow_op_result, asyncio.TimeoutError) or isinstance(slow_op_result, Exception)


# --- race_models & execute_with_fallbacks Tests ---

@pytest.fixture
def create_mock_adapter():
    """생성 가능한 LLM 어댑터 모의 객체 생성"""
    def _create_adapter(model_name, response=None, side_effect=None):
        adapter = AsyncMock()
        make_mock_awaitable(adapter)
        
        # 기본 응답 설정
        if response is None:
            response = {"choices": [{"text": f"Response from {model_name}"}]}
        
        # generate 메서드 설정
        if side_effect:
            adapter.generate = AsyncMock(side_effect=side_effect)
        else:
            adapter.generate = AsyncMock(return_value=response)
        
        # 모델명 속성 추가
        adapter.model = model_name
        
        # 테스트 어댑터 저장소에 저장
        test_adapters[model_name] = adapter
        return adapter
    
    return _create_adapter

@pytest.mark.asyncio
async def test_race_models_success(create_mock_adapter):
    """테스트: 모든 모델이 성공적으로 응답"""
    # 테스트 어댑터 생성
    adapter1 = create_mock_adapter("model1", response={"choices": [{"text": "Response from model1"}]})
    adapter2 = create_mock_adapter("model2", response={"choices": [{"text": "Response from model2"}]})
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        model, response = await race_models(
            models=["model1", "model2"],
            prompt="Test prompt",
            max_tokens=10
        )
        
        # 결과 확인
        assert model in ["model1", "model2"]
        assert response["choices"][0]["text"] in ["Response from model1", "Response from model2"]
        
        # generate 호출 확인
        assert adapter1.generate.call_count + adapter2.generate.call_count > 0

@pytest.mark.asyncio
async def test_race_models_one_fails(create_mock_adapter):
    """테스트: 하나의 모델은 실패하고 다른 하나는 성공"""
    # 테스트 어댑터 생성
    failing_adapter = create_mock_adapter("failing_model", 
                                         side_effect=LLMError(code=ErrorCode.LLM_API_ERROR, message="API error"))
    success_adapter = create_mock_adapter("success_model")
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        model, response = await race_models(
            models=["failing_model", "success_model"],
            prompt="Test prompt"
        )
        
        # 결과 확인
        assert model == "success_model"
        assert response["choices"][0]["text"] == "Response from success_model"
        
        # generate 호출 확인
        assert failing_adapter.generate.call_count == 1
        assert success_adapter.generate.call_count == 1

@pytest.mark.asyncio
async def test_race_models_all_fail(create_mock_adapter):
    """테스트: 모든 모델이 실패하는 경우"""
    # 테스트 어댑터 생성
    error = LLMError(code=ErrorCode.LLM_API_ERROR, message="API error")
    adapter1 = create_mock_adapter("model1", side_effect=error)
    adapter2 = create_mock_adapter("model2", side_effect=error)
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        with pytest.raises(LLMError) as excinfo:
            await race_models(
                models=["model1", "model2"],
                prompt="Test prompt"
            )
            
        # 오류 메시지 확인
        assert "All models failed in race" in str(excinfo.value)
        
        # generate 호출 확인
        assert adapter1.generate.call_count == 1
        assert adapter2.generate.call_count == 1

@pytest.mark.asyncio
async def test_race_models_metrics(create_mock_adapter, mock_metrics):
    """테스트: 메트릭 추적 기능 확인"""
    # 테스트 어댑터 생성
    mock_track = mock_metrics[1]  # track_llm_fallback mock
    
    # 어댑터 설정: primary 실패, fallback 성공
    create_mock_adapter("primary_model", 
                      side_effect=LLMError(code=ErrorCode.LLM_API_ERROR, message="Primary failed"))
    create_mock_adapter("fallback_model", 
                      response={"choices": [{"text": "Fallback Response"}]})
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        model, response = await race_models(
            models=["primary_model", "fallback_model"],
            prompt="Test prompt"
        )
        
        # 결과 확인
        assert model == "fallback_model"
        assert response["choices"][0]["text"] == "Fallback Response"
        
        # 메트릭 추적 확인
        mock_track.assert_called_once_with("primary_model", "fallback_model")

@pytest.mark.asyncio
async def test_execute_with_fallbacks_primary_success(create_mock_adapter):
    """테스트: primary 모델이 성공적으로 응답하는 경우"""
    # 테스트 어댑터 생성
    primary_adapter = create_mock_adapter("primary_model")
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        model, response = await execute_with_fallbacks(
            primary_model="primary_model",
            fallback_models=["fallback1", "fallback2"],
            prompt="Test prompt"
        )
        
        # 결과 확인
        assert model == "primary_model"
        assert response["choices"][0]["text"] == "Response from primary_model"
        
        # generate가 primary 모델에서만 호출되었는지 확인
        assert primary_adapter.generate.call_count == 1

@pytest.mark.asyncio
async def test_execute_with_fallbacks_primary_fails(create_mock_adapter, mock_metrics):
    """테스트: primary 모델 실패, fallback 성공"""
    # 메트릭 모킹
    mock_track = mock_metrics[1]  # track_llm_fallback mock
    
    # 테스트 어댑터 생성
    create_mock_adapter("primary_model", 
                      side_effect=LLMError(code=ErrorCode.LLM_API_ERROR, message="Primary error"))
    fallback_adapter = create_mock_adapter("fallback1")
    create_mock_adapter("fallback2")  # 이 어댑터는 호출되지 않아야 함
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        model, response = await execute_with_fallbacks(
            primary_model="primary_model",
            fallback_models=["fallback1", "fallback2"],
            prompt="Test prompt"
        )
        
        # 결과 확인
        assert model == "fallback1"
        assert response["choices"][0]["text"] == "Response from fallback1"
        
        # generate 호출 확인
        assert test_adapters["primary_model"].generate.call_count == 1
        assert fallback_adapter.generate.call_count == 1
        assert test_adapters["fallback2"].generate.call_count == 0
        
        # 메트릭 추적 확인
        mock_track.assert_called_once_with("primary_model", "fallback1")

@pytest.mark.asyncio
async def test_execute_with_fallbacks_all_fail(create_mock_adapter):
    """테스트: 모든 모델이 실패하는 경우"""
    # 테스트 어댑터 생성 - 모두 실패하도록 설정
    error = LLMError(code=ErrorCode.LLM_API_ERROR, message="API error")
    primary_adapter = create_mock_adapter("primary_model", side_effect=error)
    fallback1 = create_mock_adapter("fallback1", side_effect=error)
    fallback2 = create_mock_adapter("fallback2", side_effect=error)
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        with pytest.raises(LLMError) as excinfo:
            await execute_with_fallbacks(
                primary_model="primary_model",
                fallback_models=["fallback1", "fallback2"],
                prompt="Test prompt"
            )
            
        # 오류 메시지 확인
        assert "All models failed" in str(excinfo.value)
        
        # 모든 모델이 호출되었는지 확인
        assert primary_adapter.generate.call_count == 1
        assert fallback1.generate.call_count == 1
        assert fallback2.generate.call_count == 1

@pytest.mark.asyncio
async def test_execute_with_fallbacks_timeout(create_mock_adapter):
    """테스트: 타임아웃으로 fallback으로 넘어가는 경우"""
    # 테스트 어댑터 생성
    # primary는 타임아웃 시뮬레이션
    async def timeout_effect(*args, **kwargs):
        await asyncio.sleep(0.5)  # 의도적으로 긴 지연
        raise asyncio.TimeoutError("Simulated timeout")
    
    primary_adapter = create_mock_adapter("primary_model", side_effect=timeout_effect)
    fallback_adapter = create_mock_adapter("fallback1")
    
    # get_adapter 모킹
    async def mock_get_adapter(model_name):
        return test_adapters.get(model_name)
    
    with patch('src.llm.parallel.get_adapter', side_effect=mock_get_adapter):
        model, response = await execute_with_fallbacks(
            primary_model="primary_model",
            fallback_models=["fallback1"],
            prompt="Test prompt",
            timeout=0.1  # 짧은 타임아웃 설정
        )
        
        # 결과 확인
        assert model == "fallback1"
        assert response["choices"][0]["text"] == "Response from fallback1"
        
        # 호출 확인
        assert primary_adapter.generate.called  # primary가 호출됨
        assert fallback_adapter.generate.called  # fallback도 호출됨
import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from src.config.settings import get_settings
from src.services.llm_client import LLMClient
from src.config.errors import LLMError
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic

settings = get_settings()

async def test_llm_client_initialization():
    """LLMClient가 설정에 따라 초기화되는지 확인"""
    llm_client = LLMClient()
    assert llm_client.primary_llm is not None
    assert llm_client.primary_llm.model_name == settings.LLM_PROVIDERS[settings.PRIMARY_LLM_PROVIDER].model_name
    assert llm_client.fallback_llm is not None
    assert llm_client.fallback_llm.model_name == settings.LLM_PROVIDERS[settings.FALLBACK_LLM_PROVIDER].model_name

async def test_generate_response_success():
    """generate_response() 성공 케이스 테스트"""
    llm_client = LLMClient()
    llm_client.primary_llm.ainvoke = AsyncMock(return_value="LLM Response")
    response = await llm_client.generate_response(messages=[{"role": "user", "content": "Test"}])
    assert response == "LLM Response"

async def test_generate_response_retry():
    """generate_response() 재시도 로직 테스트"""
    llm_client = LLMClient()
    llm_client.primary_llm.ainvoke = AsyncMock(side_effect=[Exception("LLM Error"), "LLM Response"])
    response = await llm_client.generate_response(messages=[{"role": "user", "content": "Test"}])
    assert response == "LLM Response"
    assert llm_client.primary_llm.ainvoke.call_count == 2

async def test_generate_response_failure():
    """generate_response() 실패 케이스 테스트"""
    llm_client = LLMClient()
    llm_client.primary_llm.ainvoke = AsyncMock(side_effect=Exception("LLM Error"))
    with pytest.raises(LLMError):
        await llm_client.generate_response(messages=[{"role": "user", "content": "Test"}])

async def test_chat_success():
    """chat() 성공 케이스 테스트"""
    llm_client = LLMClient()
    llm_client.primary_llm.ainvoke = AsyncMock(return_value="Primary LLM Response")
    response = await llm_client.chat(messages=[{"role": "user", "content": "Chat Test"}])
    assert response == "Primary LLM Response"

async def test_chat_fallback():
    """chat() 폴백 LLM 사용 케이스 테스트"""
    llm_client = LLMClient()
    llm_client.primary_llm.ainvoke = AsyncMock(side_effect=Exception("Primary LLM Error"))
    llm_client.fallback_llm.ainvoke = AsyncMock(return_value="Fallback LLM Response")
    response = await llm_client.chat(messages=[{"role": "user", "content": "Chat Test"}])
    assert response == "Fallback LLM Response"

async def test_chat_failure():
    """chat() 실패 케이스 테스트"""
    llm_client = LLMClient()
    llm_client.primary_llm.ainvoke = AsyncMock(side_effect=Exception("Primary LLM Error"))
    llm_client.fallback_llm = None
    with pytest.raises(LLMError):
        await llm_client.chat(messages=[{"role": "user", "content": "Chat Test"}])

async def test_create_prompt_success():
    """create_prompt() 성공 케이스 테스트"""
    llm_client = LLMClient()
    template = "The capital of {country} is {capital}."
    kwargs = {"country": "France", "capital": "Paris"}
    prompt = await llm_client.create_prompt(template, **kwargs)
    assert prompt == "The capital of France is Paris."

async def test_create_prompt_missing_variable():
    """create_prompt() 필수 변수 누락 케이스 테스트"""
    llm_client = LLMClient()
    template = "The capital of {country} is {capital}."
    kwargs = {"country": "France"}
    with pytest.raises(LLMError):
        await llm_client.create_prompt(template, **kwargs)

async def test_create_prompt_invalid_template():
    """create_prompt() 잘못된 템플릿 형식 테스트"""
    llm_client = LLMClient()
    template = "The capital of {country} is {capital"  # 닫는 괄호 누락
    kwargs = {"country": "France", "capital": "Paris"}
    with pytest.raises(LLMError):
        await llm_client.create_prompt(template, **kwargs)
        
async def test_generate_response_timeout_then_success_on_retry(monkeypatch):
    """generate_response() 타임아웃 후 재시도 성공 테스트"""
    # 재시도 1회로 설정
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)

    llm_client = LLMClient()
    # 첫 번째는 TimeoutError, 두 번째는 성공
    llm_client.primary_llm.ainvoke = AsyncMock(side_effect=[
        asyncio.TimeoutError(),
        "Successful Response After Timeout",
    ])

    response = await llm_client.generate_response(
        messages=[{"role": "user", "content": "Test timeout"}]
    )
    assert response == "Successful Response After Timeout"
    assert llm_client.primary_llm.ainvoke.call_count == 2


async def test_generate_response_timeout_then_fallback(monkeypatch):
    """generate_response() 타임아웃 후 재시도 모두 실패 시 chat()에서 폴백 사용 테스트"""
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 1)

    llm_client = LLMClient()
    # primary는 계속 TimeoutError
    llm_client.primary_llm.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
    # fallback LLM이 존재해야 함
    assert llm_client.fallback_llm is not None

    llm_client.fallback_llm.ainvoke = AsyncMock(return_value="Fallback Response Due to Timeout")

    response = await llm_client.chat(
        messages=[{"role": "user", "content": "Test timeout leading to fallback"}]
    )
    assert response == "Fallback Response Due to Timeout"
    # primary 호출: LLM_MAX_RETRIES + 1 번
    assert llm_client.primary_llm.ainvoke.call_count == 2
    # fallback 호출: 1번
    assert llm_client.fallback_llm.ainvoke.call_count == 1


@pytest.mark.parametrize("target_provider, expected_type, mock_model_name", [
    ("anthropic", ChatAnthropic, "claude-test"),
])
async def test_llm_client_selects_different_provider(
    monkeypatch, target_provider, expected_type, mock_model_name
):
    """설정에 따라 다른 LLM 공급자가 선택되는지 테스트"""
    # 1) mock provider 설정 객체(SimpleNamespace)
    mock_dict = {"api_key": f"{target_provider}_key", "model_name": mock_model_name}
    mock_provider_settings = SimpleNamespace(**mock_dict)

    # 2) settings 패치
    monkeypatch.setattr(settings, "LLM_PROVIDERS", {target_provider: mock_provider_settings})
    monkeypatch.setattr(settings, "PRIMARY_LLM_PROVIDER", target_provider)
    monkeypatch.setattr(settings, "FALLBACK_LLM_PROVIDER", None)

    # 3) LLMClient 생성 후 검증
    llm_client = LLMClient()
    # 래퍼 내부 실제 LLM 객체 타입 확인
    inner = llm_client.primary_llm._llm
    
    # 여기서 모킹 관련 클래스 타입 패치
    from src.services.llm_client import MockChatAnthropic
    
    # 테스트 환경에서는 실제 ChatAnthropic 또는 MockChatAnthropic 둘 다 허용
    assert isinstance(inner, expected_type) or isinstance(inner, MockChatAnthropic), \
        f"Expected {expected_type.__name__} or MockChatAnthropic, but got {type(inner).__name__}"
    
    # 모델 이름은 정확히 설정되어야 함
    assert llm_client.primary_llm.model_name == mock_model_name, \
        f"Expected model name to be {mock_model_name}, but got {llm_client.primary_llm.model_name}"
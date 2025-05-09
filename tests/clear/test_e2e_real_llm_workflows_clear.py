# tests/e2e/test_e2e_real_llm_workflows.py
import asyncio
import os
import time
import pytest
from typing import Any, Dict, Optional, List
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

# --- 디버깅 코드 섹션 ---
# 이 섹션은 다른 import 보다 먼저 실행되어 초기 환경 상태를 확인합니다.
print("\n--- E2E TEST FILE LOADED: INITIAL ENVIRONMENT CHECK ---")
print(f"Current working directory: {os.getcwd()}")
print(f"Is .env file present in CWD? {'Yes' if os.path.exists('.env') else 'No'}")

# pytest-dotenv가 작동했다면 이 시점에 .env 파일의 변수들이 로드되어 있어야 합니다.
INITIAL_PRIMARY_LLM_PROVIDER_FROM_OS_ENV = os.getenv("PRIMARY_LLM_PROVIDER")
print(f"INITIAL_PRIMARY_LLM_PROVIDER_FROM_OS_ENV (before get_settings): '{INITIAL_PRIMARY_LLM_PROVIDER_FROM_OS_ENV}'")

if INITIAL_PRIMARY_LLM_PROVIDER_FROM_OS_ENV:
    _api_key_var_name_debug = f"LLM_PROVIDERS__{INITIAL_PRIMARY_LLM_PROVIDER_FROM_OS_ENV}__API_KEY"
    _api_key_value_debug = os.getenv(_api_key_var_name_debug)
    if _api_key_value_debug:
        print(f"INITIAL_API_KEY_FOR_PRIMARY_PROVIDER (os.getenv('{_api_key_var_name_debug}')): '******{_api_key_value_debug[-4:]}'")
    else:
        print(f"INITIAL_API_KEY_FOR_PRIMARY_PROVIDER (os.getenv('{_api_key_var_name_debug}')): Not found or empty")
else:
    print("INITIAL_API_KEY_FOR_PRIMARY_PROVIDER: Cannot determine as PRIMARY_LLM_PROVIDER is not set in os.environ")
print("--- END INITIAL ENVIRONMENT CHECK ---\n")
# --- 디버깅 코드 섹션 끝 ---

# 애플리케이션 및 설정 로드
# 이 시점에서 get_settings()가 호출되면, pydantic-settings가 .env를 (다시) 읽으려고 시도할 수 있습니다.
# pytest-dotenv가 이미 로드했다면, pydantic-settings는 기존 환경 변수를 사용합니다.
try:
    from src.api.app import app
    from src.config.settings import get_settings
    from src.services.llm_client import LLMClient # LLM Fallback 테스트 시 타입 힌트 및 패치 대상
except ImportError as e:
    print(f"FATAL E2E IMPORT ERROR: {e}. Check PYTHONPATH or project structure.")
    # 임포트 에러 시 pytest가 파일을 제대로 실행할 수 없으므로, 여기서 테스트를 중단시키는 것이 나을 수 있습니다.
    # 하지만 skipif 조건이 먼저 평가되도록 일단 진행합니다.
    app = None
    get_settings = None
    LLMClient = None

# settings 객체는 skipif 조건문 및 테스트 함수에서 사용됩니다.
# get_settings() 호출은 설정 로드 시점에 .env 파일이 (pytest-dotenv에 의해) 이미 환경 변수로 반영되었기를 기대합니다.
settings = None
if get_settings:
    try:
        settings = get_settings()
        print("\n--- SETTINGS OBJECT AFTER get_settings() CALL ---")
        print(f"settings.PRIMARY_LLM_PROVIDER: '{getattr(settings, 'PRIMARY_LLM_PROVIDER', 'NOT_FOUND')}'")
        if hasattr(settings, 'LLM_PROVIDERS') and settings.PRIMARY_LLM_PROVIDER in settings.LLM_PROVIDERS:
            primary_provider_conf = settings.LLM_PROVIDERS[settings.PRIMARY_LLM_PROVIDER]
            print(f"Primary Provider ('{settings.PRIMARY_LLM_PROVIDER}') Config API Key in settings: {'******' + primary_provider_conf.api_key[-4:] if primary_provider_conf.api_key else 'NOT_SET'}")
        else:
            print(f"Primary Provider ('{getattr(settings, 'PRIMARY_LLM_PROVIDER', 'N/A')}') config not found in settings.LLM_PROVIDERS")

        # Fallback Provider 정보 (디버깅용)
        if hasattr(settings, 'FALLBACK_LLM_PROVIDER') and settings.FALLBACK_LLM_PROVIDER:
            print(f"settings.FALLBACK_LLM_PROVIDER: '{settings.FALLBACK_LLM_PROVIDER}'")
            if settings.FALLBACK_LLM_PROVIDER in settings.LLM_PROVIDERS:
                 fallback_provider_conf = settings.LLM_PROVIDERS[settings.FALLBACK_LLM_PROVIDER]
                 print(f"Fallback Provider ('{settings.FALLBACK_LLM_PROVIDER}') Config API Key in settings: {'******' + fallback_provider_conf.api_key[-4:] if fallback_provider_conf.api_key else 'NOT_SET'}")
            else:
                print(f"Fallback Provider ('{settings.FALLBACK_LLM_PROVIDER}') config not found in settings.LLM_PROVIDERS")
        else:
            print("settings.FALLBACK_LLM_PROVIDER: Not set or empty")

        print("--- END SETTINGS OBJECT DEBUG ---\n")

    except Exception as e_settings:
        print(f"ERROR: Could not load settings via get_settings(): {e_settings}")
        # settings가 None이면 skipif 조건에서 AttributeError 발생 가능, 아래에서 처리

# skipif 조건 정의
_skip_e2e_real_llm_tests = True
_skip_reason = "Default skip: Settings or os.environ not properly evaluated."

if settings:
    primary_llm_provider_name_from_settings = getattr(settings, 'PRIMARY_LLM_PROVIDER', None)
    if primary_llm_provider_name_from_settings:
        # pydantic-settings가 환경변수와 .env를 종합해서 settings.LLM_PROVIDERS[provider].api_key 를 채움
        # 따라서 settings 객체에서 직접 API 키를 확인하는 것이 더 정확합니다.
        primary_provider_config = settings.LLM_PROVIDERS.get(primary_llm_provider_name_from_settings)
        if primary_provider_config and primary_provider_config.api_key:
            _skip_e2e_real_llm_tests = False
            _skip_reason = "" # 테스트 실행 가능
            print(f"SKIP_INFO: Primary LLM ('{primary_llm_provider_name_from_settings}') API key FOUND in settings object. Tests will attempt to run.")
        else:
            _skip_reason = (f"Real LLM API key for primary provider '{primary_llm_provider_name_from_settings}' "
                            f"NOT FOUND or EMPTY in the 'settings.LLM_PROVIDERS' object. "
                            "Check .env and its loading by pydantic-settings (via pytest-dotenv).")
            print(f"SKIP_INFO: {_skip_reason}")
    else:
        _skip_reason = ("'PRIMARY_LLM_PROVIDER' is NOT SET in the 'settings' object. "
                        "Ensure it's defined in your .env file and loaded correctly.")
        print(f"SKIP_INFO: {_skip_reason}")
else:
    _skip_reason = "Global 'settings' object could not be initialized. Cannot determine API key presence."
    print(f"SKIP_INFO: {_skip_reason}")


e2e_real_llm = pytest.mark.skipif(
    _skip_e2e_real_llm_tests,
    reason=_skip_reason
)

# --- Fixture 및 테스트 함수 ---
@pytest.fixture(scope="module")
def client():
    if not app: # app 임포트 실패 시
        pytest.skip("FastAPI app could not be imported. Skipping E2E tests.")
        return None
    with TestClient(app) as c:
        yield c

async def poll_for_status(
    client: Optional[TestClient],
    task_id: str,
    timeout_seconds: int = 180,
    interval_seconds: int = 5,
) -> Optional[Dict[str, Any]]:
    if not client: return None # client가 None이면 (app 임포트 실패 등) 아무것도 하지 않음

    start_time = time.time()
    print(f"\nPolling status for task_id: {task_id} (timeout: {timeout_seconds}s)")
    while time.time() - start_time < timeout_seconds:
        response = client.get(f"/api/v1/status/{task_id}")
        if response.status_code == 200:
            status_data = response.json()
            current_status = status_data.get("status")
            print(f"Task {task_id} status: {current_status}, Detail: {status_data.get('error_message') or status_data.get('final_answer') or 'Running...'}")
            if current_status in ["completed", "failed"]:
                return status_data
        elif response.status_code == 404:
            print(f"Task {task_id} status: Not found yet (404)")
        else:
            print(f"Task {task_id} status check failed with code {response.status_code}: {response.text}")
            try:
                response.raise_for_status()
            except Exception as e:
                print(f"Error during polling: {e}")
                return {"status": "error_polling", "error_message": str(e)}
        await asyncio.sleep(interval_seconds)
    print(f"Polling timed out for task_id: {task_id}")
    return {"status": "timeout_polling", "error_message": "Polling for status timed out."}


@e2e_real_llm
@pytest.mark.asyncio
async def test_e2e_simple_prompt_agent_real_llm(client: Optional[TestClient]):
    if not client: pytest.skip("TestClient not available.")
    workflow_name = "simple_prompt_agent"
    user_question = "What is the capital of France? Respond in one word."

    request_payload = {
        "graph_config_name": workflow_name,
        "original_input": {"original_input": user_question},
        "initial_metadata": {"test_type": "e2e_real_llm_simple"}
    }

    response_run = client.post("/api/v1/run", json=request_payload)
    assert response_run.status_code == 202, f"Run endpoint failed: {response_run.text}"
    run_data = response_run.json()
    task_id = run_data.get("task_id")
    assert task_id, "Task ID not found in run response"

    final_status = await poll_for_status(client, task_id)

    assert final_status is not None, f"Polling resulted in None for task {task_id}"
    assert final_status.get("status") == "completed", f"Task {task_id} did not complete successfully: Status '{final_status.get('status')}', Error: {final_status.get('error_message')}, LLM Output: {final_status.get('last_llm_output')}"
    assert final_status.get("error_message") is None, f"Task {task_id} completed with error: {final_status.get('error_message')}"

    final_answer = final_status.get("final_answer")
    assert final_answer, "Final answer is empty"
    assert "Paris" in final_answer, f"Expected 'Paris' in final answer, but got: {final_answer}"

    print(f"\n[E2E Simple Prompt Agent Test] Task ID: {task_id}")
    print(f"Question: {user_question}")
    print(f"LLM Answer: {final_answer}")

@e2e_real_llm
@pytest.mark.asyncio
async def test_e2e_react_tool_agent_calculator_real_llm(client: Optional[TestClient]):
    if not client: pytest.skip("TestClient not available.")
    workflow_name = "react_tool_workflow"
    user_query = "What is 123 plus 456, and then multiply the sum by 2?"

    request_payload = {
        "graph_config_name": workflow_name,
        "original_input": user_query,
        "initial_metadata": {"test_type": "e2e_real_llm_react_calculator"}
    }

    response_run = client.post("/api/v1/run", json=request_payload)
    assert response_run.status_code == 202, f"Run endpoint failed: {response_run.text}"
    run_data = response_run.json()
    task_id = run_data.get("task_id")
    assert task_id

    final_status = await poll_for_status(client, task_id, timeout_seconds=300)

    assert final_status is not None, f"Polling resulted in None for task {task_id}"
    assert final_status.get("status") == "completed", f"Task {task_id} did not complete successfully: Status '{final_status.get('status')}', Error: {final_status.get('error_message')}, LLM Output: {final_status.get('last_llm_output')}"
    assert final_status.get("error_message") is None, f"Task {task_id} completed with error: {final_status.get('error_message')}"

    final_answer = final_status.get("final_answer")
    assert final_answer, "Final answer is empty"
    expected_result = "1158"
    assert expected_result in final_answer, \
        f"Expected calculation result '{expected_result}' in final answer, but got: {final_answer}"

    print(f"\n[E2E ReAct Calculator Test] Task ID: {task_id}")
    print(f"Query: {user_query}")
    print(f"LLM Final Answer: {final_answer}")


@e2e_real_llm
@pytest.mark.asyncio
async def test_e2e_memory_persistence_real_llm(client: Optional[TestClient]):
    """
    메모리 시스템의 지속성 및 상태 복원 기능을 검증
    - 로드맵 Stage 2.5에서 강조된 메모리 시스템 검증
    """
    if not client: pytest.skip("TestClient not available.")
    
    # 1. 첫 번째 작업 실행 - 결과가 메모리에 저장됨
    workflow_name = "simple_prompt_agent"
    user_question = "What's the formula for calculating the area of a circle?"
    
    print(f"\n[E2E Memory Test] Starting with question: '{user_question}'")
    
    request_payload = {
        "graph_config_name": workflow_name,
        "original_input": {"original_input": user_question},
        "initial_metadata": {"test_type": "e2e_real_llm_memory"}
    }
    
    # 첫 번째 작업 시작
    response_run = client.post("/api/v1/run", json=request_payload)
    assert response_run.status_code == 202, f"Run endpoint failed: {response_run.text}"
    task_id = response_run.json().get("task_id")
    assert task_id, "Task ID not found in run response"
    
    print(f"[E2E Memory Test] Task ID: {task_id}, waiting for completion...")
    
    # 작업 완료 대기
    final_status = await poll_for_status(client, task_id)
    assert final_status is not None, f"Polling resulted in None for task {task_id}"
    assert final_status.get("status") == "completed", f"Task {task_id} did not complete successfully"
    
    first_answer = final_status.get("final_answer")
    assert first_answer, "First answer is empty"
    print(f"[E2E Memory Test] First task completed. Answer: {first_answer[:100]}...")
    
    # 2. 메모리에서 작업 상태 직접 확인 (API를 통해)
    print(f"[E2E Memory Test] Checking task state via status API...")
    direct_check_response = client.get(f"/api/v1/status/{task_id}")
    assert direct_check_response.status_code == 200, f"Status endpoint failed: {direct_check_response.text}"
    direct_check_data = direct_check_response.json()
    
    # 3. API 응답과 폴링 결과 비교 (메모리 일관성 검증)
    assert direct_check_data.get("final_answer") == first_answer, "Memory state is inconsistent"
    
    # 4. 작업 상태 가져오기 (30초 후 - 캐싱이 없다면 메모리에서 다시 로드됨)
    print(f"[E2E Memory Test] Waiting 30 seconds before re-checking memory persistence...")
    await asyncio.sleep(30)
    
    # 메모리 지속성 확인
    persistence_check_response = client.get(f"/api/v1/status/{task_id}")
    assert persistence_check_response.status_code == 200, f"Status endpoint failed on persistence check: {persistence_check_response.text}"
    persistence_data = persistence_check_response.json()
    
    assert persistence_data.get("final_answer") == first_answer, "Memory persistence failed - state changed or was lost"
    
    print(f"\n[E2E Memory Test] Memory persistence verified successfully")
    print(f"Initial status API response matched polling result")
    print(f"Status API response after 30s delay matched initial response")
    print(f"Task state was correctly persisted in the memory system")
    
@e2e_real_llm
@pytest.mark.asyncio
async def test_e2e_llm_fallback_scenario(client: Optional[TestClient], monkeypatch):
    if not client: pytest.skip("TestClient not available.")
    if not settings: pytest.skip("Settings object not available for fallback test.")

    if not settings.FALLBACK_LLM_PROVIDER or not settings.LLM_PROVIDERS.get(settings.FALLBACK_LLM_PROVIDER):
        pytest.skip("Fallback LLM provider is not configured in settings. Skipping LLM fallback test.")

    fallback_provider_name = settings.FALLBACK_LLM_PROVIDER
    fallback_provider_config = settings.LLM_PROVIDERS.get(fallback_provider_name)

    if not (fallback_provider_config and fallback_provider_config.api_key):
        pytest.skip(f"Real LLM API key for fallback provider '{fallback_provider_name}' "
                    f"NOT FOUND or EMPTY in 'settings.LLM_PROVIDERS'. Skipping LLM fallback test.")

    workflow_name = "simple_prompt_agent"
    user_question = "Tell me a very short, one-sentence joke using the fallback LLM."

    # Primary LLM의 ainvoke가 실패하도록 모킹합니다.
    # LLMClient가 사용하는 Langchain 모델 (예: ChatOpenAI)의 ainvoke를 패치합니다.
    primary_llm_provider_for_patch = settings.PRIMARY_LLM_PROVIDER
    langchain_model_path_to_patch = ""
    if primary_llm_provider_for_patch == "openai":
        langchain_model_path_to_patch = "langchain_community.chat_models.ChatOpenAI.ainvoke"
    elif primary_llm_provider_for_patch == "anthropic":
        langchain_model_path_to_patch = "langchain_community.chat_models.ChatAnthropic.ainvoke"
    # Gemini 등 다른 provider에 대한 경로 추가 가능
    else:
        pytest.skip(f"Primary LLM provider '{primary_llm_provider_for_patch}' ainvoke patching not implemented for this test.")
        return

    num_primary_failures = settings.LLM_MAX_RETRIES + 1
    primary_failure_side_effect = [asyncio.TimeoutError("Simulated Primary LLM ainvoke Timeout")] * num_primary_failures
    
    mocked_primary_ainvoke = AsyncMock(side_effect=primary_failure_side_effect)

    with monkeypatch.context() as m:
        # m.setattr을 사용하여 실제 Langchain 모델의 ainvoke를 모킹
        # 이 모킹은 LLMClient가 해당 모델의 인스턴스를 생성하고 ainvoke를 호출할 때 적용됩니다.
        try:
            m.setattr(langchain_model_path_to_patch, mocked_primary_ainvoke)
            print(f"\n[MOCKING] Patched '{langchain_model_path_to_patch}' to simulate primary LLM failure.")
        except AttributeError: # 경로가 정확하지 않거나 모듈이 없을 경우
             pytest.skip(f"Could not patch '{langchain_model_path_to_patch}'. Check path or if provider is installed.")
             return


        request_payload = {
            "graph_config_name": workflow_name,
            "original_input": {"original_input": user_question},
            "initial_metadata": {"test_type": "e2e_real_llm_fallback"}
        }

        response_run = client.post("/api/v1/run", json=request_payload)
        assert response_run.status_code == 202, f"Run endpoint failed for fallback test: {response_run.text}"
        task_id = response_run.json().get("task_id")
        assert task_id

        final_status = await poll_for_status(client, task_id, timeout_seconds=240)

    # monkeypatch.context() 블록을 벗어나면 자동으로 원래대로 복구됩니다.

    assert final_status is not None, f"Polling resulted in None for fallback task {task_id}"
    assert final_status.get("status") == "completed", f"Fallback task {task_id} failed: Status '{final_status.get('status')}', Error: {final_status.get('error_message')}, LLM Output: {final_status.get('last_llm_output')}"
    assert final_status.get("error_message") is None

    final_answer = final_status.get("final_answer")
    assert final_answer, "Final answer from fallback is empty"
    assert len(final_answer) > 1, f"Fallback answer seems too short: '{final_answer}'"

    print(f"\n[E2E LLM Fallback Test] Task ID: {task_id}")
    print(f"Question: {user_question}")
    print(f"LLM (Fallback) Answer: {final_answer}")
    print(f"Primary LLM ainvoke was called {mocked_primary_ainvoke.call_count} times (expected to fail {num_primary_failures} times).")

    # Primary LLM이 LLM_MAX_RETRIES + 1 만큼 호출(실패)되었는지 확인
    assert mocked_primary_ainvoke.call_count == num_primary_failures, \
        f"Expected primary LLM to be called {num_primary_failures} times, but was {mocked_primary_ainvoke.call_count}"

    # TODO: 더 나아가서, telemetry나 로그를 통해 실제로 fallback model_name이 사용되었는지 확인하면 더 좋습니다.
    # 예를 들어, Orchestrator가 최종 상태의 metadata에 사용된 LLM 모델 정보를 기록하도록 수정하고,
    # final_status.get("metadata", {}).get("model_used") 와 fallback_model_name_expected 를 비교할 수 있습니다.
    fallback_model_name_expected = fallback_provider_config.model_name
    print(f"Expected fallback model to be used: {fallback_model_name_expected}")
    # 현재 응답 모델에는 이 정보가 없으므로, 이 부분은 추가적인 로깅/메타데이터 저장 구현 후 검증 가능.
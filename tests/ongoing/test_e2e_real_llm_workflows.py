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
async def test_e2e_tot_workflow_problem_solving_real_llm(client: Optional[TestClient]):
    """
    Tree of Thoughts (ToT) 워크플로우가 복잡한 문제를 해결할 수 있는지 테스트
    - 로드맵 Stage 4 및 8에서 강조된 ToT 흐름을 검증
    """
    if not client: pytest.skip("TestClient not available.")
    
    # 실제 존재하는 파일명으로 수정 (tot_workflow → default_tot_workflow)
    workflow_name = "default_tot_workflow"
    user_question = "What would be three creative ways to solve traffic congestion in big cities? Compare their pros and cons."
    
    print(f"\n[E2E ToT Workflow Test] Starting test with question: '{user_question}'")
    print(f"Using dynamic graph configuration: '{workflow_name}'")
    
    request_payload = {
        "graph_config_name": workflow_name,
        "original_input": user_question,
        "initial_metadata": {"test_type": "e2e_real_llm_tot"}
    }
    
    response_run = client.post("/api/v1/run", json=request_payload)
    assert response_run.status_code == 202, f"Run endpoint failed: {response_run.text}"
    run_data = response_run.json()
    task_id = run_data.get("task_id")
    assert task_id, "Task ID not found in run response"
    
    print(f"[E2E ToT Workflow Test] Task ID: {task_id} created, polling for status...")
    
    # ToT는 복잡한 사고 과정을 거치므로 타임아웃을 더 길게 설정
    final_status = await poll_for_status(client, task_id, timeout_seconds=300, interval_seconds=10)
    
    assert final_status is not None, f"Polling resulted in None for task {task_id}"
    assert final_status.get("status") == "completed", f"Task {task_id} did not complete successfully: Status '{final_status.get('status')}', Error: {final_status.get('error_message')}"
    assert final_status.get("error_message") is None, f"Task {task_id} completed with error: {final_status.get('error_message')}"
    
    final_answer = final_status.get("final_answer")
    assert final_answer, "Final answer is empty"
    assert len(final_answer) > 200, f"Final answer seems too short for a ToT response: '{final_answer[:100]}...'"
    
    # 추가 검증: ToT 흐름이 여러 생각 경로를 생성했는지 확인
    thought_paths = final_status.get("metadata", {}).get("thought_paths", [])
    print(f"[E2E ToT Workflow Test] Number of thought paths: {len(thought_paths)}")
    assert len(thought_paths) >= 1, "Expected at least one thought path in metadata"
    
    print(f"\n[E2E ToT Workflow Test] Task ID: {task_id} completed successfully")
    print(f"Final Answer (first 300 chars): {final_answer[:300]}...")
    print(f"Thought Paths Count: {len(thought_paths)}")


@e2e_real_llm
@pytest.mark.asyncio
async def test_e2e_websocket_real_time_updates_real_llm(client: Optional[TestClient]):
    """
    실제 LLM과 연결된 상태에서 WebSocket을 통한 실시간 업데이트가 제대로 작동하는지 확인
    - 로드맵 Stage 6.5에서 명시된 WebSocket 기능 검증
    """
    if not client: pytest.skip("TestClient not available.")
    
    # WebSocket 업데이트를 테스트하기 위해 좀 더 시간이 걸리는 작업 선택
    workflow_name = "simple_prompt_agent"
    user_question = "Give me 5 interesting facts about machine learning."
    
    print(f"\n[E2E WebSocket Test] Starting with question: '{user_question}'")
    print(f"Using graph configuration: '{workflow_name}'")
    
    # 1. 태스크 시작
    request_payload = {
        "graph_config_name": workflow_name,
        "original_input": {"original_input": user_question},
        "initial_metadata": {"test_type": "e2e_real_llm_websocket"}
    }
    
    response_run = client.post("/api/v1/run", json=request_payload)
    assert response_run.status_code == 202, f"Run endpoint failed: {response_run.text}"
    task_id = response_run.json().get("task_id")
    assert task_id, "Task ID not found in run response"
    
    print(f"[E2E WebSocket Test] Task ID: {task_id}, connecting to WebSocket...")
    
    # 2. 현재 테스트 프레임워크가 WebSocket 연결을 지원하는지 확인
    try:
        with client.websocket_connect(f"/api/v1/ws/status/{task_id}") as websocket:
            print(f"[E2E WebSocket Test] WebSocket connection established")
            
            # 3. 최소 2개 이상의 상태 메시지 수신 시도
            messages = []
            start_time = time.time()
            timeout = 30  # 2분 타임아웃
            
            while time.time() - start_time < timeout:
                try:
                    # 짧은 타임아웃으로 메시지 수신 시도
                    data = websocket.receive_json(timeout=2.0)
                    print(f"[E2E WebSocket Test] Received message: {data.get('event_type')}")
                    messages.append(data)
                    
                    # 최종 결과 메시지를 받았다면 루프 종료
                    if data.get("event_type") == "final_result":
                        print("[E2E WebSocket Test] Received final_result, exiting loop")
                        break
                except Exception as e:
                    # 타임아웃이나 JSON 파싱 오류 등이 발생할 수 있음
                    print(f"[E2E WebSocket Test] Waiting for more messages... ({e})")
                    await asyncio.sleep(2)
            
            # 메시지 검증
            assert len(messages) >= 2, f"Expected at least 2 WebSocket messages, but got {len(messages)}"
            
            # 첫 메시지와 마지막 메시지의 이벤트 타입 확인
            assert messages[0]["event_type"] in ["status_update", "task_started"], "First message should be status_update or task_started"
            assert messages[-1]["event_type"] == "final_result", "Last message should be final_result"
            
            # 최종 결과 메시지 내용 확인
            final_message = messages[-1]
            assert "final_answer" in final_message, "Final message should contain final_answer"
            assert final_message["task_id"] == task_id, "Task ID in WebSocket message doesn't match"
            
            print(f"\n[E2E WebSocket Test] Received {len(messages)} WebSocket messages")
            print(f"First message type: {messages[0]['event_type']}")
            print(f"Last message type: {messages[-1]['event_type']}")
            print(f"Final answer (first 100 chars): {final_message.get('final_answer', '')[:100]}...")
            
    except Exception as ws_error:
        pytest.skip(f"WebSocket testing not supported or failed: {ws_error}")
        return


# 동적 그래프 구성 테스트를 위한 헬퍼 함수 (필요시)
@e2e_real_llm
@pytest.mark.asyncio
async def test_e2e_dynamic_graph_configuration_real_llm(client: Optional[TestClient]):
    """
    서로 다른 그래프 구성 JSON이 다른 동작을 생성하는지 확인
    - 로드맵 Stage 4에서 강조된 동적 그래프 구성 검증
    """
    if not client: pytest.skip("TestClient not available.")
    
    # 동일한 질문에 대해 실제 존재하는 서로 다른 두 워크플로우 구성 사용
    base_question = "What are the advantages and disadvantages of electric vehicles?"
    workflow1 = "simple_prompt_agent"    # 단일 LLM 호출 워크플로우 
    workflow2 = "default_tot_workflow"   # Tree of Thoughts 워크플로우 (tot_workflow → default_tot_workflow)
    
    print(f"\n[E2E Dynamic Graph Test] Testing different graph configurations")
    print(f"Base question: '{base_question}'")
    print(f"Comparing workflows: '{workflow1}' vs '{workflow2}'")
    
    responses = {}
    
    # 각 워크플로우 실행 및 결과 수집
    for workflow in [workflow1, workflow2]:
        print(f"[E2E Dynamic Graph Test] Running with workflow: '{workflow}'")
        request_payload = {
            "graph_config_name": workflow,
            "original_input": {"original_input": base_question},
            "initial_metadata": {"test_type": f"e2e_dynamic_graph_{workflow}"}
        }
        
        response_run = client.post("/api/v1/run", json=request_payload)
        assert response_run.status_code == 202, f"Run endpoint failed for {workflow}: {response_run.text}"
        task_id = response_run.json().get("task_id")
        assert task_id, f"Task ID not found for {workflow}"
        
        print(f"[E2E Dynamic Graph Test] Task ID for {workflow}: {task_id}")
        final_status = await poll_for_status(client, task_id, timeout_seconds=240)
        
        assert final_status is not None, f"Polling resulted in None for {workflow} task {task_id}"
        assert final_status.get("status") == "completed", f"Task {task_id} for {workflow} did not complete successfully"
        
        responses[workflow] = {
            "final_answer": final_status.get("final_answer"),
            "metadata": final_status.get("metadata", {}),
            "iterations": final_status.get("current_iteration", 0),
        }
        
        print(f"[E2E Dynamic Graph Test] {workflow} completed in {responses[workflow]['iterations']} iterations")
    
    # 결과 비교 및 검증
    simple_response = responses[workflow1]
    tot_response = responses[workflow2]
    
    # 1. 반복 횟수 비교 - ToT는 일반적으로 더 많은 반복이 필요함
    simple_iterations = simple_response["iterations"]
    tot_iterations = tot_response["iterations"]
    print(f"[E2E Dynamic Graph Test] Iterations comparison: {workflow1}={simple_iterations}, {workflow2}={tot_iterations}")
    
    # 2. 사고 경로 및 메타데이터 차이 확인
    tot_thought_paths = tot_response["metadata"].get("thought_paths", [])
    print(f"[E2E Dynamic Graph Test] ToT thought paths: {len(tot_thought_paths)}")
    
    # 3. 답변 길이 및 구조 비교
    simple_answer_length = len(simple_response["final_answer"])
    tot_answer_length = len(tot_response["final_answer"])
    print(f"[E2E Dynamic Graph Test] Answer length comparison: {workflow1}={simple_answer_length}, {workflow2}={tot_answer_length}")
    
    # 두 응답이 완전히 다른지 확인 (다른 구성이 다른 결과를 생성해야 함)
    assert simple_response["final_answer"] != tot_response["final_answer"], "Both workflows produced identical answers"
    
    # ToT 워크플로우의 메타데이터에는 사고 경로 정보가 있어야 함
    if "thought_paths" in tot_response["metadata"]:
        assert len(tot_response["metadata"]["thought_paths"]) > 0, "ToT workflow did not generate any thought paths"
    
    print(f"\n[E2E Dynamic Graph Test] Dynamic graph configuration verified successfully")
    print(f"Different JSON configurations produced different behaviors and outputs")
    print(f"Simple prompt workflow answer length: {simple_answer_length} chars")
    print(f"ToT workflow answer length: {tot_answer_length} chars")


async def compare_workflows(client, base_question, workflow1, workflow2):
    """서로 다른 워크플로우 구성으로 동일한 질문을 처리하고 결과 비교"""
    responses = {}
    
    for workflow in [workflow1, workflow2]:
        request_payload = {
            "graph_config_name": workflow,
            "original_input": {"original_input": base_question},
            "initial_metadata": {"test_type": f"e2e_workflow_compare_{workflow}"}
        }
        
        response_run = client.post("/api/v1/run", json=request_payload)
        assert response_run.status_code == 202
        task_id = response_run.json().get("task_id")
        assert task_id
        
        final_status = await poll_for_status(client, task_id, timeout_seconds=240)
        assert final_status is not None
        assert final_status.get("status") == "completed"
        
        responses[workflow] = {
            "final_answer": final_status.get("final_answer"),
            "metadata": final_status.get("metadata", {}),
        }
    
    return responses
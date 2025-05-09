# tests/test_graph_nodes.py

import pytest
import os
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# --- 테스트 대상 임포트 ---
from src.agents.graph_nodes.generic_llm_node import GenericLLMNode, ParsedLLMResponse
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager # ToolManager 임포트
from src.schemas.mcp_models import AgentGraphState
from src.tools.base import BaseTool # MockTool 정의 위해 필요
from pydantic import BaseModel, Field # MockTool 정의 위해 필요

# --- Fixtures ---

@pytest.fixture
def mock_llm_client():
    """LLMClient 모의 객체"""
    client = MagicMock(spec=LLMClient)
    client.generate_response = AsyncMock(return_value="Default Mock LLM Response")
    client.primary_llm = MagicMock() # LLMClient의 내부 속성 모킹 추가
    client.primary_llm.model_name = "mock_model"
    client.provider_name = "mock_provider" # provider_name 추가
    return client

@pytest.fixture
def mock_memory_manager():
    """
    Conversation history 조회를 담당하는 MemoryManager의 목 객체.
    get_history() 는 비동기 코루틴이어야 하므로 AsyncMock 으로 정의한다.
    """
    mgr = MagicMock()
    mgr.get_history = AsyncMock(return_value=[])   # 빈 히스토리 반환
    return mgr

# MockTool 정의 (test_services.py와 중복되므로 conftest.py로 옮기는 것이 좋음)
class MockToolInput(BaseModel):
    arg1: str = Field(...)

class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "Mock tool"
    args_schema: type[BaseModel] = MockToolInput
    def _run(self, arg1: str) -> str: return f"Mock run: {arg1}"
    async def _arun(self, arg1: str) -> str: return f"Mock arun: {arg1}"

# CalculatorTool 임포트 (mock_tool_manager_with_tools에서 사용)
from src.tools.calculator import CalculatorTool

@pytest.fixture
def mock_tool_manager_with_tools(): # tool_manager_instance 의존성 제거
    """미리 등록된 도구가 있는 ToolManager 인스턴스 생성"""
    # 각 테스트마다 새로운 manager 생성
    manager = ToolManager(name=f"graph_node_test_manager_{os.urandom(4).hex()}")
    # 필요한 MockTool과 CalculatorTool 등록
    manager.register(MockTool)
    manager.register(CalculatorTool)
    return manager # MagicMock 대신 실제 ToolManager 인스턴스 반환

@pytest.fixture
def basic_agent_state():
    """기본 AgentGraphState 객체"""
    return AgentGraphState(
        task_id="task123",
        original_input="Calculate 5 plus 7",
        dynamic_data={"scratchpad": "Initial state.", "tool_call_history": []}
    )

@pytest.fixture
def react_prompt_content():
    """react_tool_agent.txt 프롬프트 내용 모방"""
    return """
Overall Goal: {original_input}
Available Tools: {available_tools}
Execution History & Scratchpad: {scratchpad}
Tool Call History (if any): {tool_call_history}
Instructions: Think -> Act -> Observation. Output JSON {{"action": "...", "action_input": ...}}.
"""

# --- GenericLLMNode 도구 호출 테스트 (테스트 함수 본문은 동일) ---

@pytest.mark.asyncio
async def test_generic_llm_node_tool_use_disabled(mock_llm_client, mock_tool_manager_with_tools, mock_memory_manager, mock_notification_service, basic_agent_state):
    # ... (기존 테스트 함수 본문 유지) ...
    prompt_content = "Input: {original_input}"
    m = mock_open(read_data=prompt_content)
    with patch("builtins.open", m), \
         patch("os.path.isabs", return_value=False), \
         patch("src.agents.graph_nodes.generic_llm_node.settings", PROMPT_TEMPLATE_DIR="dummy"):

        node = GenericLLMNode(
            llm_client=mock_llm_client,
            tool_manager=mock_tool_manager_with_tools,
            notification_service=mock_notification_service,
            prompt_template_path="simple_prompt.txt",
            output_field_name="final_answer",
            input_keys_for_prompt=["original_input"],
            node_id="test_no_tool_node",
            enable_tool_use=False,
            memory_manager=mock_memory_manager
        )
        mock_llm_client.generate_response.return_value = "Simple answer without tools"
        result_update = await node(basic_agent_state)
        mock_llm_client.generate_response.assert_awaited_once()
        # get_tool 호출 안됨 확인 - ToolManager가 실제 인스턴스이므로 직접 메서드 호출 검증은 어려움
        # 대신 결과 상태를 통해 도구가 호출되지 않았음을 간접 확인
        assert "tool_call_history" not in result_update.get("dynamic_data", {})
        assert result_update.get("final_answer") == "Simple answer without tools"

@pytest.mark.asyncio
async def test_generic_llm_node_single_tool_call_success(mock_llm_client, mock_tool_manager_with_tools, mock_memory_manager, mock_notification_service, basic_agent_state, react_prompt_content):
    # ... (기존 테스트 함수 본문 유지) ...
    m = mock_open(read_data=react_prompt_content)
    with patch("builtins.open", m), \
         patch("os.path.isabs", return_value=False), \
         patch("src.agents.graph_nodes.generic_llm_node.settings", PROMPT_TEMPLATE_DIR="dummy"):

        node = GenericLLMNode(
            llm_client=mock_llm_client,
            tool_manager=mock_tool_manager_with_tools, # 실제 ToolManager 전달
            memory_manager=mock_memory_manager,
            notification_service=mock_notification_service,
            prompt_template_path="react_prompt.txt",
            output_field_name="final_answer",
            input_keys_for_prompt=["original_input", "scratchpad", "tool_call_history", "available_tools"],
            node_id="test_single_tool_node",
            enable_tool_use=True,
            allowed_tools=["calculator"],
        )
        # --- LLM 응답 시나리오 설정 ---
        llm_response_tool_call = json.dumps({"action": "calculator", "action_input": {"expression": "5 + 7"}})
        llm_response_final_answer = json.dumps({"action": "finish", "action_input": "The result of 5 plus 7 is 12."})
        mock_llm_client.generate_response.side_effect = [llm_response_tool_call, llm_response_final_answer]

        result_update = await node(basic_agent_state)

        assert mock_llm_client.generate_response.call_count == 2
        assert result_update.get("final_answer") == "The result of 5 plus 7 is 12."
        assert result_update.get("error_message") is None
        dynamic_data = result_update.get("dynamic_data", {})
        assert "tool_call_history" in dynamic_data
        history = dynamic_data["tool_call_history"]
        assert len(history) == 1
        assert history[0]["tool_name"] == "calculator"
        assert history[0]["args"] == {"expression": "5 + 7"}
        assert history[0]["result"] == "12.0" or history[0]["result"] == "12"
        assert history[0]["error"] is False
        assert "Observation: 12.0" in dynamic_data.get("scratchpad", "") or "Observation: 12" in dynamic_data.get("scratchpad", "")


@pytest.mark.asyncio
async def test_generic_llm_node_parse_response_formats(mock_llm_client, mock_tool_manager_with_tools, mock_memory_manager, mock_notification_service):
    """LLM 응답 파싱 로직 (JSON, Text) 테스트"""
    with patch.object(GenericLLMNode, '_load_prompt_template', return_value="Dummy template content"):
        node = GenericLLMNode(
            llm_client=mock_llm_client,
            tool_manager=mock_tool_manager_with_tools,
            memory_manager=mock_memory_manager,
            notification_service=mock_notification_service,
            prompt_template_path="dummy.txt",
            node_id="test_parser_node",
            enable_tool_use=True,
        )

    # 1. 정상 JSON 응답 (```json 포함)
    json_response_str_block = '```json\n{\n  "action": "web_search",\n  "action_input": {\n    "query": "LangGraph"\n  }\n}\n```'
    parsed_block = node._parse_llm_response(json_response_str_block)
    assert parsed_block is not None
    assert parsed_block["action"] == "web_search"
    assert parsed_block["action_input"] == {"query": "LangGraph"}

    # 1.1 정상 JSON 응답 (```json 없음)
    json_response_str_plain = '{\n  "action": "calculator",\n  "action_input": {\n    "expression": "1+1"\n  }\n}'
    parsed_plain = node._parse_llm_response(json_response_str_plain)
    assert parsed_plain is not None
    assert parsed_plain["action"] == "calculator"
    assert parsed_plain["action_input"] == {"expression": "1+1"}


    # 2. 텍스트 기반 응답 (Action: Args: 패턴 파싱 성공 검증)
    text_response_str_action_args = "Thought: I need to calculate the result.\nAction: calculator\nArgs: {\"expression\": \"10 / 2\"}"
    parsed_text = node._parse_llm_response(text_response_str_action_args)
    # 이제 텍스트 파싱이 성공해야 함
    assert parsed_text is not None
    assert parsed_text["action"] == "calculator"
    assert parsed_text["action_input"] == {"expression": "10 / 2"} # JSON 파싱 성공 가정

    # 2.1 텍스트 기반 응답 (Args가 JSON이 아닐 때)
    text_response_str_action_args_plain = "Thought: Search needed.\nAction: web_search\nArgs: simple query"
    parsed_text_plain = node._parse_llm_response(text_response_str_action_args_plain)
    assert parsed_text_plain is not None
    assert parsed_text_plain["action"] == "web_search"
    assert parsed_text_plain["action_input"] == {"input": "simple query"} # 문자열은 'input' 키로 래핑됨

    # 3. 파싱 실패 케이스
    invalid_response = "Just some text without action."
    parsed_invalid = node._parse_llm_response(invalid_response)
    assert parsed_invalid is None
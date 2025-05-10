import pytest
import asyncio
import json
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from langgraph.graph import END


# 테스트 대상 모듈 임포트
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.agents.orchestrator import Orchestrator, REGISTERED_NODE_TYPES
from src.config.settings import get_settings

# graph_nodes 임포트 (Orchestrator가 사용하므로 필요)
from src.agents.graph_nodes.generic_llm_node import GenericLLMNode
from src.agents.graph_nodes.thought_generator_node import ThoughtGeneratorNode
from src.agents.graph_nodes.state_evaluator_node import StateEvaluatorNode
from src.agents.graph_nodes.search_strategy_node import SearchStrategyNode
from src.services.tool_manager import ToolManager
from src.memory.memory_manager import MemoryManager


# --- 테스트용 Fixtures ---
@pytest.fixture
def mock_llm_client_for_orchestrator():
    client = MagicMock(spec=LLMClient)
    client.generate_response = AsyncMock(return_value="Mocked LLM Response for Orchestrator Test")
    client.primary_llm = MagicMock()
    client.primary_llm.model_name = "mock_orchestrator_model"
    # provider_name 추가 (LLMClient 수정 제안에 따라)
    client.provider_name = "mock_provider"
    return client

@pytest.fixture
def mock_tool_manager():
    return MagicMock(spec=ToolManager)

@pytest.fixture
def mock_memory_manager():
    return MagicMock(spec=MemoryManager)

@pytest.fixture
def orchestrator_instance(mock_llm_client_for_orchestrator, mock_tool_manager, mock_memory_manager, mock_notification_service):
    """테스트용 Orchestrator 인스턴스를 생성합니다."""
    return Orchestrator(llm_client=mock_llm_client_for_orchestrator, tool_manager=mock_tool_manager, memory_manager=mock_memory_manager, notification_service=mock_notification_service)

@pytest.fixture
def simple_graph_config_content():
    """simple_prompt_agent.json 파일의 내용과 유사한 테스트용 설정입니다."""
    return {
        "name": "TestSimplePromptAgentWorkflow",
        "description": "A test simple workflow.",
        "entry_point": "input_node",
        "nodes": [
            {
                "id": "input_node", # Changed to input_node to match entry_point
                "node_type": "generic_llm_node",
                "parameters": {
                    "prompt_template_path": "test/simple_test_prompt.txt",
                    "output_field_name": "final_answer",
                    "input_keys_for_prompt": ["original_input"],
                    "node_id": "test_simple_responder"
                }
            }
        ],
        "edges": [
            {
                "type": "standard",
                "source": "input_node",
                "target": "__end__"
            }
        ]
    }

@pytest.fixture
def tot_graph_config_content():
    """default_tot_workflow.json 파일의 내용과 유사한 테스트용 설정입니다."""
    return {
        "name": "TestDefaultTreeOfThoughtsWorkflow",
        "description": "A test ToT workflow.",
        "entry_point": "thought_generator",
        "nodes": [
            {"id": "thought_generator", "node_type": "thought_generator_node", "parameters": {"num_thoughts": 1, "prompt_template_path": "tot/test_generate.txt", "node_id": "TestToTGen"}},
            {"id": "state_evaluator", "node_type": "state_evaluator_node", "parameters": {"prompt_template_path": "tot/test_evaluate.txt", "node_id": "TestToTEval"}},
            {"id": "search_strategy", "node_type": "search_strategy_node", "parameters": {"beam_width": 1, "score_threshold_to_finish": 0.98, "node_id": "TestToTStrat"}}
        ],
        "edges": [
            {"type": "standard", "source": "thought_generator", "target": "state_evaluator"},
            {"type": "standard", "source": "state_evaluator", "target": "search_strategy"},
            {
                "type": "conditional",
                "source": "search_strategy",
                "condition_key": "final_answer", 
                "targets": {
                    "value_is_not_none": "__end__", 
                    "value_is_none": "thought_generator"
                },
                "default_target": "thought_generator" 
            }
        ]
    }

# --- Orchestrator 테스트 ---

# Updated test_orchestrator_load_graph_config
def test_orchestrator_load_graph_config(orchestrator_instance, simple_graph_config_content, tmp_path):
    """Orchestrator가 JSON 설정 파일을 올바르게 로드하는지 테스트합니다."""
    config_dir = tmp_path / "agent_graphs"
    config_dir.mkdir()
    config_file = config_dir / "test_simple_config.json"
    config_file.write_text(json.dumps(simple_graph_config_content))

    # settings의 AGENT_GRAPH_CONFIG_DIR를 임시 디렉토리로 패치
    # MagicMock 대신 직접 설정하여 파일 로드 로직 테스트
    with patch("src.agents.orchestrator.settings") as mock_settings:
        mock_settings.AGENT_GRAPH_CONFIG_DIR = str(config_dir)
        # 필요한 경우 직접 load_graph_config 구현 제공 가능
        # mock_settings.load_graph_config.return_value = simple_graph_config_content
        
        loaded_config = orchestrator_instance._load_graph_config_from_file("test_simple_config")
        assert loaded_config["name"] == "TestSimplePromptAgentWorkflow"
        assert len(loaded_config["nodes"]) == 1
        
# 필요하다면 다른 테스트도 비슷하게 수정
# 하지만 _load_graph_config_from_file 메서드 수정으로 기존 테스트도 작동해야 함


@pytest.mark.asyncio
async def test_orchestrator_build_simple_graph(orchestrator_instance, simple_graph_config_content):
    """Orchestrator가 간단한 그래프를 올바르게 빌드하는지 테스트합니다."""
    # _create_node_instance 내부의 프롬프트 파일 로딩 모킹
    with patch("builtins.open", mock_open(read_data="Test prompt: {original_input}")) as mocked_prompt_file:
        with patch("os.path.isabs", return_value=False): # 경로 관련 로직을 단순화
             with patch("src.agents.graph_nodes.generic_llm_node.settings", PROMPT_TEMPLATE_DIR="dummy_prompts"):
                graph = orchestrator_instance.build_graph(simple_graph_config_content)
                assert graph is not None
                assert "input_node" in graph.nodes
                assert simple_graph_config_content["entry_point"] in graph.nodes # type: ignore
                mocked_prompt_file.assert_called_once_with(os.path.join("dummy_prompts", "test/simple_test_prompt.txt"), 'r', encoding='utf-8')


@pytest.mark.asyncio
async def test_orchestrator_run_simple_workflow(orchestrator_instance, simple_graph_config_content, tmp_path):
    """Orchestrator가 간단한 워크플로우를 실행하고 예상 결과를 반환하는지 테스트합니다."""
    config_dir = tmp_path / "agent_graphs"
    config_dir.mkdir()
    config_file = config_dir / "simple_workflow.json"
    config_file.write_text(json.dumps(simple_graph_config_content))

    # 프롬프트 파일 모킹
    prompt_dir = tmp_path / "prompts" / "test"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "simple_test_prompt.txt"
    prompt_file.write_text("User query: {original_input} -> LLM says:")

    # LLMClient의 generate_response 모킹
    orchestrator_instance.llm_client.generate_response.return_value = "LLM answer to simple input"

    with patch("src.agents.orchestrator.settings", AGENT_GRAPH_CONFIG_DIR=str(config_dir)):
        # GenericLLMNode 내부에서도 settings.PROMPT_TEMPLATE_DIR를 사용하므로 패치
        with patch("src.agents.graph_nodes.generic_llm_node.settings", PROMPT_TEMPLATE_DIR=str(tmp_path / "prompts")):
            final_state = await orchestrator_instance.run_workflow(
                graph_config_name="simple_workflow",
                task_id="task_simple_123",
                original_input="Hello world"
            )
            assert final_state.task_id == "task_simple_123"
            assert final_state.original_input == "Hello world"
            assert final_state.final_answer == "LLM answer to simple input" # output_field_name에 따라
            assert final_state.error_message is None
            orchestrator_instance.llm_client.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_run_tot_workflow_mocked_llm(orchestrator_instance, tot_graph_config_content, tmp_path):
    """Orchestrator가 ToT 워크플로우를 실행하는지 테스트 (LLM 응답 모킹)."""
    tot_graph_config_content = {
        "description": "A test ToT workflow.",
        "name": "TestDefaultTreeOfThoughtsWorkflow",
        "entry_point": "thought_generator",
        "nodes": [
            {"id": "thought_generator", "node_type": "thought_generator_node", 
             "parameters": {"prompt_template": "tot/test_generate.txt", "node_id": "TestToTGen", "num_thoughts": 1}},
            {"id": "state_evaluator", "node_type": "state_evaluator_node", 
             "parameters": {"prompt_template": "tot/test_evaluate.txt", "node_id": "TestToTEval"}},
            {"id": "search_strategy", "node_type": "search_strategy_node", 
             "parameters": {"node_id": "TestToTStrat", "beam_width": 1, "finish_threshold": 0.98}}
        ],
        "edges": [
            {"source": "thought_generator", "target": "state_evaluator", "type": "standard"},
            {"source": "state_evaluator", "target": "search_strategy", "type": "standard"},
            {"source": "search_strategy", "target": "thought_generator", "type": "conditional", 
             "condition_key": "next_action", "targets": {"continue": "thought_generator", "finish": "__end__"}}
        ]
    }

    config_dir = tmp_path / "agent_graphs"
    config_dir.mkdir()
    config_file = config_dir / "default_tot_workflow.json"
    config_file.write_text(json.dumps(tot_graph_config_content))

    # 프롬프트 파일 모킹
    prompt_tot_dir = tmp_path / "prompts" / "tot"
    prompt_tot_dir.mkdir(parents=True, exist_ok=True)
    (prompt_tot_dir / "test_generate.txt").write_text("Generate thoughts for: {original_input}")
    (prompt_tot_dir / "test_evaluate.txt").write_text("Evaluate thought: {thought_to_evaluate_content}")

    # LLM 응답 모킹 함수
    async def mock_llm_responses(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_content = messages[0]["content"] if messages and messages[0].get("content") else ""

        if "Generate thoughts for" in prompt_content:
            if mock_llm_responses.call_count == 0:
                mock_llm_responses.call_count += 1
                return "Thought: Mocked Thought Alpha"
            else:
                mock_llm_responses.call_count += 1
                return "Thought: Mocked Thought Beta from depth 1"
        elif "Evaluate thought" in prompt_content:
            if "Mocked Thought Alpha" in prompt_content:
                return "Score: 0.85, Reasoning: Alpha seems good."
            elif "Mocked Thought Beta" in prompt_content:
                return "Score: 0.99, Reasoning: Beta is excellent, finish."
            return "Score: 0.5, Reasoning: Default mock eval."
        
        return "Score: 0.96, Reasoning: Unexpected request but valid format."
    
    mock_llm_responses.call_count = 0
    orchestrator_instance.llm_client.generate_response.side_effect = mock_llm_responses

    mock_settings_object = MagicMock()
    setattr(mock_settings_object, 'AGENT_GRAPH_CONFIG_DIR', str(config_dir))
    setattr(mock_settings_object, 'PROMPT_TEMPLATE_DIR', str(tmp_path / "prompts"))

    # 중요: CasePreservingStr 클래스를 표준 str로 패치하는 부분
        # json.dumps의 default 파라미터를 설정하여 직렬화 안전하게 처리
    with patch("src.agents.orchestrator.settings", mock_settings_object), \
         patch("src.agents.graph_nodes.generic_llm_node.settings", mock_settings_object), \
         patch("src.agents.graph_nodes.thought_generator_node.settings", mock_settings_object), \
         patch("src.agents.graph_nodes.state_evaluator_node.settings", mock_settings_object), \
         patch("src.agents.graph_nodes.search_strategy_node.settings", mock_settings_object):

        task_id = "task_tot_mock_123"
        final_state = await orchestrator_instance.run_workflow(
            graph_config_name="default_tot_workflow",
            task_id=task_id,
            original_input="Solve complex problem X"
        )

        # 검증 로직
        assert final_state.task_id == task_id

        assert final_state.error_message is None, f"Workflow errored: {final_state.error_message}"
        assert final_state.final_answer is not None, "Final answer should be set"
        assert orchestrator_instance.llm_client.generate_response.call_count >= 2
        assert final_state is not None
        assert final_state.final_answer is not None
        assert "Score:" in final_state.final_answer
        assert "Reasoning:" in final_state.final_answer
        final_thought = final_state.get_thought_by_id(final_state.current_best_thought_id)
        assert final_thought is not None
        assert "Score:" in final_thought.content
        assert "Reasoning:" in final_thought.content

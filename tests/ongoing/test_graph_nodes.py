import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

# 테스트 대상 모듈 임포트
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState, Thought
from src.agents.graph_nodes.generic_llm_node import GenericLLMNode
from src.agents.graph_nodes.thought_generator_node import ThoughtGeneratorNode
from src.agents.graph_nodes.state_evaluator_node import StateEvaluatorNode
from src.agents.graph_nodes.search_strategy_node import SearchStrategyNode
from src.config.settings import get_settings


# --- 테스트용 Fixtures ---
@pytest.fixture
def mock_llm_client():
    """LLMClient의 모의(mock) 객체를 생성합니다."""
    client = MagicMock(spec=LLMClient)
    # generate_response가 코루틴이므로 AsyncMock으로 설정
    client.generate_response = AsyncMock(return_value="Mocked LLM Response")
    # 실제 LLMClient에 있을 수 있는 다른 속성/메서드도 필요에 따라 모킹
    client.primary_llm = MagicMock()
    client.primary_llm.model_name = "mock_primary_model"
    client.fallback_llm = None # 또는 모킹된 폴백 LLM
    # provider_name 추가 (LLMClient 수정 제안에 따라)
    client.provider_name = "mock_provider"
    return client

@pytest.fixture
def initial_agent_graph_state():
    """테스트용 초기 AgentGraphState 객체를 생성합니다."""
    return AgentGraphState(
        task_id="test_task_001",
        original_input="Test original input for the graph.",
        metadata={"user_id": "test_user"}
    )

# --- GenericLLMNode 테스트 ---
@pytest.mark.asyncio
async def test_generic_llm_node_execution_success(mock_llm_client, initial_agent_graph_state):
    """GenericLLMNode가 성공적으로 실행되고 상태를 업데이트하는지 테스트합니다."""
    prompt_content = "User: {original_input} AI:"
    # Patch 'open' to mock file reading
    with patch("builtins.open", mock_open(read_data=prompt_content)) as mocked_file:
        # Patch os.path.isabs to always return False to force joining with base_prompt_dir
        with patch("os.path.isabs", return_value=False):
            # Patch settings.PROMPT_TEMPLATE_DIR if it's used by the node
            with patch("src.agents.graph_nodes.generic_llm_node.settings", PROMPT_TEMPLATE_DIR="dummy_prompts"):
                node = GenericLLMNode(
                    llm_client=mock_llm_client,
                    prompt_template_path="test_prompt.txt", # 경로는 실제 로직에 맞게
                    output_field_name="dynamic_data.llm_result",
                    input_keys_for_prompt=["original_input"],
                    node_id="test_generic_node"
                )
                
                # 예상되는 LLM 응답 설정
                mock_llm_client.generate_response.return_value = "This is the LLM's answer."

                output_state_update = await node(initial_agent_graph_state)

                mocked_file.assert_called_once_with(os.path.join("dummy_prompts", "test_prompt.txt"), 'r', encoding='utf-8')
                mock_llm_client.generate_response.assert_awaited_once()
                
                # generate_response 호출 시 messages 인자 검증
                args, kwargs = mock_llm_client.generate_response.call_args
                assert "messages" in kwargs
                assert kwargs["messages"] == [{"role": "user", "content": f"User: {initial_agent_graph_state.original_input} AI:"}]

                assert "dynamic_data" in output_state_update
                assert output_state_update["dynamic_data"].get("llm_result") == "This is the LLM's answer."
                assert output_state_update.get("last_llm_input") == f"User: {initial_agent_graph_state.original_input} AI:"
                assert output_state_update.get("error_message") is None

@pytest.mark.asyncio
async def test_generic_llm_node_llm_failure(mock_llm_client, initial_agent_graph_state):
    """GenericLLMNode가 LLM 호출 실패 시 에러 메시지를 상태에 기록하는지 테스트합니다."""
    prompt_content = "Prompt: {original_input}"
    with patch("builtins.open", mock_open(read_data=prompt_content)):
        with patch("os.path.isabs", return_value=False):
            with patch("src.agents.graph_nodes.generic_llm_node.settings", PROMPT_TEMPLATE_DIR="dummy_prompts"):
                node = GenericLLMNode(
                    llm_client=mock_llm_client,
                    prompt_template_path="fail_prompt.txt",
                    output_field_name="dynamic_data.llm_result",
                    input_keys_for_prompt=["original_input"],
                    node_id="test_failure_node"
                )
                mock_llm_client.generate_response.side_effect = Exception("LLM API Unreachable")

                output_state_update = await node(initial_agent_graph_state)

                assert "error_message" in output_state_update
                assert "LLM API Unreachable" in output_state_update["error_message"]
                assert output_state_update.get("dynamic_data", {}).get("llm_result") is None # 또는 output_field_name에 따라 다름


# --- ThoughtGeneratorNode 테스트 ---
@pytest.mark.asyncio
async def test_thought_generator_node_generates_thoughts(mock_llm_client, initial_agent_graph_state):
    """ThoughtGeneratorNode가 여러 생각을 생성하고 상태를 업데이트하는지 테스트합니다."""
    node = ThoughtGeneratorNode(
        llm_client=mock_llm_client,
        num_thoughts=2,
        node_id="test_thought_gen"
    )
    # LLM이 "Thought: ..." 형식으로 응답한다고 가정
    mock_llm_client.generate_response.return_value = "Thought: Idea 1\nThought: Idea 2"
    
    # 초기 상태에 max_search_depth 설정 (노드가 사용하므로)
    initial_agent_graph_state.max_search_depth = 5
    initial_agent_graph_state.search_depth = 0


    output_state_update = await node(initial_agent_graph_state)

    assert mock_llm_client.generate_response.called
    assert "thoughts" in output_state_update
    assert "current_thoughts_to_evaluate" in output_state_update
    
    newly_generated_thoughts = [
        t for t in output_state_update["thoughts"] 
        if t.id in output_state_update["current_thoughts_to_evaluate"]
    ]
    assert len(newly_generated_thoughts) == 2
    assert newly_generated_thoughts[0].content == "Idea 1"
    assert newly_generated_thoughts[1].content == "Idea 2"
    assert output_state_update.get("error_message") is None

@pytest.mark.asyncio
async def test_thought_generator_node_max_depth_reached(mock_llm_client, initial_agent_graph_state):
    """ThoughtGeneratorNode가 최대 탐색 깊이에 도달하면 생각 생성을 중단하는지 테스트합니다."""
    initial_agent_graph_state.search_depth = 5
    initial_agent_graph_state.max_search_depth = 5
    
    node = ThoughtGeneratorNode(llm_client=mock_llm_client, num_thoughts=3)
    output_state_update = await node(initial_agent_graph_state)

    assert not mock_llm_client.generate_response.called # LLM 호출 안됨
    assert output_state_update.get("current_thoughts_to_evaluate") == []
    assert "Max search depth reached" in output_state_update.get("error_message", "")


# --- StateEvaluatorNode 테스트 ---
@pytest.mark.asyncio
async def test_state_evaluator_node_evaluates_thoughts(mock_llm_client, initial_agent_graph_state):
    """StateEvaluatorNode가 생각들을 평가하고 점수를 업데이트하는지 테스트합니다."""
    node = StateEvaluatorNode(llm_client=mock_llm_client, node_id="test_evaluator")

    # 평가할 생각들을 상태에 추가
    thought1 = initial_agent_graph_state.add_thought("Thought content 1")
    thought2 = initial_agent_graph_state.add_thought("Thought content 2")
    initial_agent_graph_state.current_thoughts_to_evaluate = [thought1.id, thought2.id]
    
    # LLM 응답 모킹 (각 생각에 대해 다른 점수 반환)
    mock_llm_client.generate_response.side_effect = [
        "Score: 0.8, Reasoning: Good idea.",
        "Score: 0.4, Reasoning: Less promising."
    ]

    output_state_update = await node(initial_agent_graph_state)

    assert mock_llm_client.generate_response.call_count == 2
    assert "thoughts" in output_state_update
    assert output_state_update.get("current_thoughts_to_evaluate") == [] # 평가 후 비워져야 함

    evaluated_thought1 = next(t for t in output_state_update["thoughts"] if t.id == thought1.id)
    evaluated_thought2 = next(t for t in output_state_update["thoughts"] if t.id == thought2.id)

    assert evaluated_thought1.evaluation_score == 0.8
    assert evaluated_thought1.status == "evaluated"
    assert "Good idea" in evaluated_thought1.metadata.get("eval_reasoning", "").lower()

    assert evaluated_thought2.evaluation_score == 0.4
    assert evaluated_thought2.status == "evaluated"
    assert "Less promising" in evaluated_thought2.metadata.get("eval_reasoning", "").lower()
    
    assert output_state_update.get("error_message") is None


# --- SearchStrategyNode 테스트 ---
@pytest.mark.asyncio
async def test_search_strategy_node_selects_best_and_continues(initial_agent_graph_state):
    """SearchStrategyNode가 최상의 생각을 선택하고 탐색을 계속하는지 테스트합니다."""
    node = SearchStrategyNode(beam_width=1, score_threshold_to_finish=0.95, node_id="test_search_strat")

    # 평가된 생각들을 상태에 추가
    t1 = Thought(id="t1", content="High score thought", evaluation_score=0.9, status="evaluated")
    t2 = Thought(id="t2", content="Medium score thought", evaluation_score=0.7, status="evaluated")
    t3 = Thought(id="t3", content="Low score thought", evaluation_score=0.3, status="evaluated")
    initial_agent_graph_state.thoughts = [t1, t2, t3]
    initial_agent_graph_state.search_depth = 1
    initial_agent_graph_state.max_search_depth = 5
    initial_agent_graph_state.current_best_thought_id = None # 초기에는 없음

    output_state_update = await node(initial_agent_graph_state)

    assert output_state_update.get("current_best_thought_id") == "t1" # 가장 점수가 높은 생각
    assert output_state_update.get("search_depth") == 2 # 깊이 증가
    assert output_state_update.get("final_answer") is None # 아직 종료 아님
    assert output_state_update.get("error_message") is None

@pytest.mark.asyncio
async def test_search_strategy_node_reaches_max_depth(initial_agent_graph_state):
    """SearchStrategyNode가 최대 깊이에 도달하면 종료하는지 테스트합니다."""
    node = SearchStrategyNode(node_id="test_max_depth_strat")

    t1 = Thought(id="t1", content="Final thought at max depth", evaluation_score=0.8, status="evaluated")
    initial_agent_graph_state.thoughts = [t1]
    initial_agent_graph_state.search_depth = 4 # 다음 깊이가 max_depth가 되도록 설정
    initial_agent_graph_state.max_search_depth = 5 
    initial_agent_graph_state.current_best_thought_id = "t1"

    output_state_update = await node(initial_agent_graph_state)

    assert output_state_update.get("search_depth") == 5
    assert output_state_update.get("final_answer") == "Final thought at max depth"
    assert "Max search depth reached" in output_state_update.get("error_message", "") # 또는 None일 수 있음 (final_answer가 설정되면 성공 간주)

@pytest.mark.asyncio
async def test_search_strategy_node_high_score_finish(initial_agent_graph_state):
    """SearchStrategyNode가 높은 점수로 조기 종료하는지 테스트합니다."""
    node = SearchStrategyNode(score_threshold_to_finish=0.9, node_id="test_high_score_finish")

    t1 = Thought(id="t1", content="Excellent thought", evaluation_score=0.95, status="evaluated")
    initial_agent_graph_state.thoughts = [t1]
    initial_agent_graph_state.search_depth = 1
    initial_agent_graph_state.max_search_depth = 5
    initial_agent_graph_state.current_best_thought_id = None

    output_state_update = await node(initial_agent_graph_state)

    assert output_state_update.get("current_best_thought_id") == "t1"
    assert output_state_update.get("final_answer") == "Excellent thought"
    assert output_state_update.get("search_depth") == 1 # 조기 종료했으므로 깊이 유지 (또는 +1 될 수도, 설계에 따라)
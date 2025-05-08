# tests/test_schemas.py
import pytest
import time
import uuid
import msgspec
from pydantic import ValidationError
from typing import Any # Any 임포트 추가

# 테스트 대상 스키마 임포트
from src.schemas.enums import TaskPriority, TaskState # TaskState는 WorkflowStatusResponse에서 문자열로 대체될 수 있음
# from src.schemas.request_models import CreateTaskRequest # 주석 처리 또는 삭제
from src.schemas.request_models import RunWorkflowRequest, ToolExecutionRequest # RunWorkflowRequest 임포트
from src.schemas.response_models import (
    TaskSubmittedResponse,
    # TaskStatusResponse, # 주석 처리 또는 삭제
    WorkflowStatusResponse, # 새로 추가
    ToolInfo, AgentInfo,
    ContextResponse, ContextOperationResponse, HealthCheckResponse,
    GraphInfo # 필요시 추가
)
from src.schemas.mcp_protocol import BaseContextSchema, TaskContext
from src.schemas.mcp_models import (
    LLMInputMessage, LLMParameters, LLMInputContext,
    LLMOutputChoice, LLMUsage, LLMOutputContext, ConversationTurn,
    Thought, AgentGraphState # ToT/AgentGraphState 관련 모델 임포트
)
from src.schemas.agent_graph_config import (
    NodeConfig, EdgeConfig, ConditionalEdgeConfig, AgentGraphConfig
)

# --- Pydantic 모델 테스트 ---

# test_create_task_request_validation 함수는 주석 처리 또는 삭제
# def test_create_task_request_validation():
#     """CreateTaskRequest 모델 유효성 검사 (정상 케이스)"""
#     # ... (기존 코드) ...

def test_run_workflow_request_validation():
    """RunWorkflowRequest 모델 유효성 검사 (정상 케이스)"""
    data = {
        "graph_config_name": "my_test_workflow",
        "original_input": {"query": "Hello?", "user_id": 123},
        "initial_metadata": {"priority": "high"}
    }
    req = RunWorkflowRequest(**data)
    assert req.graph_config_name == "my_test_workflow"
    assert req.original_input == {"query": "Hello?", "user_id": 123}
    assert req.initial_metadata == {"priority": "high"}
    assert req.task_id is None # Optional 필드는 기본값이 None

    # 필수 필드 누락 시 에러 확인
    with pytest.raises(ValidationError):
        RunWorkflowRequest(original_input="test") # graph_config_name 누락
    with pytest.raises(ValidationError):
        RunWorkflowRequest(graph_config_name="test") # original_input 누락

def test_tool_execution_request_validation():
    """ToolExecutionRequest 모델 유효성 검사"""
    data = {"args": {"param1": "value1", "count": 5}}
    req = ToolExecutionRequest(**data)
    assert req.args["param1"] == "value1"
    assert req.args["count"] == 5

def test_response_models_instantiation():
    """Response 모델들이 정상적으로 인스턴스화되는지 확인"""
    # TaskSubmittedResponse: status가 'accepted'로 변경됨
    task_submit_res = TaskSubmittedResponse(task_id="task_123") # status 기본값 사용
    assert task_submit_res.task_id == "task_123"
    assert task_submit_res.status == "accepted"

    # TaskStatusResponse 대신 WorkflowStatusResponse 테스트
    workflow_status_res = WorkflowStatusResponse(
        task_id="task_456",
        status="completed", # 문자열 상태 사용
        final_answer="Workflow finished successfully.",
        current_iteration=5,
        metadata={"model_used": "gpt-4"}
    )
    assert workflow_status_res.status == "completed"
    assert workflow_status_res.final_answer == "Workflow finished successfully."
    assert workflow_status_res.current_iteration == 5

    # 나머지 응답 모델 테스트는 기존과 동일하게 유지 가능
    tool_info = ToolInfo(name="calculator", description="Calculates things", args_schema_summary={"expression": "string (required)"})
    assert tool_info.name == "calculator"
    assert tool_info.args_schema_summary == {"expression": "string (required)"} # args_schema_summary 검증 추가

    agent_info = AgentInfo(name="planner", agent_type="mcp_planner", version="1.0")
    assert agent_info.agent_type == "mcp_planner"

    context_res = ContextResponse(context_id="ctx_abc", data={"key": "value"})
    assert context_res.context_id == "ctx_abc"

    context_op_res = ContextOperationResponse(context_id="ctx_abc", status="updated", message="Context updated.")
    assert context_op_res.status == "updated"

    health_res = HealthCheckResponse(status="ok")
    assert health_res.status == "ok"

    graph_info = GraphInfo(name="my_graph", description="A test graph")
    assert graph_info.name == "my_graph"


# --- MCP 스키마 테스트 ---

def test_base_context_schema_instantiation():
    """BaseContextSchema 기본 인스턴스화 및 serialize 테스트"""
    ctx = BaseContextSchema()
    assert isinstance(ctx.context_id, str)
    assert is_valid_uuid(ctx.context_id) # UUID 형식인지 확인 (utils.ids 필요)
    assert isinstance(ctx.timestamp, float)
    assert ctx.metadata == {}
    serialized = ctx.serialize()
    assert serialized['context_id'] == ctx.context_id
    assert serialized['version'] == '1.0.0'

def test_task_context_instantiation():
    """TaskContext 인스턴스화 테스트"""
    task_id = f"task_{uuid.uuid4()}"
    data = {
        "task_id": task_id,
        "task_type": "planning",
        "input_data": {"goal": "plan the test"},
        "metadata": {"priority": 3}
    }
    ctx = TaskContext(**data)
    assert ctx.task_id == task_id
    assert ctx.task_type == "planning"
    assert ctx.input_data["goal"] == "plan the test"
    serialized = ctx.serialize()
    assert serialized['task_id'] == task_id

# --- msgspec 모델 테스트 ---

# msgspec 인코더/디코더 (기존 유지)
msgpack_encoder = msgspec.msgpack.Encoder()
json_encoder = msgspec.json.Encoder()

# 디코더는 특정 타입 대신 Any 또는 object를 사용하여 유연하게 처리할 수 있음
msgpack_llm_input_decoder = msgspec.msgpack.Decoder(LLMInputContext)
json_llm_input_decoder = msgspec.json.Decoder(LLMInputContext)
msgpack_llm_output_decoder = msgspec.msgpack.Decoder(LLMOutputContext)
json_llm_output_decoder = msgspec.json.Decoder(LLMOutputContext)
msgpack_conv_turn_decoder = msgspec.msgpack.Decoder(ConversationTurn)

def test_llm_input_context_msgspec_roundtrip():
    """LLMInputContext의 msgspec 직렬화/역직렬화 왕복 테스트"""
    params = LLMParameters(max_tokens=100, temperature=0.7)
    # content가 list of dict 형태일 경우도 테스트 (Anthropic, Gemini 등)
    messages = [
        LLMInputMessage(role="user", content="Hello"),
        LLMInputMessage(role="assistant", content="Hi there!"),
        LLMInputMessage(role="user", content=[{"type": "text", "text": "How are you?"}])
    ]
    ctx = LLMInputContext(model="test-model", messages=messages, parameters=params, use_cache=False)

    # MessagePack
    encoded_msgpack = msgpack_encoder.encode(ctx)
    decoded_msgpack = msgpack_llm_input_decoder.decode(encoded_msgpack)
    assert ctx == decoded_msgpack
    assert decoded_msgpack.model == "test-model"
    assert not decoded_msgpack.use_cache
    assert decoded_msgpack.parameters.temperature == 0.7
    assert len(decoded_msgpack.messages) == 3
    assert decoded_msgpack.messages[0].content == "Hello"
    assert isinstance(decoded_msgpack.messages[2].content, list)
    assert decoded_msgpack.messages[2].content[0]['text'] == "How are you?"

    # JSON
    encoded_json = json_encoder.encode(ctx)
    decoded_json = json_llm_input_decoder.decode(encoded_json)
    assert ctx == decoded_json
    assert decoded_json.model == "test-model"
    assert not decoded_json.use_cache
    assert decoded_json.parameters.temperature == 0.7
    assert len(decoded_json.messages) == 3
    assert decoded_json.messages[0].content == "Hello"
    assert isinstance(decoded_json.messages[2].content, list)
    assert decoded_json.messages[2].content[0]['text'] == "How are you?"

def test_llm_output_context_msgspec_roundtrip():
    """LLMOutputContext의 msgspec 직렬화/역직렬화 왕복 테스트"""
    usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    choice = LLMOutputChoice(text="World", index=0, finish_reason="stop")
    ctx = LLMOutputContext(success=True, result_text="World", choices=[choice], usage=usage, model_used="test-model")

    # MessagePack
    encoded_msgpack = msgpack_encoder.encode(ctx)
    decoded_msgpack = msgpack_llm_output_decoder.decode(encoded_msgpack)
    assert ctx == decoded_msgpack
    assert decoded_msgpack.success
    assert decoded_msgpack.result_text == "World"
    assert decoded_msgpack.usage.total_tokens == 30
    assert decoded_msgpack.choices[0].finish_reason == "stop"

    # JSON
    encoded_json = json_encoder.encode(ctx)
    decoded_json = json_llm_output_decoder.decode(encoded_json)
    assert ctx == decoded_json
    assert decoded_json.success
    assert decoded_json.result_text == "World"
    assert decoded_json.usage.total_tokens == 30
    assert decoded_json.choices[0].finish_reason == "stop"

def test_conversation_turn_msgspec():
    """ConversationTurn msgspec 모델 테스트"""
    turn = ConversationTurn(role="assistant", content="How can I help?")
    assert turn.role == "assistant"
    encoded = msgpack_encoder.encode(turn)
    decoded = msgpack_conv_turn_decoder.decode(encoded)
    assert turn == decoded

def test_agent_graph_state_msgspec_roundtrip():
    """AgentGraphState의 msgspec 직렬화/역직렬화 왕복 테스트"""
    thought1 = Thought(content="Initial idea")
    thought2 = Thought(content="Refined idea", parent_id=thought1.id, evaluation_score=0.8, status="evaluated")
    state = AgentGraphState(
        task_id="graph_task_1",
        original_input={"problem": "Solve complex issue"},
        thoughts=[thought1, thought2],
        current_best_thought_id=thought2.id,
        search_depth=1,
        dynamic_data={"plan_step": 2, "intermediate_result": "Partial solution"}
    )

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(AgentGraphState)

    encoded = encoder.encode(state)
    decoded = decoder.decode(encoded)

    assert state == decoded
    assert decoded.task_id == "graph_task_1"
    assert len(decoded.thoughts) == 2
    assert decoded.thoughts[1].evaluation_score == 0.8
    assert decoded.dynamic_data["plan_step"] == 2


# --- Agent Graph Config 스키마 테스트 (기존 유지) ---

def test_agent_graph_config_validation_success():
    """정상적인 AgentGraphConfig 유효성 검사"""
    config_data = {
        "name": "Test Workflow",
        "description": "A simple test workflow.",
        "entry_point": "start_node",
        "nodes": [
            {"id": "start_node", "node_type": "input_parser", "parameters": {}},
            {"id": "tool_node", "node_type": "calculator", "parameters": {"expression": "1+1"}},
            {"id": "final_node", "node_type": "output_formatter", "parameters": {}}
        ],
        "edges": [
            {"type": "standard", "source": "start_node", "target": "tool_node"},
            {
                "type": "conditional",
                "source": "tool_node",
                "condition_key": "dynamic_data.tool_result_status", # 예시: dynamic_data 사용
                "targets": {
                    "success": "final_node",
                    "error": "__end__"
                },
                "default_target": "__end__"
            }
        ]
    }
    config = AgentGraphConfig(**config_data)
    assert config.name == "Test Workflow"
    assert len(config.nodes) == 3
    assert len(config.edges) == 2
    assert isinstance(config.edges[1], ConditionalEdgeConfig)
    assert config.edges[1].condition_key == "dynamic_data.tool_result_status"

def test_agent_graph_config_validation_failure_missing_node():
    """엣지에서 참조하는 노드가 없을 때 유효성 검사 실패"""
    config_data = {
        "name": "Invalid Workflow",
        "entry_point": "start_node",
        "nodes": [
            {"id": "start_node", "node_type": "input_parser"}
        ],
        "edges": [
            {"type": "standard", "source": "start_node", "target": "missing_node"} # 존재하지 않는 노드
        ]
    }
    with pytest.raises(ValueError, match="Target node ID 'missing_node' not found"):
        AgentGraphConfig(**config_data)

def test_agent_graph_config_validation_failure_bad_entrypoint():
    """잘못된 Entry Point ID 유효성 검사 실패"""
    config_data = {
        "name": "Invalid Entry",
        "entry_point": "wrong_entry", # 존재하지 않는 노드
        "nodes": [
            {"id": "start_node", "node_type": "input_parser"}
        ],
        "edges": []
    }
    with pytest.raises(ValueError, match="Entry point 'wrong_entry' does not match"):
        AgentGraphConfig(**config_data)

# --- Helper Functions ---
def is_valid_uuid(uuid_to_test: str, version=4) -> bool:
    """Checks if a string is a valid UUID."""
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test
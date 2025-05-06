# tests/test_schemas.py
import pytest
import time
import uuid
import msgspec
from pydantic import ValidationError

# 테스트 대상 스키마 임포트
from src.schemas.enums import TaskPriority, TaskState
from src.schemas.request_models import CreateTaskRequest, ToolExecutionRequest
from src.schemas.response_models import (
    TaskSubmittedResponse, TaskStatusResponse, ToolInfo, AgentInfo,
    ContextResponse, ContextOperationResponse, HealthCheckResponse
)
from src.schemas.mcp_protocol import BaseContextSchema, TaskContext
from src.schemas.mcp_models import (
    LLMInputMessage, LLMParameters, LLMInputContext,
    LLMOutputChoice, LLMUsage, LLMOutputContext, ConversationTurn
)
from src.schemas.agent_graph_config import (
    NodeConfig, EdgeConfig, ConditionalEdgeConfig, AgentGraphConfig
)

# --- Pydantic 모델 테스트 ---

def test_create_task_request_validation():
    """CreateTaskRequest 모델 유효성 검사 (정상 케이스)"""
    data = {
        "goal": "Test the API",
        "task_type": "api_test",
        "input_data": {"url": "/test"},
        "priority": TaskPriority.HIGH, # Enum 사용
        "metadata": {"user": "tester"}
    }
    req = CreateTaskRequest(**data)
    assert req.goal == "Test the API"
    assert req.priority == TaskPriority.HIGH

    # 정수 Priority 테스트
    data_int_prio = {**data, "priority": 3} # HIGH에 해당하는 정수 값 사용 (Enum 정의 참고)
    req_int = CreateTaskRequest(**data_int_prio)
    assert req_int.priority == TaskPriority.HIGH # Pydantic이 Enum으로 변환해야 함

    # 필수 필드 누락 시 에러 발생하는지 확인
    with pytest.raises(ValidationError):
        CreateTaskRequest() # goal 누락

def test_tool_execution_request_validation():
    """ToolExecutionRequest 모델 유효성 검사"""
    data = {"args": {"param1": "value1", "count": 5}}
    req = ToolExecutionRequest(**data)
    assert req.args["param1"] == "value1"
    assert req.args["count"] == 5

def test_response_models_instantiation():
    """Response 모델들이 정상적으로 인스턴스화되는지 확인"""
    task_submit_res = TaskSubmittedResponse(task_id="task_123", status="submitted")
    assert task_submit_res.task_id == "task_123"

    task_status_res = TaskStatusResponse(id="task_123", state=TaskState.COMPLETED, result={"output": "done"})
    assert task_status_res.state == TaskState.COMPLETED

    tool_info = ToolInfo(name="calculator", description="Calculates things", args_schema_summary={"expression": "string"})
    assert tool_info.name == "calculator"

    agent_info = AgentInfo(name="planner", agent_type="mcp_planner", version="1.0")
    assert agent_info.agent_type == "mcp_planner"

    context_res = ContextResponse(context_id="ctx_abc", data={"key": "value"})
    assert context_res.context_id == "ctx_abc"

    context_op_res = ContextOperationResponse(context_id="ctx_abc", status="updated", message="Context updated.")
    assert context_op_res.status == "updated"

    health_res = HealthCheckResponse(status="ok")
    assert health_res.status == "ok"

# --- MCP 스키마 테스트 ---

def test_base_context_schema_instantiation():
    """BaseContextSchema 기본 인스턴스화 및 serialize 테스트"""
    ctx = BaseContextSchema()
    assert isinstance(ctx.context_id, str)
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

# msgspec 인코더/디코더
msgpack_encoder = msgspec.msgpack.Encoder()
msgpack_decoder = msgspec.msgpack.Decoder(LLMInputContext) # 특정 타입 지정 또는 Any 사용 가능
json_encoder = msgspec.json.Encoder()
json_decoder = msgspec.json.Decoder(LLMInputContext)

def test_llm_input_context_msgspec_roundtrip():
    """LLMInputContext의 msgspec 직렬화/역직렬화 왕복 테스트"""
    params = LLMParameters(max_tokens=100, temperature=0.7)
    messages = [LLMInputMessage(role="user", content="Hello")]
    ctx = LLMInputContext(model="test-model", messages=messages, parameters=params, use_cache=False)

    # MessagePack
    encoded_msgpack = msgpack_encoder.encode(ctx)
    decoded_msgpack = msgspec.msgpack.decode(encoded_msgpack, type=LLMInputContext)
    assert ctx == decoded_msgpack
    assert decoded_msgpack.model == "test-model"
    assert not decoded_msgpack.use_cache
    assert decoded_msgpack.parameters.temperature == 0.7
    assert decoded_msgpack.messages[0].content == "Hello"

    # JSON
    encoded_json = json_encoder.encode(ctx)
    decoded_json = msgspec.json.decode(encoded_json, type=LLMInputContext)
    assert ctx == decoded_json
    assert decoded_json.model == "test-model"
    assert not decoded_json.use_cache
    assert decoded_json.parameters.temperature == 0.7
    assert decoded_json.messages[0].content == "Hello"

def test_llm_output_context_msgspec_roundtrip():
    """LLMOutputContext의 msgspec 직렬화/역직렬화 왕복 테스트"""
    usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    choice = LLMOutputChoice(text="World", index=0, finish_reason="stop")
    ctx = LLMOutputContext(success=True, result_text="World", choices=[choice], usage=usage, model_used="test-model")

    # MessagePack
    encoded_msgpack = msgpack_encoder.encode(ctx)
    decoded_msgpack = msgspec.msgpack.decode(encoded_msgpack, type=LLMOutputContext)
    assert ctx == decoded_msgpack
    assert decoded_msgpack.success
    assert decoded_msgpack.result_text == "World"
    assert decoded_msgpack.usage.total_tokens == 30
    assert decoded_msgpack.choices[0].finish_reason == "stop"

    # JSON
    encoded_json = json_encoder.encode(ctx)
    decoded_json = msgspec.json.decode(encoded_json, type=LLMOutputContext)
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
    decoded = msgspec.msgpack.decode(encoded, type=ConversationTurn)
    assert turn == decoded

# --- Agent Graph Config 스키마 테스트 ---

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
                "condition_key": "tool_result_status",
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
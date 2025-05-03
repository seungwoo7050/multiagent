import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.mcp.schema import BaseContextSchema, TaskContext
from src.core.mcp.adapters.agent_adapter import AgentAdapter, AgentInputContext, AgentOutputContext
from src.core.mcp.adapters.llm_adapter import LLMAdapter, LLMInputContext
from src.core.mcp.adapters.memory_adapter import MemoryAdapter, MemoryInputContext
from src.core.agent import BaseAgent, AgentResult, AgentContext
from src.core.task import BaseTask, TaskState
from src.memory.base import BaseMemory
from src.memory.manager import MemoryManager
from src.config.errors import LLMError, ErrorCode

@pytest.fixture(autouse=True)
def mock_env_vars():
    mock_environ = {
        "GEMINI_API_KEY": "fake-gemini-key",
        "OPENAI_API_KEY": "fake-openai-key",
        "ANTHROPIC_API_KEY": "fake-anthropic-key"
    }
    with patch.dict(os.environ, mock_environ, clear=True):
        yield

class ConcreteAgentAdapter(AgentAdapter):
    async def adapt_output(self, result: AgentResult, input_context: AgentInputContext) -> AgentOutputContext:
        error_message = None
        if result.error and isinstance(result.error, dict):
            error_message = result.error.get("message")
        elif isinstance(result.error, Exception):
             error_message = str(result.error)

        metadata = result.metadata or {}
        agent_name = metadata.get("agent_name")

        return AgentOutputContext(
            context_id=input_context.context_id,
            success=result.success,
            output_data=result.output,
            error_message=error_message,
            agent_name=agent_name,
            metadata=metadata
        )

class TestAdapterBase:
    @pytest.fixture
    def mock_context(self) -> BaseContextSchema:
        return BaseContextSchema(
            context_id="test-context-id",
            metadata={"test_key": "test_value"}
        )

    @pytest.fixture
    def mock_task_context(self) -> TaskContext:
        return TaskContext(
            context_id="test-task-context-id",
            task_id="task-123",
            task_type="test_task",
            input_data={"param1": "value1"}
        )


class TestAgentAdapter(TestAdapterBase):
    @pytest.fixture
    def mock_agent(self) -> BaseAgent:
        agent = MagicMock(spec=BaseAgent)
        agent.execute = AsyncMock(return_value=AgentResult(
            success=True,
            output={"result": "Test agent output"},
            execution_time=0.5,
            metadata={"agent_name": "test_agent"}
        ))
        return agent

    @pytest.fixture
    def agent_adapter(self, mock_agent) -> ConcreteAgentAdapter:
        return ConcreteAgentAdapter(target_component=mock_agent)

    @pytest.fixture
    def agent_input_context(self) -> AgentInputContext:
        return AgentInputContext(
            context_id="test-agent-context",
            agent_type="test_agent",
            parameters={"param1": "value1"},
            metadata={"trace_id": "trace-123"}
        )

    @pytest.mark.asyncio
    async def test_adapt_input(self, agent_adapter, agent_input_context):
        core_context = await agent_adapter.adapt_input(agent_input_context)

        assert isinstance(core_context, AgentContext)
        assert core_context.parameters["param1"] == "value1"
        assert core_context.metadata["trace_id"] == "trace-123"
        assert core_context.task is None

    @pytest.mark.asyncio
    async def test_adapt_input_with_task_context(self, agent_adapter, mock_task_context):
        task_adapter = AsyncMock()
        mock_task = BaseTask(id="task-123", type="test_task", state=TaskState.PENDING)
        task_adapter.adapt_input = AsyncMock(return_value=mock_task)

        agent_adapter._task_adapter = task_adapter

        agent_input = AgentInputContext(
            context_id="test-agent-context",
            agent_type="test_agent",
            task_context=mock_task_context
        )

        core_context = await agent_adapter.adapt_input(agent_input)

        assert core_context.task is not None
        assert core_context.task.id == "task-123"
        assert core_context.task.type == "test_task"

        task_adapter.adapt_input.assert_called_once_with(mock_task_context)

    @pytest.mark.asyncio
    async def test_adapt_output(self, agent_adapter, agent_input_context):
        agent_result = AgentResult(
            success=True,
            output={"key1": "value1", "key2": 123},
            execution_time=0.25,
            metadata={"agent_name": "test_agent"}
        )

        output_context = await agent_adapter.adapt_output(agent_result, agent_input_context)

        assert output_context.success is True
        assert output_context.output_data == {"key1": "value1", "key2": 123}
        assert output_context.error_message is None
        assert output_context.agent_name == "test_agent"
        assert output_context.context_id == agent_input_context.context_id

    @pytest.mark.asyncio
    async def test_adapt_output_with_error(self, agent_adapter, agent_input_context):
        agent_result = AgentResult(
            success=False,
            output={},
            error={"message": "Test error message", "type": "test_error"},
            execution_time=0.25,
            metadata={"agent_name": "test_agent"}
        )

        output_context = await agent_adapter.adapt_output(agent_result, agent_input_context)

        assert output_context.success is False
        assert output_context.error_message == "Test error message"
        assert output_context.agent_name == "test_agent"


class TestLLMAdapter(TestAdapterBase):
    @pytest.fixture
    def mock_llm_provider(self):
        provider = AsyncMock()
        provider.model = "test-model"
        provider.execute = AsyncMock(return_value={
            "id": "llm-response-123",
            "choices": [{"message": {"content": "Test LLM response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        })
        return provider

    @pytest.fixture
    def llm_adapter(self, mock_llm_provider):
         # This patching might still need adjustment depending on how LLMAdapter internally uses settings
        with patch("src.llm.get_adapter", return_value=mock_llm_provider), \
             patch("src.llm.selector.select_models", return_value=("test-model", [])), \
             patch("src.llm.parallel.execute_with_fallbacks",
                   return_value=("test-model", {"id": "llm-response-123",
                                                "choices": [{"message": {"content": "Test LLM response"}}],
                                                "usage": {"prompt_tokens": 10, "completion_tokens": 20}})):
            return LLMAdapter()


    @pytest.fixture
    def llm_input_context(self):
        return LLMInputContext(
            context_id="test-llm-context",
            model="test-model",
            prompt="Test prompt",
            parameters={"temperature": 0.7, "max_tokens": 100}
        )

    @pytest.mark.asyncio
    async def test_adapt_input(self, llm_adapter, llm_input_context):
        call_args = await llm_adapter.adapt_input(llm_input_context)

        assert call_args["primary_model"] == "test-model"
        assert call_args["prompt"] == "Test prompt"
        assert call_args["temperature"] == 0.7
        assert call_args["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_adapt_output(self, llm_adapter, llm_input_context):
        llm_response = {
            "id": "llm-response-123",
            "model": "test-model",
            "choices": [{"message": {"content": "Test LLM response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }

        output_context = await llm_adapter.adapt_output(llm_response, llm_input_context)

        assert output_context.success is True
        assert output_context.result_text == "Test LLM response"
        assert output_context.model_used == "test-model"
        assert output_context.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert output_context.context_id == llm_input_context.context_id

    @pytest.mark.asyncio
    async def test_adapt_output_with_error(self, llm_adapter, llm_input_context):
        llm_error = LLMError(
            message="API error",
            model="test-model",
            code=ErrorCode.LLM_PROVIDER_ERROR
        )

        output_context = await llm_adapter.adapt_output(llm_error, llm_input_context, model_used="test-model")

        assert output_context.success is False
        assert output_context.error_message == "API error"
        assert output_context.model_used == "test-model"

    @pytest.fixture
    def mock_openai_adapter(self, monkeypatch):
        """OpenAI 어댑터를 모킹하여 테스트 응답 반환"""
        mock_response = {
            "id": "test-response-id",
            "object": "chat.completion",
            "created": 1746253556,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "message": {"content": "Test LLM response"},
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        # 비동기 함수 모킹용 메서드 생성
        async def async_mock_execute(*args, **kwargs):
            return "gpt-3.5-turbo", mock_response
        
        # LLMAdapter._execute_llm_request 메서드를 모킹
        with patch('src.core.mcp.adapters.llm_adapter.LLMAdapter._execute_llm_request', 
                new=AsyncMock(side_effect=async_mock_execute)):
            yield
    
    @pytest.mark.asyncio
    async def test_process_with_mcp(self, llm_adapter, llm_input_context, mock_openai_adapter):
        output_context = await llm_adapter.process_with_mcp(llm_input_context)
        
        assert output_context.success is True
        assert output_context.result_text == "Test LLM response"


class TestMemoryAdapter(TestAdapterBase):
    @pytest.fixture
    def mock_memory_manager(self):
        manager = AsyncMock(spec=MemoryManager)
        manager.load = AsyncMock(return_value={"key": "value"})
        manager.save = AsyncMock(return_value=True)
        manager.delete = AsyncMock(return_value=True)
        manager.exists = AsyncMock(return_value=True)
        manager.search_vectors = AsyncMock(return_value=[{"id": "vec1", "text": "Test text", "score": 0.95}])
        manager.vector_store = AsyncMock()
        return manager

    @pytest.fixture
    def memory_adapter(self, mock_memory_manager):
        return MemoryAdapter(target_component=mock_memory_manager)

    @pytest.fixture
    def load_input_context(self):
        return MemoryInputContext(
            context_id="test-memory-context",
            operation="load",
            key="test-key",
            metadata={"use_cache": True}
        )

    @pytest.fixture
    def save_input_context(self):
        return MemoryInputContext(
            context_id="test-memory-context",
            operation="save",
            key="test-key",
            data={"field1": "value1"},
            ttl=3600
        )

    @pytest.fixture
    def vector_search_context(self):
        return MemoryInputContext(
            context_id="test-memory-context",
            operation="search_vectors",
            query="test query",
            k=5
        )

    @pytest.mark.asyncio
    async def test_adapt_load_operation(self, memory_adapter, load_input_context):
        args_dict, method_name = await memory_adapter._adapt_load_operation(load_input_context)

        assert method_name == "load"
        assert args_dict["key"] == "test-key"
        assert args_dict["context_id"] == "test-memory-context"
        assert args_dict["use_cache"] is True

    @pytest.mark.asyncio
    async def test_adapt_save_operation(self, memory_adapter, save_input_context):
        args_dict, method_name = await memory_adapter._adapt_save_operation(save_input_context)

        assert method_name == "save"
        assert args_dict["key"] == "test-key"
        assert args_dict["context_id"] == "test-memory-context"
        assert args_dict["data"] == {"field1": "value1"}
        assert args_dict["ttl"] == 3600

    @pytest.mark.asyncio
    async def test_adapt_vector_search_operation(self, memory_adapter, vector_search_context):
        args_dict, method_name = await memory_adapter._adapt_search_vectors_operation(vector_search_context)

        assert method_name == "search_vectors"
        assert args_dict["query"] == "test query"
        assert args_dict["k"] == 5
        assert args_dict["context_id"] == "test-memory-context"

    @pytest.mark.asyncio
    async def test_adapt_input(self, memory_adapter, load_input_context):
        result = await memory_adapter.adapt_input(load_input_context)

        assert result["operation"] == "load"
        assert result["args"]["key"] == "test-key"
        assert result["args"]["context_id"] == "test-memory-context"

    @pytest.mark.asyncio
    async def test_adapt_output_load(self, memory_adapter, load_input_context):
        result_data = {"key": "value"}
        output_context = await memory_adapter.adapt_output(result_data, load_input_context, operation="load")

        assert output_context.success is True
        assert output_context.result == {"key": "value"}
        assert output_context.error_message is None
        assert output_context.metadata["operation_performed"] == "load"

    @pytest.mark.asyncio
    async def test_adapt_output_error(self, memory_adapter, load_input_context):
        error = Exception("Test memory error")
        output_context = await memory_adapter.adapt_output(error, load_input_context, operation="load")

        assert output_context.success is False
        assert output_context.result is None
        assert output_context.error_message == "Test memory error"

    @pytest.mark.asyncio
    async def test_process_with_mcp(self, memory_adapter, load_input_context):
        output_context = await memory_adapter.process_with_mcp(load_input_context)

        assert output_context.success is True
        assert output_context.result == {"key": "value"}
        assert output_context.metadata["operation_performed"] == "load"


class TestAdapterPerformance:
    @pytest.mark.asyncio
    async def test_agent_adapter_performance(self):
        agent = MagicMock(spec=BaseAgent)
        agent.execute = AsyncMock()
        adapter = ConcreteAgentAdapter(agent)

        context = AgentInputContext(
            context_id="perf-test",
            agent_type="test_agent"
        )

        start_time = asyncio.get_event_loop().time()
        await adapter.adapt_input(context)
        end_time = asyncio.get_event_loop().time()

        assert (end_time - start_time) < 0.001

    @pytest.mark.asyncio
    async def test_llm_adapter_performance(self):
        adapter = LLMAdapter() # Note: This might still try to access settings on init depending on implementation

        context = LLMInputContext(
            context_id="perf-test",
            model="test-model",
            prompt="test"
        )

        output = {"choices": [{"message": {"content": "test"}}]}

        start_time = asyncio.get_event_loop().time()
        # Assuming adapt_output doesn't rely heavily on settings after mocking env vars
        await adapter.adapt_output(output, context)
        end_time = asyncio.get_event_loop().time()

        assert (end_time - start_time) < 0.001

    @pytest.mark.asyncio
    async def test_memory_adapter_performance(self):
        memory = MagicMock(spec=BaseMemory) # Use spec here
        memory.exists = AsyncMock(return_value=True)
        adapter = MemoryAdapter(memory)

        context = MemoryInputContext(
            context_id="perf-test",
            operation="exists",
            key="test-key"
        )

        adapter._adapt_exists_operation = AsyncMock(return_value=({"key": "test-key", "context_id": "perf-test"}, "exists"))

        start_time = asyncio.get_event_loop().time()
        await adapter.adapt_input(context)
        end_time = asyncio.get_event_loop().time()

        assert (end_time - start_time) < 0.001


if __name__ == "__main__":
    pytest.main(["-v", __file__])
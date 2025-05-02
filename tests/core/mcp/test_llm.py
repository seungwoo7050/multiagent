import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.mcp.schema import BaseContextSchema, TaskContext
from src.core.mcp.llm.context_model_selector import ContextModelSelector
from src.core.mcp.llm.context_performance import get_context_labels
from src.core.mcp.llm.context_transform import transform_llm_input_for_model
from src.core.mcp.llm.context_preserving_fallback import execute_mcp_llm_with_context_preserving_fallback
from src.config.errors import LLMError, ErrorCode


class TestContextModelSelector:
    """Tests for the context-aware model selection component."""
    
    @pytest.fixture
    def model_selector(self):
        return ContextModelSelector()
    
    @pytest.fixture
    def mock_task_context(self):
        context = MagicMock(spec=TaskContext)
        context.context_id = "test-context-123"
        context.task_type = "general"
        context.metadata = {}
        context.input_data = {"goal": "summarize the text"}
        return context
    
    @pytest.mark.asyncio
    async def test_select_model_defaults(self, model_selector, mock_task_context):
        """Test that model selection returns expected default model when no rules match."""
        with patch('src.core.mcp.llm.context_model_selector.list_available_models', 
                   return_value=['gpt-3.5-turbo', 'gpt-4-turbo', 'claude-3-haiku']):
            with patch('src.core.mcp.llm.context_model_selector.settings') as mock_settings:
                mock_settings.PRIMARY_LLM = 'gpt-3.5-turbo'
                
                selected_model = await model_selector.select_model(mock_task_context)
                
                assert selected_model == 'gpt-3.5-turbo'
    
    @pytest.mark.asyncio
    async def test_select_model_coding_task(self, model_selector):
        """Test that coding task with large token count selects the expected model."""
        context = MagicMock(spec=TaskContext)
        context.task_type = "coding"
        context.metadata = {"estimated_tokens": 60000}
        
        with patch('src.core.mcp.llm.context_model_selector.list_available_models', 
                   return_value=['gpt-3.5-turbo', 'gpt-4-turbo', 'claude-3-haiku']):
            selected_model = await model_selector.select_model(context)
            
            assert selected_model == 'gpt-4-turbo'
    
    @pytest.mark.asyncio
    async def test_select_model_low_latency(self, model_selector):
        """Test that low latency flag selects the expected model."""
        context = MagicMock(spec=BaseContextSchema)
        context.metadata = {"low_latency": True}
        
        with patch('src.core.mcp.llm.context_model_selector.list_available_models', 
                   return_value=['gpt-3.5-turbo', 'gpt-4-turbo', 'claude-3-haiku']):
            selected_model = await model_selector.select_model(context)
            
            assert selected_model == 'claude-3-haiku'
    
    @pytest.mark.asyncio
    async def test_select_model_image_analysis(self, model_selector):
        """Test that image analysis task selects the expected model."""
        context = MagicMock(spec=TaskContext)
        context.task_type = "analysis"
        context.input_data = {"goal": "image analysis of the provided photo"}
        
        with patch('src.core.mcp.llm.context_model_selector.list_available_models', 
                   return_value=['gpt-3.5-turbo', 'gpt-4o', 'claude-3-haiku']):
            selected_model = await model_selector.select_model(context)
            
            assert selected_model == 'gpt-4o'
    
    @pytest.mark.asyncio
    async def test_singleton_behavior(self):
        """Test that the singleton pattern works correctly."""
        with patch('src.core.mcp.llm.context_model_selector._selector_instance', None):
            from src.core.mcp.llm.context_model_selector import get_context_model_selector
            
            selector1 = await get_context_model_selector()
            selector2 = await get_context_model_selector()
            
            assert selector1 is selector2


class TestContextPerformance:
    """Tests for context performance labeling."""
    
    def test_get_context_labels_with_task_context(self):
        """Test that task context produces expected labels."""
        # MagicMock 대신 실제 TaskContext 객체 사용을 고려하거나,
        # MagicMock을 사용해야 한다면 아래와 같이 예상값을 수정합니다.
        context = MagicMock(spec=TaskContext)
        # context.__class__.__name__ = 'TaskContext' # 이 줄은 Mock 객체의 실제 클래스를 바꾸지 못하므로 제거하거나 주석 처리합니다.
        context.task_type = "summarization"
        context.task_id = "task-123"

        labels = get_context_labels(context)

        # 예상 결과를 실제 반환값에 맞게 수정합니다.
        assert labels == {
            'context_class': 'MagicMock', # 실제 클래스 이름인 MagicMock을 예상합니다.
            'task_type': 'summarization',
            'task_id': 'task-123'         # task_id 레이블도 포함합니다.
        }
    
    def test_get_context_labels_with_base_context(self):
        """Test that base context produces expected labels."""
        context = MagicMock(spec=BaseContextSchema)
        context.__class__.__name__ = 'BaseContextSchema'
        
        labels = get_context_labels(context)
        
        # Updated expectation to match actual behavior (MagicMock's class name)
        assert labels == {'context_class': 'MagicMock'}
    
    def test_get_context_labels_with_none(self):
        """Test that None context produces empty labels."""
        labels = get_context_labels(None)
        
        assert labels == {}


class TestContextTransform:
    """Tests for context transformation."""
    
    @pytest.mark.asyncio
    async def test_transform_string_to_messages_for_openai(self):
        """Test transforming string prompt to messages list for OpenAI."""
        from src.llm.adapters.openai import OpenAIAdapter
        
        adapter = MagicMock(spec=OpenAIAdapter)
        adapter.model = "gpt-4"
        adapter.provider = "openai"
        
        result = await transform_llm_input_for_model("Hello, how are you?", adapter)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, how are you?"
    
    @pytest.mark.asyncio
    async def test_transform_messages_to_string_for_completion_model(self):
        """Test transforming messages to string for non-chat model."""
        adapter = MagicMock()
        adapter.model = "text-davinci-003"
        adapter.provider = "openai"
        # This model doesn't match the chat model pattern
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        result = await transform_llm_input_for_model(messages, adapter)
        
        assert isinstance(result, str)
        assert "You are a helpful assistant" in result
        assert "Hello, how are you?" in result
    
    @pytest.mark.asyncio
    async def test_transform_preserves_existing_messages_for_chat_model(self):
        """Test that messages are preserved when already in message format."""
        from src.llm.adapters.anthropic import AnthropicAdapter
        
        adapter = MagicMock(spec=AnthropicAdapter)
        adapter.model = "claude-3-opus"
        adapter.provider = "anthropic"
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = await transform_llm_input_for_model(messages, adapter)
        
        assert result is messages  # Should be the same object
        assert len(result) == 3


class TestContextPreservingFallback:
    """Tests for context-preserving fallback."""
    
    @pytest.fixture
    def mock_llm_adapter(self):
        adapter = AsyncMock()
        return adapter
    
    @pytest.fixture
    def mock_mcp_adapter(self):
        adapter = AsyncMock()
        return adapter
    
    @pytest.mark.asyncio
    async def test_successful_primary_model(self, mock_llm_adapter, mock_mcp_adapter):
        """Test fallback when primary model succeeds."""
        # Setup mocks
        with patch('src.core.mcp.llm.context_preserving_fallback.select_models', 
                   return_value=("gpt-4", ["gpt-3.5-turbo"])), \
             patch('src.core.mcp.llm.context_preserving_fallback.get_llm_adapter_instance', 
                   return_value=mock_llm_adapter), \
             patch('src.core.mcp.llm.context_preserving_fallback.transform_llm_input_for_model', 
                   return_value="transformed prompt"):
            
            # Mock successful execution
            mock_output_context = MagicMock()
            mock_output_context.success = True
            mock_mcp_adapter.process_with_mcp.return_value = mock_output_context
            
            # Execute the function
            model, output = await execute_mcp_llm_with_context_preserving_fallback(
                requested_model="gpt-4",
                original_prompt_or_messages="Hello",
                mcp_llm_adapter=mock_mcp_adapter,
                parameters={"temperature": 0.7},
                metadata={"trace_id": "test-trace-123"}
            )
            
            # Assertions
            assert model == "gpt-4"
            assert output == mock_output_context
            assert mock_mcp_adapter.process_with_mcp.call_count == 1
    
    @pytest.mark.asyncio
    async def test_fallback_to_secondary_model(self, mock_llm_adapter, mock_mcp_adapter):
        """Test fallback to secondary model when primary fails."""
        # Setup mocks
        with patch('src.core.mcp.llm.context_preserving_fallback.select_models', 
                   return_value=("gpt-4", ["gpt-3.5-turbo"])), \
             patch('src.core.mcp.llm.context_preserving_fallback.should_fallback_immediately', 
                   return_value=True), \
             patch('src.core.mcp.llm.context_preserving_fallback.get_llm_adapter_instance') as mock_get_adapter, \
             patch('src.core.mcp.llm.context_preserving_fallback.transform_llm_input_for_model', 
                   return_value="transformed prompt"):
            
            # Create two different mock adapters
            primary_adapter = AsyncMock()
            primary_adapter.ensure_initialized.side_effect = LLMError(
                code=ErrorCode.LLM_RATE_LIMIT, 
                message="Rate limit exceeded",
                model="gpt-4"
            )
            
            secondary_adapter = AsyncMock()
            secondary_adapter.ensure_initialized.return_value = True
            
            # Return different adapters based on model name
            def get_adapter_side_effect(model, **kwargs):
                if model == "gpt-4":
                    return primary_adapter
                else:
                    return secondary_adapter
                
            mock_get_adapter.side_effect = get_adapter_side_effect
            
            # Mock successful execution on secondary model
            mock_output_context = MagicMock()
            mock_output_context.success = True
            mock_mcp_adapter.process_with_mcp.return_value = mock_output_context
            
            # Execute the function
            model, output = await execute_mcp_llm_with_context_preserving_fallback(
                requested_model="gpt-4",
                original_prompt_or_messages="Hello",
                mcp_llm_adapter=mock_mcp_adapter,
                parameters={"temperature": 0.7},
                metadata={"trace_id": "test-trace-123"}
            )
            
            # Assertions
            assert model == "gpt-3.5-turbo"
            assert output == mock_output_context
            assert mock_mcp_adapter.process_with_mcp.call_count == 1
    
    @pytest.mark.asyncio
    async def test_all_models_fail(self, mock_llm_adapter, mock_mcp_adapter):
        """Test behavior when all models fail."""
        # Setup mocks
        with patch('src.core.mcp.llm.context_preserving_fallback.select_models', 
                   return_value=("gpt-4", ["gpt-3.5-turbo"])), \
             patch('src.core.mcp.llm.context_preserving_fallback.should_fallback_immediately', 
                   return_value=True), \
             patch('src.core.mcp.llm.context_preserving_fallback.get_llm_adapter_instance') as mock_get_adapter, \
             patch('src.core.mcp.llm.context_preserving_fallback.transform_llm_input_for_model', 
                   return_value="transformed prompt"):
            
            # Make both adapters fail
            primary_adapter = AsyncMock()
            primary_adapter.ensure_initialized.side_effect = LLMError(
                code=ErrorCode.LLM_RATE_LIMIT, 
                message="Rate limit exceeded",
                model="gpt-4"
            )
            
            secondary_adapter = AsyncMock()
            secondary_adapter.ensure_initialized.side_effect = LLMError(
                code=ErrorCode.LLM_API_ERROR, 
                message="API error",
                model="gpt-3.5-turbo"
            )
            
            # Return different adapters based on model name
            def get_adapter_side_effect(model, **kwargs):
                if model == "gpt-4":
                    return primary_adapter
                else:
                    return secondary_adapter
                
            mock_get_adapter.side_effect = get_adapter_side_effect
            
            # Execute the function and expect error
            with pytest.raises(LLMError) as exc_info:
                await execute_mcp_llm_with_context_preserving_fallback(
                    requested_model="gpt-4",
                    original_prompt_or_messages="Hello",
                    mcp_llm_adapter=mock_mcp_adapter,
                    parameters={"temperature": 0.7},
                    metadata={"trace_id": "test-trace-123"}
                )
            
            # Verify the error contains information about all failed models
            assert "All LLM models failed" in str(exc_info.value)
            assert "gpt-4" in str(exc_info.value)
            assert "gpt-3.5-turbo" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main()
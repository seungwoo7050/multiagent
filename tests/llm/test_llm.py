"""
Comprehensive integration tests for the LLM package.

These tests verify the functionality of LLM adapters, fallback mechanisms,
connection pooling, caching, and other core LLM functionality.

Note: Some tests require valid API keys to be set in environment variables.
"""

import os
import time
import asyncio
import pytest
from unittest import mock
from typing import Dict, List, Any


from src.llm.base import BaseLLMAdapter
from src.llm.adapters import get_adapter
from src.llm.adapters.openai import OpenAIAdapter
from src.llm.adapters.anthropic import AnthropicAdapter
from src.llm.adapters.gemini import GeminiAdapter
from src.llm.cache import get_cache, clear_cache
from src.llm.connection_pool import get_connection_pool, cleanup_connection_pools
from src.llm.tokenizer import count_tokens, get_token_limit
from src.llm.models import get_model_info, list_available_models
from src.llm.fallback import execute_llm_with_fallback
from src.llm.parallel import execute_parallel, race_models
from src.llm.factory import get_llm_factory
from src.config.errors import LLMError
from src.config.settings import get_settings

settings = get_settings()

# Test helpers
async def fake_success_operation():
    await asyncio.sleep(0.1)
    return {"success": True, "value": 42}

async def fake_error_operation():
    await asyncio.sleep(0.1)
    raise ValueError("Test error")

async def fake_timeout_operation():
    await asyncio.sleep(10)
    return {"success": True}

# Fixtures
@pytest.fixture
def test_prompt():
    return "Write a haiku about testing."

@pytest.fixture
def test_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant for testing."},
        {"role": "user", "content": "Write a haiku about testing."}
    ]

@pytest.fixture
async def openai_adapter():
    adapter = get_adapter("gpt-3.5-turbo", api_key="sk-test-openai-key")

    adapter = get_adapter("gpt-3.5-turbo", api_key="sk-test-openai-key")
    await adapter.ensure_initialized()
    return adapter

@pytest.fixture
async def anthropic_adapter():
    """Create an Anthropic adapter instance."""
    adapter = get_adapter("claude-3-sonnet", api_key="sk-test-anthropic-key")
    await adapter.ensure_initialized()
    return adapter

@pytest.fixture
async def gemini_adapter():
    """Create a Gemini adapter instance."""
    adapter = get_adapter("gemini-pro", api_key="test-gemini-key")
    await adapter.ensure_initialized()
    return adapter

@pytest.fixture
async def llm_cache():
    """Get the LLM cache and clear it before/after tests."""
    cache = await get_cache()
    await clear_cache()
    return cache

@pytest.fixture
def mock_openai_adapter():
    """Create a properly mocked OpenAI adapter."""
    mock_adapter = mock.MagicMock(spec=OpenAIAdapter)
    mock_adapter.model = "gpt-3.5-turbo"
    mock_adapter.provider = "openai"
    mock_adapter.initialized = True
    mock_adapter.ensure_initialized.return_value = True
    
    # Mock the generate method
    mock_adapter.generate.return_value = {
        "choices": [{"text": "Mock response"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    }
    
    # Mock the _generate_text method (for internal patching)
    mock_adapter._generate_text = mock.AsyncMock()
    mock_adapter._generate_text.return_value = {
        "choices": [{"text": "Mock response"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    }
    
    return mock_adapter


class TestLLMAdapters:
    """Tests for the LLM adapter functionality."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(self):
        """Test that adapters can be properly initialized."""
        # Use direct API key passing to bypass environment variable issues
        openai_adapter = get_adapter("gpt-3.5-turbo", api_key="sk-test-openai-key")
        assert isinstance(openai_adapter, OpenAIAdapter)
        assert openai_adapter.model == "gpt-3.5-turbo"
        assert openai_adapter.provider == "openai"
        
        # Test Anthropic adapter
        anthropic_adapter = get_adapter("claude-3-haiku", api_key="sk-test-anthropic-key")
        assert isinstance(anthropic_adapter, AnthropicAdapter)
        assert anthropic_adapter.model == "claude-3-haiku"
        assert anthropic_adapter.provider == "anthropic"
        
        # Test Gemini adapter
        gemini_adapter = get_adapter("gemini-pro", api_key="test-gemini-key")
        assert isinstance(gemini_adapter, GeminiAdapter)
        assert gemini_adapter.model == "gemini-pro"
        assert gemini_adapter.provider == "gemini"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="No OpenAI API key available."
    )
    async def test_openai_text_generation(self, openai_adapter, test_prompt):
        """Test OpenAI text generation with a basic prompt."""
        response = await openai_adapter.generate(prompt=test_prompt)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "text" in response["choices"][0]
        assert len(response["choices"][0]["text"]) > 0
        assert "usage" in response
        assert response["usage"]["prompt_tokens"] > 0
        assert response["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="No Anthropic API key available."
    )
    async def test_anthropic_text_generation(self, anthropic_adapter, test_prompt):
        """Test Anthropic text generation with a basic prompt."""
        response = await anthropic_adapter.generate(prompt=test_prompt)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "text" in response["choices"][0]
        assert len(response["choices"][0]["text"]) > 0
        assert "usage" in response
        assert response["usage"]["prompt_tokens"] > 0
        assert response["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="No Gemini API key available."
    )
    async def test_gemini_text_generation(self, gemini_adapter, test_prompt):
        """Test Gemini text generation with a basic prompt."""
        response = await gemini_adapter.generate(prompt=test_prompt)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "text" in response["choices"][0]
        assert len(response["choices"][0]["text"]) > 0
        assert "usage" in response
        assert response["usage"]["prompt_tokens"] > 0
        assert response["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="No OpenAI API key available."
    )
    async def test_adapter_chat_messages(self, openai_adapter, test_messages):
        """Test adapter handling of chat message formats."""
        response = await openai_adapter.generate(prompt=test_messages)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "text" in response["choices"][0]
        assert len(response["choices"][0]["text"]) > 0

    @pytest.mark.asyncio
    async def test_adapter_health_check(self, openai_adapter):
        """Test adapter health check functionality."""
        adapter = await openai_adapter
        health_status = await adapter.health_check()
        
        assert "status" in health_status
        assert "adapter" in health_status
        assert "model" in health_status
        assert "provider" in health_status
        assert "latency_sec" in health_status
        
        assert health_status["adapter"] == "OpenAIAdapter"
        assert health_status["model"] == "gpt-3.5-turbo"
        assert health_status["provider"] == "openai"

class TestLLMFallbackAndParallel:
    """Tests for LLM fallback and parallel execution mechanisms."""

    @pytest.mark.asyncio
    async def test_fallback_mechanism_mocked(self):
        """Test the fallback mechanism with mocked adapters."""
        # Create a mock for primary model that fails
        primary_mock = mock.MagicMock(spec=BaseLLMAdapter)
        primary_mock.model = "gpt-4o"
        primary_mock.provider = "openai"
        primary_mock.generate.side_effect = LLMError(
            code="LLM_TIMEOUT",
            message="OpenAI request timed out"
        )
        
        # Create a mock for fallback model that succeeds
        fallback_mock = mock.MagicMock(spec=BaseLLMAdapter)
        fallback_mock.model = "claude-3-haiku"
        fallback_mock.provider = "anthropic"
        fallback_mock.generate.return_value = {
            "choices": [{"text": "Fallback model response"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        
        # Patch adapter creation
        with mock.patch("src.llm.fallback.get_llm_adapter_instance") as mock_get_adapter:
            # Return different adapters based on model name
            def side_effect(model, **kwargs):
                if model == "gpt-4o":
                    return primary_mock
                else:
                    return fallback_mock
            
            mock_get_adapter.side_effect = side_effect
            
            # Patch model selection to return our test models
            with mock.patch("src.llm.fallback.select_models") as mock_select:
                mock_select.return_value = ("gpt-4o", ["claude-3-haiku"])
                
                # Execute with fallback
                model_name, response = await execute_llm_with_fallback(
                    prompt="Test prompt",
                    requested_model="gpt-4o"
                )
                
                # Verify correct model was used
                assert model_name == "claude-3-haiku"
                assert response["choices"][0]["text"] == "Fallback model response"
                
                # Verify the primary model was attempted first
                primary_mock.generate.assert_called_once()
                fallback_mock.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_parallel_operations(self):
        """Test parallel execution of multiple operations."""
        # Create test operations
        operations = [
            fake_success_operation,
            fake_success_operation,
            fake_error_operation,
            fake_success_operation,
        ]
        
        # Execute operations in parallel with return_exceptions=True
        results = await execute_parallel(
            operations=operations,
            timeout=1.0,
            return_exceptions=True
        )
        
        # Verify results
        assert len(results) == 4
        assert isinstance(results[2], ValueError)
        assert results[0]["success"] is True
        assert results[1]["success"] is True
        assert results[3]["success"] is True

    @pytest.mark.asyncio
    async def test_race_models_mocked(self):
        """Test racing multiple models with mocked adapters."""
        # Create mocks for different model adapters
        fast_mock = mock.MagicMock(spec=BaseLLMAdapter)
        fast_mock.model = "gpt-3.5-turbo"
        fast_mock.provider = "openai"
        fast_mock.generate.return_value = {
            "choices": [{"text": "Fast model response"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        
        slow_mock = mock.MagicMock(spec=BaseLLMAdapter)
        slow_mock.model = "gpt-4o"
        slow_mock.provider = "openai"
        slow_mock.generate.side_effect = lambda **kwargs: asyncio.sleep(0.5) and {
            "choices": [{"text": "Slow model response"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15}
        }
        
        # Patch adapter creation
        with mock.patch("src.llm.parallel._create_adapters_concurrently") as mock_create:
            mock_create.return_value = {"gpt-3.5-turbo": fast_mock, "gpt-4o": slow_mock}
            # Return different adapters based on model name
            def side_effect(model, **kwargs):
                if model == "gpt-3.5-turbo":
                    return fast_mock
                else:
                    return slow_mock
            
            # Test race_models
            winner, response = await race_models(
                models=["gpt-3.5-turbo", "gpt-4o"],
                prompt="Test prompt",
                timeout=1.0
            )
            
            # Verify the fast model won
            assert winner == "gpt-3.5-turbo"
            assert response["choices"][0]["text"] == "Fast model response"

class TestLLMCache:
    """Tests for LLM response caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit_miss(self, llm_cache, openai_adapter, test_prompt):
        """Test cache hit and miss behavior."""
        # Await the fixtures to get the actual objects
        cache = await llm_cache
        adapter = await openai_adapter
        
        # Create a mock for the adapter's _generate_text method
        with mock.patch.object(adapter, '_generate_text') as mock_generate:
            mock_generate.return_value = {
                "choices": [{"text": "Test response"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            }
            
            # First call - should miss cache and call _generate_text
            response1 = await adapter.generate(
                prompt=test_prompt,
                use_cache=True
            )
            
            # Second call with same parameters - should hit cache
            response2 = await adapter.generate(
                prompt=test_prompt,
                use_cache=True
            )
            
            # Verify _generate_text was only called once
            assert mock_generate.call_count == 1
            assert response1["choices"][0]["text"] == response2["choices"][0]["text"]
            
            # Call with different parameters - should miss cache
            await adapter.generate(
                prompt=test_prompt + " Different prompt.",
                use_cache=True
            )
            
            # Verify _generate_text was called again
            assert mock_generate.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_ttl(self, llm_cache):
        """Test cache TTL functionality."""
        # Await the fixture to get the actual cache object
        cache = await llm_cache
        
        # Add an item with a short TTL
        test_key = "ttl_test_key"
        test_value = {"result": "test"}
        
        # Set with 1 second TTL
        await cache.set(test_key, test_value, ttl=1)
        
        # Immediate get should succeed
        cached_value = await cache.get(test_key)
        assert cached_value is not None
        assert cached_value["result"] == "test"
        
        # Wait for TTL to expire
        await asyncio.sleep(1.5)
        
        # Get after expiry should fail
        expired_value = await cache.get(test_key)
        assert expired_value is None

    @pytest.mark.asyncio
    async def test_cache_clear(self, llm_cache):
        """Test cache clearing functionality."""
        # Await the fixture to get the actual cache object
        cache = await llm_cache
        
        # Add some items to the cache
        await cache.set("key1", {"value": 1})
        await cache.set("key2", {"value": 2})
        
        # Verify items are in cache
        assert await cache.get("key1") is not None
        assert await cache.get("key2") is not None
        
        # Clear cache
        cleared = await cache.clear()
        assert cleared is True
        
        # Verify items are gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

class TestLLMConnections:
    """Tests for LLM connection management."""

    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self):
        """Test that connection pools are created once and reused."""
        # Get connection pools for the same provider
        pool1 = await get_connection_pool("openai")
        pool2 = await get_connection_pool("openai")
        
        # Should be the same object
        assert pool1 is pool2
        
        # Get pool for a different provider
        pool3 = await get_connection_pool("anthropic")
        
        # Should be a different object
        assert pool1 is not pool3

    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self):
        """Test connection pool cleanup."""
        # Create a connection pool
        pool = await get_connection_pool("openai")
        assert not pool.closed
        
        # Clean up pools
        await cleanup_connection_pools()
        
        # Pool should be closed
        assert pool.closed

class TestTokenizing:
    """Tests for token counting and model token limits."""

    @pytest.mark.asyncio
    async def test_token_counting(self):
        """Test token counting accuracy."""
        test_text = "This is a simple test sentence to count tokens."
        
        # Count tokens for different models
        openai_tokens = await count_tokens("gpt-3.5-turbo", test_text)
        anthropic_tokens = await count_tokens("claude-3-haiku", test_text)
        gemini_tokens = await count_tokens("gemini-pro", test_text)
        
        # All should give similar results within a small range
        assert abs(openai_tokens - anthropic_tokens) <= 3
        assert abs(openai_tokens - gemini_tokens) <= 3
        
        # Basic sanity check - token count should be less than word count * 2
        # (assuming words are roughly 1-2 tokens)
        word_count = len(test_text.split())
        assert openai_tokens <= word_count * 2

    def test_token_limits(self):
        """Test token limit retrieval for different models."""
        # Get token limits
        gpt4_limit = get_token_limit("gpt-4o")
        claude_limit = get_token_limit("claude-3-opus")
        gemini_limit = get_token_limit("gemini-pro")
        
        # Verify values are reasonable
        assert gpt4_limit >= 8000
        assert claude_limit >= 100000
        assert gemini_limit >= 30000
        
        # Models with larger context windows should have higher limits
        assert claude_limit > gpt4_limit

class TestModelsAndFactory:
    """Tests for model registry and factory functionality."""

    def test_model_info_retrieval(self):
        """Test model information retrieval."""
        # Get info for different models
        gpt_info = get_model_info("gpt-4o")
        claude_info = get_model_info("claude-3-opus")
        gemini_info = get_model_info("gemini-pro")
        
        # Verify basic info is present
        for info in [gpt_info, claude_info, gemini_info]:
            assert "provider" in info
            assert "token_limit" in info
            assert "capabilities" in info
        
        # Verify correct providers
        assert gpt_info["provider"] == "openai"
        assert claude_info["provider"] == "anthropic"
        assert gemini_info["provider"] == "gemini"

    def test_list_available_models(self):
        """Test listing available models with filters."""
        # Get all models
        all_models = list_available_models()
        assert len(all_models) > 0
        
        # Get OpenAI models only
        openai_models = list_available_models(provider="openai")
        assert all(model.startswith("gpt") for model in openai_models)
        
        # Get models with reasoning capability
        reasoning_models = list_available_models(required_capabilities=["reasoning"])
        assert len(reasoning_models) > 0
        
        # Verify filtering by token limit
        high_capacity_models = list_available_models(min_token_limit=100000)
        for model in high_capacity_models:
            model_info = get_model_info(model)
            assert model_info["token_limit"] >= 100000

    @pytest.mark.asyncio
    async def test_llm_factory(self):
        """Test LLM factory for creating adapters."""
        # Get factory
        factory = await get_llm_factory()
        
        # Create adapters for different models
        with mock.patch("src.llm.factory.get_llm_adapter_instance") as mock_get_adapter:
            # Set up mock to return appropriate adapter type
            def side_effect(model, **kwargs):
                if model.startswith("gpt"):
                    adapter = mock.MagicMock(spec=OpenAIAdapter)
                elif model.startswith("claude"):
                    adapter = mock.MagicMock(spec=AnthropicAdapter)
                else:
                    adapter = mock.MagicMock(spec=GeminiAdapter)
                
                adapter.model = model
                adapter.initialized = True
                adapter.ensure_initialized.return_value = True
                return adapter
            
            mock_get_adapter.side_effect = side_effect
            
            # Create adapters
            openai_adapter = await factory.create_adapter("gpt-4o")
            anthropic_adapter = await factory.create_adapter("claude-3-opus")
            gemini_adapter = await factory.create_adapter("gemini-pro")
            
            # Verify adapter creation
            assert mock_get_adapter.call_count == 3
            assert openai_adapter.model == "gpt-4o"
            assert anthropic_adapter.model == "claude-3-opus"
            assert gemini_adapter.model == "gemini-pro"

class TestPerformance:
    """Performance benchmarks for LLM package."""

    @pytest.mark.asyncio
    async def test_latency_benchmark(self, openai_adapter):
        """Benchmark LLM request latency."""
        # Await the fixture to get the actual adapter
        adapter = await openai_adapter
        
        with mock.patch.object(adapter, '_generate_text') as mock_generate:
            mock_generate.return_value = {
                "choices": [{"text": "Fast response"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
            }
            
            # Warm-up
            await adapter.generate(prompt="Test", use_cache=False)
            
            # Benchmark
            start_time = time.time()
            iterations = 10
            
            for _ in range(iterations):
                await adapter.generate(prompt="Test", use_cache=False)
            
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            
            # Log benchmark results
            print(f"\nLatency benchmark results:")
            print(f"Average request time: {avg_time:.4f}s")
            print(f"Requests per second: {iterations/total_time:.2f}")
            
            # Basic assertion to ensure the test ran
            assert total_time > 0

    @pytest.mark.asyncio
    async def test_cache_performance(self, llm_cache):
        """Benchmark cache performance."""
        # Await the fixture to get the actual cache object
        cache = await llm_cache
        
        # Generate test data
        num_items = 1000
        test_data = [{"item": f"test_{i}"} for i in range(num_items)]
        
        # Benchmark set operations
        start_time = time.time()
        for i, data in enumerate(test_data):
            await cache.set(f"perf_key_{i}", data)
        set_time = time.time() - start_time
        
        # Benchmark get operations (first half should hit L1 cache)
        start_time = time.time()
        for i in range(num_items // 2):
            await cache.get(f"perf_key_{i}")
        l1_get_time = time.time() - start_time
        
        # Benchmark get operations for keys at the end (should hit L2 cache)
        start_time = time.time()
        for i in range(num_items // 2, num_items):
            await cache.get(f"perf_key_{i}")
        l2_get_time = time.time() - start_time
        
        # Log performance results
        print(f"\nCache performance results:")
        print(f"Set {num_items} items: {set_time:.4f}s ({num_items/set_time:.2f} ops/s)")
        print(f"Get from L1 cache: {l1_get_time:.4f}s ({(num_items//2)/l1_get_time:.2f} ops/s)")
        print(f"Get from L2 cache: {l2_get_time:.4f}s ({(num_items//2)/l2_get_time:.2f} ops/s)")
        
        # Basic assertion to ensure the test ran
        assert set_time > 0
        
        # Get cache stats
        cache_stats = await cache.get_stats()
        print(f"Cache stats: {cache_stats}")
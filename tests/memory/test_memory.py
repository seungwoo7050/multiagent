"""
Unit tests for the memory package components.

These tests verify the functionality of the memory storage and vector store systems.
Run with pytest: 
    python -m pytest tests/memory/test_memory.py -v
"""
import os
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.settings import get_settings
from src.memory.base import BaseMemory, BaseVectorStore
from src.memory.manager import MemoryManager
from src.memory.redis_memory import RedisMemory
from src.memory.utils import generate_memory_key, serialize_data, deserialize_data, compute_fingerprint, ExpirationPolicy
from src.memory.vector_store import VectorStore

settings = get_settings()

# Mock the metrics manager
@pytest.fixture
def mock_metrics():
    """Create a mock metrics manager"""
    metrics_mock = MagicMock()
    metrics_mock.track_memory = MagicMock()
    metrics_mock.track_cache = MagicMock()
    metrics_mock.timed_metric = MagicMock()
    metrics_mock.MEMORY_METRICS = {
        'operations': MagicMock(),
        'duration': MagicMock(),
        'size': MagicMock()
    }
    metrics_mock.timed_metric.return_value = lambda f: f  # Return the original function
    
    with patch('src.memory.utils.metrics', metrics_mock), \
         patch('src.memory.redis_memory.metrics', metrics_mock), \
         patch('src.memory.manager.metrics', metrics_mock), \
         patch('src.memory.vector_store.metrics', metrics_mock), \
         patch('src.memory.backends.chroma.metrics', metrics_mock), \
         patch('src.memory.backends.qdrant.metrics', metrics_mock), \
         patch('src.memory.backends.faiss.metrics', metrics_mock):
        yield metrics_mock

# Helper fixture for mocking Redis
@pytest.fixture
def mock_redis():
    """Create a mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=1)
    redis_mock.scan = AsyncMock(return_value=(b'0', [b'memory:test:key1', b'memory:test:key2']))
    redis_mock.mget = AsyncMock(return_value=[None, None])
    redis_mock.pipeline = AsyncMock()
    redis_mock.pipeline.return_value.__aenter__ = AsyncMock()
    redis_mock.pipeline.return_value.__aexit__ = AsyncMock()
    redis_mock.pipeline.return_value.execute = AsyncMock(return_value=[True, True])
    redis_mock.info = AsyncMock(return_value={"used_memory": 1024})
    redis_mock.dbsize = AsyncMock(return_value=10)
    return redis_mock

# Fixture for mocking MemoryManager
@pytest.fixture
def memory_manager(mock_metrics):
    """Create a memory manager with mocked dependencies"""
    # Create mock primary memory
    primary_memory = AsyncMock(spec=BaseMemory)
    primary_memory.load_context = AsyncMock(return_value="test_value")
    primary_memory.save_context = AsyncMock(return_value=True)
    primary_memory.delete_context = AsyncMock(return_value=True)
    primary_memory.clear = AsyncMock(return_value=True)
    primary_memory.list_keys = AsyncMock(return_value=["key1", "key2"])
    primary_memory.exists = AsyncMock(return_value=True)
    primary_memory.bulk_load = AsyncMock(return_value={"key1": "value1", "key2": "value2"})
    primary_memory.bulk_save = AsyncMock(return_value=True)
    primary_memory.get_stats = AsyncMock(return_value={"implementation_type": "MockMemory", "total_keys": 10})
    
    # Create mock vector store
    vector_store = AsyncMock(spec=BaseVectorStore)
    vector_store.store_vector = AsyncMock(return_value="vector_id")
    vector_store.search_vectors = AsyncMock(return_value=[
        {"id": "vec1", "score": 0.95, "metadata": {"text": "test"}},
        {"id": "vec2", "score": 0.85, "metadata": {"text": "test2"}}
    ])
    vector_store.delete_vectors = AsyncMock(return_value=True)
    vector_store.get_stats = AsyncMock(return_value={"type": "mock", "total_vectors": 5})
    
    # Create memory manager with mocks
    manager = MemoryManager(
        primary_memory=primary_memory,
        vector_store=vector_store,
        cache_size=100,
        cache_ttl=60,
        memory_ttl=3600
    )
    
    return manager

# Test memory utils functions
class TestMemoryUtils:
    """Test memory utility functions"""
    
    def test_generate_memory_key(self):
        """Test memory key generation"""
        key = generate_memory_key("test_key", "context123")
        assert key == "memory:context123:test_key"
    
    @pytest.mark.asyncio
    async def test_serialize_deserialize_roundtrip(self, mock_metrics):
        """Test serialization and deserialization roundtrip"""
        # Test with different data types
        test_data = [
            {"name": "Test", "value": 123},
            ["a", "b", "c", 1, 2, 3],
            "Simple string",
            123,
            True
        ]
        
        for data in test_data:
            # Serialize
            serialized = await serialize_data(data)
            assert isinstance(serialized, bytes)
            
            # Deserialize
            deserialized = await deserialize_data(serialized)
            assert deserialized == data
    
    def test_compute_fingerprint(self):
        """Test fingerprint generation for different data types"""
        # Same data should produce same fingerprint
        data1 = {"a": 1, "b": 2}
        data2 = {"a": 1, "b": 2}
        
        fp1 = compute_fingerprint(data1)
        fp2 = compute_fingerprint(data2)
        assert fp1 == fp2
        
        # Different data should produce different fingerprints
        data3 = {"a": 1, "b": 3}
        fp3 = compute_fingerprint(data3)
        assert fp1 != fp3
    
    def test_expiration_policy(self):
        """Test TTL policy calculations"""
        # Test with override TTL
        assert ExpirationPolicy.get_ttl(3600, 1800) == 1800
        
        # Test with zero or negative TTL (should return None)
        assert ExpirationPolicy.get_ttl(3600, 0) is None
        assert ExpirationPolicy.get_ttl(3600, -1) is None
        
        # Test default TTL
        assert ExpirationPolicy.get_ttl(3600, None) == 3600
        
        # Test key types
        # Temporary keys (25% of default, minimum 60s)
        assert ExpirationPolicy.get_ttl(3600, None, "temporary") == 900
        assert ExpirationPolicy.get_ttl(100, None, "temporary") == 60  # Minimum 60s
        
        # Persistent keys (4x default)
        assert ExpirationPolicy.get_ttl(3600, None, "persistent") == 14400

# Test Redis memory implementation
@pytest.mark.asyncio
class TestRedisMemory:
    """Test Redis memory implementation"""
    
    async def test_load_context(self, mock_redis, mock_metrics):
        """Test loading data from Redis"""
        # Setup
        with patch('src.memory.redis_memory.conn_manager.get_redis_async_connection', return_value=mock_redis):

            # Create test instance
            redis_memory = RedisMemory(default_ttl=3600)
            
            # Case 1: Key not found
            mock_redis.get.return_value = None
            result = await redis_memory.load_context("test_key", "context1", default="default_value")
            assert result == "default_value"
            mock_redis.get.assert_called_with("memory:context1:test_key")
            
            # Case 2: Key found
            # Mock serialized data
            test_data = {"name": "Test", "value": 123}
            serialized = await serialize_data(test_data)
            mock_redis.get.return_value = serialized
            
            result = await redis_memory.load_context("test_key", "context1")
            assert result == test_data
    
    async def test_save_context(self, mock_redis, mock_metrics):
        """Test saving data to Redis"""
        # Setup
        with patch('src.memory.redis_memory.conn_manager.get_redis_async_connection', return_value=mock_redis):

            # Create test instance
            redis_memory = RedisMemory(default_ttl=3600)
            
            # Case 1: Save with TTL
            test_data = {"name": "Test", "value": 123}
            result = await redis_memory.save_context("test_key", "context1", test_data, ttl=1800)
            assert result is True
            mock_redis.setex.assert_called_once()
            
            # Reset mock
            mock_redis.setex.reset_mock()
            mock_redis.set.reset_mock()
            
            # Case 2: Save without TTL
            result = await redis_memory.save_context("test_key", "context1", test_data, ttl=0)
            assert result is True
            mock_redis.set.assert_called_once()
            mock_redis.setex.assert_not_called()
    
    async def test_delete_context(self, mock_redis, mock_metrics):
        """Test deleting data from Redis"""
        # Setup
        with patch('src.memory.redis_memory.conn_manager.get_redis_async_connection', return_value=mock_redis):

            # Create test instance
            redis_memory = RedisMemory()
            
            # Case 1: Successful deletion
            mock_redis.delete.return_value = 1
            result = await redis_memory.delete_context("test_key", "context1")
            assert result is True
            mock_redis.delete.assert_called_with("memory:context1:test_key")
            
            # Case 2: Key not found
            mock_redis.delete.return_value = 0
            result = await redis_memory.delete_context("nonexistent", "context1")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, mock_redis: AsyncMock, mock_metrics: MagicMock):
        """Test bulk operations in Redis (Revised Mocking Strategy)"""
        with patch('src.memory.redis_memory.conn_manager.get_redis_async_connection', return_value=mock_redis):

            redis_memory = RedisMemory()

            keys_to_load = ["key1", "key2"]
            expected_serialized_values = [await serialize_data("value1"), await serialize_data("value2")]
            mock_redis.mget.return_value = expected_serialized_values

            load_result = await redis_memory.bulk_load(keys_to_load, "context1")
            assert load_result == {"key1": "value1", "key2": "value2"}
            mock_redis.mget.assert_awaited_once_with('memory:context1:key1', 'memory:context1:key2')

            data_to_save = {"key1": "new_value1", "key2": "new_value2"}

            mock_pipeline_cm = AsyncMock()

            mock_pipe = AsyncMock()

            mock_pipeline_cm.__aenter__.return_value = mock_pipe
            mock_pipeline_cm.__aexit__.return_value = None

            mock_pipe.execute.return_value = [True] * len(data_to_save)

            mock_redis.pipeline = MagicMock(return_value=mock_pipeline_cm)

            save_result = await redis_memory.bulk_save(data_to_save, "context1")

            assert save_result is True

            mock_redis.pipeline.assert_called_once_with(transaction=False)
            mock_pipeline_cm.__aenter__.assert_awaited_once()
            mock_pipe.execute.assert_awaited_once()          
            mock_pipeline_cm.__aexit__.assert_awaited_once() 

            assert mock_pipe.setex.call_count == len(data_to_save) 

# Test MemoryManager
@pytest.mark.asyncio
class TestMemoryManager:
    """Test MemoryManager functionality"""
    
    async def test_load(self, memory_manager):
        """Test loading data with caching"""
        # Case 1: Cache miss, load from primary
        result = await memory_manager.load("test_key", "context1")
        assert result == "test_value"
        memory_manager.primary.load_context.assert_called_once()
        
        # Case 2: Cache hit, no call to primary
        memory_manager.primary.load_context.reset_mock()
        # Manually add to cache to simulate previous load
        memory_manager.cache[memory_manager._get_cache_key("test_key", "context1")] = "cached_value"
        
        result = await memory_manager.load("test_key", "context1")
        assert result == "cached_value"
        memory_manager.primary.load_context.assert_not_called()
    
    async def test_save(self, memory_manager):
        """Test saving data with cache update"""
        # Save data
        result = await memory_manager.save("test_key", "context1", "test_value")
        assert result is True
        memory_manager.primary.save_context.assert_called_once()
        
        # Check that cache was updated
        cache_key = memory_manager._get_cache_key("test_key", "context1")
        assert memory_manager.cache[cache_key] == "test_value"
    
    async def test_delete(self, memory_manager):
        """Test deleting data with cache invalidation"""
        # Setup: Add to cache
        cache_key = memory_manager._get_cache_key("test_key", "context1")
        memory_manager.cache[cache_key] = "cached_value"
        
        # Delete
        result = await memory_manager.delete("test_key", "context1")
        assert result is True
        memory_manager.primary.delete_context.assert_called_once()
        
        # Verify cache was invalidated
        assert cache_key not in memory_manager.cache
    
    async def test_clear(self, memory_manager):
        """Test clearing data"""
        # Setup: Add to cache
        memory_manager.cache["memory:context1:key1"] = "value1"
        memory_manager.cache["memory:context1:key2"] = "value2"
        memory_manager.cache["memory:context2:key3"] = "value3"
        
        # Clear specific context
        result = await memory_manager.clear("context1")
        assert result is True
        memory_manager.primary.clear.assert_called_once()
        
        # Verify cache for context1 was cleared
        assert "memory:context1:key1" not in memory_manager.cache
        assert "memory:context1:key2" not in memory_manager.cache
        assert "memory:context2:key3" in memory_manager.cache
    
    async def test_vector_operations(self, memory_manager):
        """Test vector operations"""
        # Test store vector
        vector_id = await memory_manager.store_vector("test text", {"tag": "test"}, context_id="context1")
        assert vector_id == "vector_id"
        memory_manager.vector_store.store_vector.assert_called_once()
        
        # Test search vectors
        results = await memory_manager.search_vectors("query", k=2, context_id="context1")
        assert len(results) == 2
        assert results[0]["id"] == "vec1"
        memory_manager.vector_store.search_vectors.assert_called_once()

# Test VectorStore
@pytest.mark.asyncio
class TestVectorStore:
    """Test VectorStore functionality"""
    
    @patch('src.memory.vector_store.metrics')
    async def test_in_memory_vector_store(self, mock_metrics):
        """Test in-memory vector store implementation"""
        # Create in-memory vector store
        vector_store = VectorStore()
        assert vector_store.vector_db_type == "none"
        
        # Store vector
        text = "This is a test document"
        metadata = {"category": "test"}
        with patch.object(vector_store, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            vector_id = await vector_store.store_vector(text, metadata, context_id="test_context")
        
        assert isinstance(vector_id, str)
        
        # Search vectors
        with patch.object(vector_store, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            results = await vector_store.search_vectors("test", k=1, context_id="test_context")
        
        assert len(results) == 1
        assert "id" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]
        
        # Delete vector
        success = await vector_store.delete_vectors([results[0]["id"]], context_id="test_context")
        assert success is True
        
        # Verify deletion
        with patch.object(vector_store, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            results = await vector_store.search_vectors("test", k=1, context_id="test_context")
        
        assert len(results) == 0

# Test Backend Registration
class TestVectorBackends:
    """Test vector store backend registration"""
    
    def test_backend_registration(self):
        """Test backend registration process"""
        # Create mock vector store class
        class MockVectorStore:
            pass
        
        # Import registration function
        from src.memory.backends import register_backends
        
        # Register backends
        register_backends(MockVectorStore)
        
        # Check if backend methods were attached
        assert hasattr(MockVectorStore, '_store_vector_chroma') or \
               hasattr(MockVectorStore, '_store_vector_qdrant') or \
               hasattr(MockVectorStore, '_store_vector_faiss')

# Integration test for the full memory stack
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("ENABLE_INTEGRATION_TESTS"), 
                    reason="Integration tests disabled")
class TestMemoryIntegration:
    """Integration tests for memory components"""
    
    @pytest.mark.asyncio
    async def test_memory_manager_with_redis(self):
        """Test MemoryManager with actual Redis connection"""
        # This requires a running Redis instance
        redis_memory = RedisMemory()
        
        # Create memory manager with Redis
        manager = MemoryManager(
            primary_memory=redis_memory,
            vector_store=None,
            cache_size=100,
            cache_ttl=60,
            memory_ttl=3600
        )
        
        # Test key with random suffix to avoid conflicts
        test_key = f"test_key_{int(time.time())}"
        test_value = {"name": "Integration Test", "value": 42}
        
        try:
            # Save data
            await manager.save(test_key, "integration_test", test_value)
            
            # Load data
            loaded = await manager.load(test_key, "integration_test")
            assert loaded == test_value
            
            # List keys
            keys = await manager.list_keys("integration_test")
            assert test_key in keys
            
            # Delete data
            await manager.delete(test_key, "integration_test")
            
            # Verify deletion
            loaded_after_delete = await manager.load(test_key, "integration_test")
            assert loaded_after_delete is None
            
        finally:
            # Clean up
            await manager.clear("integration_test")
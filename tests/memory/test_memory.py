"""Tests for the Memory module."""

import asyncio
import os
import pytest
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.memory.base import BaseMemory, BaseVectorStore
from src.memory.redis_memory import RedisMemory
from src.memory.manager import MemoryManager
from src.memory.vector_store import VectorStore
from src.memory.utils import (
    serialize_data,
    deserialize_data,
    generate_memory_key,
    compute_fingerprint,
    AsyncLock,
    ExpirationPolicy,
)

# Fixture for mock Redis service
# In real tests, use fakeredis or a Redis test container
class MockRedisMemory(BaseMemory):
    """Mock Redis memory for testing."""
    
    def __init__(self):
        self.data = {}
        self.ttls = {}
        
    async def load_context(self, key: str, context_id: str, default: Any = None) -> Any:
        full_key = generate_memory_key(key, context_id)
        if full_key in self.data:
            return self.data[full_key]
        return default
    
    async def save_context(self, key: str, context_id: str, data: Any, ttl: Optional[int] = None) -> bool:
        full_key = generate_memory_key(key, context_id)
        self.data[full_key] = data
        if ttl:
            self.ttls[full_key] = time.time() + ttl
        return True
    
    async def delete_context(self, key: str, context_id: str) -> bool:
        full_key = generate_memory_key(key, context_id)
        if full_key in self.data:
            del self.data[full_key]
            if full_key in self.ttls:
                del self.ttls[full_key]
            return True
        return False
    
    async def clear(self, context_id: Optional[str] = None) -> bool:
        if context_id:
            prefix = f"memory:{context_id}:"
            keys_to_delete = [k for k in self.data if k.startswith(prefix)]
            for k in keys_to_delete:
                del self.data[k]
                if k in self.ttls:
                    del self.ttls[k]
        else:
            self.data.clear()
            self.ttls.clear()
        return True
    
    async def list_keys(self, context_id: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        if context_id:
            prefix = f"memory:{context_id}:"
            keys = [k.split(":", 2)[2] for k in self.data.keys() if k.startswith(prefix)]
        else:
            keys = [k.split(":", 2)[2] if ":" in k else k for k in self.data.keys()]
        
        if pattern:
            # Simple pattern matching (only * wildcard supported)
            if pattern == "*":
                return keys
            elif "*" not in pattern:
                return [k for k in keys if k == pattern]
            elif pattern.startswith("*") and pattern.endswith("*"):
                pattern = pattern[1:-1]
                return [k for k in keys if pattern in k]
            elif pattern.startswith("*"):
                pattern = pattern[1:]
                return [k for k in keys if k.endswith(pattern)]
            elif pattern.endswith("*"):
                pattern = pattern[:-1]
                return [k for k in keys if k.startswith(pattern)]
            else:
                return []
        
        return keys
    
    async def exists(self, key: str, context_id: str) -> bool:
        full_key = generate_memory_key(key, context_id)
        return full_key in self.data
    
    async def bulk_load(self, keys: List[str], context_id: str, default: Any = None) -> Dict[str, Any]:
        result = {}
        for key in keys:
            result[key] = await self.load_context(key, context_id, default)
        return result
    
    async def bulk_save(self, data: Dict[str, Any], context_id: str, ttl: Optional[int] = None) -> bool:
        for key, value in data.items():
            await self.save_context(key, context_id, value, ttl)
        return True


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing."""
    
    def __init__(self):
        self.vectors = {}
    
    async def store_vector(self, text: str, metadata: Dict[str, Any], 
                          vector: Optional[List[float]] = None, 
                          context_id: Optional[str] = None) -> str:
        import hashlib
        # Generate a stable ID for the vector
        vector_id = hashlib.md5(f"{text}:{context_id or 'global'}".encode()).hexdigest()
        
        # Create a simple vector from the text for testing
        if vector is None:
            vector = [hash(c) % 100 / 100 for c in text[:10]]
            vector = vector + [0] * (10 - len(vector))  # Pad to length 10
        
        collection = context_id or "global"
        if collection not in self.vectors:
            self.vectors[collection] = {}
        
        self.vectors[collection][vector_id] = {
            "id": vector_id,
            "text": text,
            "vector": vector,
            "metadata": metadata
        }
        
        return vector_id
    
    async def search_vectors(self, query: str, k: int = 5, 
                            context_id: Optional[str] = None, 
                            filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Mock search just returns vectors with metadata containing any words from query
        collection = context_id or "global"
        if collection not in self.vectors:
            return []
        
        results = []
        query_words = set(query.lower().split())
        
        for vector_id, item in self.vectors[collection].items():
            # Apply metadata filter if specified
            if filter_metadata and not all(item["metadata"].get(key) == value 
                                          for key, value in filter_metadata.items()):
                continue
            
            # Check for word matches
            text_words = set(item["text"].lower().split())
            match_count = len(query_words.intersection(text_words))
            
            if match_count > 0:
                # Calculate a mock similarity score
                score = match_count / len(query_words) if query_words else 0
                
                results.append({
                    "id": vector_id,
                    "score": score,
                    "text": item["text"],
                    "metadata": item["metadata"]
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    async def delete_vectors(self, ids: Optional[List[str]] = None, 
                            context_id: Optional[str] = None) -> bool:
        if context_id and context_id in self.vectors:
            if ids:
                # Delete specific vectors in the context
                for vector_id in ids:
                    if vector_id in self.vectors[context_id]:
                        del self.vectors[context_id][vector_id]
            else:
                # Delete the entire context
                del self.vectors[context_id]
        elif ids:
            # Delete specific vectors across all contexts
            for collection in self.vectors:
                for vector_id in ids:
                    if vector_id in self.vectors[collection]:
                        del self.vectors[collection][vector_id]
        
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        stats = {
            "collections": len(self.vectors),
            "total_vectors": sum(len(vectors) for vectors in self.vectors.values())
        }
        return stats


@pytest.fixture
def mock_redis_memory():
    return MockRedisMemory()


@pytest.fixture
def mock_vector_store():
    return MockVectorStore()


@pytest.fixture
def memory_manager(mock_redis_memory, mock_vector_store):
    return MemoryManager(
        primary_memory=mock_redis_memory,
        vector_store=mock_vector_store,
        cache_size=100,
        cache_ttl=60,
        memory_ttl=3600,
    )


# Test utility functions
@pytest.mark.asyncio
async def test_memory_key_generation():
    """Test memory key generation."""
    key = generate_memory_key("test_key", "context123")
    assert key == "memory:context123:test_key"


@pytest.mark.asyncio
async def test_serialization_deserialization():
    """Test data serialization and deserialization."""
    data = {
        "string": "test",
        "int": 123,
        "float": 3.14,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "bool": True,
        "none": None,
    }
    
    # Serialize and deserialize
    serialized = await serialize_data(data)
    deserialized = await deserialize_data(serialized)
    
    assert deserialized == data, "Data should be identical after serialization and deserialization"


@pytest.mark.asyncio
async def test_compute_fingerprint():
    """Test fingerprint computation."""
    data1 = {"a": 1, "b": 2}
    data2 = {"b": 2, "a": 1}  # Same content, different order
    data3 = {"a": 1, "b": 3}  # Different content
    
    fp1 = compute_fingerprint(data1)
    fp2 = compute_fingerprint(data2)
    fp3 = compute_fingerprint(data3)
    
    assert fp1 == fp2, "Fingerprints should be identical for same content"
    assert fp1 != fp3, "Fingerprints should differ for different content"


# Test RedisMemory via mock
@pytest.mark.asyncio
async def test_redis_memory_basic_operations(mock_redis_memory):
    """Test basic RedisMemory operations."""
    # Save data
    result = await mock_redis_memory.save_context("test_key", "context1", "test_value")
    assert result == True
    
    # Load data
    value = await mock_redis_memory.load_context("test_key", "context1")
    assert value == "test_value"
    
    # Check existence
    exists = await mock_redis_memory.exists("test_key", "context1")
    assert exists == True
    
    # List keys
    keys = await mock_redis_memory.list_keys("context1")
    assert "test_key" in keys
    
    # Delete data
    result = await mock_redis_memory.delete_context("test_key", "context1")
    assert result == True
    
    # Verify deleted
    value = await mock_redis_memory.load_context("test_key", "context1", "default")
    assert value == "default"


@pytest.mark.asyncio
async def test_redis_memory_bulk_operations(mock_redis_memory):
    """Test bulk operations in RedisMemory."""
    # Bulk save
    data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
    }
    result = await mock_redis_memory.bulk_save(data, "context1")
    assert result == True
    
    # Bulk load
    values = await mock_redis_memory.bulk_load(["key1", "key2", "key3", "key4"], "context1", "default")
    assert values["key1"] == "value1"
    assert values["key2"] == "value2"
    assert values["key3"] == "value3"
    assert values["key4"] == "default"
    
    # Clear context
    result = await mock_redis_memory.clear("context1")
    assert result == True
    
    # Verify cleared
    keys = await mock_redis_memory.list_keys("context1")
    assert len(keys) == 0


# Test MemoryManager
@pytest.mark.asyncio
async def test_memory_manager_caching(memory_manager):
    """Test MemoryManager caching."""
    # Save data
    result = await memory_manager.save("test_key", "context1", "test_value")
    assert result == True
    
    # Modify underlying storage directly to test cache
    await memory_manager.primary.save_context("test_key", "context1", "modified_value")
    
    # Load should return cached value
    value = await memory_manager.load("test_key", "context1")
    assert value == "test_value"  # Not "modified_value"
    
    # Load without cache should return modified value
    value = await memory_manager.load("test_key", "context1", use_cache=False)
    assert value == "modified_value"
    
    # Invalidate cache
    count = await memory_manager.invalidate_cache("test_key", "context1")
    assert count == 1
    
    # Load should return modified value after invalidation
    value = await memory_manager.load("test_key", "context1")
    assert value == "modified_value"


@pytest.mark.asyncio
async def test_memory_manager_bulk_operations(memory_manager):
    """Test MemoryManager bulk operations."""
    # Bulk save
    data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
    }
    result = await memory_manager.bulk_save(data, "context1")
    assert result == True
    
    # Bulk load (should use cache)
    values = await memory_manager.bulk_load(["key1", "key2", "key3", "key4"], "context1", "default")
    assert values["key1"] == "value1"
    assert values["key2"] == "value2"
    assert values["key3"] == "value3"
    assert values["key4"] == "default"
    
    # Modify underlying storage directly
    await memory_manager.primary.save_context("key1", "context1", "modified1")
    
    # Bulk load should still use cache
    values = await memory_manager.bulk_load(["key1", "key2"], "context1")
    assert values["key1"] == "value1"  # Not "modified1"
    
    # Bulk load without cache should see modifications
    values = await memory_manager.bulk_load(["key1", "key2"], "context1", use_cache=False)
    assert values["key1"] == "modified1"
    
    # Clear should invalidate cache
    result = await memory_manager.clear("context1", clear_cache=True)
    assert result == True
    
    # Verify cleared
    exists = await memory_manager.exists("key1", "context1")
    assert exists == False


@pytest.mark.asyncio
async def test_memory_manager_with_cache_decorator(memory_manager):
    """Test MemoryManager with_cache decorator function."""
    # Define function to cache
    call_count = 0
    
    async def expensive_function():
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"
    
    # First call should execute function
    result1 = await memory_manager.with_cache(
        expensive_function, "cached_func", "context1"
    )
    assert result1 == "result_1"
    assert call_count == 1
    
    # Add a small delay to ensure cache operations complete
    await asyncio.sleep(0.1)
    
    # Second call should use cache
    result2 = await memory_manager.with_cache(
        expensive_function, "cached_func", "context1"
    )
    assert result2 == "result_1"
    assert call_count == 1  # Still 1
    
    # Add a small delay to ensure cache operations complete
    await asyncio.sleep(0.1)
    
    # Force refresh should execute again
    result3 = await memory_manager.with_cache(
        expensive_function, "cached_func", "context1", force_refresh=True
    )
    assert result3 == "result_2"
    assert call_count == 2


@pytest.mark.asyncio
async def test_memory_manager_with_bulk_cache(memory_manager):
    """Test MemoryManager with_bulk_cache function."""
    # Define bulk function to cache
    call_count = 0
    
    async def expensive_bulk_function(keys):
        nonlocal call_count
        call_count += 1
        # Add a small delay to make it more realistic
        await asyncio.sleep(0.05)
        return {key: f"{key}_result_{call_count}" for key in keys}
    
    # First call should execute function
    results1 = await memory_manager.with_bulk_cache(
        expensive_bulk_function, ["key1", "key2"], "context1"
    )
    assert results1["key1"] == "key1_result_1"
    assert results1["key2"] == "key2_result_1"
    assert call_count == 1
    
    # Add a small delay to ensure cache operations complete
    await asyncio.sleep(0.1)
    
    # Second call should use cache
    results2 = await memory_manager.with_bulk_cache(
        expensive_bulk_function, ["key1", "key2", "key3"], "context1"
    )
    assert results2["key1"] == "key1_result_1"  # Cached
    assert results2["key2"] == "key2_result_1"  # Cached
    assert results2["key3"] == "key3_result_2"  # New computation
    assert call_count == 2  # One more call for key3
    
    # Add a small delay to ensure cache operations complete
    await asyncio.sleep(0.1)
    
    # Force refresh should execute again for all keys
    results3 = await memory_manager.with_bulk_cache(
        expensive_bulk_function, ["key1", "key2"], "context1", force_refresh=True
    )
    assert results3["key1"] == "key1_result_3"
    assert results3["key2"] == "key2_result_3"
    assert call_count == 3


# Test VectorStore
@pytest.mark.asyncio
async def test_vector_store_operations(mock_vector_store):
    """Test VectorStore operations."""
    # Store vectors
    vector_id1 = await mock_vector_store.store_vector(
        "The quick brown fox jumps over the lazy dog",
        {"source": "test1", "category": "animals"},
        context_id="context1"
    )
    
    vector_id2 = await mock_vector_store.store_vector(
        "The fast red fox runs past the sleeping dog",
        {"source": "test2", "category": "animals"},
        context_id="context1"
    )
    
    vector_id3 = await mock_vector_store.store_vector(
        "The sun rises in the east and sets in the west",
        {"source": "test3", "category": "nature"},
        context_id="context1"
    )
    
    # Search vectors
    results = await mock_vector_store.search_vectors(
        "fox dog",
        k=2,
        context_id="context1"
    )
    
    assert len(results) == 2
    # Both documents have fox and dog, should be found
    assert any("fox" in result["text"].lower() for result in results)
    assert any("dog" in result["text"].lower() for result in results)
    
    # Search with metadata filter
    results = await mock_vector_store.search_vectors(
        "fox",
        k=5,
        context_id="context1",
        filter_metadata={"category": "animals"}
    )
    
    assert len(results) > 0
    assert all(result["metadata"]["category"] == "animals" for result in results)
    
    # Delete vectors
    await mock_vector_store.delete_vectors(
        ids=[vector_id1],
        context_id="context1"
    )
    
    # Search again after deletion
    results = await mock_vector_store.search_vectors(
        "fox",
        k=5,
        context_id="context1"
    )
    
    # Should not find the deleted vector
    assert all(result["id"] != vector_id1 for result in results)
    
    # Delete entire context
    await mock_vector_store.delete_vectors(context_id="context1")
    
    # Search again after context deletion
    results = await mock_vector_store.search_vectors(
        "fox",
        k=5,
        context_id="context1"
    )
    
    assert len(results) == 0


@pytest.mark.asyncio
async def test_memory_manager_vector_integration(memory_manager):
    """Test MemoryManager integration with VectorStore."""
    # Store vectors
    vector_id1 = await memory_manager.store_vector(
        "The quick brown fox jumps over the lazy dog",
        {"source": "test1", "category": "animals"},
        context_id="context1"
    )
    
    vector_id2 = await memory_manager.store_vector(
        "Artificial intelligence is transforming how we live and work",
        {"source": "test2", "category": "technology"},
        context_id="context1"
    )
    
    # Search vectors
    results = await memory_manager.search_vectors(
        "intelligence AI technology",
        k=1,
        context_id="context1"
    )
    
    assert len(results) == 1
    assert "intelligence" in results[0]["text"].lower()
    
    # Clear context including vectors
    await memory_manager.clear("context1", clear_vectors=True)
    
    # Search again after clearing
    results = await memory_manager.search_vectors(
        "intelligence",
        k=5,
        context_id="context1"
    )
    
    assert len(results) == 0


# Run tests if executed directly
if __name__ == "__main__":
    asyncio.run(pytest.main(["-xvs", __file__]))
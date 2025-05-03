"""
Test for the optimized vector search implementation.
Compares performance between optimized vector search and the fallback method.
"""
import pytest
import time
import numpy as np
from unittest.mock import patch

from src.memory.vector_store import VectorStore

@pytest.mark.asyncio
class TestVectorStoreOptimization:
    """Test vector store search optimization."""
    
    async def test_optimized_vector_search(self):
        """Test that optimized vector search is faster than fallback method."""
        # Create vector store instance
        vector_store = VectorStore()
        
        # Generate some test data - 1000 random vectors
        num_vectors = 1000
        vector_dim = 384  # Typical dimension for small embedding models
        np.random.seed(42)  # For reproducible results
        
        # Create random unit vectors
        test_vectors = []
        for i in range(num_vectors):
            vec = np.random.randn(vector_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            test_vectors.append(vec.tolist())
        
        # Store vectors in memory
        for i, vector in enumerate(test_vectors):
            vector_store._in_memory_vectors.setdefault("test_collection", {})
            vector_store._in_memory_vectors["test_collection"][f"id_{i}"] = {
                'id': f"id_{i}",
                'vector': vector,
                'metadata': {
                    'text': f"Test document {i}",
                    'category': 'test' if i % 2 == 0 else 'other'
                }
            }
        
        # Create query vector
        query_vector = np.random.randn(vector_dim).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.tolist()
        
        # Test optimized search (with NumPy)
        start_time = time.time()
        optimized_results = await vector_store._search_vectors_memory(
            query_vector=query_vector,
            k=10,
            collection="test_collection",
            filter_metadata={"category": "test"}
        )
        optimized_time = time.time() - start_time
        
        # Force fallback to pure Python implementation
        with patch('numpy.array', side_effect=ImportError("Forced fallback")):
            start_time = time.time()
            fallback_results = await vector_store._search_vectors_memory(
                query_vector=query_vector,
                k=10,
                collection="test_collection",
                filter_metadata={"category": "test"}
            )
            fallback_time = time.time() - start_time
        
        # Verify results are consistent
        assert len(optimized_results) == len(fallback_results)
        
        # Check that the same IDs are returned (order might differ slightly due to floating point precision)
        optimized_ids = {r['id'] for r in optimized_results}
        fallback_ids = {r['id'] for r in fallback_results}
        assert optimized_ids == fallback_ids
        
        # Verify performance improvement
        # The NumPy version should be significantly faster (at least 2x)
        performance_ratio = fallback_time / optimized_time
        print("\nPerformance comparison:")
        print(f"Optimized implementation: {optimized_time:.6f} seconds")
        print(f"Fallback implementation: {fallback_time:.6f} seconds")
        print(f"Speed improvement: {performance_ratio:.2f}x faster")
        
        assert performance_ratio > 2.0, \
            f"Expected at least 2x performance improvement, but got {performance_ratio:.2f}x"
    
    async def test_filtered_search_accuracy(self):
        """Test that filtering in vector search works correctly."""
        # Create vector store instance
        vector_store = VectorStore()
        
        # Create test vectors with different categories
        test_vectors = [
            # Category A vectors
            ([0.8, 0.1, 0.0, 0.1], "A", "doc1"),
            ([0.7, 0.2, 0.0, 0.1], "A", "doc2"),
            ([0.6, 0.3, 0.0, 0.1], "A", "doc3"),
            
            # Category B vectors
            ([0.1, 0.8, 0.0, 0.1], "B", "doc4"),
            ([0.2, 0.7, 0.0, 0.1], "B", "doc5"),
            ([0.3, 0.6, 0.0, 0.1], "B", "doc6"),
            
            # Category C vectors
            ([0.1, 0.1, 0.8, 0.0], "C", "doc7"),
            ([0.1, 0.1, 0.7, 0.1], "C", "doc8"),
            ([0.1, 0.1, 0.6, 0.2], "C", "doc9"),
        ]
        
        # Store vectors in memory
        for i, (vector, category, doc_id) in enumerate(test_vectors):
            vector_store._in_memory_vectors.setdefault("test_collection", {})
            vector_store._in_memory_vectors["test_collection"][doc_id] = {
                'id': doc_id,
                'vector': vector,
                'metadata': {
                    'text': f"Test document {i+1}",
                    'category': category
                }
            }
        
        # Query vectors closer to each category
        query_a = [0.9, 0.1, 0.0, 0.0]  # Close to category A
        query_b = [0.1, 0.9, 0.0, 0.0]  # Close to category B
        query_c = [0.0, 0.0, 0.9, 0.1]  # Close to category C
        
        # Test unfiltered search - should return closest match regardless of category
        results_a = await vector_store._search_vectors_memory(
            query_vector=query_a,
            k=1,
            collection="test_collection"
        )
        results_b = await vector_store._search_vectors_memory(
            query_vector=query_b,
            k=1,
            collection="test_collection"
        )
        results_c = await vector_store._search_vectors_memory(
            query_vector=query_c,
            k=1,
            collection="test_collection"
        )
        
        # Verify unfiltered searches return closest match (without filtering)
        assert results_a[0]['metadata']['category'] == "A"
        assert results_b[0]['metadata']['category'] == "B"
        assert results_c[0]['metadata']['category'] == "C"
        
        # Test with filtering - should only return matches from the specified category
        filtered_results = await vector_store._search_vectors_memory(
            query_vector=query_a,  # Query close to category A
            k=3,
            collection="test_collection",
            filter_metadata={"category": "B"}  # But only return category B
        )
        
        # Verify all results match the filter
        assert len(filtered_results) == 3
        assert all(r['metadata']['category'] == "B" for r in filtered_results)
        
        # Check if results are ordered by similarity
        scores = [r['score'] for r in filtered_results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by similarity score"
    
    async def test_empty_collection_handling(self):
        """Test handling of empty collections."""
        vector_store = VectorStore()
        
        # Test with non-existent collection
        results = await vector_store._search_vectors_memory(
            query_vector=[0.5, 0.5],
            k=5,
            collection="nonexistent"
        )
        assert len(results) == 0
        
        # Test with empty collection
        vector_store._in_memory_vectors["empty_collection"] = {}
        results = await vector_store._search_vectors_memory(
            query_vector=[0.5, 0.5],
            k=5,
            collection="empty_collection"
        )
        assert len(results) == 0
        
        # Test with collection that has no matches after filtering
        vector_store._in_memory_vectors["test_collection"] = {
            "id1": {
                'id': "id1",
                'vector': [0.1, 0.9],
                'metadata': {'category': 'A'}
            }
        }
        
        results = await vector_store._search_vectors_memory(
            query_vector=[0.5, 0.5],
            k=5,
            collection="test_collection",
            filter_metadata={"category": "B"}  # No matches with this category
        )
        assert len(results) == 0
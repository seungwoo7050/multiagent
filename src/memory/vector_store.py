"""Vector store integration for similarity search and RAG functionality."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp

from src.config.connections import get_http_session
from src.config.errors import ErrorCode, MemoryError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import (
    MEMORY_OPERATION_DURATION,
    timed_metric,
    track_memory_operation,
    track_memory_operation_completed,
    track_memory_size,
)
from src.config.settings import get_settings
from src.memory.base import BaseVectorStore
from src.memory.utils import AsyncLock, generate_vector_key
from src.utils.timing import async_timed

logger = get_logger(__name__)
settings = get_settings()


class VectorStore(BaseVectorStore):
    """Vector store implementation for similarity search.
    
    Features:
    - Async API calls to vector database 
    - Support for multiple vector DB providers
    - Context segmentation for isolation
    - Optimized embedding generation
    """
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize vector store.
        
        Args:
            api_url: Vector database API URL (defaults to settings)
            api_key: Vector database API key (defaults to settings)
        """
        self.vector_db_type = settings.VECTOR_DB_TYPE
        self.api_url = api_url or settings.VECTOR_DB_URL
        self.api_key = api_key
        
        # Set appropriate client methods based on DB type
        if self.vector_db_type == "chroma":
            self._store_fn = self._store_vector_chroma
            self._search_fn = self._search_vectors_chroma
            self._delete_fn = self._delete_vectors_chroma
        elif self.vector_db_type == "qdrant":
            self._store_fn = self._store_vector_qdrant
            self._search_fn = self._search_vectors_qdrant
            self._delete_fn = self._delete_vectors_qdrant
        elif self.vector_db_type == "faiss":
            self._store_fn = self._store_vector_faiss
            self._search_fn = self._search_vectors_faiss
            self._delete_fn = self._delete_vectors_faiss
        else:
            # Default to in-memory implementation for testing/development
            self._store_fn = self._store_vector_memory
            self._search_fn = self._search_vectors_memory
            self._delete_fn = self._delete_vectors_memory
            self._in_memory_vectors = {}
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "store_vector"})
    async def store_vector(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        vector: Optional[List[float]] = None, 
        context_id: Optional[str] = None
    ) -> str:
        """Store vector embedding.
        
        Args:
            text: Text content to embed
            metadata: Associated metadata
            vector: Optional pre-computed vector
            context_id: Optional context ID
            
        Returns:
            ID of stored vector
        """
        track_memory_operation("store_vector")
        
        try:
            vector_id = str(uuid.uuid4())
            collection = generate_vector_key(context_id)
            
            # If no vector provided, generate embedding
            if vector is None:
                vector = await self._generate_embedding(text)
            
            # Add required fields to metadata
            enhanced_metadata = {
                "id": vector_id,
                "text": text,
                "created_at": time.time(),
                "context_id": context_id or "global",
                **metadata
            }
            
            # Store vector using appropriate method
            start_time = time.time()
            success = await self._store_fn(vector_id, vector, enhanced_metadata, collection)
            track_memory_operation_completed("vector_store", time.time() - start_time)
            
            if not success:
                raise MemoryError(
                    code=ErrorCode.VECTOR_DB_ERROR,
                    message=f"Failed to store vector in collection {collection}"
                )
            
            # Track vector storage size
            vector_size = len(vector) * 4  # Approximate size in bytes (float32)
            metadata_size = len(json.dumps(enhanced_metadata))
            track_memory_size("vector_store", vector_size + metadata_size)
            
            return vector_id
            
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.VECTOR_DB_ERROR,
                f"Failed to store vector for context {context_id or 'global'}"
            )
            error.log_error(logger)
            raise error
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "search_vectors"})
    async def search_vectors(
        self, 
        query: str, 
        k: int = 5, 
        context_id: Optional[str] = None, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query: Query text
            k: Number of results to return
            context_id: Optional context ID to limit search
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching documents with similarity scores
        """
        track_memory_operation("search_vectors")
        
        try:
            collection = generate_vector_key(context_id)
            
            # Generate embedding for query
            query_vector = await self._generate_embedding(query)
            
            # Search using appropriate method
            start_time = time.time()
            results = await self._search_fn(query_vector, k, collection, filter_metadata)
            track_memory_operation_completed("vector_search", time.time() - start_time)
            
            return results
            
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.VECTOR_DB_ERROR,
                f"Failed to search vectors for context {context_id or 'global'}"
            )
            error.log_error(logger)
            return []
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "delete_vectors"})
    async def delete_vectors(
        self, 
        ids: Optional[List[str]] = None, 
        context_id: Optional[str] = None
    ) -> bool:
        """Delete vectors from store.
        
        Args:
            ids: Optional list of vector IDs to delete
            context_id: Optional context ID to delete all vectors
            
        Returns:
            True if successful, False otherwise
        """
        track_memory_operation("delete_vectors")
        
        if ids is None and context_id is None:
            logger.warning("Both ids and context_id are None, no vectors will be deleted")
            return False
            
        try:
            collection = generate_vector_key(context_id)
            
            # Delete using appropriate method
            start_time = time.time()
            success = await self._delete_fn(ids, collection, context_id)
            track_memory_operation_completed("vector_delete", time.time() - start_time)
            
            return success
            
        except Exception as e:
            error = convert_exception(
                e,
                ErrorCode.VECTOR_DB_ERROR,
                f"Failed to delete vectors for context {context_id or 'global'}"
            )
            error.log_error(logger)
            return False
    
    @timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "get_stats"})
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary of store stats
        """
        try:
            if self.vector_db_type == "none":
                # In-memory stats
                return {
                    "type": "in_memory",
                    "collections": len(self._in_memory_vectors),
                    "total_vectors": sum(len(vectors) for vectors in self._in_memory_vectors.values())
                }
                
            # For actual vector DBs, stats depend on implementation
            # This is a simplified implementation
            return {
                "type": self.vector_db_type,
                "url": self.api_url,
                "status": "connected"
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {str(e)}")
            return {
                "type": self.vector_db_type,
                "status": "error",
                "error": str(e)
            }
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text.
        
        Uses OpenAI-compatible embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            Float vector embedding
        """
        # For demo purposes, we'll use a simple embedding function
        # In production, this would call an embedding API like OpenAI
        
        if settings.ENVIRONMENT == "development" or self.vector_db_type == "none":
            # Simple hash-based pseudo-embedding for development
            # This is NOT for production use!
            import hashlib
            import struct
            
            # Create a deterministic but unique float vector from text
            hash_val = hashlib.md5(text.encode()).digest()
            floats = [struct.unpack('f', hash_val[i:i+4])[0] for i in range(0, len(hash_val), 4)]
            
            # Normalize to unit vector
            magnitude = sum(x*x for x in floats) ** 0.5
            if magnitude > 0:
                normalized = [x/magnitude for x in floats]
            else:
                normalized = [0.0] * len(floats)
                
            return normalized
        
        # In a real implementation, we would call an embedding API:
        try:
            async with await get_http_session() as session:
                start_time = time.time()
                
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                data = {
                    "input": text,
                    "model": "text-embedding-ada-002"  # Example model
                }
                
                async with session.post(
                    f"{self.api_url}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=10.0
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise MemoryError(
                            code=ErrorCode.VECTOR_DB_ERROR,
                            message=f"Embedding API returned {response.status}: {error_text}"
                        )
                    
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    
                    track_memory_operation_completed("generate_embedding", time.time() - start_time)
                    return embedding
                    
        except Exception as e:
            # Fall back to development mode embedding
            logger.warning(f"Error generating embedding, using fallback: {str(e)}")
            return await self._generate_embedding(text)
    
    # Implementation for in-memory vector store (development only)
    async def _store_vector_memory(
        self, 
        vector_id: str, 
        vector: List[float], 
        metadata: Dict[str, Any], 
        collection: str
    ) -> bool:
        """Store vector in memory (for development)."""
        if collection not in self._in_memory_vectors:
            self._in_memory_vectors[collection] = {}
            
        self._in_memory_vectors[collection][vector_id] = {
            "id": vector_id,
            "vector": vector,
            "metadata": metadata
        }
        
        return True
    
    async def _search_vectors_memory(
        self, 
        query_vector: List[float], 
        k: int, 
        collection: str, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search vectors in memory (for development)."""
        if collection not in self._in_memory_vectors:
            return []
            
        # Simple cosine similarity calculation
        def cosine_similarity(a, b):
            dot_product = sum(x*y for x, y in zip(a, b))
            norm_a = sum(x*x for x in a) ** 0.5
            norm_b = sum(x*x for x in b) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0
                
            return dot_product / (norm_a * norm_b)
        
        # Calculate scores
        results = []
        for vector_id, item in self._in_memory_vectors[collection].items():
            # Apply metadata filters if specified
            if filter_metadata:
                if not all(item["metadata"].get(key) == value 
                           for key, value in filter_metadata.items()):
                    continue
            
            score = cosine_similarity(query_vector, item["vector"])
            results.append({
                "id": vector_id,
                "score": score,
                "metadata": item["metadata"]
            })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    async def _delete_vectors_memory(
        self, 
        ids: Optional[List[str]], 
        collection: str, 
        context_id: Optional[str] = None
    ) -> bool:
        """Delete vectors from memory (for development)."""
        if collection not in self._in_memory_vectors:
            return False
            
        if ids:
            # Delete specific IDs
            for vector_id in ids:
                if vector_id in self._in_memory_vectors[collection]:
                    del self._in_memory_vectors[collection][vector_id]
        elif context_id:
            # Delete all vectors for context
            # This requires a linear scan in our simple implementation
            ids_to_delete = []
            for vector_id, item in self._in_memory_vectors[collection].items():
                if item["metadata"].get("context_id") == context_id:
                    ids_to_delete.append(vector_id)
                    
            for vector_id in ids_to_delete:
                del self._in_memory_vectors[collection][vector_id]
                
        return True
    
    # Chroma implementation
    async def _store_vector_chroma(
        self, 
        vector_id: str, 
        vector: List[float], 
        metadata: Dict[str, Any], 
        collection: str
    ) -> bool:
        """Store vector in ChromaDB."""
        raise NotImplementedError("ChromaDB integration to be implemented")
    
    async def _search_vectors_chroma(
        self, 
        query_vector: List[float], 
        k: int, 
        collection: str, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search vectors in ChromaDB."""
        raise NotImplementedError("ChromaDB integration to be implemented")
    
    async def _delete_vectors_chroma(
        self, 
        ids: Optional[List[str]], 
        collection: str, 
        context_id: Optional[str] = None
    ) -> bool:
        """Delete vectors from ChromaDB."""
        raise NotImplementedError("ChromaDB integration to be implemented")
    
    # Qdrant implementation
    async def _store_vector_qdrant(
        self, 
        vector_id: str, 
        vector: List[float], 
        metadata: Dict[str, Any], 
        collection: str
    ) -> bool:
        """Store vector in Qdrant."""
        raise NotImplementedError("Qdrant integration to be implemented")
    
    async def _search_vectors_qdrant(
        self, 
        query_vector: List[float], 
        k: int, 
        collection: str, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search vectors in Qdrant."""
        raise NotImplementedError("Qdrant integration to be implemented")
    
    async def _delete_vectors_qdrant(
        self, 
        ids: Optional[List[str]], 
        collection: str, 
        context_id: Optional[str] = None
    ) -> bool:
        """Delete vectors from Qdrant."""
        raise NotImplementedError("Qdrant integration to be implemented")
    
    # FAISS implementation
    async def _store_vector_faiss(
        self, 
        vector_id: str, 
        vector: List[float], 
        metadata: Dict[str, Any], 
        collection: str
    ) -> bool:
        """Store vector in FAISS."""
        raise NotImplementedError("FAISS integration to be implemented")
    
    async def _search_vectors_faiss(
        self, 
        query_vector: List[float], 
        k: int, 
        collection: str, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search vectors in FAISS."""
        raise NotImplementedError("FAISS integration to be implemented")
    
    async def _delete_vectors_faiss(
        self, 
        ids: Optional[List[str]], 
        collection: str, 
        context_id: Optional[str] = None
    ) -> bool:
        """Delete vectors from FAISS."""
        raise NotImplementedError("FAISS integration to be implemented")
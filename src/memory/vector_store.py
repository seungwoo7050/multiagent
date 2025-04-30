"""
Vector storage system for the Multi-Agent Platform.
Supports multiple vector database backends with optimized performance.
"""
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import aiohttp
from src.config.connections import get_connection_manager
from src.config.errors import ErrorCode, MemoryError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager, MEMORY_METRICS
from src.config.settings import get_settings
from src.memory.base import BaseVectorStore
from src.memory.utils import generate_vector_key
from src.utils.timing import async_timed

logger = get_logger(__name__)
metrics = get_metrics_manager()
settings = get_settings()
conn_manager = get_connection_manager()

class VectorStore(BaseVectorStore):
    """
    Vector storage system supporting multiple backends.
    
    Implements vector embedding storage and similarity search functionality.
    """

    def __init__(self, api_url: Optional[str]=None, api_key: Optional[str]=None, backend_options: Optional[Dict[str, Any]]=None):
        """
        Initialize vector store with configuration.
        
        Args:
            api_url: URL for the vector database API
            api_key: Optional API key for authentication
            backend_options: Optional additional configuration for specific backends
        """
        self.vector_db_type: str = settings.VECTOR_DB_TYPE
        self.api_url: Optional[str] = api_url or settings.VECTOR_DB_URL
        self.api_key: Optional[str] = api_key
        self.backend_options = backend_options or {}
        
        # Log initialization
        logger.info(f'Initializing VectorStore with type: {self.vector_db_type}, URL: {self.api_url or "In-memory"}')
        
        # FAISS-specific configuration for persistent storage
        if self.vector_db_type == 'faiss' and self.backend_options.get('storage_dir'):
            self.faiss_directory = self.backend_options['storage_dir']
            logger.info(f"FAISS vectors will be stored in: {self.faiss_directory}")
        
        # Set default in-memory implementation
        self._store_fn = self._store_vector_memory
        self._search_fn = self._search_vectors_memory
        self._delete_fn = self._delete_vectors_memory
        self._in_memory_vectors: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Initialize in-memory storage by default
        logger.info("In-memory vector storage initialized")
        
        # Backend-specific implementations will be attached by backends.__init__.register_backends()

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'store_vector'})
    async def store_vector(self, text: str, metadata: Dict[str, Any], vector: Optional[List[float]]=None, context_id: Optional[str]=None) -> str:
        """
        Store a vector with associated text and metadata.
        
        Args:
            text: The text associated with the vector
            metadata: Additional metadata to store
            vector: Optional pre-computed vector embedding
            context_id: Optional context identifier
            
        Returns:
            str: ID of the stored vector
            
        Raises:
            MemoryError: If vector storage fails
        """
        metrics.track_memory('operations', operation_type='store_vector')
        operation_desc = f"context '{context_id or 'global'}'"
        logger.debug(f'Storing vector for {operation_desc} with text (length: {len(text)})')
        
        try:
            # Generate unique ID for the vector
            vector_id: str = str(uuid.uuid4())
            collection: str = generate_vector_key(context_id)
            
            # Generate embedding if not provided
            if vector is None:
                logger.debug(f"Vector not provided for '{vector_id}'. Generating embedding...")
                start_embed_time = time.monotonic()
                vector = await self._generate_embedding(text)
                embed_duration = time.monotonic() - start_embed_time
                logger.debug(f"Embedding generated for '{vector_id}' in {embed_duration:.4f}s (Vector dim: {len(vector)})")
            
            # Enhance metadata with additional information
            enhanced_metadata: Dict[str, Any] = {
                'id': vector_id,
                'text': text,
                'created_at': time.time(),
                'context_id': context_id or 'global',
                **metadata
            }
            
            # Store vector using backend-specific function
            start_store_time = time.monotonic()
            success: bool = await self._store_fn(vector_id, vector, enhanced_metadata, collection)
            store_duration = time.monotonic() - start_store_time
            metrics.track_memory('duration', operation_type=f'vector_store_{self.vector_db_type}', value=store_duration)
            
            if not success:
                raise MemoryError(
                    code=ErrorCode.VECTOR_DB_ERROR, 
                    message=f"Backend function failed to store vector '{vector_id}' in collection '{collection}'"
                )
            
            # Track size metrics
            try:
                vector_size_bytes = len(vector) * 4  # Approximate size of float32 vector
                metadata_size_bytes = len(json.dumps(enhanced_metadata))
                total_size_bytes = vector_size_bytes + metadata_size_bytes
                metrics.track_memory('size', memory_type=f'vector_store_{self.vector_db_type}', value=total_size_bytes)
                logger.debug(f"Stored vector '{vector_id}'. Approx size: {total_size_bytes} bytes.")
            except Exception as size_err:
                logger.warning(f"Could not estimate size for stored vector '{vector_id}': {size_err}")
            
            return vector_id
            
        except Exception as e:
            error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, f'Failed to store vector for {operation_desc}')
            error.log_error(logger)
            raise error

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'search_vectors'})
    async def search_vectors(self, query: str, k: int=5, context_id: Optional[str]=None, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        """
        Search for vectors similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            context_id: Optional context identifier
            filter_metadata: Optional filter to apply to metadata
            
        Returns:
            List[Dict[str, Any]]: Search results with similarity scores
        """
        metrics.track_memory('operations', operation_type='search_vectors')
        operation_desc = f"context '{context_id or 'global'}'"
        logger.debug(f'Searching vectors for query (length: {len(query)}) in {operation_desc} (k={k})')
        
        try:
            collection: str = generate_vector_key(context_id)
            
            # Generate embedding for query
            start_embed_time = time.monotonic()
            query_vector: List[float] = await self._generate_embedding(query)
            embed_duration = time.monotonic() - start_embed_time
            logger.debug(f'Query embedding generated in {embed_duration:.4f}s (Vector dim: {len(query_vector)})')
            
            # Search vectors using backend-specific function
            start_search_time = time.monotonic()
            results: List[Dict[str, Any]] = await self._search_fn(query_vector, k, collection, filter_metadata)
            search_duration = time.monotonic() - start_search_time
            metrics.track_memory('duration', operation_type=f'vector_search_{self.vector_db_type}', value=search_duration)
            
            logger.info(f'Vector search completed. Found {len(results)} results for query in {operation_desc}.')
            return results
            
        except Exception as e:
            error = convert_exception(
                e, 
                ErrorCode.VECTOR_DB_ERROR, 
                f"Failed to search vectors for query '{query[:50]}...' in {operation_desc}"
            )
            error.log_error(logger)
            return []

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'delete_vectors'})
    async def delete_vectors(self, ids: Optional[List[str]]=None, context_id: Optional[str]=None) -> bool:
        """
        Delete vectors by ID or context.
        
        Args:
            ids: Optional list of vector IDs to delete
            context_id: Optional context identifier
            
        Returns:
            bool: True if deletion was successful
        """
        metrics.track_memory('operations', operation_type='delete_vectors')
        
        if ids is None and context_id is None:
            logger.warning('Attempted to delete vectors without specifying IDs or context_id.')
            return False
            
        operation_desc = f'IDs {ids}' if ids else f"context '{context_id}'"
        logger.info(f'Deleting vectors for {operation_desc}')
        
        try:
            collection: str = generate_vector_key(context_id)
            
            # Delete vectors using backend-specific function
            start_time = time.monotonic()
            success: bool = await self._delete_fn(ids, collection, context_id)
            duration = time.monotonic() - start_time
            metrics.track_memory('duration', operation_type=f'vector_delete_{self.vector_db_type}', value=duration)
            
            if success:
                logger.info(f'Successfully deleted vectors for {operation_desc}.')
            else:
                logger.warning(f'Vector deletion reported failure for {operation_desc}.')
                
            return success
            
        except Exception as e:
            error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, f'Failed to delete vectors for {operation_desc}')
            error.log_error(logger)
            return False

    @metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'get_vector_stats'})
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        logger.debug(f'Getting stats for vector store (type: {self.vector_db_type})')
        
        try:
            # In-memory stats
            if self.vector_db_type == 'none':
                total_vectors = 0
                collections = {}
                
                if hasattr(self, '_in_memory_vectors'):
                    for collection_name, vectors in self._in_memory_vectors.items():
                        count = len(vectors)
                        total_vectors += count
                        collections[collection_name] = count
                
                return {
                    'type': 'in_memory',
                    'collections': len(self._in_memory_vectors) if hasattr(self, '_in_memory_vectors') else 0,
                    'collection_counts': collections,
                    'total_vectors': total_vectors
                }
            
            # Backend-specific stats (limited in this implementation)
            logger.warning(f'Advanced stats retrieval not fully implemented for vector DB type: {self.vector_db_type}.')
            return {
                'type': self.vector_db_type,
                'url': self.api_url,
                'status': 'connected (assumed)'
            }
            
        except Exception as e:
            logger.error(f'Failed to get vector store stats (type: {self.vector_db_type}): {str(e)}', exc_info=True)
            return {
                'type': self.vector_db_type,
                'status': 'error',
                'error_message': str(e)
            }

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            MemoryError: If embedding generation fails
        """
        # Use mock embedding for development or in-memory backend
        if settings.ENVIRONMENT == 'development' or self.vector_db_type == 'none':
            logger.debug('Generating MOCK embedding using hash function (for dev/testing).')
            import hashlib
            import struct
            
            # Use MD5 hash of text to generate reproducible mock embedding
            hash_bytes = hashlib.md5(text.encode('utf-8')).digest()
            num_floats = len(hash_bytes) // 4
            embedding_dim = 1536  # Match OpenAI embedding dimension
            mock_vector = [0.0] * embedding_dim
            
            # Convert hash bytes to floats
            float_index = 0
            for i in range(0, num_floats * 4, 4):
                if float_index < embedding_dim:
                    mock_vector[float_index] = struct.unpack('<f', hash_bytes[i:i + 4])[0]
                    float_index += 1
            
            # Normalize vector
            magnitude = sum((x * x for x in mock_vector)) ** 0.5
            if magnitude > 1e-09:
                normalized_vector = [x / magnitude for x in mock_vector]
            else:
                normalized_vector = mock_vector
                
            return normalized_vector
        
        # Generate real embedding using API
        logger.debug(f'Generating REAL embedding for text (length: {len(text)}) using external API.')
        
        try:
            async with conn_manager.http_session() as session:
                embedding_start_time = time.monotonic()
                
                # OpenAI embedding endpoint
                embedding_api_url = 'https://api.openai.com/v1/embeddings'
                embedding_model = 'text-embedding-ada-002'
                
                # Get API key from settings
                openai_api_key = settings.LLM_PROVIDERS_CONFIG.get('openai', {}).get('api_key')
                if not openai_api_key:
                    raise MemoryError(
                        code=ErrorCode.CONFIG_ERROR, 
                        message='OpenAI API key not configured for embeddings.'
                    )
                
                # Request headers and data
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {openai_api_key}'
                }
                data = {
                    'input': text,
                    'model': embedding_model
                }
                
                # Make API request
                async with session.post(embedding_api_url, headers=headers, json=data, timeout=15.0) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise MemoryError(
                            code=ErrorCode.LLM_API_ERROR,
                            message=f'Embedding API request failed with status {response.status}: {error_text[:100]}',
                            details={'status': response.status, 'model': embedding_model}
                        )
                    
                    result = await response.json()
                    embedding_vector = result['data'][0]['embedding']
                
                # Track metrics
                duration = time.monotonic() - embedding_start_time
                metrics.track_memory('duration', operation_type='generate_embedding_api', value=duration)
                logger.debug(f'Successfully generated embedding via API in {duration:.4f}s')
                
                return embedding_vector
                
        except Exception as e:
            logger.error(f'Failed to generate embedding using API: {e}', exc_info=True)
            error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, 'Embedding generation failed')
            raise error

    async def _store_vector_memory(self, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
        """
        Store a vector in the in-memory store.
        
        Args:
            vector_id: ID for the vector
            vector: Vector embedding
            metadata: Associated metadata
            collection: Collection name
            
        Returns:
            bool: True if successful
        """
        logger.debug(f"Storing vector '{vector_id}' in memory collection '{collection}'")
        
        if collection not in self._in_memory_vectors:
            self._in_memory_vectors[collection] = {}
            
        self._in_memory_vectors[collection][vector_id] = {
            'id': vector_id,
            'vector': vector,
            'metadata': metadata
        }
        
        return True

    async def _search_vectors_memory(self, query_vector: List[float], k: int, collection: str, 
                                    filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        """
        Search vectors in the in-memory store using efficient numpy operations.
        
        Args:
            query_vector: Query vector embedding
            k: Number of results to return
            collection: Collection name
            filter_metadata: Optional filter criteria
            
        Returns:
            List[Dict[str, Any]]: Search results
        """
        logger.debug(f"Searching in-memory collection '{collection}' for {k} nearest neighbors.")
        
        if collection not in self._in_memory_vectors:
            return []
        
        try:
            # Use NumPy for efficient vector operations
            import numpy as np
            
            # Convert query to numpy array
            query_np = np.array(query_vector, dtype=np.float32)
            
            target_collection = self._in_memory_vectors.get(collection, {})
            
            # Filter vectors by metadata if needed
            filtered_items = []
            for vector_id, item in target_collection.items():
                if filter_metadata:
                    match = True
                    for filter_key, filter_value in filter_metadata.items():
                        if item['metadata'].get(filter_key) != filter_value:
                            match = False
                            break
                    if not match:
                        continue
                filtered_items.append((vector_id, item))
            
            # If no vectors match the filter, return empty results
            if not filtered_items:
                return []
            
            # Batch process vectors for efficiency
            vector_ids = [vid for vid, _ in filtered_items]
            vectors = np.array([item['vector'] for _, item in filtered_items], dtype=np.float32)
            
            # Normalize vectors and query for cosine similarity
            vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = np.divide(vectors, vector_norms, where=vector_norms!=0)
            
            query_norm = np.linalg.norm(query_np)
            if query_norm > 0:
                query_np = query_np / query_norm
            
            # Calculate similarities in one batch operation (dot product of normalized vectors = cosine similarity)
            similarities = np.dot(vectors, query_np)
            
            # Create results with indices and scores
            results_with_scores = [(float(similarities[idx]), vector_ids[idx], filtered_items[idx][1]['metadata']) 
                                for idx in range(len(vector_ids))]
            
            # Sort by score in descending order and take top k
            results_with_scores.sort(reverse=True)
            top_k_results = results_with_scores[:k]
            
            # Format results
            formatted_results = [{
                'id': vector_id,
                'score': score,
                'metadata': metadata
            } for score, vector_id, metadata in top_k_results]
            
            logger.debug(f'In-memory search found {len(formatted_results)} results.')
            return formatted_results
            
        except ImportError:
            # Fallback to pure Python implementation if NumPy is not available
            logger.warning("NumPy not available. Using slower pure Python vector search.")
            
            def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
                """Calculate cosine similarity between two vectors."""
                if len(vec_a) != len(vec_b) or len(vec_a) == 0:
                    return 0.0
                    
                dot_product = sum((a * b for a, b in zip(vec_a, vec_b)))
                norm_a = sum((a * a for a in vec_a)) ** 0.5
                norm_b = sum((b * b for b in vec_b)) ** 0.5
                
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                    
                return dot_product / (norm_a * norm_b)
            
            # Calculate similarity for each vector
            results: List[Tuple[float, Dict[str, Any]]] = []
            target_collection = self._in_memory_vectors.get(collection, {})
            
            for vector_id, item in target_collection.items():
                # Apply metadata filter if provided
                if filter_metadata:
                    match = True
                    for filter_key, filter_value in filter_metadata.items():
                        if item['metadata'].get(filter_key) != filter_value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Calculate similarity
                try:
                    score = cosine_similarity(query_vector, item['vector'])
                except Exception as sim_err:
                    logger.warning(f"Error calculating similarity for vector '{vector_id}': {sim_err}")
                    score = 0.0
                
                # Add to results
                results.append((score, {
                    'id': vector_id,
                    'score': score,
                    'metadata': item['metadata']
                }))
            
            # Sort and return top k
            results.sort(key=lambda x: x[0], reverse=True)
            top_k_results = [res_dict for score, res_dict in results[:k]]
            
            logger.debug(f'In-memory search found {len(top_k_results)} results.')
            return top_k_results

    async def _delete_vectors_memory(self, ids: Optional[List[str]], collection: str, 
                                    context_id: Optional[str]=None) -> bool:
        """
        Delete vectors from the in-memory store.
        
        Args:
            ids: Optional list of vector IDs to delete
            collection: Collection name
            context_id: Optional context identifier
            
        Returns:
            bool: True if successful
        """
        logger.debug(f"Deleting vectors from in-memory collection '{collection}'. IDs: {ids}, Context ID: {context_id}")
        
        if collection not in self._in_memory_vectors:
            logger.warning(f"Collection '{collection}' not found for deletion.")
            return False
            
        target_collection = self._in_memory_vectors[collection]
        deleted_count = 0
        
        if ids:
            # Delete specific vectors by ID
            for vector_id in ids:
                if vector_id in target_collection:
                    del target_collection[vector_id]
                    deleted_count += 1
            logger.debug(f"Deleted {deleted_count} vectors by specific IDs from '{collection}'.")
            
        elif context_id:
            # Delete vectors by context ID
            ids_to_delete = [
                vector_id for vector_id, item in target_collection.items() 
                if item.get('metadata', {}).get('context_id') == context_id
            ]
            
            if not ids_to_delete:
                logger.debug(f"No vectors found for context_id '{context_id}' in collection '{collection}' to delete.")
                return True
                
            for vector_id in ids_to_delete:
                if vector_id in target_collection:
                    del target_collection[vector_id]
                    deleted_count += 1
                    
            logger.debug(f"Deleted {deleted_count} vectors for context_id '{context_id}' from '{collection}'.")
            
        else:
            logger.warning('Deletion called without specific IDs or context_id for in-memory store.')
            return False
        
        # Remove empty collection to save memory
        if not target_collection:
            del self._in_memory_vectors[collection]
            logger.debug(f"Removed empty collection '{collection}' after deletion.")
            
        return True
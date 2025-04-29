import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import aiohttp
from src.config.connections import get_http_session
from src.config.errors import ErrorCode, MemoryError, convert_exception
from src.config.logger import get_logger
from src.config.metrics import MEMORY_OPERATION_DURATION, timed_metric, track_memory_operation, track_memory_operation_completed, track_memory_size
from src.config.settings import get_settings
from src.memory.base import BaseVectorStore
from src.memory.utils import AsyncLock, generate_vector_key
from src.utils.timing import async_timed
logger = get_logger(__name__)
settings = get_settings()

class VectorStore(BaseVectorStore):

    def __init__(self, api_url: Optional[str]=None, api_key: Optional[str]=None):
        self.vector_db_type: str = settings.VECTOR_DB_TYPE
        self.api_url: Optional[str] = api_url or settings.VECTOR_DB_URL
        self.api_key: Optional[str] = api_key
        logger.info(f'Initializing VectorStore with type: {self.vector_db_type}, URL: {self.api_url or 'In-memory'}')
        if self.vector_db_type == 'chroma':
            self._store_fn = self._store_vector_chroma
            self._search_fn = self._search_vectors_chroma
            self._delete_fn = self._delete_vectors_chroma
        elif self.vector_db_type == 'qdrant':
            self._store_fn = self._store_vector_qdrant
            self._search_fn = self._search_vectors_qdrant
            self._delete_fn = self._delete_vectors_qdrant
        elif self.vector_db_type == 'faiss':
            self._store_fn = self._store_vector_faiss
            self._search_fn = self._search_vectors_faiss
            self._delete_fn = self._delete_vectors_faiss
        else:
            if self.vector_db_type != 'none':
                logger.warning(f"Unsupported VECTOR_DB_TYPE '{self.vector_db_type}'. Defaulting to in-memory implementation.")
            self.vector_db_type = 'none'
            self._store_fn = self._store_vector_memory
            self._search_fn = self._search_vectors_memory
            self._delete_fn = self._delete_vectors_memory
            self._in_memory_vectors: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'store_vector'})
    async def store_vector(self, text: str, metadata: Dict[str, Any], vector: Optional[List[float]]=None, context_id: Optional[str]=None) -> str:
        track_memory_operation('store_vector')
        operation_desc = f"context '{context_id or 'global'}'"
        logger.debug(f'Storing vector for {operation_desc} with text (length: {len(text)})')
        try:
            vector_id: str = str(uuid.uuid4())
            collection: str = generate_vector_key(context_id)
            if vector is None:
                logger.debug(f"Vector not provided for '{vector_id}'. Generating embedding...")
                start_embed_time = time.monotonic()
                vector = await self._generate_embedding(text)
                embed_duration = time.monotonic() - start_embed_time
                logger.debug(f"Embedding generated for '{vector_id}' in {embed_duration:.4f}s (Vector dim: {len(vector)})")
            enhanced_metadata: Dict[str, Any] = {'id': vector_id, 'text': text, 'created_at': time.time(), 'context_id': context_id or 'global', **metadata}
            start_store_time = time.monotonic()
            success: bool = await self._store_fn(vector_id, vector, enhanced_metadata, collection)
            store_duration = time.monotonic() - start_store_time
            track_memory_operation_completed(f'vector_store_{self.vector_db_type}', store_duration)
            if not success:
                raise MemoryError(code=ErrorCode.VECTOR_DB_ERROR, message=f"Backend function failed to store vector '{vector_id}' in collection '{collection}'")
            try:
                vector_size_bytes = len(vector) * 4
                metadata_size_bytes = len(json.dumps(enhanced_metadata))
                total_size_bytes = vector_size_bytes + metadata_size_bytes
                track_memory_size(f'vector_store_{self.vector_db_type}', total_size_bytes)
                logger.debug(f"Stored vector '{vector_id}'. Approx size: {total_size_bytes} bytes.")
            except Exception as size_err:
                logger.warning(f"Could not estimate size for stored vector '{vector_id}': {size_err}")
            return vector_id
        except Exception as e:
            error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, f'Failed to store vector for {operation_desc}')
            error.log_error(logger)
            raise error

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'search_vectors'})
    async def search_vectors(self, query: str, k: int=5, context_id: Optional[str]=None, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        track_memory_operation('search_vectors')
        operation_desc = f"context '{context_id or 'global'}'"
        logger.debug(f'Searching vectors for query (length: {len(query)}) in {operation_desc} (k={k})')
        try:
            collection: str = generate_vector_key(context_id)
            start_embed_time = time.monotonic()
            query_vector: List[float] = await self._generate_embedding(query)
            embed_duration = time.monotonic() - start_embed_time
            logger.debug(f'Query embedding generated in {embed_duration:.4f}s (Vector dim: {len(query_vector)})')
            start_search_time = time.monotonic()
            results: List[Dict[str, Any]] = await self._search_fn(query_vector, k, collection, filter_metadata)
            search_duration = time.monotonic() - start_search_time
            track_memory_operation_completed(f'vector_search_{self.vector_db_type}', search_duration)
            logger.info(f'Vector search completed. Found {len(results)} results for query in {operation_desc}.')
            return results
        except Exception as e:
            error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, f"Failed to search vectors for query '{query[:50]}...' in {operation_desc}")
            error.log_error(logger)
            return []

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'delete_vectors'})
    async def delete_vectors(self, ids: Optional[List[str]]=None, context_id: Optional[str]=None) -> bool:
        track_memory_operation('delete_vectors')
        if ids is None and context_id is None:
            logger.warning('Attempted to delete vectors without specifying IDs or context_id.')
            return False
        operation_desc = f'IDs {ids}' if ids else f"context '{context_id}'"
        logger.info(f'Deleting vectors for {operation_desc}')
        try:
            collection: str = generate_vector_key(context_id)
            start_time = time.monotonic()
            success: bool = await self._delete_fn(ids, collection, context_id)
            duration = time.monotonic() - start_time
            track_memory_operation_completed(f'vector_delete_{self.vector_db_type}', duration)
            if success:
                logger.info(f'Successfully deleted vectors for {operation_desc}.')
            else:
                logger.warning(f'Vector deletion reported failure for {operation_desc}.')
            return success
        except Exception as e:
            error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, f'Failed to delete vectors for {operation_desc}')
            error.log_error(logger)
            return False

    @timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'get_vector_stats'})
    async def get_stats(self) -> Dict[str, Any]:
        logger.debug(f'Getting stats for vector store (type: {self.vector_db_type})')
        try:
            if self.vector_db_type == 'none':
                total_vectors = 0
                if hasattr(self, '_in_memory_vectors'):
                    total_vectors = sum((len(vectors) for vectors in self._in_memory_vectors.values()))
                return {'type': 'in_memory', 'collections': len(self._in_memory_vectors) if hasattr(self, '_in_memory_vectors') else 0, 'total_vectors': total_vectors}
            logger.warning(f'Stats retrieval not fully implemented for vector DB type: {self.vector_db_type}. Returning basic info.')
            return {'type': self.vector_db_type, 'url': self.api_url, 'status': 'connected (assumed)'}
        except Exception as e:
            logger.error(f'Failed to get vector store stats (type: {self.vector_db_type}): {str(e)}', exc_info=True)
            return {'type': self.vector_db_type, 'status': 'error', 'error_message': str(e)}

    async def _generate_embedding(self, text: str) -> List[float]:
        if settings.ENVIRONMENT == 'development' or self.vector_db_type == 'none':
            logger.debug('Generating MOCK embedding using hash function (for dev/testing).')
            import hashlib
            import struct
            hash_bytes = hashlib.md5(text.encode('utf-8')).digest()
            num_floats = len(hash_bytes) // 4
            embedding_dim = 1536
            mock_vector = [0.0] * embedding_dim
            float_index = 0
            for i in range(0, num_floats * 4, 4):
                if float_index < embedding_dim:
                    mock_vector[float_index] = struct.unpack('<f', hash_bytes[i:i + 4])[0]
                    float_index += 1
            magnitude = sum((x * x for x in mock_vector)) ** 0.5
            if magnitude > 1e-09:
                normalized_vector = [x / magnitude for x in mock_vector]
            else:
                normalized_vector = mock_vector
            return normalized_vector
        logger.debug(f'Generating REAL embedding for text (length: {len(text)}) using external API.')
        try:
            session = await get_http_session()
            embedding_start_time = time.monotonic()
            embedding_api_url = 'https://api.openai.com/v1/embeddings'
            embedding_model = 'text-embedding-ada-002'
            openai_api_key = settings.LLM_PROVIDERS_CONFIG.get('openai', {}).get('api_key')
            if not openai_api_key:
                raise MemoryError(code=ErrorCode.CONFIG_ERROR, message='OpenAI API key not configured for embeddings.')
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {openai_api_key}'}
            data = {'input': text, 'model': embedding_model}
            async with session.post(embedding_api_url, headers=headers, json=data, timeout=15.0) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise MemoryError(code=ErrorCode.LLM_API_ERROR, message=f'Embedding API request failed with status {response.status}: {error_text[:100]}', details={'status': response.status, 'model': embedding_model})
                result = await response.json()
                embedding_vector = result['data'][0]['embedding']
            duration = time.monotonic() - embedding_start_time
            track_memory_operation_completed('generate_embedding_api', duration)
            logger.debug(f'Successfully generated embedding via API in {duration:.4f}s')
            return embedding_vector
        except Exception as e:
            logger.error(f'Failed to generate embedding using API: {e}', exc_info=True)
            error = convert_exception(e, ErrorCode.VECTOR_DB_ERROR, 'Embedding generation failed')
            raise error

    async def _store_vector_memory(self, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
        logger.debug(f"Storing vector '{vector_id}' in memory collection '{collection}'")
        if collection not in self._in_memory_vectors:
            self._in_memory_vectors[collection] = {}
        self._in_memory_vectors[collection][vector_id] = {'id': vector_id, 'vector': vector, 'metadata': metadata}
        return True

    async def _search_vectors_memory(self, query_vector: List[float], k: int, collection: str, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        logger.debug(f"Searching in-memory collection '{collection}' for {k} nearest neighbors.")
        if collection not in self._in_memory_vectors:
            return []

        def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
            if len(vec_a) != len(vec_b) or len(vec_a) == 0:
                return 0.0
            dot_product = sum((a * b for a, b in zip(vec_a, vec_b)))
            norm_a = sum((a * a for a in vec_a)) ** 0.5
            norm_b = sum((b * b for b in vec_b)) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
        results: List[Tuple[float, Dict[str, Any]]] = []
        target_collection = self._in_memory_vectors.get(collection, {})
        for vector_id, item in target_collection.items():
            if filter_metadata:
                match = True
                for filter_key, filter_value in filter_metadata.items():
                    if item['metadata'].get(filter_key) != filter_value:
                        match = False
                        break
                if not match:
                    continue
            try:
                score = cosine_similarity(query_vector, item['vector'])
            except Exception as sim_err:
                logger.warning(f"Error calculating similarity for vector '{vector_id}': {sim_err}")
                score = 0.0
            results.append((score, {'id': vector_id, 'score': score, 'metadata': item['metadata']}))
        results.sort(key=lambda x: x[0], reverse=True)
        top_k_results = [res_dict for score, res_dict in results[:k]]
        logger.debug(f'In-memory search found {len(top_k_results)} results.')
        return top_k_results

    async def _delete_vectors_memory(self, ids: Optional[List[str]], collection: str, context_id: Optional[str]=None) -> bool:
        logger.debug(f"Deleting vectors from in-memory collection '{collection}'. IDs: {ids}, Context ID: {context_id}")
        if collection not in self._in_memory_vectors:
            logger.warning(f"Collection '{collection}' not found for deletion.")
            return False
        target_collection = self._in_memory_vectors[collection]
        deleted_count = 0
        if ids:
            for vector_id in ids:
                if vector_id in target_collection:
                    del target_collection[vector_id]
                    deleted_count += 1
            logger.debug(f"Deleted {deleted_count} vectors by specific IDs from '{collection}'.")
        elif context_id:
            ids_to_delete = [vector_id for vector_id, item in target_collection.items() if item.get('metadata', {}).get('context_id') == context_id]
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
        if not target_collection:
            del self._in_memory_vectors[collection]
            logger.debug(f"Removed empty collection '{collection}' after deletion.")
        return True

    async def _store_vector_chroma(self, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
        logger.warning('ChromaDB integration is not implemented.')
        raise NotImplementedError('ChromaDB vector storage is not implemented yet.')

    async def _search_vectors_chroma(self, query_vector: List[float], k: int, collection: str, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        logger.warning('ChromaDB integration is not implemented.')
        raise NotImplementedError('ChromaDB vector search is not implemented yet.')

    async def _delete_vectors_chroma(self, ids: Optional[List[str]], collection: str, context_id: Optional[str]=None) -> bool:
        logger.warning('ChromaDB integration is not implemented.')
        raise NotImplementedError('ChromaDB vector deletion is not implemented yet.')

    async def _store_vector_qdrant(self, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
        logger.warning('Qdrant integration is not implemented.')
        raise NotImplementedError('Qdrant vector storage is not implemented yet.')

    async def _search_vectors_qdrant(self, query_vector: List[float], k: int, collection: str, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        logger.warning('Qdrant integration is not implemented.')
        raise NotImplementedError('Qdrant vector search is not implemented yet.')

    async def _delete_vectors_qdrant(self, ids: Optional[List[str]], collection: str, context_id: Optional[str]=None) -> bool:
        logger.warning('Qdrant integration is not implemented.')
        raise NotImplementedError('Qdrant vector deletion is not implemented yet.')

    async def _store_vector_faiss(self, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
        logger.warning('FAISS integration is not implemented.')
        raise NotImplementedError('FAISS vector storage is not implemented yet.')

    async def _search_vectors_faiss(self, query_vector: List[float], k: int, collection: str, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        logger.warning('FAISS integration is not implemented.')
        raise NotImplementedError('FAISS vector search is not implemented yet.')

    async def _delete_vectors_faiss(self, ids: Optional[List[str]], collection: str, context_id: Optional[str]=None) -> bool:
        logger.warning('FAISS integration is not implemented.')
        raise NotImplementedError('FAISS vector deletion is not implemented yet.')
"""
Qdrant implementation for vector storage.

This module provides Qdrant-specific implementations for the vector store methods.
"""
from typing import Any, Dict, List, Optional

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager

logger = get_logger(__name__)
metrics = get_metrics_manager()

async def store_vector(vector_store, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
    """
    Store a vector in Qdrant.
    
    Args:
        vector_store: The VectorStore instance
        vector_id: ID for the vector
        vector: Vector embedding
        metadata: Associated metadata
        collection: Collection name
        
    Returns:
        bool: True if successful
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        # Get or create client
        if not hasattr(vector_store, '_qdrant_client'):
            if not vector_store.api_url:
                raise ValueError("Qdrant requires an API URL")
            
            vector_store._qdrant_client = QdrantClient(url=vector_store.api_url)
            logger.info(f"Initialized Qdrant client with URL: {vector_store.api_url}")
        
        # Check if collection exists, create if not
        collections = vector_store._qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection not in collection_names:
            # Create a new collection with the appropriate dimensions
            vector_store._qdrant_client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=len(vector),
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new Qdrant collection: {collection} with dimension {len(vector)}")
        
        # Store the vector
        vector_store._qdrant_client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )
        
        logger.debug(f"Vector stored in Qdrant collection '{collection}' with ID '{vector_id}'")
        return True
        
    except ImportError:
        logger.error("Qdrant client package not installed. Install with 'pip install qdrant-client'")
        return False
    except Exception as e:
        logger.error(f"Error storing vector in Qdrant: {e}", exc_info=True)
        return False

async def search_vectors(vector_store, query_vector: List[float], k: int, collection: str, 
                       filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
    """
    Search vectors in Qdrant.
    
    Args:
        vector_store: The VectorStore instance
        query_vector: Query vector embedding
        k: Number of results to return
        collection: Collection name
        filter_metadata: Optional filter criteria
        
    Returns:
        List[Dict[str, Any]]: Search results
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        # Get client
        if not hasattr(vector_store, '_qdrant_client'):
            if not vector_store.api_url:
                raise ValueError("Qdrant requires an API URL")
                
            vector_store._qdrant_client = QdrantClient(url=vector_store.api_url)
        
        # Check if collection exists
        collections = vector_store._qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection not in collection_names:
            logger.warning(f"Qdrant collection '{collection}' not found")
            return []
        
        # Prepare filter if provided
        filter_obj = None
        if filter_metadata and len(filter_metadata) > 0:
            must_conditions = []
            for key, value in filter_metadata.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            filter_obj = models.Filter(must=must_conditions)
        
        # Query collection
        results = vector_store._qdrant_client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=k,
            query_filter=filter_obj,
            with_payload=True
        )
        
        # Format results
        formatted_results = []
        for result in results:
            try:
                # Qdrant ScoredPoint objects have id, score, and payload
                formatted_results.append({
                    'id': str(result.id),
                    'score': float(result.score),
                    'metadata': dict(result.payload) if result.payload else {}
                })
            except Exception as e:
                logger.warning(f"Error formatting Qdrant result: {e}")
        
        logger.debug(f"Qdrant search returned {len(formatted_results)} results")
        return formatted_results
        
    except ImportError:
        logger.error("Qdrant client package not installed. Install with 'pip install qdrant-client'")
        return []
    except Exception as e:
        logger.error(f"Error searching vectors in Qdrant: {e}", exc_info=True)
        return []

async def delete_vectors(vector_store, ids: Optional[List[str]], collection: str, 
                       context_id: Optional[str]=None) -> bool:
    """
    Delete vectors from Qdrant.
    
    Args:
        vector_store: The VectorStore instance
        ids: Optional list of vector IDs to delete
        collection: Collection name
        context_id: Optional context identifier for filtering
        
    Returns:
        bool: True if successful
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        # Get client
        if not hasattr(vector_store, '_qdrant_client'):
            if not vector_store.api_url:
                raise ValueError("Qdrant requires an API URL")
                
            vector_store._qdrant_client = QdrantClient(url=vector_store.api_url)
        
        # Check if collection exists
        collections = vector_store._qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection not in collection_names:
            logger.warning(f"Qdrant collection '{collection}' not found")
            return True  # No collection = nothing to delete = success
        
        # Delete by IDs
        if ids:
            vector_store._qdrant_client.delete(
                collection_name=collection,
                points_selector=ids
            )
            logger.debug(f"Deleted {len(ids)} vectors from Qdrant collection '{collection}'")
            return True
            
        # Delete by context
        elif context_id:
            filter_obj = models.Filter(
                must=[
                    models.FieldCondition(
                        key="context_id",
                        match=models.MatchValue(value=context_id)
                    )
                ]
            )
            
            # In newer versions, count operation requires separate API call
            try:
                # First, try to count the matching points
                count_result = vector_store._qdrant_client.count(
                    collection_name=collection,
                    count_filter=filter_obj
                )
                count = count_result.count
                logger.debug(f"Found {count} vectors to delete for context '{context_id}'")
            except Exception:
                count = "unknown"
            
            # Delete points matching the filter
            vector_store._qdrant_client.delete(
                collection_name=collection,
                points_selector=filter_obj
            )
            
            logger.debug(f"Deleted {count} vectors for context '{context_id}' from Qdrant collection '{collection}'")
            return True
                
        # No IDs or context specified
        else:
            logger.warning("No vector IDs or context_id provided for Qdrant deletion")
            return False
            
    except ImportError:
        logger.error("Qdrant client package not installed. Install with 'pip install qdrant-client'")
        return False
    except Exception as e:
        logger.error(f"Error deleting vectors from Qdrant: {e}", exc_info=True)
        return False
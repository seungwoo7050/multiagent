"""
ChromaDB implementation for vector storage.

This module provides ChromaDB-specific implementations for the vector store methods.
"""
from typing import Any, Dict, List, Optional

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager

logger = get_logger(__name__)
metrics = get_metrics_manager()

async def store_vector(vector_store, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
    """
    Store a vector in ChromaDB.
    
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
        import chromadb
        from chromadb.utils import embedding_functions
        
        # Get or create client
        if not hasattr(vector_store, '_chroma_client'):
            if vector_store.api_url:
                vector_store._chroma_client = chromadb.HttpClient(url=vector_store.api_url)
            else:
                # Use in-memory client if no URL provided
                vector_store._chroma_client = chromadb.Client()
            logger.info(f"Initialized ChromaDB client: {'HTTP' if vector_store.api_url else 'In-memory'}")
        
        # Get or create collection
        try:
            chroma_collection = vector_store._chroma_client.get_collection(name=collection)
            logger.debug(f"Using existing ChromaDB collection: {collection}")
        except Exception:
            # Collection doesn't exist, create it
            chroma_collection = vector_store._chroma_client.create_collection(
                name=collection,
                metadata={"description": f"Vector collection for {collection}"}
            )
            logger.info(f"Created new ChromaDB collection: {collection}")
        
        # Extract text from metadata for document content
        document_text = metadata.get('text', '')
        
        # Add document to collection
        chroma_collection.add(
            ids=[vector_id],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[document_text]
        )
        
        logger.debug(f"Vector stored in ChromaDB collection '{collection}' with ID '{vector_id}'")
        return True
        
    except ImportError:
        logger.error("ChromaDB package not installed. Install with 'pip install chromadb'")
        return False
    except Exception as e:
        logger.error(f"Error storing vector in ChromaDB: {e}", exc_info=True)
        return False

async def search_vectors(vector_store, query_vector: List[float], k: int, collection: str, 
                        filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
    """
    Search vectors in ChromaDB.
    
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
        import chromadb
        
        # Get client
        if not hasattr(vector_store, '_chroma_client'):
            if vector_store.api_url:
                vector_store._chroma_client = chromadb.HttpClient(url=vector_store.api_url)
            else:
                vector_store._chroma_client = chromadb.Client()
        
        # Get collection
        try:
            chroma_collection = vector_store._chroma_client.get_collection(name=collection)
        except Exception as e:
            logger.warning(f"ChromaDB collection '{collection}' not found: {e}")
            return []
        
        # Prepare filter if provided
        where_filter = None
        if filter_metadata:
            where_filter = {}
            for key, value in filter_metadata.items():
                where_filter[key] = value
        
        # Query collection
        result = chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        if result and result.get('ids') and result.get('distances'):
            ids = result['ids'][0]  # First query results
            distances = result['distances'][0]  # First query distances
            metadatas = result.get('metadatas', [[]])[0]  # First query metadatas
            
            for i, vector_id in enumerate(ids):
                score = 1.0 - distances[i]  # Convert distance to similarity score
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                formatted_results.append({
                    'id': vector_id,
                    'score': score,
                    'metadata': metadata
                })
        
        logger.debug(f"ChromaDB search returned {len(formatted_results)} results")
        return formatted_results
        
    except ImportError:
        logger.error("ChromaDB package not installed. Install with 'pip install chromadb'")
        return []
    except Exception as e:
        logger.error(f"Error searching vectors in ChromaDB: {e}", exc_info=True)
        return []

async def delete_vectors(vector_store, ids: Optional[List[str]], collection: str, 
                        context_id: Optional[str]=None) -> bool:
    """
    Delete vectors from ChromaDB.
    
    Args:
        vector_store: The VectorStore instance
        ids: Optional list of vector IDs to delete
        collection: Collection name
        context_id: Optional context identifier for filtering
        
    Returns:
        bool: True if successful
    """
    try:
        import chromadb
        
        # Get client
        if not hasattr(vector_store, '_chroma_client'):
            if vector_store.api_url:
                vector_store._chroma_client = chromadb.HttpClient(url=vector_store.api_url)
            else:
                vector_store._chroma_client = chromadb.Client()
        
        # Check if collection exists
        try:
            chroma_collection = vector_store._chroma_client.get_collection(name=collection)
        except Exception as e:
            logger.warning(f"ChromaDB collection '{collection}' not found: {e}")
            return True  # No collection = nothing to delete = success
        
        # Delete by IDs
        if ids:
            chroma_collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} vectors from ChromaDB collection '{collection}'")
            return True
            
        # Delete by context
        elif context_id:
            # Get IDs matching context
            where_filter = {"context_id": context_id}
            results = chroma_collection.query(
                query_embeddings=[[0.0]],  # Dummy query
                where=where_filter,
                n_results=10000,  # Large number to get all matches
                include=["metadatas", "documents"]
            )
            
            if results and results.get('ids') and results['ids'][0]:
                context_ids = results['ids'][0]
                chroma_collection.delete(ids=context_ids)
                logger.debug(f"Deleted {len(context_ids)} vectors for context '{context_id}' from ChromaDB collection '{collection}'")
                return True
            else:
                logger.debug(f"No vectors found for context '{context_id}' in ChromaDB collection '{collection}'")
                return True
                
        # No IDs or context specified
        else:
            logger.warning("No vector IDs or context_id provided for ChromaDB deletion")
            return False
            
    except ImportError:
        logger.error("ChromaDB package not installed. Install with 'pip install chromadb'")
        return False
    except Exception as e:
        logger.error(f"Error deleting vectors from ChromaDB: {e}", exc_info=True)
        return False
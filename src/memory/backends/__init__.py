"""
Vector store backend implementations.

This package contains implementations for different vector database systems.
"""
from src.config.logger import get_logger

logger = get_logger(__name__)

def register_backends(vector_store_class):
    """
    Register backend implementations with the VectorStore class.
    
    This function checks for available backends and attaches their implementations
    to the provided VectorStore class.
    
    Args:
        vector_store_class: The VectorStore class to extend
    """
    # Register ChromaDB backend if available
    try:
        from src.memory.backends.chroma import store_vector, search_vectors, delete_vectors
        vector_store_class._store_vector_chroma = store_vector
        vector_store_class._search_vectors_chroma = search_vectors
        vector_store_class._delete_vectors_chroma = delete_vectors
        logger.info("ChromaDB backend registered successfully")
    except ImportError:
        logger.debug("ChromaDB backend not available")
    
    # Register Qdrant backend if available
    try:
        from src.memory.backends.qdrant import store_vector, search_vectors, delete_vectors
        vector_store_class._store_vector_qdrant = store_vector
        vector_store_class._search_vectors_qdrant = search_vectors
        vector_store_class._delete_vectors_qdrant = delete_vectors
        logger.info("Qdrant backend registered successfully")
    except ImportError:
        logger.debug("Qdrant backend not available")
    
    # Register FAISS backend if available
    try:
        from src.memory.backends.faiss import store_vector, search_vectors, delete_vectors
        vector_store_class._store_vector_faiss = store_vector
        vector_store_class._search_vectors_faiss = search_vectors
        vector_store_class._delete_vectors_faiss = delete_vectors
        logger.info("FAISS backend registered successfully")
    except ImportError:
        logger.debug("FAISS backend not available")
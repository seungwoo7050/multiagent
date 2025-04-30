"""
FAISS implementation for vector storage.

This module provides FAISS-specific implementations for the vector store methods.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager

logger = get_logger(__name__)
metrics = get_metrics_manager()

async def store_vector(vector_store, vector_id: str, vector: List[float], metadata: Dict[str, Any], collection: str) -> bool:
    """
    Store a vector in FAISS.
    
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
        import faiss
        import numpy as np
        
        # Initialize FAISS storage if needed
        if not hasattr(vector_store, '_faiss_indexes'):
            vector_store._faiss_indexes = {}
            vector_store._faiss_metadata = {}
            vector_store._faiss_id_maps = {}
            logger.info("Initialized FAISS storage")
        
        # Convert vector to numpy array
        vector_np = np.array([vector], dtype=np.float32)
        
        # Create or get index for this collection
        if collection not in vector_store._faiss_indexes:
            # Create a new index
            dimension = len(vector)
            index = faiss.IndexFlatIP(dimension)  # Inner product similarity (cosine after normalization)
            vector_store._faiss_indexes[collection] = index
            vector_store._faiss_metadata[collection] = {}
            vector_store._faiss_id_maps[collection] = {}
            logger.info(f"Created new FAISS index for collection '{collection}' with dimension {dimension}")
        else:
            index = vector_store._faiss_indexes[collection]
        
        # Normalize vector for cosine similarity
        faiss.normalize_L2(vector_np)
        
        # Store metadata and mapping
        next_id = len(vector_store._faiss_metadata[collection])
        vector_store._faiss_metadata[collection][vector_id] = metadata
        vector_store._faiss_id_maps[collection][next_id] = vector_id
        
        # Add to index
        index.add(vector_np)
        
        # Save index if persistent storage is configured
        if hasattr(vector_store, 'faiss_directory') and vector_store.faiss_directory:
            os.makedirs(vector_store.faiss_directory, exist_ok=True)
            index_path = os.path.join(vector_store.faiss_directory, f"{collection}.index")
            metadata_path = os.path.join(vector_store.faiss_directory, f"{collection}.metadata.json")
            
            # Save index
            faiss.write_index(index, index_path)
            
            # Save metadata and ID mappings
            with open(metadata_path, 'w') as f:
                json.dump({
                    'metadata': vector_store._faiss_metadata[collection],
                    'id_map': {str(k): v for k, v in vector_store._faiss_id_maps[collection].items()},  # Convert int keys to strings for JSON
                    'last_updated': datetime.now().isoformat()
                }, f)
            
            logger.debug(f"Saved FAISS index and metadata for collection '{collection}'")
        
        logger.debug(f"Vector stored in FAISS collection '{collection}' with ID '{vector_id}'")
        return True
        
    except ImportError:
        logger.error("FAISS package not installed. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
        return False
    except Exception as e:
        logger.error(f"Error storing vector in FAISS: {e}", exc_info=True)
        return False

async def search_vectors(vector_store, query_vector: List[float], k: int, collection: str, 
                      filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
    """
    Search vectors in FAISS.
    
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
        import faiss
        import numpy as np
        
        # Check if index exists
        if not hasattr(vector_store, '_faiss_indexes') or collection not in vector_store._faiss_indexes:
            logger.warning(f"FAISS index for collection '{collection}' not found")
            return []
        
        index = vector_store._faiss_indexes[collection]
        metadata_dict = vector_store._faiss_metadata[collection]
        id_map = vector_store._faiss_id_maps[collection]
        
        # Convert and normalize query vector
        query_np = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # Search in index
        k_search = min(k * 4, index.ntotal)  # Search for more to allow for filtering
        if k_search == 0:
            return []
            
        distances, indices = index.search(query_np, k_search)
        
        # Format results
        results_with_filter: List[Tuple[float, Dict[str, Any]]] = []
        
        # Process each result
        for i, idx in enumerate(indices[0]):
            # Skip invalid indices
            if idx == -1 or idx not in id_map:
                continue
                
            # Get original ID and metadata
            original_id = id_map[idx]
            metadata = metadata_dict.get(original_id, {})
            
            # Apply filter if provided
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Add to results
            results_with_filter.append((
                float(distances[0][i]),  # Convert numpy float to Python float
                {
                    'id': original_id,
                    'score': float(distances[0][i]),
                    'metadata': metadata
                }
            ))
        
        # Sort by score and take top k
        results_with_filter.sort(key=lambda x: x[0], reverse=True)
        formatted_results = [result for _, result in results_with_filter[:k]]
        
        logger.debug(f"FAISS search returned {len(formatted_results)} results")
        return formatted_results
        
    except ImportError:
        logger.error("FAISS package not installed. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
        return []
    except Exception as e:
        logger.error(f"Error searching vectors in FAISS: {e}", exc_info=True)
        return []

async def delete_vectors(vector_store, ids: Optional[List[str]], collection: str, 
                     context_id: Optional[str]=None) -> bool:
    """
    Delete vectors from FAISS.
    
    Args:
        vector_store: The VectorStore instance
        ids: Optional list of vector IDs to delete
        collection: Collection name
        context_id: Optional context identifier for filtering
        
    Returns:
        bool: True if successful
    """
    try:
        import faiss
        import numpy as np
        
        # Check if index exists
        if not hasattr(vector_store, '_faiss_indexes') or collection not in vector_store._faiss_indexes:
            logger.warning(f"FAISS index for collection '{collection}' not found")
            return True  # Nothing to delete = success
        
        # FAISS doesn't support direct deletion, so we need to rebuild the index
        original_index = vector_store._faiss_indexes[collection]
        original_metadata = vector_store._faiss_metadata[collection]
        original_id_map = vector_store._faiss_id_maps[collection]
        
        # Determine which IDs to keep
        if ids:
            # Delete specific IDs
            ids_to_delete = set(ids)
            ids_to_keep = [id for id in original_metadata.keys() if id not in ids_to_delete]
        elif context_id:
            # Delete by context
            ids_to_keep = [
                id for id, meta in original_metadata.items() 
                if meta.get('context_id') != context_id
            ]
        else:
            logger.warning("No vector IDs or context_id provided for FAISS deletion")
            return False
        
        # If nothing to keep, clear the index
        if not ids_to_keep:
            dimension = original_index.d
            vector_store._faiss_indexes[collection] = faiss.IndexFlatIP(dimension)
            vector_store._faiss_metadata[collection] = {}
            vector_store._faiss_id_maps[collection] = {}
            logger.debug(f"Cleared all vectors from FAISS collection '{collection}'")
            
            # Update on disk if persistent storage is configured
            if hasattr(vector_store, 'faiss_directory') and vector_store.faiss_directory:
                index_path = os.path.join(vector_store.faiss_directory, f"{collection}.index")
                metadata_path = os.path.join(vector_store.faiss_directory, f"{collection}.metadata.json")
                
                os.makedirs(vector_store.faiss_directory, exist_ok=True)
                
                # Save empty index
                faiss.write_index(vector_store._faiss_indexes[collection], index_path)
                
                # Save empty metadata and ID mappings
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'metadata': {},
                        'id_map': {},
                        'last_updated': datetime.now().isoformat()
                    }, f)
            
            return True
        
        # Create new index
        dimension = original_index.d
        new_index = faiss.IndexFlatIP(dimension)
        new_metadata = {}
        new_id_map = {}
        
        # Add vectors to keep
        for new_idx, vector_id in enumerate(ids_to_keep):
            # Find the original index
            original_idx = None
            for idx, id in original_id_map.items():
                if id == vector_id:
                    original_idx = idx
                    break
            
            if original_idx is not None:
                # Get the vector
                vector = original_index.reconstruct(original_idx)
                vector_np = np.array([vector], dtype=np.float32)
                
                # Add to new index
                new_index.add(vector_np)
                
                # Update metadata and mapping
                new_metadata[vector_id] = original_metadata[vector_id]
                new_id_map[new_idx] = vector_id
        
        # Replace old index and metadata
        vector_store._faiss_indexes[collection] = new_index
        vector_store._faiss_metadata[collection] = new_metadata
        vector_store._faiss_id_maps[collection] = new_id_map
        
        # Update on disk if persistent storage is configured
        if hasattr(vector_store, 'faiss_directory') and vector_store.faiss_directory:
            index_path = os.path.join(vector_store.faiss_directory, f"{collection}.index")
            metadata_path = os.path.join(vector_store.faiss_directory, f"{collection}.metadata.json")
            
            os.makedirs(vector_store.faiss_directory, exist_ok=True)
            
            # Save index
            faiss.write_index(new_index, index_path)
            
            # Save metadata and ID mappings
            with open(metadata_path, 'w') as f:
                json.dump({
                    'metadata': new_metadata,
                    'id_map': {str(k): v for k, v in new_id_map.items()},  # Convert int keys to strings for JSON
                    'last_updated': datetime.now().isoformat()
                }, f)
        
        deleted_count = len(original_metadata) - len(new_metadata)
        logger.debug(f"Deleted {deleted_count} vectors from FAISS collection '{collection}'")
        return True
            
    except ImportError:
        logger.error("FAISS package not installed. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
        return False
    except Exception as e:
        logger.error(f"Error deleting vectors from FAISS: {e}", exc_info=True)
        return False

async def load_saved_index(vector_store, collection: str) -> bool:
    """
    Load a saved FAISS index from disk.
    
    Args:
        vector_store: The VectorStore instance
        collection: Collection name
        
    Returns:
        bool: True if successful
    """
    try:
        import faiss
        
        if not hasattr(vector_store, 'faiss_directory') or not vector_store.faiss_directory:
            logger.warning("FAISS directory not configured for persistent storage")
            return False
            
        index_path = os.path.join(vector_store.faiss_directory, f"{collection}.index")
        metadata_path = os.path.join(vector_store.faiss_directory, f"{collection}.metadata.json")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"No saved index found for collection '{collection}'")
            return False
            
        # Initialize storage if needed
        if not hasattr(vector_store, '_faiss_indexes'):
            vector_store._faiss_indexes = {}
            vector_store._faiss_metadata = {}
            vector_store._faiss_id_maps = {}
            
        # Load index
        index = faiss.read_index(index_path)
        vector_store._faiss_indexes[collection] = index
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
        vector_store._faiss_metadata[collection] = data.get('metadata', {})
        
        # Convert string keys back to integers for id_map
        id_map_str = data.get('id_map', {})
        vector_store._faiss_id_maps[collection] = {int(k): v for k, v in id_map_str.items()}
        
        logger.info(f"Loaded FAISS index for collection '{collection}' with {index.ntotal} vectors")
        return True
        
    except ImportError:
        logger.error("FAISS package not installed. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
        return False
    except Exception as e:
        logger.error(f"Error loading FAISS index for collection '{collection}': {e}", exc_info=True)
        return False
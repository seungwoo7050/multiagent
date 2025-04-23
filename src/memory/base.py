"""Base memory interface and abstractions."""

import abc
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class BaseMemory(abc.ABC):
    """Base interface for memory implementations.
    
    All memory implementations must provide these basic operations.
    """
    
    @abc.abstractmethod
    async def load_context(
        self, 
        key: str, 
        context_id: str, 
        default: Any = None
    ) -> Any:
        """Load data from memory.
        
        Args:
            key: The identifier for the data
            context_id: The conversation or session ID
            default: Value to return if key not found
            
        Returns:
            The stored data or default if not found
        """
        pass
    
    @abc.abstractmethod
    async def save_context(
        self, 
        key: str, 
        context_id: str, 
        data: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Save data to memory.
        
        Args:
            key: The identifier for the data
            context_id: The conversation or session ID
            data: The data to save
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def delete_context(
        self, 
        key: str, 
        context_id: str
    ) -> bool:
        """Delete data from memory.
        
        Args:
            key: The identifier for the data
            context_id: The conversation or session ID
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def clear(
        self, 
        context_id: Optional[str] = None
    ) -> bool:
        """Clear all data for a context or all contexts.
        
        Args:
            context_id: The conversation or session ID, or None to clear all
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def list_keys(
        self, 
        context_id: Optional[str] = None, 
        pattern: Optional[str] = None
    ) -> List[str]:
        """List all keys in memory, optionally filtered by context and pattern.
        
        Args:
            context_id: Filter by this conversation or session ID
            pattern: Filter keys by this pattern
            
        Returns:
            List of matching keys
        """
        pass
    
    @abc.abstractmethod
    async def exists(
        self, 
        key: str, 
        context_id: str
    ) -> bool:
        """Check if a key exists in memory.
        
        Args:
            key: The identifier to check
            context_id: The conversation or session ID
            
        Returns:
            True if key exists, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def bulk_load(
        self, 
        keys: List[str], 
        context_id: str, 
        default: Any = None
    ) -> Dict[str, Any]:
        """Load multiple values from memory in a single operation.
        
        Args:
            keys: List of keys to retrieve
            context_id: The conversation or session ID
            default: Default value for missing keys
            
        Returns:
            Dictionary of {key: value} for all found keys
        """
        pass
    
    @abc.abstractmethod
    async def bulk_save(
        self, 
        data: Dict[str, Any], 
        context_id: str, 
        ttl: Optional[int] = None
    ) -> bool:
        """Save multiple values to memory in a single operation.
        
        Args:
            data: Dictionary of {key: value} pairs to store
            context_id: The conversation or session ID
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        # Default implementation - override for specific stats
        return {
            "type": self.__class__.__name__,
            "keys": len(await self.list_keys()),
        }


class BaseVectorStore(abc.ABC):
    """Base interface for vector store implementations."""
    
    @abc.abstractmethod
    async def store_vector(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        vector: Optional[List[float]] = None, 
        context_id: Optional[str] = None
    ) -> str:
        """Store a vector embedding with its associated text and metadata.
        
        Args:
            text: The text content
            metadata: Associated metadata
            vector: Optional pre-computed embedding vector
            context_id: Optional context identifier for segmentation
            
        Returns:
            Identifier for the stored vector
        """
        pass
    
    @abc.abstractmethod
    async def search_vectors(
        self, 
        query: str, 
        k: int = 5, 
        context_id: Optional[str] = None, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query: The search query text
            k: Number of results to return
            context_id: Optional context to limit search to
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching documents with similarity scores
        """
        pass
    
    @abc.abstractmethod
    async def delete_vectors(
        self, 
        ids: Optional[List[str]] = None, 
        context_id: Optional[str] = None
    ) -> bool:
        """Delete vectors from the store.
        
        Args:
            ids: Optional list of specific vector IDs to delete
            context_id: Optional context ID to delete all vectors for
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary of store statistics
        """
        pass
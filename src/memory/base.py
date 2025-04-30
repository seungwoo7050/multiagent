"""
Base interfaces for memory and vector storage systems.
Defines the contract that all implementations must follow.
"""
import abc
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.config.logger import get_logger

logger = get_logger(__name__)

class BaseMemory(abc.ABC):
    """
    Abstract base class for memory storage systems.
    
    Defines the interface that all memory storage implementations must follow.
    """
    
    @abc.abstractmethod
    async def load_context(self, key: str, context_id: str, default: Any=None) -> Any:
        """
        Load data from memory.
        
        Args:
            key: The key to load
            context_id: The context identifier
            default: Default value to return if key not found
            
        Returns:
            The stored data or default if not found
        """
        pass

    @abc.abstractmethod
    async def save_context(self, key: str, context_id: str, data: Any, ttl: Optional[int]=None) -> bool:
        """
        Save data to memory.
        
        Args:
            key: The key to save
            context_id: The context identifier
            data: The data to save
            ttl: Time-to-live in seconds, or None for no expiration
            
        Returns:
            bool: True if save was successful
        """
        pass

    @abc.abstractmethod
    async def delete_context(self, key: str, context_id: str) -> bool:
        """
        Delete data from memory.
        
        Args:
            key: The key to delete
            context_id: The context identifier
            
        Returns:
            bool: True if deletion was successful
        """
        pass

    @abc.abstractmethod
    async def clear(self, context_id: Optional[str]=None) -> bool:
        """
        Clear all data for a context, or all contexts if none specified.
        
        Args:
            context_id: The context identifier, or None for all contexts
            
        Returns:
            bool: True if clearing was successful
        """
        pass

    @abc.abstractmethod
    async def list_keys(self, context_id: Optional[str]=None, pattern: Optional[str]=None) -> List[str]:
        """
        List keys matching pattern within a context.
        
        Args:
            context_id: The context identifier, or None for all contexts
            pattern: Optional pattern for filtering keys
            
        Returns:
            List[str]: List of matching keys
        """
        pass

    @abc.abstractmethod
    async def exists(self, key: str, context_id: str) -> bool:
        """
        Check if a key exists in memory.
        
        Args:
            key: The key to check
            context_id: The context identifier
            
        Returns:
            bool: True if key exists
        """
        pass

    @abc.abstractmethod
    async def bulk_load(self, keys: List[str], context_id: str, default: Any=None) -> Dict[str, Any]:
        """
        Load multiple keys at once.
        
        Args:
            keys: List of keys to load
            context_id: The context identifier
            default: Default value for keys not found
            
        Returns:
            Dict[str, Any]: Dictionary of key-value pairs
        """
        pass

    @abc.abstractmethod
    async def bulk_save(self, data: Dict[str, Any], context_id: str, ttl: Optional[int]=None) -> bool:
        """
        Save multiple key-value pairs at once.
        
        Args:
            data: Dictionary of key-value pairs to save
            context_id: The context identifier
            ttl: Time-to-live in seconds, or None for no expiration
            
        Returns:
            bool: True if save was successful
        """
        pass

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory implementation.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        logger.debug(f'Getting basic stats for memory implementation: {self.__class__.__name__}')
        try:
            keys = await self.list_keys()
            num_keys = len(keys)
        except Exception as e:
            logger.warning(f'Could not retrieve key list for stats in {self.__class__.__name__}: {e}')
            num_keys = -1
        return {'implementation_type': self.__class__.__name__, 'total_keys_found': num_keys}


class BaseVectorStore(abc.ABC):
    """
    Abstract base class for vector storage systems.
    
    Defines the interface that all vector storage implementations must follow.
    """
    
    @abc.abstractmethod
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
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    async def delete_vectors(self, ids: Optional[List[str]]=None, context_id: Optional[str]=None) -> bool:
        """
        Delete vectors by ID or context.
        
        Args:
            ids: Optional list of vector IDs to delete
            context_id: Optional context identifier
            
        Returns:
            bool: True if deletion was successful
        """
        pass

    @abc.abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        pass
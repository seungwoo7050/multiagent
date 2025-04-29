import abc
from typing import Any, Dict, List, Optional, Set, Tuple, Union

class BaseMemory(abc.ABC):

    @abc.abstractmethod
    async def load_context(self, key: str, context_id: str, default: Any=None) -> Any:
        pass

    @abc.abstractmethod
    async def save_context(self, key: str, context_id: str, data: Any, ttl: Optional[int]=None) -> bool:
        pass

    @abc.abstractmethod
    async def delete_context(self, key: str, context_id: str) -> bool:
        pass

    @abc.abstractmethod
    async def clear(self, context_id: Optional[str]=None) -> bool:
        pass

    @abc.abstractmethod
    async def list_keys(self, context_id: Optional[str]=None, pattern: Optional[str]=None) -> List[str]:
        pass

    @abc.abstractmethod
    async def exists(self, key: str, context_id: str) -> bool:
        pass

    @abc.abstractmethod
    async def bulk_load(self, keys: List[str], context_id: str, default: Any=None) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def bulk_save(self, data: Dict[str, Any], context_id: str, ttl: Optional[int]=None) -> bool:
        pass

    async def get_stats(self) -> Dict[str, Any]:
        logger.debug(f'Getting basic stats for memory implementation: {self.__class__.__name__}')
        try:
            keys = await self.list_keys()
            num_keys = len(keys)
        except Exception as e:
            logger.warning(f'Could not retrieve key list for stats in {self.__class__.__name__}: {e}')
            num_keys = -1
        return {'implementation_type': self.__class__.__name__, 'total_keys_found': num_keys}

class BaseVectorStore(abc.ABC):

    @abc.abstractmethod
    async def store_vector(self, text: str, metadata: Dict[str, Any], vector: Optional[List[float]]=None, context_id: Optional[str]=None) -> str:
        pass

    @abc.abstractmethod
    async def search_vectors(self, query: str, k: int=5, context_id: Optional[str]=None, filter_metadata: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def delete_vectors(self, ids: Optional[List[str]]=None, context_id: Optional[str]=None) -> bool:
        pass

    @abc.abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass
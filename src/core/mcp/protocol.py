import abc
from typing import Any, Dict, Union
from pydantic import BaseModel

class ContextProtocol(abc.ABC, BaseModel):
    version: str = '1.0.0'

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ContextProtocol':
        pass

    @abc.abstractmethod
    def optimize(self) -> 'ContextProtocol':
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {'version': self.version, 'context_type': self.__class__.__name__}

    model_config = {
        "arbitrary_types_allowed": True,
    }
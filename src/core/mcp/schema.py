from typing import Any, Dict, Optional, cast
from pydantic import Field, Json
import json
import time
from src.core.mcp.protocol import ContextProtocol
from src.utils.ids import generate_uuid

class BaseContextSchema(ContextProtocol):
    context_id: str = Field(default_factory=generate_uuid, description='Unique identifier for this context instance')
    timestamp: float = Field(default_factory=time.time, description='Timestamp of context creation/update')
    metadata: Dict[str, Any] = Field(default_factory=dict, description='Optional metadata')

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump(mode='json')

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'BaseContextSchema':
        return cls.model_validate(data)

    def optimize(self) -> 'BaseContextSchema':
        optimized = self.model_copy()
        if not optimized.metadata:
            # Empty metadata dict should be preserved but flagged as optimized
            setattr(optimized, '_optimization_applied_inplace', True)
        return optimized

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> 'BaseContextSchema':
        return cls.model_validate_json(json_str)

class TaskContext(BaseContextSchema):
    task_id: str
    task_type: str
    input_data: Optional[Dict[str, Any]] = None
    current_step: Optional[str] = None

    def optimize(self) -> 'TaskContext':
        optimized = super().optimize()
        return cast(TaskContext, optimized)
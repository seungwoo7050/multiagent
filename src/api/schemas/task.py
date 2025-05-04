# src/api/schemas/task.py
from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class TaskPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def as_int(self) -> int:
        return {"LOW": 1, "NORMAL": 2, "HIGH": 3, "CRITICAL": 4}[self.value]

class CreateTaskRequest(BaseModel):
    goal: str
    task_type: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    # int 또는 Enum 모두 허용
    priority: Union[TaskPriority, int] = TaskPriority.NORMAL
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CreateTaskResponse(BaseModel):
    task_id: str = Field(..., description="새로 생성된 작업의 고유 ID")
    status: str = Field(default="submitted", description="작업 제출 상태")

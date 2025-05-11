from enum import Enum

class TaskPriority(str, Enum):
    """작업의 우선순위를 나타내는 Enum"""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def as_int(self) -> int:
        """우선순위를 정수 값으로 변환 (낮을수록 높음)"""
        return {"LOW": 4, "NORMAL": 3, "HIGH": 2, "CRITICAL": 1}[self.value]

class TaskState(str, Enum):
    """작업의 상태를 나타내는 Enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"                              
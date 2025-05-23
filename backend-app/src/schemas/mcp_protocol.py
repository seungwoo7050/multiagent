import abc
import time
from typing import Any, Dict, Optional, cast
from pydantic import BaseModel, Field, ConfigDict
from src.utils.ids import generate_uuid


class ContextProtocol(abc.ABC, BaseModel):
    """
    모든 MCP 컨텍스트 객체가 따라야 하는 기본 프로토콜 (인터페이스).
    버전 관리 및 기본 직렬화/역직렬화, 최적화 메서드를 정의합니다.
    Pydantic BaseModel을 상속하여 기본적인 유효성 검사 및 직렬화 기능을 활용합니다.
    """

    version: str = Field(default="1.0.0", description="MCP 버전")

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """컨텍스트 객체를 직렬화 가능한 사전(dict) 형태로 변환합니다."""

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> "ContextProtocol":
        """사전(dict) 데이터로부터 컨텍스트 객체를 역직렬화합니다."""

    @abc.abstractmethod
    def optimize(self) -> "ContextProtocol":
        """컨텍스트 데이터 최적화 (예: 압축, 불필요한 정보 제거)를 수행합니다."""

    def get_metadata(self) -> Dict[str, Any]:
        """컨텍스트의 메타데이터를 반환합니다."""

        return {"version": self.version, "context_type": self.__class__.__name__}

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class BaseContextSchema(ContextProtocol):
    """
    모든 구체적인 MCP 컨텍스트 스키마의 기본 클래스.
    공통 필드(context_id, timestamp, metadata)와 기본 구현을 제공합니다.
    """

    context_id: str = Field(
        default_factory=generate_uuid, description="컨텍스트 인스턴스의 고유 식별자"
    )
    timestamp: float = Field(
        default_factory=time.time, description="컨텍스트 생성/갱신 타임스탬프"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터 (예: trace_id, user_id)"
    )

    def serialize(self) -> Dict[str, Any]:
        """Pydantic 모델을 사용하여 직렬화합니다."""

        return self.model_dump(mode="json")

    @classmethod
    def deserialize(
        cls: type["BaseContextSchema"], data: Dict[str, Any]
    ) -> "BaseContextSchema":
        """Pydantic 모델을 사용하여 역직렬화합니다."""

        return cls.model_validate(data)

    def optimize(self) -> "BaseContextSchema":
        """기본 최적화 (메타데이터 정리 등). 하위 클래스에서 재정의 가능."""

        optimized = self.model_copy()

        if not optimized.metadata:
            pass
        return cast(BaseContextSchema, optimized)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class TaskContext(BaseContextSchema):
    """특정 작업을 나타내는 컨텍스트 예시"""

    task_id: str = Field(..., description="연관된 작업의 ID")
    task_type: str = Field(..., description="작업 유형")
    input_data: Optional[Dict[str, Any]] = Field(None, description="작업 입력 데이터")
    current_step: Optional[str] = Field(None, description="현재 진행 중인 단계")

    def optimize(self) -> "TaskContext":
        optimized = super().optimize()

        return cast(TaskContext, optimized)

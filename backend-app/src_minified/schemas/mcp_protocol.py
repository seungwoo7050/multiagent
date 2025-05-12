G=classmethod
C=True
import abc as B,time
from typing import Any,Dict,Optional,cast as D
from pydantic import BaseModel as H,Field as A,ConfigDict as E
from src.utils.ids import generate_uuid as I
class J(B.ABC,H):
	version=A(default='1.0.0',description='MCP 버전')
	@B.abstractmethod
	def serialize(self):0
	@G
	@B.abstractmethod
	def deserialize(cls,data):0
	@B.abstractmethod
	def optimize(self):0
	def get_metadata(A):return{'version':A.version,'context_type':A.__class__.__name__}
	model_config=E(arbitrary_types_allowed=C)
class F(J):
	context_id=A(default_factory=I,description='컨텍스트 인스턴스의 고유 식별자');timestamp=A(default_factory=time.time,description='컨텍스트 생성/갱신 타임스탬프');metadata=A(default_factory=dict,description='추가 메타데이터 (예: trace_id, user_id)')
	def serialize(A):return A.model_dump(mode='json')
	@G
	def deserialize(cls,data):return cls.model_validate(data)
	def optimize(B):
		A=B.model_copy()
		if not A.metadata:0
		return D(F,A)
	model_config=E(arbitrary_types_allowed=C,validate_assignment=C)
class K(F):
	task_id=A(...,description='연관된 작업의 ID');task_type=A(...,description='작업 유형');input_data=A(None,description='작업 입력 데이터');current_step=A(None,description='현재 진행 중인 단계')
	def optimize(B):A=super().optimize();return D(K,A)
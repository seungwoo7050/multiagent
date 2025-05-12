B=None
from typing import Any,Dict,Literal,Optional
from pydantic import BaseModel as D,Field as A
import datetime as E
class C(D):event_type=A(...,description='메시지 이벤트 유형');timestamp=A(default_factory=E.datetime.now,description='메시지 생성 타임스탬프 (UTC 권장)');task_id=A(...,description='관련 작업 ID')
class F(C):event_type='status_update';status=A(...,description="새로운 작업 상태 (예: 'running', 'tool_called', 'completed', 'failed')");detail=A(B,description='상태에 대한 추가 설명');current_node=A(B,description='현재 실행 중이거나 완료된 노드 ID');next_node=A(B,description='다음에 실행될 것으로 예상되는 노드 ID')
class G(C):event_type='intermediate_result';node_id=A(...,description='결과를 생성한 노드 ID');result_step_name=A(...,description='중간 결과 단계 이름 또는 설명');data=A(...,description='중간 결과 데이터')
class H(C):event_type='final_result';final_answer=A(B,description='워크플로우의 최종 답변');error_message=A(B,description='오류 발생 시 최종 오류 메시지');metadata=A(default_factory=dict,description='최종 결과 관련 추가 메타데이터')
class I(C):event_type='error';error_code=A(...,description='오류 코드 (내부 정의)');message=A(...,description='오류 메시지');details=A(B,description='오류 상세 정보')
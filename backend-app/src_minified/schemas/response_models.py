H='Context ID'
G='도구 설명'
F='도구 이름'
E=dict
D=list
B=None
from typing import Any,Dict,List,Optional,Literal
from pydantic import BaseModel as C,Field as A
class I(C):task_id=A(...,description='새로 생성된 작업의 고유 ID');status=A(default='accepted',description='작업 접수 상태')
class J(C):task_id=A(...,description='작업 ID');status=A(...,description="워크플로우의 현재 상태 (예: 'running', 'completed', 'failed', 'pending')",examples=['running','completed','failed']);final_answer=A(B,description='워크플로우 최종 결과 (완료 시). AgentGraphState의 final_answer 필드에 해당합니다.');error_message=A(B,description='오류 발생 시 메시지. AgentGraphState의 error_message 필드에 해당합니다.');current_iteration=A(B,description='현재 반복 횟수 (AgentGraphState의 current_iteration)');search_depth=A(B,description='현재 탐색 깊이 (ToT 경우, AgentGraphState의 search_depth)');last_llm_output=A(B,description='마지막 LLM 출력 (디버깅용, AgentGraphState의 last_llm_output)');metadata=A(B,description='워크플로우 관련 메타데이터 (AgentGraphState의 metadata)')
class K(C):title=B;type=B;description=B;default=B;format=B;items=B;properties=B
class L(C):title=A(...,description='도구 인자 스키마 제목 (보통 도구 이름)');type=A(default='object',description="스키마 타입 (일반적으로 'object')");properties=A(default_factory=E,description='인자 속성 정의');required=A(default_factory=D,description='필수 인자 목록')
class M(C):name=A(...,description=F);description=A(...,description=G);args_schema_summary=A(B,description='인자 스키마 요약 (인자명: 타입)')
class N(C):name=A(...,description=F);description=A(...,description=G);args_schema=A(...,description='도구의 상세 인자 스키마')
class O(C):status=A(...,description="'success' 또는 'error'");tool_name=A(...,description='실행된 도구 이름');execution_time=A(B,description='실행 시간(초)');result=A(B,description='도구 실행 결과 (성공 시)');error=A(B,description='에러 상세 정보 (실패 시)')
class P(C):context_id=A(...,description=H);data=A(...,description='Context 데이터')
class Q(C):context_id=A(...,description=H);status=A(...,description='작업 상태');message=A(...,description='결과 메시지')
class R(C):name=A(...,description='에이전트 이름');agent_type=A(...,description='에이전트 타입');description=A(B,description='에이전트 설명');version=A(...,description='에이전트 버전')
class S(C):name:0;description=B;version:0;agent_type:0;model=B;capabilities=A(default_factory=D);parameters=A(default_factory=E);max_retries:0;timeout:0;allowed_tools=A(default_factory=D);memory_keys=A(default_factory=D);metadata=A(default_factory=E);mcp_enabled:0;mcp_context_types=A(default_factory=D)
class T(C):status=A(default='ok',description='시스템 상태')
class U(C):name=A(...,description='그래프 설정 파일 이름 (확장자 제외)');description=A(B,description='그래프 설명 (설정 파일 내)')
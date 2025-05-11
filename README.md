# 멀티 에이전트 시스템 (v2.0)

이 프로젝트는 LLM 기반 멀티 에이전트 시스템으로, 프레임워크 중심 아키텍처를 활용하여 복잡한 문제 해결 및 작업 처리를 수행합니다. 이 시스템은 Tree of Thoughts(ToT), Task Division, ReAct 패턴을 통합하여 다양한 복잡도의 작업을 효율적으로 처리합니다.

## 주요 기능

- **다중 에이전트 오케스트레이션**: LangGraph를 사용한 동적 워크플로우 구성
- **Tree of Thoughts(ToT) 추론**: 복잡한 문제에 대한 다중 사고 경로 탐색
- **작업 분할 기능**: 복잡한 작업을 관리 가능한 하위 작업으로 분할
- **ReAct 도구 통합**: 외부 도구를 활용한 추론 및 행동 패턴
- **LLM 대체(Fallback) 로직**: 안정적인 LLM 호출을 위한 다중 공급자 지원
- **대화 컨텍스트 관리**: 대화 기록 유지 및 활용

## 시스템 아키텍처

### 백엔드

```
project_root/
├── api/                       # FastAPI 기반 HTTP API 레이어
├── agents/                    # 에이전트 오케스트레이션 (LangGraph 사용)
│   └── graph_nodes/           # 다양한 노드 구현 (ToT, Task Division, ReAct)
├── services/                  # 서비스 어댑터 및 유틸리티
├── tools/                     # 플러그인 방식의 도구들
├── memory/                    # 메모리 저장 및 관리 시스템
├── schemas/                   # Pydantic/msgspec 스키마 정의
├── config/                    # 설정 로더 및 파일들
│   ├── agent_graphs/          # 동적 에이전트 그래프 구성
│   └── prompts/               # 프롬프트 템플릿
└── utils/                     # 공통 유틸리티
```

### 프론트엔드

```
frontend/
├── public/                    # 정적 에셋
├── src/                       # 소스 코드
│   ├── api/                   # API 클라이언트 모듈
│   ├── components/            # 재사용 가능한 UI 컴포넌트 (채팅, 대시보드)
│   ├── contexts/              # React Context를 사용한 상태 관리
│   ├── hooks/                 # 커스텀 React 훅
│   ├── pages/                 # 최상위 페이지 컴포넌트
│   ├── App.js                 # 메인 애플리케이션 컴포넌트
│   └── index.js               # 진입점
└── package.json               # 프론트엔드 의존성
```

## 설치 방법

### 백엔드 설치

```bash
# 저장소 클론
git clone https://github.com/your-username/multi-agent-system.git
cd multi-agent-system

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키 및 설정 추가 (.env.example 참조)
```

### 프론트엔드 설치

```bash
cd frontend
npm install
```

## 실행 방법

### 백엔드 실행

```bash
# 개발 모드로 실행
poetry run uvicorn api.main:app --reload

# 또는 프로덕션 모드로 실행
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 프론트엔드 실행

```bash
cd frontend
npm run dev
```

## API 사용법

### 기본 엔드포인트

- `POST /api/v1/run`: 에이전트 워크플로우 실행
- `GET /api/v1/status/{task_id}`: 작업 상태 및 결과 확인
- `GET /api/v1/graphs`: 사용 가능한 에이전트 그래프 구성 목록 조회
- `WebSocket /ws/status/{task_id}`: 실시간 작업 상태 업데이트

### 요청 예시

```json
POST /api/v1/run
{
  "task_type": "complex_problem_solving",
  "input_data": {
    "problem_description": "효율적인 재택근무 정책을 설계하라",
    "constraints": ["예산 < 100만원", "1주일 내 구현 가능"]
  },
  "metadata": {
    "request_id": "req-abc-123",
    "graph_config_name": "default_tot_workflow",
    "conversation_uuid": "conv-xyz-789"
  }
}
```

### 응답 예시

```json
{
  "task_id": "task-xyz-789",
  "status": "completed",
  "result": {
    "solution": "제안된 재택근무 정책은...",
    "confidence_score": 0.85,
    "reasoning_steps_summary": ["사고 1 -> 평가 -> 사고 2..."]
  },
  "metadata": {
     "request_id": "req-abc-123",
     "model_used": "gpt-4",
     "total_tokens": 1500,
     "graph_used": "default_tot_workflow"
  }
}
```

## 동적 에이전트 그래프 구성

에이전트 워크플로우는 `config/agent_graphs/` 디렉토리의 JSON 파일로 정의됩니다:

### Tree of Thoughts (ToT) 그래프

```json
{
  "nodes": {
    "thought_generator": {
      "type": "ThoughtGeneratorNode",
      "config": {
        "num_thoughts": 3,
        "prompt_template": "generate_thought_style_A.txt"
      }
    },
    "state_evaluator": {
      "type": "StateEvaluatorNode",
      "config": {
        "evaluation_criteria": ["relevance", "feasibility", "originality"],
        "prompt_template": "evaluate_thought.txt"
      }
    }
  },
  "edges": [
    {
      "source": "thought_generator",
      "target": "state_evaluator"
    }
  ]
}
```

### 작업 분할 그래프

```json
{
  "nodes": {
    "task_complexity_evaluator": {
      "type": "TaskComplexityEvaluatorNode",
      "config": {
        "complexity_thresholds": {
          "simple": 0.3,
          "medium": 0.7,
          "complex": 1.0
        }
      }
    },
    "task_division": {
      "type": "TaskDivisionNode",
      "config": {
        "max_subtasks": 5
      }
    }
  },
  "edges": [
    {
      "source": "task_complexity_evaluator",
      "target": "task_division",
      "condition": "complexity > 0.7"
    }
  ]
}
```

### ReAct 도구 워크플로우

```json
{
  "nodes": {
    "react_agent": {
      "type": "GenericLLMNode",
      "config": {
        "prompt_template": "react_tool_agent.txt",
        "enable_tool_use": true,
        "allowed_tools": ["calculator", "web_search"]
      }
    }
  }
}
```

## 도구 통합

새로운 도구는 `tools/` 디렉토리에 파일을 추가하고 `BaseTool` 클래스를 상속하여 쉽게 구현할 수 있습니다:

```python
from tools.base import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(..., description="계산할 수학 표현식")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "간단한 수학 계산을 수행합니다. 사칙연산, 제곱근, 지수 등의 계산에 사용하세요."
    args_schema = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"계산 결과: {result}"
        except Exception as e:
            return f"계산 오류: {str(e)}"
```

## 메모리 시스템 사용

메모리 시스템은 대화 이력 및 상태 유지를 위해 사용됩니다:

```python
from memory.memory_manager import MemoryManager

memory_manager = MemoryManager()

# 상태 저장
await memory_manager.save_state(
    "conversation:abc123", 
    {"messages": [{"role": "user", "content": "안녕하세요"}]},
    ttl=3600  # 1시간 후 만료
)

# 상태 조회
state = await memory_manager.load_state("conversation:abc123")

# 대화 이력 조회
history = await memory_manager.get_history("conversation:abc123", limit=10)
```

## 대화 컨텍스트 관리

시스템은 `conversation_uuid`를 통해 대화 컨텍스트를 관리합니다:

```python
# API 요청에 conversation_uuid 포함
{
  "input_data": { "query": "이전 대화 내용을 참고해서 추가 질문에 답변해줘" },
  "metadata": { "conversation_uuid": "conv-xyz-789" }
}
```

백엔드에서는 이 UUID를 사용하여 메모리 시스템에서 이전 대화 내용을 검색하고 LLM 요청에 포함시킵니다.

## 프론트엔드 기능

프론트엔드는 세 개의 패널로 구성된 React 애플리케이션입니다:

1. **왼쪽 패널 - 대화 목록**:
   - 모든 대화를 시간 역순으로 표시
   - "새 대화" 버튼으로 새로운 대화 시작
   - 대화 선택하여 중앙 패널에 표시

2. **중앙 패널 - 활성 대화**:
   - 선택된 대화의 메시지 이력 표시
   - 메시지 입력 필드와 전송 버튼
   - 실시간 에이전트 응답 표시

3. **오른쪽 패널 - 작업 모니터링 대시보드**:
   - 모든 대화에 걸친 작업 목록과 상태 표시
   - WebSocket을 통한 실시간 상태 업데이트
   - 작업 상세 정보 보기 기능

## 개발자 가이드

### 새로운, 노드 추가하기

1. `agents/graph_nodes/` 디렉토리에 새 노드 클래스 구현
2. 필요한 경우 `config/prompts/`에 프롬프트 템플릿 추가
3. `config/agent_graphs/`에 노드를 사용하는 그래프 구성 추가

### 새로운 도구 추가하기

1. `tools/` 디렉토리에 `BaseTool`을 상속하는 새 도구 클래스 구현
2. 도구 이름, 설명, 인수 스키마 정의
3. `_run` 및/또는 `_arun` 메서드 구현

### 테스트 실행

```bash
# 전체 테스트 실행
poetry run pytest

# 특정 모듈 테스트
poetry run pytest tests/test_agents.py

# 특정 테스트 실행
poetry run pytest tests/test_agents.py::test_tot_workflow
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
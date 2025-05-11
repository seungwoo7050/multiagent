"""
LangGraph의 노드로 작동 (일반적으로 함수 또는 Runnable 객체).
설정 (JSON)으로부터 prompt_template_path (예: config/prompts/some_prompt.txt), 사용할 LLM 모델 이름, temperature 등의 파라미터를 받을 수 있어야 합니다.
llm_client (3단계에서 구현한 services/llm_client.py)를 사용하여 LLM 호출을 수행합니다.
그래프 상태(AgentGraphState)에서 필요한 컨텍스트를 가져와 프롬프트를 포맷팅합니다.
LLM 호출 결과를 다시 그래프 상태에 업데이트합니다.
"""
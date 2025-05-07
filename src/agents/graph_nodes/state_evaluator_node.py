# multiagent/src/agents/graph_nodes/state_evaluator_node.py

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig

from src.config.logger import get_logger
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState, Thought # Step 4.1에서 정의

logger = get_logger(__name__)

class CasePreservingStr(str):
    def lower(self):            # lower() 해도 그대로 반환
        return self

class StateEvaluatorNode:
    """
    생성된 생각(상태)들을 평가하여 점수를 매기는 ToT 노드.
    """
    def __init__(
        self,
        llm_client: LLMClient,
        max_tokens_per_eval: int = 100,
        temperature: float = 0.4,
        prompt_template_path: Optional[str] = None, # 평가용 프롬프트
        model_name: Optional[str] = None,
        node_id: str = "state_evaluator"
    ):
        self.llm_client = llm_client
        self.max_tokens_per_eval = max_tokens_per_eval
        self.temperature = temperature
        self.prompt_template_path = prompt_template_path # TODO: 로드 로직
        self.model_name = model_name
        self.node_id = node_id
        logger.info(f"StateEvaluatorNode '{self.node_id}' initialized.")

    def _construct_evaluation_prompt(self, state: AgentGraphState, thought_content: str) -> str:
        # 평가 기준을 명확히 제시
        evaluation_criteria = """
        Evaluation Criteria:
        1. Relevance: How directly does this thought address the overall Goal?
        2. Progress: How likely is this thought to lead to significant progress towards the Goal?
        3. Feasibility: How practical and achievable is this thought given potential constraints?
        4. Novelty/Insight: Does this thought offer a new perspective or a clever approach? (Less critical but good)
        5. Risk: What is the potential for this thought to lead to a dead-end or waste resources? (Lower is better)
        """

        parent_thought_content = "N/A (This is an initial thought)"
        if state.thoughts: # 현재 평가 대상 생각의 부모 찾기 (예시)
            current_thought = next((t for t in state.thoughts if t.content == thought_content and t.status == "generated"), None) # 더 나은 ID 기반 검색 필요
            if current_thought and current_thought.parent_id:
                parent = state.get_thought_by_id(current_thought.parent_id)
                if parent:
                    parent_thought_content = parent.content
        
        prompt = f"""
        You are an expert evaluator. Your task is to assess the promise of a given "Thought" in the context of achieving an overall "Goal".

        Overall Goal: {state.original_input}

        Context (Parent Thought or Current Focus):
        {parent_thought_content}

        Thought to Evaluate:
        "{thought_content}"

        {evaluation_criteria}

        Instructions:
        Provide a numerical score between 0.0 (not useful) and 1.0 (highly promising).
        Then, provide a concise reasoning for your score, referencing the criteria.

        Output Format:
        Score: [float between 0.0 and 1.0]
        Reasoning: [Your brief textual reasoning]

        Evaluation:
        """
        return prompt


    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        # ... (기존 로직 유지, 단 LLM 응답 파싱 시 "Score: "와 "Reasoning: "을 더 잘 찾도록 개선)
        # 예: 정규식 사용
        # import re
        # score_match = re.search(r"Score:\s*([0-9.]+)", eval_response_str, re.IGNORECASE)
        # reasoning_match = re.search(r"Reasoning:\s*(.+)", eval_response_str, re.IGNORECASE | re.DOTALL)
        # score = float(score_match.group(1)) if score_match else 0.0
        # reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
        # ...
        # (이전 답변의 _call__ 메서드 내용 참고하여 업데이트)
        logger.info(f"StateEvaluatorNode '{self.node_id}' execution started for task: {state.task_id}")
        
        updated_thoughts_map = {t.id: t for t in state.thoughts} # ID로 쉽게 접근하기 위한 맵
        error_message: Optional[str] = None
        any_evaluation_done = False
        
        if not state.current_thoughts_to_evaluate:
            logger.info(f"Node '{self.node_id}': No thoughts to evaluate.")
            return {"current_thoughts_to_evaluate": []} # thoughts는 변경 없으므로 전달 안함 (LangGraph가 병합)

        evaluations_summary = []

        for thought_id_to_eval in state.current_thoughts_to_evaluate:
            thought_to_eval = updated_thoughts_map.get(thought_id_to_eval)
            
            if not thought_to_eval:
                logger.warning(f"Node '{self.node_id}': Thought with ID '{thought_id_to_eval}' not found in state. Skipping evaluation.")
                continue

            try:
                eval_prompt = self._construct_evaluation_prompt(state, thought_to_eval.content)
                logger.debug(f"Node '{self.node_id}': Evaluation prompt for thought '{thought_to_eval.id}':\n{eval_prompt[:300]}...")

                messages = [{"role": "user", "content": eval_prompt}]
                eval_response_str = await self.llm_client.generate_response(
                    messages=messages,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens_per_eval
                )
                logger.debug(f"Node '{self.node_id}': Evaluation response for thought '{thought_to_eval.id}': {eval_response_str}")

                score = 0.0
                reasoning = "Could not parse evaluation."
                
                import re
                score_match = re.search(r"Score:\s*([0-9.]+)", eval_response_str, re.IGNORECASE)
                reasoning_match = re.search(r"Reasoning:\s*([\s\S]+)", eval_response_str, re.IGNORECASE) # DOTALL 대신 [\s\S]

                if score_match:
                    try:
                        score = float(score_match.group(1))
                    except ValueError:
                        logger.warning(f"Node '{self.node_id}': Could not parse score float from: '{score_match.group(1)}'")
                else:
                    logger.warning(f"Node '{self.node_id}': 'Score:' pattern not found in response: {eval_response_str}")

                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                else:
                    logger.warning(f"Node '{self.node_id}': 'Reasoning:' pattern not found in response: {eval_response_str}")

                reasoning = CasePreservingStr(reasoning)

                # 새 Thought 객체 생성 (불변성 유지 시도)
                updated_metadata = thought_to_eval.metadata.copy() if thought_to_eval.metadata else {}
                updated_metadata['eval_reasoning'] = reasoning
                updated_metadata['raw_eval_response'] = eval_response_str

                new_thought_version = Thought(
                    id=thought_to_eval.id,
                    parent_id=thought_to_eval.parent_id,
                    content=thought_to_eval.content,
                    evaluation_score=score,
                    status="evaluated",
                    metadata=updated_metadata
                )
                updated_thoughts_map[thought_id_to_eval] = new_thought_version
                any_evaluation_done = True
                evaluations_summary.append(f"Thought '{thought_to_eval.id}': Score={score:.2f}")

            except Exception as e:
                logger.error(f"Node '{self.node_id}': Error evaluating thought '{thought_id_to_eval}': {e}", exc_info=True)
                failed_metadata = thought_to_eval.metadata.copy() if thought_to_eval.metadata else {}
                failed_metadata['eval_error'] = str(e)
                
                failed_thought_version = Thought(
                    id=thought_to_eval.id,
                    parent_id=thought_to_eval.parent_id,
                    content=thought_to_eval.content,
                    evaluation_score=thought_to_eval.evaluation_score, # 기존 점수 유지 또는 None
                    status="evaluation_failed",
                    metadata=failed_metadata
                )
                updated_thoughts_map[thought_id_to_eval] = failed_thought_version
                any_evaluation_done = True # 실패도 처리된 것으로 간주
                error_message = (error_message or "") + f"Error evaluating {thought_id_to_eval}: {str(e)}; "

        final_thoughts_list = list(updated_thoughts_map.values())
        logger.info(f"Node '{self.node_id}': Evaluations completed. Summary: {', '.join(evaluations_summary if evaluations_summary else ['No new evaluations.'])}")
        
        update_payload = {
            "thoughts": final_thoughts_list,
            "current_thoughts_to_evaluate": [] # 평가 완료
        }
        if any_evaluation_done : # LLM 호출이 있었다면 로그
            update_payload["last_llm_output"] = f"Evaluations completed for thoughts: {', '.join(state.current_thoughts_to_evaluate)}."
        if error_message:
            update_payload["error_message"] = error_message
            
        return update_payload
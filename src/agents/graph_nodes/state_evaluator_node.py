# src/agents/graph_nodes/state_evaluator_node.py
import os # os 임포트 추가
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig

from src.config.logger import get_logger
from src.config.settings import get_settings # settings 임포트 추가
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState, Thought
from src.services.notification_service import NotificationService # 추가
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage # 추가

logger = get_logger(__name__)
settings = get_settings() # settings 인스턴스

# CasePreservingStr 클래스 정의는 유지

class StateEvaluatorNode:
    def __init__(
        self,
        llm_client: LLMClient,
        notification_service: NotificationService, # <--- 추가
        max_tokens_per_eval: int = 100,
        temperature: float = 0.4,
        prompt_template_path: Optional[str] = None,
        model_name: Optional[str] = None,
        node_id: str = "state_evaluator"
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service # <--- 저장
        self.max_tokens_per_eval = max_tokens_per_eval
        self.temperature = temperature
        self.prompt_template_path = prompt_template_path
        self.model_name = model_name
        self.node_id = node_id
        self.prompt_template_str = self._load_prompt_template_if_path_exists() # 프롬프트 로드
        logger.info(
            f"StateEvaluatorNode '{self.node_id}' initialized. "
            f"NotificationService injected: {'Yes' if notification_service else 'No'}. "
            f"Prompt: '{self.prompt_template_path if self.prompt_template_path else 'Default internal'}'"
        )

    def _load_prompt_template_if_path_exists(self) -> Optional[str]:
        # ThoughtGeneratorNode와 동일한 로직 사용 가능
        if not self.prompt_template_path:
            return None
        base_prompt_dir = getattr(settings, 'PROMPT_TEMPLATE_DIR', 'config/prompts')
        if os.path.isabs(self.prompt_template_path):
            full_path = self.prompt_template_path
        else:
            full_path = os.path.join(base_prompt_dir, self.prompt_template_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                logger.debug(f"Successfully loaded prompt template from: {full_path} for node '{self.node_id}'")
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Prompt template file not found for StateEvaluatorNode '{self.node_id}': {full_path}. Using default internal prompt.")
            return None
        except Exception as e:
            logger.error(f"Error loading prompt template from {full_path} for node '{self.node_id}': {e}. Using default internal prompt.")
            return None

    def _construct_evaluation_prompt(self, state: AgentGraphState, thought_content: str, parent_thought_content: str) -> str:
        if self.prompt_template_str:
            # 로드된 프롬프트 템플릿 사용
            from langchain_core.prompts import PromptTemplate
            try:
                 prompt_data = {
                     "original_input": state.original_input,
                     "parent_thought_content": parent_thought_content, # 명시적으로 전달받음
                     "thought_to_evaluate_content": thought_content
                     # 프롬프트 템플릿에 정의된 다른 변수들 추가 가능
                 }
                 template = PromptTemplate(template=self.prompt_template_str, input_variables=list(prompt_data.keys()))
                 return template.format(**prompt_data)
            except KeyError as ke:
                 logger.error(f"Missing key for prompt template in StateEvaluatorNode '{self.node_id}': {ke}. Falling back to default internal prompt.")
            except Exception as e:
                 logger.error(f"Error formatting prompt template in StateEvaluatorNode '{self.node_id}': {e}. Falling back to default internal prompt.")

        # 기본 내부 프롬프트
        evaluation_criteria = """
        Evaluation Criteria:
        1. Relevance to Goal (Weight: 40%): How directly and significantly does this thought contribute to achieving the "Overall Goal"?
        2. Progress Likelihood (Weight: 30%): How likely is pursuing this thought to lead to tangible progress or valuable new information?
        3. Feasibility & Actionability (Weight: 20%): How practical, actionable, and achievable is this thought given typical constraints (time, resources, information availability)? Is it specific enough to act upon?
        4. Novelty/Insight (Weight: 10%): Does this thought offer a new perspective, a creative approach, or a particularly insightful next step? (Bonus, not strictly required for a good score if other criteria are met).
        5. Risk of Dead-end (Negative Factor): Is there a high risk this thought will lead to a dead-end, be unproductive, or significantly deviate from the goal?
        """
        return f"""
        You are an expert AI Evaluator. Your task is to critically assess the promise and quality of a given "Thought" in the context of achieving an "Overall Goal", potentially originating from a "Parent Thought".

        Overall Goal:
        {state.original_input}

        Parent Thought (Context for the thought being evaluated, if applicable):
        {parent_thought_content}
        (Note: If the thought is an initial thought, Parent Thought might be "Initial problem analysis" or similar.)

        Thought to Evaluate:
        "{thought_content}"

        {evaluation_criteria}

        Instructions for Evaluation:
        1. Provide a numerical score between 0.0 (not at all promising) and 1.0 (highly promising) for the "Thought to Evaluate".
        2. Provide a concise reasoning for your score, briefly touching upon the relevant criteria. Your reasoning should justify the score.

        Output Format:
        Score: [A single float value between 0.0 and 1.0]
        Reasoning: [Your concise textual reasoning, typically 1-3 sentences.]

        Begin Evaluation:
        """

    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        logger.info(f"StateEvaluatorNode '{self.node_id}' execution started. Task ID: {state.task_id}")
        await self.notification_service.broadcast_to_task(
            state.task_id,
            StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' (State Evaluator) started.", current_node=self.node_id)
        )

        updated_thoughts_map = {t.id: t for t in state.thoughts}
        error_message: Optional[str] = None
        any_evaluation_done = False
        evaluations_summary_for_log = [] # 로그용 요약

        if not state.current_thoughts_to_evaluate:
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): No thoughts to evaluate.")
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_completed", detail=f"Node '{self.node_id}' finished: No thoughts to evaluate.", current_node=self.node_id, next_node="search_strategy" if state.thoughts else None) # 다음 노드 정보 추가
            )
            return {"current_thoughts_to_evaluate": []}

        for thought_id_to_eval in state.current_thoughts_to_evaluate:
            thought_to_eval = updated_thoughts_map.get(thought_id_to_eval)

            if not thought_to_eval:
                logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): Thought with ID '{thought_id_to_eval}' not found. Skipping.")
                continue

            parent_thought_content = "N/A (Initial thought or parent not found)"
            if thought_to_eval.parent_id:
                parent = state.get_thought_by_id(thought_to_eval.parent_id)
                if parent:
                    parent_thought_content = parent.content

            try:
                eval_prompt = self._construct_evaluation_prompt(state, thought_to_eval.content, parent_thought_content)
                logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): Evaluation prompt for thought '{thought_to_eval.id}'.")

                messages = [{"role": "user", "content": eval_prompt}]
                eval_response_str = await self.llm_client.generate_response(
                    messages=messages, model_name=self.model_name,
                    temperature=self.temperature, max_tokens=self.max_tokens_per_eval
                )
                logger.debug(f"Node '{self.node_id}' (Task: {state.task_id}): Evaluation response for '{thought_to_eval.id}': {eval_response_str[:100]}...")

                score = 0.0
                reasoning = "Could not parse evaluation."
                score_match = re.search(r"Score:\s*([0-9.]+)", eval_response_str, re.IGNORECASE)
                reasoning_match = re.search(r"Reasoning:\s*([\s\S]+)", eval_response_str, re.IGNORECASE | re.DOTALL) # DOTALL 추가

                if score_match:
                    try: score = float(score_match.group(1))
                    except ValueError: logger.warning(f"Node '{self.node_id}': Could not parse score from: '{score_match.group(1)}'")
                else: logger.warning(f"Node '{self.node_id}': 'Score:' pattern not found.")

                if reasoning_match: reasoning = reasoning_match.group(1).strip()
                else: logger.warning(f"Node '{self.node_id}': 'Reasoning:' pattern not found.")
                
                # CasePreservingStr 사용 여부 결정 (여기선 일단 str)
                # reasoning = CasePreservingStr(reasoning)

                updated_metadata = thought_to_eval.metadata.copy() if thought_to_eval.metadata else {}
                updated_metadata['eval_reasoning'] = reasoning
                updated_metadata['raw_eval_response'] = eval_response_str

                new_thought_version = Thought(
                    id=thought_to_eval.id, parent_id=thought_to_eval.parent_id, content=thought_to_eval.content,
                    evaluation_score=score, status="evaluated", metadata=updated_metadata
                )
                updated_thoughts_map[thought_id_to_eval] = new_thought_version
                any_evaluation_done = True
                evaluations_summary_for_log.append(f"Thought '{thought_to_eval.id}': Score={score:.2f}")

                # 각 생각 평가 완료 시 중간 결과 알림
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    IntermediateResultMessage(
                        task_id=state.task_id, node_id=self.node_id,
                        result_step_name="thought_evaluated",
                        data={"thought_id": thought_id_to_eval, "score": score, "reasoning": reasoning[:150]+"..."} # 긴 내용은 자르기
                    )
                )

            except Exception as e:
                logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): Error evaluating thought '{thought_id_to_eval}': {e}", exc_info=True)
                # 실패 처리 및 알림
                failed_metadata = thought_to_eval.metadata.copy() if thought_to_eval.metadata else {}
                failed_metadata['eval_error'] = str(e)
                failed_thought_version = Thought(
                    id=thought_to_eval.id, parent_id=thought_to_eval.parent_id, content=thought_to_eval.content,
                    evaluation_score=None, status="evaluation_failed", metadata=failed_metadata
                )
                updated_thoughts_map[thought_id_to_eval] = failed_thought_version
                any_evaluation_done = True
                error_message = (error_message or "") + f"Error evaluating {thought_id_to_eval}: {str(e)}; "
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    IntermediateResultMessage(
                        task_id=state.task_id, node_id=self.node_id,
                        result_step_name="thought_evaluation_failed",
                        data={"thought_id": thought_id_to_eval, "error": str(e)}
                    )
                )

        final_thoughts_list = list(updated_thoughts_map.values())
        log_summary_str = ', '.join(evaluations_summary_for_log) if evaluations_summary_for_log else 'No new evaluations.'
        logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Evaluations completed. Summary: {log_summary_str}")

        update_payload = {
            "thoughts": final_thoughts_list, # 수정된 thoughts 리스트 반환
            "current_thoughts_to_evaluate": [] # 평가 완료
        }
        if any_evaluation_done:
            update_payload["last_llm_output"] = f"Evaluations completed for thoughts: {', '.join(state.current_thoughts_to_evaluate)}. Summary: {log_summary_str}"
        if error_message:
            update_payload["error_message"] = error_message

        await self.notification_service.broadcast_to_task(
            state.task_id,
            StatusUpdateMessage(
                task_id=state.task_id, status="node_completed",
                detail=f"Node '{self.node_id}' (State Evaluator) finished. Evaluated {len(state.current_thoughts_to_evaluate)} thoughts.",
                current_node=self.node_id, next_node="search_strategy"
            )
        )
        return update_payload
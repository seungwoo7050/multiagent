import os 
import re
from typing import Any, Dict, List, Optional, TypedDict, Union, Tuple

from langchain_core.runnables import RunnableConfig

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState, Thought
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

logger = get_logger(__name__)
settings = get_settings()

class StateEvaluatorNode:
    def __init__(
        self,
        llm_client: LLMClient,
        notification_service: NotificationService,
        max_tokens_per_eval: int = 150,
        temperature: float = 0.4,
        prompt_template_path: Optional[str] = None,
        model_name: Optional[str] = None,
        node_id: str = "state_evaluator",
        default_score: float = 0.5  # 기본 점수 추가
    ):
        self.llm_client = llm_client
        self.notification_service = notification_service
        self.max_tokens_per_eval = max_tokens_per_eval
        self.temperature = temperature
        self.prompt_template_path = prompt_template_path
        self.model_name = model_name
        self.node_id = node_id
        self.default_score = default_score  # 기본 점수 저장
        self.prompt_template_str = self._load_prompt_template_if_path_exists()
        logger.info(
            f"StateEvaluatorNode '{self.node_id}' initialized. "
            f"NotificationService injected: {'Yes' if notification_service else 'No'}. "
            f"Prompt: '{self.prompt_template_path if self.prompt_template_path else 'Default internal'}'"
        )

    def _load_prompt_template_if_path_exists(self) -> Optional[str]:
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
            from langchain_core.prompts import PromptTemplate
            try:
                 prompt_data = {
                     "original_input": state.original_input,
                     "parent_thought_content": parent_thought_content,
                     "thought_to_evaluate_content": thought_content
                 }
                 template = PromptTemplate(template=self.prompt_template_str, input_variables=list(prompt_data.keys()))
                 return template.format(**prompt_data)
            except KeyError as ke:
                 logger.error(f"Missing key for prompt template in StateEvaluatorNode '{self.node_id}': {ke}. Falling back to default internal prompt.")
            except Exception as e:
                 logger.error(f"Error formatting prompt template in StateEvaluatorNode '{self.node_id}': {e}. Falling back to default internal prompt.")

        # 기본 내부 프롬프트 - 명확하고 간결한 지침 추가
        evaluation_criteria = """
        Evaluation Criteria:
        1. Relevance to Goal (Weight: 40%): How directly and significantly does this thought contribute to achieving the "Overall Goal"?
        2. Progress Likelihood (Weight: 30%): How likely is pursuing this thought to lead to tangible progress or valuable new information?
        3. Feasibility & Actionability (Weight: 20%): How practical, actionable, and achievable is this thought given typical constraints (time, resources, information availability)? Is it specific enough to act upon?
        4. Novelty/Insight (Weight: 10%): Does this thought offer a new perspective, a creative approach, or a particularly insightful next step? (Bonus, not strictly required for a good score if other criteria are met).
        
        IMPORTANT: Be generous with your scoring! If a thought has any merit at all, score it at least 0.3.
        - Score 0.7-1.0: An excellent thought that directly addresses the goal
        - Score 0.5-0.69: A good thought with clear value 
        - Score 0.3-0.49: An average thought with some potential
        - Score 0.1-0.29: A weak thought with limited value
        - Score 0-0.09: A completely irrelevant or impractical thought
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
        2. Remember: Be generous in your scoring. If the thought is directly useful, score it 0.7 or higher.
        3. Provide a concise reasoning for your score, briefly touching upon the relevant criteria. Your reasoning should justify the score.

        Output Format:
        Score: [A single float value between 0.0 and 1.0]
        Reasoning: [Your concise textual reasoning, typically 1-3 sentences.]

        Begin Evaluation:
        """

    def _parse_evaluation_response(self, response_str: str) -> Tuple[Optional[float], str]:
        """평가 응답 문자열을 파싱하여 점수와 이유를 추출"""
        response_str = response_str.strip()
        
        # 1. 일반적인 Score: X.Y 포맷 찾기
        score_match = re.search(r"Score:\s*([0-9.]+)", response_str, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                if 0 <= score <= 1:
                    reasoning_match = re.search(r"Reasoning:\s*([\s\S]+)", response_str, re.IGNORECASE | re.DOTALL)
                    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
                    return score, reasoning
            except ValueError:
                pass  # 숫자 변환 실패 - 다음 방법 시도
        
        # 2. 첫 줄이 숫자만 있는 경우 (Score: 없이)
        lines = response_str.split('\n')
        if lines and lines[0].strip().replace('.', '', 1).isdigit():
            try:
                score = float(lines[0].strip())
                if 0 <= score <= 1:
                    reasoning = '\n'.join(lines[1:]).strip() or "No reasoning provided."
                    return score, reasoning
            except ValueError:
                pass  # 숫자 변환 실패 - 다음 방법 시도
        
        # 3. 텍스트 중 숫자 찾기 (마지막 시도)
        number_matches = re.findall(r"([0-9](?:\.[0-9]+)?)", response_str)
        for match in number_matches:
            try:
                score = float(match)
                if 0 <= score <= 1:
                    reasoning = f"Score extracted from text. Original response: {response_str[:200]}..."
                    return score, reasoning
            except ValueError:
                continue
        
        # 파싱 실패 - 기본값 반환
        logger.warning(f"Could not parse score from evaluation response. Using default score {self.default_score}. Response: {response_str[:100]}...")
        return self.default_score, f"Score parsing failed. Original response: {response_str[:200]}..."

    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        with tracer.start_as_current_span(
            "graph.node.state_evaluator",
            attributes={
                "node_id": self.node_id,
                "task_id": state.task_id,
                "thoughts_to_eval": len(state.current_thoughts_to_evaluate),
            },
        ):

            logger.info(f"StateEvaluatorNode '{self.node_id}' execution started. Task ID: {state.task_id}")
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' (State Evaluator) started.", current_node=self.node_id)
            )

            updated_thoughts_map = {t.id: t for t in state.thoughts}
            error_message: Optional[str] = None
            any_evaluation_done = False
            evaluations_summary_for_log = []

            if not state.current_thoughts_to_evaluate:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): No thoughts to evaluate.")
                await self.notification_service.broadcast_to_task(
                    state.task_id,
                    StatusUpdateMessage(task_id=state.task_id, status="node_completed", detail=f"Node '{self.node_id}' finished: No thoughts to evaluate.", current_node=self.node_id, next_node="search_strategy" if state.thoughts else None)
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

                    # 개선된 파싱 로직 사용
                    score, reasoning = self._parse_evaluation_response(eval_response_str)
                    
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

                    # 각 생각 평가 완료 알림
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        IntermediateResultMessage(
                            task_id=state.task_id, node_id=self.node_id,
                            result_step_name="thought_evaluated",
                            data={"thought_id": thought_id_to_eval, "score": score, "reasoning": reasoning[:150]+"..."}
                        )
                    )

                except Exception as e:
                    logger.error(f"Node '{self.node_id}' (Task: {state.task_id}): Error evaluating thought '{thought_id_to_eval}': {e}", exc_info=True)
                    
                    # 실패 처리 - 기본 점수 설정, 실패로 표시하지 않고 평가 완료로 처리
                    updated_metadata = thought_to_eval.metadata.copy() if thought_to_eval.metadata else {}
                    updated_metadata['eval_error'] = str(e)
                    updated_metadata['eval_reasoning'] = f"Evaluation failed, using default score {self.default_score}. Error: {str(e)}"
                    
                    # 오류가 있어도 평가 완료로 처리하고 기본 점수 사용
                    fallback_thought_version = Thought(
                        id=thought_to_eval.id, parent_id=thought_to_eval.parent_id, content=thought_to_eval.content,
                        evaluation_score=self.default_score, status="evaluated", metadata=updated_metadata
                    )
                    updated_thoughts_map[thought_id_to_eval] = fallback_thought_version
                    any_evaluation_done = True
                    evaluations_summary_for_log.append(f"Thought '{thought_id_to_eval}': Score={self.default_score:.2f} (fallback)")
                    
                    error_message = (error_message or "") + f"Error evaluating {thought_id_to_eval}: {str(e)}; "
                    await self.notification_service.broadcast_to_task(
                        state.task_id,
                        IntermediateResultMessage(
                            task_id=state.task_id, node_id=self.node_id,
                            result_step_name="thought_evaluation_fallback",
                            data={"thought_id": thought_id_to_eval, "fallback_score": self.default_score, "error": str(e)}
                        )
                    )

            final_thoughts_list = list(updated_thoughts_map.values())
            log_summary_str = ', '.join(evaluations_summary_for_log) if evaluations_summary_for_log else 'No new evaluations.'
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Evaluations completed. Summary: {log_summary_str}")

            update_payload = {
                "thoughts": final_thoughts_list,
                "current_thoughts_to_evaluate": []
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
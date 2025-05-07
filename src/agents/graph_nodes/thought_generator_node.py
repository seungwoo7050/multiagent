# multiagent/src/agents/graph_nodes/thought_generator_node.py

import re
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from src.config.logger import get_logger
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState, Thought

logger = get_logger(__name__)

class ThoughtGeneratorNode:
    """
    주어진 상태를 바탕으로 여러 개의 다음 생각(추론 경로)을 생성하는 ToT 노드.
    """
    def __init__(
        self,
        llm_client: LLMClient,
        num_thoughts: int = 3,  # 한 번에 생성할 생각의 수
        max_tokens_per_thought: int = 200,
        temperature: float = 0.7,
        prompt_template_path: Optional[str] = None,  # 특정 프롬프트 사용 시
        model_name: Optional[str] = None,
        node_id: str = "thought_generator"
    ):
        self.llm_client = llm_client
        self.num_thoughts = num_thoughts
        self.max_tokens_per_thought = max_tokens_per_thought
        self.temperature = temperature
        self.prompt_template_path = prompt_template_path
        self.model_name = model_name
        self.node_id = node_id
        logger.info(f"ThoughtGeneratorNode '{self.node_id}' initialized. Num thoughts: {self.num_thoughts}")

    def _construct_prompt(self, state: AgentGraphState) -> str:
        # 이전 생각들을 요약하거나, 가장 유망했던 몇 개만 선택하여 컨텍스트로 제공
        relevant_history = "No prior thoughts relevant to current expansion."
        parent_thought_content = "N/A (Initial thought generation)"

        # current_best_thought_id를 확장할 부모 생각으로 간주
        if state.current_best_thought_id:
            parent_thought = state.get_thought_by_id(state.current_best_thought_id)
            if parent_thought:
                parent_thought_content = parent_thought.content

        prompt = f"""
        You are a creative and methodical problem solver. Your task is to generate a set of diverse and promising next steps (thoughts) to achieve a given goal, based on the current context.

        Goal: {state.original_input}

        Current Context / Parent Thought:
        {parent_thought_content}

        Previous relevant thoughts and outcomes (if any):
        {relevant_history} 
        {( 'Current error (if any, try to overcome this): ' + state.error_message) if state.error_message else ''}

        Instructions:
        1. Carefully analyze the Goal and the Current Context/Parent Thought.
        2. Generate exactly {self.num_thoughts} distinct, actionable, and forward-looking thoughts.
        3. Each thought should be a potential step, hypothesis, or approach to explore next.
        4. Avoid vague or overly broad thoughts. Be specific.
        5. Do not repeat thoughts that have already been evaluated negatively or led to dead ends, unless you have a new angle.
        6. Consider different perspectives or strategies for the next steps.

        Output each thought on a new line, prefixed with "Thought: ".

        Example Output:
        Thought: Research alternative_method_X for solving sub_problem_Y.
        Thought: Verify assumption_Z by querying_tool_A.
        Thought: Break down complex_task_Q into smaller manageable parts.

        Generated Thoughts:
        """
        return prompt

    @staticmethod
    def _extract_thoughts(raw: str) -> List[str]:
        """
        LLM 응답 문자열에서 실제 '생각' 부분만 추출합니다.
        허용 포맷:
          Thought: some text
          - some text
          1) some text
        """
        pattern = re.compile(
            r"""
            ^\s*                             # 앞 공백
            (?:thought\s*:|[-\d.)]+)?\s*   # Thought: 또는 불릿/번호
            (?P<content>.+?)                 # 실제 텍스트
            \s*$                            # 뒤 공백
            """,
            re.IGNORECASE | re.VERBOSE,
        )
        thoughts: List[str] = []
        for line in raw.splitlines():
            m = pattern.match(line)
            if m:
                content = m.group("content").strip()
                if content:
                    thoughts.append(content)
        return thoughts

    async def __call__(
        self,
        state: AgentGraphState,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        logger.info(f"ThoughtGeneratorNode '{self.node_id}' execution started for task: {state.task_id}")

        error_message: Optional[str] = None

        if state.search_depth >= state.max_search_depth:
            logger.info(
                f"Node '{self.node_id}': Max search depth ({state.max_search_depth}) reached. No new thoughts generated."
            )
            return {
                "current_thoughts_to_evaluate": [],
                "error_message": "Max search depth reached."
            }

        try:
            generation_prompt = self._construct_prompt(state)
            logger.debug(f"Node '{self.node_id}': Generation prompt:\n{generation_prompt}")

            full_response = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": generation_prompt}],
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens_per_thought * self.num_thoughts
            )

            generated_thought_contents = ThoughtGeneratorNode._extract_thoughts(full_response)
            generated_thought_contents = generated_thought_contents[: self.num_thoughts]

            if not generated_thought_contents:
                logger.warning(f"Node '{self.node_id}': LLM did not generate any valid thoughts.")
                error_message = f"LLM did not generate thoughts in node '{self.node_id}'."

        except Exception as e:
            logger.error(f"Node '{self.node_id}': Error during thought generation: {e}", exc_info=True)
            error_message = f"Error in ThoughtGeneratorNode '{self.node_id}': {e}"
            generated_thought_contents = []

        newly_added_thought_ids: List[str] = []
        for content in generated_thought_contents:
            new_thought = state.add_thought(content=content, parent_id=state.current_best_thought_id)
            newly_added_thought_ids.append(new_thought.id)
        if newly_added_thought_ids:
            logger.info(f"Node '{self.node_id}': Generated {len(newly_added_thought_ids)} new thoughts.")

        return {
            "thoughts": state.thoughts,
            "current_thoughts_to_evaluate": newly_added_thought_ids,
            "last_llm_output": generated_thought_contents if generated_thought_contents else error_message,
            "error_message": error_message
        }

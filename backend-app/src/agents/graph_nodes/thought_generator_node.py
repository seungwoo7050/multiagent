import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.services.llm_client import LLMClient
from src.schemas.mcp_models import AgentGraphState
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

logger = get_logger(__name__)
settings = get_settings()


class ThoughtGeneratorNode:
  def __init__(
    self,
    llm_client: LLMClient,
    notification_service: NotificationService,
    num_thoughts: int = 3,
    max_tokens_per_thought: int = 500,
    temperature: float = 0.7,
    prompt_template_path: Optional[str] = None,
    model_name: Optional[str] = None,
    node_id: str = "thought_generator",
  ):
    self.llm_client = llm_client
    self.notification_service = notification_service
    self.num_thoughts = num_thoughts
    self.max_tokens_per_thought = max_tokens_per_thought
    self.temperature = temperature
    self.prompt_template_path = prompt_template_path
    self.model_name = model_name
    self.node_id = node_id
    self.prompt_template_str = self._load_prompt_template_if_path_exists()
    logger.info(
      f"ThoughtGeneratorNode '{self.node_id}' initialized. Num thoughts: {self.num_thoughts}. "
      f"NotificationService injected: {'Yes' if notification_service else 'No'}. "
      f"Prompt: '{self.prompt_template_path if self.prompt_template_path else 'Default internal'}'"
    )

  def _load_prompt_template_if_path_exists(self) -> Optional[str]:
    if not self.prompt_template_path:
      return None
    base_prompt_dir = getattr(settings, "PROMPT_TEMPLATE_DIR", "config/prompts")
    if os.path.isabs(self.prompt_template_path):
      full_path = self.prompt_template_path
    else:
      full_path = os.path.join(base_prompt_dir, self.prompt_template_path)
    try:
      with open(full_path, "r", encoding="utf-8") as f:
        logger.debug(
          f"Successfully loaded prompt template from: {full_path} for node '{self.node_id}'"
        )
        return f.read()
    except FileNotFoundError:
      logger.warning(
        f"Prompt template file not found for ThoughtGeneratorNode '{self.node_id}': {full_path}. Using default internal prompt."
      )
      return None
    except Exception as e:
      logger.error(
        f"Error loading prompt template from {full_path} for node '{self.node_id}': {e}. Using default internal prompt."
      )
      return None

  def _construct_prompt(self, state: AgentGraphState) -> str:
    parent_thought_content = "N/A (Initial thought generation)"
    if state.current_best_thought_id:
      parent_thought = state.get_thought_by_id(state.current_best_thought_id)
      if parent_thought:
        parent_thought_content = parent_thought.content

    sibling_thoughts_summary_parts = []
    if (
      parent_thought_content != "N/A (Initial thought generation)"
      and state.current_best_thought_id
    ):
      for t in state.thoughts:
        if (
          t.parent_id == state.current_best_thought_id
          and t.status == "evaluated"
          and t.evaluation_score is not None
        ):
          sibling_thoughts_summary_parts.append(
            f"Sibling Thought (ID: {t.id}, Score: {t.evaluation_score:.2f}): {t.content[:100]}..."
          )
        elif (
          t.parent_id == state.current_best_thought_id
          and t.status == "evaluation_failed"
        ):
          sibling_thoughts_summary_parts.append(
            f"Sibling Thought (ID: {t.id}, Status: evaluation_failed): {t.content[:100]}..."
          )

    sibling_thoughts_summary = (
      "\n".join(sibling_thoughts_summary_parts)
      if sibling_thoughts_summary_parts
      else "No previously explored sibling thoughts from this parent."
    )

    if self.prompt_template_str:
      from langchain_core.prompts import PromptTemplate

      try:
        prompt_data = {
          "original_input": state.original_input,
          "parent_thought_content": parent_thought_content,
          "sibling_thoughts_summary": sibling_thoughts_summary,
          "num_thoughts": self.num_thoughts,
          "search_depth": state.search_depth,
          "max_search_depth": state.max_search_depth,
          "error_message": (
            f"Current Error (if any, try to overcome or sidestep this): {state.error_message}"
            if state.error_message
            else ""
          ),
        }

        template = PromptTemplate(
          template=self.prompt_template_str,
          input_variables=list(prompt_data.keys()),
        )
        return template.format(**prompt_data)
      except KeyError as ke:
        logger.error(
          f"Missing key for prompt template in ThoughtGeneratorNode '{self.node_id}': {ke}. Falling back to default internal prompt."
        )

      except Exception as e:
        logger.error(
          f"Error formatting prompt template in ThoughtGeneratorNode '{self.node_id}': {e}. Falling back to default internal prompt."
        )

    return f"""
    You are a creative and methodical problem solver. Your task is to generate a set of diverse and promising next steps (thoughts) to achieve a given goal, based on the current context.

    Overall Goal: {state.original_input}

    Current Context / Parent Thought for Expansion:
    {parent_thought_content}

    Previously Explored Sibling Thoughts and Their Outcomes (if any, for context):
    {sibling_thoughts_summary}

    Current Search Depth: {state.search_depth} / {state.max_search_depth}
    {("Current Error (if any, try to generate thoughts to overcome or sidestep this): " + state.error_message) if state.error_message else ""}

    Instructions:
    1. Analyze the "Overall Goal" and the "Current Context/Parent Thought".
    2. Generate exactly {self.num_thoughts} distinct, actionable, and forward-looking thoughts.
    3. Each thought should represent a potential next step, a hypothesis to test, a question to answer, or a sub-problem to solve.
    4. Thoughts should be diverse, exploring different angles or approaches if possible.
    5. Avoid thoughts that are too vague, too broad, or simply restate the current problem. Be specific and constructive.
    6. If "Previously Explored Sibling Thoughts" are provided, try not to generate highly similar thoughts unless you have a significantly new angle or refinement.
    7. Ensure thoughts are concise and clearly phrased.

    Output Format:
    Provide each thought on a new line, prefixed with "Thought: ".
    Example (if Overall Goal is "Plan a 3-day trip to Paris" and Parent Thought is "Day 1: Focus on iconic landmarks"):
    Thought: Research opening hours and ticket prices for the Eiffel Tower and Louvre Museum.
    Thought: Plan a walking route connecting the Eiffel Tower, Arc de Triomphe, and Champs-Élysées.
    Thought: Identify potential lunch spots near the Louvre with good reviews.

    Begin Generating Thoughts:
    """

  @staticmethod
  def _extract_thoughts(raw: str) -> List[str]:
    pattern = re.compile(
      r"""
      ^\s* # 앞 공백
      (?:thought\s*:|[-\d.)]+)?\s* # Thought: 또는 불릿/번호
      (?P<content>.+?)         # 실제 텍스트
      \s*$              # 뒤 공백
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
    self, state: AgentGraphState, config: Optional[RunnableConfig] = None
  ) -> Dict[str, Any]:
    with tracer.start_as_current_span(
      "graph.node.thought_generator",
      attributes={
        "node_id": self.node_id,
        "task_id": state.task_id,
        "num_thoughts": self.num_thoughts,
        "search_depth": state.search_depth,
      },
    ):
      logger.info(
        f"ThoughtGeneratorNode '{self.node_id}' execution started. Task ID: {state.task_id}"
      )
      await self.notification_service.broadcast_to_task(
        state.task_id,
        StatusUpdateMessage(
          task_id=state.task_id,
          status="node_executing",
          detail=f"Node '{self.node_id}' (Thought Generator) started.",
          current_node=self.node_id,
        ),
      )

      error_message: Optional[str] = None
      generated_thought_contents: List[str] = []

      if state.search_depth >= state.max_search_depth:
        logger.info(
          f"Node '{self.node_id}': Max search depth ({state.max_search_depth}) reached. No new thoughts generated for task {state.task_id}."
        )

        await self.notification_service.broadcast_to_task(
          state.task_id,
          StatusUpdateMessage(
            task_id=state.task_id,
            status="node_completed",
            detail=f"Node '{self.node_id}' finished: Max search depth reached.",
            current_node=self.node_id,
          ),
        )
        return {
          "current_thoughts_to_evaluate": [],
          "error_message": "Max search depth reached.",
        }

      try:
        generation_prompt = self._construct_prompt(state)
        logger.debug(
          f"Node '{self.node_id}' (Task: {state.task_id}): Generation prompt constructed."
        )

        full_response = await self.llm_client.generate_response(
          messages=[{"role": "user", "content": generation_prompt}],
          model_name=self.model_name,
          temperature=self.temperature,
          max_tokens=self.max_tokens_per_thought * self.num_thoughts,
        )
        logger.debug(
          f"Node '{self.node_id}' (Task: {state.task_id}): LLM response received."
        )

        generated_thought_contents = ThoughtGeneratorNode._extract_thoughts(
          full_response
        )
        generated_thought_contents = generated_thought_contents[
          : self.num_thoughts
        ]

        if not generated_thought_contents:
          logger.warning(
            f"Node '{self.node_id}' (Task: {state.task_id}): LLM did not generate any valid thoughts from response: {full_response[:200]}..."
          )
          error_message = (
            f"LLM did not generate thoughts in node '{self.node_id}'."
          )
        else:
          await self.notification_service.broadcast_to_task(
            state.task_id,
            IntermediateResultMessage(
              task_id=state.task_id,
              node_id=self.node_id,
              result_step_name="thoughts_generated",
              data={
                "generated_count": len(generated_thought_contents),
                "thoughts_preview": [
                  t[:100] + "..." for t in generated_thought_contents
                ],
              },
            ),
          )

      except Exception as e:
        logger.error(
          f"Node '{self.node_id}' (Task: {state.task_id}): Error during thought generation: {e}",
          exc_info=True,
        )
        error_message = f"Error in ThoughtGeneratorNode '{self.node_id}': {e}"
        generated_thought_contents = []

      newly_added_thought_ids: List[str] = []
      if generated_thought_contents:
        for content in generated_thought_contents:
          new_thought = state.add_thought(
            content=content, parent_id=state.current_best_thought_id
          )
          newly_added_thought_ids.append(new_thought.id)
        if newly_added_thought_ids:
          logger.info(
            f"Node '{self.node_id}' (Task: {state.task_id}): Generated and added {len(newly_added_thought_ids)} new thoughts to state."
          )

      await self.notification_service.broadcast_to_task(
        state.task_id,
        StatusUpdateMessage(
          task_id=state.task_id,
          status="node_completed",
          detail=f"Node '{self.node_id}' (Thought Generator) finished. Generated {len(newly_added_thought_ids)} thoughts.",
          current_node=self.node_id,
          next_node="state_evaluator",
        ),
      )

      return {
        "thoughts": state.thoughts,
        "current_thoughts_to_evaluate": newly_added_thought_ids,
        "last_llm_output": generated_thought_contents
        if generated_thought_contents
        else (error_message or "No thoughts generated."),
        "error_message": error_message,
      }

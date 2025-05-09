# src/agents/graph_nodes/search_strategy_node.py
import os # os 임포트 추가 (향후 사용 가능성)
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from src.config.logger import get_logger
from src.config.settings import get_settings # settings 임포트 추가
from src.schemas.mcp_models import AgentGraphState, Thought
from src.services.notification_service import NotificationService # 추가
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage # 추가

logger = get_logger(__name__)
settings = get_settings() # settings 인스턴스

class SearchStrategyNode:
    def __init__(
        self,
        notification_service: NotificationService, # <--- 추가
        beam_width: int = 1,
        score_threshold_to_finish: float = 0.95,
        min_score_to_continue: float = 0.1,
        node_id: str = "search_strategy"
    ):
        if beam_width < 1:
            raise ValueError("Beam width must be at least 1.")
        self.notification_service = notification_service # <--- 저장
        self.beam_width = beam_width
        self.score_threshold_to_finish = score_threshold_to_finish
        self.min_score_to_continue = min_score_to_continue
        self.node_id = node_id
        logger.info(
            f"SearchStrategyNode '{self.node_id}' initialized. "
            f"Beam width: {self.beam_width}, Finish Threshold: {self.score_threshold_to_finish}, "
            f"Min Continue Score: {self.min_score_to_continue}. "
            f"NotificationService injected: {'Yes' if notification_service else 'No'}"
        )

    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        logger.info(f"SearchStrategyNode '{self.node_id}' execution started. Task ID: {state.task_id}, Depth: {state.search_depth}")
        await self.notification_service.broadcast_to_task(
            state.task_id,
            StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' (Search Strategy) started.", current_node=self.node_id)
        )

        # ... (기존 필터링 및 정렬 로직) ...
        evaluated_thoughts: List[Thought] = sorted(
            [t for t in state.thoughts if t.status == "evaluated" and t.evaluation_score is not None],
            key=lambda t: t.evaluation_score or -1.0, # None 점수는 최하위로
            reverse=True
        )

        if not evaluated_thoughts:
            logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): No evaluated thoughts with scores found.")
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_error", detail=f"Node '{self.node_id}': No evaluated thoughts to process.", current_node=self.node_id)
            )
            # 이전 상태의 best thought (있다면) 또는 기본 에러 메시지를 final_answer로 설정하고 종료
            final_answer_content = "Could not determine a course of action due to lack of evaluated thoughts."
            if state.current_best_thought_id:
                 best_t = state.get_thought_by_id(state.current_best_thought_id)
                 if best_t: final_answer_content = best_t.content
            return {
                "error_message": "No new thoughts were evaluated or no valid scores found.",
                "final_answer": final_answer_content
            }

        top_thoughts_for_expansion = evaluated_thoughts[:self.beam_width]
        current_round_best_thought = top_thoughts_for_expansion[0]
        new_global_best_thought_id = state.current_best_thought_id # 기본값
        global_best_thought = state.get_thought_by_id(state.current_best_thought_id) if state.current_best_thought_id else None

        if global_best_thought is None or \
           (current_round_best_thought.evaluation_score is not None and \
            (global_best_thought.evaluation_score is None or \
             current_round_best_thought.evaluation_score > global_best_thought.evaluation_score)):
            new_global_best_thought_id = current_round_best_thought.id
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): New global best thought ID: {new_global_best_thought_id} (Score: {current_round_best_thought.evaluation_score})")
        elif global_best_thought:
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Global best thought remains: {global_best_thought.id} (Score: {global_best_thought.evaluation_score})")


        next_search_depth = state.search_depth + 1
        final_answer_content: Optional[str] = None
        strategy_decision: str = "continue_search" # 기본 결정
        next_node_for_ws = "thought_generator" # 기본 다음 노드 (계속 탐색 시)

        # --- 종료 조건 판단 ---
        if next_search_depth >= state.max_search_depth:
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Max search depth ({state.max_search_depth}) reached.")
            final_thought_to_use = state.get_thought_by_id(new_global_best_thought_id) if new_global_best_thought_id else current_round_best_thought
            final_answer_content = final_thought_to_use.content if final_thought_to_use else "Reached max depth, no definitive answer."
            strategy_decision = "finish_max_depth"
            next_node_for_ws = None # 종료
            update_payload = {
                "thoughts": state.thoughts, "current_best_thought_id": new_global_best_thought_id,
                "search_depth": next_search_depth, "final_answer": final_answer_content,
                "error_message": "Max search depth reached."
            }

        elif current_round_best_thought.evaluation_score is not None and \
             current_round_best_thought.evaluation_score >= self.score_threshold_to_finish:
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): High-confidence thought found (Score: {current_round_best_thought.evaluation_score}). Finalizing.")
            final_answer_content = current_round_best_thought.content
            strategy_decision = "finish_high_score"
            next_node_for_ws = None # 종료
            update_payload = {
                "thoughts": state.thoughts, "current_best_thought_id": current_round_best_thought.id,
                "search_depth": state.search_depth, "final_answer": final_answer_content,
                "error_message": None
            }

        elif all(t.evaluation_score is not None and t.evaluation_score < self.min_score_to_continue for t in top_thoughts_for_expansion):
            logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): All top {self.beam_width} thoughts below threshold. Stopping.")
            final_thought_to_use = state.get_thought_by_id(new_global_best_thought_id) if new_global_best_thought_id else current_round_best_thought
            final_answer_content = final_thought_to_use.content if final_thought_to_use else "Exploration stopped due to low scores."
            strategy_decision = "finish_low_score"
            next_node_for_ws = None # 종료
            update_payload = {
                "thoughts": state.thoughts, "current_best_thought_id": new_global_best_thought_id,
                "search_depth": next_search_depth, "final_answer": final_answer_content,
                "error_message": "Exploration stopped, all current paths unpromising."
            }
        else: # 계속 탐색
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Proceeding to next search depth ({next_search_depth}). Best thought to expand based on: {new_global_best_thought_id}")
            strategy_decision = "continue_search"
            update_payload = {
                "thoughts": state.thoughts, "current_best_thought_id": new_global_best_thought_id,
                "search_depth": next_search_depth, "error_message": None
            }

        # 결정된 전략 알림
        await self.notification_service.broadcast_to_task(
            state.task_id,
            IntermediateResultMessage(
                task_id=state.task_id, node_id=self.node_id,
                result_step_name="search_strategy_decision",
                data={
                    "decision": strategy_decision,
                    "current_best_thought_id": update_payload.get("current_best_thought_id"),
                    "current_best_score": state.get_thought_by_id(update_payload.get("current_best_thought_id")).evaluation_score if update_payload.get("current_best_thought_id") and state.get_thought_by_id(update_payload.get("current_best_thought_id")) else None,
                    "next_depth": update_payload.get("search_depth"),
                    "final_answer_preview": (final_answer_content[:100]+"..." if final_answer_content else None)
                }
            )
        )

        await self.notification_service.broadcast_to_task(
            state.task_id,
            StatusUpdateMessage(
                task_id=state.task_id, status="node_completed",
                detail=f"Node '{self.node_id}' (Search Strategy) finished. Decision: {strategy_decision}.",
                current_node=self.node_id, next_node=next_node_for_ws
            )
        )
        return update_payload
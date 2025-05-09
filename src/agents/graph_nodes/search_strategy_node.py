# src/agents/graph_nodes/search_strategy_node.py
import os
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from src.utils.logger import get_logger
from src.config.settings import get_settings
from src.schemas.mcp_models import AgentGraphState, Thought
from src.services.notification_service import NotificationService
from src.schemas.websocket_models import StatusUpdateMessage, IntermediateResultMessage
from opentelemetry import trace
tracer = trace.get_tracer(__name__)
logger = get_logger(__name__)
settings = get_settings()

class SearchStrategyNode:
    def __init__(
        self,
        notification_service: NotificationService,
        beam_width: int = 1,
        score_threshold_to_finish: float = 0.95,
        min_score_to_continue: float = 0.1,
        node_id: str = "search_strategy"
    ):
        if beam_width < 1:
            raise ValueError("Beam width must be at least 1.")
        self.notification_service = notification_service
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
        with tracer.start_as_current_span(
            "graph.node.search_strategy",
            attributes={
                "node_id": self.node_id,
                "task_id": state.task_id,
                "beam_width": self.beam_width,
                "search_depth": state.search_depth,
            },
        ):
            logger.info(f"SearchStrategyNode '{self.node_id}' execution started. Task ID: {state.task_id}, Depth: {state.search_depth}")
            await self.notification_service.broadcast_to_task(
                state.task_id,
                StatusUpdateMessage(task_id=state.task_id, status="node_executing", detail=f"Node '{self.node_id}' (Search Strategy) started.", current_node=self.node_id)
            )

            # 평가된 생각들을 점수 기준으로 정렬
            evaluated_thoughts: List[Thought] = sorted(
                [t for t in state.thoughts if t.status == "evaluated" and t.evaluation_score is not None],
                key=lambda t: t.evaluation_score or -1.0,
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
                    "final_answer": final_answer_content,
                    "error_message": "No new thoughts were evaluated or no valid scores found."
                }

            top_thoughts_for_expansion = evaluated_thoughts[:self.beam_width]
            current_round_best_thought = top_thoughts_for_expansion[0]
            new_global_best_thought_id = state.current_best_thought_id
            global_best_thought = state.get_thought_by_id(state.current_best_thought_id) if state.current_best_thought_id else None

            if global_best_thought is None or \
            (current_round_best_thought.evaluation_score is not None and \
                (global_best_thought.evaluation_score is None or \
                current_round_best_thought.evaluation_score > global_best_thought.evaluation_score)):
                new_global_best_thought_id = current_round_best_thought.id
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): New global best thought ID: {new_global_best_thought_id} (Score: {current_round_best_thought.evaluation_score})")
            elif global_best_thought:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Global best thought remains: {global_best_thought.id} (Score: {global_best_thought.evaluation_score})")

            # 종료 결정을 위한 플래그
            should_terminate = False
            next_search_depth = state.search_depth + 1
            final_answer_content = None
            strategy_decision = "continue_search"
            next_node_for_ws = "thought_generator"

            # --- 종료 조건 판단 (수정됨) ---
            # 1. 먼저 LangGraph의 recursion_limit 확인 - 이 값을 상태에 가져옴
            recursion_limit = getattr(state, 'recursion_limit', 15)
            
            # 2. 현재의 search_depth가 recursion_limit보다 크거나 같으면 강제 종료
            if next_search_depth >= recursion_limit:
                logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): Recursion limit ({recursion_limit}) nearly reached. Forcing termination.")
                should_terminate = True
                best_thought = state.get_thought_by_id(new_global_best_thought_id) if new_global_best_thought_id else current_round_best_thought
                final_answer_content = best_thought.content if best_thought else "Search stopped due to recursion limit."
                strategy_decision = "finish_recursion_limit"
                next_node_for_ws = None
            
            # 3. 사용자 지정 max_search_depth 확인 (일반적으로 recursion_limit보다 작음)
            elif next_search_depth >= state.max_search_depth:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Max search depth ({state.max_search_depth}) reached.")
                final_thought_to_use = state.get_thought_by_id(new_global_best_thought_id) if new_global_best_thought_id else current_round_best_thought
                final_answer_content = final_thought_to_use.content if final_thought_to_use else "Reached max depth, no definitive answer."
                strategy_decision = "finish_max_depth"
                next_node_for_ws = None
                should_terminate = True
            
            # 4. 높은 점수 찾음 - 대부분의 성공 케이스에서는 이걸로 종료
            elif current_round_best_thought.evaluation_score is not None and \
                current_round_best_thought.evaluation_score >= self.score_threshold_to_finish:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): High-confidence thought found (Score: {current_round_best_thought.evaluation_score}). Finalizing.")
                final_answer_content = current_round_best_thought.content
                strategy_decision = "finish_high_score"
                next_node_for_ws = None
                should_terminate = True

            # 5. 점수가 낮아서 더 이상 탐색할 가치가 없는 경우
            elif all(t.evaluation_score is not None and t.evaluation_score < self.min_score_to_continue for t in top_thoughts_for_expansion):
                logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): All top {self.beam_width} thoughts below threshold. Stopping.")
                final_thought_to_use = state.get_thought_by_id(new_global_best_thought_id) if new_global_best_thought_id else current_round_best_thought
                final_answer_content = final_thought_to_use.content if final_thought_to_use else "Exploration stopped due to low scores."
                strategy_decision = "finish_low_score"
                next_node_for_ws = None
                should_terminate = True
            
            else: # 계속 탐색
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Proceeding to next search depth ({next_search_depth}). Best thought to expand based on: {new_global_best_thought_id}")
                strategy_decision = "continue_search"

            # 상태 업데이트 생성 (단순화 및 확실한 종료 조건)
            update_payload = {
                "thoughts": state.thoughts,
                "current_best_thought_id": new_global_best_thought_id,
                "search_depth": next_search_depth
            }

            # 핵심 수정: 종료 시 final_answer를 최우선 값으로 설정하고 다른 모든 필드를 secondary로 처리
            if should_terminate:
                # LangGraph edge condition을 확실히 트리거하기 위해 최종 상태를 매우 간결하게 유지
                return {
                    "final_answer": final_answer_content
                }
            else:
                # 계속할 때는 final_answer를 명시적으로 None으로 설정
                update_payload["final_answer"] = None

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
                        "final_answer_preview": (final_answer_content[:100]+"..." if final_answer_content else None),
                        "is_terminal": should_terminate
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
            
            # 디버깅 로그 추가
            logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Returning update with final_answer: {final_answer_content is not None}")
            return update_payload
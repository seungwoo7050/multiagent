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
        beam_width: int = 2,
        score_threshold_to_finish: float = 0.7,
        min_score_to_continue: float = 0.1,
        node_id: str = "search_strategy",
        min_depth_before_finish: int = 1  # 최소 1회는 탐색하도록 추가
    ):
        if beam_width < 1:
            raise ValueError("Beam width must be at least 1.")
        self.notification_service = notification_service
        self.beam_width = beam_width
        self.score_threshold_to_finish = score_threshold_to_finish
        self.min_score_to_continue = min_score_to_continue
        self.node_id = node_id
        self.min_depth_before_finish = min_depth_before_finish  # 최소 탐색 깊이 추가
        logger.info(
            f"SearchStrategyNode '{self.node_id}' initialized. "
            f"Beam width: {self.beam_width}, Finish Threshold: {self.score_threshold_to_finish}, "
            f"Min Continue Score: {self.min_score_to_continue}, Min Depth: {self.min_depth_before_finish}. "
            f"NotificationService injected: {'Yes' if notification_service else 'No'}"
        )

    def _select_best_thought(self, thoughts: List[Thought]) -> Optional[Thought]:
        """모든 생각들 중에서 가장 높은 점수를 가진 생각 반환"""
        if not thoughts:
            return None
            
        scored_thoughts = [t for t in thoughts if t.status == "evaluated" and t.evaluation_score is not None]
        if not scored_thoughts:
            return None
            
        return max(scored_thoughts, key=lambda t: t.evaluation_score or 0.0)

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
                
                # 이전 상태의 best thought (있다면) 또는 기본 메시지를 final_answer로 설정
                final_answer_content = "No evaluated thoughts were found to solve this problem."
                best_across_all = self._select_best_thought(state.thoughts)
                if best_across_all:
                    final_answer_content = best_across_all.content
                elif state.original_input:
                    final_answer_content = f"Regarding '{state.original_input}', I couldn't develop a complete solution. Please provide more details or try a different approach."
                
                return {
                    "final_answer": final_answer_content,
                    "error_message": "No thoughts were evaluated successfully.",
                    "current_best_thought_id": state.current_best_thought_id,
                    "next_action": "finish"
                }

            # 현재 라운드의 top thoughts
            top_thoughts_for_expansion = evaluated_thoughts[:self.beam_width]
            current_round_best_thought = top_thoughts_for_expansion[0]
            
            # 전체 생각들 중 최고 점수 생각 (탐색 기록 전체에서)
            best_across_all = self._select_best_thought(state.thoughts)
            new_global_best_thought_id = best_across_all.id if best_across_all else state.current_best_thought_id
            
            # 이전 최고 점수 생각이 있다면 가져옴
            global_best_thought = state.get_thought_by_id(state.current_best_thought_id) if state.current_best_thought_id else None
            
            # 새로운 글로벌 베스트 선택 로직 개선
            if global_best_thought is None or \
                (best_across_all and best_across_all.evaluation_score is not None and \
                (global_best_thought.evaluation_score is None or \
                best_across_all.evaluation_score > global_best_thought.evaluation_score)):
                new_global_best_thought_id = best_across_all.id
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): New global best thought ID: {new_global_best_thought_id} (Score: {best_across_all.evaluation_score})")
            elif global_best_thought:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Global best thought remains: {global_best_thought.id} (Score: {global_best_thought.evaluation_score})")

            # 종료 결정을 위한 플래그
            should_terminate = False
            next_search_depth = state.search_depth + 1
            final_answer_content = None
            strategy_decision = "continue_search"
            next_node_for_ws = "thought_generator"
            next_action = "continue"

            # --- 종료 조건 판단 (수정됨) ---
            recursion_limit = getattr(state, 'recursion_limit', 15)
            
            if isinstance(state.dynamic_data, dict) and 'recursion_limit' in state.dynamic_data:
                recursion_limit = state.dynamic_data['recursion_limit']
            elif isinstance(state.metadata, dict) and 'recursion_limit' in state.metadata:
                recursion_limit = state.metadata['recursion_limit']
            
            # [수정] 매우 높은 점수에 대한 즉각 종료 처리 - 최소 탐색 깊이 조건보다 우선 적용
            if current_round_best_thought.evaluation_score is not None and current_round_best_thought.evaluation_score >= 0.85:
                logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): Very high score ({current_round_best_thought.evaluation_score}) detected. Forcing termination despite depth.")
                should_terminate = True
                final_answer_content = current_round_best_thought.content
                strategy_decision = "finish_very_high_score"
                next_node_for_ws = None
                next_action = "finish"
            
            # 점수 개선 없음 추적 로직
            no_improvement_count = 0
            if not should_terminate and isinstance(state.dynamic_data, dict):
                no_improvement_count = state.dynamic_data.get('no_improvement_count', 0)
                prev_best_score = state.dynamic_data.get('prev_best_score', 0)
                
                if best_across_all and best_across_all.evaluation_score is not None:
                    if best_across_all.evaluation_score <= prev_best_score:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0  # 점수 개선 시 카운터 리셋
                        state.dynamic_data['prev_best_score'] = best_across_all.evaluation_score
                state.dynamic_data['no_improvement_count'] = no_improvement_count

            # 정체 카운터가 3 이상이면 강제 종료 (점수 기반 종료 조건)
            if not should_terminate and no_improvement_count >= 3 and state.search_depth >= 2:
                logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): No score improvement for 3 iterations. Forcing termination.")
                should_terminate = True
                final_thought_to_use = best_across_all or current_round_best_thought
                final_answer_content = final_thought_to_use.content if final_thought_to_use else "Search stopped due to no progress."
                strategy_decision = "finish_no_improvement"
                next_node_for_ws = None
                next_action = "finish"
            
            # 1. 최소 탐색 깊이 체크 - 너무 일찍 종료되지 않도록
            if not should_terminate and state.search_depth < self.min_depth_before_finish:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Continuing search as minimum depth {self.min_depth_before_finish} not yet reached.")
                strategy_decision = "continue_min_depth"
                next_action = "continue"
            
            # 2. 반복 제한 확인
            elif not should_terminate and next_search_depth >= recursion_limit:
                logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): Recursion limit ({recursion_limit}) nearly reached. Forcing termination.")
                should_terminate = True
                final_thought_to_use = best_across_all or current_round_best_thought
                final_answer_content = final_thought_to_use.content if final_thought_to_use else "Search stopped due to recursion limit."
                strategy_decision = "finish_recursion_limit"
                next_node_for_ws = None
                next_action = "finish"
            
            # 3. 사용자 지정 max_search_depth 확인
            elif not should_terminate and next_search_depth >= state.max_search_depth:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Max search depth ({state.max_search_depth}) reached.")
                should_terminate = True
                final_thought_to_use = best_across_all or current_round_best_thought
                final_answer_content = final_thought_to_use.content if final_thought_to_use else "Reached max depth, no definitive answer."
                strategy_decision = "finish_max_depth"
                next_node_for_ws = None
                next_action = "finish"
            
            # 4. 높은 점수 찾음 - 대부분의 성공 케이스
            elif not should_terminate and current_round_best_thought.evaluation_score is not None and \
                current_round_best_thought.evaluation_score >= self.score_threshold_to_finish:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): High-confidence thought found (Score: {current_round_best_thought.evaluation_score}). Finalizing.")
                should_terminate = True
                final_answer_content = current_round_best_thought.content
                strategy_decision = "finish_high_score"
                next_node_for_ws = None
                next_action = "finish"

            # 5. 지금까지의 최고 점수가 threshold의 85%는 되는데 탐색이 일정 깊이 진행됐을 경우
            elif not should_terminate and best_across_all and best_across_all.evaluation_score is not None and \
                    best_across_all.evaluation_score >= (self.score_threshold_to_finish * 0.85) and \
                    state.search_depth >= 3:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Decent solution found after sufficient exploration (Score: {best_across_all.evaluation_score}). Finalizing.")
                should_terminate = True
                final_answer_content = best_across_all.content
                strategy_decision = "finish_sufficient_score"
                next_node_for_ws = None
                next_action = "finish"

            # 6. 점수가 낮아서 더 이상 탐색할 가치가 없는 경우
            elif not should_terminate and (state.search_depth >= 2) and all(t.evaluation_score is not None and t.evaluation_score < self.min_score_to_continue for t in top_thoughts_for_expansion):
                logger.warning(f"Node '{self.node_id}' (Task: {state.task_id}): All top {self.beam_width} thoughts below threshold. Stopping.")
                should_terminate = True
                final_thought_to_use = best_across_all or current_round_best_thought
                final_answer_content = final_thought_to_use.content
                strategy_decision = "finish_low_score"
                next_node_for_ws = None
                next_action = "finish"
            
            # 계속 탐색 (다른 종료 조건에 해당되지 않을 경우)
            elif not should_terminate:
                logger.info(f"Node '{self.node_id}' (Task: {state.task_id}): Proceeding to next search depth ({next_search_depth}). Best thought to expand based on: {new_global_best_thought_id}")
                strategy_decision = "continue_search"
                next_action = "continue"

            # 상태 업데이트 생성
            update_payload = {
                "thoughts": state.thoughts,
                "current_best_thought_id": new_global_best_thought_id,
                "next_action": next_action
            }
            
            # Update state and create return payload
            if should_terminate:
                final_answer = final_answer_content
                
                # Improve final answer formatting for the original question
                if state.original_input and isinstance(final_answer, str):
                    if not final_answer.startswith("Based on") and not final_answer.startswith("Regarding"):
                        final_answer = f"Based on your request: '{state.original_input}', here is my solution:\n\n{final_answer}"
                
                update_payload["search_depth"] = state.search_depth if strategy_decision == "finish_high_score" else next_search_depth
                update_payload["error_message"] = None  # Normal termination, not an error
                update_payload["final_answer"] = final_answer
                
                # Critical fix: preserve dynamic_data to maintain subtask processing context
                update_payload["dynamic_data"] = state.dynamic_data.copy() if state.dynamic_data else {}
            else:
                update_payload["search_depth"] = next_search_depth
                update_payload["final_answer"] = None

            # 결정된 전략 알림
            await self.notification_service.broadcast_to_task(
                state.task_id,
                IntermediateResultMessage(
                    task_id=state.task_id, node_id=self.node_id,
                    result_step_name="search_strategy_decision",
                    data={
                        "decision": strategy_decision,
                        "current_best_thought_id": new_global_best_thought_id,
                        "current_best_score": state.get_thought_by_id(new_global_best_thought_id).evaluation_score
                            if new_global_best_thought_id and state.get_thought_by_id(new_global_best_thought_id) else None,
                        "next_depth": next_search_depth if not should_terminate else state.search_depth,
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

            # --- 반환 페이로드 구성 ---
            logger.debug(
                f"[Search-Strategy] task_id={state.task_id} "
                f"depth={state.search_depth} decision={strategy_decision} "
                f"next_action={next_action}"
            )
            
            new_dynamic_data = state.dynamic_data.copy() if state.dynamic_data else {}
            current_error_message_in_state = state.error_message

            final_update_payload: Dict[str, Any] = {
                "thoughts": state.thoughts, 
                "current_best_thought_id": new_global_best_thought_id,
                "next_action": next_action, 
                "error_message": None,
                "dynamic_data": new_dynamic_data
            }

            if should_terminate:
                final_update_payload["search_depth"] = state.search_depth
                final_update_payload["final_answer"] = final_answer_content
                
                if strategy_decision == "finish_recursion_limit":
                    final_update_payload["error_message"] = "Search for subtask stopped due to recursion limit within ToT."
                elif strategy_decision == "finish_max_depth":
                    final_update_payload["error_message"] = "Reached max search depth in ToT for subtask."
                elif strategy_decision == "finish_low_score":
                    final_update_payload["error_message"] = "Stopping ToT for subtask due to consistently low scores."
            else:
                final_update_payload["search_depth"] = next_search_depth
                final_update_payload["final_answer"] = None

            if current_error_message_in_state and not final_update_payload.get("error_message"):
                final_update_payload["error_message"] = current_error_message_in_state
            
            logger.info(
                f"Node '{self.node_id}' (Task: {state.task_id}): Returning update. "
                f"Subtask ToT Final Answer Present: {final_update_payload.get('final_answer') is not None}, "
                f"Next Action: {final_update_payload.get('next_action')}, "
                f"Search Depth: {final_update_payload.get('search_depth')}"
            )
            return final_update_payload
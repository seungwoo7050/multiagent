# multiagent/src/agents/graph_nodes/search_strategy_node.py

from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from src.config.logger import get_logger
from src.schemas.mcp_models import AgentGraphState, Thought # Step 4.1에서 정의

logger = get_logger(__name__)

class SearchStrategyNode:
    """
    ToT: 평가된 생각들을 바탕으로 다음 탐색 전략을 결정하는 노드.
    (예: 최상의 생각 선택, 탐색 깊이 증가, 종료 조건 확인 등)
    """
    def __init__(
        self,
        beam_width: int = 1, # 한 번에 확장할 유망한 생각의 수 (Beam Search의 폭)
        score_threshold_to_finish: float = 0.95, # 이 점수 이상이면 성공으로 간주하고 종료
        min_score_to_continue: float = 0.1, # 이 점수 미만이면 탐색 중단 가능성
        node_id: str = "search_strategy"
    ):
        if beam_width < 1:
            raise ValueError("Beam width must be at least 1.")
        self.beam_width = beam_width
        self.score_threshold_to_finish = score_threshold_to_finish
        self.min_score_to_continue = min_score_to_continue
        self.node_id = node_id
        logger.info(
            f"SearchStrategyNode '{self.node_id}' initialized. "
            f"Beam width: {self.beam_width}, Finish Threshold: {self.score_threshold_to_finish}, "
            f"Min Continue Score: {self.min_score_to_continue}"
        )

    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        logger.info(f"SearchStrategyNode '{self.node_id}' execution started for task: {state.task_id}, depth: {state.search_depth}")        
        if not state.thoughts or not any(t.evaluation_score is not None for t in state.thoughts):
            logger.warning(f"Node '{self.node_id}': No evaluated thoughts with scores found.")
            return {"next_action": "finish"}   # error_message → 생략


        # 1. 평가 완료된 생각들 필터링 및 정렬
        #    이전에 'evaluated' 된 것들 중에서 현재 확장할 대상을 고르는 것임.
        #    ThoughtGenerator가 생성한 것들을 StateEvaluator가 평가했고, 이제 그 결과를 보는 것.
        
        # 가장 최근에 'evaluated' 상태가 된 생각들을 대상으로 함.
        # 이를 위해선 AgentGraphState에 `last_evaluated_thought_ids` 같은 필드가 있거나,
        # `current_thoughts_to_evaluate`가 비워지기 전의 값을 참조해야 함.
        # 현재 상태 설계에서는 `state.thoughts` 전체를 봐야함.
        
        candidate_thoughts: List[Thought] = []
        parent_thought_id_of_candidates: Optional[str] = None

        # current_best_thought_id를 부모로 하여 생성되고 평가된 생각들을 찾음
        # 또는, AgentGraphState에 `last_generated_parent_id` 같은 필드를 두어 추적
        # 여기서는 current_best_thought_id를 마지막으로 확장된 노드로 가정하고,
        # 그것의 자식들(즉, 최근에 평가된 생각들)을 찾으려고 시도합니다.
        # 이 로직은 ThoughtGenerator가 parent_id를 어떻게 설정하느냐에 따라 달라짐.
        # 가장 간단하게는 evaluation_score가 있고, status가 'evaluated'인 모든 생각을 고려.
        
        evaluated_thoughts: List[Thought] = sorted(
            [t for t in state.thoughts if t.status == "evaluated" and t.evaluation_score is not None],
            key=lambda t: t.evaluation_score or -1.0,
            reverse=True
        )

        if not evaluated_thoughts:
            logger.warning(f"Node '{self.node_id}': No evaluated thoughts with scores found.")
            # 이전 평가 단계에서 오류가 있었거나, 생성된 생각이 없을 수 있음.
            # 또는 아직 평가할 대상이 current_thoughts_to_evaluate에 남아있을 수도 있으나,
            # 이 노드는 평가가 끝난 후 호출된다고 가정.
            return {
                "error_message": "No new thoughts were evaluated or no valid scores found.",
                "final_answer": "Could not determine a course of action due to lack of evaluated thoughts." 
                                 if not state.current_best_thought_id else state.get_thought_by_id(state.current_best_thought_id).content
                # next_node_override: "END" 등을 설정하여 그래프 종료 유도 가능
            }

        # 2. 최상위 생각(들) 선택
        top_thoughts_for_expansion = evaluated_thoughts[:self.beam_width]
        
        if not top_thoughts_for_expansion:
            logger.warning(f"Node '{self.node_id}': No top thoughts selected for expansion.")
            # 이 경우는 evaluated_thoughts가 비어있던 경우와 유사
            return {
                 "error_message": "No promising thoughts selected for further expansion.",
                 "final_answer": "Exploration did not yield any promising paths."
                                  if not state.current_best_thought_id else state.get_thought_by_id(state.current_best_thought_id).content
            }

        # 현재 라운드에서 가장 좋은 생각
        current_round_best_thought = top_thoughts_for_expansion[0]
        new_global_best_thought_id = state.current_best_thought_id
        
        # 전역 최선 업데이트
        global_best_thought = state.get_thought_by_id(state.current_best_thought_id) if state.current_best_thought_id else None
        if global_best_thought is None or \
           (current_round_best_thought.evaluation_score is not None and \
            (global_best_thought.evaluation_score is None or \
             current_round_best_thought.evaluation_score > global_best_thought.evaluation_score)):
            new_global_best_thought_id = current_round_best_thought.id
            logger.info(f"Node '{self.node_id}': New global best thought: {new_global_best_thought_id} (Score: {current_round_best_thought.evaluation_score})")
        elif global_best_thought:
             logger.info(f"Node '{self.node_id}': Global best thought remains: {global_best_thought.id} (Score: {global_best_thought.evaluation_score})")


        # 3. 종료 조건 판단
        next_search_depth = state.search_depth + 1
        final_answer: Optional[str] = None
        
        # 조건 1: 최대 깊이 도달
        if next_search_depth >= state.max_search_depth:
            logger.info(f"Node '{self.node_id}': Max search depth ({state.max_search_depth}) reached.")
            final_thought_to_use = state.get_thought_by_id(new_global_best_thought_id) if new_global_best_thought_id else current_round_best_thought
            final_answer = final_thought_to_use.content if final_thought_to_use else "Reached max depth, no definitive answer."
            return {
                "thoughts": state.thoughts, # 최종 상태 반영
                "current_best_thought_id": new_global_best_thought_id,
                "search_depth": next_search_depth,
                "final_answer": final_answer,
                "next_action": "finish",       # ← 추가
                "error_message": "Max search depth reached."
                # 오케스트레이터는 final_answer가 있으면 종료로 판단
            }

        # 조건 2: 점수 임계값 충족
        if current_round_best_thought.evaluation_score is not None and \
           current_round_best_thought.evaluation_score >= self.score_threshold_to_finish:
            logger.info(f"Node '{self.node_id}': High-confidence thought found (Score: {current_round_best_thought.evaluation_score}). Finalizing.")
            final_answer = current_round_best_thought.content
            # 이 경우, current_best_thought_id는 current_round_best_thought.id로 설정하는 것이 자연스러움
            return {
                "thoughts": state.thoughts,
                "current_best_thought_id": current_round_best_thought.id,
                "search_depth": state.search_depth, # 현재 깊이에서 종료했음을 명시 (또는 next_search_depth)
                "final_answer": final_answer,
                "error_message": None
            }

        # 조건 3: 진행 불가 (모든 후보의 점수가 너무 낮음)
        # beam_width 만큼의 생각 모두가 min_score_to_continue 미만이면 중단
        if all(t.evaluation_score is not None and t.evaluation_score < self.min_score_to_continue for t in top_thoughts_for_expansion):
            logger.warning(f"Node '{self.node_id}': All top {self.beam_width} thoughts have scores below threshold ({self.min_score_to_continue}). Stopping exploration.")
            final_thought_to_use = state.get_thought_by_id(new_global_best_thought_id) if new_global_best_thought_id else current_round_best_thought
            final_answer = final_thought_to_use.content if final_thought_to_use else "Exploration stopped due to low scores, no definitive answer."
            return {
                "thoughts": state.thoughts,
                "current_best_thought_id": new_global_best_thought_id,
                "search_depth": next_search_depth, # 깊이는 증가했지만 더 이상 진행 안 함
                "final_answer": final_answer,
                "error_message": "Exploration stopped, all current paths deemed unpromising."
            }

        # 4. 다음 상태 결정 (계속 탐색)
        # ThoughtGenerator가 current_best_thought_id (이제 new_global_best_thought_id가 됨)를
        # 부모로 삼아 새로운 생각들을 생성하도록 유도합니다.
        # 만약 beam_width > 1 이고 여러 경로를 동시에 확장하고 싶다면,
        # "thoughts_to_expand": [t.id for t in top_thoughts_for_expansion] 와 같이 상태를 업데이트하고
        # ThoughtGenerator가 이를 처리하도록 수정해야 합니다.
        # 여기서는 가장 좋은 것 하나(new_global_best_thought_id)를 다음 생성의 기준으로 삼습니다.
        
        # 확장할 생각들의 상태를 "expanding" 등으로 변경하여 중복 확장을 막을 수 있음 (선택적)
        # for t_expand in top_thoughts_for_expansion:
        #     for i, t_state in enumerate(state.thoughts):
        #         if t_state.id == t_expand.id:
        #             state.thoughts[i].status = "expanding" # 상태 직접 변경 또는 새 객체로 교체
        #             break


        logger.info(f"Node '{self.node_id}': Proceeding to next search depth ({next_search_depth}). Best thought to expand based on: {new_global_best_thought_id}")
        
        # StateEvaluator에서 current_thoughts_to_evaluate를 비웠으므로, 
        # SearchStrategy는 다음 ThoughtGenerator가 뭘 생성할지 알려줄 필요는 없음 (Generator가 알아서 함)
        # 만약 SearchStrategy가 명시적으로 어떤 thought를 expand할지 지정해야 한다면,
        # "thoughts_pending_generation_from": [new_global_best_thought_id] 같은 필드를 추가할 수 있음.
        return {
            "thoughts": state.thoughts, # 변경된 status 등을 반영
            "current_best_thought_id": new_global_best_thought_id, 
            "search_depth": next_search_depth,
            "error_message": None 
            # current_thoughts_to_evaluate는 이전 단계에서 비워졌어야 함.
        }
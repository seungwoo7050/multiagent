from typing import List, Dict, Any, Optional, Tuple, Set, TypeVar
import time
from pydantic import BaseModel, Field, ConfigDict
from src.core.mcp.protocol import ContextProtocol
from src.config.logger import get_logger
logger = get_logger(__name__)
TContext = TypeVar('TContext', bound=ContextProtocol)

class ContextTransition(BaseModel):
    from_context_id: Optional[str] = Field(None, description='ID of the source context, if any')
    from_context_type: Optional[str] = Field(None, description='Type name of the source context')
    to_context_id: str = Field(..., description='ID of the resulting context')
    to_context_type: str = Field(..., description='Type name of the resulting context')
    component_name: str = Field(..., description='Name of the component performing the transition')
    operation: Optional[str] = Field(None, description='Specific operation performed by the component')
    timestamp: float = Field(default_factory=time.time, description='Timestamp of the transition')
    metadata: Dict[str, Any] = Field(default_factory=dict, description='Additional metadata about the transition')
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ContextFlowManager:

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.flow_history: Dict[str, List[ContextTransition]] = {}
        self.transition_log: List[ContextTransition] = []
        logger.debug(f'ContextFlowManager initialized for workflow {workflow_id}')

    def log_transition(self, to_context: ContextProtocol, component_name: str, operation: Optional[str]=None, from_context: Optional[ContextProtocol]=None, metadata: Optional[Dict[str, Any]]=None) -> None:
        from_context_id = getattr(from_context, 'context_id', None) if from_context else None
        from_context_type = type(from_context).__name__ if from_context else None
        to_context_id = getattr(to_context, 'context_id', 'unknown_id')
        to_context_type = type(to_context).__name__
        if to_context_id == 'unknown_id':
            logger.warning(f'Logging transition to context without a valid context_id (Type: {to_context_type}). Cannot properly track history by ID.')
        transition = ContextTransition(from_context_id=from_context_id, from_context_type=from_context_type, to_context_id=to_context_id, to_context_type=to_context_type, component_name=component_name, operation=operation, metadata=metadata or {})
        self.transition_log.append(transition)
        if to_context_id != 'unknown_id':
            if to_context_id not in self.flow_history:
                self.flow_history[to_context_id] = []
            self.flow_history[to_context_id].append(transition)
        log_from = from_context_type or 'Start'
        logger.debug(f'Logged context transition: {log_from} -> {to_context_type} via {component_name}.{operation or ''} (To ID: {to_context_id})')

    def get_context_history(self, context_id: str) -> List[ContextTransition]:
        return self.flow_history.get(context_id, [])

    def get_full_transition_log(self) -> List[ContextTransition]:
        return self.transition_log

    def find_originating_context(self, context_id: str, target_type: Optional[Type[TContext]]=None) -> Optional[str]:
        current_id: Optional[str] = context_id
        visited: Set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            transitions = self.flow_history.get(current_id, [])
            if not transitions:
                break
            last_transition = transitions[0]
            from_id = last_transition.from_context_id
            from_type_name = last_transition.from_context_type
            if from_id is None:
                break
            if target_type and from_type_name == target_type.__name__:
                return from_id
            current_id = from_id
        if target_type is None and current_id:
            final_transitions = self.flow_history.get(current_id, [])
            if final_transitions and final_transitions[0].from_context_id is None:
                return current_id
            elif not final_transitions:
                return current_id
        return None
from pydantic import BaseModel, Field, ConfigDict
from typing import Set
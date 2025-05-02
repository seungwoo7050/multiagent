import asyncio
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.config.logger import get_logger
from src.core.mcp.protocol import ContextProtocol

logger = get_logger(__name__)

class RoutingTarget(BaseModel):
    target_type: str = Field(..., description='Type of the routing target')
    target_id: Optional[str] = Field(None, description='Identifier for the target (e.g., pool name, agent type/id)')
    priority: Optional[int] = Field(None, description='Optional priority for routing')
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ContextRouter:

    def __init__(self):
        logger.debug('ContextRouter initialized.')
        self.type_based_rules: Dict[str, RoutingTarget] = {'TaskContext': RoutingTarget(target_type='agent_type', target_id='planner'), 'LLMInputContext': RoutingTarget(target_type='worker_pool', target_id='llm_processing_pool')}

    async def determine_route(self, context: ContextProtocol, available_targets: Optional[List[Any]]=None, current_system_state: Optional[Dict[str, Any]]=None) -> Optional[RoutingTarget]:
        context_type_name = type(context).__name__
        context_id = getattr(context, 'context_id', 'unknown_id')
        logger.debug(f'Determining route for context {context_id} (Type: {context_type_name})')
        if context_type_name in self.type_based_rules:
            target = self.type_based_rules[context_type_name]
            logger.info(f"Routing context {context_id} based on type '{context_type_name}' to target type '{target.target_type}' (ID: {target.target_id})")
            return target
        logger.warning(f"No specific routing rule found for context type '{context_type_name}'. Cannot determine route.")
        return None

_router_instance: Optional[ContextRouter] = None
_router_lock = asyncio.Lock()

async def get_context_router() -> ContextRouter:
    global _router_instance
    if _router_instance is not None:
        return _router_instance
    async with _router_lock:
        if _router_instance is None:
            _router_instance = ContextRouter()
            logger.info('Singleton ContextRouter instance created.')
    if _router_instance is None:
        raise RuntimeError('Failed to create ContextRouter instance.')
    return _router_instance
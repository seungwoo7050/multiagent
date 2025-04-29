from typing import Dict, Optional, Any
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import TaskContext

def get_context_labels(context: Optional[ContextProtocol]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if context is None:
        return labels
    if isinstance(context, TaskContext):
        labels['task_type'] = context.task_type or 'unknown'
    if 'context_class' not in labels:
        labels['context_class'] = type(context).__name__
    return {k: v for k, v in labels.items() if v}
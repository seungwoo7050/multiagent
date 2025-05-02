from typing import Dict, Optional

from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import TaskContext


def get_context_labels(context: Optional[ContextProtocol]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if context is None:
        return labels
    
    # Add context class
    labels['context_class'] = type(context).__name__
    
    # Add task-specific labels
    if isinstance(context, TaskContext):
        labels['task_type'] = context.task_type or 'unknown'
        if context.task_id:
            labels['task_id'] = context.task_id
    
    # Extract relevant metadata as labels
    if hasattr(context, 'metadata') and context.metadata:
        for key in ['priority', 'source', 'user_id']:
            if key in context.metadata:
                labels[f'metadata_{key}'] = str(context.metadata[key])
        
        # Performance metrics
        if 'estimated_tokens' in context.metadata:
            labels['estimated_tokens'] = str(context.metadata['estimated_tokens'])
    
    return {k: str(v) for k, v in labels.items() if v}
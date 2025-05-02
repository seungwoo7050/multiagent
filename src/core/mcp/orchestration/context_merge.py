import asyncio
import copy
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, cast

from pydantic import BaseModel

from src.config.logger import get_logger
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema

logger = get_logger(__name__)
TContext = TypeVar('TContext', bound=ContextProtocol)

class ContextMergeStrategy(Enum):
    OVERWRITE = 'overwrite'
    APPEND_LIST = 'append_list'
    CONCATENATE_STRING = 'concatenate_string'
    RECURSIVE_DICT_MERGE = 'recursive_dict_merge'
    CUSTOM = 'custom'

class ContextMerger:

    def __init__(self):
        logger.debug('ContextMerger initialized.')

    async def merge_contexts(self, contexts: List[ContextProtocol], target_context_type: Optional[Type[TContext]]=None, strategy: ContextMergeStrategy=ContextMergeStrategy.RECURSIVE_DICT_MERGE, custom_merge_func: Optional[Callable[[Dict, Dict], Dict]]=None, initial_context_data: Optional[Dict[str, Any]]=None) -> Optional[TContext]:
        if not contexts:
            logger.warning('No contexts provided for merging.')
            return None
        merged_data: Dict[str, Any] = copy.deepcopy(initial_context_data) if initial_context_data else {}
        all_metadata: List[Dict[str, Any]] = []
        context_ids_merged: List[str] = []
        final_version = '1.0.0'
        logger.info(f'Starting merge of {len(contexts)} contexts using strategy: {strategy.value}')
        for i, context in enumerate(contexts):
            try:
                context_data: Optional[Dict[str, Any]] = None
                if hasattr(context, 'serialize') and callable(context.serialize):
                    context_data = context.serialize()
                elif isinstance(context, BaseModel):
                    context_data = context.model_dump(mode='json')
                elif isinstance(context, dict):
                    context_data = context
                else:
                    logger.warning(f'Context at index {i} (type: {type(context).__name__}) cannot be easily converted to dict, skipping.')
                    continue
                context_id = context_data.get('context_id', f'context_{i}')
                context_ids_merged.append(context_id)
                metadata = context_data.pop('metadata', {})
                if metadata:
                    all_metadata.append(metadata)
                if strategy == ContextMergeStrategy.CUSTOM:
                    if custom_merge_func and callable(custom_merge_func):
                        merged_data = custom_merge_func(merged_data, context_data)
                    else:
                        raise ValueError('Custom merge strategy selected but no valid custom_merge_func provided.')
                else:
                    merged_data = self._apply_merge_strategy(merged_data, context_data, strategy)
            except Exception as e:
                logger.error(f'Error processing context at index {i} (ID: {context_id}) during merge: {e}', exc_info=True)
                continue
        if not merged_data and (not all_metadata):
            logger.warning('Merging resulted in empty data and metadata. No final context created.')
            return None
        final_metadata: Dict[str, Any] = {}
        for meta in all_metadata:
            if isinstance(meta, dict):
                final_metadata.update(meta)
        final_metadata['_merge_info'] = {'strategy': strategy.value, 'merged_context_count': len(context_ids_merged), 'original_context_ids': context_ids_merged, 'merged_at': time.time()}
        merged_data['metadata'] = final_metadata
        merged_data['version'] = final_version
        if 'context_id' not in merged_data:
            from src.utils.ids import generate_uuid
            merged_data['context_id'] = generate_uuid()
        TargetClass = target_context_type or BaseContextSchema
        try:
            merged_context = TargetClass.model_validate(merged_data)
            logger.info(f'Successfully merged {len(context_ids_merged)} contexts into new context (ID: {getattr(merged_context, 'context_id', 'N/A')}, Type: {TargetClass.__name__})')
            return cast(TContext, merged_context)
        except Exception as e:
            logger.error(f'Failed to create target context object of type {TargetClass.__name__} after merge: {e}', exc_info=True)
            return None

    def _apply_merge_strategy(self, base_dict: Dict[str, Any], new_dict: Dict[str, Any], strategy: ContextMergeStrategy) -> Dict[str, Any]:
        merged = base_dict.copy()
        for key, new_value in new_dict.items():
            if key not in merged:
                merged[key] = new_value
            else:
                base_value = merged[key]
                if strategy == ContextMergeStrategy.OVERWRITE:
                    merged[key] = new_value
                elif strategy == ContextMergeStrategy.APPEND_LIST:
                    if isinstance(base_value, list):
                        if isinstance(new_value, list):
                            merged[key].extend(new_value)
                        else:
                            merged[key].append(new_value)
                    else:
                        merged[key] = [base_value, new_value]
                elif strategy == ContextMergeStrategy.CONCATENATE_STRING:
                    if isinstance(base_value, str) and isinstance(new_value, str):
                        merged[key] = f'{base_value}\n---\n{new_value}'
                    else:
                        merged[key] = new_value
                elif strategy == ContextMergeStrategy.RECURSIVE_DICT_MERGE:
                    if isinstance(base_value, dict) and isinstance(new_value, dict):
                        merged[key] = self._apply_merge_strategy(base_value, new_value, strategy)
                    else:
                        merged[key] = new_value
                else:
                    logger.warning(f"Unknown or unhandled merge strategy '{strategy.value}' for key '{key}'. Falling back to OVERWRITE.")
                    merged[key] = new_value
        return merged


_merger_instance: Optional[ContextMerger] = None
_merger_lock = asyncio.Lock()

async def get_context_merger() -> ContextMerger:
    global _merger_instance
    if _merger_instance is not None:
        return _merger_instance
    async with _merger_lock:
        if _merger_instance is None:
            _merger_instance = ContextMerger()
            logger.info('Singleton ContextMerger instance created.')
    if _merger_instance is None:
        raise RuntimeError('Failed to create ContextMerger instance.')
    return _merger_instance
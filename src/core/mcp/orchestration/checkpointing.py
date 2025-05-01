import time
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from src.core.mcp.protocol import ContextProtocol
from src.orchestration.workflow import WorkflowState
from src.memory.manager import MemoryManager
from src.config.logger import get_logger
from src.config.errors import OrchestrationError
from src.core.exceptions import ErrorCode
logger = get_logger(__name__)

class CheckpointManager:

    def __init__(self, memory_manager: MemoryManager, checkpoint_prefix: str='checkpoint'):
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError('memory_manager must be an instance of MemoryManager')
        self.memory_manager = memory_manager
        self.prefix = checkpoint_prefix
        logger.debug('CheckpointManager initialized.')

    def _get_checkpoint_key(self, task_id: str, checkpoint_id: Optional[str]=None) -> str:
        if checkpoint_id is None:
            checkpoint_id = f'ts_{int(time.time() * 1000)}'
        return f'{self.prefix}:{checkpoint_id}'

    async def save_checkpoint(self, workflow_state: WorkflowState, checkpoint_id: Optional[str]=None, ttl: Optional[int]=None) -> Optional[str]:
        task_id = workflow_state.task_id
        save_checkpoint_id = checkpoint_id or f'ts_{int(time.time() * 1000)}'
        checkpoint_key = self._get_checkpoint_key(task_id, save_checkpoint_id)
        logger.info(f'Saving checkpoint for task {task_id} with ID {save_checkpoint_id} (Memory Key: {checkpoint_key})')
        try:
            state_dict = workflow_state.model_dump(mode='json')
            success = await self.memory_manager.save(key=checkpoint_key, context_id=task_id, data=state_dict, ttl=ttl, update_cache=False)
            if success:
                logger.info(f'Checkpoint {save_checkpoint_id} saved successfully for task {task_id}.')
                return save_checkpoint_id
            else:
                logger.error(f'Failed to save checkpoint {save_checkpoint_id} for task {task_id} (MemoryManager.save returned False).')
                return None
        except Exception as e:
            logger.exception(f'Error saving checkpoint {save_checkpoint_id} for task {task_id}: {e}')
            return None

    async def load_checkpoint(self, task_id: str, checkpoint_id: str) -> Optional[WorkflowState]:
        checkpoint_key = self._get_checkpoint_key(task_id, checkpoint_id)
        logger.info(f'Loading checkpoint {checkpoint_id} for task {task_id} (Memory Key: {checkpoint_key})')
        try:
            state_data = await self.memory_manager.load(key=checkpoint_key, context_id=task_id, use_cache=False)
            if state_data:
                workflow_state = WorkflowState.model_validate(state_data)
                logger.info(f'Checkpoint {checkpoint_id} loaded successfully for task {task_id}.')
                return workflow_state
            else:
                logger.warning(f'Checkpoint {checkpoint_id} not found for task {task_id}.')
                return None
        except Exception as e:
            logger.exception(f'Error loading checkpoint {checkpoint_id} for task {task_id}: {e}')
            return None

    async def load_latest_checkpoint(self, task_id: str) -> Optional[WorkflowState]:
        logger.info(f'Loading latest checkpoint for task {task_id}')
        try:
            pattern = f'{self.prefix}:ts_*'
            checkpoint_keys: List[str] = await self.memory_manager.list_keys(context_id=task_id, pattern=pattern)
            if not checkpoint_keys:
                logger.info(f'No checkpoints found for task {task_id}.')
                return None
            latest_key: Optional[str] = None
            max_ts: int = -1
            for key in checkpoint_keys:
                try:
                    parts = key.split(':')
                    if len(parts) == 2 and parts[1].startswith('ts_'):
                        ts_str = parts[1][3:]
                        ts = int(ts_str)
                        if ts > max_ts:
                            max_ts = ts
                            latest_key = key
                    else:
                        logger.warning(f'Skipping key with unexpected format: {key}')
                except (ValueError, IndexError) as parse_err:
                    logger.warning(f'Could not parse timestamp from checkpoint key: {key}. Error: {parse_err}')
                    continue
            if latest_key:
                checkpoint_id = latest_key.split(':')[-1]
                logger.info(f'Latest checkpoint found for task {task_id} is {checkpoint_id}')
                return await self.load_checkpoint(task_id, checkpoint_id)
            else:
                logger.info(f'No valid timestamped checkpoints found for task {task_id}.')
                return None
        except Exception as e:
            logger.exception(f'Error loading latest checkpoint for task {task_id}: {e}')
            return None

    async def list_checkpoints(self, task_id: str) -> List[str]:
        logger.debug(f'Listing checkpoints for task {task_id}')
        try:
            pattern = f'{self.prefix}:*'
            checkpoint_keys: List[str] = await self.memory_manager.list_keys(context_id=task_id, pattern=pattern)
            checkpoint_ids: List[str] = []
            for key in checkpoint_keys:
                parts = key.split(':', 1)
                if len(parts) == 2:
                    checkpoint_ids.append(parts[1])
                else:
                    logger.warning(f'Found checkpoint key with unexpected format: {key}')
            try:
                checkpoint_ids.sort(key=lambda id_str: int(id_str[3:]) if id_str.startswith('ts_') else -1, reverse=True)
            except ValueError:
                logger.warning('Could not sort checkpoints by timestamp due to parsing error.')
                checkpoint_ids.sort(reverse=True)
            return checkpoint_ids
        except Exception as e:
            logger.exception(f'Error listing checkpoints for task {task_id}: {e}')
            return []

    async def delete_checkpoint(self, task_id: str, checkpoint_id: str) -> bool:
        checkpoint_key = self._get_checkpoint_key(task_id, checkpoint_id)
        logger.info(f'Deleting checkpoint {checkpoint_id} for task {task_id} (Memory Key: {checkpoint_key})')
        try:
            return await self.memory_manager.delete(key=checkpoint_key, context_id=task_id)
        except Exception as e:
            logger.exception(f'Error deleting checkpoint {checkpoint_id} for task {task_id}: {e}')
            return False

    async def delete_old_checkpoints(self, task_id: str, keep_latest: int=3) -> int:
        if keep_latest < 0:
            logger.warning('keep_latest must be non-negative. Not deleting any checkpoints.')
            return 0
        logger.info(f'Deleting old checkpoints for task {task_id}, keeping latest {keep_latest}')
        deleted_count = 0
        try:
            checkpoints = await self.list_checkpoints(task_id)
            if len(checkpoints) > keep_latest:
                checkpoints_to_delete = checkpoints[keep_latest:]
                logger.debug(f'Found {len(checkpoints_to_delete)} old checkpoints to delete for task {task_id}.')
                delete_tasks = [self.delete_checkpoint(task_id, cp_id) for cp_id in checkpoints_to_delete]
                results = await asyncio.gather(*delete_tasks, return_exceptions=True)
                deleted_count = sum((1 for res in results if res is True))
                errors = [res for res in results if isinstance(res, Exception)]
                if errors:
                    logger.error(f'Encountered {len(errors)} errors while deleting old checkpoints for task {task_id}: {errors}')
            else:
                logger.info(f'No old checkpoints to delete for task {task_id} (found {len(checkpoints)}, keeping {keep_latest}).')
            logger.info(f'Deleted {deleted_count} old checkpoints for task {task_id}.')
            return deleted_count
        except Exception as e:
            logger.exception(f'Error deleting old checkpoints for task {task_id}: {e}')
            return deleted_count
_checkpoint_manager_instance: Optional[CheckpointManager] = None
_checkpoint_manager_lock = asyncio.Lock()

async def get_checkpoint_manager(memory_manager: Optional[MemoryManager]=None) -> CheckpointManager:
    global _checkpoint_manager_instance
    if _checkpoint_manager_instance is not None:
        return _checkpoint_manager_instance
    async with _checkpoint_manager_lock:
        if _checkpoint_manager_instance is None:
            if not memory_manager:
                try:
                    from src.memory.manager import get_memory_manager
                    memory_manager = await get_memory_manager()
                except ImportError:
                    logger.error('Default MemoryManager function (get_memory_manager) not found.')
                    raise ValueError('MemoryManager dependency not available for CheckpointManager.')
                except Exception as mm_err:
                    logger.error(f'Failed to get default MemoryManager: {mm_err}')
                    raise ValueError('Could not obtain MemoryManager dependency for CheckpointManager.')
            _checkpoint_manager_instance = CheckpointManager(memory_manager=memory_manager)
            logger.info('Singleton CheckpointManager instance created.')
    if _checkpoint_manager_instance is None:
        raise RuntimeError('Failed to create CheckpointManager instance.')
    return _checkpoint_manager_instance
from pydantic import BaseModel, Field, ConfigDict
import asyncio
from typing import Any, Coroutine, List, Optional

from src.config.logger import get_logger

logger = get_logger(__name__)

async def execute_in_background(tasks: List[Coroutine[Any, Any, Any]], task_names: Optional[List[str]]=None) -> None:
    num_tasks = len(tasks)
    if not tasks:
        logger.debug('No tasks provided for background execution.')
        return
    use_names = False
    if task_names:
        if len(task_names) == num_tasks:
            use_names = True
        else:
            logger.warning(f'Mismatch between number of tasks ({num_tasks}) and task names ({len(task_names)}). Task names will not be used in logs.')
    task_names_str = f' ({', '.join(task_names)})' if use_names else ''
    logger.info(f'Starting execution of {num_tasks} tasks in background{task_names_str}.')
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = 0
    failure_count = 0
    for i, result in enumerate(results):
        task_name = task_names[i] if use_names else f'Task {i}'
        if isinstance(result, Exception):
            logger.error(f"Background task '{task_name}' failed: {str(result)}", exc_info=result)
            failure_count += 1
        else:
            logger.info(f"Background task '{task_name}' completed successfully.")
            success_count += 1
    logger.info(f'Background execution finished{task_names_str}. Success: {success_count}, Failed: {failure_count}')
import abc
import random
from typing import List, Optional, Any, Dict

class BaseLoadBalancerStrategy(abc.ABC):

    @abc.abstractmethod
    def select_worker(self, available_workers: List[Any], task_info: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        pass

    def update_worker_status(self, worker_id: Any, status: Dict[str, Any]) -> None:
        pass

class RoundRobinStrategy(BaseLoadBalancerStrategy):

    def __init__(self) -> None:
        self._current_index: int = 0

    def select_worker(self, available_workers: List[Any], task_info: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        if not available_workers:
            return None
        num_workers: int = len(available_workers)
        selected_index: int = self._current_index % num_workers
        self._current_index = (self._current_index + 1) % num_workers
        selected_worker = available_workers[selected_index]
        logger.debug(f'RoundRobin selected worker at index {selected_index} (Worker: {selected_worker}, Next index: {self._current_index})')
        return selected_worker

class RandomStrategy(BaseLoadBalancerStrategy):

    def select_worker(self, available_workers: List[Any], task_info: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        if not available_workers:
            return None
        selected_worker = random.choice(available_workers)
        logger.debug(f'Random strategy selected worker: {selected_worker}')
        return selected_worker
import abc
import random
from typing import Any, Dict, List, Optional

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager

# Initialize logger and metrics manager
logger = get_logger(__name__)
metrics = get_metrics_manager()

class BaseLoadBalancerStrategy(abc.ABC):
    """
    Abstract base class for worker load balancing strategies.
    Implementations determine how tasks are distributed among workers.
    """
    @abc.abstractmethod
    def select_worker(self, available_workers: List[Any], task_info: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        """
        Select a worker from the available workers pool.
        
        Args:
            available_workers: List of available workers
            task_info: Optional task information for making informed decisions
            
        Returns:
            Any: Selected worker or None if no workers available
        """

    def update_worker_status(self, worker_id: Any, status: Dict[str, Any]) -> None:
        """
        Update status information for a worker.
        
        Args:
            worker_id: Identifier for the worker
            status: Status information dictionary
        """

class RoundRobinStrategy(BaseLoadBalancerStrategy):
    """
    Round-robin load balancing strategy.
    Distributes tasks evenly among workers in a circular sequence.
    """
    def __init__(self) -> None:
        """Initialize the round-robin strategy"""
        self._current_index: int = 0

    def select_worker(self, available_workers: List[Any], task_info: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        """
        Select the next worker in round-robin order.
        
        Args:
            available_workers: List of available workers
            task_info: Optional task information (not used in this strategy)
            
        Returns:
            Any: Selected worker or None if no workers available
        """
        if not available_workers:
            logger.debug("No workers available for RoundRobin selection")
            return None
            
        num_workers: int = len(available_workers)
        selected_index: int = self._current_index % num_workers
        self._current_index = (self._current_index + 1) % num_workers
        selected_worker = available_workers[selected_index]
        
        logger.debug(
            f'RoundRobin selected worker at index {selected_index} '
            f'(Worker: {selected_worker}, Next index: {self._current_index})'
        )
        
        # Track metrics for the selection
        metrics.track_task('worker_selection', strategy='round_robin')
        
        return selected_worker

class RandomStrategy(BaseLoadBalancerStrategy):
    """
    Random load balancing strategy.
    Randomly selects a worker for each task, useful for preventing hotspots.
    """
    def select_worker(self, available_workers: List[Any], task_info: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        """
        Select a worker randomly.
        
        Args:
            available_workers: List of available workers
            task_info: Optional task information (not used in this strategy)
            
        Returns:
            Any: Selected worker or None if no workers available
        """
        if not available_workers:
            logger.debug("No workers available for Random selection")
            return None
            
        selected_worker = random.choice(available_workers)
        logger.debug(f'Random strategy selected worker: {selected_worker}')
        
        # Track metrics for the selection
        metrics.track_task('worker_selection', strategy='random')
        
        return selected_worker

class WeightedRoundRobinStrategy(BaseLoadBalancerStrategy):
    """
    Weighted round-robin load balancing strategy.
    Distributes tasks based on worker capacity/weights.
    """
    def __init__(self) -> None:
        """Initialize the weighted round-robin strategy"""
        self._worker_weights: Dict[Any, int] = {}
        self._current_index: int = 0
        
    def set_worker_weight(self, worker_id: Any, weight: int) -> None:
        """
        Set the weight for a specific worker.
        
        Args:
            worker_id: Worker identifier
            weight: Weight value (higher means more capacity)
        """
        if weight <= 0:
            logger.warning(f"Invalid worker weight: {weight} for worker {worker_id}. Must be positive.")
            weight = 1
            
        self._worker_weights[worker_id] = weight
        logger.debug(f"Set weight {weight} for worker {worker_id}")
        
    def update_worker_status(self, worker_id: Any, status: Dict[str, Any]) -> None:
        """
        Update worker status, potentially adjusting weights.
        
        Args:
            worker_id: Worker identifier
            status: Status information dictionary
        """
        if 'weight' in status and isinstance(status['weight'], int) and status['weight'] > 0:
            self._worker_weights[worker_id] = status['weight']
            logger.debug(f"Updated weight to {status['weight']} for worker {worker_id}")

    def select_worker(self, available_workers: List[Any], task_info: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        """
        Select a worker using weighted round-robin.
        
        Args:
            available_workers: List of available workers
            task_info: Optional task information
            
        Returns:
            Any: Selected worker or None if no workers available
        """
        if not available_workers:
            logger.debug("No workers available for WeightedRoundRobin selection")
            return None
            
        # Create weighted list of workers
        weighted_workers = []
        for worker in available_workers:
            weight = self._worker_weights.get(worker, 1)
            weighted_workers.extend([worker] * weight)
            
        if not weighted_workers:
            logger.warning("No weighted workers available, falling back to simple round-robin")
            return RoundRobinStrategy().select_worker(available_workers, task_info)
            
        num_weighted_workers = len(weighted_workers)
        selected_index = self._current_index % num_weighted_workers
        self._current_index = (self._current_index + 1) % num_weighted_workers
        selected_worker = weighted_workers[selected_index]
        
        logger.debug(
            f'WeightedRoundRobin selected worker: {selected_worker} '
            f'(Weight: {self._worker_weights.get(selected_worker, 1)})'
        )
        
        # Track metrics for the selection
        metrics.track_task('worker_selection', strategy='weighted_round_robin')
        
        return selected_worker

def create_load_balancer(strategy_name: str = "round_robin") -> BaseLoadBalancerStrategy:
    """
    Factory function to create a load balancer by name.
    
    Args:
        strategy_name: Name of the strategy to create
        
    Returns:
        BaseLoadBalancerStrategy: The created load balancer instance
    """
    strategies = {
        "round_robin": RoundRobinStrategy,
        "random": RandomStrategy,
        "weighted_round_robin": WeightedRoundRobinStrategy
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        logger.warning(f"Unknown load balancer strategy '{strategy_name}'. Using RoundRobin as default.")
        return RoundRobinStrategy()
        
    return strategy_class()
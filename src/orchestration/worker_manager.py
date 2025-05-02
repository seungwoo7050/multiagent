import asyncio
import time
from typing import Optional, cast

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.core.queue_worker_pool import (QueueWorkerPool, QueueWorkerPoolConfig,
                                        QueueWorkerPoolMetrics)
from src.core.worker_pool import WorkerPoolType, get_worker_pool

metrics = get_metrics_manager()
logger = get_logger(__name__)
settings = get_settings()

class WorkerManager:

    def __init__(self, worker_pool: Optional[QueueWorkerPool]=None, monitor_interval: int=15, scale_check_interval: int=30):
        self.worker_pool: QueueWorkerPool = worker_pool or cast(QueueWorkerPool, get_worker_pool('default', WorkerPoolType.QUEUE_ASYNCIO))
        self.monitor_interval: int = monitor_interval
        self.scale_check_interval: int = max(scale_check_interval, monitor_interval)
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._last_scale_check: float = 0
        self.min_workers: int = getattr(settings, 'WORKER_MIN_COUNT', 1)
        pool_max_workers = getattr(self.worker_pool.config, 'workers', 1)
        self.max_workers: int = getattr(settings, 'WORKER_MAX_COUNT', pool_max_workers)
        self.scale_up_queue_factor: float = getattr(settings, 'WORKER_SCALE_UP_QUEUE_FACTOR', 5.0)
        self.scale_down_idle_threshold: float = getattr(settings, 'WORKER_SCALE_DOWN_IDLE_THRESHOLD', 0.2)
        self.min_scale_interval: int = getattr(settings, 'WORKER_MIN_SCALE_INTERVAL_S', 60)
        logger.info(f'WorkerManager initialized (Pool: {self.worker_pool.name}, Min: {self.min_workers}, Max: {self.max_workers}, Monitor Interval: {self.monitor_interval}s, Scale Interval: {self.scale_check_interval}s)')
        logger.info(f'Scaling Params: ScaleUpFactor={self.scale_up_queue_factor}, ScaleDownThreshold={self.scale_down_idle_threshold}, MinScaleInterval={self.min_scale_interval}s')

    async def _monitor_loop(self) -> None:
        logger.info(f"Worker monitoring loop started for pool '{self.worker_pool.name}'.")
        while self._running:
            try:
                current_time: float = time.monotonic()
                metrics: Optional[QueueWorkerPoolMetrics] = None
                active_workers: int = -1
                current_max_workers_in_pool: int = -1
                queue_size: int = -1
                if hasattr(self.worker_pool, 'metrics') and isinstance(self.worker_pool.metrics, QueueWorkerPoolMetrics):
                    metrics = self.worker_pool.metrics
                    active_workers = metrics.active_workers
                    metrics.track_task('processing', value=metrics.running_tasks) # running_tasks 로 값 설정
                else:
                    logger.warning(f"Worker pool '{self.worker_pool.name}' does not have compatible 'metrics' attribute (QueueWorkerPoolMetrics). Cannot get active workers.")
                if hasattr(self.worker_pool, 'config') and isinstance(self.worker_pool.config, QueueWorkerPoolConfig):
                    current_max_workers_in_pool = self.worker_pool.config.workers
                else:
                    logger.warning(f"Worker pool '{self.worker_pool.name}' does not have compatible 'config' attribute (QueueWorkerPoolConfig). Using manager's max_workers ({self.max_workers}).")
                    current_max_workers_in_pool = self.max_workers
                if hasattr(self.worker_pool, 'get_queue_size') and callable(self.worker_pool.get_queue_size):
                    queue_size = self.worker_pool.get_queue_size()
                    metrics.track_task('queue_depth', value=queue_size)
                else:
                    logger.warning(f"Worker pool '{self.worker_pool.name}' does not have 'get_queue_size' method. Cannot get accurate queue depth.")
                log_status: str = f"Worker Pool Status ('{self.worker_pool.name}'): Active Workers={active_workers}/{current_max_workers_in_pool} (Manager Max: {self.max_workers}), Queue Depth={queue_size}"
                if metrics:
                    log_status += f', Submitted={metrics.tasks_submitted}, Completed={metrics.tasks_completed}, Failed={metrics.tasks_failed}'
                logger.info(log_status)
                if current_time - self._last_scale_check >= self.scale_check_interval:
                    self._last_scale_check = current_time
                    if active_workers != -1 and queue_size != -1 and (current_max_workers_in_pool != -1):
                        logger.debug(f"Checking scaling conditions for pool '{self.worker_pool.name}'...")
                        await self._check_scaling(active_workers, current_max_workers_in_pool, queue_size)
                    else:
                        logger.debug('Skipping scaling check due to missing or invalid metrics.')
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                logger.info(f"Worker monitoring loop cancelled for pool '{self.worker_pool.name}'.")
                break
            except Exception:
                logger.exception(f"Error in worker monitoring loop for pool '{self.worker_pool.name}'. Continuing...")
                await asyncio.sleep(self.monitor_interval)
        logger.info(f"Worker monitoring loop finished for pool '{self.worker_pool.name}'.")

    async def _check_scaling(self, active_workers: int, current_max_workers_in_pool: int, queue_size: int) -> None:
        scale_up_threshold: float = active_workers * self.scale_up_queue_factor
        should_scale_up = queue_size > scale_up_threshold and active_workers >= current_max_workers_in_pool and (current_max_workers_in_pool < self.max_workers)
        if should_scale_up:
            new_worker_count: int = min(self.max_workers, current_max_workers_in_pool + 1)
            log_msg: str = f"[SCALING] Recommendation: Scale Up for pool '{self.worker_pool.name}'. Queue depth ({queue_size}) > threshold ({scale_up_threshold:.1f}) and pool is busy ({active_workers}/{current_max_workers_in_pool}). Target workers: {new_worker_count}"
            logger.info(log_msg)
            metrics.track_worker_scaling(action='scale_up', pool_name=self.worker_pool.name)
            return
        utilization = active_workers / current_max_workers_in_pool if current_max_workers_in_pool > 0 else 0
        should_scale_down = queue_size == 0 and current_max_workers_in_pool > self.min_workers and (utilization < self.scale_down_idle_threshold)
        if should_scale_down:
            new_worker_count: int = max(self.min_workers, current_max_workers_in_pool - 1)
            if new_worker_count < current_max_workers_in_pool:
                log_msg: str = f"[SCALING] Recommendation: Scale Down for pool '{self.worker_pool.name}'. Queue empty and low utilization ({utilization:.2f} < {self.scale_down_idle_threshold:.2f}). Target workers: {new_worker_count}"
                logger.info(log_msg)
                metrics.track_worker_scaling(action='scale_down', pool_name=self.worker_pool.name)
        else:
            logger.debug(f"No scaling action needed for pool '{self.worker_pool.name}'. Queue={queue_size}, Active={active_workers}/{current_max_workers_in_pool}, UpCond=({queue_size > scale_up_threshold}, {active_workers >= current_max_workers_in_pool}, {current_max_workers_in_pool < self.max_workers}), DownCond=({queue_size == 0}, {current_max_workers_in_pool > self.min_workers}, {utilization < self.scale_down_idle_threshold})")

    async def start(self) -> None:
        if self._running:
            logger.warning('WorkerManager is already running.')
            return
        logger.info('Starting WorkerManager...')
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info('WorkerManager monitoring task created and started.')

    async def stop(self) -> None:
        if not self._running:
            logger.warning('WorkerManager is not running or already stopping.')
            return
        if self._monitor_task is None:
            logger.warning('WorkerManager has no active monitor task to stop.')
            self._running = False
            return
        logger.info('Stopping WorkerManager...')
        self._running = False
        if self._monitor_task and (not self._monitor_task.done()):
            logger.debug('Cancelling the monitoring task...')
            self._monitor_task.cancel()
            try:
                await self._monitor_task
                logger.debug('Monitoring task successfully awaited after cancellation.')
            except asyncio.CancelledError:
                logger.info('Monitoring task successfully cancelled.')
            except Exception as e:
                logger.error(f'Error encountered while waiting for monitoring task cancellation: {e}', exc_info=True)
            finally:
                self._monitor_task = None
        logger.info('WorkerManager stopped.')
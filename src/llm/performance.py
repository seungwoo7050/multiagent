import time
import asyncio
from collections import deque
from typing import Dict, Optional, Deque, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import get_metrics_manager
from src.utils.timing import get_current_time_ms

metrics = get_metrics_manager()
settings = get_settings()
logger = get_logger(__name__)

class ModelPerformanceStats(BaseModel):
    model_name: str
    provider: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    recent_requests: Deque[Tuple[int, int, bool]] = Field(default_factory=lambda: deque(maxlen=100))
    last_request_time_ms: Optional[int] = None
    last_error_time_ms: Optional[int] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.request_count if self.request_count > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / self.success_count if self.success_count > 0 else 0.0

    @property
    def average_tokens_per_second(self) -> float:
        if self.success_count == 0 or self.total_latency_ms == 0:
            return 0.0
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        total_latency_sec = self.total_latency_ms / 1000.0
        return total_tokens / total_latency_sec if total_latency_sec > 0 else 0.0

class LLMPerformanceTracker:

    def __init__(self, history_size: int=100):
        self._model_stats: Dict[str, ModelPerformanceStats] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self.history_size = history_size
        logger.info(f'LLMPerformanceTracker initialized (History size per model: {history_size})')

    async def record_request(self, model: str, provider: str) -> None:
        async with self._lock:
            stats = self._model_stats.setdefault(model, ModelPerformanceStats(model_name=model, provider=provider, recent_requests=deque(maxlen=self.history_size)))
            stats.request_count += 1
            stats.last_request_time_ms = get_current_time_ms()
        metrics.track_llm('requests', model=model, provider=provider)
        logger.debug(f'Recorded start of request for model {model}')

    async def record_success(self, model: str, provider: str, latency_ms: int, prompt_tokens: int, completion_tokens: int, context_labels: Optional[Dict[str, str]]=None) -> None:
        async with self._lock:
            stats = self._model_stats.setdefault(model, ModelPerformanceStats(model_name=model, provider=provider, recent_requests=deque(maxlen=self.history_size)))
            stats.success_count += 1
            stats.total_latency_ms += latency_ms
            stats.total_prompt_tokens += prompt_tokens
            stats.total_completion_tokens += completion_tokens
            stats.recent_requests.append((get_current_time_ms(), latency_ms, True))
            
        metrics.track_llm('duration', model=model, provider=provider, value=latency_ms / 1000.0, **(context_labels or {}))
        metrics.track_llm('tokens', model=model, provider=provider, type='prompt', value=prompt_tokens, **(context_labels or {}))
        metrics.track_llm('tokens', model=model, provider=provider, type='completion', value=completion_tokens, **(context_labels or {}))
        logger.debug(f'Recorded successful response for model {model} (Latency: {latency_ms}ms)')

    async def record_failure(self, model: str, provider: str, error_type: str, latency_ms: Optional[int]=None, context_labels: Optional[Dict[str, str]]=None) -> None:
        current_time_ms = get_current_time_ms()
        async with self._lock:
            stats = self._model_stats.setdefault(model, ModelPerformanceStats(model_name=model, provider=provider, recent_requests=deque(maxlen=self.history_size)))
            stats.error_count += 1
            stats.last_error_time_ms = current_time_ms
            failure_latency = latency_ms if latency_ms is not None else 0
            stats.recent_requests.append((current_time_ms, failure_latency, False))
        metrics.track_llm('errors', model=model, provider=provider, error_type=error_type, **(context_labels or {}))
        logger.debug(f'Recorded failed response for model {model} (Error: {error_type})')

    async def get_performance_stats(self, model: str) -> Optional[ModelPerformanceStats]:
        async with self._lock:
            stats = self._model_stats.get(model)
            if stats:
                return stats.model_copy(deep=True)
            else:
                return None

    async def get_all_performance_stats(self) -> Dict[str, ModelPerformanceStats]:
        async with self._lock:
            return {name: stats.model_copy(deep=True) for name, stats in self._model_stats.items()}

    async def reset_stats(self, model: Optional[str]=None) -> None:
        async with self._lock:
            if model:
                if model in self._model_stats:
                    provider = self._model_stats[model].provider
                    self._model_stats[model] = ModelPerformanceStats(model_name=model, provider=provider, recent_requests=deque(maxlen=self.history_size))
                    logger.info(f'Reset performance stats for model: {model}')
                else:
                    logger.warning(f'Cannot reset stats: Model {model} not found in tracker.')
            else:
                logger.info('Resetting performance stats for all models.')
                self._model_stats.clear()
_tracker_instance: Optional[LLMPerformanceTracker] = None
_tracker_lock = asyncio.Lock()

async def get_performance_tracker(history_size: int=100) -> LLMPerformanceTracker:
    global _tracker_instance
    if _tracker_instance is not None:
        return _tracker_instance
    async with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = LLMPerformanceTracker(history_size=history_size)
            logger.info('Singleton LLMPerformanceTracker instance created.')
    if _tracker_instance is None:
        raise RuntimeError('Failed to create LLMPerformanceTracker instance.')
    return _tracker_instance

async def record_llm_success(model: str, provider: str, latency_ms: int, prompt_tokens: int, completion_tokens: int, context_labels: Optional[Dict[str, str]]=None):
    try:
        tracker = await get_performance_tracker()
        await tracker.record_success(model, provider, latency_ms, prompt_tokens, completion_tokens, context_labels)
    except Exception as e:
        logger.error(f'Failed to record LLM success metric for model {model}: {e}', exc_info=True)

async def record_llm_failure(model: str, provider: str, error_type: str, latency_ms: Optional[int]=None, context_labels: Optional[Dict[str, str]]=None):
    try:
        tracker = await get_performance_tracker()
        await tracker.record_failure(model, provider, error_type, latency_ms, context_labels)
    except Exception as e:
        logger.error(f'Failed to record LLM failure metric for model {model}: {e}', exc_info=True)
from collections import deque
from pydantic import BaseModel, Field, ConfigDict
import os
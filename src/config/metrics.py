"""
Performance metrics collection for the Multi-Agent Platform.
Uses Prometheus for metrics collection with optimized performance.
"""
import asyncio
import functools
import threading
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.exposition import start_http_server

from src.config.logger import get_logger
from src.config.settings import get_settings

settings = get_settings()
logger = get_logger(__name__)

# Define type variables for better type hinting
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Registry for all metrics
REGISTRY = prometheus_client.REGISTRY

# System information
SYSTEM_INFO = Gauge('system_info', 'System information', ['app_name', 'app_version', 'environment'])

# HTTP metrics
HTTP_REQUESTS_TOTAL = Counter('http_requests_total', 'Total number of HTTP requests', 
                              ['method', 'endpoint', 'status_code'])
HTTP_REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration in seconds', 
                                 ['method', 'endpoint', 'status_code'], 
                                 buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2, 5, 10, float('inf')))
HTTP_REQUEST_SIZE = Histogram('http_request_size_bytes', 'HTTP request size in bytes', 
                             ['method', 'endpoint'], 
                             buckets=(100, 1000, 10000, 100000, 1000000, float('inf')))
HTTP_RESPONSE_SIZE = Histogram('http_response_size_bytes', 'HTTP response size in bytes', 
                              ['method', 'endpoint'], 
                              buckets=(100, 1000, 10000, 100000, 1000000, float('inf')))

# Task metrics
TASK_METRICS = {
    'created': Counter('task_created_total', 'Total number of tasks created'),
    'consumed': Counter('task_consumed_total', 'Total number of tasks consumed', ['dispatcher_id']),
    'completed': Counter('task_completed_total', 'Total number of tasks completed', ['status']),
    'duration': Histogram('task_duration_seconds', 'Task processing duration in seconds', 
                         ['status'], 
                         buckets=(0.1, 0.5, 1.0, 2.5, 5, 10, 30, 60, 120, 300, float('inf'))),
    'queue_depth': Gauge('task_queue_depth', 'Current depth of the task queue'),
    'processing': Gauge('task_processing', 'Current number of tasks being processed'),
    'rejections': Counter('task_rejections_total', 'Total number of tasks rejected', ['reason']),
    'retries': Counter('task_retries_total', 'Total number of task retries', ['task_type']),
    'dlq': Counter('task_dlq_total', 'Total number of tasks moved to Dead Letter Queue', ['reason'])
}

# LLM metrics
LLM_METRICS = {
    'requests': Counter('llm_requests_total', 'Total number of LLM API requests', 
                      ['model', 'provider']),
    'duration': Histogram('llm_request_duration_seconds', 'LLM API request duration in seconds', 
                         ['model', 'provider'], 
                         buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, float('inf'))),
    'tokens': Counter('llm_token_usage_total', 'Total number of tokens used', 
                    ['model', 'provider', 'type']),
    'errors': Counter('llm_errors_total', 'Total number of LLM API errors', 
                     ['model', 'provider', 'error_type']),
    'fallbacks': Counter('llm_fallbacks_total', 'Total number of LLM fallbacks', 
                        ['from_model', 'to_model'])
}

# Agent metrics
AGENT_METRICS = {
    'created': Counter('agent_created_total', 'Total number of agents created', ['agent_type']),
    'operations': Counter('agent_operations_total', 'Total number of agent operations', 
                        ['agent_type', 'operation_type']),
    'duration': Histogram('agent_operation_duration_seconds', 'Agent operation duration in seconds', 
                         ['agent_type', 'operation_type'], 
                         buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf'))),
    'errors': Counter('agent_error_total', 'Total number of agent errors', 
                     ['agent_type', 'error_type'])
}

# Tool metrics
TOOL_METRICS = {
    'executions': Counter('tool_executions_total', 'Total number of tool executions', ['tool_name']),
    'duration': Histogram('tool_execution_duration_seconds', 'Tool execution duration in seconds', 
                         ['tool_name'], 
                         buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float('inf'))),
    'errors': Counter('tool_errors_total', 'Total number of tool errors', 
                     ['tool_name', 'error_type'])
}

# Memory and cache metrics
MEMORY_METRICS = {
    'operations': Counter('memory_operations_total', 'Total number of memory operations', 
                        ['operation_type']),
    'duration': Histogram('memory_operation_duration_seconds', 'Memory operation duration in seconds', 
                         ['operation_type'], 
                         buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, float('inf'))),
    'size': Gauge('memory_size_bytes', 'Current memory size in bytes', ['memory_type'])
}

CACHE_METRICS = {
    'operations': Counter('cache_operations_total', 'Total number of cache operations', 
                        ['operation_type']),
    'hits': Counter('cache_hits_total', 'Total number of cache hits', ['cache_type']),
    'misses': Counter('cache_misses_total', 'Total number of cache misses', ['cache_type']),
    'size': Gauge('cache_size_entries', 'Current number of entries in cache', ['cache_type'])
}

REGISTRY_METRICS = {
    'operations': Counter('registry_operations_total', 'Total number of registry operations', 
                        ['registry_name', 'operation_type']),
    'duration': Histogram('registry_operation_duration_seconds', 'Registry operation duration in seconds', 
                         ['operation'], 
                         buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, float('inf'))),
    'size': Gauge('registry_size_entries', 'Current number of entries in registry', ['registry_name'])
}

# Task metric constants for direct access
TASK_CREATED_TOTAL = TASK_METRICS['created']
TASK_CONSUMED_TOTAL = TASK_METRICS['consumed']
TASK_COMPLETED_TOTAL = TASK_METRICS['completed']
TASK_DURATION = TASK_METRICS['duration']
TASK_QUEUE_DEPTH = TASK_METRICS['queue_depth']
TASK_PROCESSING = TASK_METRICS['processing']
TASK_REJECTIONS_TOTAL = TASK_METRICS['rejections']

# LLM metric constants for direct access
LLM_REQUESTS_TOTAL = LLM_METRICS['requests']
LLM_REQUEST_DURATION = LLM_METRICS['duration'] 
LLM_TOKEN_USAGE = LLM_METRICS['tokens']
LLM_ERRORS_TOTAL = LLM_METRICS['errors']

# Agent metric constants for direct access
AGENT_CREATED_TOTAL = AGENT_METRICS['created']
AGENT_OPERATIONS_TOTAL = AGENT_METRICS['operations']
AGENT_OPERATION_DURATION = AGENT_METRICS['duration']
AGENT_ERROR_TOTAL = AGENT_METRICS['errors']

# Memory metric constants for direct access
MEMORY_OPERATIONS_TOTAL = MEMORY_METRICS['operations']
MEMORY_OPERATION_DURATION = MEMORY_METRICS['duration']
MEMORY_SIZE = MEMORY_METRICS['size']

# Registry metric constants for direct access (add this after creating REGISTRY_METRICS as we discussed)
REGISTRY_OPERATIONS_TOTAL = REGISTRY_METRICS['operations']
REGISTRY_OPERATION_DURATION = REGISTRY_METRICS['duration']
REGISTRY_SIZE = REGISTRY_METRICS['size']

class MetricsManager:
    """
    Manager for metrics collection functionality.
    Provides a simplified interface for tracking metrics.
    """
    
    def __init__(self):
        """Initialize the metrics manager."""
        self.settings = get_settings()
        self.enabled = self.settings.METRICS_ENABLED
        self._server_started = False
        self._server_lock = threading.Lock()
    
    def track_http_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, request_size: int, response_size: int) -> None:
        """
        Track an HTTP request metrics.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            status_code: HTTP status code
            duration: Request duration in seconds
            request_size: Size of request in bytes
            response_size: Size of response in bytes
        """
        if not self.enabled:
            return
            
        HTTP_REQUESTS_TOTAL.labels(
            method=method, endpoint=endpoint, status_code=status_code).inc()
        HTTP_REQUEST_DURATION.labels(
            method=method, endpoint=endpoint, status_code=status_code).observe(duration)
        HTTP_REQUEST_SIZE.labels(
            method=method, endpoint=endpoint).observe(request_size)
        HTTP_RESPONSE_SIZE.labels(
            method=method, endpoint=endpoint).observe(response_size)
    
    # 수정된 track_task 메서드 로직 (전체 메서드를 아래 내용으로 교체)
    def track_task(self, metric_name: str, **labels) -> None:
        """
        Track a task metric.

        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric. For Counters, 'value' key will be used for inc amount.
        """
        if not self.enabled:
            return

        if metric_name not in TASK_METRICS:
            logger.warning(f"Unknown task metric: {metric_name}")
            return

        metric = TASK_METRICS[metric_name]
        value_for_inc = None

        # Counter 타입의 경우, 'value' 레이블을 inc() 인자로 사용하기 위해 분리
        if isinstance(metric, Counter) and 'value' in labels:
             # labels 딕셔너리에서 'value'를 제거하고 그 값을 저장
             # .pop()은 해당 키가 없으면 에러를 발생시키므로 안전하게 .get()과 del 사용 가능
             # 또는 기본값 1을 사용하려면 value_for_inc = labels.pop('value', 1) 사용
             value_for_inc = labels.pop('value')

        try:
            if isinstance(metric, Gauge):
                if metric_name == 'processing':
                    # 'processing' 게이지는 특별 처리 (증가/감소)
                    if 'increment' in labels and labels.pop('increment'):
                        metric.inc()
                    else:
                        metric.dec()
                else:
                    # 다른 게이지는 값 설정
                    value = labels.pop('value', 0)
                    if labels:
                        metric.labels(**labels).set(value)
                    else:
                        metric.set(value)
            elif isinstance(metric, Histogram):
                # 히스토그램은 값 관찰
                value = labels.pop('value', 0)
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
            elif isinstance(metric, Counter):
                # 카운터는 값 증가 (분리된 value 사용)
                labeled_metric = metric.labels(**labels) if labels else metric
                if value_for_inc is not None:
                    # value_for_inc는 float일 수 있으므로 int가 필요하면 변환
                    labeled_metric.inc(float(value_for_inc))
                else:
                    # value가 제공되지 않으면 1 증가
                    labeled_metric.inc()
            else:
                logger.warning(f"Unhandled metric type for {metric_name}: {type(metric)}")

        except (ValueError, TypeError) as e:
            # Prometheus 클라이언트 라이브러리에서 발생하는 레이블 관련 오류 포함
            logger.warning(f"Error processing metric {metric_name} with labels {labels}: {e}")
        except Exception as e:
            # 기타 예외 처리
            logger.exception(f"Unexpected error processing metric {metric_name}: {e}")
    
    def track_llm(self, metric_name: str, **labels) -> None:
        """
        Track an LLM metric.
        
        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric
        """
        if not self.enabled:
            return
            
        if metric_name not in LLM_METRICS:
            logger.warning(f"Unknown LLM metric: {metric_name}")
            return
            
        metric = LLM_METRICS[metric_name]
        
        if isinstance(metric, Histogram):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        elif isinstance(metric, Counter) and 'value' in labels:
            value = labels.pop('value')
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
        else:
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
    
    def track_agent(self, metric_name: str, **labels) -> None:
        """
        Track an agent metric.
        
        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric
        """
        if not self.enabled:
            return
            
        if metric_name not in AGENT_METRICS:
            logger.warning(f"Unknown agent metric: {metric_name}")
            return
            
        metric = AGENT_METRICS[metric_name]
        
        if isinstance(metric, Histogram):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        else:
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
    
    def track_tool(self, metric_name: str, **labels) -> None:
        """
        Track a tool metric.
        
        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric
        """
        if not self.enabled:
            return
            
        if metric_name not in TOOL_METRICS:
            logger.warning(f"Unknown tool metric: {metric_name}")
            return
            
        metric = TOOL_METRICS[metric_name]
        
        if isinstance(metric, Histogram):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        else:
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
    
    def track_memory(self, metric_name: str, **labels) -> None:
        """
        Track a memory metric.
        
        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric
        """
        if not self.enabled:
            return
            
        if metric_name not in MEMORY_METRICS:
            logger.warning(f"Unknown memory metric: {metric_name}")
            return
            
        metric = MEMORY_METRICS[metric_name]
        
        if isinstance(metric, Gauge):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        elif isinstance(metric, Histogram):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        else:
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
    
    def track_cache(self, metric_name: str, **labels) -> None:
        """
        Track a cache metric.
        
        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric
        """
        if not self.enabled:
            return
            
        if metric_name not in CACHE_METRICS:
            logger.warning(f"Unknown cache metric: {metric_name}")
            return
            
        metric = CACHE_METRICS[metric_name]
        
        if isinstance(metric, Gauge):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        else:
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
    
    def timed_metric(self, metric: Union[Histogram, Summary], 
                     labels: Optional[Dict[str, str]] = None,
                     count_exceptions: bool = False) -> Callable[[F], F]:
        """
        Decorator for timing functions and recording the execution time.
        
        Args:
            metric: The histogram or summary metric to use
            labels: Optional labels for the metric
            count_exceptions: Whether to record timing even if an exception is raised
            
        Returns:
            Callable: Decorator function
        """
        def decorator(func: F) -> F:
            # Skip instrumentation if metrics are disabled
            if not self.enabled:
                return func
                
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.monotonic()
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception:
                        if count_exceptions:
                            duration = time.monotonic() - start_time
                            if labels:
                                metric.labels(**labels).observe(duration)
                            else:
                                metric.observe(duration)
                        raise
                    finally:
                        if count_exceptions or 'result' in locals():
                            duration = time.monotonic() - start_time
                            if labels:
                                metric.labels(**labels).observe(duration)
                            else:
                                metric.observe(duration)
                                
                return async_wrapper  # type: ignore
            else:
                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    start_time = time.monotonic()
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception:
                        if count_exceptions:
                            duration = time.monotonic() - start_time
                            if labels:
                                metric.labels(**labels).observe(duration)
                            else:
                                metric.observe(duration)
                        raise
                    finally:
                        if count_exceptions or 'result' in locals():
                            duration = time.monotonic() - start_time
                            if labels:
                                metric.labels(**labels).observe(duration)
                            else:
                                metric.observe(duration)
                                
                return sync_wrapper  # type: ignore
                
        return decorator
    
    def start_metrics_server(self) -> Optional[threading.Thread]:
        """
        Start the Prometheus metrics server if enabled.
        
        Returns:
            Optional[threading.Thread]: Server thread if started, None otherwise
        """
        if not self.enabled:
            logger.info("Metrics collection is disabled")
            return None
            
        with self._server_lock:
            if self._server_started:
                logger.info("Metrics server is already running")
                return None
                
            try:
                # Set system info metric
                SYSTEM_INFO.labels(
                    app_name=self.settings.APP_NAME,
                    app_version=self.settings.APP_VERSION,
                    environment=self.settings.ENVIRONMENT
                ).set(1)
                
                # Start HTTP server in a separate thread
                thread = threading.Thread(
                    target=start_http_server,
                    args=(self.settings.METRICS_PORT,),
                    daemon=True
                )
                thread.start()
                self._server_started = True
                
                logger.info(f"Prometheus metrics server started on port {self.settings.METRICS_PORT}")
                return thread
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}", exc_info=True)
                return None
            
    def track_registry(self, metric_name: str, **labels) -> None:
        """
        Track a registry metric.
        
        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric
        """
        if not self.enabled:
            return
            
        if metric_name not in REGISTRY_METRICS:
            logger.warning(f"Unknown registry metric: {metric_name}")
            return
            
        metric = REGISTRY_METRICS[metric_name]
        
        if isinstance(metric, Gauge):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        elif isinstance(metric, Histogram):
            value = labels.pop('value', 0)
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        else:
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()


# Singleton instance
_metrics_manager: Optional[MetricsManager] = None
_manager_lock = threading.Lock()


def get_metrics_manager() -> MetricsManager:
    """
    Get the metrics manager singleton instance.
    
    Returns:
        MetricsManager: The metrics manager instance
    """
    global _metrics_manager
    
    if _metrics_manager is None:
        with _manager_lock:
            if _metrics_manager is None:
                _metrics_manager = MetricsManager()
                
    return _metrics_manager


# Convenience functions for metrics tracking
def track_http_request(method: str, endpoint: str, status_code: int, 
                       duration: float, request_size: int, response_size: int) -> None:
    """Track an HTTP request."""
    get_metrics_manager().track_http_request(
        method, endpoint, status_code, duration, request_size, response_size)


def track_task_created() -> None:
    """Track a task creation."""
    get_metrics_manager().track_task('created')


def track_task_consumed(dispatcher_id: str) -> None:
    """Track a task consumption."""
    get_metrics_manager().track_task('consumed', dispatcher_id=dispatcher_id)


def track_task_started() -> None:
    """Track a task start."""
    get_metrics_manager().track_task('processing', increment=True)


def track_task_completed(status: str, duration: float) -> None:
    """Track a task completion."""
    get_metrics_manager().track_task('completed', status=status)
    get_metrics_manager().track_task('duration', status=status, value=duration)
    get_metrics_manager().track_task('processing', increment=False)


def timed_metric(metric: Union[Histogram, Summary], 
                 labels: Optional[Dict[str, str]] = None,
                 count_exceptions: bool = False) -> Callable[[F], F]:
    """Decorator for timing a function and recording the execution time."""
    return get_metrics_manager().timed_metric(metric, labels, count_exceptions)


def start_metrics_server() -> Optional[threading.Thread]:
    """Start the Prometheus metrics server."""
    return get_metrics_manager().start_metrics_server()

def track_registry_operation(registry_name: str, operation_type: str) -> None:
    """Track a registry operation."""
    get_metrics_manager().track_registry('operations', registry_name=registry_name, operation_type=operation_type)

def track_registry_size(registry_name: str, size: int) -> None:
    """Track the size of a registry."""
    get_metrics_manager().track_registry('size', registry_name=registry_name, value=size)
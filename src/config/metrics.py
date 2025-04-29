"""
Performance metrics collection for the Multi-Agent Platform.
Uses Prometheus for metrics collection with optimized performance.
"""
import time
import functools
import threading
import asyncio
from typing import Callable, Any, Dict, Optional, List, Union, Set, TypeVar

import prometheus_client
from prometheus_client import Histogram, Counter, Gauge, Summary
from prometheus_client.exposition import start_http_server

from src.config.settings import get_settings
from src.config.logger import get_logger

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
    
    def track_task(self, metric_name: str, **labels) -> None:
        """
        Track a task metric.
        
        Args:
            metric_name: Name of the metric to track
            **labels: Labels for the metric
        """
        if not self.enabled:
            return
            
        if metric_name not in TASK_METRICS:
            logger.warning(f"Unknown task metric: {metric_name}")
            return
            
        metric = TASK_METRICS[metric_name]
        
        if isinstance(metric, Gauge):
            if metric_name == 'processing':
                if 'increment' in labels and labels.pop('increment'):
                    metric.inc()
                else:
                    metric.dec()
            else:
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
            try:
                # Counter는 먼저 labels()를 호출한 후 inc()를 호출해야 함
                if labels:
                    metric.labels(**labels).inc()
                else:
                    metric.inc()
            except TypeError as e:
                logger.warning(f"Error incrementing metric {metric_name}: {e}")
    
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
                    except Exception as e:
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
                    except Exception as e:
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
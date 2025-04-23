import time
import functools
import threading
from typing import Callable, Any, Dict, Optional, List, Union

import prometheus_client
from prometheus_client import Histogram, Counter, Gauge
from prometheus_client.exposition import start_http_server

from src.config.settings import get_settings
from src.config.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

REGISTRY = prometheus_client.REGISTRY

SYSTEM_INFO = Gauge(
    "system_info",
    "System information",
    ["app_name", "app_version", "environment"]
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_request_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2, 5, 10, float("inf"))
)

HTTP_REQUEST_SIZE = Histogram(
    "http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1_000, 10_000, 100_000, 1_000_000, float("inf"))
)

HTTP_RESPONSE_SIZE = Histogram(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    buckets=(100, 1_000, 10_000, 100_000, 1_000_000, float("inf"))
)

TASK_CREATED_TOTAL = Counter(
    "task_created_total",
    "Total number of tasks created",
)

TASK_COMPLETED_TOTAL = Counter(
    "task_completed_total",
    "Total number of tasks completed",
    ["status"]
)

TASK_DURATION = Histogram(
    "task_duration_seconds",
    "Task processing duration in seconds",
    ["status"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5, 10, 30, 60, 120, 300, float("inf"))
)

TASK_QUEUE_DEPTH = Gauge(
    "task_queue_depth",
    "Current depth of the task in queue",
)

TASK_PROCESSING = Gauge(
    "task_processing",
    "Current number of tasks being processed",
)

LLM_REQUESTS_TOTAL = Counter(
    "llm_request_total",
    "Total number of LLM API requests",
    ["model", "provider"]
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM API request duration in seconds",
    ["model", "provider"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, float("inf"))
)

LLM_TOKEN_USAGE = Counter(
    "llm_token_usage",
    "Total number of tokens used in LLM API requests",
    ["model", "provider", "type"]
)

LLM_ERRORS_TOTAL = Counter(
    "LLM_ERRORS_TOTAL",
    "Total number of LLM API errors",
    ["model", "provider", "error_type"]
)

LLM_FALLBACKS_TOTAL = Counter(
    "llm_fallbacks_total",
    "Total number of LLM fallbacks triggered",
    ["from_model", "to_model"]
)

AGENT_CREATED_TOTAL = Counter(
    "agent_created_total",
    "Total number of agents created",
    ["agent_type"]
)

AGENT_OPERATIONS_TOTAL = Counter(
    "agent_operations_total",
    "Total number of agent operations",
    ["agent_type", "operation_type"]
)

AGENT_OPERATION_DURATION = Histogram(
    "agent_operation_duration_seconds",
    "Agent operation duration in seconds",
    ["agent_type", "operation_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float("inf"))
)

AGENT_ERROR_TOTAL = Counter(
    "agent_error_total",
    "Total number of agent errors",
    ["agent_type", "error_type"]
)

TOOL_EXECUTIONS_TOTAL = Counter(
    "tool_executions_total",
    "Total number of tool executions",
    ["tool_name"]
)

TOOL_EXECUTION_DURATION = Histogram(
    "tool_execution_duration_seconds",
    "Tool execution duration in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float("inf"))
)

TOOL_ERRORS_TOTAL = Counter(
    "TOOL_ERRORS_TOTAL",
    "Total number of tool errors",
    ["tool_name", "error_type"]
)

MEMORY_OPERATIONS_TOTAL = Counter(
    "memory_operations_total",
    "Total number of memory operations",
    ["operation_type"]
)

MEMORY_OPERATION_DURATION = Histogram(
    "memory_operation_duration_seconds",
    "Memory operation duration in seconds",
    ["operation_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, float("inf"))
)

MEMORY_SIZE = Gauge(
    "memory_size_bytes",
    "Current memory size in bytes",
    ["memory_type"]
)

CACHE_OPERATIONS_TOTAL = Counter(
    "cache_operations_total",
    "Total number of cache operations",
    ["operation_type"]
)

CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["cache_type"]
)

CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["cache_type"]
)

CACHE_SIZE = Gauge(
    "cache_size_entries",
    "Current number of entries in cache",
    ["cache_type"]
)

def track_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float,
    request_size: int,
    response_size: int
):
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    ).inc()

    HTTP_REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    ).observe(duration)

    HTTP_REQUEST_SIZE.labels(
        method=method,
        endpoint=endpoint
    ).observe(request_size)

    HTTP_RESPONSE_SIZE.labels(
        method=method,
        endpoint=endpoint
    ).observe(response_size)
    
def track_task_created():
    TASK_CREATED_TOTAL.inc()
    TASK_QUEUE_DEPTH.inc()
    
def track_task_started():
    TASK_QUEUE_DEPTH.dec()
    TASK_PROCESSING.inc()
    
def track_task_completed(status: str, duration: float):
    TASK_COMPLETED_TOTAL.labels(status=status).inc()
    TASK_DURATION.labels(status=status).observe(duration)
    TASK_PROCESSING.dec()
    
def track_llm_request(model: str, provider: str):
    LLM_REQUESTS_TOTAL.labels(model=model, provider=provider).inc()
    
def track_llm_response(model: str, provider: str, duration: float, prompt_tokens: int, completion_tokens: int):
    LLM_REQUEST_DURATION.labels(model=model, provider=provider).observe(duration)
    prompt_counter = LLM_TOKEN_USAGE.labels(model=model, provider=provider, type="prompt")
    prompt_counter.inc(prompt_tokens)
    completion_counter = LLM_TOKEN_USAGE.labels(model=model, provider=provider, type="completion")
    completion_counter.inc(completion_tokens)

def track_llm_error(model: str, provider: str, error_type: str):
    LLM_ERRORS_TOTAL.labels(model=model, provider=provider, error_type=error_type).inc()
    
def track_llm_fallback(from_model: str, to_model: str):
    LLM_FALLBACKS_TOTAL.labels(from_model=from_model, to_model=to_model).inc()
    
def track_agent_created(agent_type: str):
    AGENT_CREATED_TOTAL.labels(agent_type=agent_type).inc()
    
def track_agent_operation(agent_type: str, operation_type: str):
    AGENT_OPERATIONS_TOTAL.labels(agent_type=agent_type, operation_type=operation_type).inc()

def track_agent_operation_completed(agent_type: str, operation_type: str, duration: float):
    AGENT_OPERATION_DURATION.labels(agent_type=agent_type, operation_type=operation_type).observe(duration)

def track_agent_error(agent_type: str, error_type: str):
    AGENT_ERROR_TOTAL.labels(agent_type=agent_type, error_type=error_type).inc()
    
def track_tool_execution(tool_name: str):
    TOOL_EXECUTIONS_TOTAL.labels(tool_name=tool_name).inc()
    
def track_tool_execution_completed(tool_name: str, duration: float):
    TOOL_EXECUTION_DURATION.labels(tool_name=tool_name).observe(duration)
    
def track_tool_error(tool_name: str, error_type: str):
    TOOL_ERRORS_TOTAL.labels(tool_name=tool_name, error_type=error_type).inc()
    
def track_memory_operation(operation_type: str):
    MEMORY_OPERATIONS_TOTAL.labels(operation_type=operation_type).inc()
    
def track_memory_operation_completed(operation_type: str, duration: float):
    MEMORY_OPERATION_DURATION.labels(operation_type=operation_type).observe(duration)

def track_memory_size(memory_type: str, size_bytes: int):
    MEMORY_SIZE.labels(memory_type=memory_type).set(size_bytes)

def track_cache_operation(operation_type: str):
    CACHE_OPERATIONS_TOTAL.labels(operation_type=operation_type).inc()
    
def track_cache_hit(cache_type: str):
    CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()
    
def track_cache_miss(cache_type: str):
    CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc()
    
def track_cache_size(cache_type: str, size_entries: int):
    CACHE_SIZE.labels(cache_type=cache_type).set(size_entries)
    
def timed_metric(
    metric: Histogram,
    labels: Optional[Dict[str, str]] = None,
) -> Callable:
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator
            
_metrics_server = None
            
def start_metrics_server():
    global _metrics_server
    if settings.METRICS_ENABLED and _metrics_server is None:
        try:
            SYSTEM_INFO.labels(
                app_name=settings.APP_NAME,
                app_version=settings.APP_VERSION,
                environment=settings.ENVIRONMENT
            ).set(1)
            
            _metrics_server = threading.Thread(
                target=prometheus_client.start_http_server,
                args=(settings.METRICS_PORT,),
                daemon=True
            )
            _metrics_server.start()
            logger.info(f"Metrics server started on port {settings.METRICS_PORT}")
            return _metrics_server
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    return None
# src/utils/logger.py
"""
Logging configuration for the Multi-Agent Platform.
Provides structured logging with OpenTelemetry context propagation.
"""
import json
import logging
import sys
import uuid # ContextLoggerAdapter에서 trace_id 기본값 생성용으로 유지
from datetime import datetime, timezone
from typing import Optional, Set, Dict, Any

# OpenTelemetry
from opentelemetry import trace

# Import settings directly to avoid circular imports
# This works because we're not creating a settings instance here
# from src.config.settings import get_settings # setup_logging에서 직접 AppSettings 사용
from src.schemas.config import AppSettings # AppSettings 직접 임포트

# Cache of standard LogRecord attributes to avoid repeated lookups
STANDARD_LOGRECORD_ATTRIBUTES: Set[str] = {
    'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
    'funcName', 'levelname', 'levelno', 'lineno', 'module',
    'msecs', 'message', 'msg', 'name', 'pathname', 'process',
    'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName',
    # OTel 필드도 표준으로 간주하여 extra_attrs에서 제외
    'otel_trace_id', 'otel_span_id'
}


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Includes OpenTelemetry trace_id and span_id if available.
    """
    def __init__(self, include_thread_info: bool = False, include_process_info: bool = False):
        super().__init__()
        self.include_thread_info = include_thread_info
        self.include_process_info = include_process_info

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'lineno': record.lineno,
        }

        # Add OpenTelemetry Trace and Span ID if available on the record
        # These are added by the custom LogRecordFactory
        if hasattr(record, 'otel_trace_id') and record.otel_trace_id:
            log_data['otel_trace_id'] = record.otel_trace_id
        if hasattr(record, 'otel_span_id') and record.otel_span_id:
            log_data['otel_span_id'] = record.otel_span_id

        # Conditionally include thread/process info
        if self.include_thread_info:
            log_data['thread'] = record.thread
            log_data['threadName'] = record.threadName

        if self.include_process_info:
            log_data['process'] = record.process
            log_data['processName'] = record.processName

        # Exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        # Contextual IDs (from ContextLoggerAdapter or extra)
        # 'trace_id'는 OTel ID와 중복될 수 있으므로, OTel ID를 우선.
        # 여기서는 LogRecordFactory를 통해 otel_trace_id가 설정되므로,
        # record에 직접 있는 trace_id, task_id, agent_id 등을 사용.
        if hasattr(record, 'trace_id') and 'otel_trace_id' not in log_data: # OTel ID 없을 때만 기존 trace_id 사용
            log_data['trace_id'] = record.trace_id
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id
        if hasattr(record, 'agent_id'):
            log_data['agent_id'] = record.agent_id

        # Performance metrics
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time

        # Add any custom attributes from 'extra_attrs' or other non-standard fields
        extra_attrs = getattr(record, 'extra_attrs', None)
        if extra_attrs and isinstance(extra_attrs, dict):
            log_data.update(extra_attrs)
        else:
            for key, value in record.__dict__.items():
                if (key not in STANDARD_LOGRECORD_ATTRIBUTES and
                    not key.startswith('_') and
                    key not in log_data):
                    log_data[key] = value

        try:
            return json.dumps(log_data, default=str, separators=(',', ':'))
        except Exception as e:
            return json.dumps({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': 'ERROR',
                'name': 'logger.JsonFormatter',
                'message': f'Error serializing log record: {str(e)}',
                'original_message': str(record.getMessage())
            }, default=str)

# TraceLogger 클래스는 제거. setLogRecordFactory로 대체하여 모든 로거에 OTel 컨텍스트 주입.
# class TraceLogger(logging.Logger): ...

class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log records.
    Preserves context across multiple log calls.
    OpenTelemetry trace/span IDs are handled by the LogRecordFactory.
    """
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        # Add context fields to extra. 'extra_attrs' is used by JsonFormatter.
        # 'trace_id' from extra will be overridden by otel_trace_id if present.
        extra_attrs_dict = kwargs['extra'].get('extra_attrs', {})
        if not isinstance(extra_attrs_dict, dict): # 혹시 extra_attrs가 dict가 아니면 초기화
            extra_attrs_dict = {}

        if self.extra:
            for k, v in self.extra.items():
                # OTel ID와 중복될 수 있는 trace_id는 LogRecordFactory에서 처리.
                # 여기서는 task_id, agent_id 및 기타 커스텀 컨텍스트를 extra_attrs에 추가.
                if k in ('task_id', 'agent_id'): # 이들은 직접 extra에 넣어서 JsonFormatter가 바로 사용하도록
                    kwargs['extra'][k] = v
                elif k != 'trace_id': # OTel trace_id와 충돌 방지
                    extra_attrs_dict[k] = v

        if extra_attrs_dict: # 내용이 있을 때만 extra_attrs 추가
            kwargs['extra']['extra_attrs'] = extra_attrs_dict

        return msg, kwargs

_logging_setup_complete = False

def setup_logging(settings_obj: Optional[AppSettings] = None) -> None:
    """
    Set up logging for the application. Includes OpenTelemetry trace/span ID injection.
    Args:
        settings_obj: Optional settings instance. If None, will use environment variables.
    """
    global _logging_setup_complete
    if _logging_setup_complete:
        logging.getLogger(__name__).debug("Logging already configured.")
        return

    if settings_obj is None:
        from src.config.settings import get_settings as get_app_settings # 순환참조 피하기 위해 함수 내에서 임포트
        settings_obj = get_app_settings()

    # --- LogRecordFactory 설정: OTel Trace/Span ID 주입 ---
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        
        record.otel_trace_id = ""
        record.otel_span_id = ""
        
        try:
            current_span = trace.get_current_span()
            
            if current_span and hasattr(current_span, 'get_span_context'):
                ctx = current_span.get_span_context()
                if ctx and ctx.is_valid:
                    record.otel_trace_id = format(ctx.trace_id, '032x')
                    record.otel_span_id = format(ctx.span_id, '016x')
                    
                    import sys
                    sys.stderr.write(f"[OTEL-DEBUG] Added trace_id={record.otel_trace_id} to LogRecord '{record.name}'\n")
                    sys.stderr.flush()
        except Exception as e:
            import sys
            sys.stderr.write(f"[OTEL-ERROR] Failed to add trace ID: {str(e)}\n")
            sys.stderr.flush()
        
        return record


    logging.setLogRecordFactory(record_factory)
    # --- LogRecordFactory 설정 끝 ---

    # TraceLogger 클래스를 사용하지 않으므로 logging.setLoggerClass 부분 제거
    # logging.setLoggerClass(TraceLogger)

    root_logger = logging.getLogger()
    log_level_name = getattr(settings_obj, 'LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    log_format = getattr(settings_obj, 'LOG_FORMAT', 'json')

    if log_format == 'json':
        include_process_info = log_level_name.upper() == 'DEBUG'
        include_thread_info = log_level_name.upper() in ('DEBUG', 'INFO')
        formatter = JsonFormatter(
            include_thread_info=include_thread_info,
            include_process_info=include_process_info
        )
    else: # text format
        # 텍스트 포맷에도 OTel ID 포함 (LogRecordFactory에서 record.otel_trace_id 등으로 설정됨)
        text_format_str = '%(asctime)s - %(name)s - %(levelname)s - [%(otel_trace_id)s:%(otel_span_id)s] - %(message)s'
        formatter = logging.Formatter(text_format_str, datefmt='%Y-%m-%d %H:%M:%S')

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if getattr(settings_obj, 'LOG_TO_FILE', False) and getattr(settings_obj, 'LOG_FILE_PATH', None):
        try:
            file_handler = logging.FileHandler(settings_obj.LOG_FILE_PATH)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            # root_logger.info(f'Log file configured at {settings_obj.LOG_FILE_PATH}') # setup_logging 완료 후 로깅
        except Exception as e:
            root_logger.error(f'Failed to set up file logging to {settings_obj.LOG_FILE_PATH}: {e}')

    _configure_library_loggers()

    # 로깅 설정 완료 후 로그 메시지 출력
    # 이 시점에는 LogRecordFactory가 적용된 로거 사용
    final_logger = logging.getLogger(__name__) # get_logger 대신 logging.getLogger 사용
    final_logger.info(f'Logging setup complete. Level: {log_level_name}, Format: {log_format}')
    if getattr(settings_obj, 'LOG_TO_FILE', False) and getattr(settings_obj, 'LOG_FILE_PATH', None) and any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        final_logger.info(f'Log file configured at {settings_obj.LOG_FILE_PATH}')

    _logging_setup_complete = True


def _configure_library_loggers() -> None:
    """Configure third-party library loggers to appropriate levels."""
    # 이 함수는 settings_obj 에 직접 의존하지 않으므로 그대로 사용 가능
    loggers_to_configure = {
        'uvicorn': logging.WARNING,
        'uvicorn.access': logging.WARNING,
        'uvicorn.error': logging.ERROR,
        'fastapi': logging.WARNING,
        'aiohttp': logging.WARNING,
        'redis': logging.WARNING,
        'httpx': logging.WARNING, # httpx 로거 추가 (LangChain 등에서 사용)
        'openai': logging.WARNING, # OpenAI SDK 로거 추가
        'anthropic': logging.WARNING, # Anthropic SDK 로거 추가
        'langchain': logging.INFO, # LangChain 관련 로깅 레벨 (필요시 조정)
        'langgraph': logging.INFO, # LangGraph 관련 로깅 레벨
        'opentelemetry': logging.INFO, # OpenTelemetry SDK 자체 로깅 레벨
    }
    for logger_name, level in loggers_to_configure.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    Ensures logging is set up before returning a logger.
    """
    # 최초 호출 시 logging 설정이 되어 있지 않으면 자동으로 수행
    if not _logging_setup_complete:
        setup_logging()
    return logging.getLogger(name)


def get_logger_with_context(
    name: str,
    trace_id: Optional[str] = None, # 이 trace_id는 OTel ID와 별개로 사용자가 명시적으로 전달하는 컨텍스트 ID로 간주
    task_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **extra
) -> ContextLoggerAdapter:
    """
    Get a logger with context data attached.
    OTel trace/span IDs are automatically added by LogRecordFactory.
    The 'trace_id' parameter here is for application-specific tracing context,
    distinct from OTel's trace_id.
    """
    logger_instance = get_logger(name)
    context: Dict[str, Any] = {}

    # 사용자가 명시적으로 전달한 trace_id (OTel의 trace_id와 다를 수 있음)
    if trace_id:
        context['app_trace_id'] = trace_id # OTel ID와 구분하기 위해 이름 변경 가능

    if task_id:
        context['task_id'] = task_id
    if agent_id:
        context['agent_id'] = agent_id

    context.update(extra)
    return ContextLoggerAdapter(logger_instance, context)
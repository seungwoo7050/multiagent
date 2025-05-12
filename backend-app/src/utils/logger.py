import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, Set, Dict, Any

from opentelemetry import trace

from src.schemas.config import AppSettings

STANDARD_LOGRECORD_ATTRIBUTES: Set[str] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "otel_trace_id",
    "otel_span_id",
}


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Includes OpenTelemetry trace_id and span_id if available.
    """

    def __init__(
        self, include_thread_info: bool = False, include_process_info: bool = False
    ):
        super().__init__()
        self.include_thread_info = include_thread_info
        self.include_process_info = include_process_info

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "lineno": record.lineno,
        }

        if hasattr(record, "otel_trace_id") and record.otel_trace_id:
            log_data["otel_trace_id"] = record.otel_trace_id
        if hasattr(record, "otel_span_id") and record.otel_span_id:
            log_data["otel_span_id"] = record.otel_span_id

        if self.include_thread_info:
            log_data["thread"] = record.thread
            log_data["threadName"] = record.threadName

        if self.include_process_info:
            log_data["process"] = record.process
            log_data["processName"] = record.processName

        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        if hasattr(record, "trace_id") and "otel_trace_id" not in log_data:
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "agent_id"):
            log_data["agent_id"] = record.agent_id

        if hasattr(record, "execution_time"):
            log_data["execution_time"] = record.execution_time

        extra_attrs = getattr(record, "extra_attrs", None)
        if extra_attrs and isinstance(extra_attrs, dict):
            log_data.update(extra_attrs)
        else:
            for key, value in record.__dict__.items():
                if (
                    key not in STANDARD_LOGRECORD_ATTRIBUTES
                    and not key.startswith("_")
                    and key not in log_data
                ):
                    log_data[key] = value

        try:
            return json.dumps(log_data, default=str, separators=(",", ":"))
        except Exception as e:
            return json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "ERROR",
                    "name": "logger.JsonFormatter",
                    "message": f"Error serializing log record: {str(e)}",
                    "original_message": str(record.getMessage()),
                },
                default=str,
            )


class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log records.
    Preserves context across multiple log calls.
    OpenTelemetry trace/span IDs are handled by the LogRecordFactory.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        extra_attrs_dict = kwargs["extra"].get("extra_attrs", {})
        if not isinstance(extra_attrs_dict, dict):
            extra_attrs_dict = {}

        if self.extra:
            for k, v in self.extra.items():
                if k in ("task_id", "agent_id"):
                    kwargs["extra"][k] = v
                elif k != "trace_id":
                    extra_attrs_dict[k] = v

        if extra_attrs_dict:
            kwargs["extra"]["extra_attrs"] = extra_attrs_dict

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
        from src.config.settings import get_settings as get_app_settings

        settings_obj = get_app_settings()

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.otel_trace_id = ""
        record.otel_span_id = ""

        try:
            current_span = trace.get_current_span()

            if current_span and hasattr(current_span, "get_span_context"):
                ctx = current_span.get_span_context()
                if ctx and ctx.is_valid:
                    record.otel_trace_id = format(ctx.trace_id, "032x")
                    record.otel_span_id = format(ctx.span_id, "016x")

                    import sys

                    sys.stderr.write(
                        f"[OTEL-DEBUG] Added trace_id={record.otel_trace_id} to LogRecord '{record.name}'\n"
                    )
                    sys.stderr.flush()
        except Exception as e:
            import sys

            sys.stderr.write(f"[OTEL-ERROR] Failed to add trace ID: {str(e)}\n")
            sys.stderr.flush()

        return record

    logging.setLogRecordFactory(record_factory)
    root_logger = logging.getLogger()
    log_level_name = getattr(settings_obj, "LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    log_format = getattr(settings_obj, "LOG_FORMAT", "json")

    if log_format == "json":
        include_process_info = log_level_name.upper() == "DEBUG"
        include_thread_info = log_level_name.upper() in ("DEBUG", "INFO")
        formatter = JsonFormatter(
            include_thread_info=include_thread_info,
            include_process_info=include_process_info,
        )
    else:
        text_format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(otel_trace_id)s:%(otel_span_id)s] - %(message)s"
        formatter = logging.Formatter(text_format_str, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if getattr(settings_obj, "LOG_TO_FILE", False) and getattr(
        settings_obj, "LOG_FILE_PATH", None
    ):
        try:
            file_handler = logging.FileHandler(settings_obj.LOG_FILE_PATH)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        except Exception as e:
            root_logger.error(
                f"Failed to set up file logging to {settings_obj.LOG_FILE_PATH}: {e}"
            )

    _configure_library_loggers()
    final_logger = logging.getLogger(__name__)
    final_logger.info(
        f"Logging setup complete. Level: {log_level_name}, Format: {log_format}"
    )
    if (
        getattr(settings_obj, "LOG_TO_FILE", False)
        and getattr(settings_obj, "LOG_FILE_PATH", None)
        and any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
    ):
        final_logger.info(f"Log file configured at {settings_obj.LOG_FILE_PATH}")

    _logging_setup_complete = True


def _configure_library_loggers() -> None:
    """Configure third-party library loggers to appropriate levels."""

    loggers_to_configure = {
        "uvicorn": logging.WARNING,
        "uvicorn.access": logging.WARNING,
        "uvicorn.error": logging.ERROR,
        "fastapi": logging.WARNING,
        "aiohttp": logging.WARNING,
        "redis": logging.WARNING,
        "httpx": logging.WARNING,
        "openai": logging.WARNING,
        "anthropic": logging.WARNING,
        "langchain": logging.INFO,
        "langgraph": logging.INFO,
        "opentelemetry": logging.INFO,
    }
    for logger_name, level in loggers_to_configure.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    Ensures logging is set up before returning a logger.
    """

    if not _logging_setup_complete:
        setup_logging()
    return logging.getLogger(name)


def get_logger_with_context(
    name: str,
    trace_id: Optional[str] = None,
    task_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **extra,
) -> ContextLoggerAdapter:
    """
    Get a logger with context data attached.
    OTel trace/span IDs are automatically added by LogRecordFactory.
    The 'trace_id' parameter here is for application-specific tracing context,
    distinct from OTel's trace_id.
    """
    logger_instance = get_logger(name)
    context: Dict[str, Any] = {}

    if trace_id:
        context["app_trace_id"] = trace_id

    if task_id:
        context["task_id"] = task_id
    if agent_id:
        context["agent_id"] = agent_id

    context.update(extra)
    return ContextLoggerAdapter(logger_instance, context)

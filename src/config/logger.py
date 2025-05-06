# import json
# import logging
# import sys
# import time
# import uuid
# from datetime import datetime, timezone
# from typing import Optional, Dict, Union, Any
# from src.config.settings import get_settings
# settings = get_settings()

# class JsonFormatter(logging.Formatter):

#     def __init__(self):
#         super().__init__()

#     def format(self, record: logging.LogRecord) -> str:
#         log_data = {'timestamp': datetime.now(timezone.utc).isoformat(), 'level': record.levelname, 'name': record.name, 'message': record.getMessage(), 'module': record.module, 'function': record.funcName, 'lineno': record.lineno, 'process': record.process, 'thread': record.thread}
#         if record.exc_info:
#             log_data['exception'] = {'type': record.exc_info[0].__name__, 'message': str(record.exc_info[1]), 'traceback': self.formatException(record.exc_info)}
#         if hasattr(record, 'trace_id'):
#             log_data['trace_id'] = record.trace_id
#         if hasattr(record, 'task_id'):
#             log_data['task_id'] = record.task_id
#         if hasattr(record, 'agent_id'):
#             log_data['agent_id'] = record.agent_id
#         if hasattr(record, 'execution_time'):
#             log_data['execution_time'] = record.execution_time
#         standard_attrs = logging.LogRecord('', '', '', '', '', '', '', '').__dict__.keys()
#         extra_attrs = {k: v for k, v in record.__dict__.items() if k not in standard_attrs and k != 'exc_info' and (k != 'exc_text') and (k != 'stack_info') and (k != 'relativeCreated') and (k != 'args') and (k != 'message') and (k != 'asctime')}
#         if extra_attrs:
#             log_data.update(extra_attrs)
#         return json.dumps(log_data, default=str)

# class TraceLogger(logging.Logger):

#     def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
#         if extra is None:
#             extra = {}
#         if 'trace_id' not in extra:
#             extra['trace_id'] = str(uuid.uuid4())
#         super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)

# class ContextLoggerAdapter(logging.LoggerAdapter):

#     def process(self, msg, kwargs):
#         if 'extra' not in kwargs:
#             kwargs['extra'] = {}
#         if self.extra:
#             for k, v in self.extra.items():
#                 kwargs['extra'][k] = v
#         return (msg, kwargs)

#     def info(self, msg, *args, **kwargs):
#         msg, kwargs = self.process(msg, kwargs)
#         self.logger.info(msg, *args, **kwargs)

#     def warning(self, msg, *args, **kwargs):
#         msg, kwargs = self.process(msg, kwargs)
#         self.logger.warning(msg, *args, **kwargs)

#     def error(self, msg, *args, **kwargs):
#         msg, kwargs = self.process(msg, kwargs)
#         self.logger.error(msg, *args, **kwargs)

#     def debug(self, msg, *args, **kwargs):
#         msg, kwargs = self.process(msg, kwargs)
#         self.logger.debug(msg, *args, **kwargs)

# def setup_logging():
#     logging.setLoggerClass(TraceLogger)
#     root_logger = logging.getLogger()
#     root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
#     for handler in list(root_logger.handlers):
#         root_logger.removeHandler(handler)
#     console_handler = logging.StreamHandler(sys.stdout)
#     if settings.LOG_FORMAT == 'json':
#         formatter = JsonFormatter()
#     else:
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#     console_handler.setFormatter(formatter)
#     root_logger.addHandler(console_handler)
#     if settings.LOG_TO_FILE and settings.LOG_FILE_PATH:
#         try:
#             file_handler = logging.FileHandler(settings.LOG_FILE_PATH)
#             file_handler.setFormatter(formatter)
#             root_logger.addHandler(file_handler)
#         except Exception as e:
#             root_logger.error(f'Failed to set up file logging to {settings.LOG_FILE_PATH}: {e}')
#     _configure_library_loggers()
#     root_logger.info(f'Logging setup complete. Level: {settings.LOG_LEVEL}, Format: {settings.LOG_FORMAT}')

# def _configure_library_loggers():
#     logging.getLogger('uvicorn').setLevel(logging.WARNING)
#     logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
#     logging.getLogger('fastapi').setLevel(logging.WARNING)

# def get_logger(name: str) -> logging.Logger:
#     return logging.getLogger(name)

# def get_logger_with_context(name: str, trace_id: Optional[str]=None, task_id: Optional[str]=None, agent_id: Optional[str]=None, **extra) -> ContextLoggerAdapter:
#     logger_instance = logging.getLogger(name)
#     context = {'trace_id': trace_id or str(uuid.uuid4())}
#     if task_id:
#         context['task_id'] = task_id
#     if agent_id:
#         context['agent_id'] = agent_id
#     context.update(extra)
#     return ContextLoggerAdapter(logger_instance, context)

"""
Logging configuration for the Multi-Agent Platform.
Provides structured logging with context propagation.
"""
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional, Set

# Import settings directly to avoid circular imports
# This works because we're not creating a settings instance here
from src.schemas.config import AppSettings

# Cache of standard LogRecord attributes to avoid repeated lookups
STANDARD_LOGRECORD_ATTRIBUTES: Set[str] = {
    'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
    'funcName', 'levelname', 'levelno', 'lineno', 'module',
    'msecs', 'message', 'msg', 'name', 'pathname', 'process',
    'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
}


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Optimized for performance with conditional processing.
    """
    def __init__(self, include_thread_info: bool = False, include_process_info: bool = False):
        super().__init__()
        self.include_thread_info = include_thread_info
        self.include_process_info = include_process_info

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.
        Optimized to only include necessary fields.
        
        Args:
            record: The log record to format
            
        Returns:
            str: JSON-formatted log string
        """
        # Core log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'lineno': record.lineno,
        }
        
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
        
        # Tracking IDs
        if hasattr(record, 'trace_id'):
            log_data['trace_id'] = record.trace_id
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id
        if hasattr(record, 'agent_id'):
            log_data['agent_id'] = record.agent_id
            
        # Performance metrics
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time
        
        # Add any custom attributes
        extra_attrs = getattr(record, 'extra_attrs', None)
        if extra_attrs and isinstance(extra_attrs, dict):
            log_data.update(extra_attrs)
        else:
            # Fall back to scanning all attributes (slower)
            for key, value in record.__dict__.items():
                if (key not in STANDARD_LOGRECORD_ATTRIBUTES and 
                    not key.startswith('_') and 
                    key not in log_data):
                    log_data[key] = value
        
        try:
            return json.dumps(log_data, default=str, separators=(',', ':'))
        except Exception as e:
            # Fallback in case of serialization issues
            return json.dumps({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': 'ERROR',
                'name': 'logger',
                'message': f'Error serializing log record: {str(e)}',
                'original_message': str(record.getMessage())
            }, default=str)


class TraceLogger(logging.Logger):
    """
    Logger with automatic trace ID generation.
    Ensures all log records have a trace ID for tracking.
    """
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        """
        Override to add trace_id to all log records.
        """
        if extra is None:
            extra = {}
        if 'trace_id' not in extra:
            extra['trace_id'] = str(uuid.uuid4())
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, 
                    stack_info=stack_info, stacklevel=stacklevel)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log records.
    Preserves context across multiple log calls.
    """
    def process(self, msg, kwargs):
        """
        Process the logging message and add context.
        """
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add context fields to extra
        if self.extra:
            extra_attrs = kwargs['extra'].get('extra_attrs', {})
            if not isinstance(extra_attrs, dict):
                extra_attrs = {}
                
            # Add adapter's context to extra_attrs
            for k, v in self.extra.items():
                if k in ('trace_id', 'task_id', 'agent_id'):
                    # Keep trace IDs at top level
                    kwargs['extra'][k] = v
                else:
                    # Put other context in extra_attrs
                    extra_attrs[k] = v
                    
            kwargs['extra']['extra_attrs'] = extra_attrs
            
        return (msg, kwargs)


def setup_logging(settings: Optional[AppSettings] = None) -> None:
    """
    Set up logging for the application.
    
    Args:
        settings: Optional settings instance. If None, will use environment variables.
    """
    # Import settings here if not provided
    if settings is None:
        from src.config.settings import get_settings
        settings = get_settings()
    
    # Set the custom logger class
    logging.setLoggerClass(TraceLogger)
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Set log level
    log_level_name = getattr(settings, 'LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level_name)
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter based on settings
    log_format = getattr(settings, 'LOG_FORMAT', 'json')
    if log_format == 'json':
        include_process_info = log_level_name == 'DEBUG'
        include_thread_info = log_level_name in ('DEBUG', 'INFO')
        formatter = JsonFormatter(
            include_thread_info=include_thread_info,
            include_process_info=include_process_info
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if getattr(settings, 'LOG_TO_FILE', False) and getattr(settings, 'LOG_FILE_PATH', None):
        try:
            file_handler = logging.FileHandler(settings.LOG_FILE_PATH)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f'Log file configured at {settings.LOG_FILE_PATH}')
        except Exception as e:
            root_logger.error(f'Failed to set up file logging to {settings.LOG_FILE_PATH}: {e}')
    
    # Configure third-party library loggers
    _configure_library_loggers()
    
    root_logger.info(f'Logging setup complete. Level: {log_level_name}, Format: {log_format}')


def _configure_library_loggers() -> None:
    """Configure third-party library loggers to appropriate levels."""
    loggers_to_configure = {
        'uvicorn': logging.WARNING,
        'uvicorn.access': logging.WARNING,
        'uvicorn.error': logging.ERROR,
        'fastapi': logging.WARNING,
        'aiohttp': logging.WARNING,
        'redis': logging.WARNING,
    }
    
    for logger_name, level in loggers_to_configure.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    
    Args:
        name: Logger name, typically __name__
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def get_logger_with_context(
    name: str, 
    trace_id: Optional[str] = None, 
    task_id: Optional[str] = None, 
    agent_id: Optional[str] = None, 
    **extra
) -> ContextLoggerAdapter:
    """
    Get a logger with context data attached.
    
    Args:
        name: Logger name
        trace_id: Optional trace ID for request tracking
        task_id: Optional task ID
        agent_id: Optional agent ID
        **extra: Additional context key-value pairs
        
    Returns:
        ContextLoggerAdapter: Logger adapter with context
    """
    logger_instance = logging.getLogger(name)
    
    # Prepare context
    context = {'trace_id': trace_id or str(uuid.uuid4())}
    
    if task_id:
        context['task_id'] = task_id
    if agent_id:
        context['agent_id'] = agent_id
        
    # Add any extra context
    context.update(extra)
    
    return ContextLoggerAdapter(logger_instance, context)
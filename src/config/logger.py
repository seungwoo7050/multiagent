import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Union, Any

from src.config.settings import get_settings

settings = get_settings()

class JsonFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "lineno": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
            
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "agent_id"):
            log_data["agent_id"] = record.agent_id
        if hasattr(record, "execution_time"):
            log_data["execution_time"] = record.execution_time            
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            for k, v in record.extra.items():
                log_data[k] = v
                
        return json.dumps(log_data, default=str)
    
class TraceLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if extra is None:
            extra = {}
        if "trace_id" not in extra:
            extra["trace_id"] = str(uuid.uuid4())
        
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)
        
class ContextLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
            
        if self.extra:
            for k, v in self.extra.items():
                kwargs['extra'][k] = v
                
        return msg, kwargs
    
    def info(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        self.logger.error(msg, *args, **kwargs)
        
    def debug(self, msg, *args, **kwargs):
        msg, kwargs = self.process(msg, kwargs)
        self.logger.debug(msg, *args, **kwargs)
        
def setup_logging():
    logging.setLoggerClass(TraceLogger)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    
    if settings.LOG_FORMAT == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler) 
    
    if settings.LOG_TO_FILE and settings.LOG_FILE_PATH:
        file_handler = logging.FileHandler(settings.LOG_FILE_PATH)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    _configure_library_loggers()
    
def _configure_library_loggers():
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def get_logger_with_context(
    name: str,
    trace_id: Optional[str] = None,
    task_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **extra,
) -> ContextLoggerAdapter:
    logger = logging.getLogger(name)
    
    context = {
        "trace_id": trace_id or str(uuid.uuid4()),
    }
    
    if task_id:
        context['task_id'] = task_id
    if agent_id:
        context['agent_id'] = agent_id
        
    context.update(extra)
    
    return ContextLoggerAdapter(logger, context)
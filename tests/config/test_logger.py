import json
import logging
from unittest.mock import patch, MagicMock, ANY

import pytest

from src.config.logger import (
    JsonFormatter,
    TraceLogger,
    ContextLoggerAdapter,
    setup_logging,
    get_logger, 
    get_logger_with_context
)

def test_json_formatter_basic():
    formatter = JsonFormatter()
    
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path",
        lineno=42,
        msg="Test message",
        args=None,
        exc_info=None
    )
    
    formatted = formatter.format(record)
    
    parsed = json.loads(formatted)
    
    assert parsed["name"] == "test_logger"
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test message"
    assert parsed["timestamp"] is not None
    
def test_json_formatter_with_context():
    formatter = JsonFormatter()
    
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path",
        lineno=42,
        msg="Test message",
        args=None,
        exc_info=None
    )
    
    record.trace_id = "trace-42"
    record.task_id = "task-42"
    
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    
    assert parsed["trace_id"] == "trace-42"
    assert parsed["task_id"] == "task-42"
    
def test_trace_logger_auto_trace_id():
    with patch.object(logging.Logger, '_log') as mock_log:
        logger = TraceLogger(name="test_logger")
        logger.info("Test message")
        
        _, kwargs = mock_log.call_args
        assert "extra" in kwargs
        assert "trace_id" in kwargs["extra"]
        assert kwargs["extra"]["trace_id"]
        
def test_context_logger_adapter():
    mock_logger = MagicMock()
    context = {"trace_id": "trace-42", "task_id": "task-42"}
    
    adapter = ContextLoggerAdapter(mock_logger, context)
    
    adapter.info("Test message")
    
    _, kwargs = mock_logger.info.call_args
    
    assert 'extra' in kwargs
    assert kwargs['extra']['trace_id'] == "trace-42"
    assert kwargs['extra']['task_id'] == "task-42"
    
    
def test_setup_logging_basic():
    with patch('src.config.logger.settings') as mock_settings, \
         patch('logging.setLoggerClass') as mock_set_logger_class, \
         patch('logging.getLogger') as mock_get_logger, \
         patch('logging.StreamHandler') as mock_stream_handler:
        
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.LOG_FORMAT = "json"
        mock_settings.LOG_TO_FILE = False
        
        mock_root_logger = MagicMock()
        mock_get_logger.return_value = mock_root_logger
        
        mock_handler = MagicMock()
        mock_stream_handler.return_value = mock_handler
        
        setup_logging()
        
        mock_set_logger_class.assert_called_once_with(TraceLogger)
        mock_root_logger.addHandler.assert_called_once_with(mock_handler)
        # why is this called four times?
        # once for the root logger, and then once for each of the other loggers:
        # uvicorn, uvicorn.access, and fastapi
        assert mock_root_logger.setLevel.call_count == 4
        
def test_get_logger_returns_logger():
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        logger = get_logger("test_logger")
        
        mock_get_logger.assert_called_once_with("test_logger")
        assert logger == mock_logger
        
def test_get_logger_with_context_returns_adapter():
    logger = get_logger_with_context("test_logger")
    
    assert isinstance(logger, ContextLoggerAdapter)
    assert 'trace_id' in logger.extra
    assert logger.extra['trace_id'] is not None
    
def test_get_logger_with_context_uses_provided_context():
    test_trace_id = "test-trace-id"
    test_task_id = "test-task-id"
    
    logger = get_logger_with_context("test_logger", trace_id=test_trace_id, task_id=test_task_id)
    
    assert logger.extra['trace_id'] == test_trace_id
    assert logger.extra['task_id'] == test_task_id
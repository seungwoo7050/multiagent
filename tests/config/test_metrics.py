import time
from unittest.mock import patch, MagicMock, ANY

import pytest
from prometheus_client import Histogram, Counter, Gauge

from src.config.metrics import (
    timed_metric,
    track_http_request,
    track_task_created,
    track_llm_request,
    track_llm_response,
    track_agent_created,
    track_memory_operation,
    start_metrics_server,
    
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,  
    HTTP_REQUEST_SIZE,  
    HTTP_RESPONSE_SIZE,
    TASK_CREATED_TOTAL,
    TASK_QUEUE_DEPTH,
    LLM_REQUESTS_TOTAL,
    LLM_REQUEST_DURATION,
    LLM_TOKEN_USAGE,
    MEMORY_OPERATIONS_TOTAL
)

def test_timed_metric():
    mock_histogram = MagicMock()
    @timed_metric(mock_histogram)
    def test_function():
        time.sleep(0.01)
        return "result"
    
    result = test_function()
    
    assert result == "result"
    assert mock_histogram.observe.called
    
def test_timed_metric_with_labels():
    mock_histogram = MagicMock()
    mock_labed_histogram = MagicMock()
    mock_histogram.labels.return_value = mock_labed_histogram
    
    test_labels = {"operation": "test", "component": "metrics"}
    
    @timed_metric(mock_histogram, labels=test_labels)
    def test_function():
        time.sleep(0.01)
        return "result"
    
    test_function()
    
    mock_histogram.labels.assert_called_once_with(**test_labels)
    assert mock_labed_histogram.observe.called
    
def test_track_http_request():
    with patch.object(HTTP_REQUESTS_TOTAL, 'labels') as mock_total, \
         patch.object(HTTP_REQUEST_SIZE, 'labels') as mock_request_size, \
         patch.object(HTTP_REQUEST_DURATION, 'labels') as mock_duration, \
         patch.object(HTTP_RESPONSE_SIZE, 'labels') as mock_response_size:
        
        mock_total.return_value = MagicMock()
        mock_duration.return_value = MagicMock()
        mock_request_size.return_value = MagicMock()
        mock_response_size.return_value = MagicMock()
        
        track_http_request(
            method="GET",
            endpoint="/api/test",
            status_code=200,
            duration=0.5,
            request_size=1000,
            response_size=2000
        )
        
        assert mock_total.called
        assert mock_duration.called
        assert mock_request_size.called
        assert mock_response_size.called
        
        assert mock_total.return_value.inc.called
        assert mock_duration.return_value.observe.called
        assert mock_request_size.return_value.observe.called
        assert mock_response_size.return_value.observe.called
        
def test_track_task_created():
    with patch.object(TASK_CREATED_TOTAL, 'inc') as mock_total_inc, \
         patch.object(TASK_QUEUE_DEPTH, 'inc') as mock_queue_inc:
    
        track_task_created()
        assert mock_total_inc.called
        assert mock_queue_inc.called
        
def test_track_llm_request():
    with patch.object(LLM_REQUESTS_TOTAL, 'labels') as mock_total:
        mock_total.return_value = MagicMock()
        
        track_llm_request(
            model="gpt-3.5-turbo",
            provider="openai",
        )
        
        assert mock_total.called
        assert "gpt-3.5-turbo" in str(mock_total.call_args)
        assert "openai" in str(mock_total.call_args)
        assert mock_total.return_value.inc.called
        
def test_track_llm_response():
    with patch.object(LLM_REQUEST_DURATION, 'labels') as mock_duration, \
         patch.object(LLM_TOKEN_USAGE, 'labels') as mock_tokens:
        
        mock_duration.return_value = MagicMock()
        mock_tokens.return_value = MagicMock()
        
        track_llm_response(
            model="gpt-3.5-turbo",
            provider="openai",
            duration=0.5,
            prompt_tokens=100,
            completion_tokens=50
        )
        
        assert mock_duration.called
        assert mock_duration.return_value.observe.called
        assert mock_tokens.call_count >= 2
        
def test_track_memory_operation():
    with patch.object(MEMORY_OPERATIONS_TOTAL, 'labels') as mock_ops:
        mock_ops.return_value = MagicMock()
        
        track_memory_operation(operation_type="read")
        
        assert mock_ops.called
        assert "read" in str(mock_ops.call_args)
        assert mock_ops.return_value.inc.called
        
def test_start_metrics_server_enabled():
    with patch('src.config.metrics.settings') as mock_settings, \
         patch('threading.Thread') as mock_thread, \
         patch('prometheus_client.start_http_server'), \
         patch('src.config.metrics.SYSTEM_INFO.labels') as mock_system_info:
             
        mock_settings.METRICS_ENABLED = True
        mock_settings.METRICS_PORT = 9090
        mock_settings.APP_NAME = "test_app"
        mock_settings.APP_VERSION = "1.0.0"
        mock_settings.ENVIRONMENT = "test"
        
        mock_system_info.return_value = MagicMock()
        
        result = start_metrics_server()
        
        assert mock_thread.called
        assert mock_thread.return_value.start.called
        assert result is not None
        
def test_start_metrics_server_disabled():
    with patch('src.config.metrics.settings') as mock_settings, \
         patch('threading.Thread') as mock_thread:
             
        mock_settings.METRICS_ENABLED = False
        
        result = start_metrics_server()
        
        assert not mock_thread.called
        assert result is None
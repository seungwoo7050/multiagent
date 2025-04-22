from unittest.mock import patch, MagicMock

import pytest

from src.config.errors import (
    ErrorCode,
    BaseError,
    SystemError,
    APIError,
    ValidationError,
    NotFoundError,
    TaskError,
    LLMError,
    MemoryError,
    AgentError,
    ToolError,
    OrchestrationError,
    ConnectionError,
    create_error_from_code,
    convert_exception,
    RETRYABLE_ERRORS,
    ERROR_TO_HTTP_STATUS,
)


def test_base_error_creation():
    error = BaseError(
        code=ErrorCode.SYSTEM_ERROR,
        message="Test error message",
        details={"key": "value"},
        original_error=ValueError("Original error")
    )
    
    assert error.code == ErrorCode.SYSTEM_ERROR
    assert error.message == "Test error message"
    assert error.details == {"key": "value"}
    assert isinstance(error.original_error, ValueError)
    assert str(error.original_error) == "Original error"


def test_base_error_to_dict():
    error = BaseError(
        code=ErrorCode.SYSTEM_ERROR,
        message="Test error message",
        details={"key": "value"},
    )
    
    error_dict = error.to_dict()
    
    assert "code" in error_dict
    assert "message" in error_dict
    assert "details" in error_dict
    assert error_dict["code"] == ErrorCode.SYSTEM_ERROR.value
    assert error_dict["message"] == "Test error message"


def test_base_error_log_error():
    mock_logger = MagicMock()
    
    error = BaseError(
        code=ErrorCode.SYSTEM_ERROR,
        message="Test error message",
    )
    
    error.log_error(mock_logger)
    
    assert mock_logger.error.called


def test_api_error_with_status_code():
    error = APIError(
        code=ErrorCode.VALIDATION_ERROR,
        message="Validation failed",
        status_code=400
    )
    
    assert error.status_code == 400
    
    error_dict = error.to_dict()
    assert "status_code" in error_dict
    assert error_dict["status_code"] == 400


def test_validation_error_default_status():
    error = ValidationError(
        message="Invalid input"
    )
    
    assert error.status_code == 400
    assert error.code == ErrorCode.VALIDATION_ERROR


def test_not_found_error_generates_message():
    error = NotFoundError(
        resource_type="User",
        resource_id="123"
    )
    
    assert "User" in error.message
    assert "123" in error.message
    assert error.status_code == 404


def test_task_error_with_task_id():
    task_id = "task-123"
    error = TaskError(
        message="Task failed",
        task_id=task_id
    )
    
    assert "task_id" in error.details
    assert error.details["task_id"] == task_id


def test_llm_error_with_model_and_provider():
    model = "gpt-4"
    provider = "openai"
    
    error = LLMError(
        message="LLM request failed",
        model=model,
        provider=provider
    )
    
    assert "model" in error.details
    assert "provider" in error.details
    assert error.details["model"] == model
    assert error.details["provider"] == provider


def test_create_error_from_code_known_code():
    error = create_error_from_code(
        code=ErrorCode.TASK_TIMEOUT,
        message="Task timed out"
    )
    
    assert isinstance(error, TaskError)
    assert error.code == ErrorCode.TASK_TIMEOUT
    assert error.message == "Task timed out"


def test_create_error_from_code_string():
    error = create_error_from_code(
        code="VALIDATION_ERROR_2001",
        message="Validation failed"
    )
    
    assert isinstance(error, ValidationError)
    assert error.code == ErrorCode.VALIDATION_ERROR


def test_create_error_from_code_with_kwargs():
    error = create_error_from_code(
        code=ErrorCode.TOOL_ERROR,
        message="Tool failed",
        tool_name="calculator"
    )
    
    assert isinstance(error, ToolError)
    assert "tool_name" in error.details
    assert error.details["tool_name"] == "calculator"


def test_convert_exception_standard_exception():
    original = ValueError("Test value error")
    
    error = convert_exception(
        exception=original,
        default_code=ErrorCode.VALIDATION_ERROR,
        default_message="Validation error"
    )
    
    assert isinstance(error, ValidationError)
    assert error.code == ErrorCode.VALIDATION_ERROR
    assert error.message == "Validation error"
    assert error.original_error is original


def test_convert_exception_already_base_error():
    original = TaskError(
        code=ErrorCode.TASK_EXECUTION_ERROR,
        message="Task execution failed"
    )
    
    error = convert_exception(
        exception=original,
        default_code=ErrorCode.SYSTEM_ERROR
    )
    
    assert error is original


def test_retryable_errors_list():
    assert len(RETRYABLE_ERRORS) > 0
    assert ErrorCode.LLM_TIMEOUT in RETRYABLE_ERRORS
    assert ErrorCode.NETWORK_ERROR in RETRYABLE_ERRORS


def test_error_to_http_status_mapping():
    assert ERROR_TO_HTTP_STATUS[ErrorCode.VALIDATION_ERROR] == 400
    assert ERROR_TO_HTTP_STATUS[ErrorCode.AUTHORIZATION_ERROR] == 403
    assert ERROR_TO_HTTP_STATUS[ErrorCode.ENDPOINT_NOT_FOUND] == 404
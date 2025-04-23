import pytest
from typing import Dict, Any, Optional

from src.config.errors import BaseError, ErrorCode
from src.core.exceptions import (
    CoreError,
    TaskError,
    TaskNotFoundError,
    TaskCreationError,
    TaskExecutionError,
    TaskTimeoutError,
    AgentError,
    AgentNotFoundError,
    AgentCreationError,
    AgentInitializationError,
    AgentExecutionError,
    AgentTimeoutError,
    FactoryError,
    RegistryError,
    CircuitBreakerError,
    BackpressureError,
    WorkerPoolError,
    SerializationError,
    ValidationError,
    ConfigurationError
)


class TestCoreExceptions:
    """Test suite for core exceptions."""

    def test_core_error_base(self):
        """Test the CoreError base class."""
        error = CoreError(
            code=ErrorCode.SYSTEM_ERROR,
            message="Test error message",
            details={"key": "value"},
            original_error=ValueError("Original error")
        )
        
        # Check properties
        assert error.code == ErrorCode.SYSTEM_ERROR
        assert error.message == "Test error message"
        assert error.details == {"key": "value"}
        assert isinstance(error.original_error, ValueError)
        assert str(error.original_error) == "Original error"
        
        # Check inheritance
        assert isinstance(error, CoreError)
        assert isinstance(error, BaseError)
        assert isinstance(error, Exception)
        
        # Check string representation
        assert str(error) == "Test error message"

    def test_task_errors(self):
        """Test task-related error classes."""
        # Generic TaskError
        task_error = TaskError(
            message="Task error message",
            task_id="task123",
            error_code=ErrorCode.TASK_ERROR,
            details={"source": "test"}
        )
        
        assert task_error.message == "Task error message"
        assert task_error.details["task_id"] == "task123"
        assert task_error.details["source"] == "test"
        assert task_error.code == ErrorCode.TASK_ERROR
        
        # TaskNotFoundError
        not_found_error = TaskNotFoundError(
            task_id="missing_task",
            message="Custom not found message"
        )
        
        assert not_found_error.code == ErrorCode.TASK_NOT_FOUND
        assert not_found_error.message == "Custom not found message"
        assert not_found_error.details["task_id"] == "missing_task" # task_id는 details에서 확인
        
        # Auto-generated message when not provided
        default_not_found = TaskNotFoundError(task_id="task456")
        assert "task456" in default_not_found.message
        
        # TaskCreationError
        creation_error = TaskCreationError(
            message="Failed to create task",
            task_type="test_task"
        )
        
        assert creation_error.code == ErrorCode.TASK_CREATION_ERROR
        assert creation_error.details["task_type"] == "test_task"
        
        # TaskExecutionError
        execution_error = TaskExecutionError(
            message="Failed to execute task",
            task_id="task789"
        )
        
        assert execution_error.code == ErrorCode.TASK_EXECUTION_ERROR
        assert execution_error.details["task_id"] == "task789"
        
        # TaskTimeoutError
        timeout_error = TaskTimeoutError(
            task_id="task_timeout",
            timeout_seconds=30.0
        )
        
        assert timeout_error.code == ErrorCode.TASK_TIMEOUT
        assert "30.0 seconds" in timeout_error.message
        assert timeout_error.details["task_id"] == "task_timeout"
        assert timeout_error.details["timeout_seconds"] == 30.0

    def test_agent_errors(self):
        """Test agent-related error classes."""
        # Generic AgentError
        agent_error = AgentError(
            message="Agent error message",
            agent_type="test_agent",
            agent_id="agent123",
            error_code=ErrorCode.AGENT_ERROR
        )
        
        assert agent_error.message == "Agent error message"
        assert agent_error.details["agent_type"] == "test_agent"
        assert agent_error.details["agent_id"] == "agent123"
        assert agent_error.code == ErrorCode.AGENT_ERROR
        
        # Test default error_code
        default_agent_error = AgentError(message="Default code")
        assert default_agent_error.code == ErrorCode.AGENT_ERROR
        
        # AgentNotFoundError
        not_found_error = AgentNotFoundError(agent_type="missing_agent")
        
        assert not_found_error.code == ErrorCode.AGENT_NOT_FOUND
        assert "missing_agent" in not_found_error.message
        assert not_found_error.details["agent_type"] == "missing_agent"
        
        # Custom message
        custom_not_found = AgentNotFoundError(
            agent_type="custom_agent",
            message="Custom not found message"
        )
        assert custom_not_found.message == "Custom not found message"
        
        # AgentCreationError
        creation_error = AgentCreationError(
            message="Failed to create agent",
            agent_type="test_agent"
        )
        
        assert creation_error.code == ErrorCode.AGENT_CREATION_ERROR
        assert creation_error.details["agent_type"] == "test_agent"
        
        # AgentInitializationError
        init_error = AgentInitializationError(
            message="Failed to initialize agent",
            agent_type="test_agent",
            agent_id="agent456"
        )
        
        assert init_error.code == ErrorCode.AGENT_CREATION_ERROR
        assert init_error.details["agent_type"] == "test_agent"
        assert init_error.details["agent_id"] == "agent456"
        
        # AgentExecutionError
        execution_error = AgentExecutionError(
            message="Failed to execute agent",
            agent_type="test_agent",
            agent_id="agent789"
        )
        
        assert execution_error.code == ErrorCode.AGENT_EXECUTION_ERROR
        assert execution_error.details["agent_type"] == "test_agent"
        assert execution_error.details["agent_id"] == "agent789"
        
        # AgentTimeoutError
        timeout_error = AgentTimeoutError(
            agent_type="test_agent",
            timeout_seconds=60.0,
            operation="process",
            agent_id="agent_timeout"
        )
        
        assert timeout_error.code == ErrorCode.AGENT_TIMEOUT
        assert "process" in timeout_error.message
        assert "60.0 seconds" in timeout_error.message
        assert timeout_error.details["agent_type"] == "test_agent"
        assert timeout_error.details["agent_id"] == "agent_timeout"
        assert timeout_error.details["timeout_seconds"] == 60.0
        assert timeout_error.details["operation"] == "process"
        
        # Without operation
        simple_timeout = AgentTimeoutError(
            agent_type="simple_agent",
            timeout_seconds=30.0
        )
        assert "Agent timed out" in simple_timeout.message
        assert "operation" not in simple_timeout.details

    def test_factory_error(self):
        """Test FactoryError class."""
        error = FactoryError(
            message="Factory error message",
            factory_name="test_factory",
            details={"key": "value"},
            original_error=RuntimeError("Original error")
        )
        
        assert error.message == "Factory error message"
        assert error.details["factory_name"] == "test_factory"
        assert error.details["key"] == "value"
        assert isinstance(error.original_error, RuntimeError)
        assert error.code == ErrorCode.SYSTEM_ERROR

    def test_registry_error(self):
        """Test RegistryError class."""
        error = RegistryError(
            message="Registry error message",
            registry_name="test_registry",
            details={"component": "test"},
            original_error=KeyError("Original error")
        )
        
        assert error.message == "Registry error message"
        assert error.details["registry_name"] == "test_registry"
        assert error.details["component"] == "test"
        assert isinstance(error.original_error, KeyError)
        assert error.code == ErrorCode.SYSTEM_ERROR

    def test_circuit_breaker_error(self):
        """Test CircuitBreakerError class."""
        error = CircuitBreakerError(
            message="Circuit breaker error message",
            circuit_name="test_circuit",
            details={"state": "open"},
            original_error=ConnectionError("Original error")
        )
        
        assert error.message == "Circuit breaker error message"
        assert error.details["circuit_name"] == "test_circuit"
        assert error.details["state"] == "open"
        assert isinstance(error.original_error, ConnectionError)
        assert error.code == ErrorCode.CIRCUIT_BREAKER_OPEN

    def test_backpressure_error(self):
        """Test BackpressureError class."""
        error = BackpressureError(
            message="Backpressure error message",
            controller_name="test_controller",
            details={"strategy": "reject"},
            original_error=OverflowError("Original error")
        )
        
        assert error.message == "Backpressure error message"
        assert error.details["controller_name"] == "test_controller"
        assert error.details["strategy"] == "reject"
        assert isinstance(error.original_error, OverflowError)
        assert error.code == ErrorCode.SYSTEM_ERROR

    def test_worker_pool_error(self):
        """Test WorkerPoolError class."""
        error = WorkerPoolError(
            message="Worker pool error message",
            details={"pool_type": "thread"},
            original_error=RuntimeError("Original error")
        )
        
        assert error.message == "Worker pool error message"
        assert error.details["pool_type"] == "thread"
        assert isinstance(error.original_error, RuntimeError)
        assert error.code == ErrorCode.WORKER_ERROR

    def test_serialization_error(self):
        """Test SerializationError class."""
        error = SerializationError(
            message="Serialization error message",
            details={"format": "json"},
            original_error=TypeError("Original error")
        )
        
        assert error.message == "Serialization error message"
        assert error.details["format"] == "json"
        assert isinstance(error.original_error, TypeError)
        assert error.code == ErrorCode.SYSTEM_ERROR

    def test_validation_error(self):
        """Test ValidationError class."""
        error = ValidationError(
            message="Validation error message",
            field="username",
            details={"constraint": "min_length"},
            original_error=ValueError("Original error")
        )
        
        assert error.message == "Validation error message"
        assert error.details["field"] == "username"
        assert error.details["constraint"] == "min_length"
        assert isinstance(error.original_error, ValueError)
        assert error.code == ErrorCode.VALIDATION_ERROR
        
        # Without field
        simple_error = ValidationError(message="Simple validation error")
        assert "field" not in simple_error.details

    def test_configuration_error(self):
        """Test ConfigurationError class."""
        error = ConfigurationError(
            message="Configuration error message",
            component="database",
            details={"setting": "connection_string"},
            original_error=ValueError("Original error")
        )
        
        assert error.message == "Configuration error message"
        assert error.details["component"] == "database"
        assert error.details["setting"] == "connection_string"
        assert isinstance(error.original_error, ValueError)
        assert error.code == ErrorCode.CONFIG_ERROR
        
        # Without component
        simple_error = ConfigurationError(message="Simple config error")
        assert "component" not in simple_error.details
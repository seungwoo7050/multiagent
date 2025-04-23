from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.config.errors import BaseError, ErrorCode


class CoreError(BaseError):
    """Base exception class for core module errors."""
    
    def __init__(
        self,
        code: Union[ErrorCode, str],
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(code, message, details, original_error)


class TaskError(CoreError):
    """Exception raised for task-related errors."""
    
    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.TASK_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
        
        if task_id:
            details["task_id"] = task_id
            
        super().__init__(error_code, message, details, original_error)


class TaskNotFoundError(TaskError):
    """Exception raised when a task is not found."""
    
    def __init__(
        self,
        task_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if message is None:
            message = f"Task not found: {task_id}"
            
        super().__init__(
            message=message,
            task_id=task_id,
            error_code=ErrorCode.TASK_NOT_FOUND,
            details=details,
            original_error=original_error
        )


class TaskCreationError(TaskError):
    """Exception raised when task creation fails."""
    
    def __init__(
        self,
        message: str,
        task_type: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        details["task_type"] = task_type
        
        super().__init__(
            message=message,
            error_code=ErrorCode.TASK_CREATION_ERROR,
            details=details,
            original_error=original_error
        )


class TaskExecutionError(TaskError):
    """Exception raised when task execution fails."""
    
    def __init__(
        self,
        message: str,
        task_id: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            task_id=task_id,
            error_code=ErrorCode.TASK_EXECUTION_ERROR,
            details=details,
            original_error=original_error
        )


class TaskTimeoutError(TaskError):
    """Exception raised when a task times out."""
    
    def __init__(
        self,
        task_id: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"Task timed out after {timeout_seconds} seconds: {task_id}"
        
        if details is None:
            details = {}
            
        details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=message,
            task_id=task_id,
            error_code=ErrorCode.TASK_TIMEOUT,
            details=details
        )


class AgentError(CoreError):
    """Exception raised for agent-related errors."""
    
    def __init__(
        self,
        message: str,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.AGENT_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        if agent_type:
            details["agent_type"] = agent_type
            
        if agent_id:
            details["agent_id"] = agent_id
            
        super().__init__(error_code, message, details, original_error)


class AgentNotFoundError(AgentError):
    """Exception raised when an agent is not found."""
    
    def __init__(
        self,
        agent_type: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if message is None:
            message = f"Agent not found: {agent_type}"
            
        super().__init__(
            message=message,
            agent_type=agent_type,
            error_code=ErrorCode.AGENT_NOT_FOUND,
            details=details,
            original_error=original_error
        )


class AgentCreationError(AgentError):
    """Exception raised when agent creation fails."""
    
    def __init__(
        self,
        message: str,
        agent_type: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            agent_type=agent_type,
            error_code=ErrorCode.AGENT_CREATION_ERROR,
            details=details,
            original_error=original_error
        )


class AgentInitializationError(AgentError):
    """Exception raised when agent initialization fails."""
    
    def __init__(
        self,
        message: str,
        agent_type: str,
        agent_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            agent_type=agent_type,
            agent_id=agent_id,
            error_code=ErrorCode.AGENT_CREATION_ERROR,
            details=details,
            original_error=original_error
        )


class AgentExecutionError(AgentError):
    """Exception raised when agent execution fails."""
    
    def __init__(
        self,
        message: str,
        agent_type: str,
        agent_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            agent_type=agent_type,
            agent_id=agent_id,
            error_code=ErrorCode.AGENT_EXECUTION_ERROR,
            details=details,
            original_error=original_error
        )


class AgentTimeoutError(AgentError):
    """Exception raised when an agent operation times out."""
    
    def __init__(
        self,
        agent_type: str,
        timeout_seconds: float,
        operation: Optional[str] = None,
        agent_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if operation:
            message = f"Agent operation '{operation}' timed out after {timeout_seconds} seconds"
        else:
            message = f"Agent timed out after {timeout_seconds} seconds"
            
        if details is None:
            details = {}
            
        details["timeout_seconds"] = timeout_seconds
        
        if operation:
            details["operation"] = operation
            
        super().__init__(
            message=message,
            agent_type=agent_type,
            agent_id=agent_id,
            error_code=ErrorCode.AGENT_TIMEOUT,
            details=details
        )


class FactoryError(CoreError):
    """Exception raised for factory-related errors."""
    
    def __init__(
        self,
        message: str,
        factory_name: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        details["factory_name"] = factory_name
        
        super().__init__(
            ErrorCode.SYSTEM_ERROR,
            message,
            details,
            original_error
        )


class RegistryError(CoreError):
    """Exception raised for registry-related errors."""
    
    def __init__(
        self,
        message: str,
        registry_name: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        details["registry_name"] = registry_name
        
        super().__init__(
            ErrorCode.SYSTEM_ERROR,
            message,
            details,
            original_error
        )


class CircuitBreakerError(CoreError):
    """Exception raised for circuit breaker-related errors."""
    
    def __init__(
        self,
        message: str,
        circuit_name: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        details["circuit_name"] = circuit_name
        
        super().__init__(
            ErrorCode.CIRCUIT_BREAKER_OPEN,
            message,
            details,
            original_error
        )


class BackpressureError(CoreError):
    """Exception raised for backpressure-related errors."""
    
    def __init__(
        self,
        message: str,
        controller_name: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        details["controller_name"] = controller_name
        
        super().__init__(
            ErrorCode.SYSTEM_ERROR,
            message,
            details,
            original_error
        )


class WorkerPoolError(CoreError):
    """Exception raised for worker pool-related errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            ErrorCode.WORKER_ERROR,
            message,
            details,
            original_error
        )


class SerializationError(CoreError):
    """Exception raised for serialization-related errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            ErrorCode.SYSTEM_ERROR,
            message,
            details,
            original_error
        )


class ValidationError(CoreError):
    """Exception raised for validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        if field:
            details["field"] = field
            
        super().__init__(
            ErrorCode.VALIDATION_ERROR,
            message,
            details,
            original_error
        )


class ConfigurationError(CoreError):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        if details is None:
            details = {}
            
        if component:
            details["component"] = component
            
        super().__init__(
            ErrorCode.CONFIG_ERROR,
            message,
            details,
            original_error
        )
        
def safe_message(self) -> str:
    """Return a safe version of the error message."""
    from src.config.settings import get_settings
    
    if get_settings().ENVIRONMENT == "development":
        return self.message
    
    # Generic messages for production
    if isinstance(self, TaskError):
        return "An error occurred while processing your task."
    elif isinstance(self, AgentError):
        return "The system encountered an issue with the requested operation."
    # Other error types...
    else:
        return "An unexpected error occurred."
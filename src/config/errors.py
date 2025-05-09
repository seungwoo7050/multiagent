from enum import Enum
from typing import Any, Dict, Optional, Type, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ErrorCode(str, Enum):
    SYSTEM_ERROR = 'SYSTEM_ERROR_1000'
    CONFIG_ERROR = 'CONFIG_ERROR_1001'
    INITIALIZATION_ERROR = 'INITIALIZATION_ERROR_1002'
    SHUTDOWN_ERROR = 'SHUTDOWN_ERROR_1003'
    API_ERROR = 'API_ERROR_2000'
    VALIDATION_ERROR = 'VALIDATION_ERROR_2001'
    AUTHENTICATION_ERROR = 'AUTHENTICATION_ERROR_2002'
    AUTHORIZATION_ERROR = 'AUTHORIZATION_ERROR_2003'
    RATE_LIMIT_ERROR = 'RATE_LIMIT_ERROR_2004'
    ENDPOINT_NOT_FOUND = 'ENDPOINT_NOT_FOUND_2005'
    BAD_REQUEST = 'BAD_REQUEST_2006'
    TASK_ERROR = 'TASK_ERROR_3000'
    TASK_NOT_FOUND = 'TASK_NOT_FOUND_3001'
    TASK_CREATION_ERROR = 'TASK_CREATION_ERROR_3002'
    TASK_EXECUTION_ERROR = 'TASK_EXECUTION_ERROR_3003'
    TASK_TIMEOUT = 'TASK_TIMEOUT_3004'
    TASK_CANCELED = 'TASK_CANCELED_3005'
    LLM_ERROR = 'LLM_ERROR_4000'
    LLM_API_ERROR = 'LLM_API_ERROR_4001'
    LLM_TIMEOUT = 'LLM_TIMEOUT_4002'
    LLM_RATE_LIMIT = 'LLM_RATE_LIMIT_4003'
    LLM_CONTENT_FILTER = 'LLM_CONTENT_FILTER_4004'
    LLM_CONTEXT_LIMIT = 'LLM_CONTEXT_LIMIT_4005'
    LLM_TOKEN_LIMIT = 'LLM_TOKEN_LIMIT_4006'
    LLM_PROVIDER_ERROR = 'LLM_PROVIDER_ERROR_4007'
    MEMORY_ERROR = 'MEMORY_ERROR_5000'
    REDIS_CONNECTION_ERROR = 'REDIS_CONNECTION_ERROR_5001'
    REDIS_OPERATION_ERROR = 'REDIS_OPERATION_ERROR_5002'
    MEMORY_RETRIEVAL_ERROR = 'MEMORY_RETRIEVAL_ERROR_5003'
    MEMORY_STORAGE_ERROR = 'MEMORY_STORAGE_ERROR_5004'
    VECTOR_DB_ERROR = 'VECTOR_DB_ERROR_5005'
    AGENT_ERROR = 'AGENT_ERROR_6000'
    AGENT_NOT_FOUND = 'AGENT_NOT_FOUND_6001'
    AGENT_CREATION_ERROR = 'AGENT_CREATION_ERROR_6002'
    AGENT_EXECUTION_ERROR = 'AGENT_EXECUTION_ERROR_6003'
    AGENT_TIMEOUT = 'AGENT_TIMEOUT_6004'
    TOOL_ERROR = 'TOOL_ERROR_7000'
    TOOL_NOT_FOUND = 'TOOL_NOT_FOUND_7001'
    TOOL_EXECUTION_ERROR = 'TOOL_EXECUTION_ERROR_7002'
    TOOL_TIMEOUT = 'TOOL_TIMEOUT_7003'
    TOOL_VALIDATION_ERROR = 'TOOL_VALIDATION_ERROR_7004'
    ORCHESTRATION_ERROR = 'ORCHESTRATION_ERROR_8000'
    WORKFLOW_ERROR = 'WORKFLOW_ERROR_8001'
    DISPATCHER_ERROR = 'DISPATCHER_ERROR_8002'
    WORKER_ERROR = 'WORKER_ERROR_8003'
    CIRCUIT_BREAKER_OPEN = 'CIRCUIT_BREAKER_OPEN_8004'
    CONNECTION_ERROR = 'CONNECTION_ERROR_9000'
    HTTP_ERROR = 'HTTP_ERROR_9001'
    NETWORK_ERROR = 'NETWORK_ERROR_9002'
    TIMEOUT_ERROR = 'TIMEOUT_ERROR_9003'

class BaseError(Exception):

    def __init__(self, code: Union[ErrorCode, str], message: str, details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None):
        self.code = code
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        error_dict = {'code': self.code.value if isinstance(self.code, ErrorCode) else self.code, 'message': self.message}
        if self.details:
            error_dict['details'] = self.details
        if self.original_error:
            error_dict['original_error'] = str(self.original_error)
        return error_dict

    def log_error(self, logger_instance=None):
        logger_to_use = logger_instance or logger
        error_dict = self.to_dict()
        error_code_str = self.code.value if isinstance(self.code, ErrorCode) else self.code
        if self.original_error:
            logger_to_use.error(f'{error_code_str}: {self.message}', extra={'error_details': error_dict}, exc_info=self.original_error)
        else:
            logger_to_use.error(f'{error_code_str}: {self.message}', extra={'error_details': error_dict})

class SystemError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.SYSTEM_ERROR, message: str='A system error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None):
        super().__init__(code, message, details, original_error)

class APIError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.API_ERROR, message: str='An API error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None, status_code: int=500):
        self.status_code = status_code
        super().__init__(code, message, details, original_error)

    def to_dict(self) -> Dict[str, Any]:
        error_dict = super().to_dict()
        error_dict['status_code'] = self.status_code
        return error_dict

class ValidationError(APIError):

    def __init__(self, message: str='Validation error', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None):
        super().__init__(ErrorCode.VALIDATION_ERROR, message, details, original_error, status_code=400)

class NotFoundError(APIError):

    def __init__(self, resource_type: str, resource_id: str, message: Optional[str]=None, details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None):
        if not message:
            message = f"{resource_type} with ID '{resource_id}' not found"
        if not details:
            details = {'resource_type': resource_type, 'resource_id': resource_id}
        super().__init__(ErrorCode.ENDPOINT_NOT_FOUND, message, details, original_error, status_code=404)

class TaskError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.TASK_ERROR, message: str='A task error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None, task_id: Optional[str]=None):
        if task_id:
            if details is None:
                details = {'task_id': task_id}
            else:
                details['task_id'] = task_id
        super().__init__(code, message, details, original_error)

class LLMError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.LLM_ERROR, message: str='An LLM error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None, model: Optional[str]=None, provider: Optional[str]=None):
        if details is None:
            details = {}
        if model:
            details['model'] = model
        if provider:
            details['provider'] = provider
        super().__init__(code, message, details, original_error)

class MemoryError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.MEMORY_ERROR, message: str='A memory error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None):
        super().__init__(code, message, details, original_error)

class AgentError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.AGENT_ERROR, message: str='An agent error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None, agent_type: Optional[str]=None, agent_id: Optional[str]=None):
        if details is None:
            details = {}
        if agent_type:
            details['agent_type'] = agent_type
        if agent_id:
            details['agent_id'] = agent_id
        super().__init__(code, message, details, original_error)

class ToolError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.TOOL_ERROR, message: str='A tool error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None, tool_name: Optional[str]=None):
        if details is None:
            details = {}
        if tool_name:
            details['tool_name'] = tool_name
        super().__init__(code, message, details, original_error)

class OrchestrationError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.ORCHESTRATION_ERROR, message: str='An orchestration error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None):
        super().__init__(code, message, details, original_error)

class ConnectionError(BaseError):

    def __init__(self, code: Union[ErrorCode, str]=ErrorCode.CONNECTION_ERROR, message: str='A connection error occurred', details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None, service: Optional[str]=None):
        if details is None:
            details = {}
        if service:
            details['service'] = service
        super().__init__(code, message, details, original_error)
ERROR_CLASS_REGISTRY: Dict[ErrorCode, Type[BaseError]] = {ErrorCode.SYSTEM_ERROR: SystemError, ErrorCode.CONFIG_ERROR: SystemError, ErrorCode.INITIALIZATION_ERROR: SystemError, ErrorCode.SHUTDOWN_ERROR: SystemError, ErrorCode.API_ERROR: APIError, ErrorCode.VALIDATION_ERROR: ValidationError, ErrorCode.AUTHENTICATION_ERROR: APIError, ErrorCode.AUTHORIZATION_ERROR: APIError, ErrorCode.RATE_LIMIT_ERROR: APIError, ErrorCode.ENDPOINT_NOT_FOUND: NotFoundError, ErrorCode.BAD_REQUEST: APIError, ErrorCode.TASK_ERROR: TaskError, ErrorCode.TASK_NOT_FOUND: TaskError, ErrorCode.TASK_CREATION_ERROR: TaskError, ErrorCode.TASK_EXECUTION_ERROR: TaskError, ErrorCode.TASK_TIMEOUT: TaskError, ErrorCode.TASK_CANCELED: TaskError, ErrorCode.LLM_ERROR: LLMError, ErrorCode.LLM_API_ERROR: LLMError, ErrorCode.LLM_TIMEOUT: LLMError, ErrorCode.LLM_RATE_LIMIT: LLMError, ErrorCode.LLM_CONTENT_FILTER: LLMError, ErrorCode.LLM_CONTEXT_LIMIT: LLMError, ErrorCode.LLM_TOKEN_LIMIT: LLMError, ErrorCode.LLM_PROVIDER_ERROR: LLMError, ErrorCode.MEMORY_ERROR: MemoryError, ErrorCode.REDIS_CONNECTION_ERROR: MemoryError, ErrorCode.REDIS_OPERATION_ERROR: MemoryError, ErrorCode.MEMORY_RETRIEVAL_ERROR: MemoryError, ErrorCode.MEMORY_STORAGE_ERROR: MemoryError, ErrorCode.VECTOR_DB_ERROR: MemoryError, ErrorCode.AGENT_ERROR: AgentError, ErrorCode.AGENT_NOT_FOUND: AgentError, ErrorCode.AGENT_CREATION_ERROR: AgentError, ErrorCode.AGENT_EXECUTION_ERROR: AgentError, ErrorCode.AGENT_TIMEOUT: AgentError, ErrorCode.TOOL_ERROR: ToolError, ErrorCode.TOOL_NOT_FOUND: ToolError, ErrorCode.TOOL_EXECUTION_ERROR: ToolError, ErrorCode.TOOL_TIMEOUT: ToolError, ErrorCode.TOOL_VALIDATION_ERROR: ToolError, ErrorCode.ORCHESTRATION_ERROR: OrchestrationError, ErrorCode.WORKFLOW_ERROR: OrchestrationError, ErrorCode.DISPATCHER_ERROR: OrchestrationError, ErrorCode.WORKER_ERROR: OrchestrationError, ErrorCode.CIRCUIT_BREAKER_OPEN: OrchestrationError, ErrorCode.CONNECTION_ERROR: ConnectionError, ErrorCode.HTTP_ERROR: ConnectionError, ErrorCode.NETWORK_ERROR: ConnectionError, ErrorCode.TIMEOUT_ERROR: ConnectionError}

def create_error_from_code(code: Union[ErrorCode, str], message: str, details: Optional[Dict[str, Any]]=None, original_error: Optional[Exception]=None, **kwargs) -> BaseError:
    if isinstance(code, str):
        try:
            code = ErrorCode(code)
        except ValueError:
            logger.warning(f"Invalid error code string '{code}'. Using SYSTEM_ERROR.")
            code = ErrorCode.SYSTEM_ERROR
    error_class = ERROR_CLASS_REGISTRY.get(code, BaseError)
    if error_class == ValidationError:
        return error_class(message=message, details=details, original_error=original_error)
    elif error_class == NotFoundError:
        resource_id = kwargs.get('resource_id', 'unknown')
        resource_type = kwargs.get('resource_type', 'unknown')
        return error_class(resource_type=resource_type, resource_id=resource_id, message=message, details=details, original_error=original_error)
    else:
        return error_class(code=code, message=message, details=details, original_error=original_error, **kwargs)

def convert_exception(exception: Exception, default_code: ErrorCode=ErrorCode.SYSTEM_ERROR, default_message: Optional[str]=None) -> BaseError:
    if isinstance(exception, BaseError):
        return exception
    message = default_message or str(exception)
    return create_error_from_code(code=default_code, message=message, original_error=exception)
RETRYABLE_ERRORS = [ErrorCode.LLM_TIMEOUT, ErrorCode.LLM_RATE_LIMIT, ErrorCode.REDIS_CONNECTION_ERROR, ErrorCode.CONNECTION_ERROR, ErrorCode.HTTP_ERROR, ErrorCode.NETWORK_ERROR, ErrorCode.TIMEOUT_ERROR]
ERROR_TO_HTTP_STATUS = {ErrorCode.VALIDATION_ERROR: 400, ErrorCode.BAD_REQUEST: 400, ErrorCode.AUTHENTICATION_ERROR: 401, ErrorCode.AUTHORIZATION_ERROR: 403, ErrorCode.ENDPOINT_NOT_FOUND: 404, ErrorCode.TASK_NOT_FOUND: 404, ErrorCode.AGENT_NOT_FOUND: 404, ErrorCode.TOOL_NOT_FOUND: 404, ErrorCode.RATE_LIMIT_ERROR: 429, ErrorCode.LLM_RATE_LIMIT: 429, ErrorCode.LLM_CONTEXT_LIMIT: 400, ErrorCode.LLM_TOKEN_LIMIT: 400, ErrorCode.SYSTEM_ERROR: 500, ErrorCode.CONFIG_ERROR: 500, ErrorCode.INITIALIZATION_ERROR: 500, ErrorCode.SHUTDOWN_ERROR: 500, ErrorCode.API_ERROR: 500, ErrorCode.TASK_ERROR: 500, ErrorCode.TASK_CREATION_ERROR: 500, ErrorCode.TASK_EXECUTION_ERROR: 500, ErrorCode.LLM_ERROR: 502, ErrorCode.LLM_API_ERROR: 502, ErrorCode.LLM_PROVIDER_ERROR: 502, ErrorCode.MEMORY_ERROR: 500, ErrorCode.REDIS_CONNECTION_ERROR: 503, ErrorCode.REDIS_OPERATION_ERROR: 500, ErrorCode.MEMORY_RETRIEVAL_ERROR: 500, ErrorCode.MEMORY_STORAGE_ERROR: 500, ErrorCode.VECTOR_DB_ERROR: 503, ErrorCode.AGENT_ERROR: 500, ErrorCode.AGENT_CREATION_ERROR: 500, ErrorCode.AGENT_EXECUTION_ERROR: 500, ErrorCode.TOOL_ERROR: 500, ErrorCode.TOOL_EXECUTION_ERROR: 500, ErrorCode.TOOL_VALIDATION_ERROR: 400, ErrorCode.ORCHESTRATION_ERROR: 500, ErrorCode.WORKFLOW_ERROR: 500, ErrorCode.DISPATCHER_ERROR: 500, ErrorCode.WORKER_ERROR: 500, ErrorCode.CIRCUIT_BREAKER_OPEN: 503, ErrorCode.CONNECTION_ERROR: 503, ErrorCode.HTTP_ERROR: 502, ErrorCode.NETWORK_ERROR: 504, ErrorCode.TIMEOUT_ERROR: 504, ErrorCode.LLM_TIMEOUT: 504, ErrorCode.TASK_TIMEOUT: 504, ErrorCode.AGENT_TIMEOUT: 504, ErrorCode.TOOL_TIMEOUT: 504, ErrorCode.LLM_CONTENT_FILTER: 451}
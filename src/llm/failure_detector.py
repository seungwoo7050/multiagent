from src.config.errors import ErrorCode, LLMError
from src.config.logger import get_logger

logger = get_logger(__name__)
FALLBACK_IMMEDIATELY_CODES = {ErrorCode.AUTHENTICATION_ERROR, ErrorCode.AUTHORIZATION_ERROR, ErrorCode.LLM_PROVIDER_ERROR, ErrorCode.BAD_REQUEST, ErrorCode.LLM_CONTENT_FILTER, ErrorCode.LLM_CONTEXT_LIMIT, ErrorCode.LLM_TOKEN_LIMIT}

def should_fallback_immediately(error: Exception) -> bool:
    if isinstance(error, LLMError):
        error_code = error.code
        code_value = error_code.value if isinstance(error_code, ErrorCode) else error_code
        if error_code in FALLBACK_IMMEDIATELY_CODES:
            logger.debug(f'Error code {code_value} requires immediate fallback.')
            return True
        if error.code == ErrorCode.LLM_API_ERROR and error.details:
            status_code = error.details.get('status_code')
            if status_code in [400, 401, 403, 404, 413]:
                logger.debug(f'LLM API Error status code {status_code} suggests immediate fallback.')
                return True
    logger.debug(f'Error type {type(error).__name__} considered potentially retryable (or fallback after retries).')
    return False
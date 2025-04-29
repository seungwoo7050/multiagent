# import os
# import json
# import sys
# import logging
# from typing import Dict, List, Optional, Set, Literal, Union, Any
# from functools import lru_cache
# from copy import deepcopy
# from pydantic import Field, field_validator, ConfigDict, ValidationInfo
# from pydantic_settings import BaseSettings
# logger = logging.getLogger(__name__)

# class LLMProviderConfig(BaseSettings):
#     api_key: str = ''
#     api_base: Optional[str] = None
#     timeout: float = 60.0
#     max_retries: int = 3
#     connection_pool_size: int = 10

# class Settings(BaseSettings):
#     APP_NAME: str = 'WooMultiAgent'
#     APP_VERSION: str = '0.1.0'
#     ENVIRONMENT: Literal['development', 'production'] = 'development'
#     DEBUG: bool = False
#     LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
#     LOG_FORMAT: Literal['json', 'text'] = 'json'
#     LOG_TO_FILE: bool = False
#     LOG_FILE_PATH: Optional[str] = None
#     WORKER_COUNT: int = Field(default_factory=lambda: max(os.cpu_count() or 1, 1))
#     MAX_CONCURRENT_TASKS: int = 100
#     REQUEST_TIMEOUT: float = 30.0
#     ENABLE_PERFORMANCE_TRACKING: bool = True
#     REDIS_URL: str = 'redis://localhost:6379/0'
#     REDIS_PASSWORD: Optional[str] = None
#     REDIS_CONNECTION_POOL_SIZE: int = 20
#     MEMORY_TTL: int = 86400
#     CACHE_TTL: int = 3600
#     PRIMARY_LLM: str = 'gpt-3.5-turbo'
#     FALLBACK_LLM: str = 'gpt-3.5-turbo'
#     LLM_RETRY_MAX_ATTEMPTS: int = 3
#     LLM_RETRY_BACKOFF_FACTOR: float = 0.5
#     LLM_RETRY_JITTER: bool = True
#     ENABLED_MODELS_SET: Union[str, Set[str]] = Field(default_factory=lambda: {'gpt-3.5-turbo', 'gpt-4o', 'claude-3-opus', 'claude-3-sonnet'})
#     LLM_MODEL_PROVIDER_MAP: Dict[str, str] = {'gpt-3.5-turbo': 'openai', 'gpt-4o': 'openai', 'claude-3-opus': 'anthropic', 'claude-3-sonnet': 'anthropic'}
#     LLM_PROVIDERS_CONFIG: Dict[str, Dict[str, Any]] = {'openai': {'api_key': '', 'timeout': 30.0, 'connection_pool_size': 10}, 'anthropic': {'api_key': '', 'timeout': 30.0, 'connection_pool_size': 10}}
#     API_HOST: str = 'localhost'
#     API_PORT: int = 8000
#     API_PREFIX: str = '/api/v1'
#     CORS_ORIGINS: List[str] = ['*']
#     AUTH_REQUIRED: bool = False
#     AUTH_TOKEN_EXPIRY: int = 86400
#     METRICS_ENABLED: bool = True
#     METRICS_PORT: int = 9090
#     VECTOR_DB_URL: Optional[str] = None
#     VECTOR_DB_TYPE: Literal['chroma', 'qdrant', 'faiss', 'none'] = 'none'

#     @field_validator('ENABLED_MODELS_SET', mode='before')
#     @classmethod
#     def parse_enabled_models(cls, v: Union[str, List, Set]) -> Set[str]:
#         if isinstance(v, str):
#             try:
#                 parsed_list = json.loads(v)
#                 if isinstance(parsed_list, list):
#                     return {str(model).strip() for model in parsed_list if str(model).strip()}
#             except json.JSONDecodeError:
#                 if not v:
#                     return set()
#                 return set((model.strip() for model in v.split(',') if str(model).strip()))
#         elif isinstance(v, list):
#             return {str(model).strip() for model in v if str(model).strip()}
#         elif isinstance(v, set):
#             return {str(model).strip() for model in v if str(model).strip()}
#         raise ValueError('Invalid value for ENABLED_MODELS_SET. Expected string, list, or set.')

#     @field_validator('LLM_PROVIDERS_CONFIG', mode='before')
#     @classmethod
#     def parse_provider_configs(cls, v: Union[str, Dict[str, Dict[str, Any]]], info: ValidationInfo) -> Dict[str, Dict[str, Any]]:
#         final_processed_config: Dict[str, Dict[str, Any]] = {}
#         default_config = deepcopy(cls.model_fields[info.field_name].default or {})
#         parsed_config: Dict[str, Dict[str, Any]] = {}
#         if isinstance(v, str):
#             if not v:
#                 parsed_config = {}
#             else:
#                 try:
#                     loaded_json = json.loads(v)
#                     if not isinstance(loaded_json, dict):
#                         raise ValueError('LLM_PROVIDERS_CONFIG JSON string must resolve to a dictionary')
#                     parsed_config = loaded_json
#                 except json.JSONDecodeError:
#                     raise ValueError('Invalid JSON string for LLM_PROVIDERS_CONFIG')
#         elif isinstance(v, dict):
#             parsed_config = deepcopy(v)
#         else:
#             raise ValueError('LLM_PROVIDERS_CONFIG must be a dictionary or a valid JSON string representation of a dictionary')
#         merged_config = deepcopy(default_config)
#         for provider, config_values in parsed_config.items():
#             if isinstance(config_values, dict):
#                 merged_config.setdefault(provider, {}).update(config_values)
#             else:
#                 logger.warning(f"LLM_PROVIDERS_CONFIG for provider '{provider}' is not a dictionary. Using default values if available.")
#         for provider, config_values in merged_config.items():
#             if not isinstance(config_values, dict):
#                 config_values = {}
#             env_key = f'{provider.upper()}_API_KEY'
#             api_key_from_env = os.getenv(env_key)
#             if api_key_from_env:
#                 config_values['api_key'] = api_key_from_env
#             elif 'api_key' not in config_values:
#                 config_values['api_key'] = ''
#             final_processed_config[provider] = config_values
#         return final_processed_config
#     model_config = ConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

# @lru_cache
# def get_settings() -> Settings:
#     try:
#         settings = Settings()
#         return settings
#     except Exception as e:
#         print(f'FATAL: Failed to load settings: {e}', file=sys.stderr)
#         raise

"""
Settings management for the Multi-Agent Platform.
Uses Pydantic for validation and environment variable loading.
"""
import os
import json
import sys
import logging
from typing import Dict, List, Optional, Set, Literal, Union, Any
from functools import lru_cache
from copy import deepcopy
from pydantic import Field, field_validator, ConfigDict, ValidationInfo
from pydantic_settings import BaseSettings

# Use a simple logger for bootstrap operations
_logger = logging.getLogger("config.settings")


class LLMProviderConfig(BaseSettings):
    """Configuration for an LLM provider."""
    api_key: str = ''
    api_base: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    connection_pool_size: int = 10


class Settings(BaseSettings):
    """
    Application settings with validation.
    
    Loads from environment variables and .env file.
    """
    # Application info
    APP_NAME: str = 'MultiAgentPlatform'
    APP_VERSION: str = '0.1.0'
    ENVIRONMENT: Literal['development', 'production'] = 'development'
    DEBUG: bool = False
    
    # Logging settings
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    LOG_FORMAT: Literal['json', 'text'] = 'json'
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: Optional[str] = None
    
    # Worker settings
    WORKER_COUNT: int = Field(default_factory=lambda: max(os.cpu_count() or 1, 1))
    MAX_CONCURRENT_TASKS: int = 100
    
    # Connection settings
    REQUEST_TIMEOUT: float = 30.0
    ENABLE_PERFORMANCE_TRACKING: bool = True
    
    # Redis settings
    REDIS_URL: str = 'redis://localhost:6379/0'
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CONNECTION_POOL_SIZE: int = 20
    
    # Memory settings
    MEMORY_TTL: int = 86400  # 1 day in seconds
    CACHE_TTL: int = 3600    # 1 hour in seconds
    
    # LLM settings
    PRIMARY_LLM: str = 'gpt-3.5-turbo'
    FALLBACK_LLM: str = 'gpt-3.5-turbo'
    LLM_RETRY_MAX_ATTEMPTS: int = 3
    LLM_RETRY_BACKOFF_FACTOR: float = 0.5
    LLM_RETRY_JITTER: bool = True
    
    # Enabled models
    ENABLED_MODELS_SET: Union[str, Set[str]] = Field(
        default_factory=lambda: {'gpt-3.5-turbo', 'gpt-4o', 'claude-3-opus', 'claude-3-sonnet'}
    )
    
    # Model-provider mapping
    LLM_MODEL_PROVIDER_MAP: Dict[str, str] = {
        'gpt-3.5-turbo': 'openai',
        'gpt-4o': 'openai',
        'claude-3-opus': 'anthropic',
        'claude-3-sonnet': 'anthropic'
    }
    
    # LLM provider configurations
    LLM_PROVIDERS_CONFIG: Dict[str, Dict[str, Any]] = {
        'openai': {
            'api_key': '',
            'timeout': 30.0,
            'connection_pool_size': 10
        },
        'anthropic': {
            'api_key': '',
            'timeout': 30.0,
            'connection_pool_size': 10
        }
    }
    
    # API settings
    API_HOST: str = 'localhost'
    API_PORT: int = 8000
    API_PREFIX: str = '/api/v1'
    CORS_ORIGINS: List[str] = ['*']
    AUTH_REQUIRED: bool = False
    AUTH_TOKEN_EXPIRY: int = 86400  # 1 day in seconds
    
    # Metrics settings
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    
    # Vector DB settings
    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_TYPE: Literal['chroma', 'qdrant', 'faiss', 'none'] = 'none'

    @field_validator('ENABLED_MODELS_SET', mode='before')
    @classmethod
    def parse_enabled_models(cls, v: Union[str, List, Set]) -> Set[str]:
        """
        Parse the ENABLED_MODELS_SET from various input formats.
        
        Accepts:
        - JSON string list: '["model1", "model2"]'
        - Comma-separated string: "model1,model2"
        - Python list: ["model1", "model2"]
        - Python set: {"model1", "model2"}
        
        Returns:
            Set[str]: Set of enabled model names
        """
        try:
            if isinstance(v, str):
                # Try to parse as JSON
                try:
                    parsed_list = json.loads(v)
                    if isinstance(parsed_list, list):
                        return {str(model).strip() for model in parsed_list if str(model).strip()}
                except json.JSONDecodeError:
                    # If not JSON, try as comma-separated string
                    if not v:
                        return set()
                    return {model.strip() for model in v.split(',') if model.strip()}
            elif isinstance(v, (list, set)):
                return {str(model).strip() for model in v if str(model).strip()}
            
            # If we get here, the format wasn't recognized
            _logger.warning(f"Unrecognized format for ENABLED_MODELS_SET: {type(v)}")
            return set()
        except Exception as e:
            _logger.error(f"Error parsing ENABLED_MODELS_SET: {str(e)}")
            raise ValueError(f'Invalid value for ENABLED_MODELS_SET: {str(e)}')

    @field_validator('LLM_PROVIDERS_CONFIG', mode='before')
    @classmethod
    def parse_provider_configs(cls, v: Union[str, Dict[str, Dict[str, Any]]], 
                               info: ValidationInfo) -> Dict[str, Dict[str, Any]]:
        """
        Parse and validate the LLM provider configurations.
        
        Merges defaults with provided values and loads API keys from environment.
        
        Args:
            v: The provider config value (string or dict)
            info: Validation context information
            
        Returns:
            Dict[str, Dict[str, Any]]: Processed provider configurations
        """
        try:
            final_config: Dict[str, Dict[str, Any]] = {}
            default_config = deepcopy(cls.model_fields[info.field_name].default or {})
            parsed_config: Dict[str, Dict[str, Any]] = {}
            
            # Parse the input value
            if isinstance(v, str):
                if not v:
                    parsed_config = {}
                else:
                    try:
                        loaded_json = json.loads(v)
                        if not isinstance(loaded_json, dict):
                            raise ValueError('LLM_PROVIDERS_CONFIG JSON string must resolve to a dictionary')
                        parsed_config = loaded_json
                    except json.JSONDecodeError:
                        raise ValueError('Invalid JSON string for LLM_PROVIDERS_CONFIG')
            elif isinstance(v, dict):
                parsed_config = deepcopy(v)
            else:
                raise ValueError('LLM_PROVIDERS_CONFIG must be a dictionary or a valid JSON string')
            
            # Merge with defaults
            merged_config = deepcopy(default_config)
            for provider, config_values in parsed_config.items():
                if isinstance(config_values, dict):
                    merged_config.setdefault(provider, {}).update(config_values)
                else:
                    _logger.warning(
                        f"LLM_PROVIDERS_CONFIG for provider '{provider}' is not a dictionary. "
                        "Using default values."
                    )
            
            # Process each provider config
            for provider, config_values in merged_config.items():
                if not isinstance(config_values, dict):
                    config_values = {}
                
                # Check for provider-specific API key in environment
                env_key = f'{provider.upper()}_API_KEY'
                api_key_from_env = os.getenv(env_key)
                
                if api_key_from_env:
                    config_values['api_key'] = api_key_from_env
                elif 'api_key' not in config_values:
                    config_values['api_key'] = ''
                
                final_config[provider] = config_values
            
            return final_config
        except Exception as e:
            _logger.error(f"Error parsing LLM_PROVIDERS_CONFIG: {str(e)}")
            raise ValueError(f'Invalid value for LLM_PROVIDERS_CONFIG: {str(e)}')

    # Pydantic configuration
    model_config = ConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    def validate_settings(self) -> List[str]:
        """
        Perform additional validation beyond Pydantic's field validation.
        
        Returns:
            List[str]: List of validation warnings/errors
        """
        warnings = []
        
        # Check if PRIMARY_LLM is in enabled models
        if self.PRIMARY_LLM not in self.ENABLED_MODELS_SET:
            warnings.append(
                f"PRIMARY_LLM '{self.PRIMARY_LLM}' is not in ENABLED_MODELS_SET. "
                "This may cause fallbacks."
            )
            
        # Check if FALLBACK_LLM is in enabled models
        if self.FALLBACK_LLM not in self.ENABLED_MODELS_SET:
            warnings.append(
                f"FALLBACK_LLM '{self.FALLBACK_LLM}' is not in ENABLED_MODELS_SET. "
                "This may cause errors when fallbacks are needed."
            )
            
        # Check for API keys for used providers
        used_providers = set(self.LLM_MODEL_PROVIDER_MAP.values())
        for provider in used_providers:
            if provider in self.LLM_PROVIDERS_CONFIG:
                if not self.LLM_PROVIDERS_CONFIG[provider].get('api_key'):
                    warnings.append(
                        f"No API key found for provider '{provider}'. "
                        f"Set {provider.upper()}_API_KEY environment variable."
                    )
            else:
                warnings.append(
                    f"Provider '{provider}' is referenced in LLM_MODEL_PROVIDER_MAP "
                    "but has no configuration in LLM_PROVIDERS_CONFIG."
                )
                
        return warnings


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings, using cache for performance.
    
    Returns:
        Settings: Application settings instance
    
    Raises:
        Exception: If settings cannot be loaded
    """
    try:
        settings = Settings()
        
        # Perform additional validation
        warnings = settings.validate_settings()
        for warning in warnings:
            _logger.warning(warning)
            
        return settings
    except Exception as e:
        error_msg = f'Failed to load settings: {e}'
        _logger.error(error_msg, exc_info=True)
        print(f'FATAL: {error_msg}', file=sys.stderr)
        raise
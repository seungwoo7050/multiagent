import os
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Set, Union, Any
from pydantic import Field, field_validator, ValidationInfo, ConfigDict, BaseModel
from pydantic_settings import BaseSettings
import json
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)            

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

class LLMProviderSettings(BaseModel):
    api_key: str = Field(..., alias="API_KEY", description="LLM API 키")
    model_name: str = Field(..., alias="MODEL_NAME", description="사용할 LLM 모델 이름")
    endpoint: Optional[str] = Field(None, alias="ENDPOINT")
    model_config = ConfigDict(
        case_sensitive=False,
        populate_by_name=True,                        
    )

class AppSettings(BaseSettings):
    """
    Application settings schema loaded from environment variables.
    Defines all configuration variables for the application.
    """
              
    APP_NAME: str = 'MultiAgentPlatform'
    APP_VERSION: str = '0.1.0'
    ENVIRONMENT: Literal['development', 'production'] = 'development'
    DEBUG: bool = False
    
             
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    LOG_FORMAT: Literal['json', 'text'] = 'json'
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: Optional[str] = None
    
    AGENT_GRAPH_CONFIG_DIR: str = Field(
        default=str(PROJECT_ROOT_DIR / "config" / "agent_graphs"),
        description="Directory for dynamic agent graph configurations (e.g., JSON files)"
    )
    PROMPT_TEMPLATE_DIR: str = Field(                       
        default=str(PROJECT_ROOT_DIR / "config" / "prompts"),
        description="Directory for prompt template files"
    )
    
    def load_graph_config(self, graph_name: str) -> dict:
        """
        지정한 그래프 이름의 JSON 파일을 읽어서 dict 로 반환합니다.
        파일이 없거나 파싱 오류 시 예외를 발생시킵니다.
        graph_name에 .json 확장자가 있든 없든 처리합니다.
        """
        if not graph_name.endswith(".json"):
            config_file_name = f"{graph_name}.json"
        else:
            config_file_name = graph_name

        path = Path(self.AGENT_GRAPH_CONFIG_DIR) / config_file_name         

        _logger.debug(f"Attempting to load graph config from: {path}")            
        if not path.exists():
            _logger.error(f"Graph config file not found at path: {path}")
            raise FileNotFoundError(f"Graph config not found: {path}")
        try:
            config_data = json.loads(path.read_text(encoding="utf-8"))
            _logger.debug(f"Successfully loaded and parsed graph config: {config_file_name}")
            return config_data
        except json.JSONDecodeError as e:
            _logger.error(f"JSONDecodeError for graph config {config_file_name}: {e}", exc_info=True)
            raise
    
                        
    WORKER_COUNT: int = Field(default_factory=lambda: max(os.cpu_count() or 1, 1))
    MAX_CONCURRENT_TASKS: int = 100
    TASK_STATUS_TTL: int = 86400                    

                            
    REQUEST_TIMEOUT: float = Field(60.0, alias='DEFAULT_REQUEST_TIMEOUT')                   
    ENABLE_PERFORMANCE_TRACKING: bool = True

                                                       
    REDIS_URL: str = 'redis://localhost:6379/0'
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CONNECTION_POOL_SIZE: int = 20
                                       
                                                 
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
             

                      
    MEMORY_TTL: int = 86400
    CACHE_TTL: int = 3600
    MEMORY_MANAGER_CACHE_SIZE: int = 10000
    MEMORY_MANAGER_CHAT_HISTORY_PREFIX: str = Field(default="chat_history", description="Prefix for chat history keys in memory manager")

                
    PRIMARY_LLM_PROVIDER: str = Field(..., description="기본 LLM 제공자 (openai, anthropic 등)")
    FALLBACK_LLM_PROVIDER: Optional[str] = Field(None, description="폴백 LLM 제공자 (선택 사항)")
    LLM_PROVIDERS: Dict[str, LLMProviderSettings] = Field(..., description="LLM 제공자별 설정")
    LLM_REQUEST_TIMEOUT: int = Field(60, description="LLM 요청 타임아웃(초)")
    LLM_MAX_RETRIES: int = Field(3, description="LLM 요청 최대 재시도 횟수")
    
                 
    PLANNER_AGENT_NAME: str = 'default_planner'
    EXECUTOR_AGENT_NAME: str = 'default_executor'

                
    API_HOST: str = '0.0.0.0'                    
    API_PORT: int = 8000
    API_PREFIX: str = '/api/v1'
    CORS_ORIGINS: List[str] = ['*']

    WEBSOCKET_KEEP_ALIVE_INTERVAL: int = Field(60, description="WebSocket 연결 유지를 위한 서버 측 sleep 간격(초)")

                    
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090

                      
    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_TYPE: Literal['chroma', 'qdrant', 'faiss', 'none'] = 'none'

                       
    TASK_QUEUE_STREAM_NAME: str = "multi_agent_tasks"
    TASK_QUEUE_GROUP_NAME: str = "agent_workers"

    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(None, description='OpenTelemetry OTLP Exporter Endpoint')
    LANGCHAIN_TRACING_V2: bool = Field(False, description='Enable LangSmith tracing V2')
    LANGCHAIN_ENDPOINT: Optional[str] = Field("https://api.smith.langchain.com", description='LangSmith API Endpoint')
    LANGCHAIN_API_KEY: Optional[str] = Field(None, description='LangSmith API Key')
    LANGCHAIN_PROJECT: Optional[str] = Field(None, description='LangSmith Project Name')

    def validate_settings(self) -> List[str]:
        """Perform additional validation beyond Pydantic's field validation."""
        warnings = []
        if self.PRIMARY_LLM_PROVIDER not in self.LLM_PROVIDERS:
            warnings.append(f"PRIMARY_LLM_PROVIDER '{self.PRIMARY_LLM_PROVIDER}' not in LLM_PROVIDERS.")
        if self.FALLBACK_LLM_PROVIDER and self.FALLBACK_LLM_PROVIDER not in self.LLM_PROVIDERS:
            warnings.append(f"FALLBACK_LLM_PROVIDER '{self.FALLBACK_LLM_PROVIDER}' not in LLM_PROVIDERS.")
                                                                                      
        if self.REDIS_URL and (self.REDIS_HOST == 'localhost' or self.REDIS_PORT == 6379 or self.REDIS_DB == 0):
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(self.REDIS_URL)
                if parsed_url.hostname and self.REDIS_HOST == 'localhost':
                    self.REDIS_HOST = parsed_url.hostname
                if parsed_url.port and self.REDIS_PORT == 6379:
                    self.REDIS_PORT = parsed_url.port
                if parsed_url.path and len(parsed_url.path) > 1 and parsed_url.path[1:].isdigit() and self.REDIS_DB == 0:
                   self.REDIS_DB = int(parsed_url.path[1:])
            except Exception as parse_err:
                warnings.append(f"Could not parse REDIS_HOST/PORT/DB from REDIS_URL: {parse_err}")

        return warnings

    model_config = ConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,                       
        env_nested_delimiter='__'                                                 
    )
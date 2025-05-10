# src/schemas/config.py
import os
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Set, Union, Any
from pydantic import Field, field_validator, ValidationInfo, ConfigDict, BaseModel
from pydantic_settings import BaseSettings
import json
import logging
from pathlib import Path


_logger = logging.getLogger(__name__) # 로깅을 위해 추가

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

class LLMProviderSettings(BaseModel):
    api_key: str = Field(..., alias="API_KEY", description="LLM API 키")
    model_name: str = Field(..., alias="MODEL_NAME", description="사용할 LLM 모델 이름")
    endpoint: Optional[str] = Field(None, alias="ENDPOINT")
    model_config = ConfigDict(
        case_sensitive=False,
        populate_by_name=True,  # 필드 이름 또는 별칭으로 채우기 허용
    )

class AppSettings(BaseSettings):
    """
    Application settings schema loaded from environment variables.
    Defines all configuration variables for the application.
    """
    # App Info
    APP_NAME: str = 'MultiAgentPlatform'
    APP_VERSION: str = '0.1.0'
    ENVIRONMENT: Literal['development', 'production'] = 'development'
    DEBUG: bool = False
    
    # Logging
    LOG_LEVEL: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    LOG_FORMAT: Literal['json', 'text'] = 'json'
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: Optional[str] = None
    
    AGENT_GRAPH_CONFIG_DIR: str = Field(
        default=str(PROJECT_ROOT_DIR / "config" / "agent_graphs"),
        description="Directory for dynamic agent graph configurations (e.g., JSON files)"
    )
    PROMPT_TEMPLATE_DIR: str = Field( # 로드맵에 언급된 프롬프트 경로도 추가
        default=str(PROJECT_ROOT_DIR / "config" / "prompts"),
        description="Directory for prompt template files"
    )
    
    def load_graph_config(self, graph_name: str) -> dict:
        """
        지정한 그래프 이름의 JSON 파일을 읽어서 dict 로 반환합니다.
        파일이 없거나 파싱 오류 시 예외를 발생시킵니다.
        """
        path = Path(self.AGENT_GRAPH_CONFIG_DIR) / f"{graph_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Graph config not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    
    # Worker/Task Config
    WORKER_COUNT: int = Field(default_factory=lambda: max(os.cpu_count() or 1, 1))
    MAX_CONCURRENT_TASKS: int = 100
    TASK_STATUS_TTL: int = 86400 # Task 상태 유지 시간 (초)

    # General Service Config
    REQUEST_TIMEOUT: float = Field(60.0, alias='DEFAULT_REQUEST_TIMEOUT') # .env.example과 맞춤
    ENABLE_PERFORMANCE_TRACKING: bool = True

    # Redis Config (현재 REDIS_URL에서 파생될 수 있지만, 명시적으로 분리)
    REDIS_URL: str = 'redis://localhost:6379/0'
    REDIS_PASSWORD: Optional[str] = None
    REDIS_CONNECTION_POOL_SIZE: int = 20
    # --- 로드맵 .env.example 기준 추가/수정 ---
    # MEMORY_TYPE: str = 'redis' # 필요시 메모리 타입 선택용
    REDIS_HOST: str = 'localhost'
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    # --- ---

    # Memory/Cache TTL
    MEMORY_TTL: int = 86400
    CACHE_TTL: int = 3600
    MEMORY_MANAGER_CACHE_SIZE: int = 10000

    # LLM Config
    PRIMARY_LLM_PROVIDER: str = Field(..., description="기본 LLM 제공자 (openai, anthropic 등)")
    FALLBACK_LLM_PROVIDER: Optional[str] = Field(None, description="폴백 LLM 제공자 (선택 사항)")
    LLM_PROVIDERS: Dict[str, LLMProviderSettings] = Field(..., description="LLM 제공자별 설정")
    LLM_REQUEST_TIMEOUT: int = Field(60, description="LLM 요청 타임아웃(초)")
    LLM_MAX_RETRIES: int = Field(3, description="LLM 요청 최대 재시도 횟수")
    
    # Agent Names
    PLANNER_AGENT_NAME: str = 'default_planner'
    EXECUTOR_AGENT_NAME: str = 'default_executor'

    # API Config
    API_HOST: str = '0.0.0.0' # .env.example 과 맞춤
    API_PORT: int = 8000
    API_PREFIX: str = '/api/v1'
    CORS_ORIGINS: List[str] = ['*']

    WEBSOCKET_KEEP_ALIVE_INTERVAL: int = Field(60, description="WebSocket 연결 유지를 위한 서버 측 sleep 간격(초)")


    # Metrics Config
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090

    # Vector DB Config
    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_TYPE: Literal['chroma', 'qdrant', 'faiss', 'none'] = 'none'

    # Task Queue Config
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
        # Derive REDIS_HOST, REDIS_PORT, REDIS_DB from REDIS_URL if not set explicitly
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
        case_sensitive=False, # 환경 변수 이름 대소문자 구분 안 함
        env_nested_delimiter='__' # LLM_PROVIDERS_CONFIG__OPENAI__API_KEY 같은 형식 지원
    )
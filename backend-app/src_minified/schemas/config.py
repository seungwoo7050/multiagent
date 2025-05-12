L='utf-8'
K='config'
F='localhost'
E=True
C=False
B=None
import os
from typing import Dict,List,Literal,Optional
from pydantic import Field as A,ConfigDict as G,BaseModel as M
from pydantic_settings import BaseSettings as N
import json as H,logging as O
from pathlib import Path as I
D=O.getLogger(__name__)
J=I(__file__).resolve().parent.parent
class P(M):api_key=A(...,alias='API_KEY',description='LLM API 키');model_name=A(...,alias='MODEL_NAME',description='사용할 LLM 모델 이름');endpoint=A(B,alias='ENDPOINT');model_config=G(case_sensitive=C,populate_by_name=E)
class Q(N):
	APP_NAME='MultiAgentPlatform';APP_VERSION='0.1.0';ENVIRONMENT='development';DEBUG=C;LOG_LEVEL='INFO';LOG_FORMAT='json';LOG_TO_FILE=C;LOG_FILE_PATH=B;AGENT_GRAPH_CONFIG_DIR=A(default=str(J/K/'agent_graphs'),description='Directory for dynamic agent graph configurations (e.g., JSON files)');PROMPT_TEMPLATE_DIR=A(default=str(J/K/'prompts'),description='Directory for prompt template files')
	def load_graph_config(F,graph_name):
		C=graph_name
		if not C.endswith('.json'):B=f"{C}.json"
		else:B=C
		A=I(F.AGENT_GRAPH_CONFIG_DIR)/B;D.debug(f"Attempting to load graph config from: {A}")
		if not A.exists():D.error(f"Graph config file not found at path: {A}");raise FileNotFoundError(f"Graph config not found: {A}")
		try:G=H.loads(A.read_text(encoding=L));D.debug(f"Successfully loaded and parsed graph config: {B}");return G
		except H.JSONDecodeError as J:D.error(f"JSONDecodeError for graph config {B}: {J}",exc_info=E);raise
	WORKER_COUNT=A(default_factory=lambda:max(os.cpu_count()or 1,1));MAX_CONCURRENT_TASKS=100;TASK_STATUS_TTL=86400;REQUEST_TIMEOUT=A(6e1,alias='DEFAULT_REQUEST_TIMEOUT');ENABLE_PERFORMANCE_TRACKING=E;REDIS_URL='redis://localhost:6379/0';REDIS_PASSWORD=B;REDIS_CONNECTION_POOL_SIZE=20;REDIS_HOST=F;REDIS_PORT=6379;REDIS_DB=0;MEMORY_TTL=86400;CACHE_TTL=3600;MEMORY_MANAGER_CACHE_SIZE=10000;MEMORY_MANAGER_CHAT_HISTORY_PREFIX=A(default='chat_history',description='Prefix for chat history keys in memory manager');PRIMARY_LLM_PROVIDER=A(...,description='기본 LLM 제공자 (openai, anthropic 등)');FALLBACK_LLM_PROVIDER=A(B,description='폴백 LLM 제공자 (선택 사항)');LLM_PROVIDERS=A(...,description='LLM 제공자별 설정');LLM_REQUEST_TIMEOUT=A(60,description='LLM 요청 타임아웃(초)');LLM_MAX_RETRIES=A(3,description='LLM 요청 최대 재시도 횟수');PLANNER_AGENT_NAME='default_planner';EXECUTOR_AGENT_NAME='default_executor';API_HOST='0.0.0.0';API_PORT=8000;API_PREFIX='/api/v1';CORS_ORIGINS=['*'];WEBSOCKET_KEEP_ALIVE_INTERVAL=A(60,description='WebSocket 연결 유지를 위한 서버 측 sleep 간격(초)');METRICS_ENABLED=E;METRICS_PORT=9090;VECTOR_DB_URL=B;VECTOR_DB_TYPE='none';TASK_QUEUE_STREAM_NAME='multi_agent_tasks';TASK_QUEUE_GROUP_NAME='agent_workers';OTEL_EXPORTER_OTLP_ENDPOINT=A(B,description='OpenTelemetry OTLP Exporter Endpoint');LANGCHAIN_TRACING_V2=A(C,description='Enable LangSmith tracing V2');LANGCHAIN_ENDPOINT=A('https://api.smith.langchain.com',description='LangSmith API Endpoint');LANGCHAIN_API_KEY=A(B,description='LangSmith API Key');LANGCHAIN_PROJECT=A(B,description='LangSmith Project Name')
	def validate_settings(A):
		C=[]
		if A.PRIMARY_LLM_PROVIDER not in A.LLM_PROVIDERS:C.append(f"PRIMARY_LLM_PROVIDER '{A.PRIMARY_LLM_PROVIDER}' not in LLM_PROVIDERS.")
		if A.FALLBACK_LLM_PROVIDER and A.FALLBACK_LLM_PROVIDER not in A.LLM_PROVIDERS:C.append(f"FALLBACK_LLM_PROVIDER '{A.FALLBACK_LLM_PROVIDER}' not in LLM_PROVIDERS.")
		if A.REDIS_URL and(A.REDIS_HOST==F or A.REDIS_PORT==6379 or A.REDIS_DB==0):
			try:
				from urllib.parse import urlparse as D;B=D(A.REDIS_URL)
				if B.hostname and A.REDIS_HOST==F:A.REDIS_HOST=B.hostname
				if B.port and A.REDIS_PORT==6379:A.REDIS_PORT=B.port
				if B.path and len(B.path)>1 and B.path[1:].isdigit()and A.REDIS_DB==0:A.REDIS_DB=int(B.path[1:])
			except Exception as E:C.append(f"Could not parse REDIS_HOST/PORT/DB from REDIS_URL: {E}")
		return C
	model_config=G(env_file='.env',env_file_encoding=L,extra='ignore',case_sensitive=C,env_nested_delimiter='__')
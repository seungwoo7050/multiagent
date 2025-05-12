Q='detail'
P='url'
O=RuntimeError
I='development'
G=print
D=True
C=Exception
import contextlib as R,os,sys as E,traceback as J
from typing import AsyncGenerator
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor as S
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor as T
from opentelemetry.instrumentation.redis import RedisInstrumentor as U
try:
	K=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
	if K not in E.path:E.path.insert(0,K)
except C as V:G(f"Error setting up project root path: {V}",file=E.stderr);E.exit(1)
try:from src.config.settings import get_settings as W;from src.utils.logger import get_logger as L,setup_logging as X;from src.utils.telemetry import setup_telemetry as Y;B=W();A=L(__name__)
except C as Z:G(f"FATAL: Could not initialize basic settings or logging: {Z}\n{J.format_exc()}",file=E.stderr);E.exit(1)
import uvicorn as a
from fastapi import FastAPI as b,Request,status as M
from fastapi.exceptions import RequestValidationError as c
from fastapi.middleware.cors import CORSMiddleware as d
from fastapi.responses import JSONResponse as H
try:from src.config.errors import ERROR_TO_HTTP_STATUS as e,BaseError as f,ErrorCode as g;from src.config.connections import setup_connection_pools as h,cleanup_connection_pools as i;from src.memory.memory_manager import get_memory_manager as j;from src.services.tool_manager import get_tool_manager as k,ToolManager;from src.schemas.response_models import HealthCheckResponse as N;from src.api.routers import router as l
except ImportError as m:A.critical(f"Failed to import core application components: {m}",exc_info=D);E.exit(1)
@R.asynccontextmanager
async def n(app):
	global A
	try:G('Attempting to setup Telemetry...');Y();G('Telemetry setup completed (or skipped if already done).')
	except C as P:G(f"FATAL: Failed to setup OpenTelemetry: {P}",file=E.stderr)
	try:G('Attempting to setup Logging utility...');X(B);A=L(__name__);A.info('Logging utility setup successfully.')
	except C as Q:G(f"FATAL: Failed to setup logging utility: {Q}\n{J.format_exc()}",file=E.stderr);E.exit(1)
	A.info('Application startup sequence initiated via lifespan...');F=0
	try:
		try:await h();A.info('Connection pools setup successfully.')
		except C as R:A.error(f"Failed to setup connection pools: {R}",exc_info=D);F+=1
		try:j();A.info('Memory Manager initialized or confirmed.')
		except C as S:A.error(f"Failed to initialize Memory Manager: {S}",exc_info=D);F+=1
		try:
			A.info('Initializing Tool Manager and loading tools...');H=k('global_tools');T='src/tools';U=H.load_tools_from_directory(T,auto_register=D);K=list(H.get_names());A.info(f"Tool loading complete. Imported {U} modules. Registered tools: {K}")
			if not K:A.warning('No tools were registered. Check the tool directory and implementations.')
		except C as V:A.error(f"Error during Tool Manager initialization or tool loading: {V}",exc_info=D);F+=1
		if F>0:
			M=f"{F} critical component(s) failed to initialize during startup. Check logs for details.";A.critical(M)
			if B.ENVIRONMENT==I:raise O(M)
		else:A.info('Core components initialized successfully.')
		A.info(f"Application '{B.APP_NAME}' v{B.APP_VERSION} startup sequence finished successfully.");yield;A.info('Application shutdown sequence initiated...')
		try:await i();A.info('Connection pools cleaned up successfully.')
		except C as W:A.error(f"Error cleaning up connection pools during shutdown: {W}",exc_info=D)
		A.info('Application shutdown sequence finished.')
	except C as N:A.critical(f"Fatal error during application lifespan management: {N}",exc_info=D);raise O('Application failed during startup or shutdown')from N
F=b(title=B.APP_NAME+' (Framework-Centric)',version=B.APP_VERSION,description='Framework-Centric Multi-Agent System API using LangGraph, FastAPI, and best practices.',debug=B.DEBUG,lifespan=n)
try:
	if B.OTEL_EXPORTER_OTLP_ENDPOINT:S.instrument_app(F);T().instrument();U().instrument();A.info('FastAPIInstrumentor applied to the application for OpenTelemetry tracing.')
	else:A.info('FastAPIInstrumentor skipped as OTEL_EXPORTER_OTLP_ENDPOINT is not set.')
except C as o:A.error(f"Failed to apply FastAPIInstrumentor: {o}",exc_info=D)
if B.CORS_ORIGINS:F.add_middleware(d,allow_origins=B.CORS_ORIGINS,allow_credentials=D,allow_methods=['*'],allow_headers=['*']);A.info(f"CORS middleware added. Allowed origins: {B.CORS_ORIGINS}")
else:A.warning('CORS_ORIGINS not set in settings. CORS middleware not added.')
@F.exception_handler(f)
async def r(request,exc):B=exc;D=e.get(B.code,500);C=B.to_dict();A.error(f"API Error Handled ({B.code}): {B.message}",extra={'error_details':C,P:str(request.url)},exc_info=B.original_error);return H(status_code=D,content=C)
@F.exception_handler(c)
async def s(request,exc):D='errors';C=request;B=exc;A.warning(f"Request validation failed: {B.errors()}",extra={D:B.errors(),P:str(C.url),'body':await C.body()});return H(status_code=M.HTTP_422_UNPROCESSABLE_ENTITY,content={Q:'Validation Error',D:B.errors()})
@F.exception_handler(C)
async def t(request,exc):A.exception(f"Unhandled exception during request to {request.url.path}: {exc}");return H(status_code=M.HTTP_500_INTERNAL_SERVER_ERROR,content={Q:'An internal server error occurred.','code':g.SYSTEM_ERROR.value})
@F.get('/health',tags=['System'],summary='Health Check',description='Performs a basic health check of the API server.',response_model=N)
async def u():A.debug('Health check endpoint called');return N(status='ok')
try:F.include_router(l,prefix=B.API_PREFIX);A.info(f"Included API routes from src.api.routers under prefix: {B.API_PREFIX}")
except NameError as p:A.error(f"Failed to include routers: '{p}' likely not defined or imported correctly.",exc_info=D)
except C as q:A.error(f"Error including API routers: {q}",exc_info=D)
if __name__=='__main__':A.info(f"Starting API server directly using Uvicorn (Host: {B.API_HOST}, Port: {B.API_PORT})...");a.run(app='src.api.app:app'if B.ENVIRONMENT==I else F,host=B.API_HOST,port=B.API_PORT,reload=B.ENVIRONMENT==I,log_level=B.LOG_LEVEL.lower())
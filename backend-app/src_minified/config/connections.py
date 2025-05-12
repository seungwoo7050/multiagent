J='redis'
G=True
F=Exception
D=None
import redis.asyncio as H
from redis.asyncio.connection import ConnectionPool as K
from typing import Optional
from src.config.settings import get_settings as E
from src.utils.logger import get_logger as L
from src.config.errors import ConnectionError,ErrorCode as I
B=L(__name__)
C=E()
A=D
async def M():
	global A
	if A is not D:B.warning('Redis connection pool already initialized.');return
	try:B.info(f"Setting up Redis connection pool for URL: {C.REDIS_URL} (DB: {C.REDIS_DB})");A=K.from_url(C.REDIS_URL,max_connections=C.REDIS_CONNECTION_POOL_SIZE,decode_responses=False,health_check_interval=30);L=H.Redis(connection_pool=A);await L.ping();B.info('Redis connection pool initialized and ping successful.')
	except F as E:B.error(f"Failed to initialize Redis connection pool: {E}",exc_info=G);A=D;raise ConnectionError(code=I.REDIS_CONNECTION_ERROR,message=f"Failed to connect to Redis at {C.REDIS_URL}: {E}",original_error=E,service=J)from E
async def N():
	global A
	if A:
		B.info('Closing Redis connection pool...')
		try:await A.disconnect();A=D;B.info('Redis connection pool closed.')
		except F as C:B.error(f"Error closing Redis connection pool: {C}",exc_info=G)
	else:B.info('Redis connection pool was not initialized or already closed.')
async def O():
	global A
	if A is D:B.critical('Attempted to get Redis connection before the pool was initialized.');raise RuntimeError('Redis connection pool is not available. Ensure setup_connection_pools() is called during application startup.')
	try:return H.Redis(connection_pool=A)
	except F as C:B.error(f"Failed to create Redis client instance from pool: {C}",exc_info=G);raise ConnectionError(code=I.REDIS_CONNECTION_ERROR,message=f"Failed to get Redis connection from pool: {C}",original_error=C,service=J)from C
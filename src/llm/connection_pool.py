import asyncio
import time
from typing import Dict, Optional, Any, Mapping, Set
import aiohttp
from aiohttp.client import ClientTimeout
from src.config.settings import get_settings
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager, MEMORY_METRICS
from src.config.errors import ConnectionError, ErrorCode, BaseError, convert_exception

settings = get_settings()
logger = get_logger(__name__)
metrics = get_metrics_manager()

_CONNECTION_POOLS: Dict[str, aiohttp.ClientSession] = {}
_POOL_CREATION_LOCKS: Dict[str, asyncio.Lock] = {}

@metrics.timed_metric(MEMORY_METRICS['duration'], {'operation_type': 'get_llm_connection_pool'})
async def get_connection_pool(provider: str) -> aiohttp.ClientSession:
    provider = provider.lower()
    if provider in _CONNECTION_POOLS:
        session = _CONNECTION_POOLS[provider]
        if not session.closed:
            logger.debug(f'Reusing existing connection pool for LLM provider: {provider}')
            return session
        else:
            logger.warning(f"Connection pool for LLM provider '{provider}' was closed. Creating a new one.")
            _CONNECTION_POOLS.pop(provider, None)
            _POOL_CREATION_LOCKS.pop(provider, None)
    if provider not in _POOL_CREATION_LOCKS:
        _POOL_CREATION_LOCKS[provider] = asyncio.Lock()
    provider_lock = _POOL_CREATION_LOCKS[provider]
    async with provider_lock:
        if provider in _CONNECTION_POOLS and (not _CONNECTION_POOLS[provider].closed):
            logger.debug(f'Connection pool for {provider} created by another coroutine while waiting for lock.')
            return _CONNECTION_POOLS[provider]
        logger.info(f'Creating new connection pool (aiohttp.ClientSession) for LLM provider: {provider}')
        try:
            start_time = time.monotonic()
            provider_config = settings.LLM_PROVIDERS_CONFIG.get(provider, {})
            pool_size = provider_config.get('connection_pool_size', 10)
            timeout_seconds = provider_config.get('timeout', settings.REQUEST_TIMEOUT)
            connector = aiohttp.TCPConnector(limit=pool_size, limit_per_host=pool_size, ttl_dns_cache=300, enable_cleanup_closed=True, force_close=False)
            timeout = aiohttp.ClientTimeout(total=timeout_seconds, connect=10.0, sock_connect=10.0, sock_read=timeout_seconds)
            session = aiohttp.ClientSession(connector=connector, timeout=timeout, raise_for_status=False, trust_env=True)
            _CONNECTION_POOLS[provider] = session
            duration = time.monotonic() - start_time
            logger.info(f'Successfully created connection pool for {provider} (Size: {pool_size}, Timeout: {timeout_seconds}s)')
            return session
        except Exception as e:
            error = ConnectionError(code=ErrorCode.CONNECTION_ERROR, message=f"Failed to create connection pool (aiohttp.ClientSession) for LLM provider '{provider}': {str(e)}", service=provider, original_error=e)
            error.log_error(logger)
            raise error

async def close_connection_pool(provider: str) -> bool:
    provider = provider.lower()
    logger.debug(f'Attempting to close connection pool for LLM provider: {provider}')
    session = _CONNECTION_POOLS.pop(provider, None)
    _POOL_CREATION_LOCKS.pop(provider, None)
    if session:
        if not session.closed:
            try:
                await session.close()
                logger.info(f'Closed connection pool for LLM provider: {provider}')
                return True
            except Exception as e:
                logger.error(f'Error closing connection pool for {provider}: {e}', exc_info=True)
                return False
        else:
            logger.debug(f'Connection pool for {provider} was already closed.')
            return True
    else:
        logger.debug(f'No active connection pool found for LLM provider: {provider}. Nothing to close.')
        return False

async def cleanup_connection_pools() -> None:
    logger.info('Cleaning up all LLM provider connection pools...')
    providers = list(_CONNECTION_POOLS.keys())
    closed_count = 0
    error_count = 0
    for provider in providers:
        try:
            success = await close_connection_pool(provider)
            if success:
                closed_count += 1
        except Exception as e:
            error_count += 1
            logger.error(f'Unexpected error during cleanup for provider {provider}: {e}', exc_info=True)
    logger.info(f'LLM connection pool cleanup finished. Closed: {closed_count}, Errors: {error_count}, Remaining in registry: {len(_CONNECTION_POOLS)}')

def get_active_providers() -> Set[str]:
    active_providers = set()
    for provider, session in _CONNECTION_POOLS.items():
        if not session.closed:
            active_providers.add(provider)
    return active_providers

def get_pool_metrics() -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    for provider, session in _CONNECTION_POOLS.items():
        if session.closed:
            continue
        connector = session.connector
        if connector and isinstance(connector, aiohttp.TCPConnector):
            try:
                metrics[provider] = {'provider': provider, 'connector_limit': connector.limit, 'acquired_connections': len(getattr(connector, '_acquired', [])), 'limit_per_host': connector.limit_per_host, 'connector_closed': connector.closed}
            except Exception as e:
                logger.warning(f'Could not retrieve full connector metrics for {provider}: {e}')
                metrics[provider] = {'provider': provider, 'status': 'error retrieving metrics'}
        else:
            metrics[provider] = {'provider': provider, 'status': 'connector info unavailable'}
    return metrics

async def health_check() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for provider, session in _CONNECTION_POOLS.items():
        provider_status: Dict[str, Any] = {'provider': provider, 'status': 'unknown', 'healthy': False, 'message': ''}
        if session.closed:
            provider_status.update({'status': 'closed', 'message': 'Connection pool (ClientSession) is closed.'})
            results[provider] = provider_status
            continue
        try:
            connector = session.connector
            if connector and (not connector.closed):
                provider_status.update({'status': 'ok', 'healthy': True, 'message': 'Connection pool (ClientSession & Connector) is healthy.', 'active_connections': len(getattr(connector, '_acquired', []))})
            elif connector and connector.closed:
                provider_status.update({'status': 'error', 'message': 'Connector associated with the session is closed.'})
            else:
                provider_status.update({'status': 'error', 'message': 'Session connector is unavailable or invalid.'})
        except Exception as e:
            provider_status.update({'status': 'error', 'message': f'Health check failed with error: {str(e)}'})
        results[provider] = provider_status
    return results
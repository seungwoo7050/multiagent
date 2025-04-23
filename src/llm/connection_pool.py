"""
Connection pooling for LLM API requests.

This module manages connection pools for different LLM providers to 
reduce connection overhead and improve performance.
"""

import asyncio
import time
from typing import Dict, Optional, Any, Mapping, Set
import aiohttp
from aiohttp.client import ClientTimeout

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.config.metrics import (
    MEMORY_OPERATION_DURATION,
    track_memory_operation,
    track_memory_operation_completed,
    timed_metric
)
from src.config.errors import ConnectionError, ErrorCode

settings = get_settings()
logger = get_logger(__name__)

# Global registry of connection pools
_CONNECTION_POOLS: Dict[str, aiohttp.ClientSession] = {}
_POOL_CREATION_LOCKS: Dict[str, asyncio.Lock] = {}


@timed_metric(MEMORY_OPERATION_DURATION, {"operation_type": "get_connection_pool"})
async def get_connection_pool(provider: str) -> aiohttp.ClientSession:
    """
    Get or create a connection pool for the specified provider.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
    
    Returns:
        aiohttp.ClientSession: A client session from the pool
        
    Raises:
        ConnectionError: If connection pool creation fails
    """
    provider = provider.lower()
    
    # Check if we already have a pool
    if provider in _CONNECTION_POOLS:
        session = _CONNECTION_POOLS[provider]
        # Check if the session is still active
        if not session.closed:
            return session
        # If closed, we'll create a new one
        logger.warning(f"Connection pool for {provider} was closed, creating new one")
        _CONNECTION_POOLS.pop(provider, None)
    
    # Get or create the lock for this provider
    if provider not in _POOL_CREATION_LOCKS:
        _POOL_CREATION_LOCKS[provider] = asyncio.Lock()
    
    # Use the lock to prevent multiple simultaneous pool creations
    async with _POOL_CREATION_LOCKS[provider]:
        # Double-check in case another task created the pool while we were waiting
        if provider in _CONNECTION_POOLS and not _CONNECTION_POOLS[provider].closed:
            return _CONNECTION_POOLS[provider]
        
        # Create a new connection pool
        try:
            # Track operation
            track_memory_operation(f"create_connection_pool_{provider}")
            start_time = time.time()
            
            # Get provider-specific settings
            provider_config = settings.LLM_PROVIDERS_CONFIG.get(provider, {})
            pool_size = provider_config.get("connection_pool_size", 10)
            timeout_seconds = provider_config.get("timeout", settings.REQUEST_TIMEOUT)
            
            # Create connector
            connector = aiohttp.TCPConnector(
                limit=pool_size,  # Max number of connections
                ttl_dns_cache=300,  # DNS cache TTL in seconds
                enable_cleanup_closed=True,
                force_close=False,
                limit_per_host=pool_size,  # Max connections per host
            )
            
            # Create timeout
            timeout = aiohttp.ClientTimeout(
                total=timeout_seconds,
                connect=10.0,
                sock_connect=10.0,
                sock_read=timeout_seconds,
            )  # Aligned with global config settings
            
            # Create session
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=False,
                trust_env=True,
            )
            
            # Store in registry
            _CONNECTION_POOLS[provider] = session
            
            # Track completion
            duration = time.time() - start_time
            track_memory_operation_completed(f"create_connection_pool_{provider}", duration)
            
            logger.info(f"Created connection pool for {provider} with size {pool_size}")
            return session
        
        except Exception as e:
            error = ConnectionError(
                code=ErrorCode.CONNECTION_ERROR,
                message=f"Failed to create connection pool for {provider}: {str(e)}",
                service=provider,
                original_error=e
            )
            error.log_error(logger)
            raise error


async def close_connection_pool(provider: str) -> bool:
    """
    Close the connection pool for the specified provider.
    
    Args:
        provider: The provider name (e.g., "openai", "anthropic")
    
    Returns:
        bool: True if the pool was closed, False if it didn't exist
    """
    provider = provider.lower()
    
    if provider in _CONNECTION_POOLS:
        session = _CONNECTION_POOLS.pop(provider)
        if not session.closed:
            await session.close()
            logger.info(f"Closed connection pool for {provider}")
        return True
    
    return False


async def cleanup_connection_pools() -> None:
    """Close all connection pools."""
    logger.info("Cleaning up all connection pools")
    
    # Get a copy of the keys to avoid modification during iteration
    providers = list(_CONNECTION_POOLS.keys())
    
    for provider in providers:
        await close_connection_pool(provider)
    
    logger.info("All connection pools closed")


def get_active_providers() -> Set[str]:
    """
    Get the set of providers with active connection pools.
    
    Returns:
        Set[str]: Set of provider names
    """
    return {
        provider for provider, session in _CONNECTION_POOLS.items()
        if not session.closed
    }


def get_pool_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Get metrics for all connection pools.
    
    Returns:
        Dict[str, Dict[str, Any]]: Metrics for each provider
    """
    metrics = {}
    
    for provider, session in _CONNECTION_POOLS.items():
        if session.closed:
            continue
        
        connector = session.connector
        if connector:
            metrics[provider] = {
                "limit": connector.limit,
                "acquired_connections": len(connector._acquired),
                "acquired_per_host": {
                    str(key): len(value) for key, value in connector._acquired_per_host.items()
                },
                "limit_per_host": connector.limit_per_host,
                "is_closed": connector.closed,
            }
    
    return metrics


async def health_check() -> Dict[str, Dict[str, Any]]:
    """
    Perform a health check on all connection pools.
    
    Returns:
        Dict[str, Dict[str, Any]]: Health status for each provider
    """
    results = {}
    
    for provider, session in _CONNECTION_POOLS.items():
        if session.closed:
            results[provider] = {
                "status": "closed",
                "healthy": False,
                "message": "Connection pool is closed",
            }
            continue
        
        try:
            # Simple health check - just check if the connector is healthy
            connector = session.connector
            if connector and not connector.closed:
                results[provider] = {
                    "status": "ok",
                    "healthy": True,
                    "message": "Connection pool is healthy",
                    "active_connections": len(connector._acquired),
                }
            else:
                results[provider] = {
                    "status": "error",
                    "healthy": False,
                    "message": "Connector is closed or not available",
                }
        except Exception as e:
            results[provider] = {
                "status": "error",
                "healthy": False,
                "message": f"Health check failed: {str(e)}",
            }
    
    return results
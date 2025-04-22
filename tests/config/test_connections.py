import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, ANY

import pytest
import redis
import redis.asyncio as aioredis
import aiohttp

from src.config.connections import (
    get_redis_connection,
    get_redis_async_connection,
    get_http_session,
    ConnectionManager,
    setup_connection_pools,
    cleanup_connection_pools,
)

def test_get_redis_connection_creates_pool_once():
    with patch('redis.ConnectionPool') as mock_pool, \
         patch('redis.Redis') as mock_redis, \
         patch('src.config.connections.settings') as mock_settings:
        
        mock_settings.REDIS_URL = "redis://localhost:6379/0"
        mock_settings.REDIS_PASSWORD = None
        mock_settings.REDIS_CONNECTION_POOL_SIZE = 10
        
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        
        result1 = get_redis_connection()
        
        assert mock_pool.from_url.called
        assert mock_redis.called
        assert result1 is mock_redis_instance
        
        mock_pool.reset_mock()
        mock_redis.reset_mock()
        
        result2 = get_redis_connection()
        
        assert not mock_pool.from_url.called
        assert mock_redis.called
        assert result2 is mock_redis_instance


@pytest.mark.asyncio
async def test_get_redis_async_connection_creates_pool_once():
    with patch('redis.asyncio.ConnectionPool') as mock_pool, \
         patch('redis.asyncio.Redis') as mock_redis, \
         patch('src.config.connections.settings') as mock_settings:
        
        mock_settings.REDIS_URL = "redis://localhost:6379/0"
        mock_settings.REDIS_PASSWORD = None
        mock_settings.REDIS_CONNECTION_POOL_SIZE = 10
        
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance
        
        result1 = await get_redis_async_connection()
        
        assert mock_pool.from_url.called
        assert mock_redis.called
        assert result1 is mock_redis_instance
        
        mock_pool.reset_mock()
        mock_redis.reset_mock()
        
        result2 = await get_redis_async_connection()
        
        assert not mock_pool.from_url.called
        assert mock_redis.called
        assert result2 is mock_redis_instance


@pytest.mark.asyncio
async def test_get_http_session_creates_session_once():
    with patch('aiohttp.TCPConnector') as mock_connector, \
         patch('aiohttp.ClientSession') as mock_session, \
         patch('src.config.connections.settings') as mock_settings:
        
        mock_settings.REQUEST_TIMEOUT = 30.0
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        result1 = await get_http_session()
        
        assert mock_connector.called
        assert mock_session.called
        assert result1 is mock_session_instance
        
        mock_connector.reset_mock()
        mock_session.reset_mock()
        
        result2 = await get_http_session()
        
        assert not mock_connector.called
        assert not mock_session.called
        assert result2 is mock_session_instance


def test_connection_manager_close_all_connections():
    with patch('src.config.connections._redis_sync_pool') as mock_sync_pool, \
         patch('src.config.connections._redis_async_pool') as mock_async_pool, \
         patch('src.config.connections._http_session_pool') as mock_session, \
         patch('asyncio.create_task') as mock_create_task, \
         patch('src.config.connections.logger') as mock_logger:
        
        ConnectionManager.close_all_connections()
        assert mock_sync_pool.disconnect.called
        assert mock_create_task.call_count >= 1
        assert mock_logger.info.call_count >= 1

def test_setup_connection_pools():
    with patch('src.config.connections.get_redis_connection') as mock_get_redis, \
         patch('src.config.connections.logger') as mock_logger:
        setup_connection_pools()
        assert mock_get_redis.called
        assert mock_logger.info.call_count >= 1

def test_cleanup_connection_pools():
    with patch('src.config.connections.ConnectionManager') as mock_manager, \
         patch('src.config.connections.logger') as mock_logger:
        
        cleanup_connection_pools()
        assert mock_manager.close_all_connections.called
        assert mock_logger.info.called        
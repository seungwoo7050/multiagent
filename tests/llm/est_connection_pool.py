import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import aiohttp
from src.llm.connection_pool import (
    get_connection_pool,
    close_connection_pool,
    cleanup_connection_pools,
    get_active_providers,
    get_pool_metrics,
    health_check,
    _CONNECTION_POOLS,
    _POOL_CREATION_LOCKS # Lock 관리도 초기화 필요
)
from src.config.errors import ConnectionError, ErrorCode
# settings 모킹을 위해 Settings 클래스 임포트 (타입 힌팅 등에 사용 가능)
# 실제 Settings 클래스 대신 MagicMock을 사용할 것이므로 필수 아님
# from src.config.settings import Settings 


# --- Fixtures ---

# @pytest.fixture -> @pytest.fixture(autouse=True) 또는 각 테스트에 명시적 사용
# @pytest.mark.asyncio 를 fixture에 직접 사용할 수는 없으므로,
# async def 로 변경하고, 이를 사용하는 테스트는 @pytest.mark.asyncio 여야 함.
@pytest.fixture
async def clear_pools(): # async def 로 변경
    """Clear connection pools and locks before and after tests."""
    # Store original state
    original_pools = _CONNECTION_POOLS.copy()
    original_locks = _POOL_CREATION_LOCKS.copy()
    
    # Clear for test
    _CONNECTION_POOLS.clear()
    _POOL_CREATION_LOCKS.clear()
    
    yield # 테스트 실행
    
    # Clean up after test using await
    tasks = []
    for provider, session in list(_CONNECTION_POOLS.items()):
        if session and not session.closed:
            # session.close()가 비동기 함수이므로 await 사용
            # create_task 대신 직접 await하거나, asyncio.gather 사용
            tasks.append(session.close()) 
            
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True) # 여러 세션 동시 종료 시도

    # Clear again after closing
    _CONNECTION_POOLS.clear()
    _POOL_CREATION_LOCKS.clear()
    
    # Restore original state (주의: 세션 객체는 닫혔을 수 있음)
    # 일반적으로 테스트 후 복원보다는 완전 초기화가 더 안전함
    # _CONNECTION_POOLS.update(original_pools)
    # _POOL_CREATION_LOCKS.update(original_locks)


@pytest.fixture
def mock_connector():
    """Create a mock aiohttp connector."""
    connector = MagicMock(spec=aiohttp.TCPConnector) # spec을 사용하여 더 정확한 모킹
    connector.closed = False
    connector.limit = 10
    connector.limit_per_host = 10
    # Mock internal tracking attributes (실제 속성 이름 확인 필요)
    # aiohttp 버전에 따라 내부 속성 이름이 다를 수 있음
    # 여기서는 예시로 설정, 실제 테스트 시 필요에 따라 조정
    connector._acquired = set() # 실제론 ConnectionKey 등을 포함할 수 있음
    connector._acquired_per_host = {} 
    # connector.is_acquired(key) 같은 메소드를 모킹해야 할 수도 있음
    return connector


@pytest.fixture
def mock_session(mock_connector):
    """Create a mock aiohttp ClientSession."""
    # AsyncMock으로 비동기 메소드 모킹
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.connector = mock_connector
    session.close = AsyncMock() # close는 비동기 함수이므로 AsyncMock 사용
    return session


@pytest.fixture(scope="module") # settings는 모듈 범위에서 동일하게 사용 가능
def mock_settings():
    """Create a mock Settings object for connection pool tests."""
    settings = MagicMock()
    # connection_pool.py에서 사용하는 속성 설정
    settings.LLM_PROVIDERS_CONFIG = {
        "test_provider": {
            "connection_pool_size": 5,
            "timeout": 15.0
        },
        "error_provider": {
             "connection_pool_size": 1,
             "timeout": 5.0
        },
        "openai": { # 실제 provider 이름 예시
            "connection_pool_size": 20,
            "timeout": 60.0
        }
        # 필요에 따라 다른 provider 설정 추가
    }
    settings.REQUEST_TIMEOUT = 30.0 # 전역 타임아웃 설정
    # 다른 필요한 설정값들 추가 가능
    return settings


# --- Test Functions ---

# 모든 비동기 테스트에 @pytest.mark.asyncio 적용
# patch 데코레이터 추가 및 mock_settings fixture 사용
@pytest.mark.asyncio
# patch 데코레이터 순서 중요: 함수에 가까울수록 먼저 적용됨 (인자 순서 반대)
@patch('src.llm.connection_pool.aiohttp.ClientSession') 
@patch('src.llm.connection_pool.settings')
async def test_get_connection_pool(mock_settings_obj, mock_client_session_cls, mock_settings, clear_pools, mock_session):
    """Test getting a connection pool with mocked settings."""
    # Mock 설정 적용
    mock_settings_obj.return_value = mock_settings
    mock_client_session_cls.return_value = mock_session # ClientSession 클래스 자체를 모킹

    provider = "test_provider"
    pool = await get_connection_pool(provider)
    
    assert pool is mock_session
    assert provider in _CONNECTION_POOLS
    assert _CONNECTION_POOLS[provider] is mock_session
    # ClientSession이 호출되었는지 확인 (풀 생성 시)
    mock_client_session_cls.assert_called_once() 


@pytest.mark.asyncio
@patch('src.llm.connection_pool.settings') # settings만 모킹해도 되는 경우
async def test_get_connection_pool_reuse(mock_settings_obj, mock_settings, clear_pools, mock_session):
    """Test that connection pools are reused."""
    mock_settings_obj.return_value = mock_settings
    provider = "test_provider"
    
    _CONNECTION_POOLS[provider] = mock_session
    
    # ClientSession이 호출되지 않음을 확인
    with patch('src.llm.connection_pool.aiohttp.ClientSession') as mock_client_session_cls:
        pool = await get_connection_pool(provider)
        assert pool is mock_session
        mock_client_session_cls.assert_not_called()


@pytest.mark.asyncio
@patch('src.llm.connection_pool.aiohttp.ClientSession') 
@patch('src.llm.connection_pool.settings')
async def test_get_connection_pool_recreate(mock_settings_obj, mock_client_session_cls, mock_settings, clear_pools, mock_session):
    """Test that closed pools are recreated."""
    mock_settings_obj.return_value = mock_settings
    mock_client_session_cls.return_value = mock_session # 새로 생성될 세션의 Mock
    provider = "test_provider"

    # spec 인자를 제거하고 MagicMock 객체를 생성합니다.
    # 이 테스트에서는 closed 속성만 True이면 충분합니다.
    closed_session = MagicMock()  # <- spec 인자 제거!
    closed_session.closed = True
    
    _CONNECTION_POOLS[provider] = closed_session # 기존에 닫힌 세션 추가

    pool = await get_connection_pool(provider)

    assert pool is mock_session # 새로 생성된 세션이 mock_session인지 확인
    assert _CONNECTION_POOLS[provider] is mock_session
    mock_client_session_cls.assert_called_once() # 새 세션이 한 번 생성되었는지 확인


@pytest.mark.asyncio
@patch('src.llm.connection_pool.settings')
async def test_close_connection_pool(mock_settings_obj, mock_settings, clear_pools, mock_session):
    """Test closing a specific connection pool."""
    mock_settings_obj.return_value = mock_settings
    provider = "test_provider"
    _CONNECTION_POOLS[provider] = mock_session
    
    result = await close_connection_pool(provider)
    
    assert result is True
    mock_session.close.assert_called_once()
    assert provider not in _CONNECTION_POOLS


@pytest.mark.asyncio
@patch('src.llm.connection_pool.settings')
async def test_close_nonexistent_pool(mock_settings_obj, mock_settings, clear_pools):
    """Test closing a non-existent pool."""
    mock_settings_obj.return_value = mock_settings
    result = await close_connection_pool("nonexistent")
    assert result is False


@pytest.mark.asyncio
@patch('src.llm.connection_pool.settings')
async def test_cleanup_connection_pools(mock_settings_obj, mock_settings, clear_pools):
    """Test cleaning up all connection pools."""
    mock_settings_obj.return_value = mock_settings
    
    # 각기 다른 Mock 세션 사용 권장 (호출 횟수 등을 정확히 추적하기 위해)
    session1 = MagicMock(spec=aiohttp.ClientSession); session1.closed=False; session1.close=AsyncMock()
    session2 = MagicMock(spec=aiohttp.ClientSession); session2.closed=False; session2.close=AsyncMock()
    
    _CONNECTION_POOLS["provider1"] = session1
    _CONNECTION_POOLS["provider2"] = session2
    
    await cleanup_connection_pools()
    
    session1.close.assert_called_once()
    session2.close.assert_called_once()
    assert len(_CONNECTION_POOLS) == 0


# 동기 함수 테스트는 @pytest.mark.asyncio 불필요
# settings 모킹이 필요 없을 수도 있지만, 일관성을 위해 추가 가능
@patch('src.llm.connection_pool.settings') 
def test_get_active_providers(mock_settings_obj, mock_settings, clear_pools, mock_session):
    """Test getting active providers."""
    mock_settings_obj.return_value = mock_settings
    
    _CONNECTION_POOLS["active1"] = mock_session
    _CONNECTION_POOLS["active2"] = mock_session
    
    closed_session = MagicMock(spec=aiohttp.ClientSession)
    closed_session.closed = True
    _CONNECTION_POOLS["inactive"] = closed_session
    
    active = get_active_providers()
    
    assert active == {"active1", "active2"} # Set 비교


@patch('src.llm.connection_pool.settings')
def test_get_pool_metrics(mock_settings_obj, mock_settings, clear_pools, mock_session): # mock_connector 불필요
    """Test getting metrics for connection pools."""
    mock_settings_obj.return_value = mock_settings
    
    # mock_session fixture가 mock_connector를 사용하도록 설정됨
    _CONNECTION_POOLS["test_provider"] = mock_session 
    
    metrics = get_pool_metrics()
    
    assert "test_provider" in metrics
    provider_metrics = metrics["test_provider"]
    # mock_connector fixture에서 설정한 값과 일치하는지 확인
    assert provider_metrics["limit"] == 10 
    # 내부 속성 접근 대신 connector 상태 확인 로직이 있다면 그것을 테스트하는 것이 더 안정적
    # assert provider_metrics["acquired_connections"] == 1 # 내부 구현에 따라 달라질 수 있음
    # assert len(provider_metrics["acquired_per_host"]) == 1 # 내부 구현에 따라 달라질 수 있음
    assert provider_metrics["limit_per_host"] == 10
    assert provider_metrics["is_closed"] is False


@pytest.mark.asyncio
@patch('src.llm.connection_pool.settings')
async def test_health_check(mock_settings_obj, mock_settings, clear_pools, mock_session): # mock_connector 불필요
    """Test connection pool health check."""
    mock_settings_obj.return_value = mock_settings

    _CONNECTION_POOLS["healthy_provider"] = mock_session
    
    closed_session = MagicMock(spec=aiohttp.ClientSession)
    closed_session.closed = True
    closed_session.connector = None # 커넥터가 없는 경우도 고려
    _CONNECTION_POOLS["unhealthy_provider"] = closed_session
    
    health_results = await health_check()
    
    assert "healthy_provider" in health_results
    assert health_results["healthy_provider"]["status"] == "ok"
    assert health_results["healthy_provider"]["healthy"] is True
    # assert health_results["healthy_provider"]["active_connections"] == 1 # 내부 구현 의존적

    assert "unhealthy_provider" in health_results
    assert health_results["unhealthy_provider"]["status"] == "closed"
    assert health_results["unhealthy_provider"]["healthy"] is False


@pytest.mark.asyncio
@patch('src.llm.connection_pool.aiohttp.TCPConnector') 
@patch('src.llm.connection_pool.settings')
async def test_connection_error_handling(mock_settings_obj, mock_tcp_connector_cls, mock_settings, clear_pools):
    """Test error handling when TCPConnector creation fails."""
    mock_settings_obj.return_value = mock_settings
    mock_tcp_connector_cls.side_effect = Exception("TCP Connection error") 

    with pytest.raises(ConnectionError) as excinfo:
        await get_connection_pool("error_provider")
        
    # 에러 메시지 문자열 포함 여부 확인
    assert "TCP Connection error" in str(excinfo.value) 
    # 에러 코드를 ErrorCode Enum 멤버와 비교하도록 수정
    assert excinfo.value.code == ErrorCode.CONNECTION_ERROR # <- 수정됨
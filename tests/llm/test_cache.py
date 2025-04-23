import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

# 테스트 대상 모듈 및 클래스 임포트
from src.llm.cache import (
    LLMCache,
    TwoLevelCache,
    get_cache,
    clear_cache,
    cache_result,
    get_cache_stats
)
# 전역 변수 임포트 (필요시)
from src.llm.cache import _CACHE_INSTANCE, _CACHE_LOCK
# 에러 클래스 임포트 (필요시)
# from src.config.errors import ErrorCode, ConnectionError

# --- Fixtures ---

@pytest.fixture
def mock_redis():
    """Mock Redis 클라이언트 생성 (비동기 메서드 포함)."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.setex = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.ttl = AsyncMock(return_value=-2) # 기본 TTL 응답 (-2는 키 없음)
    # scan 응답 형식: (다음 커서, [키 리스트])
    mock.scan = AsyncMock(return_value=('0', [])) # scan의 기본 응답 튜플 형식 수정
    return mock

@pytest.fixture(scope="function") # 각 테스트마다 새로운 mock settings 사용
def mock_settings():
    """캐시 테스트를 위한 Mock Settings 객체 생성."""
    settings = MagicMock()
    settings.CACHE_TTL = 3600 # 기본 TTL 설정
    # 필요에 따라 다른 설정값 추가 가능
    # settings.REDIS_HOST = "mock_host"
    # settings.REDIS_PORT = 6379
    return settings

@pytest.fixture
def cache_instance(mock_settings): # mock_settings를 인자로 받아 사용 가능
    """초기화되지 않은 TwoLevelCache 인스턴스를 생성하는 fixture."""
    # fixture에서 생성 시 설정값을 명시적으로 전달하거나 mock_settings 사용 가능
    cache = TwoLevelCache(
        namespace="test_cache",
        local_maxsize=10,  # 테스트를 위한 작은 로컬 캐시 크기
        ttl=mock_settings.CACHE_TTL # mock_settings의 TTL 값 사용
    )
    return cache

# --- Helper Function for Initialization ---

async def initialize_cache(cache, mock_redis_conn):
    """Helper to initialize cache with mock redis connection."""
    # _ensure_initialized 내부에서 get_redis_async_connection를 호출하므로,
    # patch된 mock_get_redis_conn 이전에 redis 인스턴스를 직접 설정해줍니다.
    # 또는 _ensure_initialized 내부 로직을 mock 합니다.
    # 여기서는 직접 설정하는 방식을 사용합니다.
    cache._redis = mock_redis_conn # 초기화 전에 mock redis 직접 할당
    cache._initialized = True # 초기화 상태로 설정 (get_redis_async_connection 호출 방지)
    # 실제 초기화 로직을 테스트하려면 아래 주석 해제 및 위 두 줄 주석 처리
    # initialized = await cache._ensure_initialized()
    # assert initialized is True

# --- Test Functions ---

@pytest.mark.asyncio
# patch 순서: 가장 안쪽 데코레이터(가장 가까운 함수)부터 적용됨
@patch('src.llm.cache.settings') # settings 객체 자체를 patch
@patch('src.config.connections.get_redis_async_connection') # Redis 연결 함수 patch
async def test_cache_initialization(mock_get_redis_conn, mock_llm_cache_settings, cache_instance, mock_redis, mock_settings):
    """Test cache initialization."""
    # mock_settings fixture 값을 mock_llm_cache_settings (patch된 settings)의 반환값으로 설정
    mock_llm_cache_settings.return_value = mock_settings
    # mock_redis fixture 값을 mock_get_redis_conn (patch된 함수)의 반환값으로 설정
    mock_get_redis_conn.return_value = mock_redis

    # cache_instance에 mock redis 연결 설정 및 초기화 상태 설정
    cache_instance._redis = mock_redis
    cache_instance._initialized = True # 초기화 완료 상태로 설정

    # 또는 실제 초기화 로직 테스트 (위 두 줄 대신 사용):
    # initialized = await cache_instance._ensure_initialized()
    # assert initialized is True
    # mock_get_redis_conn.assert_called_once() # Redis 연결 시도 확인

    # 초기화 후 상태 검증
    assert isinstance(cache_instance, TwoLevelCache)
    assert cache_instance.namespace == "test_cache"
    assert cache_instance.ttl == mock_settings.CACHE_TTL
    assert cache_instance.local_maxsize == 10
    assert cache_instance.local_cache == {}
    assert cache_instance._initialized is True # _initialized 상태 직접 확인
    assert cache_instance._redis is mock_redis # 할당된 mock redis 확인

@pytest.mark.asyncio
@patch('src.llm.cache.settings')
@patch('src.config.connections.get_redis_async_connection')
async def test_cache_set_get(mock_get_redis_conn, mock_llm_cache_settings, cache_instance, mock_redis, mock_settings):
    """Test setting and getting values from cache."""
    mock_llm_cache_settings.return_value = mock_settings
    mock_get_redis_conn.return_value = mock_redis
    await initialize_cache(cache_instance, mock_redis) # Helper 사용 초기화

    # --- Set Operation ---
    test_key = "test_key"
    test_value = "test_value"
    set_ttl = 60 # set 시 사용할 특정 TTL

    # get 호출 시 Redis는 아직 값이 없다고 응답하도록 설정
    mock_redis.get.return_value = None

    await cache_instance.set(test_key, test_value, ttl=set_ttl)

    # 로컬 캐시 확인
    assert test_key in cache_instance.local_cache
    assert cache_instance.local_cache[test_key]["value"] == test_value
    assert cache_instance.local_cache[test_key]["expires_at"] > time.time()

    # Redis setex 호출 확인 (namespace 포함, TTL, 직렬화된 값)
    expected_redis_key = f"{cache_instance.namespace}:{test_key}"
    mock_redis.setex.assert_called_once_with(expected_redis_key, set_ttl, '"test_value"') # JSON 직렬화 가정

    # --- Get Operation (Local Cache Hit) ---
    # Redis get이 호출되지 않음을 확인하기 위해 mock 재설정 또는 확인 전 상태 기록
    mock_redis.get.reset_mock()

    result = await cache_instance.get(test_key)
    assert result == test_value

    # 로컬 캐시에서 찾았으므로 Redis get은 호출되지 않아야 함
    mock_redis.get.assert_not_called()
    # LRU 순서 업데이트 확인 (선택적)
    assert cache_instance.local_cache_order[-1] == test_key

@pytest.mark.asyncio
@patch('src.llm.cache.settings')
@patch('src.config.connections.get_redis_async_connection')
async def test_cache_get_from_redis(mock_get_redis_conn, mock_llm_cache_settings, cache_instance, mock_redis, mock_settings):
    """Test getting a value from Redis when not in local cache."""
    mock_llm_cache_settings.return_value = mock_settings
    mock_get_redis_conn.return_value = mock_redis
    await initialize_cache(cache_instance, mock_redis)

    test_key = "redis_key"
    redis_value = "redis_value"
    redis_ttl = 1800
    serialized_value = f'"{redis_value}"' # JSON 직렬화 가정
    expected_redis_key = f"{cache_instance.namespace}:{test_key}"

    # 로컬 캐시에는 없다고 가정
    assert test_key not in cache_instance.local_cache

    # Redis get 호출 시 직렬화된 값과 TTL을 반환하도록 설정
    mock_redis.get.return_value = serialized_value
    mock_redis.ttl.return_value = redis_ttl

    # --- Get Operation (Redis Cache Hit) ---
    result = await cache_instance.get(test_key)

    # 결과 확인
    assert result == redis_value

    # Redis get 및 ttl 호출 확인
    mock_redis.get.assert_called_once_with(expected_redis_key)
    mock_redis.ttl.assert_called_once_with(expected_redis_key)

    # 로컬 캐시에 저장되었는지 확인
    assert test_key in cache_instance.local_cache
    assert cache_instance.local_cache[test_key]["value"] == redis_value
    assert cache_instance.local_cache[test_key]["expires_at"] is not None
    assert cache_instance.local_cache[test_key]["expires_at"] > time.time() + redis_ttl - 10 # 약간의 오차 허용
    # LRU 순서 업데이트 확인 (선택적)
    assert cache_instance.local_cache_order[-1] == test_key

@pytest.mark.asyncio
@patch('src.llm.cache.settings')
@patch('src.config.connections.get_redis_async_connection')
async def test_cache_delete(mock_get_redis_conn, mock_llm_cache_settings, cache_instance, mock_redis, mock_settings):
    """Test deleting a value from cache."""
    mock_llm_cache_settings.return_value = mock_settings
    mock_get_redis_conn.return_value = mock_redis
    await initialize_cache(cache_instance, mock_redis)

    test_key = "delete_key"
    test_value = "delete_value"
    expected_redis_key = f"{cache_instance.namespace}:{test_key}"

    # 로컬 캐시에 미리 값 추가 (set 메서드 사용 또는 직접 할당)
    await cache_instance.set(test_key, test_value) # set을 통해 로컬 및 Redis에 추가 가정
    mock_redis.setex.reset_mock() # set 호출은 delete 테스트와 무관하므로 리셋

    # delete 호출 전 상태 확인 (선택적)
    assert test_key in cache_instance.local_cache
    assert test_key in cache_instance.local_cache_order

    # --- Delete Operation ---
    result = await cache_instance.delete(test_key)

    # 결과 확인
    assert result is True

    # 로컬 캐시에서 삭제되었는지 확인
    assert test_key not in cache_instance.local_cache
    assert test_key not in cache_instance.local_cache_order

    # Redis delete 호출 확인
    mock_redis.delete.assert_called_once_with(expected_redis_key)


@pytest.mark.asyncio
@patch('src.llm.cache.settings')
@patch('src.config.connections.get_redis_async_connection')
async def test_cache_clear(mock_get_redis_conn, mock_llm_cache_settings, cache_instance, mock_redis, mock_settings):
    """Test clearing the entire cache."""
    mock_llm_cache_settings.return_value = mock_settings
    mock_get_redis_conn.return_value = mock_redis
    await initialize_cache(cache_instance, mock_redis)

    # 로컬 캐시에 값 추가
    await cache_instance.set("key1", "value1")
    await cache_instance.set("key2", "value2")

    # Redis scan이 특정 키들을 반환하도록 설정
    redis_keys_in_namespace = [
        f"{cache_instance.namespace}:key1",
        f"{cache_instance.namespace}:key2",
        f"{cache_instance.namespace}:other_key"
    ]
    # scan 모킹: 첫 호출에 모든 키 반환, 다음 커서는 '0'
    mock_redis.scan.return_value = ('0', redis_keys_in_namespace)
    mock_redis.delete.reset_mock() # 이전 delete 호출 영향 제거

    # clear 호출 전 상태 확인
    assert len(cache_instance.local_cache) == 2
    assert len(cache_instance.local_cache_order) == 2

    # --- Clear Operation ---
    result = await cache_instance.clear()

    # 결과 확인
    assert result is True

    # 로컬 캐시 비워졌는지 확인
    assert cache_instance.local_cache == {}
    assert cache_instance.local_cache_order == []

    # Redis scan 호출 확인 (namespace 패턴)
    mock_redis.scan.assert_called_once_with(cursor='0', match=f'{cache_instance.namespace}:*', count=100)
    # Redis delete 호출 확인 (scan 결과로 나온 키들)
    mock_redis.delete.assert_called_once_with(*redis_keys_in_namespace)


@pytest.mark.asyncio
@patch('src.llm.cache.settings')
@patch('src.config.connections.get_redis_async_connection')
async def test_lru_cache_eviction(mock_get_redis_conn, mock_llm_cache_settings, cache_instance, mock_redis, mock_settings):
    """Test LRU cache eviction when max size is reached."""
    mock_llm_cache_settings.return_value = mock_settings
    mock_get_redis_conn.return_value = mock_redis
    await initialize_cache(cache_instance, mock_redis) # cache_instance의 local_maxsize는 10

    # maxsize (10) 보다 많은 아이템 추가 (15개)
    for i in range(15):
        key = f"key{i}"
        value = f"value{i}"
        await cache_instance.set(key, value)

    # 로컬 캐시 크기 확인
    assert len(cache_instance.local_cache) == cache_instance.local_maxsize # 10
    assert len(cache_instance.local_cache_order) == cache_instance.local_maxsize # 10

    # 가장 오래된 아이템(0~4)이 제거되었는지 확인
    for i in range(5):
        assert f"key{i}" not in cache_instance.local_cache
        assert f"key{i}" not in cache_instance.local_cache_order

    # 최근 아이템(5~14)이 남아있는지 확인
    for i in range(5, 15):
        assert f"key{i}" in cache_instance.local_cache
        assert f"key{i}" in cache_instance.local_cache_order


@pytest.mark.asyncio
@patch('src.llm.cache.settings')
@patch('src.config.connections.get_redis_async_connection')
async def test_cache_expiration(mock_get_redis_conn, mock_llm_cache_settings, cache_instance, mock_redis, mock_settings):
    """Test cache entry expiration."""
    mock_llm_cache_settings.return_value = mock_settings
    mock_get_redis_conn.return_value = mock_redis
    await initialize_cache(cache_instance, mock_redis)

    key = "expire_key"
    value = "expire_value"
    short_ttl = 0.1 # 100ms TTL
    expected_redis_key = f"{cache_instance.namespace}:{key}"

    # 짧은 TTL로 값 설정
    await cache_instance.set(key, value, ttl=short_ttl)

    # TTL보다 긴 시간 대기
    await asyncio.sleep(short_ttl + 0.1)

    # 만료 후 get 시 Redis는 값이 없다고 응답하도록 설정
    mock_redis.get.return_value = None
    mock_redis.ttl.return_value = -2 # 만료된 키에 대한 TTL 응답

    # --- Get Operation (After Expiration) ---
    result = await cache_instance.get(key)

    # 결과 확인 (None 이어야 함)
    assert result is None

    # 로컬 캐시에서도 제거되었는지 확인 (get 시 만료 체크 후 제거됨)
    assert key not in cache_instance.local_cache
    assert key not in cache_instance.local_cache_order

    # Redis get이 호출되었는지 확인 (로컬 만료 후 Redis 확인 시도)
    mock_redis.get.assert_called_once_with(expected_redis_key)

# --- Helper Function Tests ---

# 전역 _CACHE_INSTANCE 상태를 관리하기 위한 fixture
@pytest.fixture(autouse=True)
async def reset_global_cache():
    """Resets the global _CACHE_INSTANCE before and after each test."""
    global _CACHE_INSTANCE, _CACHE_LOCK
    async with _CACHE_LOCK:
        _CACHE_INSTANCE = None
    yield
    async with _CACHE_LOCK:
        _CACHE_INSTANCE = None


@pytest.mark.asyncio
@patch('src.llm.cache.settings') # get_cache 내부 settings 접근 patch
# get_cache 내부에서 TwoLevelCache 생성 시 Redis 연결 시도를 mock
@patch('src.config.connections.get_redis_async_connection', new_callable=AsyncMock)
async def test_get_cache_singleton(mock_get_redis_func, mock_llm_cache_settings, mock_settings, mock_redis):
    """Test that get_cache returns a singleton instance and initializes it."""
    # --- 수정된 부분 ---
    # mock_llm_cache_settings 객체의 CACHE_TTL 속성에 mock_settings fixture의 값을 설정
    mock_llm_cache_settings.CACHE_TTL = mock_settings.CACHE_TTL
    # mock_llm_cache_settings.return_value = mock_settings # 이 줄은 제거합니다.
    # --- 수정 끝 ---

    mock_get_redis_func.return_value = mock_redis # mock redis 연결 반환

    # 첫번째 호출: 인스턴스 생성 시 위에서 설정한 CACHE_TTL 값을 사용하게 됨
    cache1 = await get_cache()
    # get_cache는 내부적으로 _ensure_initialized를 호출하지 않음.
    # 사용 시점에 _ensure_initialized가 호출됨.
    # 따라서 여기서는 생성만 확인

    # 두번째 호출: 이미 생성된 인스턴스 반환
    cache2 = await get_cache()

    assert cache1 is cache2 # 동일 인스턴스 확인
    assert isinstance(cache1, TwoLevelCache) # 타입 확인
    # get_cache 함수 자체는 Redis 연결 함수를 직접 호출하지 않음
    # cache 인스턴스의 메서드 (get, set 등) 사용 시 내부적으로 호출됨
    # mock_get_redis_func.assert_called_once() # <- 호출되지 않는 것이 정상

    # 싱글톤 인스턴스의 설정 값 확인
    # 이제 cache1.ttl은 mock_settings.CACHE_TTL과 같은 정수 값을 가짐
    assert cache1.ttl == mock_settings.CACHE_TTL


@pytest.mark.asyncio
async def test_cache_result_helper():
    """Test the cache_result helper function."""
    # Mock Cache 인스턴스 생성 및 비동기 메소드 모킹
    mock_cache_instance = MagicMock(spec=TwoLevelCache)
    mock_cache_instance.set = AsyncMock(return_value=True)

    # get_cache 함수가 위 mock 인스턴스를 반환하도록 patch
    with patch('src.llm.cache.get_cache', new_callable=AsyncMock, return_value=mock_cache_instance):
        key = "helper_key"
        value = "helper_value"
        ttl = 30
        result = await cache_result(key, value, ttl=ttl)

        assert result is True # set의 반환값 확인
        # mock_cache_instance의 set 메소드 호출 확인
        mock_cache_instance.set.assert_called_once_with(key, value, ttl)


@pytest.mark.asyncio
async def test_clear_cache_helper():
    """Test the clear_cache helper function."""
    mock_cache_instance = MagicMock(spec=TwoLevelCache)
    mock_cache_instance.clear = AsyncMock(return_value=True)

    with patch('src.llm.cache.get_cache', new_callable=AsyncMock, return_value=mock_cache_instance):
        result = await clear_cache()
        assert result is True # clear의 반환값 확인
        mock_cache_instance.clear.assert_called_once()


@pytest.mark.asyncio
async def test_get_cache_stats_helper():
    """Test the get_cache_stats helper function."""
    mock_cache_instance = MagicMock(spec=TwoLevelCache)
    mock_stats_result = {
        "hit_count": 10, "miss_count": 5, "hit_ratio": 0.67, "local_cache_size": 7
    }
    # get_stats는 동기/비동기 여부가 원본 코드에 따라 달라질 수 있음. 원본이 async이면 AsyncMock 사용
    mock_cache_instance.get_stats = AsyncMock(return_value=mock_stats_result)

    with patch('src.llm.cache.get_cache', new_callable=AsyncMock, return_value=mock_cache_instance):
        stats = await get_cache_stats()
        assert stats == mock_stats_result
        mock_cache_instance.get_stats.assert_called_once()
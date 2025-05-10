# tests/test_memory.py
import pytest
import pytest_asyncio
import fakeredis.aioredis
import asyncio
import time
import msgspec # 테스트 데이터 생성을 위해 추가
from typing import Any, Dict, List, Optional

from cachetools import TTLCache
from unittest.mock import MagicMock, AsyncMock

# 테스트 대상 모듈 임포트
from src.memory import memory_store
from src.config import settings # 설정값 사용을 위해 임포트
from src.utils.serialization import SerializationFormat # 직렬화 포맷 사용
from src.config.errors import ErrorCode, MemoryError, ConnectionError
from src.memory.memory_manager import MemoryManager, DEFAULT_KEY_PREFIX

# --- 테스트 Fixtures ---

@pytest.fixture
def mock_memory_store_funcs(mocker) -> Dict[str, AsyncMock]:
    """memory_store의 함수들을 모킹하여 반환하는 fixture."""
    return {
        "save_state": mocker.patch("src.memory.memory_store.save_state", new_callable=AsyncMock, return_value=True),
        "load_state": mocker.patch("src.memory.memory_store.load_state", new_callable=AsyncMock, return_value=None), # 기본적으로 None 반환
        "delete_state": mocker.patch("src.memory.memory_store.delete_state", new_callable=AsyncMock, return_value=True),
        "exists": mocker.patch("src.memory.memory_store.exists", new_callable=AsyncMock, return_value=False),
        "get_history": mocker.patch("src.memory.memory_store.get_history", new_callable=AsyncMock, return_value=[]),
    }

@pytest.fixture
def memory_manager(mock_memory_store_funcs) -> MemoryManager:
    """테스트를 위한 MemoryManager 인스턴스를 생성하는 fixture."""
    # MemoryManager 생성 시 memory_store 함수들이 이미 패치된 상태
    # 기본 캐시 설정 사용 (cache_ttl > 0)
    manager = MemoryManager(default_ttl=3600, cache_ttl=60, cache_size=10)
    # 내부 함수 참조가 patch된 함수를 가리키도록 강제 업데이트 (필요한 경우)
    # 생성자에서 이미 memory_store를 import하므로 일반적으로는 불필요
    # manager._save_func = mock_memory_store_funcs["save_state"]
    # manager._load_func = mock_memory_store_funcs["load_state"]
    # ... 등
    return manager

@pytest.fixture
def memory_manager_no_cache() -> MemoryManager:
    """캐시가 비활성화된 MemoryManager 인스턴스를 생성하는 fixture."""
    manager = MemoryManager(default_ttl=3600, cache_ttl=0) # cache_ttl=0으로 캐시 비활성화
    return manager

@pytest_asyncio.fixture(scope="function") # 각 테스트 함수마다 새로운 fakeredis 인스턴스 사용
async def fake_redis_client():
    """Fakeredis 비동기 클라이언트를 제공하는 pytest fixture."""
    # decode_responses=False 로 설정해야 bytes 로 반환됨 (memory_store 와 동일하게)
    client = await fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.flushall() # 테스트 후 데이터 정리
    await client.close() # 클라이언트 닫기

@pytest_asyncio.fixture(autouse=True) # 모든 테스트 함수에 자동으로 적용
async def mock_redis_connection(mocker, fake_redis_client):
    """memory_store의 get_redis_async_connection을 mock하여 fakeredis 클라이언트를 반환."""
    # mocker.patch를 사용하여 memory_store 모듈 내의 get_redis_async_connection 함수를 mock
    mocked_get_conn = mocker.patch(
        "src.memory.memory_store.get_redis_async_connection",
        return_value=fake_redis_client # 호출 시 fake_redis_client 반환
    )
    yield mocked_get_conn # 테스트 함수 실행
    # 테스트 종료 후 mock 복원 (pytest-mock이 자동으로 처리)

# --- 테스트 데이터 ---
test_key_1 = "test_key_1"
test_value_1 = {"a": 1, "b": "hello"}
test_value_bytes_1 = msgspec.msgpack.Encoder().encode(test_value_1)

test_key_2 = "test_key_2"
test_value_2 = [1, 2, "world", {"nested": True}]
test_value_bytes_2 = msgspec.msgpack.Encoder().encode(test_value_2)

history_prefix = "history:user123:"
history_data = [
    (f"{history_prefix}1678886400.1", {"role": "user", "content": "Hello"}),
    (f"{history_prefix}1678886401.2", {"role": "assistant", "content": "Hi there!"}),
    (f"{history_prefix}1678886402.3", {"role": "user", "content": "How are you?"}),
]
history_data_bytes = {k: msgspec.msgpack.Encoder().encode(v) for k, v in history_data}


# --- TestMemoryStoreRedis 클래스 ---
# @pytest.mark.usefixtures("mock_redis_connection") # 클래스 레벨 적용 가능
@pytest.mark.asyncio # 비동기 테스트 클래스임을 명시
class TestMemoryStoreRedis:
    """memory_store.py의 Redis 백엔드 함수들을 테스트합니다."""

    async def test_redis_save_and_load_state(self, fake_redis_client):
        """_redis_save_state와 _redis_load_state 기본 기능 테스트."""
        # 저장 테스트
        success = await memory_store._redis_save_state(test_key_1, test_value_1)
        assert success is True

        # Fakeredis에서 직접 확인
        stored_bytes = await fake_redis_client.get(test_key_1)
        assert stored_bytes == test_value_bytes_1

        # 로드 테스트
        loaded_value = await memory_store._redis_load_state(test_key_1)
        assert loaded_value == test_value_1

    async def test_redis_load_state_not_found(self):
        """키가 없을 때 _redis_load_state가 default를 반환하는지 테스트."""
        default_value = "not_found"
        loaded_value = await memory_store._redis_load_state("non_existent_key", default=default_value)
        assert loaded_value == default_value

        loaded_value_none = await memory_store._redis_load_state("non_existent_key_2")
        assert loaded_value_none is None

    async def test_redis_save_state_with_ttl(self, fake_redis_client):
        """_redis_save_state의 TTL 기능 테스트."""
        ttl_seconds = 1 # 짧은 TTL 설정
        success = await memory_store._redis_save_state(test_key_2, test_value_2, ttl=ttl_seconds)
        assert success is True

        # TTL 확인 (fakeredis는 실제 시간 기반 TTL 지원)
        stored_value = await memory_store._redis_load_state(test_key_2)
        assert stored_value == test_value_2

        # TTL 이후 확인
        await asyncio.sleep(ttl_seconds + 0.1)
        expired_value = await memory_store._redis_load_state(test_key_2)
        assert expired_value is None # 만료되어 None 반환되어야 함

        # TTL=0 또는 음수일 때 영구 저장 확인
        success_neg_ttl = await memory_store._redis_save_state(test_key_1, test_value_1, ttl=-1)
        assert success_neg_ttl is True
        key_ttl = await fake_redis_client.ttl(test_key_1)
        assert key_ttl == -1 # -1은 TTL이 없음을 의미

    async def test_redis_delete_state(self):
        """_redis_delete_state 기능 테스트."""
        # 먼저 저장
        await memory_store._redis_save_state(test_key_1, test_value_1)
        assert await memory_store._redis_load_state(test_key_1) == test_value_1

        # 삭제 테스트
        deleted = await memory_store._redis_delete_state(test_key_1)
        assert deleted is True

        # 삭제 후 로드 확인
        assert await memory_store._redis_load_state(test_key_1) is None

        # 없는 키 삭제 시도
        deleted_non_existent = await memory_store._redis_delete_state("non_existent_key")
        assert deleted_non_existent is False

    async def test_redis_exists(self):
        """_redis_exists 기능 테스트."""
        # 키 없을 때
        assert await memory_store._redis_exists("non_existent_key") is False

        # 키 저장 후
        await memory_store._redis_save_state(test_key_1, test_value_1)
        assert await memory_store._redis_exists(test_key_1) is True

        # 키 삭제 후
        await memory_store._redis_delete_state(test_key_1)
        assert await memory_store._redis_exists(test_key_1) is False

    async def test_redis_get_history(self, fake_redis_client):
        """_redis_get_history 기능 테스트 (키 형식 및 정렬 가정)."""
        # 테스트 데이터 저장
        for key, value_bytes in history_data_bytes.items():
            await fake_redis_client.set(key, value_bytes)

        # 전체 기록 조회 (최신순 정렬 확인)
        history = await memory_store._redis_get_history(history_prefix)
        assert len(history) == 3
        assert history[0] == history_data[2][1] # 가장 최신 기록 (1678886402.3)
        assert history[1] == history_data[1][1]
        assert history[2] == history_data[0][1] # 가장 오래된 기록

        # Limit 적용 테스트
        history_limit_2 = await memory_store._redis_get_history(history_prefix, limit=2)
        assert len(history_limit_2) == 2
        assert history_limit_2[0] == history_data[2][1]
        assert history_limit_2[1] == history_data[1][1]

        # 없는 접두사 테스트
        history_none = await memory_store._redis_get_history("non_existent_prefix:")
        assert history_none == []

    async def test_redis_serialization_error(self, mocker):
        """직렬화 실패 시 MemoryError 발생하는지 테스트."""
        # serialize 함수가 예외를 발생시키도록 mock
        mocker.patch("src.memory.memory_store.serialize", side_effect=TypeError("Cannot serialize"))

        with pytest.raises(MemoryError) as exc_info:
            await memory_store._redis_save_state(test_key_1, {"unserializable": lambda: None})

        assert exc_info.value.code == ErrorCode.REDIS_OPERATION_ERROR
        assert "Failed to save key" in exc_info.value.message

    async def test_redis_deserialization_error(self, fake_redis_client):
        """역직렬화 실패 시 MemoryError 발생하는지 테스트."""
        # 잘못된 형식의 바이트 저장
        await fake_redis_client.set(test_key_1, b"invalid msgpack data")

        with pytest.raises(MemoryError) as exc_info:
            await memory_store._redis_load_state(test_key_1)

        assert exc_info.value.code == ErrorCode.MEMORY_RETRIEVAL_ERROR
        assert "Failed to deserialize data" in exc_info.value.message

    async def test_redis_connection_error(self, mocker):
        """Redis 연결 실패 시 MemoryError 발생하는지 테스트."""
        # get_redis_async_connection 함수가 예외를 발생시키도록 mock
        mocker.patch(
            "src.memory.memory_store.get_redis_async_connection",
            side_effect=ConnectionError(code=ErrorCode.REDIS_CONNECTION_ERROR, message="Connection failed")
        )

        with pytest.raises(MemoryError) as exc_info:
            await memory_store._redis_load_state(test_key_1)

        assert exc_info.value.code == ErrorCode.REDIS_OPERATION_ERROR # convert_exception 결과 확인
        assert "Failed to load key" in exc_info.value.message
        assert isinstance(exc_info.value.original_error, ConnectionError)


@pytest.mark.asyncio
class TestMemoryManager:
    """MemoryManager 클래스의 기능을 테스트합니다."""

    context_id = "test_context"
    key = "my_key"
    value = {"data": "some_value"}
    full_key = f"{DEFAULT_KEY_PREFIX}:{context_id}:{key}" # 예상되는 전체 키

    async def test_get_full_key(self, memory_manager: MemoryManager):
        """_get_full_key 메서드가 올바른 키를 생성하는지 테스트."""
        assert memory_manager._get_full_key(self.context_id, self.key) == self.full_key
        with pytest.raises(ValueError):
            memory_manager._get_full_key("", self.key)
        with pytest.raises(ValueError):
            memory_manager._get_full_key(self.context_id, "")

    async def test_get_effective_ttl(self, memory_manager: MemoryManager):
        """_get_effective_ttl 메서드가 TTL을 올바르게 계산하는지 테스트."""
        manager_default_ttl = memory_manager.default_ttl # 예: 3600
        assert memory_manager._get_effective_ttl(None) == manager_default_ttl # None -> 기본값
        assert memory_manager._get_effective_ttl(100) == 100 # 명시적 양수 TTL
        assert memory_manager._get_effective_ttl(0) is None # 0 -> 영구
        assert memory_manager._get_effective_ttl(-1) is None # 음수 -> 영구

        # 기본 TTL이 0 또는 None일 때 테스트
        manager_no_ttl = MemoryManager(default_ttl=0)
        assert manager_no_ttl._get_effective_ttl(None) is None
        assert manager_no_ttl._get_effective_ttl(100) == 100

    async def test_save_state_calls_store_with_correct_args(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """save_state가 올바른 인수로 memory_store.save_state를 호출하는지 테스트."""
        ttl = 600
        expected_ttl = 600 # _get_effective_ttl 결과 예상
        success = await memory_manager.save_state(self.context_id, self.key, self.value, ttl=ttl)

        assert success is True
        mock_memory_store_funcs["save_state"].assert_awaited_once_with(self.full_key, self.value, expected_ttl)

    async def test_save_state_updates_cache(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """save_state가 성공 시 L1 캐시를 업데이트하는지 테스트."""
        await memory_manager.save_state(self.context_id, self.key, self.value)

        # 캐시 확인 (내부 캐시 직접 접근 - 테스트 목적)
        assert memory_manager._cache is not None
        assert self.full_key in memory_manager._cache
        assert memory_manager._cache[self.full_key] == self.value
        mock_memory_store_funcs["save_state"].assert_awaited_once() # 저장소 호출 확인

    async def test_save_state_no_cache_update_if_disabled(self, memory_manager_no_cache: MemoryManager, mock_memory_store_funcs):
        """캐시 비활성화 시 save_state가 캐시를 업데이트하지 않는지 테스트."""
        await memory_manager_no_cache.save_state(self.context_id, self.key, self.value)

        assert memory_manager_no_cache._cache is None # 캐시 자체가 없음
        mock_memory_store_funcs["save_state"].assert_awaited_once()

    async def test_save_state_handles_store_failure(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """저장소 저장 실패 시 save_state가 False를 반환하고 캐시 업데이트 안 하는지 테스트."""
        mock_memory_store_funcs["save_state"].return_value = False # 저장 실패 시뮬레이션

        success = await memory_manager.save_state(self.context_id, self.key, self.value)
        assert success is False
        assert memory_manager._cache is not None
        assert self.full_key not in memory_manager._cache # 캐시 업데이트 안 됨

    async def test_load_state_cache_hit(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """load_state 캐시 히트 시나리오 테스트."""
        # 캐시에 미리 값 저장
        memory_manager._cache[self.full_key] = self.value

        loaded_value = await memory_manager.load_state(self.context_id, self.key)

        assert loaded_value == self.value
        # 캐시 히트 시 저장소 함수는 호출되지 않아야 함
        mock_memory_store_funcs["load_state"].assert_not_awaited()

    async def test_load_state_cache_miss(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """load_state 캐시 미스 시나리오 테스트 (저장소에서 로드 및 캐시 저장)."""
        # 저장소 함수가 값을 반환하도록 설정
        mock_memory_store_funcs["load_state"].return_value = self.value

        loaded_value = await memory_manager.load_state(self.context_id, self.key)

        assert loaded_value == self.value
        # 저장소 함수가 호출되었는지 확인
        mock_memory_store_funcs["load_state"].assert_awaited_once_with(self.full_key, None) # default=None
        # 로드 후 캐시에 저장되었는지 확인
        assert memory_manager._cache is not None
        assert self.full_key in memory_manager._cache
        assert memory_manager._cache[self.full_key] == self.value

    async def test_load_state_cache_miss_not_found(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """캐시와 저장소 모두에 키가 없을 때 default 값이 반환되는지 테스트."""
        # 저장소 함수가 default 값 (None)을 반환하도록 설정 (기본 mock 설정)
        mock_memory_store_funcs["load_state"].return_value = None
        default_val = "i_am_default"

        loaded_value = await memory_manager.load_state(self.context_id, self.key, default=default_val)

        assert loaded_value == default_val
        mock_memory_store_funcs["load_state"].assert_awaited_once_with(self.full_key, default_val)
        # 캐시에는 저장되지 않아야 함 (default 값이므로)
        assert memory_manager._cache is not None
        assert self.full_key not in memory_manager._cache

    async def test_load_state_cache_disabled(self, memory_manager_no_cache: MemoryManager, mock_memory_store_funcs):
        """캐시 비활성화 시 load_state가 항상 저장소를 호출하는지 테스트."""
        mock_memory_store_funcs["load_state"].return_value = self.value

        loaded_value = await memory_manager_no_cache.load_state(self.context_id, self.key)

        assert loaded_value == self.value
        mock_memory_store_funcs["load_state"].assert_awaited_once_with(
             memory_manager_no_cache._get_full_key(self.context_id, self.key), None
        )
        assert memory_manager_no_cache._cache is None

    async def test_load_state_cache_ttl(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """캐시 TTL이 만료된 후 저장소를 호출하는지 테스트."""
        cache_ttl = 0.1 # 매우 짧은 캐시 TTL 설정 (테스트 목적)
        manager = MemoryManager(cache_ttl=cache_ttl)
        # Mock 함수 재설정 (manager 재생성 시 patch가 풀릴 수 있음)
        manager._load_func = mock_memory_store_funcs["load_state"]
        mock_memory_store_funcs["load_state"].return_value = self.value

        # 1. 캐시에 저장
        await manager.save_state(self.context_id, self.key, self.value)
        assert manager._cache[self.full_key] == self.value

        # 2. TTL 만료 전 로드 (캐시 히트)
        loaded_before_expiry = await manager.load_state(self.context_id, self.key)
        assert loaded_before_expiry == self.value
        mock_memory_store_funcs["load_state"].assert_not_awaited() # 아직 저장소 호출 안 됨

        # 3. TTL 만료 대기
        await asyncio.sleep(cache_ttl + 0.05)

        # 4. TTL 만료 후 로드 (캐시 미스 -> 저장소 호출)
        loaded_after_expiry = await manager.load_state(self.context_id, self.key)
        assert loaded_after_expiry == self.value
        mock_memory_store_funcs["load_state"].assert_awaited_once() # 저장소 호출됨

    async def test_delete_state_calls_store_and_clears_cache(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """delete_state가 저장소 함수를 호출하고 캐시를 제거하는지 테스트."""
        # 캐시에 미리 값 저장
        memory_manager._cache[self.full_key] = self.value
        assert self.full_key in memory_manager._cache

        # 삭제 함수 호출
        success = await memory_manager.delete_state(self.context_id, self.key)

        assert success is True
        # 저장소 함수 호출 확인
        mock_memory_store_funcs["delete_state"].assert_awaited_once_with(self.full_key)
        # 캐시 제거 확인
        assert memory_manager._cache is not None
        assert self.full_key not in memory_manager._cache

    async def test_delete_state_cache_disabled(self, memory_manager_no_cache: MemoryManager, mock_memory_store_funcs):
        """캐시 비활성화 시 delete_state가 저장소 함수만 호출하는지 테스트."""
        success = await memory_manager_no_cache.delete_state(self.context_id, self.key)
        assert success is True
        mock_memory_store_funcs["delete_state"].assert_awaited_once_with(
             memory_manager_no_cache._get_full_key(self.context_id, self.key)
        )
        assert memory_manager_no_cache._cache is None

    async def test_exists_cache_hit(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """exists 캐시 히트 시나리오 테스트."""
        memory_manager._cache[self.full_key] = self.value # 캐시에 저장
        result = await memory_manager.exists(self.context_id, self.key)
        assert result is True
        mock_memory_store_funcs["exists"].assert_not_awaited() # 저장소 호출 안 됨

    async def test_exists_cache_miss_store_hit(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """exists 캐시 미스, 저장소 히트 시나리오 테스트."""
        mock_memory_store_funcs["exists"].return_value = True # 저장소에 존재
        result = await memory_manager.exists(self.context_id, self.key)
        assert result is True
        mock_memory_store_funcs["exists"].assert_awaited_once_with(self.full_key)
        # 캐시에는 저장되지 않음 (exists는 캐시 업데이트 안 함)
        assert self.full_key not in memory_manager._cache

    async def test_exists_cache_miss_store_miss(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """exists 캐시 미스, 저장소 미스 시나리오 테스트."""
        mock_memory_store_funcs["exists"].return_value = False # 저장소에 없음
        result = await memory_manager.exists(self.context_id, self.key)
        assert result is False
        mock_memory_store_funcs["exists"].assert_awaited_once_with(self.full_key)

    async def test_get_history_calls_store(self, memory_manager: MemoryManager, mock_memory_store_funcs):
        """get_history가 memory_store.get_history를 호출하는지 테스트."""
        history_key_prefix = "chat_history"
        expected_prefix_arg = f"{DEFAULT_KEY_PREFIX}:{self.context_id}:{history_key_prefix}"
        limit = 10
        mock_return = [{"role": "user", "content": "hi"}]
        mock_memory_store_funcs["get_history"].return_value = mock_return

        result = await memory_manager.get_history(self.context_id, history_key_prefix, limit=limit)

        assert result == mock_return
        mock_memory_store_funcs["get_history"].assert_awaited_once_with(expected_prefix_arg, limit)
        # get_history는 캐시 사용 안 함 확인 (캐시에 관련 키 없어야 함)
        assert memory_manager._cache is not None
        # history 결과는 캐시에 저장되지 않음
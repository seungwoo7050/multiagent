import json
import time
import uuid
import datetime
import asyncio
import pytest
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Import modules to test
from src.utils.ids import (
    generate_uuid, generate_short_uuid, generate_sequential_id,
    generate_snowflake_id, generate_prefixed_id, generate_task_id,
    generate_agent_id, generate_trace_id, generate_fingerprint,
    generate_short_fingerprint, is_valid_uuid, is_valid_snowflake,
    extract_timestamp_from_snowflake, generate_collision_resistant_id
)
from src.utils.serialization import (
    serialize, deserialize, serialize_to_json, deserialize_from_json,
    SerializationFormat, model_to_dict, model_to_json
)
from src.utils.timing import (
    timed, async_timed, Timer, AsyncTimer,
    get_current_time_ms, sleep_with_jitter
)

# Test classes for serialization
class TestEnum(Enum):
    ONE = 1
    TWO = 2
    THREE = 3

class TestModel(BaseModel):
    id: str
    name: str
    value: int
    tags: List[str]
    optional: Optional[str] = None

# Tests for ids.py
class TestIds:
    def test_generate_uuid(self):
        uuid_str = generate_uuid()
        assert is_valid_uuid(uuid_str)
        assert len(uuid_str) == 36
    
    def test_generate_short_uuid(self):
        short_uuid = generate_short_uuid()
        assert len(short_uuid) >= 16
        assert len(short_uuid) <= 24  # URL-safe base64 is variable length
    
    def test_generate_sequential_id(self):
        seq_id1 = generate_sequential_id()
        time.sleep(0.001)  # Ensure different timestamp
        seq_id2 = generate_sequential_id()
        
        # Sequential IDs should be different and follow pattern
        assert seq_id1 != seq_id2
        assert '-' in seq_id1
        
        # Test with prefix
        prefix_id = generate_sequential_id("test")
        assert prefix_id.startswith("test-")
    
    def test_generate_snowflake_id(self):
        snowflake_id = generate_snowflake_id()
        
        # Snowflake IDs should be integers fitting in 64 bits
        assert isinstance(snowflake_id, int)
        assert is_valid_snowflake(snowflake_id)
        
        # Generate a few more to check they're increasing
        prev_id = snowflake_id
        for _ in range(5):
            new_id = generate_snowflake_id()
            assert new_id > prev_id
            prev_id = new_id
    
    def test_extract_timestamp_from_snowflake(self):
        current_time_ms = int(time.time() * 1000)
        snowflake_id = generate_snowflake_id()
        
        # Extract timestamp and verify it's close to current time
        timestamp = extract_timestamp_from_snowflake(snowflake_id)
        assert timestamp <= current_time_ms + 10  # Allow small delay
        assert timestamp >= current_time_ms - 1000  # Allow reasonable past
    
    def test_generate_prefixed_id(self):
        prefixed_id = generate_prefixed_id("test")
        
        # Check format
        assert prefixed_id.startswith("test-")
        assert len(prefixed_id) >= 21  # prefix + hyphen + 16 chars
        
        # Test custom length
        custom_length_id = generate_prefixed_id("test", length=8)
        assert len(custom_length_id) == len("test-") + 8
    
    def test_generate_task_id(self):
        task_id = generate_task_id()
        assert task_id.startswith("task-")
        
        # With task type
        typed_task_id = generate_task_id("analysis")
        assert typed_task_id.startswith("task-analysis-")
    
    def test_generate_agent_id(self):
        agent_id = generate_agent_id("worker")
        assert agent_id.startswith("agent-worker-")
    
    def test_generate_trace_id(self):
        trace_id = generate_trace_id()
        assert trace_id.startswith("trace-")
    
    def test_fingerprinting(self):
        test_content = "test content for fingerprinting"
        fingerprint = generate_fingerprint(test_content)
        
        # SHA-256 produces 64 hex characters
        assert len(fingerprint) == 64
        
        # Same content should produce same fingerprint
        assert generate_fingerprint(test_content) == fingerprint
        
        # Test short fingerprint
        short_fingerprint = generate_short_fingerprint(test_content, length=8)
        assert len(short_fingerprint) == 8
        assert short_fingerprint == fingerprint[:8]
    
    def test_collision_resistant_id(self):
        # Test with string content
        cr_id1 = generate_collision_resistant_id("test", "content1")
        cr_id2 = generate_collision_resistant_id("test", "content2")
        
        # Different content should produce different IDs
        assert cr_id1 != cr_id2
        
        # Same prefix and content should produce deterministic part
        content_hash1 = cr_id1.split("-")[2]
        repro_id = generate_collision_resistant_id("test", "content1")
        content_hash2 = repro_id.split("-")[2]
        assert content_hash1 == content_hash2
        
        # Test with dict content
        dict_id = generate_collision_resistant_id("test", {"key": "value"})
        assert dict_id.startswith("test-")

# Tests for serialization.py
class TestSerialization:
    def test_simple_types_serialization(self):
        # Test basic types
        test_data = {
            "string": "test string",
            "int": 123,
            "float": 123.456,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }
        
        # JSON serialization
        json_bytes = serialize(test_data, format=SerializationFormat.JSON)
        json_result = deserialize(json_bytes, format=SerializationFormat.JSON)
        assert json_result == test_data
        
        # MessagePack serialization
        msgpack_bytes = serialize(test_data, format=SerializationFormat.MSGPACK)
        msgpack_result = deserialize(msgpack_bytes, format=SerializationFormat.MSGPACK)
        assert msgpack_result == test_data
    
    def test_complex_types_serialization(self):
        test_date = datetime.date.today()
        test_time = datetime.datetime.now().time()
        test_datetime = datetime.datetime.now()
        test_uuid = uuid.uuid4()
        test_set = {1, 2, 3}
        test_bytes = b'binary data'
        test_enum = TestEnum.TWO
        
        test_data = {
            "date": test_date,
            "time": test_time,
            "datetime": test_datetime,
            "uuid": test_uuid,
            "set": test_set,
            "bytes": test_bytes,
            "enum": test_enum
        }
        
        # JSON serialization
        json_bytes = serialize(test_data, format=SerializationFormat.JSON)
        json_result = deserialize(json_bytes, format=SerializationFormat.JSON)
        
        assert json_result["date"] == test_date
        assert json_result["time"] == test_time
        assert json_result["datetime"] == test_datetime
        assert json_result["uuid"] == test_uuid
        assert json_result["set"] == test_set
        assert json_result["bytes"] == test_bytes
        assert json_result["enum"] == test_enum
        
        # MessagePack serialization
        msgpack_bytes = serialize(test_data, format=SerializationFormat.MSGPACK)
        msgpack_result = deserialize(msgpack_bytes, format=SerializationFormat.MSGPACK)
        
        assert msgpack_result["date"] == test_date
        assert msgpack_result["time"] == test_time
        assert msgpack_result["datetime"] == test_datetime
        assert msgpack_result["uuid"] == test_uuid
        assert msgpack_result["set"] == test_set
        assert msgpack_result["bytes"] == test_bytes
        assert msgpack_result["enum"] == test_enum
    
    def test_model_serialization(self):
        model = TestModel(
            id="test-id",
            name="Test Model",
            value=123,
            tags=["tag1", "tag2"]
        )
        
        # Test model_to_dict and model_to_json remain unchanged
        model_dict = model_to_dict(model)
        assert model_dict["id"] == "test-id"
        assert model_dict["name"] == "Test Model"
        
        model_json = model_to_json(model)
        assert isinstance(model_json, str)
        assert json.loads(model_json)["id"] == "test-id"
        
        # Test full serialization/deserialization
        serialized = serialize(model, format=SerializationFormat.JSON)
        deserialized = deserialize(serialized, format=SerializationFormat.JSON)
        
        # Updated assertions for model instance
        assert deserialized.id == model.id
        assert deserialized.name == model.name
        assert deserialized.value == model.value
        assert deserialized.tags == model.tags
    
    def test_json_conversion(self):
        test_data = {"key": "value", "list": [1, 2, 3]}
        
        # Convert to JSON string
        json_str = serialize_to_json(test_data)
        assert isinstance(json_str, str)
        
        # Convert back from JSON string
        result = deserialize_from_json(json_str)
        assert result == test_data

# Tests for timing.py
class TestTiming:
    def test_timed_decorator(self):
        @timed()
        def example_function(sleep_time):
            time.sleep(sleep_time)
            return "result"
        
        # Call the decorated function
        result = example_function(0.01)
        assert result == "result"
        
        # Test with custom name
        @timed(name="custom_timer")
        def another_function():
            time.sleep(0.01)
            return "done"
        
        result = another_function()
        assert result == "done"
    
    def test_timer_context_manager(self):
        with Timer("test_timer") as timer:
            time.sleep(0.01)
        
        # Check that execution time was recorded
        assert timer.execution_time > 0
        assert timer.execution_time < 1  # Reasonable upper bound
        
        # Test with different log level
        with Timer("info_timer", log_level="info") as timer:
            time.sleep(0.01)
        
        assert timer.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_async_timed_decorator(self):
        @async_timed()
        async def example_async_function(sleep_time):
            await asyncio.sleep(sleep_time)
            return "async result"
        
        # Call the decorated async function
        result = await example_async_function(0.01)
        assert result == "async result"
    
    @pytest.mark.asyncio
    async def test_async_timer_context_manager(self):
        async with AsyncTimer("test_async_timer") as timer:
            await asyncio.sleep(0.01)
        
        # Check that execution time was recorded
        assert timer.execution_time > 0
        assert timer.execution_time < 1  # Reasonable upper bound
        
        # Test with different log level
        async with AsyncTimer("info_async_timer", log_level="info") as timer:
            await asyncio.sleep(0.01)
        
        assert timer.execution_time > 0
    
    def test_current_time_ms(self):
        # Get millisecond time
        time_ms = get_current_time_ms()
        
        # Should be close to current time
        current_ms = int(time.time() * 1000)
        assert abs(time_ms - current_ms) < 100  # Allow small difference
    
    @pytest.mark.asyncio
    async def test_sleep_with_jitter(self):
        start_time = time.time()
        await sleep_with_jitter(0.1, jitter_factor=0.5)
        duration = time.time() - start_time
        
        # Sleep should take around base time, but with jitter
        assert duration > 0.05  # Could be shorter with negative jitter
        assert duration < 0.2   # Could be longer with positive jitter
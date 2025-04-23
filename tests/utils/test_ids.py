import pytest
import uuid
from src.utils.ids import (
    generate_uuid,
    generate_short_uuid,
    generate_sequential_id,
    generate_snowflake_id,
    generate_prefixed_id,
    generate_task_id,
    generate_agent_id,
    generate_trace_id,
    generate_fingerprint,
    generate_short_fingerprint,
    is_valid_uuid,
    is_valid_snowflake,
    extract_timestamp_from_snowflake,
    generate_collision_resistant_id
)


class TestIdGeneration:
    """Test suite for ID generation utilities."""

    def test_generate_uuid(self):
        """Test generation of standard UUIDs."""
        # Generate a UUID and verify it's a valid UUID
        generated_id = generate_uuid()
        assert isinstance(generated_id, str)
        assert len(generated_id) == 36  # Standard UUID length
        assert is_valid_uuid(generated_id)

    def test_generate_short_uuid(self):
        """Test generation of shorter UUIDs."""
        # Generate a short UUID
        short_id = generate_short_uuid()
        assert isinstance(short_id, str)
        assert len(short_id) == 22  # Base64 encoding of 16 bytes without padding

    def test_generate_sequential_id(self):
        """Test generation of sequential IDs."""
        # Generate sequential IDs and verify they follow the expected pattern
        seq_id1 = generate_sequential_id()
        seq_id2 = generate_sequential_id()
        
        # IDs should be strings
        assert isinstance(seq_id1, str)
        assert isinstance(seq_id2, str)
        
        # Second ID should be different from first
        assert seq_id1 != seq_id2
        
        # Test with prefix
        prefixed_id = generate_sequential_id(prefix="test")
        assert prefixed_id.startswith("test-")

    def test_generate_snowflake_id(self):
        """Test generation of Snowflake-like IDs."""
        # Generate Snowflake IDs
        snowflake_id1 = generate_snowflake_id()
        snowflake_id2 = generate_snowflake_id()
        
        # IDs should be integers
        assert isinstance(snowflake_id1, int)
        assert isinstance(snowflake_id2, int)
        
        # Second ID should be different from first
        assert snowflake_id1 != snowflake_id2
        
        # Should be valid snowflake IDs
        assert is_valid_snowflake(snowflake_id1)
        assert is_valid_snowflake(snowflake_id2)

    def test_generate_prefixed_id(self):
        """Test generation of prefixed IDs."""
        # Generate prefixed ID
        prefixed_id = generate_prefixed_id("test", length=8)
        
        # ID should be a string and start with the prefix
        assert isinstance(prefixed_id, str)
        assert prefixed_id.startswith("test-")
        
        # Random part should have the specified length
        random_part = prefixed_id.split("-")[1]
        assert len(random_part) == 8
        
        # Test with different prefix and length
        custom_id = generate_prefixed_id("custom", length=12)
        assert custom_id.startswith("custom-")
        random_part = custom_id.split("-")[1]
        assert len(random_part) == 12
        
        # Empty prefix should raise ValueError
        with pytest.raises(ValueError):
            generate_prefixed_id("", length=8)

    def test_generate_task_id(self):
        """Test generation of task IDs."""
        # Generate task ID
        task_id = generate_task_id()
        
        # ID should be a string and start with 'task'
        assert isinstance(task_id, str)
        assert task_id.startswith("task-")
        
        # Test with task type
        typed_task_id = generate_task_id("planning")
        assert typed_task_id.startswith("task-planning-")
        
        # Test with task type containing spaces
        space_task_id = generate_task_id("complex planning")
        assert space_task_id.startswith("task-complex-planning-")

    def test_generate_agent_id(self):
        """Test generation of agent IDs."""
        # Generate agent ID
        agent_id = generate_agent_id("planner")
        
        # ID should be a string and have the expected format
        assert isinstance(agent_id, str)
        assert agent_id.startswith("agent-planner-")
        
        # Test with agent type containing spaces
        space_agent_id = generate_agent_id("executor agent")
        assert space_agent_id.startswith("agent-executor-agent-")

    def test_generate_trace_id(self):
        """Test generation of trace IDs."""
        # Generate trace ID
        trace_id = generate_trace_id()
        
        # ID should be a string and start with 'trace'
        assert isinstance(trace_id, str)
        assert trace_id.startswith("trace-")
        
        # Format should be trace-timestamp-node_id-random
        parts = trace_id.split("-")
        assert len(parts) == 4
        assert parts[0] == "trace"
        # parts[1] should be a timestamp (integer)
        assert parts[1].isdigit()

    def test_generate_fingerprint(self):
        """Test generation of deterministic fingerprints."""
        # Test string input
        content = "test content"
        fingerprint = generate_fingerprint(content)
        
        # Fingerprint should be a string of expected length (64 chars for SHA-256)
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64
        
        # Same content should generate same fingerprint
        assert generate_fingerprint(content) == fingerprint
        
        # Different content should generate different fingerprint
        assert generate_fingerprint("different content") != fingerprint
        
        # Test bytes input
        bytes_content = b"test content"
        bytes_fingerprint = generate_fingerprint(bytes_content)
        assert isinstance(bytes_fingerprint, str)
        assert len(bytes_fingerprint) == 64
        assert bytes_fingerprint == fingerprint  # Same underlying content

    def test_generate_short_fingerprint(self):
        """Test generation of shortened fingerprints."""
        content = "test content"
        
        # Test default length
        short_fingerprint = generate_short_fingerprint(content)
        assert isinstance(short_fingerprint, str)
        assert len(short_fingerprint) == 16
        
        # Test custom length
        custom_length = 8
        custom_fingerprint = generate_short_fingerprint(content, length=custom_length)
        assert len(custom_fingerprint) == custom_length
        
        # Should be prefix of full fingerprint
        full_fingerprint = generate_fingerprint(content)
        assert full_fingerprint.startswith(custom_fingerprint)
        
        # Test length clamping
        long_fingerprint = generate_short_fingerprint(content, length=100)
        assert len(long_fingerprint) == 64  # Max SHA-256 length
        assert long_fingerprint == full_fingerprint

    def test_is_valid_uuid(self):
        """Test UUID validation."""
        # Valid UUID should return True
        valid_uuid = str(uuid.uuid4())
        assert is_valid_uuid(valid_uuid) is True
        
        # Invalid UUIDs should return False
        assert is_valid_uuid("not-a-uuid") is False
        assert is_valid_uuid("123e4567-e89b-12d3-a456-42661417400") is False  # Too short
        assert is_valid_uuid("") is False
        assert is_valid_uuid(None) is False

    def test_is_valid_snowflake(self):
        """Test Snowflake ID validation."""
        # Valid Snowflake ID should return True
        valid_snowflake = generate_snowflake_id()
        assert is_valid_snowflake(valid_snowflake) is True
        
        # Invalid Snowflake IDs should return False
        assert is_valid_snowflake("not-an-int") is False
        assert is_valid_snowflake(-1) is False
        assert is_valid_snowflake(2**64) is False  # Too large
        assert is_valid_snowflake(None) is False

    def test_extract_timestamp_from_snowflake(self):
        """Test extraction of timestamp from Snowflake ID."""
        # Generate a Snowflake ID
        snowflake_id = generate_snowflake_id()
        
        # Extract timestamp
        timestamp = extract_timestamp_from_snowflake(snowflake_id)
        
        # Timestamp should be an integer
        assert isinstance(timestamp, int)
        
        # Should be a reasonable timestamp (within the last minute)
        import time
        current_time = int(time.time() * 1000)
        assert current_time - timestamp < 60000  # Within the last minute
        
        # Invalid Snowflake ID should raise ValueError
        with pytest.raises(ValueError):
            extract_timestamp_from_snowflake("not-a-snowflake")

    def test_generate_collision_resistant_id(self):
        """Test generation of collision-resistant IDs."""
        # Generate ID with prefix only
        prefix_id = generate_collision_resistant_id("test")
        assert isinstance(prefix_id, str)
        assert prefix_id.startswith("test-")
        
        # Parts should be: prefix-timestamp-content_hash-random
        parts = prefix_id.split("-")
        assert len(parts) == 4
        assert parts[0] == "test"
        assert parts[1].isdigit()  # timestamp
        assert len(parts[2]) == 8  # content hash (zeros if no content)
        
        # Generate ID with content
        content_id = generate_collision_resistant_id("test", "test content")
        assert content_id.startswith("test-")
        
        # Different content should produce different hash part
        assert content_id.split("-")[2] != "0" * 8
        assert content_id.split("-")[2] != prefix_id.split("-")[2]
        
        # Same content should produce same hash part
        duplicate_id = generate_collision_resistant_id("test", "test content")
        assert duplicate_id.split("-")[2] == content_id.split("-")[2]
        
        # Test with dictionary content
        dict_id = generate_collision_resistant_id("test", {"key": "value"})
        assert dict_id.startswith("test-")
        
        # Same dictionary content should produce same hash part
        duplicate_dict_id = generate_collision_resistant_id("test", {"key": "value"})
        assert duplicate_dict_id.split("-")[2] == dict_id.split("-")[2]
        
        # Different dictionary content should produce different hash part
        diff_dict_id = generate_collision_resistant_id("test", {"key": "different"})
        assert diff_dict_id.split("-")[2] != dict_id.split("-")[2]
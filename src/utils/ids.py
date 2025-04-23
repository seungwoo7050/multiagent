import hashlib
import os
import random
import string
import time
import uuid
from typing import Optional, Union

from src.config.logger import get_logger

# Module logger
logger = get_logger(__name__)

# Counter for sequential IDs
_id_counter = 0

# Node ID for distributed environments (randomly generated at module load time)
_node_id = ''.join(random.choices(string.hexdigits, k=6)).lower()

# Process ID for better uniqueness
_process_id = str(os.getpid() % 10000).zfill(4)


def generate_uuid() -> str:
    """Generate a standard UUID4.
    
    Returns:
        str: UUID4 string.
    """
    return str(uuid.uuid4())


def generate_short_uuid() -> str:
    """Generate a shorter UUID (22 chars) using base64 encoding.
    
    Returns:
        str: Short UUID string.
    """
    # Generate a UUID and convert to bytes
    uuid_bytes = uuid.uuid4().bytes
    
    # Use base64 encoding for more compact representation, remove padding
    import base64
    return base64.urlsafe_b64encode(uuid_bytes).decode('utf-8').rstrip('=')


def generate_sequential_id(prefix: str = "") -> str:
    """Generate a sequential ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID.
        
    Returns:
        str: Sequential ID.
    """
    global _id_counter
    _id_counter += 1
    
    timestamp = int(time.time() * 1000)
    
    # Combine timestamp, counter, and process ID for uniqueness
    if prefix:
        return f"{prefix}-{timestamp}-{_id_counter}"
    else:
        return f"{timestamp}-{_id_counter}"


def generate_snowflake_id() -> int:
    """Generate a Snowflake-like ID (64-bit integer).
    
    This combines timestamp, node ID, and sequence for distributed uniqueness.
    
    Returns:
        int: Snowflake-like ID.
    """
    global _id_counter
    _id_counter = (_id_counter + 1) & 0xFFF  # 12 bits for counter
    
    # Current timestamp (milliseconds since epoch) - 41 bits
    timestamp = int(time.time() * 1000)
    
    # Convert node_id hex string to int - 10 bits
    node_int = int(_node_id, 16) & 0x3FF
    
    # Combine: 41 bits timestamp + 10 bits node + 12 bits counter
    snowflake = (timestamp << 22) | (node_int << 12) | _id_counter
    
    return snowflake


def generate_prefixed_id(prefix: str, length: int = 16) -> str:
    """Generate a prefixed ID with specified length.
    
    Args:
        prefix: Prefix for the ID.
        length: Length of the random part.
        
    Returns:
        str: Prefixed ID.
    """
    if not prefix:
        raise ValueError("Prefix cannot be empty")
    
    # Generate random part (alphanumeric)
    random_part = ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=length)
    )
    
    return f"{prefix}-{random_part}"


def generate_task_id(task_type: Optional[str] = None) -> str:
    """Generate an ID for a task.
    
    Args:
        task_type: Optional task type to include in the ID.
        
    Returns:
        str: Task ID.
    """
    prefix = "task"
    if task_type:
        # Convert to lowercase and replace spaces with dashes
        task_type_clean = task_type.lower().replace(" ", "-")
        prefix = f"{prefix}-{task_type_clean}"
    
    timestamp = int(time.time())
    random_part = ''.join(random.choices(string.hexdigits, k=6)).lower()
    
    return f"{prefix}-{timestamp}-{random_part}"


def generate_agent_id(agent_type: str) -> str:
    """Generate an ID for an agent.
    
    Args:
        agent_type: Type of agent.
        
    Returns:
        str: Agent ID.
    """
    # Convert to lowercase and replace spaces with dashes
    agent_type_clean = agent_type.lower().replace(" ", "-")
    prefix = f"agent-{agent_type_clean}"
    
    timestamp = int(time.time())
    random_part = ''.join(random.choices(string.hexdigits, k=6)).lower()
    
    return f"{prefix}-{timestamp}-{random_part}"


def generate_trace_id() -> str:
    """Generate a trace ID for request tracing.
    
    Returns:
        str: Trace ID.
    """
    # Combine timestamp, node ID, and random string
    timestamp = int(time.time())
    random_part = ''.join(random.choices(string.hexdigits, k=8)).lower()
    
    return f"trace-{timestamp}-{_node_id}-{random_part}"


def generate_fingerprint(content: Union[str, bytes]) -> str:
    """Generate a deterministic fingerprint for content.
    
    This is useful for caching based on content.
    
    Args:
        content: Content to fingerprint.
        
    Returns:
        str: Hexadecimal fingerprint.
    """
    if isinstance(content, str):
        content_bytes = content.encode('utf-8')
    else:
        content_bytes = content
    
    return hashlib.sha256(content_bytes).hexdigest()


def generate_short_fingerprint(content: Union[str, bytes], length: int = 16) -> str:
    """Generate a shortened deterministic fingerprint for content.
    
    Args:
        content: Content to fingerprint.
        length: Length of the fingerprint (max 64).
        
    Returns:
        str: Shortened hexadecimal fingerprint.
    """
    if length > 64:
        length = 64  # SHA256 produces 64 hex chars
    
    full_fingerprint = generate_fingerprint(content)
    return full_fingerprint[:length]


def is_valid_uuid(id_string: Optional[str]) -> bool: # 타입 힌트를 Optional[str]로 변경하여 None도 받을 수 있음을 명시
    """Check if a string is a valid UUID.

    Args:
        id_string: String to check.

    Returns:
        bool: True if valid UUID, False otherwise.
    """
    # 먼저 id_string이 문자열인지 확인합니다.
    # None 이거나 다른 타입이면 False를 반환합니다.
    if not isinstance(id_string, str):
        return False

    try:
        # 문자열인 경우에만 UUID 생성을 시도합니다.
        uuid.UUID(id_string)
        return True
    except ValueError:
        # uuid.UUID()는 유효하지 않은 형식의 문자열에 대해 ValueError를 발생시킵니다.
        return False


def is_valid_snowflake(id_int: int) -> bool:
    """Check if an integer is a valid Snowflake ID.
    
    Args:
        id_int: Integer to check.
        
    Returns:
        bool: True if valid Snowflake ID, False otherwise.
    """
    # Basic validation: Snowflake IDs are positive 64-bit integers
    if not isinstance(id_int, int):
        return False
    
    return 0 <= id_int < (1 << 64)


def extract_timestamp_from_snowflake(snowflake_id: int) -> int:
    """Extract the timestamp from a Snowflake ID.
    
    Args:
        snowflake_id: Snowflake ID.
        
    Returns:
        int: Timestamp in milliseconds.
    """
    if not is_valid_snowflake(snowflake_id):
        raise ValueError("Invalid Snowflake ID")
    
    # Extract timestamp (first 41 bits)
    return snowflake_id >> 22


def generate_collision_resistant_id(
    prefix: str,
    content: Optional[Union[str, dict]] = None,
    entropy_bits: int = 32
) -> str:
    """Generate a collision-resistant ID.
    
    This combines a prefix, content hash (if provided), and random bits.
    
    Args:
        prefix: ID prefix.
        content: Optional content to hash for deterministic part.
        entropy_bits: Number of random bits to add (multiple of 8).
        
    Returns:
        str: Collision-resistant ID.
    """
    # Ensure entropy_bits is a multiple of 8
    entropy_bytes = (entropy_bits + 7) // 8
    
    # Generate timestamp part
    timestamp = int(time.time())
    
    # Hash content if provided
    if content is not None:
        if isinstance(content, dict):
            # Sort keys for deterministic serialization
            import json
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        # Generate short hash
        content_hash = generate_short_fingerprint(content_str, 8)
    else:
        content_hash = "0" * 8
    
    # Generate random part
    random_part = ''.join(random.choices(string.hexdigits, k=entropy_bytes * 2)).lower()
    
    # Combine all parts
    return f"{prefix}-{timestamp}-{content_hash}-{random_part}"
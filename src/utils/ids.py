import hashlib
import os
import random
import string
import threading
import time
import uuid
from typing import Optional, Union

from src.config.logger import get_logger

logger = get_logger(__name__)

# Thread-safe counter
class _Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self, max_value=None):
        with self.lock:
            self.value += 1
            if max_value and self.value >= max_value:
                self.value = 0
            return self.value

# Initialize global variables
_id_counter = _Counter()
_node_id = ''.join(random.choices(string.hexdigits, k=6)).lower()
_process_id = str(os.getpid() % 10000).zfill(4)

def generate_uuid() -> str:
    return str(uuid.uuid4())

def generate_short_uuid() -> str:
    uuid_bytes = uuid.uuid4().bytes
    import base64
    return base64.urlsafe_b64encode(uuid_bytes).decode('utf-8').rstrip('=')

def generate_sequential_id(prefix: str='') -> str:
    counter_value = _id_counter.increment()
    timestamp = int(time.time() * 1000)
    if prefix:
        return f'{prefix}-{timestamp}-{counter_value}'
    else:
        return f'{timestamp}-{counter_value}'

def generate_snowflake_id() -> int:
    counter_value = _id_counter.increment(4095)  # Reset at 4095 (12 bits max)
    timestamp = int(time.time() * 1000)
    node_int = int(_node_id, 16) & 1023
    snowflake = (timestamp << 22) | (node_int << 12) | counter_value
    return snowflake

def generate_prefixed_id(prefix: str, length: int=16) -> str:
    if not prefix:
        raise ValueError('Prefix cannot be empty for generate_prefixed_id')
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f'{prefix}-{random_part}'

def generate_task_id(task_type: Optional[str]=None) -> str:
    prefix = 'task'
    if task_type:
        task_type_clean = task_type.lower().replace(' ', '-')
        prefix = f'{prefix}-{task_type_clean}'
    timestamp = int(time.time())
    random_part = ''.join(random.choices(string.hexdigits, k=6)).lower()
    return f'{prefix}-{timestamp}-{random_part}'

def generate_agent_id(agent_type: str) -> str:
    agent_type_clean = agent_type.lower().replace(' ', '-')
    prefix = f'agent-{agent_type_clean}'
    timestamp = int(time.time())
    random_part = ''.join(random.choices(string.hexdigits, k=6)).lower()
    return f'{prefix}-{timestamp}-{random_part}'

def generate_trace_id() -> str:
    timestamp = int(time.time())
    random_part = ''.join(random.choices(string.hexdigits, k=8)).lower()
    return f'trace-{timestamp}-{_node_id}-{random_part}'

def generate_fingerprint(content: Union[str, bytes]) -> str:
    if isinstance(content, str):
        content_bytes = content.encode('utf-8')
    else:
        content_bytes = content
    return hashlib.sha256(content_bytes).hexdigest()

def generate_short_fingerprint(content: Union[str, bytes], length: int=16) -> str:
    if length > 64:
        length = 64
    full_fingerprint = generate_fingerprint(content)
    return full_fingerprint[:length]

def is_valid_uuid(id_string: Optional[str]) -> bool:
    if not isinstance(id_string, str):
        return False
    try:
        uuid.UUID(id_string)
        return True
    except ValueError:
        return False
    except Exception:
        return False

def is_valid_snowflake(id_int: int) -> bool:
    if not isinstance(id_int, int):
        return False
    return 0 <= id_int < 1 << 64

def extract_timestamp_from_snowflake(snowflake_id: int) -> int:
    if not is_valid_snowflake(snowflake_id):
        raise ValueError('Invalid Snowflake ID provided for timestamp extraction')
    return snowflake_id >> 22

def generate_collision_resistant_id(prefix: str, content: Optional[Union[str, dict]]=None, entropy_bits: int=32) -> str:
    entropy_bytes = (entropy_bits + 7) // 8
    timestamp = int(time.time())
    if content is not None:
        if isinstance(content, dict):
            import json
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        content_hash = generate_short_fingerprint(content_str, 8)
    else:
        content_hash = '0' * 8
    random_part = ''.join(random.choices(string.hexdigits, k=entropy_bytes * 2)).lower()
    return f'{prefix}-{timestamp}-{content_hash}-{random_part}'
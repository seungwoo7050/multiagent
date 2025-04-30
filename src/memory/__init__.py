"""
Memory package for the Multi-Agent Platform.

This package provides memory storage and vector embedding functionality.
"""
from src.memory.base import BaseMemory, BaseVectorStore
from src.memory.manager import MemoryManager
from src.memory.redis_memory import RedisMemory
from src.memory.vector_store import VectorStore

# Initialize backends if available
try:
    from src.memory.backends import register_backends
    register_backends(VectorStore)
except ImportError:
    # Backends package not available, use default implementations
    pass

__all__ = [
    'BaseMemory',
    'BaseVectorStore',
    'MemoryManager',
    'RedisMemory',
    'VectorStore',
]
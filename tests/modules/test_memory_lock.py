"""
Test for the improved memory manager lock cleanup implementation.
Verifies that lock cleanup happens deterministically based on time rather than random chance.
"""
import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock

from src.memory.manager import MemoryManager
from src.memory.base import BaseMemory

@pytest.mark.asyncio
class TestMemoryManagerLocks:
    """Test memory manager lock management."""
    
    async def test_lock_cleanup_timing(self):
        """Test that lock cleanup happens based on time intervals."""
        # Create mock primary memory
        mock_primary = AsyncMock(spec=BaseMemory)
        
        # Create memory manager with small cache
        memory_manager = MemoryManager(
            primary_memory=mock_primary,
            cache_size=100,
            cache_ttl=60
        )
        
        # Verify initial state
        assert len(memory_manager._cache_locks) == 0
        
        # Set the last cleanup time to a known value
        memory_manager._last_locks_cleanup = time.monotonic() - 70  # 70 seconds ago, should trigger cleanup
        
        # Create a large number of locks
        test_keys = [f"test_key_{i}" for i in range(200)]
        created_locks = []
        
        # Request 150 locks - all should be created
        for i in range(150):
            lock = await memory_manager._get_cache_lock(test_keys[i])
            created_locks.append(lock)
            
        # Verify locks were created
        assert len(memory_manager._cache_locks) > 0
        
        # Artificially mark some locks as "in use" by acquiring them
        active_locks = created_locks[:30]  # First 30 locks will be active
        for lock in active_locks:
            # Acquire without waiting to simulate active locks
            acquire_success = lock.locked() or lock._locked
            
        # Request another lock, which should trigger cleanup
        with patch.object(memory_manager, '_locks_lock', new_callable=AsyncMock) as mock_locks_lock:
            # Mock the lock context manager
            mock_locks_lock.__aenter__.return_value = None
            mock_locks_lock.__aexit__.return_value = None
            
            # Get a new lock which should trigger cleanup
            new_lock = await memory_manager._get_cache_lock(test_keys[150])
            
            # Verify lock was created
            assert test_keys[150] in memory_manager._cache_locks
            
            # Record the new last_locks_cleanup time
            new_cleanup_time = memory_manager._last_locks_cleanup
            
        # Verify cleanup happened
        assert new_cleanup_time > time.monotonic() - 5  # Should be recent
        
        # Verify that inactive locks were removed while active ones remain
        total_remaining = len(memory_manager._cache_locks)
        assert total_remaining < 150, "Some locks should have been cleaned up"
        
        # Check if lock cleanup works as expected when time hasn't elapsed
        memory_manager._last_locks_cleanup = time.monotonic()  # Just cleaned up
        
        # Add more locks
        for i in range(151, 160):
            lock = await memory_manager._get_cache_lock(test_keys[i])
            
        # The cleanup shouldn't happen again because it's too soon
        assert len(memory_manager._cache_locks) > total_remaining
    
    async def test_concurrent_lock_requests(self):
        """Test that concurrent lock requests work correctly."""
        # Create mock primary memory
        mock_primary = AsyncMock(spec=BaseMemory)
        
        # Create memory manager
        memory_manager = MemoryManager(
            primary_memory=mock_primary,
            cache_size=100,
            cache_ttl=60
        )
        
        # Create a shared counter to track concurrent access
        counter = 0
        
        async def increment_with_lock(key: str, delay: float):
            nonlocal counter
            # Get lock for the key
            lock = await memory_manager._get_cache_lock(key)
            async with lock:
                # Read the current value
                current = counter
                # Simulate some work
                await asyncio.sleep(delay)
                # Increment
                counter = current + 1
                # Return the value that was used for increment
                return current
        
        # Run concurrent tasks using the same lock
        tasks = []
        for i in range(5):
            tasks.append(asyncio.create_task(
                increment_with_lock("same_key", 0.05)
            ))
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks)
        
        # If locks work correctly, each task should see a different counter value
        assert len(set(results)) == 5, "Each task should see a unique counter value"
        assert counter == 5, "Final counter should equal number of tasks"
        
        # Verify that only one lock was created
        assert len(memory_manager._cache_locks) == 1
        assert "same_key" in memory_manager._cache_locks
    
    async def test_load_with_locking(self):
        """Test that load operation uses locks correctly to prevent cache stampede."""
        # Create mock primary memory that simulates slow response
        mock_primary = AsyncMock(spec=BaseMemory)
        
        async def slow_load_context(key, context_id, default=None):
            await asyncio.sleep(0.1)  # Simulate slow DB access
            return f"value_for_{key}"
        
        mock_primary.load_context.side_effect = slow_load_context
        
        # Create memory manager
        memory_manager = MemoryManager(
            primary_memory=mock_primary,
            cache_size=100,
            cache_ttl=60
        )
        
        # Run concurrent load operations for the same key
        async def load_key():
            return await memory_manager.load("test_key", "context1")
        
        # Run multiple concurrent loads
        tasks = []
        for _ in range(5):
            tasks.append(asyncio.create_task(load_key()))
        
        # Wait for all tasks
        results = await asyncio.gather(*tasks)
        
        # All results should be the same
        assert len(set(results)) == 1, "All concurrent loads should return the same value"
        
        # The primary storage should only be called once if locking works
        mock_primary.load_context.assert_called_once_with("test_key", "context1", None)
        
        # Verify lock cleanup doesn't affect locks that are still in use
        assert "memory:context1:test_key" in memory_manager._cache_locks
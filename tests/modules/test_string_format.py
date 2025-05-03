"""
Test for the string formatting fixes in the memory package.
Verifies that f-strings with embedded quotes are correctly formatted.
"""
from unittest.mock import patch, AsyncMock

from src.memory.redis_memory import RedisMemory
from src.memory.manager import MemoryManager

class TestStringFormatting:
    """Test string formatting in memory components."""
    
    def test_redis_memory_string_formatting(self):
        """Test that RedisMemory class formats strings correctly."""
        # Create instance with mock Redis
        with patch('src.memory.redis_memory.conn_manager') as mock_conn_manager:
            redis_memory = RedisMemory()
            
            # Test the clear method implementation which had formatting issues
            with patch.object(redis_memory, '_get_redis') as mock_get_redis:
                mock_redis = AsyncMock()
                mock_redis.scan.return_value = (b'0', [])
                mock_get_redis.return_value = mock_redis
                
                # Capture log messages
                with patch('src.memory.redis_memory.logger') as mock_logger:
                    # Call the method that previously had formatting issues
                    context_id = None
                    asyncio_run(redis_memory.clear(context_id))
                    
                    # Verify formatted log message - context is None
                    mock_logger.info.assert_any_call('Clearing Redis memory for all contexts...')
                    
                    # Test with specific context
                    context_id = "test_context"
                    asyncio_run(redis_memory.clear(context_id))
                    
                    # Verify log message with context
                    format_message = f"Clearing Redis memory for context '{context_id}'..."
                    mock_logger.info.assert_any_call(format_message)
    
    def test_memory_manager_string_formatting(self):
        """Test that MemoryManager class formats strings correctly."""
        # Create memory manager with mock components
        mock_primary = AsyncMock()
        memory_manager = MemoryManager(primary_memory=mock_primary)
        
        # Patch the logger to capture messages
        with patch('src.memory.manager.logger') as mock_logger:
            # Test list_keys method that had formatting issues
            asyncio_run(memory_manager.list_keys(None, None))
            
            # Verify formatted log message with all contexts
            mock_logger.debug.assert_any_call('Listing keys from primary storage (context: all, pattern: none)')
            
            # Test with specific context and pattern
            context_id = "test_context"
            pattern = "user*"
            asyncio_run(memory_manager.list_keys(context_id, pattern))
            
            # Verify formatted log message with specific context and pattern
            mock_logger.debug.assert_any_call(f'Listing keys from primary storage (context: {context_id}, pattern: {pattern})')
            
            # Test with None context but specific pattern
            asyncio_run(memory_manager.list_keys(None, pattern))
            mock_logger.debug.assert_any_call(f'Listing keys from primary storage (context: all, pattern: {pattern})')
            
            # Test with specific context but None pattern
            asyncio_run(memory_manager.list_keys(context_id, None))
            mock_logger.debug.assert_any_call(f'Listing keys from primary storage (context: {context_id}, pattern: none)')

def asyncio_run(coroutine):
    """Helper to run coroutines synchronously in a test."""
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)
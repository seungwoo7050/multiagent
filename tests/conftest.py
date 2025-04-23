import os
import sys
import pytest
import logging
from typing import Any, Dict

# Add the src directory to path so that imports work correctly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Disable verbose logging during tests
logging.basicConfig(level=logging.ERROR)

# Reset all global state between tests
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state between tests to prevent cross-test contamination."""
    from src.core.registry import clear_all_registries
    from src.core.worker_pool import shutdown_all_worker_pools
    import asyncio
    
    # Before test
    yield
    
    # After test cleanup
    clear_all_registries()
    
    # Create event loop to execute async cleanup
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # Create new event loop if the current one is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run worker pool cleanup
        if not loop.is_closed():
            loop.run_until_complete(shutdown_all_worker_pools())
    except Exception as e:
        # Log but don't fail tests on cleanup errors
        logging.error(f"Error during cleanup: {e}")
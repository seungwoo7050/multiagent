"""Unit tests for the web search tool."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from src.config.errors import ToolError
from src.tools.web_search import WebSearchTool, WebSearchInput


@pytest.fixture
def web_search():
    """Create a web search tool for tests."""
    return WebSearchTool()


# Mock the HTTP session to avoid the settings dependency
@pytest.fixture(autouse=True)
def mock_http_session():
    """Mock the HTTP session for all tests."""
    with patch('src.tools.web_search.get_http_session') as mock:
        mock_session = AsyncMock()
        mock.return_value = mock_session
        yield mock_session


# Mock Redis to avoid external dependencies
@pytest.fixture(autouse=True)
def mock_redis():
    """Mock Redis connection for all tests."""
    with patch('src.tools.web_search.get_redis_async_connection') as mock:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Default to cache miss
        mock_redis.setex.return_value = True
        mock.return_value = mock_redis
        yield mock_redis


# Basic tests remain the same as before...


@pytest.mark.asyncio
async def test_web_search_execution(web_search):
    """Test the web search execution."""
    # Override _perform_search to avoid using HTTP session
    with patch.object(web_search, '_perform_search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            {"title": "Test Result", "snippet": "This is a test", "url": "https://example.com", "position": 1}
        ]
        
        result = await web_search.arun(query="test query")
        
        assert result["query"] == "test query"
        assert len(result["results"]) == 1
        assert mock_search.called
        assert mock_search.call_args[0][0] == "test query"  # First arg is query


def test_web_search_initialization(web_search):
    """Test web search tool initialization."""
    assert web_search.name == "web_search"
    assert "search" in web_search.description.lower()
    assert web_search.args_schema == WebSearchInput


def test_web_search_input_validation():
    """Test web search input validation."""
    # Valid input
    valid_input = WebSearchInput(query="test query")
    assert valid_input.query == "test query"
    assert valid_input.num_results == 5  # Default value
    
    # Custom num_results
    valid_input_with_num = WebSearchInput(query="test query", num_results=3)
    assert valid_input_with_num.num_results == 3
    
    # Empty query
    with pytest.raises(ValidationError):
        WebSearchInput(query="")
    
    # Too long query
    with pytest.raises(ValidationError):
        WebSearchInput(query="test query" * 500)  # Over 1000 chars
    
    # Invalid num_results (too small)
    with pytest.raises(ValidationError):
        WebSearchInput(query="test query", num_results=0)
    
    # Invalid num_results (too large)
    with pytest.raises(ValidationError):
        WebSearchInput(query="test query", num_results=20)


@pytest.mark.asyncio
async def test_web_search_execution(web_search):
    """Test the web search execution."""
    # We're not actually calling any external service
    result = await web_search.arun(query="test query")
    
    assert "query" in result
    assert result["query"] == "test query"
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0
    
    # Check result structure
    first_result = result["results"][0]
    assert "title" in first_result
    assert "snippet" in first_result
    assert "url" in first_result
    assert "position" in first_result


@pytest.mark.asyncio
async def test_web_search_num_results(web_search):
    """Test that num_results parameter is respected."""
    result = await web_search.arun(query="test query", num_results=3)
    
    assert len(result["results"]) == 3
    
    result = await web_search.arun(query="test query", num_results=7)
    
    assert len(result["results"]) == 7


@pytest.mark.asyncio
async def test_web_search_cache_hit():
    """Test cache hit scenario."""
    web_search = WebSearchTool()
    
    # Mock cache hit
    web_search._get_from_cache = AsyncMock(return_value={
        "query": "cached query",
        "results": [{"title": "Cached Result", "snippet": "This is from cache", "url": "https://example.com/cached", "position": 1}]
    })
    
    result = await web_search.arun(query="cached query")
    
    assert result["query"] == "cached query"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Cached Result"
    assert web_search._get_from_cache.called
    
    # The _perform_search method shouldn't be called on cache hit
    web_search._perform_search = AsyncMock()
    result = await web_search.arun(query="cached query")
    assert not web_search._perform_search.called


@pytest.mark.asyncio
async def test_web_search_cache_miss():
    """Test cache miss scenario."""
    web_search = WebSearchTool()
    
    # Mock cache miss
    web_search._get_from_cache = AsyncMock(return_value=None)
    web_search._save_to_cache = AsyncMock(return_value=True)
    
    # We still want the real search functionality
    original_perform_search = web_search._perform_search
    web_search._perform_search = AsyncMock(side_effect=original_perform_search)
    
    result = await web_search.arun(query="new query")
    
    assert result["query"] == "new query"
    assert web_search._get_from_cache.called
    assert web_search._perform_search.called
    assert web_search._save_to_cache.called


@pytest.mark.asyncio
async def test_web_search_error_handling():
    """Test error handling in the web search tool."""
    web_search = WebSearchTool()
    
    # Simulate a search error
    web_search._perform_search = AsyncMock(side_effect=Exception("Test search error"))
    
    with pytest.raises(ToolError) as excinfo:
        await web_search.arun(query="error query")
    
    assert "Web search failed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_web_search_cache_key_generation(web_search):
    """Test that cache keys are deterministic and unique."""
    # Same query and num_results should generate the same key
    key1 = web_search._get_cache_key("test query", 5)
    key2 = web_search._get_cache_key("test query", 5)
    assert key1 == key2
    
    # Different queries should generate different keys
    key3 = web_search._get_cache_key("different query", 5)
    assert key1 != key3
    
    # Different num_results should generate different keys
    key4 = web_search._get_cache_key("test query", 3)
    assert key1 != key4
    
    # Case insensitive and whitespace normalization
    key5 = web_search._get_cache_key("  TEST QUERY  ", 5)
    assert key1 == key5


@pytest.mark.asyncio
async def test_web_search_format_results(web_search):
    """Test that results are properly formatted."""
    # Create some mock results
    raw_results = [
        {"title": "Result 1", "snippet": "Snippet 1", "url": "https://example.com/1", "position": 1, "extra_field": "should be removed"},
        {"title": "Result 2", "snippet": "Snippet 2", "url": "https://example.com/2", "position": 2, "another_field": "also removed"},
    ]
    
    formatted = web_search._format_results(raw_results, "test query")
    
    assert formatted["query"] == "test query"
    assert formatted["total_results"] == 2
    assert len(formatted["results"]) == 2
    
    # Check that only the expected fields are included
    for result in formatted["results"]:
        assert "title" in result
        assert "snippet" in result
        assert "url" in result
        assert "position" in result
        assert "extra_field" not in result
        assert "another_field" not in result


@pytest.mark.asyncio
async def test_web_search_mock_results(web_search):
    """Test the mock results generation."""
    mock_results = web_search._generate_mock_results("test query", 3)
    
    assert len(mock_results) == 3
    
    # Check that the query terms are included in the results
    for result in mock_results:
        assert "test" in result["snippet"].lower()
        assert "query" in result["snippet"].lower()
        assert "test" in result["url"].lower()
        assert "query" in result["url"].lower()


@pytest.mark.asyncio
async def test_web_search_sync_execution(web_search):
    """Test the synchronous execution path."""
    # The sync version should create an event loop and run the async version
    with patch('asyncio.new_event_loop') as mock_loop:
        # Create a mock event loop
        mock_event_loop = MagicMock()
        mock_loop.return_value = mock_event_loop
        
        # Mock the async result that would be returned by run_until_complete
        mock_event_loop.run_until_complete.return_value = {
            "query": "sync test",
            "results": [{"title": "Sync Result", "snippet": "Test", "url": "https://example.com", "position": 1}]
        }
        
        # Call the synchronous method
        result = web_search.run(query="sync test")
        
        # Verify the loop was used correctly
        assert mock_loop.called
        assert mock_event_loop.run_until_complete.called
        assert mock_event_loop.close.called
        
        # Check that we got the expected result
        assert result["query"] == "sync test"


@pytest.mark.asyncio
async def test_web_search_redis_interaction():
    """Test the interaction with Redis."""
    web_search = WebSearchTool()
    
    # Create mock Redis connection
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None  # Cache miss
    mock_redis.setex.return_value = True  # Successful cache set
    
    # Mock the connection function
    with patch('src.tools.web_search.get_redis_async_connection', return_value=mock_redis):
        # First call should miss the cache and save results
        result = await web_search.arun(query="redis test")
        
        # Verify Redis operations
        assert mock_redis.get.called
        assert mock_redis.setex.called
        
        # Set up for cache hit
        cache_data = json.dumps(result)
        mock_redis.get.return_value = cache_data
        
        # Second call should hit the cache
        result2 = await web_search.arun(query="redis test")
        
        # Should be the same result
        assert result == result2
        
        # Redis get should be called twice, but setex only once
        assert mock_redis.get.call_count == 2
        assert mock_redis.setex.call_count == 1
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import json

from src.tools.web_search import WebSearchTool, WebSearchInput
from src.tools.web_search_google import GoogleSearchTool
from src.config.errors import ToolError


class TestWebSearchTool(unittest.TestCase):
    """Tests for the WebSearchTool (DuckDuckGo) implementation"""
    
    def setUp(self):
        self.web_search = WebSearchTool()
        # Mock the settings for testing
        self.web_search.api_url = "https://api.duckduckgo.com/"
        self.web_search.cache_ttl = 3600
    
    @patch("aiohttp.ClientSession.get")
    @patch("src.tools.web_search.get_redis_async_connection")
    async def test_duckduckgo_search(self, mock_redis, mock_get):
        """Test the DuckDuckGo search functionality"""
        # Mock the Redis connection
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None  # No cache hit
        mock_redis_instance.setex.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.raise_for_status = AsyncMock()
        mock_response.json.return_value = {
            "AbstractSource": "Wikipedia",
            "Abstract": "Python is a high-level programming language.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "Heading": "Python (programming language)",
            "RelatedTopics": [
                {
                    "FirstURL": "https://duckduckgo.com/Python_Software_Foundation",
                    "Text": "Python Software Foundation - The organization that owns Python."
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Run the search
        result = await self.web_search._arun(query="python programming", num_results=2)
        
        # Verify the results
        self.assertEqual(result["query"], "python programming")
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["search_engine"], "duckduckgo")
        
        # Verify the first result is from the Abstract
        self.assertEqual(result["results"][0]["title"], "Python (programming language)")
        self.assertEqual(result["results"][0]["snippet"], "Python is a high-level programming language.")
        
        # Check that the API was called with correct parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args[1]["params"]
        self.assertEqual(call_args["q"], "python programming")
        self.assertEqual(call_args["format"], "json")
        
        # Verify cache was checked and set
        mock_redis_instance.get.assert_called_once()
        mock_redis_instance.setex.assert_called_once()
    
    @patch("aiohttp.ClientSession.get")
    @patch("src.tools.web_search.get_redis_async_connection")
    async def test_duckduckgo_search_no_results(self, mock_redis, mock_get):
        """Test the fallback when DuckDuckGo returns no results"""
        # Mock the Redis connection
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None  # No cache hit
        mock_redis_instance.setex.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        # Mock the HTTP response with no results
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.raise_for_status = AsyncMock()
        mock_response.json.return_value = {
            "Abstract": "",
            "RelatedTopics": []
        }
        mock_get.return_value = mock_response
        
        # Run the search
        result = await self.web_search._arun(query="very obscure query", num_results=3)
        
        # Verify fallback results were generated
        self.assertEqual(result["query"], "very obscure query")
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["search_engine"], "duckduckgo")
        
        # Check fallback source
        for item in result["results"]:
            self.assertEqual(item["source"], "fallback")
    
    @patch("aiohttp.ClientSession.get")
    async def test_duckduckgo_search_api_error(self, mock_get):
        """Test error handling when the DuckDuckGo API fails"""
        # Mock the HTTP request to raise an exception
        mock_get.side_effect = Exception("API connection error")
        
        # Try to run the search and expect an error
        with self.assertRaises(ToolError) as context:
            await self.web_search._arun(query="python", num_results=2)
        
        # Verify the error details
        self.assertEqual(context.exception.code, "TOOL_EXECUTION_ERROR")
        self.assertIn("Web search execution failed", context.exception.message)


class TestGoogleSearchTool(unittest.TestCase):
    """Tests for the GoogleSearchTool implementation"""
    
    def setUp(self):
        self.google_search = GoogleSearchTool()
        # Set up test API credentials
        self.google_search.api_key = "test_api_key"
        self.google_search.search_engine_id = "test_search_engine_id"
        self.google_search.cache_ttl = 3600
    
    @patch("aiohttp.ClientSession.get")
    @patch("src.tools.web_search_google.get_redis_async_connection")
    async def test_google_search(self, mock_redis, mock_get):
        """Test the Google search functionality"""
        # Mock the Redis connection
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None  # No cache hit
        mock_redis_instance.setex.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.__aenter__.return_value = mock_response
        mock_response.raise_for_status = AsyncMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Python Programming Language",
                    "snippet": "Python is a programming language that lets you work quickly and integrate systems more effectively.",
                    "link": "https://www.python.org/",
                    "formattedUrl": "www.python.org",
                    "pagemap": {
                        "metatags": [
                            {
                                "og:type": "website"
                            }
                        ]
                    }
                },
                {
                    "title": "Python Tutorial",
                    "snippet": "Learn Python programming with tutorials and examples.",
                    "link": "https://www.w3schools.com/python/"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Run the search
        result = await self.google_search._arun(query="python programming", num_results=2)
        
        # Verify the results
        self.assertEqual(result["query"], "python programming")
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["search_engine"], "google")
        
        # Verify the first result
        self.assertEqual(result["results"][0]["title"], "Python Programming Language")
        self.assertIn("programming language", result["results"][0]["snippet"])
        self.assertEqual(result["results"][0]["url"], "www.python.org")
        
        # Check that the API was called with correct parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args[1]["params"]
        self.assertEqual(call_args["q"], "python programming")
        self.assertEqual(call_args["key"], "test_api_key")
        self.assertEqual(call_args["cx"], "test_search_engine_id")
        
        # Verify cache was checked and set
        mock_redis_instance.get.assert_called_once()
        mock_redis_instance.setex.assert_called_once()
    
    async def test_google_search_missing_credentials(self):
        """Test error handling when Google API credentials are missing"""
        # Clear API credentials
        self.google_search.api_key = ""
        self.google_search.search_engine_id = ""
        
        # Try to run the search and expect a validation error
        with self.assertRaises(ToolError) as context:
            await self.google_search._arun(query="python", num_results=2)
        
        # Verify the error details
        self.assertEqual(context.exception.code, "TOOL_VALIDATION_ERROR")
        self.assertIn("API key and Search Engine ID must be configured", context.exception.message)
    
    @patch("aiohttp.ClientSession.get")
    async def test_google_search_api_error(self, mock_get):
        """Test error handling when the Google API fails"""
        # Mock the HTTP request to raise an exception
        mock_get.side_effect = Exception("API connection error")
        
        # Try to run the search and expect an error
        with self.assertRaises(ToolError) as context:
            await self.google_search._arun(query="python", num_results=2)
        
        # Verify the error details
        self.assertEqual(context.exception.code, "TOOL_EXECUTION_ERROR")
        self.assertIn("Google search failed", context.exception.message)


# Helper function to run async tests
def async_test(coro):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


if __name__ == "__main__":
    # Patch the actual test methods to use our async_test wrapper
    TestWebSearchTool.test_duckduckgo_search = async_test(TestWebSearchTool.test_duckduckgo_search)
    TestWebSearchTool.test_duckduckgo_search_no_results = async_test(TestWebSearchTool.test_duckduckgo_search_no_results)
    TestWebSearchTool.test_duckduckgo_search_api_error = async_test(TestWebSearchTool.test_duckduckgo_search_api_error)
    TestGoogleSearchTool.test_google_search = async_test(TestGoogleSearchTool.test_google_search)
    TestGoogleSearchTool.test_google_search_missing_credentials = async_test(TestGoogleSearchTool.test_google_search_missing_credentials)
    TestGoogleSearchTool.test_google_search_api_error = async_test(TestGoogleSearchTool.test_google_search_api_error)
    
    unittest.main()
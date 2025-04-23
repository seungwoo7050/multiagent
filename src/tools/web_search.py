"""
Web Search Tool - High-Performance Implementation.

This module provides an asynchronous web search tool that can retrieve
information from search engines with proper caching and error handling.
"""

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator

from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.config.metrics import CACHE_HITS_TOTAL, CACHE_MISSES_TOTAL
from src.config.connections import get_http_session, get_redis_async_connection
from src.tools.base import BaseTool
from src.tools.registry import register_tool
from src.utils.timing import AsyncTimer

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""
    
    query: str = Field(
        ...,
        description="The search query to execute"
    )
    
    num_results: int = Field(
        5,
        description="Number of search results to return",
        ge=1,
        le=10
    )
    
    @validator("query")
    def validate_query(cls, v: str) -> str:
        """Validate the search query."""
        if not v or not isinstance(v, str):
            raise ValueError("Query must be a non-empty string")
        
        # Basic validation
        if len(v) > 1000:
            raise ValueError("Query is too long (max 1000 chars)")
        
        v = v.strip()
        
        return v


@register_tool()
class WebSearchTool(BaseTool):
    """
    A tool for performing web searches.
    
    This tool implements an asynchronous web search that retrieves
    information from search engines with proper caching and error handling.
    """
    
    def __init__(self):
        """Initialize the web search tool."""
        self.cache_ttl = 3600  # 1 hour cache TTL
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information on a given query. Returns top search results."
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return WebSearchInput
    
    def _run(self, **kwargs: Any) -> Any:
        """
        Execute the web search synchronously.
        
        Note: This will run the async version in a new event loop.
        """
        # Create a new event loop for the async operation
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._arun(**kwargs))
        finally:
            loop.close()
    
    async def _arun(self, **kwargs: Any) -> Any:
        """
        Execute the web search asynchronously.
        
        Args:
            query: The search query to execute.
            num_results: Number of results to return.
            
        Returns:
            A dictionary with search results.
            
        Raises:
            ToolError: If the search fails.
        """
        query = kwargs["query"]
        num_results = kwargs.get("num_results", 5)
        
        logger.info(f"Performing web search for: {query}")
        
        # Check cache first
        cache_key = self._get_cache_key(query, num_results)
        cached_result = await self._get_from_cache(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for query: {query}")
            CACHE_HITS_TOTAL.labels(cache_type="web_search").inc()
            return cached_result
        
        logger.info(f"Cache miss for query: {query}")
        CACHE_MISSES_TOTAL.labels(cache_type="web_search").inc()
        
        try:
            # Perform the search
            async with AsyncTimer("web_search_execution"):
                results = await self._perform_search(query, num_results)
            
            # Format the results
            search_results = self._format_results(results, query)
            
            # Cache the results
            await self._save_to_cache(cache_key, search_results)
            
            return search_results
            
        except Exception as e:
            logger.error(
                f"Web search failed: {str(e)}",
                extra={"query": query, "error": str(e)},
                exc_info=e
            )
            
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Web search failed: {str(e)}",
                details={"query": query, "error": str(e)},
                original_error=e,
                tool_name=self.name
            )
    
    async def _perform_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Perform the actual search using an external service.
        
        Args:
            query: The search query.
            num_results: Number of results to retrieve.
            
        Returns:
            A list of search result dictionaries.
            
        Raises:
            ToolError: If the search request fails.
        """
        # In a real implementation, this would call a search API
        # For this example, we'll simulate a search response
        
        # Example search API call using aiohttp
        try:
            session = await get_http_session()
            
            # Replace with actual search API endpoint
            search_url = "https://api.search.example.com/search"
            
            # Prepare query parameters
            params = {
                "q": query,
                "num": num_results,
                "format": "json"
            }
            
            # Simulated API call - in a real implementation this would be uncommented
            # async with session.get(search_url, params=params) as response:
            #     if response.status != 200:
            #         raise ToolError(
            #             code=ErrorCode.CONNECTION_ERROR,
            #             message=f"Search API returned status {response.status}",
            #             details={"status": response.status, "query": query}
            #         )
            #     
            #     result_data = await response.json()
            #     return result_data.get("results", [])
            
            # For demonstration, simulate a delay and return mock results
            await asyncio.sleep(0.2)  # Simulate network delay
            
            # Mock results based on query
            return self._generate_mock_results(query, num_results)
            
        except Exception as e:
            logger.error(
                f"Search API request failed: {str(e)}",
                extra={"query": query},
                exc_info=e
            )
            
            raise ToolError(
                code=ErrorCode.CONNECTION_ERROR,
                message=f"Failed to connect to search service: {str(e)}",
                details={"query": query},
                original_error=e,
                tool_name=self.name
            )
    
    def _generate_mock_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Generate mock search results for demonstration.
        
        Args:
            query: The search query.
            num_results: Number of results to generate.
            
        Returns:
            A list of mock search results.
        """
        # This would be replaced with actual API integration
        results = []
        query_terms = query.lower().split()
        
        for i in range(min(num_results, 10)):
            # Generate a result with some relation to the query
            title = f"Result {i+1} for {query}"
            snippet = f"This is a sample search result about {query}. It contains information related to {', '.join(query_terms)}."
            url = f"https://example.com/result-{i+1}?q={'-'.join(query_terms)}"
            
            results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "position": i + 1
            })
        
        return results
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Format the search results for the tool response.
        
        Args:
            results: The raw search results.
            query: The original query.
            
        Returns:
            A formatted response dictionary.
        """
        formatted_results = []
        
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("url", ""),
                "position": result.get("position", 0)
            })
        
        return {
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
    
    def _get_cache_key(self, query: str, num_results: int) -> str:
        """
        Generate a deterministic cache key for the search query.
        
        Args:
            query: The search query.
            num_results: Number of results.
            
        Returns:
            A cache key string.
        """
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Create a hash of the query parameters
        hash_input = f"{normalized_query}:{num_results}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Create a namespaced Redis key
        return f"tool:web_search:{hash_value}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get search results from cache if available.
        
        Args:
            cache_key: The cache key to lookup.
            
        Returns:
            The cached results or None if not found.
        """
        try:
            try:
            # Get Redis connection
                redis = await get_redis_async_connection()
            except Exception as e:
                class MinimalMock:
                    async def get(self, *args, **kwargs):
                        return None
                redis = MinimalMock()
            
            # Get from cache
            cached_data = await redis.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            # Log but don't fail on cache errors
            logger.warning(
                f"Cache retrieval failed: {str(e)}")
            return None
    
    async def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """
        Save search results to cache.
        
        Args:
            cache_key: The cache key.
            data: The data to cache.
            
        Returns:
            True if caching succeeded, False otherwise.
        """
        try:
            # Get Redis connection
            redis = await get_redis_async_connection()
            
            # Serialize data
            serialized = json.dumps(data)
            
            # Save to cache with TTL
            await redis.setex(cache_key, self.cache_ttl, serialized)
            
            return True
            
        except Exception as e:
            # Log but don't fail on cache errors
            logger.warning(
                f"Cache save failed: {str(e)}",
                extra={"cache_key": cache_key},
                exc_info=e
            )
            
            return False
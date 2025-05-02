import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel

from src.config.connections import get_connection_manager
from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.tools.base import BaseTool
from src.tools.registry import register_tool
from src.tools.web_search import WebSearchInput, track_cache_miss
from src.utils.timing import AsyncTimer

logger = get_logger(__name__)
settings = get_settings()
metrics = get_metrics_manager()
conn_manager = get_connection_manager()

@register_tool()
class GoogleSearchTool(BaseTool):
    """Tool for searching the web using Google Custom Search API."""
    
    def __init__(self):
        self.cache_ttl = getattr(settings, 'SEARCH_CACHE_TTL', 3600)
        self.api_key = getattr(settings, 'GOOGLE_SEARCH_API_KEY', '')
        self.search_engine_id = getattr(settings, 'GOOGLE_SEARCH_ENGINE_ID', '')
        
        # API endpoint for Google Custom Search
        self.api_url = "https://customsearch.googleapis.com/customsearch/v1"
        
        logger.debug(
            f'GoogleSearchTool initialized with cache TTL: {self.cache_ttl}s, '
            f'API key present: {bool(self.api_key)}, '
            f'Search engine ID present: {bool(self.search_engine_id)}'
        )
        
        # Check if API credentials are configured
        if not self.api_key or not self.search_engine_id:
            logger.warning(
                "GoogleSearchTool is missing API key or search engine ID. "
                "Please set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID "
                "in your .env file or environment variables."
            )
        
    @property
    def name(self) -> str:
        return 'google_search'
        
    @property
    def description(self) -> str:
        return 'Search the web using Google Custom Search API. Returns top search results with higher quality than basic web search.'
    
    @property
    def args_schema(self) -> type[BaseModel]:
        return WebSearchInput
    
    def _run(self, **kwargs: Any) -> Any:
        """Run the tool synchronously by deferring to the async implementation."""
        import asyncio
        
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No event loop exists, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._arun(**kwargs))
                loop.close()
                return result
            else:
                # Event loop exists but we can't use run_until_complete in it
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(self._arun(**kwargs))
                finally:
                    new_loop.close()
        except Exception as e:
            logger.error(f"Error running sync GoogleSearchTool: {e}", exc_info=True)
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Sync execution failed: {str(e)}",
                original_error=e,
                tool_name=self.name
            )
    
    async def _arun(self, **kwargs: Any) -> Any:
        """Run the Google search asynchronously."""
        query: str = kwargs['query']
        num_results: int = kwargs.get('num_results', 5)
        safe_search: bool = kwargs.get('safe_search', True)
        
        # Validate API credentials
        if not self.api_key or not self.search_engine_id:
            raise ToolError(
                code=ErrorCode.TOOL_VALIDATION_ERROR, 
                message="Google Search API key and Search Engine ID must be configured",
                details={
                    "api_key_present": bool(self.api_key),
                    "search_engine_id_present": bool(self.search_engine_id)
                },
                tool_name=self.name
            )
        
        logger.info(f"Performing Google search for query: '{query}', num_results: {num_results}")
        
        # Check cache first
        cache_key = self._get_cache_key(query, num_results, safe_search)
        cached_result = await self._get_from_cache(cache_key)
        
        if cached_result is not None:
            logger.info(f"Cache hit for Google search query: '{query}' (Key: {cache_key})")
            metrics.track_cache('hits', cache_type='google_search')
            return cached_result
        
        logger.info(f"Cache miss for Google search query: '{query}' (Key: {cache_key}). Performing search.")
        metrics.track_cache('misses', cache_type='google_search')
        
        try:
            async with AsyncTimer('google_search_api_call'):
                raw_results = await self._perform_search(query, num_results, safe_search)
            
            search_results = self._format_results(raw_results, query)
            await self._save_to_cache(cache_key, search_results)
            
            return search_results
        except Exception as e:
            logger.error(
                f"Google search failed for query '{query}': {str(e)}",
                extra={'query': query, 'error': str(e)},
                exc_info=e
            )
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Google search failed: {str(e)}",
                details={'query': query, 'num_results': num_results, 'error': str(e)},
                original_error=e,
                tool_name=self.name
            )
    
    async def _perform_search(self, query: str, num_results: int, safe_search: bool = True) -> List[Dict[str, Any]]:
        """
        Perform a search using Google Custom Search API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            safe_search: Whether to enable safe search filtering
            
        Returns:
            List of search result items
        """
        logger.debug(f"Calling Google Search API for query: '{query}' (num_results: {num_results})")
        
        # Prepare request parameters
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10)  # Google API max is 10 results per request
        }
        
        if safe_search:
            params["safe"] = "active"
        
        try:
            # Make the request to Google API
            async with conn_manager.http_session() as session: # http_session() 컨텍스트 매니저 사용 권장
                async with session.get(self.api_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            # Extract search results
            results = []
            if "items" in data:
                for i, item in enumerate(data["items"]):
                    title = item.get("title", "No Title")
                    snippet = item.get("snippet", "")
                    
                    # Format the URL, preferring formattedUrl if available
                    url = item.get("formattedUrl", item.get("link", ""))
                    
                    # Get page type information if available
                    page_type = None
                    if "pagemap" in item and "metatags" in item["pagemap"]:
                        for metatag in item["pagemap"]["metatags"]:
                            if "og:type" in metatag:
                                page_type = metatag["og:type"]
                                break
                    
                    # Get search result metadata
                    metadata = {
                        "position": i + 1,
                        "source": "google",
                        "mime_type": item.get("mime", ""),
                        "file_format": item.get("fileFormat", ""),
                        "page_type": page_type
                    }
                    
                    # Clean up metadata by removing empty values
                    metadata = {k: v for k, v in metadata.items() if v}
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "position": i + 1,
                        "source": "google",
                        "metadata": metadata
                    })
            
            # Log search statistics
            logger.debug(f"Google search returned {len(results)} results for query: '{query}'")
            
            return results
            
        except aiohttp.ClientError as e:
            logger.error(f"Google API request failed: {str(e)}", exc_info=True)
            raise ToolError(
                code=ErrorCode.CONNECTION_ERROR,
                message=f"Failed to connect to Google Search API: {str(e)}",
                details={'query': query},
                original_error=e,
                tool_name=self.name
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Google API response: {str(e)}", exc_info=True)
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to parse Google API response: {str(e)}",
                details={'query': query},
                original_error=e,
                tool_name=self.name
            )
        except Exception as e:
            logger.error(f"Unexpected error during Google search: {str(e)}", exc_info=True)
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Google search failed: {str(e)}",
                details={'query': query},
                original_error=e,
                tool_name=self.name
            )
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Format raw API results into a standardized structure."""
        formatted_results = []
        
        for result in results:
            formatted_results.append({
                "title": result.get("title", "No Title"),
                "snippet": result.get("snippet", "No Snippet"),
                "url": result.get("url", ""),
                "position": result.get("position", 0),
                "source": "google",
                "metadata": result.get("metadata", {})
            })
        
        return {
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results,
            "search_engine": "google"
        }
    
    def _get_cache_key(self, query: str, num_results: int, safe_search: bool) -> str:
        """Generate a cache key specifically for Google searches."""
        normalized_query = query.lower().strip()
        hash_input = f'google:{normalized_query}:{num_results}:{safe_search}'
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        return f'tool:google_search:{hash_value}'
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached search results."""
        logger.debug(f'Attempting to get Google search result from cache (Key: {cache_key})')
        
        try:
            redis = await conn_manager.get_redis_async_connection()
            time.time()
            cached_data_bytes: Optional[bytes] = await redis.get(cache_key)
            
            if cached_data_bytes:
                cached_result: Dict[str, Any] = json.loads(cached_data_bytes.decode('utf-8'))
                logger.debug(f'Cache hit in Redis for key: {cache_key}')
                return cached_result
            else:
                logger.debug(f'Cache miss in Redis for key: {cache_key}')
                return None
                
        except Exception as e:
            logger.warning(
                f'Failed to retrieve from Google search cache (Key: {cache_key}): {str(e)}', 
                exc_info=True
            )
            track_cache_miss('google_search_redis_error')
            return None
    
    async def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """Save search results to cache."""
        logger.debug(f'Attempting to save Google search result to cache (Key: {cache_key})')
        
        try:
            redis = await conn_manager.get_redis_async_connection()
            serialized_data: str = json.dumps(data)
            serialized_bytes: bytes = serialized_data.encode('utf-8')
            
            await redis.setex(cache_key, self.cache_ttl, serialized_bytes)
            
            logger.debug(f'Successfully saved result to cache (Key: {cache_key}, TTL: {self.cache_ttl}s)')
            return True
            
        except Exception as e:
            logger.warning(
                f'Failed to save result to Google search cache (Key: {cache_key}): {str(e)}', 
                extra={'cache_key': cache_key}, 
                exc_info=e
            )
            return False
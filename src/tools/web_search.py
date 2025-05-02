import asyncio
import hashlib
import json
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Type

import aiohttp
from pydantic import BaseModel, Field, field_validator

from src.config.connections import get_connection_manager
from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager
from src.config.settings import get_settings
from src.tools.base import BaseTool
from src.tools.registry import register_tool
from src.utils.timing import AsyncTimer

logger = get_logger(__name__)
settings = get_settings()
metrics = get_metrics_manager()
conn_manager = get_connection_manager()

class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(..., description='The search query to execute')
    num_results: int = Field(5, description='Number of search results to return', ge=1, le=10)
    safe_search: bool = Field(True, description='Whether to enable safe search filtering')

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate search query."""
        if not v or not isinstance(v, str):
            raise ValueError('Query must be a non-empty string')
        if len(v) > 1000:
            raise ValueError('Query is too long (max 1000 chars)')
        v = v.strip()
        if not v:
            raise ValueError('Query must not be empty after stripping whitespace')
        return v

def track_cache_miss(reason: str) -> None:
    """Track a cache miss with the given reason."""
    metrics.track_cache('misses', cache_type='web_search')
    logger.debug(f"Cache miss tracked: {reason}")

@register_tool()
class WebSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo."""

    def __init__(self):
        self.cache_ttl: int = getattr(settings, 'SEARCH_CACHE_TTL', 3600)
        self.api_url: str = getattr(settings, 'DUCKDUCKGO_PROXY_API_URL', 'https://api.duckduckgo.com/')
        logger.debug(f'WebSearchTool initialized with cache TTL: {self.cache_ttl}s and API URL: {self.api_url}')

    @property
    def name(self) -> str:
        return 'web_search'

    @property
    def description(self) -> str:
        return 'Search the web for information on a given query using DuckDuckGo. Returns top search results.'

    @property
    def args_schema(self) -> Type[BaseModel]:
        return WebSearchInput

    def _run(self, **kwargs: Any) -> Any:
        """Run web search synchronously."""
        logger.debug('WebSearchTool._run called. Running async version in new event loop.')
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
            logger.error(f'Error running sync WebSearchTool: {e}', exc_info=True)
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR, 
                message=f'Sync execution failed: {str(e)}', 
                original_error=e, 
                tool_name=self.name
            ) from e

    async def _arun(self, **kwargs: Any) -> Any:
        """Run web search asynchronously."""
        query: str = kwargs['query']
        num_results: int = kwargs.get('num_results', 5)
        safe_search: bool = kwargs.get('safe_search', True)
        
        logger.info(f"Performing DuckDuckGo search for query: '{query}', num_results: {num_results}")
        
        cache_key = self._get_cache_key(query, num_results, safe_search)
        cached_result = await self._get_from_cache(cache_key)
        
        if cached_result is not None:
            logger.info(f"Cache hit for web search query: '{query}' (Key: {cache_key})")
            metrics.track_cache('hits', cache_type='web_search')

            return cached_result
            
        logger.info(f"Cache miss for web search query: '{query}' (Key: {cache_key}). Performing search.")
        metrics.track_cache('misses', cache_type='web_search')
        
        try:
            async with AsyncTimer('web_search_api_call'):
                raw_results: List[Dict[str, Any]] = await self._perform_search(query, num_results, safe_search)
                
            search_results: Dict[str, Any] = self._format_results(raw_results, query)
            await self._save_to_cache(cache_key, search_results)
            
            return search_results
        except Exception as e:
            logger.error(
                f"Web search failed for query '{query}': {str(e)}", 
                extra={'query': query, 'error': str(e)}, 
                exc_info=e
            )
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR, 
                message=f'Web search execution failed: {str(e)}', 
                details={'query': query, 'num_results': num_results, 'error': str(e)}, 
                original_error=e, 
                tool_name=self.name
            )

    async def _perform_search(self, query: str, num_results: int, safe_search: bool = True) -> List[Dict[str, Any]]:
        """
        Perform the actual search using DuckDuckGo's API.
        
        DuckDuckGo doesn't have an official API for search results, but we can use their
        instant answer API and parse the results.
        """
        logger.debug(f"Performing DuckDuckGo search for: '{query}' (num_results: {num_results})")
        
        # Prepare request parameters
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'no_redirect': '1',
            'kl': 'us-en',  # Region and language
            'kd': '-1',     # No time limit
            't': 'web_search_tool'
        }
        
        if safe_search:
            params['kp'] = '1'  # Safe search on
            
        try:
            # Make the request to DuckDuckGo API
            async with conn_manager.http_session() as session: # http_session() 컨텍스트 매니저 사용 권장
                async with session.get(self.api_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
            results = []
            
            # Extract abstract (main result)
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'DuckDuckGo Result'),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', ''),
                    'position': 1,
                    'source': 'duckduckgo_abstract'
                })
                
            # Extract related topics
            if data.get('RelatedTopics'):
                for i, topic in enumerate(data.get('RelatedTopics', []), start=len(results) + 1):
                    if i > num_results:
                        break
                    
                    # Skip category headers
                    if 'Topics' in topic:
                        continue
                        
                    # Extract info
                    url = topic.get('FirstURL', '')
                    text = topic.get('Text', '')
                    
                    # Sometimes the first part of Text is the title
                    title_parts = text.split(' - ', 1)
                    if len(title_parts) > 1:
                        title, snippet = title_parts
                    else:
                        title = url.split('/')[-1].replace('_', ' ').title()
                        snippet = text
                        
                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'url': url,
                        'position': i,
                        'source': 'duckduckgo_related'
                    })
                    
            # If no results from abstract or related topics, use our fallback method
            if not results and not data.get('Redirect'):
                # If DuckDuckGo's API doesn't return results, we'd use a fallback
                # This could be switching to a different search method or using cached data
                logger.warning(f"No results from DuckDuckGo API for query: '{query}'. Using fallback.")
                results = self._generate_fallback_results(query, num_results)
                
            # Limit to requested number
            return results[:num_results]
                
        except aiohttp.ClientError as e:
            logger.error(f"DuckDuckGo API request failed: {str(e)}", exc_info=True)
            raise ToolError(
                code=ErrorCode.CONNECTION_ERROR,
                message=f"Failed to connect to DuckDuckGo API: {str(e)}",
                details={'query': query},
                original_error=e,
                tool_name=self.name
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse DuckDuckGo API response: {str(e)}", exc_info=True)
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"Failed to parse DuckDuckGo API response: {str(e)}",
                details={'query': query},
                original_error=e,
                tool_name=self.name
            )
        except Exception as e:
            logger.error(f"Unexpected error during DuckDuckGo search: {str(e)}", exc_info=True)
            raise ToolError(
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                message=f"DuckDuckGo search failed: {str(e)}",
                details={'query': query},
                original_error=e,
                tool_name=self.name
            )

    def _generate_fallback_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Generate fallback search results when the API fails to return results."""
        logger.debug(f"Generating fallback results for query: '{query}'")
        results = []
        query_terms = query.lower().split()
        
        for i in range(min(num_results, 10)):
            title = f"Search Result {i + 1} for '{query}'"
            snippet = f"This is a fallback result for '{query}'. It may contain information about: {', '.join(query_terms)}."
            url = f'https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}'
            
            results.append({
                'title': title, 
                'snippet': snippet, 
                'url': url, 
                'position': i + 1,
                'source': 'fallback'
            })
            
        return results

    def _format_results(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Format search results into a standard structure."""
        formatted_results: List[Dict[str, Any]] = []
        
        for result in results:
            formatted_results.append({
                'title': result.get('title', 'No Title'),
                'snippet': result.get('snippet', 'No Snippet'),
                'url': result.get('url', ''),
                'position': result.get('position', 0),
                'source': result.get('source', 'duckduckgo')
            })
            
        return {
            'query': query,
            'total_results': len(formatted_results),
            'results': formatted_results,
            'search_engine': 'duckduckgo'
        }

    def _get_cache_key(self, query: str, num_results: int, safe_search: bool) -> str:
        """Generate a cache key for the search."""
        normalized_query = query.lower().strip()
        hash_input = f'duckduckgo:{normalized_query}:{num_results}:{safe_search}'
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        return f'tool:web_search:{hash_value}'

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached search results."""
        logger.debug(f'Attempting to get web search result from cache (Key: {cache_key})')
        
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
                f'Failed to retrieve from web search cache (Key: {cache_key}): {str(e)}', 
                exc_info=True
            )
            track_cache_miss('web_search_redis_error')
            return None

    async def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """Save search results to cache."""
        logger.debug(f'Attempting to save web search result to cache (Key: {cache_key})')
        
        try:
            redis = await conn_manager.get_redis_async_connection()
            serialized_data: str = json.dumps(data)
            serialized_bytes: bytes = serialized_data.encode('utf-8')
            
            time.time()
            await redis.setex(cache_key, self.cache_ttl, serialized_bytes)
            
            logger.debug(f'Successfully saved result to cache (Key: {cache_key}, TTL: {self.cache_ttl}s)')
            return True
            
        except Exception as e:
            logger.warning(
                f'Failed to save result to web search cache (Key: {cache_key}): {str(e)}', 
                extra={'cache_key': cache_key}, 
                exc_info=e
            )
            return False
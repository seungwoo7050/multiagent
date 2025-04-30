import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator
from src.config.errors import ErrorCode, ToolError
from src.config.logger import get_logger
from src.config.metrics import CACHE_HITS_TOTAL, CACHE_MISSES_TOTAL
from src.config.connections import get_http_session, get_redis_async_connection
from src.tools.base import BaseTool
from src.tools.registry import register_tool
from src.utils.timing import AsyncTimer
logger = get_logger(__name__)

class WebSearchInput(BaseModel):
    query: str = Field(..., description='The search query to execute')
    num_results: int = Field(5, description='Number of search results to return', ge=1, le=10)

    @field_validator('query')
    def validate_query(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError('Query must be a non-empty string')
        if len(v) > 1000:
            raise ValueError('Query is too long (max 1000 chars)')
        v = v.strip()
        if not v:
            raise ValueError('Query must not be empty after stripping whitespace')
        return v

@register_tool()
class WebSearchTool(BaseTool):

    def __init__(self):
        self.cache_ttl: int = 3600
        logger.debug(f'WebSearchTool initialized with cache TTL: {self.cache_ttl}s')

    @property
    def name(self) -> str:
        return 'web_search'

    @property
    def description(self) -> str:
        return 'Search the web for information on a given query. Returns top search results.'

    @property
    def args_schema(self) -> Type[BaseModel]:
        return WebSearchInput

    def _run(self, **kwargs: Any) -> Any:
        logger.debug('WebSearchTool._run called. Running async version in new event loop.')
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run(self._arun(**kwargs))
        except RuntimeError as e:
            logger.warning(f'Could not get or run event loop for sync WebSearchTool: {e}. Creating new loop.')
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(self._arun(**kwargs))
            finally:
                new_loop.close()
        except Exception as e:
            logger.error(f'Error running sync WebSearchTool: {e}', exc_info=True)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Sync execution failed: {str(e)}', original_error=e, tool_name=self.name) from e

    async def _arun(self, **kwargs: Any) -> Any:
        query: str = kwargs['query']
        num_results: int = kwargs.get('num_results', 5)
        logger.info(f"Performing async web search for query: '{query}', num_results: {num_results}")
        cache_key = self._get_cache_key(query, num_results)
        cached_result = await self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for web search query: '{query}' (Key: {cache_key})")
            CACHE_HITS_TOTAL.labels(cache_type='web_search').inc()
            return cached_result
        logger.info(f"Cache miss for web search query: '{query}' (Key: {cache_key}). Performing search.")
        CACHE_MISSES_TOTAL.labels(cache_type='web_search').inc()
        try:
            async with AsyncTimer('web_search_api_call'):
                raw_results: List[Dict[str, Any]] = await self._perform_search(query, num_results)
            search_results: Dict[str, Any] = self._format_results(raw_results, query)
            await self._save_to_cache(cache_key, search_results)
            return search_results
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {str(e)}", extra={'query': query, 'error': str(e)}, exc_info=e)
            raise ToolError(code=ErrorCode.TOOL_EXECUTION_ERROR, message=f'Web search execution failed: {str(e)}', details={'query': query, 'num_results': num_results, 'error': str(e)}, original_error=e, tool_name=self.name)

    async def _perform_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        logger.debug(f"Performing mock web search for: '{query}' (num_results: {num_results})")
        await asyncio.sleep(0.1)
        mock_results = self._generate_mock_results(query, num_results)
        logger.debug(f"Generated {len(mock_results)} mock search results for query '{query}'")
        return mock_results

    def _generate_mock_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        query_terms = query.lower().split()
        for i in range(min(num_results, 10)):
            title = f"Mock Result {i + 1} about '{query}'"
            snippet = f"This is a generated mock snippet for result {i + 1}. It mentions terms like: {', '.join(query_terms)}. More details about '{query}' can be found here."
            url = f'https://mock-search.example.com/result{i + 1}?query={'_'.join(query_terms)}'
            results.append({'title': title, 'snippet': snippet, 'url': url, 'position': i + 1})
        return results

    def _format_results(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        formatted_results: List[Dict[str, Any]] = []
        for result in results:
            formatted_results.append({'title': result.get('title', 'No Title'), 'snippet': result.get('snippet', 'No Snippet'), 'url': result.get('url', ''), 'position': result.get('position', 0)})
        return {'query': query, 'total_results': len(formatted_results), 'results': formatted_results}

    def _get_cache_key(self, query: str, num_results: int) -> str:
        normalized_query = query.lower().strip()
        hash_input = f'{normalized_query}:{num_results}'
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        return f'tool:web_search:{hash_value}'

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        logger.debug(f'Attempting to get web search result from cache (Key: {cache_key})')
        try:
            redis = await get_redis_async_connection()
            start_time = time.time()
            cached_data_bytes: Optional[bytes] = await redis.get(cache_key)
            track_memory_operation_completed('redis_get_web_cache', time.time() - start_time)
            if cached_data_bytes:
                cached_result: Dict[str, Any] = json.loads(cached_data_bytes.decode('utf-8'))
                logger.debug(f'Cache hit in Redis for key: {cache_key}')
                return cached_result
            else:
                logger.debug(f'Cache miss in Redis for key: {cache_key}')
                return None
        except Exception as e:
            logger.warning(f'Failed to retrieve from web search cache (Key: {cache_key}): {str(e)}', exc_info=True)
            track_cache_miss('web_search_redis_error')
            return None

    async def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> bool:
        logger.debug(f'Attempting to save web search result to cache (Key: {cache_key})')
        try:
            redis = await get_redis_async_connection()
            serialized_data: str = json.dumps(data)
            serialized_bytes: bytes = serialized_data.encode('utf-8')
            start_time = time.time()
            await redis.setex(cache_key, self.cache_ttl, serialized_bytes)
            track_memory_operation_completed('redis_set_web_cache', time.time() - start_time)
            logger.debug(f'Successfully saved result to cache (Key: {cache_key}, TTL: {self.cache_ttl}s)')
            return True
        except Exception as e:
            logger.warning(f'Failed to save result to web search cache (Key: {cache_key}): {str(e)}', extra={'cache_key': cache_key}, exc_info=e)
            return False
# src/tools/web_search.py
import asyncio
from typing import Any, Dict, List, Optional, Type

# 필요한 LangChain 및 Pydantic import
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# BaseTool 및 오류/로거 import
from src.tools.base import BaseTool
from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger
from src.services.tool_manager import register_tool

logger = get_logger(__name__)

# 간단한 입력 스키마 정의 (query만 필요)
class DuckDuckGoInput(BaseModel):
    query: str = Field(..., description="The search query string.")

# @register_tool() 데코레이터 유지
@register_tool()
class WebSearchTool(BaseTool):
    """
    Performs a web search using the DuckDuckGo search engine.
    Useful for finding current information, answering general knowledge questions,
    or looking up specific topics online. Input should be the search query.
    """
    # 클래스 변수로 name, description, args_schema 정의
    name: str = "web_search" # 또는 "duckduckgo_search"
    description: str = (
        "Performs a web search using DuckDuckGo to find current information, "
        "answer general knowledge questions, or look up specific topics. "
        "Input 'query' is the search term."
    )
    args_schema: Type[BaseModel] = DuckDuckGoInput

    # 내부에서 사용할 LangChain 도구 인스턴스 (초기화 시 생성하거나 메서드 내에서 생성)
    # 여기서는 메서드 내에서 필요시 생성
    _search_instance: Optional[DuckDuckGoSearchRun] = None

    def _get_search_instance(self) -> DuckDuckGoSearchRun:
        """DuckDuckGoSearchRun 인스턴스를 가져오거나 생성합니다."""
        # 간단한 싱글톤 패턴 또는 매번 생성 (DuckDuckGoSearchRun은 상태가 거의 없음)
        if self._search_instance is None:
            try:
                self._search_instance = DuckDuckGoSearchRun()
            except Exception as e:
                 logger.error(f"Failed to initialize DuckDuckGoSearchRun: {e}", exc_info=True)
                 raise ToolError(
                     message=f"Failed to initialize the DuckDuckGo search tool: {e}",
                     code=ErrorCode.TOOL_CREATION_ERROR,
                     tool_name=self.name,
                     original_error=e
                 )
        return self._search_instance

    # _run 메서드 구현
    def _run(self, query: str) -> str:
        """
        Synchronously performs a web search using DuckDuckGoSearchRun.
        """
        logger.debug(f"WebSearchTool: Executing search for query: '{query}'")
        try:
            search_tool = self._get_search_instance()
            # LangChain 도구의 run 메서드 호출
            result: str = search_tool.run(tool_input=query) # run 메서드는 문자열 query를 직접 받음
            logger.info(f"WebSearchTool: Search successful for query '{query}'. Result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"WebSearchTool: Error during DuckDuckGo search for query '{query}': {e}", exc_info=True)
            raise ToolError(
                message=f"Web search failed: {str(e)}",
                tool_name=self.name,
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                original_error=e,
                details={'query': query}
            )

    # _arun 메서드 구현
    async def _arun(self, query: str) -> str:
        """
        Asynchronously performs a web search using DuckDuckGoSearchRun.
        """
        logger.debug(f"WebSearchTool: Asynchronously executing search for query: '{query}'")
        try:
            search_tool = self._get_search_instance()
            # LangChain 도구의 arun 메서드 호출
            result: str = await search_tool.arun(tool_input=query) # arun도 문자열 query를 받음
            logger.info(f"WebSearchTool: Async search successful for query '{query}'. Result length: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"WebSearchTool: Error during async DuckDuckGo search for query '{query}': {e}", exc_info=True)
            raise ToolError(
                message=f"Async web search failed: {str(e)}",
                tool_name=self.name,
                code=ErrorCode.TOOL_EXECUTION_ERROR,
                original_error=e,
                details={'query': query}
            )

    # 기존 _perform_search, _format_results, 캐싱 관련 메서드들은 모두 제거
                         
import asyncio
from typing import Any, Dict, List, Optional, Type

                                 
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

                         
from src.tools.base import BaseTool
from src.config.errors import ErrorCode, ToolError
from src.utils.logger import get_logger
from src.services.tool_manager import register_tool

logger = get_logger(__name__)

                           
class DuckDuckGoInput(BaseModel):
    query: str = Field(..., description="The search query string.")

                           
@register_tool()
class WebSearchTool(BaseTool):
    """
    Performs a web search using the DuckDuckGo search engine.
    Useful for finding current information, answering general knowledge questions,
    or looking up specific topics online. Input should be the search query.
    """
                                               
    name: str = "web_search"                         
    description: str = (
        "Performs a web search using DuckDuckGo to find current information, "
        "answer general knowledge questions, or look up specific topics. "
        "Input 'query' is the search term."
    )
    args_schema: Type[BaseModel] = DuckDuckGoInput

                                                         
                         
    _search_instance: Optional[DuckDuckGoSearchRun] = None

    def _get_search_instance(self) -> DuckDuckGoSearchRun:
        """DuckDuckGoSearchRun 인스턴스를 가져오거나 생성합니다."""
                                                              
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

                 
    def _run(self, query: str) -> str:
        """
        Synchronously performs a web search using DuckDuckGoSearchRun.
        """
        logger.debug(f"WebSearchTool: Executing search for query: '{query}'")
        try:
            search_tool = self._get_search_instance()
                                      
            result: str = search_tool.run(tool_input=query)                            
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

                  
    async def _arun(self, query: str) -> str:
        """
        Asynchronously performs a web search using DuckDuckGoSearchRun.
        """
        logger.debug(f"WebSearchTool: Asynchronously executing search for query: '{query}'")
        try:
            search_tool = self._get_search_instance()
                                       
            result: str = await search_tool.arun(tool_input=query)                      
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

                                                            
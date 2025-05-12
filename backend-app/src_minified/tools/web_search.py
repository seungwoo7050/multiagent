H='query'
F=True
E=Exception
from typing import Optional,Type
from langchain_community.tools import DuckDuckGoSearchRun as I
from pydantic import BaseModel as B,Field
from src.tools.base import BaseTool as G
from src.config.errors import ErrorCode as C,ToolError as D
from src.utils.logger import get_logger as J
from src.services.tool_manager import register_tool as K
A=J(__name__)
class L(B):query=Field(...,description='The search query string.')
@K()
class M(G):
	name='web_search';description="Performs a web search using DuckDuckGo to find current information, answer general knowledge questions, or look up specific topics. Input 'query' is the search term.";args_schema=L;_search_instance=None
	def _get_search_instance(B):
		if B._search_instance is None:
			try:B._search_instance=I()
			except E as G:A.error(f"Failed to initialize DuckDuckGoSearchRun: {G}",exc_info=F);raise D(message=f"Failed to initialize the DuckDuckGo search tool: {G}",code=C.TOOL_CREATION_ERROR,tool_name=B.name,original_error=G)
		return B._search_instance
	def _run(I,query):
		B=query;A.debug(f"WebSearchTool: Executing search for query: '{B}'")
		try:K=I._get_search_instance();J=K.run(tool_input=B);A.info(f"WebSearchTool: Search successful for query '{B}'. Result length: {len(J)}");return J
		except E as G:A.error(f"WebSearchTool: Error during DuckDuckGo search for query '{B}': {G}",exc_info=F);raise D(message=f"Web search failed: {str(G)}",tool_name=I.name,code=C.TOOL_EXECUTION_ERROR,original_error=G,details={H:B})
	async def _arun(I,query):
		B=query;A.debug(f"WebSearchTool: Asynchronously executing search for query: '{B}'")
		try:K=I._get_search_instance();J=await K.arun(tool_input=B);A.info(f"WebSearchTool: Async search successful for query '{B}'. Result length: {len(J)}");return J
		except E as G:A.error(f"WebSearchTool: Error during async DuckDuckGo search for query '{B}': {G}",exc_info=F);raise D(message=f"Async web search failed: {str(G)}",tool_name=I.name,code=C.TOOL_EXECUTION_ERROR,original_error=G,details={H:B})
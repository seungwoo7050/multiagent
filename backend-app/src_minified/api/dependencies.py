I=Exception
H=True
G=None
import asyncio as M
from typing import Annotated as D,Optional
from fastapi import Depends as E,HTTPException as B,status as C
from src.utils.logger import get_logger as N
from src.config.settings import get_settings as O
from src.memory.memory_manager import MemoryManager as P,get_memory_manager as Q
from src.agents.orchestrator import Orchestrator as J
from src.services.llm_client import LLMClient as K
from src.services.tool_manager import ToolManager as R,get_tool_manager as S
from src.services.notification_service import NotificationService as L
Z=O()
A=N(__name__)
F=G
T=M.Lock()
async def U():
	global F
	if F is G:
		async with T:
			if F is G:F=L();A.info('NotificationService singleton instance created.')
	return F
a=D[L,E(U)]
async def V():
	try:
		D=Q()
		if D is G:raise B(status_code=C.HTTP_503_SERVICE_UNAVAILABLE,detail='Memory service is not available.')
		return D
	except I as E:A.error(f"Failed to get MemoryManager dependency: {E}",exc_info=H);raise B(status_code=C.HTTP_500_INTERNAL_SERVER_ERROR,detail='Could not initialize memory service.')
b=D[P,E(V)]
async def W():
	try:D=K();return D
	except I as E:A.error(f"Failed to get LLMClient dependency: {E}",exc_info=H);raise B(status_code=C.HTTP_500_INTERNAL_SERVER_ERROR,detail='Could not initialize LLM service.')
c=D[K,E(W)]
async def X():
	try:
		D=S('global_tools')
		if D is G:raise B(status_code=C.HTTP_503_SERVICE_UNAVAILABLE,detail='Tool service is not available.')
		return D
	except I as E:A.error(f"Failed to get ToolManager dependency: {E}",exc_info=H);raise B(status_code=C.HTTP_500_INTERNAL_SERVER_ERROR,detail='Could not initialize tool service.')
d=D[R,E(X)]
async def Y(llm_client,tool_manager,memory_manager,notification_service):
	A.debug('Resolving New Orchestrator dependency with LLMClient, ToolManager, MemoryManager, and NotificationService...')
	try:E=J(llm_client=llm_client,tool_manager=tool_manager,memory_manager=memory_manager,notification_service=notification_service);A.debug('New Orchestrator instance created successfully.');return E
	except ValueError as D:A.error(f"New Orchestrator dependency error: {D}",exc_info=H);raise B(status_code=C.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"Orchestrator configuration error: {D}")
	except I as F:A.error(f"Failed to get New Orchestrator dependency: {F}",exc_info=H);raise B(status_code=C.HTTP_500_INTERNAL_SERVER_ERROR,detail='Could not initialize task processing service.')
e=D[J,E(Y)]
A.info('API Dependencies configured for framework-centric Orchestrator.')
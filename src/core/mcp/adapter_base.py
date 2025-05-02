import abc
import asyncio
from typing import Any, Optional, Type

from src.config.logger import get_logger
from src.core.mcp.protocol import ContextProtocol

logger = get_logger(__name__)

class MCPAdapterBase(abc.ABC):

    def __init__(self, target_component: Any, mcp_context_type: Optional[Type[ContextProtocol]]=None):
        self.target_component = target_component
        self.mcp_context_type = mcp_context_type
        logger.debug(f'Initialized {self.__class__.__name__} for target {type(target_component).__name__}')

    @abc.abstractmethod
    async def adapt_input(self, context: ContextProtocol, **kwargs: Any) -> Any:
        pass

    @abc.abstractmethod
    async def adapt_output(self, component_output: Any, original_context: Optional[ContextProtocol]=None, **kwargs: Any) -> ContextProtocol:
        pass

    async def process_with_mcp(self, context: ContextProtocol, **kwargs: Any) -> ContextProtocol:
        try:
            logger.debug(f'[{self.__class__.__name__}] Adapting input context...')
            adapted_input = await self.adapt_input(context, **kwargs)
            logger.debug(f'[{self.__class__.__name__}] Executing target component ({type(self.target_component).__name__})...')
            component_output: Any
            if hasattr(self.target_component, 'execute') and callable(self.target_component.execute):
                if asyncio.iscoroutinefunction(self.target_component.execute):
                    component_output = await self.target_component.execute(adapted_input)
                else:
                    logger.warning(f"Target component {type(self.target_component).__name__}'s execute method is synchronous.")
                    component_output = self.target_component.execute(adapted_input)
            elif hasattr(self.target_component, 'process') and callable(self.target_component.process):
                if asyncio.iscoroutinefunction(self.target_component.process):
                    component_output = await self.target_component.process(adapted_input)
                else:
                    logger.warning(f"Target component {type(self.target_component).__name__}'s process method is synchronous.")
                    component_output = self.target_component.process(adapted_input)
            else:
                raise NotImplementedError(f"Target component {type(self.target_component).__name__} lacks a standard execution method (e.g., 'execute', 'process'). MCPAdapterBase.process_with_mcp needs to be overridden in {self.__class__.__name__}.")
            logger.debug(f'[{self.__class__.__name__}] Target component execution finished.')
            logger.debug(f'[{self.__class__.__name__}] Adapting component output...')
            output_context = await self.adapt_output(component_output, original_context=context, **kwargs)
            logger.debug(f'[{self.__class__.__name__}] Output adaptation finished.')
            return output_context
        except Exception as e:
            logger.error(f'Error during MCP processing via adapter {self.__class__.__name__}: {e}', exc_info=True)
            raise
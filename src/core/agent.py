import abc
import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast
from pydantic import BaseModel, Field, field_validator
from src.config.logger import get_logger
from src.config.metrics import track_agent_created, track_agent_operation, track_agent_operation_completed, track_agent_error
from src.core.task import BaseTask, TaskResult
from src.utils.timing import AsyncTimer, get_current_time_ms
logger = get_logger(__name__)

class AgentState(str, Enum):
    IDLE = 'idle'
    INITIALIZING = 'initializing'
    PROCESSING = 'processing'
    ERROR = 'error'
    TERMINATED = 'terminated'

class AgentCapability(str, Enum):
    PLANNING = 'planning'
    EXECUTION = 'execution'
    REASONING = 'reasoning'
    TOOL_USE = 'tool_use'
    CODE_GENERATION = 'code_generation'
    INFORMATION_RETRIEVAL = 'information_retrieval'
    CREATIVE_WRITING = 'creative_writing'
    CONVERSATION = 'conversation'
    SUMMARIZATION = 'summarization'
    TEXT_ANALYSIS = 'text_analysis'

class AgentContext(BaseModel):
    task: Optional[BaseTask] = None
    parent_agent_id: Optional[str] = None
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    memory: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_param(self, key: str, default: Any=None) -> Any:
        return self.parameters.get(key, default)

    def update_memory(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def get_memory(self, key: str, default: Any=None) -> Any:
        return self.memory.get(key, default)

    class Config:
        arbitrary_types_allowed = True

class AgentConfig(BaseModel):
    name: str
    description: str = ''
    version: str = '1.0.0'
    agent_type: str
    model: Optional[str] = None
    capabilities: Set[AgentCapability] = Field(default_factory=set)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 3
    timeout: float = 30.0
    allowed_tools: List[str] = Field(default_factory=list)
    memory_keys: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('agent_type')
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        if not v:
            raise ValueError('Agent type cannot be empty')
        return v

    @field_validator('capabilities')
    @classmethod
    def validate_capabilities(cls, v: Set[AgentCapability]) -> Set[AgentCapability]:
        if not v:
            raise ValueError('At least one agent capability must be specified')
        return v

class AgentResult(BaseModel):
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    task_result: Optional[TaskResult] = None
    execution_time: float = 0.0

    @staticmethod
    def success_result(output: Dict[str, Any], execution_time: float, task_result: Optional[TaskResult]=None, metadata: Optional[Dict[str, Any]]=None) -> 'AgentResult':
        return AgentResult(success=True, output=output, execution_time=execution_time, task_result=task_result, metadata=metadata or {})

    @staticmethod
    def error_result(error: Dict[str, Any], execution_time: float, task_result: Optional[TaskResult]=None, metadata: Optional[Dict[str, Any]]=None) -> 'AgentResult':
        return AgentResult(success=False, error=error, execution_time=execution_time, task_result=task_result, metadata=metadata or {})

class BaseAgent(abc.ABC):

    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState.IDLE
        self.last_activity_time = get_current_time_ms()
        self.execution_count = 0
        self.error_count = 0
        track_agent_created(self.config.agent_type)
        logger.info(f'Agent initialized: {self.config.name} (type: {self.config.agent_type})', extra={'agent_type': self.config.agent_type, 'agent_name': self.config.name, 'agent_version': self.config.version})

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def agent_type(self) -> str:
        return self.config.agent_type

    @property
    def is_idle(self) -> bool:
        return self.state == AgentState.IDLE

    @property
    def is_busy(self) -> bool:
        return self.state in {AgentState.INITIALIZING, AgentState.PROCESSING}

    @property
    def is_terminated(self) -> bool:
        return self.state == AgentState.TERMINATED

    @property
    def idle_time_ms(self) -> int:
        if self.is_busy:
            return 0
        return get_current_time_ms() - self.last_activity_time

    @abc.abstractmethod
    async def initialize(self) -> bool:
        if self.state == AgentState.IDLE:
            self.state = AgentState.INITIALIZING
            self.state = AgentState.IDLE
            return True
        return True

    @abc.abstractmethod
    async def process(self, context: AgentContext) -> AgentResult:
        pass

    @abc.abstractmethod
    async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
        pass

    async def execute(self, context: AgentContext) -> AgentResult:
        self.state = AgentState.PROCESSING
        self.last_activity_time = get_current_time_ms()
        self.execution_count += 1
        operation_type = f'execute_{self.agent_type}'
        track_agent_operation(self.agent_type, operation_type)
        async with AsyncTimer(f'agent_{self.agent_type}_execution') as timer:
            try:
                initialized = await self.initialize()
                if not initialized:
                    self.state = AgentState.ERROR
                    error_info = {'type': 'initialization_error', 'message': f'Failed to initialize agent {self.name}'}
                    logger.error(f'Agent initialization failed: {self.name}', extra={'agent_type': self.agent_type, 'error': error_info})
                    track_agent_error(self.agent_type, 'initialization_error')
                    return AgentResult.error_result(error_info, timer.execution_time)
                result = await self.process(context)
                self.state = AgentState.IDLE
                self.last_activity_time = get_current_time_ms()
                track_agent_operation_completed(self.agent_type, operation_type, timer.execution_time)
                return result
            except Exception as e:
                self.state = AgentState.ERROR
                self.error_count += 1
                logger.exception(f'Error during agent execution: {self.name}', extra={'agent_type': self.agent_type, 'error_type': type(e).__name__, 'error_message': str(e)})
                track_agent_error(self.agent_type, type(e).__name__)
                try:
                    return await self.handle_error(e, context)
                except Exception as handler_error:
                    logger.exception(f'Critical error in agent error handler: {self.name}', extra={'agent_type': self.agent_type, 'handler_error_type': type(handler_error).__name__, 'handler_error_message': str(handler_error), 'original_error_type': type(e).__name__, 'original_error_message': str(e)})
                    error_info = {'type': 'unhandled_error', 'message': f'Agent execution failed: {str(e)}', 'handler_error': f'Error handler also failed: {str(handler_error)}'}
                    return AgentResult.error_result(error_info, timer.execution_time)

    async def terminate(self) -> None:
        self.state = AgentState.TERMINATED
        logger.info(f'Agent terminated: {self.name}', extra={'agent_type': self.agent_type})

    async def __aenter__(self) -> 'BaseAgent':
        if not await self.initialize():
            raise RuntimeError(f'Failed to initialize agent {self.name} for async context management')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.terminate()
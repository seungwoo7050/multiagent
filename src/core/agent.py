import abc
import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, cast

from pydantic import BaseModel, Field, field_validator

from src.config.logger import get_logger
from src.config.metrics import (
    track_agent_created,
    track_agent_operation,
    track_agent_operation_completed,
    track_agent_error,
)
from src.core.task import BaseTask, TaskResult
from src.utils.timing import AsyncTimer, get_current_time_ms

# Module logger
logger = get_logger(__name__)


class AgentState(str, Enum):
    """Enum representing the possible states of an agent."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentCapability(str, Enum):
    """Enum representing agent capabilities."""
    PLANNING = "planning"
    EXECUTION = "execution"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    CODE_GENERATION = "code_generation"
    INFORMATION_RETRIEVAL = "information_retrieval"
    CREATIVE_WRITING = "creative_writing"
    CONVERSATION = "conversation"
    SUMMARIZATION = "summarization"
    TEXT_ANALYSIS = "text_analysis"


class AgentContext(BaseModel):
    """Context information provided to an agent."""
    
    task: Optional[BaseTask] = None
    parent_agent_id: Optional[str] = None
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    memory: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with a default fallback."""
        return self.parameters.get(key, default)
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update a memory item."""
        self.memory[key] = value
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Get a memory item with a default fallback."""
        return self.memory.get(key, default)
    
    class Config:
        arbitrary_types_allowed = True


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    
    name: str
    description: str = ""
    version: str = "1.0.0"
    agent_type: str
    model: Optional[str] = None
    capabilities: Set[AgentCapability] = Field(default_factory=set)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    max_retries: int = 3
    timeout: float = 30.0
    allowed_tools: List[str] = Field(default_factory=list)
    memory_keys: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        """Validate agent type."""
        if not v:
            raise ValueError("Agent type cannot be empty")
        return v
    
    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: Set[AgentCapability]) -> Set[AgentCapability]:
        """Validate agent capabilities."""
        # Ensure at least one capability is specified
        if not v:
            raise ValueError("At least one agent capability must be specified")
        return v


class AgentResult(BaseModel):
    """Result returned by an agent."""
    
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    task_result: Optional[TaskResult] = None
    execution_time: float = 0.0
    
    @staticmethod
    def success_result(
        output: Dict[str, Any],
        execution_time: float,
        task_result: Optional[TaskResult] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AgentResult":
        """Create a successful result."""
        return AgentResult(
            success=True,
            output=output,
            execution_time=execution_time,
            task_result=task_result,
            metadata=metadata or {}
        )
    
    @staticmethod
    def error_result(
        error: Dict[str, Any],
        execution_time: float,
        task_result: Optional[TaskResult] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AgentResult":
        """Create an error result."""
        return AgentResult(
            success=False,
            error=error,
            execution_time=execution_time,
            task_result=task_result,
            metadata=metadata or {}
        )


class BaseAgent(abc.ABC):
    """Abstract base class for all agent types with performance metrics."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with the provided configuration."""
        self.config = config
        self.state = AgentState.IDLE
        self.last_activity_time = get_current_time_ms()
        self.execution_count = 0
        self.error_count = 0
        
        # Track metrics
        track_agent_created(self.config.agent_type)
        
        logger.info(
            f"Agent initialized: {self.config.name} (type: {self.config.agent_type})",
            extra={
                "agent_type": self.config.agent_type,
                "agent_name": self.config.name,
                "agent_version": self.config.version
            }
        )
    
    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.config.name
    
    @property
    def agent_type(self) -> str:
        """Get the agent type."""
        return self.config.agent_type
    
    @property
    def is_idle(self) -> bool:
        """Check if the agent is idle."""
        return self.state == AgentState.IDLE
    
    @property
    def is_busy(self) -> bool:
        """Check if the agent is busy."""
        return self.state in {AgentState.INITIALIZING, AgentState.PROCESSING}
    
    @property
    def is_terminated(self) -> bool:
        """Check if the agent is terminated."""
        return self.state == AgentState.TERMINATED
    
    @property
    def idle_time_ms(self) -> int:
        """Get the time since last activity in milliseconds."""
        if self.is_busy:
            return 0
        return get_current_time_ms() - self.last_activity_time
    
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent with any setup work.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass
    
    @abc.abstractmethod
    async def process(self, context: AgentContext) -> AgentResult:
        """Process a task with the given context.
        
        Args:
            context: The agent context including task and parameters.
            
        Returns:
            AgentResult: The result of the agent's processing.
        """
        pass
    
    @abc.abstractmethod
    async def handle_error(self, error: Exception, context: AgentContext) -> AgentResult:
        """Handle an error that occurred during processing.
        
        Args:
            error: The exception that occurred.
            context: The agent context when the error occurred.
            
        Returns:
            AgentResult: The result after error handling.
        """
        pass
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent with performance tracking and error handling.
        
        This method handles state transitions, metrics tracking, and error handling.
        
        Args:
            context: The agent context including task and parameters.
            
        Returns:
            AgentResult: The result of the agent's execution.
        """
        # Update state
        self.state = AgentState.PROCESSING
        self.last_activity_time = get_current_time_ms()
        self.execution_count += 1
        
        # Track the operation
        operation_type = f"execute_{self.agent_type}"
        track_agent_operation(self.agent_type, operation_type)
        
        # Execute with timing
        async with AsyncTimer(f"agent_{self.agent_type}_execution") as timer:
            try:
                # Initialize if not already done
                if not await self.initialize():
                    self.state = AgentState.ERROR
                    error_info = {
                        "type": "initialization_error",
                        "message": f"Failed to initialize agent {self.name}"
                    }
                    logger.error(
                        f"Agent initialization failed: {self.name}",
                        extra={"agent_type": self.agent_type, "error": error_info}
                    )
                    track_agent_error(self.agent_type, "initialization_error")
                    return AgentResult.error_result(error_info, timer.execution_time)
                
                # Process the task
                result = await self.process(context)
                
                # Update state
                self.state = AgentState.IDLE
                self.last_activity_time = get_current_time_ms()
                
                # Track completion
                track_agent_operation_completed(
                    self.agent_type,
                    operation_type,
                    timer.execution_time
                )
                
                return result
            except Exception as e:
                # Update error stats
                self.state = AgentState.ERROR
                self.error_count += 1
                
                # Log the error
                logger.exception(
                    f"Error in agent execution: {self.name}",
                    extra={
                        "agent_type": self.agent_type,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                
                # Track the error
                track_agent_error(self.agent_type, type(e).__name__)
                
                try:
                    # Try to handle the error
                    return await self.handle_error(e, context)
                except Exception as handler_error:
                    # If error handling fails, return a generic error result
                    logger.exception(
                        f"Error in error handler: {self.name}",
                        extra={
                            "agent_type": self.agent_type,
                            "error_type": type(handler_error).__name__,
                            "error_message": str(handler_error)
                        }
                    )
                    
                    # Return error result
                    error_info = {
                        "type": "unhandled_error",
                        "message": str(e),
                        "handler_error": str(handler_error)
                    }
                    return AgentResult.error_result(error_info, timer.execution_time)
    
    async def terminate(self) -> None:
        """Terminate the agent, releasing any resources.
        
        This method should be called when the agent is no longer needed.
        """
        self.state = AgentState.TERMINATED
        # Subclasses should override this method to release resources
        logger.info(
            f"Agent terminated: {self.name}",
            extra={"agent_type": self.agent_type}
        )
    
    async def __aenter__(self) -> "BaseAgent":
        """Context manager entry point."""
        if not await self.initialize():
            raise RuntimeError(f"Failed to initialize agent {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        await self.terminate()
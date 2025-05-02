from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.agent import AgentCapability


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
    mcp_enabled: bool = Field(default=False, description='Whether this agent instance uses the Model Context Protocol.')
    mcp_context_types: List[str] = Field(default_factory=list, description='List of primary MCP context types this agent interacts with.')

    @field_validator('agent_type')
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        if not v:
            raise ValueError('Agent type cannot be empty')
        return v.strip()

    @field_validator('capabilities', mode='before')
    @classmethod
    def validate_capabilities(cls, v: Union[Set[str], List[str], str]) -> Set[AgentCapability]:
        processed_set: Set[str]
        if isinstance(v, str):
            processed_set = {item.strip() for item in v.split(',') if item.strip()}
        elif isinstance(v, list):
            processed_set = set(v)
        elif isinstance(v, set):
            processed_set = v
        else:
            raise ValueError(f'Invalid input type for capabilities: {type(v)}. Expected str, list, or set.')
        valid_capabilities: Set[AgentCapability] = set()
        invalid_capabilities: Set[str] = set()
        for cap_str in processed_set:
            try:
                valid_capabilities.add(AgentCapability(str(cap_str).lower().strip()))
            except ValueError:
                invalid_capabilities.add(cap_str)
        if invalid_capabilities:
            valid_options = ', '.join([e.value for e in AgentCapability])
            raise ValueError(f'Invalid agent capabilities found: {', '.join(invalid_capabilities)}. Valid options are: {valid_options}')
        if not valid_capabilities:
            raise ValueError('At least one valid agent capability must be specified.')
        return valid_capabilities
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True, validate_assignment=True)
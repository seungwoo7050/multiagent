"""
Focused test to verify MCPExecutorAgent response handling from LLM adapter.

This test specifically examines how the executor agent processes different 
response formats from the LLM adapter to identify issues in the format handling.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.config import AgentConfig
from src.agents.mcp_executor import MCPExecutorAgent
from src.core.agent import AgentContext
from src.core.task import BaseTask, TaskState, TaskPriority
from src.tools.registry import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.datetime_tool import DateTimeTool

@pytest.fixture
def test_task():
    """Create a sample task with a plan."""
    task = MagicMock(spec=BaseTask)
    task.id = "test-task-id"
    task.type = "execute"
    task.state = TaskState.PENDING
    task.priority = TaskPriority.NORMAL
    task.input = {
        "goal": "Execute a plan",
        "plan": [
            {
                "step": 1,
                "action": "calculator",
                "args": {"expression": "2 + 2"},
                "reasoning": "Need to perform a simple calculation"
            },
            {
                "step": 2,
                "action": "datetime",
                "args": {"operation": "current"},
                "reasoning": "Need to check the current time"
            },
            {
                "step": 3,
                "action": "finish",
                "args": {"answer": "Here's the calculation result and current time."},
                "reasoning": "Providing final answer to the user"
            }
        ]
    }
    return task

@pytest.fixture
def tool_registry():
    """Create a tool registry with calculator and datetime tools."""
    registry = ToolRegistry()
    registry.register(CalculatorTool)
    registry.register(DateTimeTool)
    return registry

@pytest.fixture
def executor_agent(tool_registry):
    """Create an executor agent for testing."""
    config = AgentConfig(
        name="test_executor",
        agent_type="mcp_executor",
        model="gpt-3.5-turbo",
        capabilities={"execution", "tool_use"},
        allowed_tools=["calculator", "datetime"],
        parameters={
            "temperature": 0.2,
            "max_react_iterations": 3
        }
    )
    
    agent = MCPExecutorAgent(
        config=config,
        tool_registry=tool_registry
    )
    return agent

@pytest.mark.asyncio
class TestExecutorResponseHandling:
    """Tests for verifying executor response handling from LLM adapter."""
    
    async def test_execute_with_action_format(self, executor_agent, test_task):
        """Test executor agent with properly formatted action responses."""
        # Create context
        agent_context = AgentContext(
            task=test_task,
            trace_id="test-trace-id",
            tools=["calculator", "datetime"]
        )
        
        # Initialize agent
        await executor_agent.initialize()
        
        try:
            # Patch the LLM adapter.process_with_mcp method
            with patch("src.core.mcp.adapters.llm_adapter.LLMAdapter.process_with_mcp") as mock_process:
                # Set up responses for each iteration - proper format
                # Set up responses for each iteration - both thoughts and actions
                responses = [
                    # First iteration
                    AsyncMock(success=True, result_text="I should perform the calculation first", model_used="gpt-3.5-turbo"),
                    AsyncMock(success=True, result_text="calculator[{\"expression\": \"2 + 2\"}]", model_used="gpt-3.5-turbo"),
                    
                    # Second iteration
                    AsyncMock(success=True, result_text="Now I'll check the current time", model_used="gpt-3.5-turbo"),
                    AsyncMock(success=True, result_text="datetime[{\"operation\": \"current\"}]", model_used="gpt-3.5-turbo"),
                    
                    # Third iteration
                    AsyncMock(success=True, result_text="I'll provide the final answer now", model_used="gpt-3.5-turbo"),
                    AsyncMock(success=True, result_text="finish[{\"answer\": \"The result is 4 and here's the current time.\"}]", model_used="gpt-3.5-turbo")
                ]
                
                # Set up the mock to return different responses for each call
                mock_process.side_effect = responses
                
                # Patch the tool runner to return success
                with patch.object(executor_agent, "tool_runner") as mock_tool_runner:
                    mock_tool_runner.run_tool = AsyncMock(return_value={
                        "status": "success", 
                        "result": {"result": 4}
                    })
                    
                    # Execute the agent
                    result = await executor_agent.process(agent_context)
                    
                    # Verify success
                    assert result.success is True
                    assert "final_answer" in result.output
                    assert "The result is 4" in result.output["final_answer"]
                    
                    # Verify tool was called
                    assert mock_tool_runner.run_tool.call_count >= 1
        finally:
            # Clean up
            await executor_agent.terminate()
    
    async def test_execute_with_plan_format(self, executor_agent, test_task):
        """Test executor agent with incorrectly formatted plan responses."""
        # Create context
        agent_context = AgentContext(
            task=test_task,
            trace_id="test-trace-id",
            tools=["calculator", "datetime"]
        )
        
        # Initialize agent
        await executor_agent.initialize()
        
        try:
            # Patch the LLM adapter
            with patch("src.core.mcp.adapters.llm_adapter.LLMAdapter.process_with_mcp") as mock_process:
                # Return plan format (incorrect) for each iteration
                plan_response = AsyncMock(
                    success=True,
                    result_text=json.dumps({
                        "plan": [
                            {
                                "step": 1,
                                "action": "calculator",
                                "args": {"expression": "2 + 2"},
                                "reasoning": "Need to perform a simple calculation"
                            },
                            {
                                "step": 2,
                                "action": "datetime",
                                "args": {"operation": "current"},
                                "reasoning": "Need to check the current time"
                            },
                            {
                                "step": 3,
                                "action": "finish",
                                "args": {"answer": "Here's the calculation result and current time."},
                                "reasoning": "Providing final answer to the user"
                            }
                        ]
                    }),
                    model_used="gpt-3.5-turbo"
                )
                
                # Set all responses to return the same plan format (incorrect)
                mock_process.side_effect = [plan_response, plan_response, plan_response]
                
                # Patch the tool runner
                with patch.object(executor_agent, "tool_runner") as mock_tool_runner:
                    mock_tool_runner.run_tool = AsyncMock(return_value={
                        "status": "success", 
                        "result": {"result": 4}
                    })
                    
                    # Execute the agent
                    result = await executor_agent.process(agent_context)
                    
                    # Verify execution still completes but with limited success
                    assert result is not None
                    
                    # Check if any tools were actually called
                    assert mock_tool_runner.run_tool.call_count == 0
                    
                    # Log information about what happened
                    print(f"Response success: {result.success}")
                    print(f"Response output keys: {result.output.keys() if result.output else 'No output'}")
                    if "scratchpad" in result.output:
                        print(f"Scratchpad entries: {len(result.output['scratchpad'])}")
        finally:
            # Clean up
            await executor_agent.terminate()
    
    async def test_inspect_llm_adapter_interaction(self, executor_agent, test_task):
        """Test to inspect the exact interaction between executor and LLM adapter."""
        # Create context
        agent_context = AgentContext(
            task=test_task,
            trace_id="test-trace-id",
            tools=["calculator", "datetime"]
        )
        
        # Initialize agent
        await executor_agent.initialize()
        
        try:
            # Patch the LLM adapter to see what's being sent to it
            with patch("src.core.mcp.adapters.llm_adapter.LLMAdapter.process_with_mcp") as mock_process:
                # Return a successful response to avoid errors
                mock_process.return_value = AsyncMock(
                    success=True,
                    result_text="calculator[{\"expression\": \"2 + 2\"}]",
                    model_used="gpt-3.5-turbo"
                )
                
                # Execute with minimal steps to inspect
                with patch.object(executor_agent, "max_react_iterations", 1):
                    await executor_agent.process(agent_context)
                    
                    # Check what was sent to the LLM adapter
                    assert mock_process.call_count >= 1
                    
                    # Get the first call args
                    first_call = mock_process.call_args_list[0][0][0] if mock_process.call_args_list else None
                    
                    # Print the first call details for debugging
                    if first_call:
                        print("\nLLM Adapter Input Details:")
                        print(f"Context type: {type(first_call)}")
                        print(f"Context properties: {dir(first_call)}")
                        
                        # Check if the context has attributes we expect
                        if hasattr(first_call, 'prompt'):
                            print(f"Prompt preview: {str(first_call.prompt)[:200]}...")
                        
                        if hasattr(first_call, 'model'):
                            print(f"Model: {first_call.model}")
        finally:
            # Clean up
            await executor_agent.terminate()
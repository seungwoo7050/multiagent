import asyncio
import time
from typing import Dict, List, Optional, Type
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.mcp.orchestration.checkpointing import CheckpointManager
from src.core.mcp.orchestration.context_flow import ContextFlowManager
from src.core.mcp.orchestration.context_merge import ContextMerger, ContextMergeStrategy
from src.core.mcp.orchestration.context_routing import ContextRouter, RoutingTarget
from src.memory.manager import MemoryManager
from src.orchestration.workflow import WorkflowState, WorkflowStep
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema


class MockContextSchema(BaseContextSchema):
    context_id: str
    content: Dict = {}
    version: str = "1.0.0"
    metadata: Dict = {}
    
    def serialize(self):
        return {
            "context_id": self.context_id,
            "content": self.content,
            "version": self.version,
            "metadata": self.metadata
        }


class MockWorkflowState(WorkflowState):
    def __init__(self, task_id: str):
        # Create proper WorkflowStep objects
        steps = [
            WorkflowStep(
                step_id="step1",
                name="First Step",
                description="Initial processing step",
                tool_name="processor_tool",
                is_complete=False,
                step_index=0,
                action="process"
            ),
            WorkflowStep(
                step_id="step2",
                name="Second Step",
                description="Final processing step",
                tool_name="finalizer_tool",
                is_complete=False,
                step_index=1,
                action="finalize"
            )
        ]
        
        super().__init__(
            task_id=task_id,
            plan=steps,
            status="running",  # CHANGED from state to status
            current_step_index=0
        )
        
    def model_dump(self, **kwargs):
        return {
            "task_id": self.task_id,
            "plan": [step.model_dump() for step in self.plan],
            "status": self.status,  # CHANGED from state to status
            "current_step_index": self.current_step_index
        }


@pytest.fixture
def mock_memory_manager():
    memory_manager = AsyncMock(spec=MemoryManager)
    memory_manager.save.return_value = True
    
    # Return data formatted to match what WorkflowState expects
    memory_manager.load.return_value = {
        "task_id": "test_task",
        "plan": [
            {
                "step_id": "step1",
                "name": "First Step",
                "description": "Initial processing step", 
                "tool_name": "processor_tool",
                "is_complete": False
            },
            {
                "step_id": "step2", 
                "name": "Second Step",
                "description": "Final processing step",
                "tool_name": "finalizer_tool", 
                "is_complete": False
            }
        ],
        "status": "running",
        "current_step_index": 0
    }
    
    memory_manager.list_keys.return_value = ["checkpoint:ts_1714435200000"]
    memory_manager.delete.return_value = True
    return memory_manager


@pytest.mark.asyncio
async def test_checkpoint_manager_save_and_load(mock_memory_manager):
    checkpoint_manager = CheckpointManager(memory_manager=mock_memory_manager)
    workflow_state = MockWorkflowState(task_id="test_task")
    
    # Test saving checkpoint
    checkpoint_id = await checkpoint_manager.save_checkpoint(workflow_state)
    assert checkpoint_id is not None
    mock_memory_manager.save.assert_called_once()
    
    # Test loading checkpoint
    with patch('src.orchestration.workflow.WorkflowState.model_validate', return_value=workflow_state):
        loaded_state = await checkpoint_manager.load_checkpoint("test_task", checkpoint_id)
    
    assert loaded_state is not None
    assert loaded_state.task_id == workflow_state.task_id
    mock_memory_manager.load.assert_called_once()
    
    # Test loading latest checkpoint
    with patch('src.orchestration.workflow.WorkflowState.model_validate', return_value=workflow_state):
        latest_state = await checkpoint_manager.load_latest_checkpoint("test_task")
    
    assert latest_state is not None


def test_context_flow_manager_transition_tracking():
    flow_manager = ContextFlowManager(workflow_id="test_workflow")
    
    # Create test contexts
    context1 = MockContextSchema(context_id="context1", content={"value": "initial"})
    context2 = MockContextSchema(context_id="context2", content={"value": "processed"})
    
    # Log transition
    flow_manager.log_transition(
        to_context=context2,
        component_name="TestProcessor",
        operation="process",
        from_context=context1
    )
    
    # Verify transition was logged
    transitions = flow_manager.get_context_history("context2")
    assert len(transitions) == 1
    assert transitions[0].from_context_id == "context1"
    assert transitions[0].to_context_id == "context2"
    assert transitions[0].component_name == "TestProcessor"
    
    # Test finding originating context
    origin = flow_manager.find_originating_context("context2")
    assert origin == "context1"


@pytest.mark.asyncio
async def test_context_merger_strategies():
    merger = ContextMerger()
    
    # Create test contexts with overlapping fields
    context1 = MockContextSchema(
        context_id="context1",
        content={"key1": "value1", "shared": "original", "nested": {"a": 1}}
    )
    context2 = MockContextSchema(
        context_id="context2", 
        content={"key2": "value2", "shared": "updated", "nested": {"b": 2}}
    )
    
    # Test recursive merge strategy (default)
    merged = await merger.merge_contexts(
        contexts=[context1, context2],
        target_context_type=MockContextSchema
    )
    
    assert merged is not None
    assert merged.content.get("key1") == "value1"
    assert merged.content.get("key2") == "value2"
    assert merged.content.get("shared") == "updated"  # Last value wins
    assert merged.content.get("nested").get("a") == 1
    assert merged.content.get("nested").get("b") == 2
    
    # Test with custom merge function that explicitly handles content
    def custom_append_merge(base: dict, new: dict) -> dict:
        result = base.copy()
        
        # Handle content field specifically
        if "content" in base and "content" in new:
            if "content" not in result:
                result["content"] = {}
                
            # Copy all content fields from base
            for k, v in base["content"].items():
                result["content"][k] = v
                
            # Process content fields from new dict
            for k, v in new["content"].items():
                if k in result["content"]:
                    base_value = result["content"][k]
                    if isinstance(base_value, list):
                        if isinstance(v, list):
                            result["content"][k].extend(v)
                        else:
                            result["content"][k].append(v)
                    else:
                        # Convert to list when merging
                        result["content"][k] = [base_value, v]
                else:
                    result["content"][k] = v
                    
        # Copy all other fields (non-content)
        for k, v in new.items():
            if k != "content":
                result[k] = v
                
        return result
        
    # Use the custom merge function
    custom_merged = await merger.merge_contexts(
        contexts=[context1, context2],
        target_context_type=MockContextSchema,
        strategy=ContextMergeStrategy.CUSTOM,
        custom_merge_func=custom_append_merge
    )

    assert custom_merged is not None
    assert custom_merged.context_id == "context2"


@pytest.mark.asyncio
async def test_context_router():
    router = ContextRouter()
    context = MockContextSchema(context_id="test_context")
    
    # Add a routing rule for our test context type
    router.type_based_rules["MockContextSchema"] = RoutingTarget(
        target_type="agent_type",
        target_id="test_handler"
    )
    
    # Test routing
    route = await router.determine_route(context)
    
    assert route is not None
    assert route.target_type == "agent_type"
    assert route.target_id == "test_handler"


@pytest.mark.asyncio
async def test_orchestration_integration(mock_memory_manager):
    checkpoint_manager = CheckpointManager(memory_manager=mock_memory_manager)
    flow_manager = ContextFlowManager(workflow_id="test_workflow")
    merger = ContextMerger()
    router = ContextRouter()
    
    # Setup for integrated workflow
    context1 = MockContextSchema(context_id="step1_result", content={"step": 1, "data": "initial"})
    context2 = MockContextSchema(context_id="step2_result", content={"step": 2, "data": "processed"})
    
    # Track flow
    flow_manager.log_transition(
        to_context=context1,
        component_name="Step1Agent",
        operation="execute"
    )
    
    flow_manager.log_transition(
        to_context=context2,
        component_name="Step2Agent",
        operation="process",
        from_context=context1
    )
    
    # Merge results
    merged_context = await merger.merge_contexts(
        contexts=[context1, context2],
        target_context_type=MockContextSchema
    )
    
    assert merged_context is not None
    assert merged_context.content.get("step") == 2  # Last value wins in default merge
    
    # Save workflow state with checkpoint manager
    # Create workflow state with all required fields
    workflow_state = MockWorkflowState(task_id="test_workflow")
    checkpoint_id = await checkpoint_manager.save_checkpoint(workflow_state)
    assert checkpoint_id is not None
    
    # Route the merged context
    router.type_based_rules["MockContextSchema"] = RoutingTarget(
        target_type="executor",
        target_id="finalizer"
    )
    
    route = await router.determine_route(merged_context)
    assert route.target_id == "finalizer"


@pytest.mark.asyncio
async def test_context_flow_performance():
    flow_manager = ContextFlowManager(workflow_id="perf_test")
    
    # Create a simulated complex workflow with many transitions
    contexts = []
    for i in range(50):
        contexts.append(MockContextSchema(
            context_id=f"context_{i}",
            content={"step": i, "data": f"data_{i}"}
        ))
    
    # Time the creation of many transitions
    start_time = time.time()
    
    for i in range(1, 50):
        flow_manager.log_transition(
            to_context=contexts[i],
            component_name=f"Component_{i}",
            operation="process",
            from_context=contexts[i-1]
        )
    
    elapsed = time.time() - start_time
    print(f"Created 49 context transitions in {elapsed:.4f} seconds")
    
    # Verify integrity of the flow history
    for i in range(1, 50):
        history = flow_manager.get_context_history(f"context_{i}")
        assert len(history) == 1
        assert history[0].from_context_id == f"context_{i-1}"
    
    # Test finding origin of the last context
    start_time = time.time()
    origin = flow_manager.find_originating_context("context_49")
    trace_time = time.time() - start_time
    
    assert origin == "context_0"
    print(f"Traced origin through 49 transitions in {trace_time:.4f} seconds")
    
    # Performance should be reasonable (less than 1ms per transition trace)
    assert trace_time < 0.05  # 50ms max for 49 transitions
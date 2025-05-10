#!/usr/bin/env python3
"""
Task Division Testing Script for Multi-Agent System
Tests task division workflow with direct orchestrator calls
"""

import os
import asyncio
import uuid
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Project setup
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception as path_setup_error:
    print(f"ERROR: Error setting up sys.path: {path_setup_error}", file=sys.stderr)

# Core imports
from src.config.settings import settings
from src.utils.logger import get_logger, setup_logging
from src.config.connections import setup_connection_pools, cleanup_connection_pools

from src.services.llm_client import LLMClient
from src.memory.memory_manager import get_memory_manager
from src.services.tool_manager import get_tool_manager
from src.services.notification_service import NotificationService
from src.agents.orchestrator import Orchestrator

logger = get_logger(__name__)

class TaskDivisionTester:
    def __init__(self):
        logger.info("Initializing TaskDivisionTester...")
        self.llm_client = LLMClient()
        self.memory_manager = get_memory_manager()
        self.tool_manager = get_tool_manager('global_tools')
        self.notification_service = NotificationService()
        
        self.orchestrator = Orchestrator(
            llm_client=self.llm_client,
            memory_manager=self.memory_manager,
            tool_manager=self.tool_manager,
            notification_service=self.notification_service
        )

    async def _run_workflow(self, task_name: str, test_input: str) -> str:
        """Run a workflow with the given input and return the task_id"""
        task_id = f"taskdiv-{uuid.uuid4()}"
        
        print(f"\n{'='*80}")
        print(f"TESTING TASK DIVISION: {task_name}")
        print(f"Task ID: {task_id}")
        print(f"{'='*80}")
        print(f"INPUT:\n{test_input}\n")
        print("Starting workflow execution...\n")
        
        try:
            # Run the workflow (task division is hardcoded)
            final_state = await self.orchestrator.run_workflow(
                graph_config_name="task_division_workflow",
                task_id=task_id,
                original_input=test_input,
                initial_metadata={"test_name": task_name}
            )
            
            if final_state:
                print(f"\nFINAL ANSWER:\n{final_state.final_answer or 'No final answer produced'}\n")
                if final_state.error_message:
                    print(f"ERROR: {final_state.error_message}")
                
                # Display subtask results if available
                if hasattr(final_state, 'dynamic_data') and final_state.dynamic_data and "subtasks" in final_state.dynamic_data:
                    subtasks = final_state.dynamic_data["subtasks"]
                    print(f"\n----- Subtask Results ({len(subtasks)}) -----")
                    for i, subtask in enumerate(subtasks):
                        print(f"\nSubtask {i+1}: {subtask.get('title', 'Untitled')}")
                        print(f"Complexity: {'Complex (ToT)' if subtask.get('is_complex') else 'Simple (GenericLLM)'}")
                        print(f"Result: {subtask.get('result', 'No result')[:150]}...")
            else:
                print("\nERROR: No final state returned")
                
            return task_id
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}", exc_info=True)
            print(f"\nERROR: Exception during workflow: {str(e)}")
            return task_id

    async def run_tests(self):
        """Run a series of tests with different inputs"""
        
        test_inputs = [
            {
                "name": "AI Course Design",
                "input": "Design a comprehensive AI course for undergraduate students including lectures, exercises, assessments, and final projects."
            },
            # {
            #     "name": "Marketing Strategy",
            #     "input": "Create a marketing strategy for a new fitness app targeting working professionals aged 25-40. Include social media, content marketing, and partnership approaches."
            # },
            # {
            #     "name": "Research Analysis",
            #     "input": "Analyze the recent developments in quantum computing and their potential impact on cryptography and data security over the next decade."
            # }
        ]
        
        for test in test_inputs:
            await self._run_workflow(test["name"], test["input"])
            print("\nWaiting before running next test...\n")
            await asyncio.sleep(2)  # Brief pause between tests


async def main():
    # Setup logging and connections
    try:
        logger.info("Setting up logging...")
        setup_logging(settings)
        
        logger.info("Setting up connection pools...")
        await setup_connection_pools()
    except Exception as e:
        logger.critical(f"Failed to setup: {e}", exc_info=True)
        return
    
    tester = None
    try:
        tester = TaskDivisionTester()
        await tester.run_tests()
        
    except Exception as e:
        logger.critical(f"Tests failed with error: {e}", exc_info=True)
    finally:
        try:
            await cleanup_connection_pools()
        except Exception as e:
            logger.error(f"Error cleaning up: {e}", exc_info=True)
        
        logger.info("Task division tests completed")

if __name__ == "__main__":
    print("Starting task division test script...")
    asyncio.run(main())
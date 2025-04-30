import json
import asyncio
import re
import time
from typing import Any, Dict, Optional, cast, List, Tuple
from src.core.agent import BaseAgent, AgentContext as CoreAgentContext, AgentResult, AgentState
from src.agents.config import AgentConfig
from src.agents.context_manager import AgentContextManager
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import BaseContextSchema, TaskContext
from src.core.mcp.adapters.llm_adapter import LLMInputContext, LLMOutputContext, LLMAdapter
from src.tools.runner import ToolRunner
from src.tools.registry import ToolRegistry
from src.config.logger import get_logger_with_context, ContextLoggerAdapter
from src.core.exceptions import AgentExecutionError, TaskError
from src.prompts.templates import PromptTemplate
from src.config.settings import get_settings
from src.config.errors import ErrorCode, LLMError, ToolError
from src.core.task import TaskResult

settings = get_settings()
logger: ContextLoggerAdapter = get_logger_with_context(__name__)

class MCPExecutorAgent(BaseAgent):

    def __init__(self, config: AgentConfig, tool_registry: Optional[ToolRegistry]=None):
        if config.agent_type not in ['mcp_executor', 'executor']:
            logger.warning(f'MCPExecutorAgent initialized with potentially mismatched config type: {config.agent_type}')
        super().__init__(config)
        self.context_manager = AgentContextManager(agent_id=self.config.name)
        self.llm_adapter = LLMAdapter()
        self.tool_runner = ToolRunner()
        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self.max_react_iterations = self.config.parameters.get('max_react_iterations', 5)

    async def initialize(self) -> bool:
        self.state = AgentState.INITIALIZING
        logger.info(f'Initializing MCPExecutorAgent: {self.name}')
        self.state = AgentState.IDLE
        logger.info(f'MCPExecutorAgent {self.name} initialized successfully.')
        return True

    async def process(self, context: CoreAgentContext) -> AgentResult:
        # Update logger with context
        global logger
        logger = get_logger_with_context(
            __name__, 
            agent_id=self.config.name, 
            task_id=getattr(context.task, 'id', None), 
            trace_id=context.trace_id
        )
        
        task_id_log = getattr(context.task, 'id', 'N/A')
        logger.info(f'MCPExecutorAgent {self.name} starting execution for task {task_id_log}')
        
        # Validate task input contains required plan
        if not context.task or not context.task.input or 'plan' not in context.task.input or (not isinstance(context.task.input['plan'], list)):
            logger.error('No valid plan (list of steps) found in the task input.')
            raise TaskError("Execution requires a 'plan' (list of steps) in the task input.", task_id=task_id_log)
        
        # Extract plan and goal from task
        plan: List[Dict[str, Any]] = context.task.input['plan']
        goal: str = context.task.input.get('goal', 'No goal specified.')
        logger.debug(f'Received plan with {len(plan)} steps for goal: {goal}')
        
        # Initialize ReAct loop variables
        scratchpad: str = ''
        final_answer: Optional[str] = None
        executed_steps: int = 0
        start_time = time.time()  # Track execution time
        
        # Begin ReAct loop
        for i in range(self.max_react_iterations):
            iteration = i + 1
            logger.info(f'ReAct Iteration {iteration}/{self.max_react_iterations}')
            
            # Generate thought
            thought: str = await self._generate_thought(goal, plan, scratchpad, executed_steps)
            scratchpad += f'Thought: {thought}\n'
            logger.debug(f'Iteration {iteration} Thought: {thought}')
            
            # Generate action
            action_str: str = await self._generate_action(goal, plan, scratchpad, executed_steps)
            scratchpad += f'Action: {action_str}\n'
            logger.debug(f'Iteration {iteration} Action String: {action_str}')
            
            # Parse and execute action
            action_type, action_input = self._parse_action(action_str)
            logger.info(f'Iteration {iteration} Parsed Action - Type: {action_type}, Input: {action_input}')
            
            # Handle different action types
            if action_type.lower() == 'finish':
                final_answer = action_input.get('answer', 'Execution finished.')
                logger.info(f'Finish action received. Final Answer: {final_answer}')
                break
            elif action_type.lower() == 'think':
                observation = 'Acknowledged internal thought process. Continuing evaluation.'
            elif self.tool_registry.has(action_type):
                logger.info(f'Executing tool: {action_type} with args: {action_input}')
                try:
                    tool_result: Dict[str, Any] = await self.tool_runner.run_tool(
                        tool=action_type, 
                        tool_registry=self.tool_registry, 
                        args=action_input, 
                        trace_id=context.trace_id
                    )
                    
                    if tool_result.get('status') == 'success':
                        observation = json.dumps(tool_result.get('result', 'Tool executed successfully.'))
                    else:
                        observation = f'Error executing tool {action_type}: {tool_result.get("error", {}).get("message", "Unknown tool error")}'
                        logger.warning(f'Tool execution failed: {observation}')
                except ToolError as tool_err:
                    observation = f'Tool Error during execution ({tool_err.code}): {tool_err.message}'
                    logger.error(f'Tool execution resulted in ToolError: {observation}', exc_info=True)
                except Exception as tool_run_e:
                    observation = f'Unexpected error running tool {action_type}: {str(tool_run_e)}'
                    logger.exception(f'Unexpected error during tool execution {action_type}')
            else:
                observation = f"Error: Unknown action type '{action_type}'. Valid actions are 'finish', 'think', or one of the available tools: {list(self.tool_registry.get_names())}"
                logger.error(observation)
            
            # Record observation in scratchpad
            scratchpad += f'Observation: {observation}\n'
            logger.debug(f'Iteration {iteration} Observation: {observation}')
            executed_steps += 1
        else:
            # This executes if the for loop completes without a break (max iterations reached)
            logger.warning(f"ReAct loop reached max iterations ({self.max_react_iterations}) without a 'finish' action.")
            final_answer = f'Execution stopped after reaching maximum iterations ({self.max_react_iterations}). Final observation: {(observation if "observation" in locals() else "N/A")}'
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare agent output
        agent_output: Dict[str, Any] = {
            'final_answer': final_answer, 
            'scratchpad': scratchpad, 
            'iterations': executed_steps
        }
        
        # Prepare task result
        task_result: Optional[TaskResult] = None
        
        return AgentResult.success_result(
            output=agent_output, 
            execution_time=execution_time,
            task_result=task_result, 
            metadata={'agent_name': self.config.name}
        )

    async def _generate_thought(self, goal: str, plan: List[Dict[str, Any]], scratchpad: str, executed_steps: int) -> str:
        """Generate a thought using LLM based on current execution state"""
        prompt = f"""
        You are an Executor Agent following a plan to achieve a goal.
        Goal: {goal}
        Plan: {json.dumps(plan, indent=2)}
        Execution History (Scratchpad):
        {scratchpad}

        Based on the goal, the overall plan, and the execution history (especially the last Observation), what is your reasoning and thought process for the *next* action you should take?
        Consider the current step based on 'executed_steps' ({executed_steps}) and the remaining plan.
        Is the last observation what you expected? Does it mean the last action succeeded or failed?
        What is the *most logical* next step according to the plan and the current situation?
        Keep your thought concise and focused on determining the next action.

        Thought:"""
        
        llm_input_context = LLMInputContext(
            model=self.config.model or settings.PRIMARY_LLM, 
            messages=[{'role': 'user', 'content': prompt}], 
            parameters={
                'max_tokens': 150, 
                'temperature': 0.5, 
                'stop_sequences': ['\nAction:']
            }, 
            use_cache=False
        )
        
        try:
            llm_output_context = await self.llm_adapter.process_with_mcp(llm_input_context)
            if llm_output_context.success and llm_output_context.result_text:
                thought = llm_output_context.result_text.strip()
                if thought.lower().startswith('thought:'):
                    thought = thought[len('thought:'):].strip()
                return thought
            else:
                error_msg = llm_output_context.error_message or 'Unknown LLM error'
                logger.warning(f'LLM failed to generate thought: {error_msg}')
                return f'Error: Could not generate thought. LLM Error: {error_msg}'
        except Exception as e:
            logger.error(f'Error generating thought via LLM adapter: {e}', exc_info=True)
            return f'Error: Exception during thought generation - {str(e)}'

    async def _generate_action(self, goal: str, plan: List[Dict[str, Any]], scratchpad: str, executed_steps: int) -> str:
        """Generate an action using LLM based on current execution state"""
        available_tools = list(self.tool_registry.get_names())
        prompt = f"""
        You are an Executor Agent following a plan.
        Goal: {goal}
        Plan: {json.dumps(plan, indent=2)}
        Execution History (Scratchpad):
        {scratchpad}
        Available Tools: {available_tools}

        Based on your last thought and the overall progress, what is the *exact next action* to take?
        Your action *must* be in one of the following formats:
        1. Tool Call: `ToolName[Input JSON]` (e.g., `web_search[{{"query": "latest AI news"}}]`) Use a tool from the Available Tools list. Input JSON must be valid.
        2. Internal Thought: `think[]` (Use this if you need to reflect or plan internally without an external action).
        3. Finish Execution: `finish[Output JSON]` (e.g., `finish[{{"answer": "The capital of France is Paris."}}]`) Use this only when the goal is fully achieved based on the observations. Output JSON should contain the final answer.

        Analyze the last Thought in the scratchpad and choose the most logical next Action based on the plan and available tools.
        Output *only* the Action string in the specified format, starting with the action name, followed by square brackets containing the JSON input (or empty for 'think').

        Action:"""
        
        llm_input_context = LLMInputContext(
            model=self.config.model or settings.PRIMARY_LLM, 
            messages=[{'role': 'user', 'content': prompt}], 
            parameters={
                'max_tokens': 150, 
                'temperature': 0.3, 
                'stop_sequences': ['\nObservation:']
            }, 
            use_cache=False
        )
        
        try:
            llm_output_context = await self.llm_adapter.process_with_mcp(llm_input_context)
            if llm_output_context.success and llm_output_context.result_text:
                action_str = llm_output_context.result_text.strip()
                if action_str.lower().startswith('action:'):
                    action_str = action_str[len('action:'):].strip()
                
                # Validate action format but continue even if it doesn't fully match
                if not re.match(r'^[a-zA-Z0-9_]+\s*\[.*?\]?\s*$', action_str):
                    logger.warning(f"LLM generated action in unexpected format: '{action_str}'. Attempting to use anyway.")
                
                return action_str
            else:
                error_msg = llm_output_context.error_message or 'Unknown LLM error'
                logger.warning(f'LLM failed to generate action: {error_msg}')
                return f'finish[{{"answer": "Error: Could not generate next action due to LLM failure: {error_msg}"}}]'
        except Exception as e:
            logger.error(f'Error generating action via LLM adapter: {e}', exc_info=True)
            return f'finish[{{"answer": "Error: Exception during action generation - {str(e)}"}}]'

    def _parse_action(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """Parse action string into action type and arguments"""
        match = re.match(r'^\s*([a-zA-Z0-9_]+)\s*\[(.*?)\]?\s*$', action_str, re.DOTALL)
        if not match:
            logger.warning(f"Could not parse action string: '{action_str}'. Defaulting to 'think'.")
            return ('think', {})
        
        action_type: str = match.group(1).strip()
        input_str: str = match.group(2).strip()
        
        if not input_str:
            if action_type.lower() == 'think':
                return ('think', {})
            else:
                logger.debug(f"Action '{action_type}' called with empty arguments '[]'.")
                return (action_type, {})
        
        try:
            action_input: Any = json.loads(input_str)
            if not isinstance(action_input, dict):
                logger.warning(f"Action input for '{action_type}' is valid JSON but not an object/dict: '{input_str}'. Wrapping in 'value' key.")
                action_input = {'value': action_input}
            return (action_type, action_input)
        except json.JSONDecodeError:
            logger.warning(f"Action input is not valid JSON: '{input_str}'. Treating as raw string argument with key 'input'.")
            return (action_type, {'input': input_str})

    async def handle_error(self, error: Exception, context: CoreAgentContext) -> AgentResult:
        """Handle execution errors and return appropriate result"""
        logger.error(f'MCPExecutorAgent {self.name} handling error during execution: {error}', exc_info=True)
        
        error_details: Dict[str, Any] = {
            'message': str(error), 
            'type': type(error).__name__, 
            'agent_name': self.config.name, 
            'agent_type': self.config.agent_type
        }
        
        if hasattr(error, 'details'):
            original_details = getattr(error, 'details')
            error_details['details'] = original_details if isinstance(original_details, dict) else str(original_details)
        
        return AgentResult.error_result(
            error=error_details, 
            execution_time=0,  # Consider tracking the actual time spent before the error
            metadata={'agent_name': self.config.name}
        )
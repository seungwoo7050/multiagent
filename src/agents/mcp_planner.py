import json
import re
from typing import Any, Dict, Optional, cast, List
from src.core.agent import BaseAgent, AgentContext as CoreAgentContext, AgentResult, AgentState
from src.agents.config import AgentConfig
from src.agents.context_manager import AgentContextManager
from src.core.mcp.protocol import ContextProtocol
from src.core.mcp.schema import TaskContext, BaseContextSchema
from src.core.mcp.adapters.llm_adapter import LLMInputContext, LLMOutputContext, LLMAdapter
from src.config.logger import get_logger_with_context, ContextLoggerAdapter
from src.core.exceptions import AgentExecutionError, TaskError
from src.prompts.templates import PromptTemplate
from src.config.settings import get_settings
from src.config.errors import LLMError
from src.core.task import TaskResult  # Added missing import

# Get settings instance
settings = get_settings()
logger: ContextLoggerAdapter = get_logger_with_context(__name__)

class MCPPlannerAgent(BaseAgent):

    def __init__(self, config: AgentConfig, memory_manager: Optional[Any]=None):
        if config.agent_type not in ['mcp_planner', 'planner']:
            logger.warning(f'MCPPlannerAgent initialized with potentially mismatched config type: {config.agent_type}')
        super().__init__(config)
        self.context_manager = AgentContextManager(agent_id=self.config.name)
        self.llm_adapter = LLMAdapter()

    async def initialize(self) -> bool:
        self.state = AgentState.INITIALIZING
        logger.info(f'Initializing MCPPlannerAgent: {self.name}')
        self.state = AgentState.IDLE
        logger.info(f'MCPPlannerAgent {self.name} initialized successfully.')
        return True

    async def process(self, context: CoreAgentContext) -> AgentResult:
        global logger
        logger = get_logger_with_context(__name__, agent_id=self.config.name, task_id=getattr(context.task, 'id', None), trace_id=context.trace_id)
        task_id_log = getattr(context.task, 'id', 'N/A')
        logger.info(f'MCPPlannerAgent {self.name} starting process for task {task_id_log}')
        if context.task:
            try:
                task_mcp_context = TaskContext(task_id=context.task.id, task_type=context.task.type, input_data=context.task.input, metadata={'state': context.task.state.value, 'priority': context.task.priority.value})
                self.context_manager.update_context(task_mcp_context)
                logger.debug(f'Task context (ID: {task_mcp_context.context_id}) updated in context manager.')
            except Exception as task_ctx_e:
                logger.error(f'Failed to create TaskContext from BaseTask: {task_ctx_e}', exc_info=True)
                raise TaskError(f'Failed to process task data: {task_ctx_e}', task_id=task_id_log)
        else:
            logger.error('No task provided in CoreAgentContext for planning.')
            raise TaskError('Planning requires a task context.', task_id=task_id_log)
        
        # Get conversation history with fallback to placeholder
        conversation_history = context.get_memory('conversation_history', 'Placeholder: Previous conversation history...')
        logger.debug('Retrieved necessary context (using placeholder history).')
        
        prompt_data: Dict[str, Any] = {
            'goal': task_mcp_context.input_data.get('goal', 'No goal provided.'), 
            'available_tools': self.config.allowed_tools, 
            'conversation_history': conversation_history
        }
        
        try:
            planning_prompt = f"""\n            **Goal:** {prompt_data['goal']}\n\n            **Available Tools:** {(', '.join(prompt_data['available_tools']) if prompt_data['available_tools'] else 'None')}\n\n            **Conversation History (if relevant):**\n            {prompt_data['conversation_history']}\n\n            **Instructions:**\n            Based on the goal, conversation history, and available tools, create a step-by-step plan to achieve the goal.\n            The plan should be in JSON format, following this schema:\n            {{\n              "plan": [\n                {{ "step": 1, "action": "tool_name or think", "args": {{ "arg_name": "value" }}, "reasoning": "Why this step?" }},\n                ...\n              ]\n            }}\n            Choose appropriate tools from the list. If no tool is suitable, use the 'think' action (e.g., {{"action": "think", "args": {{"thought": "Plan next step based on info"}}}}). Provide clear reasoning for each step. Ensure the plan directly helps achieve the goal.\n            Output ONLY the JSON plan, nothing else.\n            """
            logger.debug('Generated planning prompt.')
        except Exception as e:
            logger.error(f'Failed to render planning prompt: {e}', exc_info=True)
            raise AgentExecutionError(f'Prompt template rendering error: {e}', agent_type=self.config.agent_type, agent_id=self.config.name)
        
        llm_input_context = LLMInputContext(
            model=self.config.model or settings.PRIMARY_LLM, 
            messages=[{'role': 'user', 'content': planning_prompt}], 
            parameters={
                'max_tokens': self.config.parameters.get('max_tokens', 1500), 
                'temperature': self.config.parameters.get('temperature', 0.5), 
                'top_p': self.config.parameters.get('top_p', 1.0)
            }, 
            use_cache=True
        )
        
        self.context_manager.update_context(llm_input_context)
        logger.info(f'Sending planning request to LLM (Model: {llm_input_context.model})')
        
        try:
            llm_output_context = await self.llm_adapter.process_with_mcp(llm_input_context)
            self.context_manager.update_context(llm_output_context)
        except LLMError as llm_e:
            logger.error(f'LLM call failed with LLMError: {llm_e}', exc_info=True)
            error_ctx = LLMOutputContext(
                success=False, 
                error_message=str(llm_e), 
                model_used=llm_input_context.model
            )
            self.context_manager.update_context(error_ctx)
            raise AgentExecutionError(
                f'LLM interaction failed: {llm_e}', 
                agent_type=self.config.agent_type, 
                agent_id=self.config.name
            )
        except Exception as e:
            logger.error(f'LLM call failed via adapter: {e}', exc_info=True)
            error_ctx = LLMOutputContext(
                success=False, 
                error_message=str(e), 
                model_used=llm_input_context.model
            )
            self.context_manager.update_context(error_ctx)
            raise AgentExecutionError(
                f'LLM interaction failed: {e}', 
                agent_type=self.config.agent_type, 
                agent_id=self.config.name
            )
            
        if not llm_output_context.success or not llm_output_context.result_text:
            error_msg = llm_output_context.error_message or 'LLM generation failed without specific error.'
            logger.error(f'LLM generation failed: {error_msg}')
            raise AgentExecutionError(
                f'LLM generation failed: {error_msg}', 
                agent_type=self.config.agent_type, 
                agent_id=self.config.name
            )
            
        raw_plan_text = llm_output_context.result_text
        logger.debug(f'Received raw plan from LLM: {raw_plan_text[:200]}...')
        
        try:
            parsed_plan: Optional[Dict[str, Any]] = None
            json_match = False
            
            # Try to parse as direct JSON first
            try:
                parsed_plan = json.loads(raw_plan_text)
                json_match = True
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_plan_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    try:
                        parsed_plan = json.loads(json_str)
                        json_match = True
                        logger.debug('Parsed plan from JSON code block.')
                    except json.JSONDecodeError:
                        pass
                else:
                    # Try to extract JSON by finding braces
                    start = raw_plan_text.find('{')
                    end = raw_plan_text.rfind('}')
                    if start != -1 and end != -1 and (start < end):
                        json_str = raw_plan_text[start:end + 1]
                        try:
                            parsed_plan = json.loads(json_str)
                            json_match = True
                            logger.debug('Parsed plan from direct brace matching.')
                        except json.JSONDecodeError:
                            pass
                            
            # Validate parsed result
            if not json_match or parsed_plan is None:
                raise json.JSONDecodeError('No valid JSON plan found in LLM response.', raw_plan_text, 0)
                
            if not isinstance(parsed_plan, dict) or 'plan' not in parsed_plan or (not isinstance(parsed_plan['plan'], list)):
                raise ValueError("Invalid plan structure: 'plan' key with a list of steps is required.")
                
            # Validate each step in the plan
            for step in parsed_plan['plan']:
                if not isinstance(step, dict) or not all((k in step for k in ['step', 'action', 'args', 'reasoning'])):
                    raise ValueError(f'Invalid step structure in plan: {step}. Missing required keys (step, action, args, reasoning).')
                    
            logger.info(f'Successfully parsed and validated plan with {len(parsed_plan["plan"])} steps.')
            
            # Create successful result
            agent_output: Dict[str, Any] = {
                'plan': parsed_plan['plan'], 
                'raw_llm_response': raw_plan_text, 
                'llm_usage': llm_output_context.usage
            }
            
            # Use appropriate TaskResult type
            task_result: Optional[TaskResult] = None
            
            return AgentResult.success_result(
                output=agent_output, 
                execution_time=0,  # Consider using a timer to track actual execution time
                task_result=task_result, 
                metadata={
                    'agent_name': self.config.name, 
                    'model_used': llm_output_context.model_used
                }
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Handle parsing errors with detailed logging
            logger.error(
                f'Failed to parse or validate plan JSON: {e}\nRaw response snippet: {raw_plan_text[:500]}', 
                exc_info=True
            )
            raise AgentExecutionError(
                f'Failed to process LLM plan output: {e}. Raw Response Snippet: {raw_plan_text[:500]}...', 
                agent_type=self.config.agent_type, 
                agent_id=self.config.name, 
                details={'raw_output': raw_plan_text}
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f'Unexpected error during plan processing: {e}')
            raise AgentExecutionError(
                f'Unexpected error processing plan: {e}', 
                agent_type=self.config.agent_type, 
                agent_id=self.config.name, 
                original_error=e
            )

    async def handle_error(self, error: Exception, context: CoreAgentContext) -> AgentResult:
        logger.error(f'MCPPlannerAgent {self.name} handling error during planning: {error}', exc_info=True)
        error_details: Dict[str, Any] = {
            'message': str(error), 
            'type': type(error).__name__, 
            'agent_name': self.config.name, 
            'agent_type': self.config.agent_type
        }
        
        if hasattr(error, 'details'):
            error_details['details'] = getattr(error, 'details')
            
        return AgentResult.error_result(
            error=error_details, 
            execution_time=0, 
            metadata={'agent_name': self.config.name}
        )
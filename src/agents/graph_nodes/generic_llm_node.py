# src/agents/graph_nodes/generic_llm_node.py

import os
import builtins as _bt
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.config.errors import ToolError
from src.services.llm_client import LLMClient
from src.services.tool_manager import ToolManager
# AgentGraphState의 정확한 위치 확인 필요 (여기서는 schemas.mcp_models로 가정)
from src.schemas.mcp_models import AgentGraphState

_bt.os = os
logger = get_logger(__name__)
settings = get_settings()

# LLM 응답 파싱 결과를 위한 타입 정의 (선택 사항)
class ParsedLLMResponse(TypedDict):
    action: str
    action_input: Union[Dict[str, Any], str]

class GenericLLMNode:
    """
    설정 가능한 LLM 호출을 수행하고, enable_tool_use 설정에 따라 동적으로 도구를 호출하는 LangGraph 노드.
    도구 사용 시 ReAct (Reasoning-Action) 패턴의 일부를 구현합니다.
    """
    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: ToolManager, # ToolManager 주입 받음
        prompt_template_path: str,
        output_field_name: Optional[str] = None,
        input_keys_for_prompt: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        node_id: str = "generic_llm_node",
        max_react_iterations: int = 5,
        # 도구 사용 관련 설정
        enable_tool_use: bool = False, # 기본값을 False로 변경 (명시적 활성화 필요)
        allowed_tools: Optional[List[str]] = None
    ):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.prompt_template_path = prompt_template_path
        self.output_field_name = output_field_name
        self.input_keys_for_prompt = input_keys_for_prompt or []
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.node_id = node_id
        self.max_react_iterations = max_react_iterations
        self.enable_tool_use = enable_tool_use # 전달된 값 사용
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None

        self.prompt_template_str = self._load_prompt_template()
        # 프롬프트 변수 설정 시 enable_tool_use 고려
        prompt_vars = list(set(self.input_keys_for_prompt))
        if self.enable_tool_use:
            prompt_vars.extend(['available_tools', 'tool_call_history', 'scratchpad'])

        self.prompt_template = PromptTemplate(
            template=self.prompt_template_str,
            input_variables=list(set(prompt_vars)) # 중복 제거
        )
        logger.info(f"GenericLLMNode '{self.node_id}' initialized. Tool use enabled: {self.enable_tool_use}. Prompt: {prompt_template_path}")

    def _load_prompt_template(self) -> str:
        base_prompt_dir = getattr(settings, 'PROMPT_TEMPLATE_DIR', 'config/prompts')
        if os.path.isabs(self.prompt_template_path):
            full_path = self.prompt_template_path
        else:
            full_path = os.path.join(base_prompt_dir, self.prompt_template_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {full_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt template from {full_path}: {e}")
            raise

    def _get_available_tools_for_prompt(self) -> str:
        """ToolManager에서 허용된 도구 목록과 설명을 가져와 프롬프트에 넣을 문자열 생성"""
        # 이 메서드는 enable_tool_use가 True일 때만 호출될 것이므로 내부 체크 제거 가능
        # (또는 추가적인 방어로 남겨둘 수 있음)
        if not self.enable_tool_use:
             return "Tool use is disabled for this node."

        available_tool_names = self.tool_manager.get_names()
        tools_to_list = []

        if self.allowed_tools is not None:
            allowed_and_available = available_tool_names.intersection(self.allowed_tools)
            if not allowed_and_available:
                 return "No tools available (allowed list is empty or no matches found)."
            tools_to_list = list(allowed_and_available)
        else:
            tools_to_list = list(available_tool_names)

        if not tools_to_list:
             return "No tools available."

        tool_summaries = self.tool_manager.get_tool_summaries_for_llm()
        filtered_summaries = [s for s in tool_summaries if s['name'] in tools_to_list]

        if not filtered_summaries:
             return "No details found for available tools."

        prompt_lines = ["Available Tools:"]
        for summary in filtered_summaries:
             prompt_lines.append(f"- {summary['name']}: {summary['description']}")
        return "\n".join(prompt_lines)

    def _prepare_prompt_input(self, state: AgentGraphState) -> Dict[str, Any]:
        """AgentGraphState에서 프롬프트 포맷팅에 필요한 입력을 추출합니다."""
        prompt_input = {}
        all_expected_vars = self.prompt_template.input_variables # 현재 프롬프트가 필요로 하는 변수 목록

        for key in all_expected_vars:
            value = None
            # 1. 상태 객체 속성에서 직접 찾기
            if hasattr(state, key):
                value = getattr(state, key)
            # 2. dynamic_data 딕셔너리에서 찾기
            elif hasattr(state, 'dynamic_data') and isinstance(state.dynamic_data, dict) and key in state.dynamic_data:
                value = state.dynamic_data[key]
            # 3. metadata 딕셔너리에서 찾기 (덜 일반적)
            elif hasattr(state, 'metadata') and isinstance(state.metadata, dict) and key in state.metadata:
                value = state.metadata[key]

            # 특별 처리: 도구 목록 및 이력 (도구 사용 시에만)
            if self.enable_tool_use:
                if key == 'available_tools':
                    value = self._get_available_tools_for_prompt()
                elif key == 'tool_call_history':
                    history = state.dynamic_data.get('tool_call_history', []) if isinstance(state.dynamic_data, dict) else []
                    value = "\n".join([f"Tool: {call.get('tool_name')}, Args: {call.get('args')}, Result: {call.get('result')}" for call in history]) if history else "No tool calls yet."
                elif key == 'scratchpad':
                    value = state.dynamic_data.get('scratchpad', "") if isinstance(state.dynamic_data, dict) else ""

            # 값이 None일 경우 처리
            if value is None:
                # input_keys_for_prompt에 명시된 필수 키가 누락된 경우 경고
                if key in self.input_keys_for_prompt:
                    logger.warning(f"Key '{key}' specified in input_keys_for_prompt not found in state for node '{self.node_id}'. Using empty string.")
                    prompt_input[key] = ""
                # 자동으로 추가된 변수(available_tools 등)나 명시되지 않은 키는 빈 문자열
                else:
                    prompt_input[key] = ""
            else:
                # 복잡한 객체 문자열 변환
                if isinstance(value, (list, dict)) and key != 'messages':
                    try:
                        prompt_input[key] = json.dumps(value, indent=2, default=str)
                    except Exception:
                        prompt_input[key] = str(value)
                else:
                    prompt_input[key] = value

        return prompt_input


    def _parse_llm_response(self, response_str: str) -> Optional[ParsedLLMResponse]:
        """LLM 응답 문자열을 파싱하여 Action과 Input을 추출합니다."""
        # 이 메서드는 도구 사용 시에만 호출되므로, 내부 로직은 그대로 유지
        response_str = response_str.strip()
        logger.debug(f"Node '{self.node_id}': Parsing LLM response: {response_str[:300]}...")

        try:
            if response_str.startswith("```json") and response_str.endswith("```"):
                response_str = response_str[7:-3].strip()
            elif response_str.startswith("```") and response_str.endswith("```"):
                 response_str = response_str[3:-3].strip()

            parsed_json = json.loads(response_str)
            if isinstance(parsed_json, dict) and 'action' in parsed_json and 'action_input' in parsed_json:
                 action = parsed_json['action']
                 action_input = parsed_json['action_input']
                 logger.info(f"Node '{self.node_id}': Successfully parsed JSON response.")
                 return ParsedLLMResponse(action=action, action_input=action_input)
            else:
                 logger.warning(f"Node '{self.node_id}': Parsed JSON lacks 'action' or 'action_input' keys.")
                 return None

        except json.JSONDecodeError:
            logger.warning(f"Node '{self.node_id}': Response is not valid JSON. Attempting text parsing.")
            action_match = re.search(r"^Action:\s*([a-zA-Z0-9_]+)", response_str, re.MULTILINE)
            args_match = re.search(r"Args:\s*(\{.*\}|[^\n]*)", response_str, re.DOTALL | re.MULTILINE)

            if action_match:
                action = action_match.group(1).strip()
                action_input_str = args_match.group(1).strip() if args_match else "{}"
                action_input: Union[Dict, str] = {}
                try:
                    action_input = json.loads(action_input_str)
                    if not isinstance(action_input, dict):
                         action_input = {"input": action_input_str}
                except json.JSONDecodeError:
                    action_input = {"input": action_input_str}

                logger.info(f"Node '{self.node_id}': Successfully parsed text response (Action: {action}).")
                return ParsedLLMResponse(action=action, action_input=action_input)

            logger.error(f"Node '{self.node_id}': Failed to parse LLM response using known formats.")
            return None

    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """LangGraph 노드 실행 메서드"""
        logger.info(f"GenericLLMNode '{self.node_id}' execution started. Tool use enabled: {self.enable_tool_use}")

        # --------------------------------------------------
        # Case 1: 도구 사용 비활성화 (enable_tool_use is False)
        # --------------------------------------------------
        if not self.enable_tool_use:
            logger.debug(f"Node '{self.node_id}': Executing simple LLM call (tool use disabled).")
            try:
                prompt_input_values = self._prepare_prompt_input(state)
                formatted_prompt = self.prompt_template.format(**prompt_input_values)
                logger.debug(f"Node '{self.node_id}': Formatted prompt (tools disabled):\n{formatted_prompt[:500]}...")

                llm_params = {}
                if self.temperature is not None: llm_params['temperature'] = self.temperature
                if self.max_tokens is not None: llm_params['max_tokens'] = self.max_tokens

                # 메시지 형식으로 LLM 호출
                messages_for_llm = [{"role": "user", "content": formatted_prompt}]
                llm_response_str = await self.llm_client.generate_response(
                    messages=messages_for_llm,
                    model_name=self.model_name,
                    **llm_params
                )
                logger.debug(f"Node '{self.node_id}': LLM raw response (tools disabled): {llm_response_str[:200]}...")

                # 결과 저장
                update_dict: Dict[str, Any] = {
                    "error_message": None,
                    "last_llm_input": formatted_prompt,   # <-- 추가
                    "last_llm_output": llm_response_str   # <-- 추가
                }
                output_key = self.output_field_name or "final_answer"

                if '.' in output_key:
                     parent_key, child_key = output_key.split('.', 1)
                     if parent_key == "dynamic_data":
                          # Ensure dynamic_data exists in the update dictionary
                          if "dynamic_data" not in update_dict:
                               update_dict["dynamic_data"] = {}
                          # Add the result to dynamic_data
                          update_dict["dynamic_data"][child_key] = llm_response_str
                          # Add last input/output to dynamic_data as well
                          update_dict["dynamic_data"]["last_llm_input"] = formatted_prompt
                          update_dict["dynamic_data"]["last_llm_output"] = llm_response_str
                     else:
                          logger.warning(f"Cannot set output to non-dynamic_data nested field '{output_key}'. Storing in 'final_answer'.")
                          update_dict["final_answer"] = llm_response_str
                else:
                     update_dict[output_key] = llm_response_str


                return update_dict

            except Exception as e:
                logger.error(f"Node '{self.node_id}': Error during simple LLM call: {e}", exc_info=True)
                return {"error_message": f"Error in node '{self.node_id}' (tools disabled): {e}"}

        # --- ReAct Loop Logic ---
        logger.debug(f"Node '{self.node_id}': Executing ReAct loop (tool use enabled).")
        current_error_message = None
        # Start with a copy of dynamic_data from the input state for this invocation
        current_dynamic_data = state.dynamic_data.copy() if state.dynamic_data else {}
        # Ensure essential keys exist
        current_dynamic_data.setdefault('scratchpad', "")
        current_dynamic_data.setdefault('tool_call_history', [])

        for i in range(self.max_react_iterations):
            logger.info(f"Node '{self.node_id}': ReAct Iteration {i + 1}/{self.max_react_iterations}")

            # Create a temporary state representation for prompt preparation for this iteration
            temp_state_for_prompt = AgentGraphState(
                 task_id=state.task_id,
                 original_input=state.original_input,
                 current_iteration=i, # Pass current iteration number
                 thoughts=state.thoughts, # Pass through existing thoughts if any
                 current_thoughts_to_evaluate=state.current_thoughts_to_evaluate,
                 current_best_thought_id=state.current_best_thought_id,
                 search_depth=state.search_depth,
                 max_search_depth=state.max_search_depth,
                 dynamic_data=current_dynamic_data, # Use the evolving dynamic data
                 metadata=state.metadata,
                 # Omit fields not needed for prompt or that are updated later
            )

            # --- 1. 프롬프트 준비 ---
            try:
                prompt_input_values = self._prepare_prompt_input(temp_state_for_prompt)
                formatted_prompt = self.prompt_template.format(**prompt_input_values)
                logger.debug(f"Node '{self.node_id}' Iteration {i+1}: Formatted prompt ready.")
            except Exception as prompt_err:
                 logger.error(f"Node '{self.node_id}': Error preparing prompt: {prompt_err}", exc_info=True)
                 current_error_message = f"Error preparing prompt in node '{self.node_id}': {prompt_err}"
                 break

            # --- 2. LLM 호출 ---
            try:
                llm_params = {}
                if self.temperature is not None: llm_params['temperature'] = self.temperature
                llm_params['max_tokens'] = self.max_tokens if self.max_tokens else 1000

                messages_for_llm = [{"role": "user", "content": formatted_prompt}]
                llm_response_str = await self.llm_client.generate_response(
                    messages=messages_for_llm, model_name=self.model_name, **llm_params
                )
                logger.debug(f"Node '{self.node_id}': LLM raw response received.")
                # Update last LLM input/output in our working copy
                current_dynamic_data["last_llm_input"] = formatted_prompt
                current_dynamic_data["last_llm_output"] = llm_response_str

            except Exception as llm_err:
                logger.error(f"Node '{self.node_id}': LLM call failed: {llm_err}", exc_info=True)
                current_error_message = f"LLM call failed in node '{self.node_id}': {llm_err}"
                break

            # --- 3. LLM 응답 파싱 ---
            parsed_response = self._parse_llm_response(llm_response_str)
            if parsed_response is None:
                logger.error(f"Node '{self.node_id}': Failed to parse LLM response. Response: {llm_response_str[:500]}...")
                current_error_message = f"Failed to parse LLM response in node '{self.node_id}'."
                break

            action = parsed_response['action']
            action_input = parsed_response['action_input'] # Can be dict or str
            logger.info(f"Node '{self.node_id}': Parsed Action: {action}, Input type: {type(action_input).__name__}")

            # --- 4. Action 실행 및 Observation 생성 ---
            observation = ""
            tool_call_entry = None # To store tool call details

            if action.lower() == "final_answer" or action.lower() == "finish":
                logger.info(f"Node '{self.node_id}': Received 'final_answer' action. Finishing.")
                final_answer = action_input if isinstance(action_input, str) else json.dumps(action_input, default=str)
                # Prepare final state update and exit
                return {
                    "final_answer": final_answer,
                    "error_message": None,
                    "dynamic_data": current_dynamic_data # Include final history etc.
                }

            elif action.lower() == "think":
                logger.info(f"Node '{self.node_id}': Received 'think' action.")
                observation = f"Thought processed: {action_input if isinstance(action_input, str) else json.dumps(action_input, default=str)}"
                # Append thought to scratchpad
                current_scratchpad = current_dynamic_data.get('scratchpad', "")
                current_dynamic_data['scratchpad'] = current_scratchpad + f"\nThought: {observation}"

            elif self.tool_manager.has(action):
                if self.allowed_tools is not None and action not in self.allowed_tools:
                    observation = f"Error: Tool '{action}' is not allowed for this node."
                    logger.warning(observation)
                    tool_call_entry = {"tool_name": action, "args": action_input, "result": observation, "error": True}
                else:
                    logger.info(f"Node '{self.node_id}': Executing tool '{action}'.")
                    tool_args = action_input if isinstance(action_input, dict) else {}
                    tool_result_str = ""
                    tool_error = False
                    try:
                        tool_instance = self.tool_manager.get_tool(action)
                        # Use ainvoke for Langchain standard invocation
                        tool_result = await tool_instance.ainvoke(tool_args)
                        tool_result_str = str(tool_result) # Convert result to string for observation
                        observation = f"Tool {action} execution successful."
                        logger.debug(f"Tool '{action}' Result: {tool_result_str[:200]}...")
                    except ToolError as tool_err:
                         observation = f"Error executing tool '{action}': {tool_err.message}"
                         logger.error(f"ToolError during execution of '{action}': {tool_err.message}", exc_info=False)
                         tool_error = True
                         tool_result_str = observation # Store error message as result
                    except Exception as tool_run_e:
                         observation = f"Unexpected error running tool '{action}': {str(tool_run_e)}"
                         logger.exception(f"Unexpected error during execution of tool '{action}'")
                         tool_error = True
                         tool_result_str = observation # Store error message as result

                    tool_call_entry = {
                         "tool_name": action,
                         "args": tool_args,
                         "result": tool_result_str, # Store string result
                         "error": tool_error
                    }
                    # Observation should include the result string
                    observation = f"Observation: {tool_result_str}"

            else: # Unknown action
                observation = f"Error: Unknown action '{action}'."
                logger.error(observation)
                tool_call_entry = {"tool_name": action, "args": action_input, "result": observation, "error": True}

            # --- 5. 상태 업데이트 (다음 루프를 위해) ---
            # Append Observation to scratchpad
            current_scratchpad = current_dynamic_data.get('scratchpad', "")
            current_dynamic_data['scratchpad'] = current_scratchpad + f"\n{observation}"
            # Append tool call entry if one was made
            if tool_call_entry:
                current_dynamic_data['tool_call_history'].append(tool_call_entry)

            # !! crucial: DO NOT modify the input 'state' object directly !!
            # The loop continues, and the next iteration will use the modified
            # 'current_dynamic_data' when preparing the prompt via _prepare_prompt_input.

        # --- Loop Finished (Max Iterations Reached or Error) ---
        logger.warning(f"Node '{self.node_id}': Reached max iterations ({self.max_react_iterations}) or error occurred.")
        final_output_value = "Reached maximum iterations without a final answer."
        if current_error_message:
             final_output_value = f"Finished due to error: {current_error_message}"
        elif 'observation' in locals() and observation:
             final_output_value = f"Max iterations reached. Last observation: {observation}"

        # Return the final state update dictionary
        final_state_update: Dict[str, Any] = {
            "error_message": current_error_message or f"Reached max iterations ({self.max_react_iterations})",
            "dynamic_data": current_dynamic_data # Include final history, scratchpad etc.
        }
        # Add final answer if applicable, otherwise error message dictates outcome
        output_key = self.output_field_name or "final_answer"

        if '.' in output_key:
            parent_key, child_key = output_key.split('.', 1)
            if parent_key == "dynamic_data":
                final_state_update["dynamic_data"][child_key] = final_output_value
            else:
                 logger.warning(f"Cannot set output to non-dynamic_data nested field '{output_key}'. Storing in 'final_answer'.")
                 final_state_update["final_answer"] = final_output_value
        else:
            final_state_update[output_key] = final_output_value

        # Check if 'final_answer' key was explicitly set by finish action earlier (unlikely if loop finished)
        if "final_answer" not in final_state_update:
             # If loop ended due to iterations/error, set final_answer field if it was the target
             if output_key == "final_answer":
                 final_state_update["final_answer"] = final_output_value
             # Otherwise, the output is in dynamic_data or handled by error_message

        return final_state_update
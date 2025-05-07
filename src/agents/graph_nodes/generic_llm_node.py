# multiagent/src/agents/graph_nodes/generic_llm_node.py

import os
import builtins as _bt
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate

from src.config.settings import get_settings
from src.config.logger import get_logger
from src.services.llm_client import LLMClient # 3단계에서 구현한 LLMClient
from src.schemas.mcp_models import AgentGraphState # Step 4.1에서 정의/수정한 상태 모델

_bt.os = os 
logger = get_logger(__name__)
settings = get_settings()

class GenericLLMNodeInput(TypedDict):
    """GenericLLMNode에 전달될 것으로 예상되는 상태의 일부 (예시)"""
    # 이 TypedDict는 실제 AgentGraphState의 부분집합을 나타낼 수 있으며,
    # LangGraph에서 노드로 전달되는 상태 객체 전체를 의미합니다.
    # 실제로는 AgentGraphState 전체를 state 파라미터로 받게 됩니다.
    pass # 실제로는 AgentGraphState가 이 역할을 합니다.

class GenericLLMNode:
    """
    설정 가능한 단일 LLM 호출을 수행하는 LangGraph 노드입니다.
    """
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template_path: str,
        output_field_name: str, # 결과를 저장할 AgentGraphState 내 필드명 (예: 'last_llm_output' 또는 'dynamic_data.some_key')
        input_keys_for_prompt: Optional[List[str]] = None, # 프롬프트 포맷팅에 사용할 상태 키 목록
        model_name: Optional[str] = None, # llm_client의 기본 모델을 오버라이드할 경우
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        node_id: str = "generic_llm_node" # 로깅 및 식별을 위한 노드 ID
    ):
        self.llm_client = llm_client
        self.prompt_template_path = prompt_template_path
        self.output_field_name = output_field_name
        self.input_keys_for_prompt = input_keys_for_prompt or []
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.node_id = node_id

        # 프롬프트 템플릿 로드
        self.prompt_template_str = self._load_prompt_template()
        # PromptTemplate 객체 생성 시 input_variables 자동 감지 기능을 사용하거나,
        # 명시적으로 input_keys_for_prompt를 사용할 수 있습니다.
        # 여기서는 명시적 키를 기준으로 생성합니다.
        self.prompt_template = PromptTemplate(
            template=self.prompt_template_str,
            input_variables=self.input_keys_for_prompt
        )
        logger.info(f"GenericLLMNode '{self.node_id}' initialized. Prompt template loaded from: {prompt_template_path}, Output field: {output_field_name}")

    def _load_prompt_template(self) -> str:
        """지정된 경로에서 프롬프트 템플릿 파일을 로드합니다."""
        # PROMPT_TEMPLATE_DIR 설정을 사용하여 전체 경로 구성
        # settings.PROMPT_TEMPLATE_DIR는 .env 또는 AppSettings에 정의되어 있어야 합니다.
        base_prompt_dir = getattr(settings, 'PROMPT_TEMPLATE_DIR', 'config/prompts')
        
        # template_path가 절대 경로인지 확인
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

    def _prepare_prompt_input(self, state: AgentGraphState) -> Dict[str, Any]:
        """AgentGraphState에서 프롬프트 포맷팅에 필요한 입력을 추출합니다."""
        prompt_input = {}
        for key in self.input_keys_for_prompt:
            # AgentGraphState의 여러 레벨에서 값을 가져올 수 있도록 로직 추가 가능
            # 예: 'original_input.some_field', 'dynamic_data.another_field'
            if hasattr(state, key):
                prompt_input[key] = getattr(state, key)
            elif key in state.dynamic_data:
                prompt_input[key] = state.dynamic_data[key]
            elif key in state.metadata:
                 prompt_input[key] = state.metadata[key]
            else:
                # 값이 없는 경우 빈 문자열 또는 기본값으로 처리할 수 있음
                logger.warning(f"Key '{key}' not found in AgentGraphState for prompt input in node '{self.node_id}'. Using empty string.")
                prompt_input[key] = ""
        return prompt_input

    async def __call__(self, state: AgentGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """LangGraph 노드 실행 메서드"""
        logger.info(f"GenericLLMNode '{self.node_id}' execution started for task: {state.task_id}")
        
        try:
            # 1. 프롬프트 입력 준비
            prompt_input_values = self._prepare_prompt_input(state)
            logger.debug(f"Node '{self.node_id}': Prompt input values: {prompt_input_values}")

            # 2. 프롬프트 포맷팅
            formatted_prompt = self.prompt_template.format(**prompt_input_values)
            logger.debug(f"Node '{self.node_id}': Formatted prompt:\n{formatted_prompt}")

            # 3. LLM 호출 파라미터 설정
            llm_params = {}
            if self.temperature is not None:
                llm_params['temperature'] = self.temperature
            if self.max_tokens is not None:
                llm_params['max_tokens'] = self.max_tokens
            
            # 4. LLM 호출 (ChatOpenAI 등은 messages 형식을 선호할 수 있으므로, generate_response가 이를 처리해야 함)
            #    여기서는 간단히 문자열 프롬프트를 전달한다고 가정합니다.
            #    LLMClient.generate_response는 내부적으로 messages 형식으로 변환할 수 있습니다.
            try:
                # llm_client.generate_response가 provider에 따라 messages 형식으로 변환
                messages_for_llm = [{"role": "user", "content": formatted_prompt}]
                llm_response_str = await self.llm_client.generate_response(
                    messages=messages_for_llm,
                    model_name=self.model_name # 지정된 경우 해당 모델 사용
                    # TODO: **llm_params 를 generate_response에 전달하는 기능 추가 필요 (LLMClient 수정)
                )
                logger.info(f"Node '{self.node_id}': LLM call successful.")
                logger.debug(f"Node '{self.node_id}': LLM response: {llm_response_str[:200]}...") # 너무 길면 일부만 로깅
                
                # 5. 상태 업데이트 준비
                update_dict = {
                    "last_llm_input": formatted_prompt,
                    "last_llm_output": llm_response_str,
                }
                # output_field_name에 따라 dynamic_data 또는 직접 필드 업데이트
                if '.' in self.output_field_name: # 예: "dynamic_data.my_custom_output"
                    parent_key, child_key = self.output_field_name.split('.', 1)
                    if parent_key == "dynamic_data":
                        # 기존 dynamic_data를 유지하면서 새 값 추가/업데이트
                        current_dynamic_data = state.dynamic_data.copy() if state.dynamic_data else {}
                        current_dynamic_data[child_key] = llm_response_str
                        update_dict["dynamic_data"] = current_dynamic_data
                    else:
                        logger.warning(f"Node '{self.node_id}': Unsupported parent key '{parent_key}' in output_field_name. Storing in dynamic_data.{self.output_field_name.replace('.', '_')}")
                        current_dynamic_data = state.dynamic_data.copy() if state.dynamic_data else {}
                        current_dynamic_data[self.output_field_name.replace('.', '_')] = llm_response_str
                        update_dict["dynamic_data"] = current_dynamic_data
                else: # 예: "final_answer"
                    update_dict[self.output_field_name] = llm_response_str

                update_dict["error_message"] = None # 성공 시 에러 메시지 초기화
                return update_dict

            except Exception as e:
                logger.error(f"Node '{self.node_id}': LLM call failed: {e}", exc_info=True)
                return {
                    "last_llm_input": formatted_prompt, # 실패했어도 어떤 프롬프트를 사용했는지 기록
                    "error_message": f"LLM call failed in node '{self.node_id}': {str(e)}"
                }

        except Exception as e:
            logger.error(f"Node '{self.node_id}': Unexpected error during execution: {e}", exc_info=True)
            return {
                "error_message": f"Unexpected error in node '{self.node_id}': {str(e)}"
            }

# LangGraph에 노드를 추가할 때는 이 클래스의 인스턴스를 생성하여 __call__ 메서드를 사용합니다.
# 예시:
# from src.services.llm_client import LLMClient # 실제 사용 시
# llm_client_instance = LLMClient() # 실제 사용 시
# generic_node_instance = GenericLLMNode(
#     llm_client=llm_client_instance,
#     prompt_template_path="path/to/your/prompt.txt",
#     output_field_name="dynamic_data.generation_result",
#     input_keys_for_prompt=["original_input", "dynamic_data.previous_step_output"],
#     node_id="my_generator"
# )
# graph.add_node("generator_node_name_in_graph", generic_node_instance)

"""
주요 변경 및 고려사항:

__init__:
llm_client: 의존성으로 주입받습니다. (3단계에서 구현된 LLMClient 인스턴스)
prompt_template_path: 프롬프트 템플릿 파일의 상대 경로 (예: thought_generation/style_a.txt). 전체 경로는 settings.PROMPT_TEMPLATE_DIR (예: config/prompts/)와 결합하여 구성됩니다.
output_field_name: LLM 응답을 AgentGraphState의 어떤 필드에 저장할지 지정합니다. 점(.)을 사용하여 dynamic_data 내의 중첩된 필드도 지정 가능합니다 (예: dynamic_data.thought_output).
input_keys_for_prompt: AgentGraphState에서 어떤 키들의 값을 프롬프트 포맷팅에 사용할지 리스트로 지정합니다.
node_id: 로깅 및 디버깅을 위해 노드에 고유 ID를 부여합니다.
_load_prompt_template(): 프롬프트 파일을 로드합니다. settings.PROMPT_TEMPLATE_DIR를 사용하여 기본 경로를 설정할 수 있도록 했습니다.
PromptTemplate: Langchain Core의 PromptTemplate을 사용하여 프롬프트 관리를 명시적으로 합니다.
_prepare_prompt_input: AgentGraphState에서 input_keys_for_prompt에 지정된 키들을 사용하여 프롬프트 포맷팅에 필요한 딕셔너리를 생성합니다. 상태 객체의 여러 위치 (state.some_field, state.dynamic_data.some_key, state.metadata.some_key)에서 값을 찾도록 간단한 예시 로직을 추가했습니다.
__call__: LangGraph 노드가 실행될 때 호출되는 메인 로직입니다.
프롬프트 입력 준비, 포맷팅, LLM 호출, 상태 업데이트 순으로 진행됩니다.
LLM 호출 시 llm_client.generate_response()를 사용합니다. LLMClient의 generate_response 메서드는 내부적으로 model_name을 인자로 받아 특정 모델을 사용하거나, 메시지 형식을 처리할 수 있어야 합니다. (필요시 LLMClient 수정)
상태 업데이트: output_field_name에 따라 AgentGraphState의 해당 필드를 LLM 응답으로 업데이트합니다. dynamic_data 내의 필드를 업데이트하는 경우, 기존 dynamic_data를 복사하여 수정 후 전체를 업데이트해야 msgspec.Struct의 불변성을 유지하면서 부분 업데이트 효과를 낼 수 있습니다. (예시에서는 state.dynamic_data.copy() 사용)
성공 시 error_message를 None으로 설정하여 이전 오류 상태를 지웁니다.
오류 처리: 파일 로딩, LLM 호출, 예기치 않은 오류 발생 시 AgentGraphState의 error_message 필드를 업데이트하여 그래프의 다음 단계에서 오류를 인지하고 처리할 수 있도록 합니다.
의존성:
src.services.llm_client.LLMClient
src.schemas.mcp_models.AgentGraphState
src.config.settings.get_settings
langchain_core.prompts.PromptTemplate (필요시 설치: pip install langchain-core)
LLMClient 수정 제안 (generate_response에 model_name 및 추가 파라미터 전달):

GenericLLMNode에서 model_name 및 temperature, max_tokens와 같은 파라미터를 llm_client.generate_response에 전달하려면 LLMClient의 해당 메서드가 이를 지원해야 합니다. 현재 제공해주신 LLMClient에는 generate_response에 model_name만 있고, 다른 파라미터는 없습니다. chat 메서드는 폴백 로직을 포함하고 있습니다.
"""
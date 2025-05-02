from typing import Dict, List, Union

from src.config.logger import get_logger
from src.llm.adapters.anthropic import AnthropicAdapter
from src.llm.adapters.gemini import GeminiAdapter
from src.llm.adapters.openai import OpenAIAdapter
from src.llm.base import BaseLLMAdapter

logger = get_logger(__name__)
SUPPORTED_INPUT_TYPES = Union[str, List[Dict[str, str]]]

async def transform_llm_input_for_model(original_input: SUPPORTED_INPUT_TYPES, target_adapter: BaseLLMAdapter) -> SUPPORTED_INPUT_TYPES:
    target_model_name = target_adapter.model
    target_provider = target_adapter.provider
    logger.debug(f"Transforming input for model '{target_model_name}' (Provider: {target_provider})")
    expects_messages_list = False
    if isinstance(target_adapter, OpenAIAdapter) and ('turbo' in target_model_name or target_model_name.startswith('gpt-4')):
        expects_messages_list = True
    elif isinstance(target_adapter, AnthropicAdapter):
        expects_messages_list = True
    elif isinstance(target_adapter, GeminiAdapter):
        expects_messages_list = True
    transformed_input = original_input
    if expects_messages_list:
        if isinstance(original_input, str):
            transformed_input = [{'role': 'user', 'content': original_input}]
            logger.debug(f'Transformed str prompt to messages list for {target_model_name}')
        elif isinstance(original_input, list):
            logger.debug(f'Input is already messages list, using as is for {target_model_name}')
        else:
            logger.warning(f"Unsupported original input type '{type(original_input)}' when expecting messages list for {target_model_name}. Trying str conversion.")
            transformed_input = [{'role': 'user', 'content': str(original_input)}]
    elif isinstance(original_input, list):
        transformed_input = '\n'.join([msg.get('content', '') for msg in original_input if msg.get('content')])
        logger.debug(f'Transformed messages list to single string for {target_model_name}')
    elif isinstance(original_input, str):
        logger.debug(f'Input is already string, using as is for {target_model_name}')
    else:
        logger.warning(f"Unsupported original input type '{type(original_input)}' when expecting string for {target_model_name}. Using str conversion.")
        transformed_input = str(original_input)
    return transformed_input
import string
from typing import Any, Dict, Optional, List, Set
from src.config.logger import get_logger
logger = get_logger(__name__)

class PromptTemplate:

    def __init__(self, template_id: str, template_string: str, input_variables: Optional[List[str]]=None):
        self.template_id: str = template_id
        self.template_string: str = template_string
        self.input_variables: Set[str] = set(input_variables) if input_variables else self._extract_variables(template_string)
        if not self._validate_template():
            logger.warning(f"Template '{template_id}' might have formatting issues or unmatched braces.")
        logger.debug(f"PromptTemplate '{template_id}' initialized. Variables: {self.input_variables}")

    def _extract_variables(self, template_string: str) -> Set[str]:
        variables: Set[str] = set()
        try:
            for _, field_name, _, _ in string.Formatter().parse(template_string):
                if field_name is not None:
                    base_variable = field_name.split('.')[0].split('[')[0]
                    if base_variable:
                        variables.add(base_variable)
        except ValueError as e:
            logger.warning(f"Error parsing variables from template '{self.template_id}': {e}")
        return variables

    def _validate_template(self) -> bool:
        balance = 0
        escaped = False
        for char in self.template_string:
            if char == '{' and (not escaped):
                if balance < 0:
                    return False
                balance += 1
                escaped = True
            elif char == '}' and (not escaped):
                balance -= 1
                if balance < 0:
                    return False
                escaped = True
            else:
                escaped = False
        return balance == 0

    def render(self, **kwargs: Any) -> str:
        provided_keys = set(kwargs.keys())
        missing_keys = self.input_variables - provided_keys
        if missing_keys:
            logger.error(f"Missing required variables for template '{self.template_id}': {missing_keys}")
            raise KeyError(f'Missing required variables: {', '.join(missing_keys)}')
        try:
            rendered_prompt = self.template_string.format(**kwargs)
            logger.debug(f"Successfully rendered template '{self.template_id}'")
            return rendered_prompt
        except KeyError as e:
            logger.error(f"Error rendering template '{self.template_id}': Missing key {e}", exc_info=True)
            raise
        except ValueError as e:
            logger.error(f"Error rendering template '{self.template_id}': Invalid template format. {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"Unexpected error rendering template '{self.template_id}': {e}")
            raise

class PromptTemplateLoader:

    def __init__(self, template_source: str):
        self.template_source = template_source
        self._cache: Dict[str, PromptTemplate] = {}
        logger.info(f'PromptTemplateLoader initialized with source: {template_source}')

    async def load_template(self, template_id: str) -> Optional[PromptTemplate]:
        if template_id in self._cache:
            logger.debug(f'Returning cached template for ID: {template_id}')
            return self._cache[template_id]
        logger.debug(f"Loading template '{template_id}' from source: {self.template_source} (Placeholder)")
        if template_id == 'planner_prompt':
            template_string = '\n             **Goal:** {goal}\n             **Available Tools:** {available_tools}\n             **Conversation History (if relevant):**\n             {conversation_history}\n             **Instructions:** Create a JSON plan... Output ONLY the JSON plan...\n             '
            template = PromptTemplate(template_id, template_string)
            self._cache[template_id] = template
            return template
        elif template_id == 'react_executor_prompt':
            template_string = '\n             Goal: {goal}\n             Plan: {plan}\n             History: {scratchpad}\n             Available Tools: {available_tools}\n             Thought: {thought} Action:\n             '
            template = PromptTemplate(template_id, template_string)
            self._cache[template_id] = template
            return template
        else:
            logger.warning(f'No placeholder found for template ID: {template_id}')
            return None
_default_loader: Optional[PromptTemplateLoader] = None
_loader_lock = asyncio.Lock()

async def get_prompt_loader(template_source: Optional[str]=None) -> PromptTemplateLoader:
    global _default_loader
    if _default_loader is None:
        async with _loader_lock:
            if _default_loader is None:
                source = template_source or 'prompts/'
                _default_loader = PromptTemplateLoader(template_source=source)
                logger.info(f'Singleton PromptTemplateLoader created (Source: {source})')
    if _default_loader is None:
        raise RuntimeError('Failed to create PromptTemplateLoader instance.')
    return _default_loader
import string
import asyncio
import os
import string
import asyncio
import os
import json
from typing import Any, Dict, Optional, List, Set, Tuple
from pathlib import Path

from src.config.logger import get_logger
logger = get_logger(__name__)


class DictFormatter(string.Formatter):
    """
    Custom formatter that supports accessing dictionary keys using dot notation.
    """
    def get_field(self, field_name: str, args: Tuple, kwargs: Dict) -> Any:
        """
        Override get_field to support dictionary key access with dot notation.
        """
        # Parse the field name for dots and brackets
        parts = self._parse_field_name(field_name)
        
        # Get the first part of the field name (the variable name)
        obj = self.get_value(parts[0], args, kwargs)
        
        # Process the remaining parts of the field name
        for part in parts[1:]:
            # Check if it's an index access like [0]
            if part.startswith('[') and part.endswith(']'):
                try:
                    # Try to parse as integer index
                    index = int(part[1:-1])
                    obj = obj[index]
                except ValueError:
                    # If not an integer, use as string key
                    key = part[1:-1]
                    if isinstance(obj, dict):
                        obj = obj.get(key, f"<Missing key: {key}>")
                    else:
                        obj = obj[key]
            else:
                # It's a dot access like .name
                if isinstance(obj, dict):
                    obj = obj.get(part, f"<Missing key: {part}>")
                else:
                    obj = getattr(obj, part)
        
        return obj, field_name
    
    def _parse_field_name(self, field_name: str) -> List[str]:
        """
        Parse a field name into parts split by dots and brackets.
        
        Args:
            field_name: The field name to parse (e.g., "person.hobbies[0]")
            
        Returns:
            A list of parts (e.g., ["person", "hobbies", "[0]"])
        """
        parts = []
        current = ""
        i = 0
        
        while i < len(field_name):
            if field_name[i] == '.':
                if current:
                    parts.append(current)
                    current = ""
            elif field_name[i] == '[':
                if current:
                    parts.append(current)
                    current = ""
                # Capture the entire bracket expression
                bracket = "["
                i += 1
                while i < len(field_name) and field_name[i] != ']':
                    bracket += field_name[i]
                    i += 1
                if i < len(field_name):  # Add the closing bracket if it exists
                    bracket += ']'
                parts.append(bracket)
            else:
                current += field_name[i]
            i += 1
        
        if current:  # Add the last part if it exists
            parts.append(current)
            
        return parts


class PromptTemplate:
    """
    A template for generating prompts with variable substitution.
    
    Attributes:
        template_id: Unique identifier for the template
        template_string: The template string with {variable} placeholders
        input_variables: Set of variable names required by the template
    """

    def __init__(self, template_id: str, template_string: str, input_variables: Optional[List[str]]=None):
        self.template_id: str = template_id
        self.template_string: str = template_string
        self.formatter = DictFormatter()
        self.input_variables: Set[str] = set(input_variables) if input_variables else self._extract_variables(template_string)
        if not self._validate_template():
            logger.warning(f"Template '{template_id}' might have formatting issues or unmatched braces.")
        logger.debug(f"PromptTemplate '{template_id}' initialized. Variables: {self.input_variables}")

    def _extract_variables(self, template_string: str) -> Set[str]:
        """
        Extract variable names from a template string.
        
        Args:
            template_string: The string containing {variable} placeholders
            
        Returns:
            A set of variable names found in the template
        """
        variables: Set[str] = set()
        try:
            for _, field_name, _, _ in self.formatter.parse(template_string):
                if field_name is not None:
                    base_variable = field_name.split('.')[0].split('[')[0]
                    if base_variable:
                        variables.add(base_variable)
        except ValueError as e:
            logger.warning(f"Error parsing variables from template '{self.template_id}': {e}")
        return variables

    def _validate_template(self) -> bool:
        """
        Validate the template string for properly matched braces.
        
        Returns:
            True if the template string has properly matched braces, False otherwise
        """
        stack = []
        i = 0
        while i < len(self.template_string):
            char = self.template_string[i]
            if char == '{' and (i + 1 < len(self.template_string)) and self.template_string[i + 1] != '{':
                stack.append('{')
            elif char == '}' and (i + 1 >= len(self.template_string) or self.template_string[i + 1] != '}'):
                if not stack or stack[-1] != '{':
                    return False
                stack.pop()
            elif char == '{' and (i + 1 < len(self.template_string)) and self.template_string[i + 1] == '{':
                i += 1  # Skip the escaped brace
            elif char == '}' and (i + 1 < len(self.template_string)) and self.template_string[i + 1] == '}':
                i += 1  # Skip the escaped brace
            i += 1
        return len(stack) == 0

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with the provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            The rendered template string
            
        Raises:
            KeyError: If a required variable is missing
            ValueError: If the template format is invalid
        """
        provided_keys = set(kwargs.keys())
        missing_keys = self.input_variables - provided_keys
        if missing_keys:
            logger.error(f"Missing required variables for template '{self.template_id}': {missing_keys}")
            raise KeyError(f"Missing required variables: {', '.join(missing_keys)}")
        try:
            rendered_prompt = self.formatter.format(self.template_string, **kwargs)
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
    """
    Loads and caches prompt templates from a source (file or directory).
    
    Attributes:
        template_source: Path to the template source (file or directory)
        _cache: Dictionary of cached templates
    """

    def __init__(self, template_source: str):
        self.template_source = template_source
        self._cache: Dict[str, PromptTemplate] = {}
        logger.info(f'PromptTemplateLoader initialized with source: {template_source}')

    async def load_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Load a template by its ID, either from cache or from the source.
        
        Args:
            template_id: The unique identifier for the template
            
        Returns:
            The loaded template, or None if not found
        """
        if template_id in self._cache:
            logger.debug(f'Returning cached template for ID: {template_id}')
            return self._cache[template_id]
        
        template = await self._load_from_source(template_id)
        if template:
            self._cache[template_id] = template
            return template
        
        # Fallback to hardcoded templates
        template = self._get_hardcoded_template(template_id)
        if template:
            self._cache[template_id] = template
            return template
            
        logger.warning(f'No template found for ID: {template_id}')
        return None
    
    async def _load_from_source(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Load a template from the source (file or directory).
        
        Args:
            template_id: The unique identifier for the template
            
        Returns:
            The loaded template, or None if not found
        """
        source_path = Path(self.template_source)
        
        # Check for single template file (JSON or TXT)
        if source_path.is_file():
            return await self._load_from_file(source_path, template_id)
        
        # Check for directory with template files
        if source_path.is_dir():
            # Try loading from a JSON file with the template ID as the filename
            json_path = source_path / f"{template_id}.json"
            if json_path.exists():
                return await self._load_from_json_file(json_path, template_id)
            
            # Try loading from a TXT file with the template ID as the filename
            txt_path = source_path / f"{template_id}.txt"
            if txt_path.exists():
                return await self._load_from_txt_file(txt_path, template_id)
            
            # Try loading from a common templates.json file
            common_json = source_path / "templates.json"
            if common_json.exists():
                return await self._load_from_json_file(common_json, template_id, key_in_file=True)
        
        return None
    
    async def _load_from_file(self, file_path: Path, template_id: str) -> Optional[PromptTemplate]:
        """
        Load a template from a file based on its extension.
        
        Args:
            file_path: Path to the template file
            template_id: The unique identifier for the template
            
        Returns:
            The loaded template, or None if not found
        """
        if file_path.suffix.lower() == '.json':
            return await self._load_from_json_file(file_path, template_id, key_in_file=True)
        elif file_path.suffix.lower() == '.txt':
            return await self._load_from_txt_file(file_path, template_id)
        else:
            logger.warning(f"Unsupported file type for template: {file_path}")
            return None
    
    async def _load_from_json_file(self, file_path: Path, template_id: str, key_in_file: bool = False) -> Optional[PromptTemplate]:
        """
        Load a template from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            template_id: The unique identifier for the template
            key_in_file: Whether the template ID is a key in the JSON file
            
        Returns:
            The loaded template, or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            json_content = await loop.run_in_executor(None, self._read_json_file, file_path)
            
            if key_in_file:
                if template_id not in json_content:
                    return None
                template_data = json_content[template_id]
            else:
                template_data = json_content
            
            if isinstance(template_data, str):
                # Simple string template
                return PromptTemplate(template_id, template_data)
            elif isinstance(template_data, dict):
                # Template with metadata
                template_string = template_data.get("template", "")
                input_vars = template_data.get("input_variables")
                return PromptTemplate(template_id, template_string, input_vars)
            else:
                logger.warning(f"Invalid template data format for '{template_id}' in {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading template '{template_id}' from JSON file {file_path}: {e}", exc_info=True)
            return None
    
    def _read_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Read and parse a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            The parsed JSON content
            
        Raises:
            Various exceptions if the file cannot be read or parsed
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def _load_from_txt_file(self, file_path: Path, template_id: str) -> Optional[PromptTemplate]:
        """
        Load a template from a text file.
        
        Args:
            file_path: Path to the text file
            template_id: The unique identifier for the template
            
        Returns:
            The loaded template, or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            file_content = await loop.run_in_executor(None, self._read_text_file, file_path)
            return PromptTemplate(template_id, file_content)
        except Exception as e:
            logger.error(f"Error loading template '{template_id}' from text file {file_path}: {e}", exc_info=True)
            return None
    
    def _read_text_file(self, file_path: Path) -> str:
        """
        Read a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            The file content as a string
            
        Raises:
            Various exceptions if the file cannot be read
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_hardcoded_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a hardcoded template by ID.
        
        Args:
            template_id: The unique identifier for the template
            
        Returns:
            The hardcoded template, or None if not found
        """
        logger.debug(f"Attempting to use hardcoded template for '{template_id}'")
        
        if template_id == 'planner_prompt':
            template_string = """
             **Goal:** {goal}
             **Available Tools:** {available_tools}
             **Conversation History (if relevant):**
             {conversation_history}
             **Instructions:** Create a JSON plan... Output ONLY the JSON plan...
             """
            return PromptTemplate(template_id, template_string)
            
        elif template_id == 'react_executor_prompt':
            template_string = """
             Goal: {goal}
             Plan: {plan}
             History: {scratchpad}
             Available Tools: {available_tools}
             Thought: {thought} Action:
             """
            return PromptTemplate(template_id, template_string)
            
        return None


# Global loader singleton
_default_loader: Optional[PromptTemplateLoader] = None
_loader_lock = asyncio.Lock()


async def get_prompt_loader(template_source: Optional[str]=None) -> PromptTemplateLoader:
    """
    Get or create the global PromptTemplateLoader instance.
    
    Args:
        template_source: Path to the template source (file or directory)
        
    Returns:
        The global PromptTemplateLoader instance
        
    Raises:
        RuntimeError: If the loader cannot be created
    """
    global _default_loader
    
    if _default_loader is None:
        async with _loader_lock:
            if _default_loader is None:
                source = template_source or os.environ.get('PROMPT_TEMPLATE_PATH', 'prompts/')
                _default_loader = PromptTemplateLoader(template_source=source)
                logger.info(f'Singleton PromptTemplateLoader created (Source: {source})')
    
    return _default_loader


async def reload_templates() -> None:
    """
    Clear the template cache to force reloading from source.
    """
    loader = await get_prompt_loader()
    loader._cache.clear()
    logger.info("Template cache cleared")
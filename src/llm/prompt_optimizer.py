import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from src.config.logger import get_logger
from src.config.settings import get_settings
from src.config.metrics import timed_metric, MEMORY_OPERATION_DURATION
from src.llm.tokenizer import count_tokens_sync
settings = get_settings()
logger = get_logger(__name__)

@timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'optimize_prompt'})
def optimize_prompt(prompt: str, model: str, target_token_count: Optional[int]=None, max_token_reduction_ratio: float=0.3, preserve_recent_context: bool=True, preserve_instructions: bool=True) -> str:
    try:
        current_token_count = count_tokens_sync(model, prompt)
    except Exception as e:
        logger.warning(f"Failed to count tokens for model '{model}' during optimization: {e}. Skipping optimization.")
        return prompt
    if target_token_count is None:
        target_token_count = int(current_token_count * (1.0 - max_token_reduction_ratio))
        if target_token_count >= current_token_count:
            return prompt
    if current_token_count <= target_token_count:
        logger.debug(f'Prompt is already within target token count ({current_token_count} <= {target_token_count}). No optimization needed.')
        return prompt
    logger.debug(f'Optimizing prompt: Current tokens={current_token_count}, Target tokens={target_token_count}')
    optimizations: List[Callable[[str], str]] = [_remove_redundant_whitespace, _compress_repeated_formats, _simplify_urls]
    optimized_prompt = prompt
    for i, optimize_func in enumerate(optimizations):
        previous_token_count = count_tokens_sync(model, optimized_prompt)
        optimized_prompt = optimize_func(optimized_prompt)
        optimized_token_count = count_tokens_sync(model, optimized_prompt)
        logger.debug(f"Applied optimization '{optimize_func.__name__}': Tokens {previous_token_count} -> {optimized_token_count}")
        if optimized_token_count <= target_token_count:
            logger.info(f'Prompt optimized to {optimized_token_count} tokens (target: {target_token_count}) after {i + 1} optimization(s).')
            return optimized_prompt
    if preserve_recent_context or preserve_instructions:
        logger.debug('Applying context trimming to reach target token count.')
        optimized_prompt = _trim_context(optimized_prompt, model, target_token_count, preserve_recent_context, preserve_instructions)
    else:
        logger.warning(f'Could not reach target token count ({target_token_count}) with light optimizations. Final tokens: {optimized_token_count}. No trimming applied as preservation flags are off.')
    final_token_count = count_tokens_sync(model, optimized_prompt)
    if final_token_count > target_token_count:
        logger.warning(f'Prompt optimization finished. Final tokens ({final_token_count}) still exceed target ({target_token_count}).')
    else:
        logger.info(f'Prompt optimization finished. Final tokens: {final_token_count} (target: {target_token_count}).')
    return optimized_prompt

def _remove_redundant_whitespace(prompt: str) -> str:
    optimized = re.sub('\\n{3,}', '\n\n', prompt)
    optimized = re.sub(' {2,}', ' ', optimized)
    optimized = re.sub('^\\s+', '', optimized, flags=re.MULTILINE)
    return optimized.strip()

def _compress_repeated_formats(prompt: str) -> str:
    optimized = prompt
    optimized = re.sub('(\\n-\\s[^\\n]+\\n-\\s[^\\n]+\\n)(?:-\\s[^\\n]+\\n){3,}', '\\1- ...(items omitted)...\\n', optimized, flags=re.IGNORECASE)
    optimized = re.sub('(\\n\\d+\\.\\s[^\\n]+\\n\\d+\\.\\s[^\\n]+\\n)(?:\\d+\\.\\s[^\\n]+\\n){2,}', '\\1...(items omitted)...\\n', optimized)
    return optimized

def _simplify_urls(prompt: str) -> str:
    try:
        optimized = re.sub('(https?://[^\\s?]+)\\?([^\\s]{30,})', '\\1?{...query_params...}', prompt)
    except re.error as e:
        logger.warning(f'Regex error during URL query simplification: {e}')
        optimized = prompt
    try:
        optimized = re.sub('(https?://[^/]+(?:/[^/]+){3})(/[^/]+){2,}', '\\1/...path_omitted...', optimized)
    except re.error as e:
        logger.warning(f'Regex error during URL path simplification: {e}')
    return optimized

def _trim_context(prompt: str, model: str, target_token_count: int, preserve_recent_context: bool=True, preserve_instructions: bool=True) -> str:
    instructions = ''
    if preserve_instructions:
        instruction_pattern = '^(?:.*?)(?:instructions?|guidelines?|rules?|task|objective|context|background|you should|please|your task is)[:\\s].*?(?=\\n\\n|$)'
        instruction_matches = list(re.finditer(instruction_pattern, prompt, re.IGNORECASE | re.DOTALL | re.MULTILINE))
        if instruction_matches:
            instructions = '\n\n'.join((match.group(0).strip() for match in instruction_matches))
            logger.debug(f'Identified instructions part (length {len(instructions)}).')
        else:
            logger.debug('No clear instruction pattern found at the beginning.')
    paragraphs = re.split('\\n\\s*\\n+', prompt.strip())
    essential_paragraphs: List[str] = []
    remaining_paragraphs: List[str] = []
    if preserve_instructions and instructions:
        instruction_para_indices = set()
        for match in instruction_matches:
            matched_text = match.group(0).strip()
            for i, p in enumerate(paragraphs):
                if p.strip() == matched_text:
                    essential_paragraphs.append(p)
                    instruction_para_indices.add(i)
                    break
        if not essential_paragraphs and instructions:
            essential_paragraphs.append(instructions)
        logger.debug(f'Added {len(essential_paragraphs)} instruction paragraph(s) to essential list.')
    recent_context_paragraph = None
    if preserve_recent_context and paragraphs:
        recent_context_paragraph = paragraphs[-1]
        last_para_index = len(paragraphs) - 1
        if last_para_index not in instruction_para_indices:
            essential_paragraphs.append(recent_context_paragraph)
            logger.debug('Added the last paragraph as recent context to essential list.')
        else:
            logger.debug('Last paragraph seems to be part of instructions, not added separately as recent context.')
    essential_set = set(essential_paragraphs)
    all_indices = set(range(len(paragraphs)))
    remaining_indices = list(all_indices - instruction_para_indices)
    if recent_context_paragraph and len(paragraphs) - 1 in remaining_indices:
        remaining_indices.remove(len(paragraphs) - 1)
    remaining_paragraphs = [paragraphs[i] for i in sorted(remaining_indices)]
    essential_content = '\n\n'.join(essential_paragraphs)
    essential_token_count = count_tokens_sync(model, essential_content)
    if essential_token_count > target_token_count:
        logger.warning(f'Essential parts alone ({essential_token_count} tokens) exceed target ({target_token_count}). Truncating essential parts.')
        prioritized_essential = ''
        if preserve_instructions and instructions:
            prioritized_essential = instructions
        elif preserve_recent_context and recent_context_paragraph:
            prioritized_essential = recent_context_paragraph
        elif essential_paragraphs:
            prioritized_essential = essential_paragraphs[0]
        estimated_chars_per_token = 4
        max_chars = target_token_count * estimated_chars_per_token
        truncated_essential = prioritized_essential[:max_chars]
        final_truncated_tokens = count_tokens_sync(model, truncated_essential)
        while final_truncated_tokens > target_token_count and len(truncated_essential) > 10:
            ratio = target_token_count / final_truncated_tokens
            max_chars = int(len(truncated_essential) * ratio * 0.9)
            truncated_essential = truncated_essential[:max_chars]
            final_truncated_tokens = count_tokens_sync(model, truncated_essential)
        return truncated_essential + '\n\n[Context heavily truncated due to length constraints]'
    final_paragraphs = essential_paragraphs.copy()
    current_token_count = essential_token_count
    truncation_notice = '\n\n[... Some context was truncated ...]\n\n'
    truncation_notice_tokens = count_tokens_sync(model, truncation_notice) if remaining_paragraphs else 0
    available_tokens = target_token_count - current_token_count - truncation_notice_tokens
    paragraphs_to_consider = remaining_paragraphs
    insertion_index = -1
    if preserve_instructions and preserve_recent_context and instructions and recent_context_paragraph:
        insertion_index = essential_paragraphs.index(instructions) + 1
    elif preserve_recent_context:
        insertion_index = len(essential_paragraphs) - 1 if recent_context_paragraph in essential_paragraphs else len(essential_paragraphs)
    else:
        insertion_index = 1 if instructions in essential_paragraphs else len(essential_paragraphs)
    added_paragraph_count = 0
    for paragraph in paragraphs_to_consider:
        paragraph_token_count = count_tokens_sync(model, paragraph)
        if paragraph_token_count <= available_tokens:
            final_paragraphs.insert(insertion_index, paragraph)
            available_tokens -= paragraph_token_count
            insertion_index += 1
            added_paragraph_count += 1
        else:
            logger.debug(f'Cannot add paragraph (tokens: {paragraph_token_count}, available: {available_tokens}). Stopping context addition.')
            break
    if len(remaining_paragraphs) > added_paragraph_count:
        notice_insertion_index = -1
        if preserve_instructions and instructions in final_paragraphs:
            notice_insertion_index = final_paragraphs.index(instructions) + 1
        elif preserve_recent_context and recent_context_paragraph in final_paragraphs:
            notice_insertion_index = final_paragraphs.index(recent_context_paragraph)
        else:
            notice_insertion_index = len(final_paragraphs) // 2 + 1
        final_paragraphs.insert(notice_insertion_index, truncation_notice.strip())
    optimized_prompt_final = '\n\n'.join(final_paragraphs)
    final_check_tokens = count_tokens_sync(model, optimized_prompt_final)
    logger.debug(f'Context trimming finished. Final tokens: {final_check_tokens} (Target: {target_token_count})')
    return optimized_prompt_final

@timed_metric(MEMORY_OPERATION_DURATION, {'operation_type': 'optimize_chat_messages'})
def optimize_chat_messages(messages: List[Dict[str, str]], model: str, target_token_count: Optional[int]=None, max_token_reduction_ratio: float=0.3, preserve_recent_messages: bool=True, preserve_system_message: bool=True) -> List[Dict[str, str]]:
    if not messages:
        return messages
    try:
        all_content = '\n'.join([msg.get('content', '') for msg in messages if msg.get('content')])
        current_token_count = count_tokens_sync(model, all_content)
    except Exception as e:
        logger.warning(f'Failed to count initial tokens for chat messages: {e}. Skipping optimization.')
        return messages
    if target_token_count is None:
        target_token_count = int(current_token_count * (1.0 - max_token_reduction_ratio))
        if target_token_count >= current_token_count:
            return messages
    if current_token_count <= target_token_count:
        logger.debug(f'Messages already within target token count ({current_token_count} <= {target_token_count}). No optimization needed.')
        return messages
    logger.debug(f'Optimizing chat messages: Current tokens={current_token_count}, Target tokens={target_token_count}')
    optimized_messages: List[Dict[str, str]] = []
    current_optimized_token_count = 0
    system_message_tokens = 0
    system_message = None
    if preserve_system_message:
        for msg in messages:
            if msg.get('role') == 'system':
                system_message = msg
                optimized_content = optimize_prompt(system_message.get('content', ''), model, None, max_token_reduction_ratio, preserve_instructions=True, preserve_recent_context=False)
                system_message['content'] = optimized_content
                system_message_tokens = count_tokens_sync(model, optimized_content)
                current_optimized_token_count += system_message_tokens
                optimized_messages.append(system_message)
                break
    for msg in messages:
        if msg == system_message:
            continue
        if not msg.get('content'):
            optimized_messages.append(msg)
            continue
        optimized_content = optimize_prompt(msg['content'], model, None, max_token_reduction_ratio, preserve_recent_context=True, preserve_instructions=False)
        optimized_msg = msg.copy()
        optimized_msg['content'] = optimized_content
        current_optimized_token_count += count_tokens_sync(model, optimized_content)
        optimized_messages.append(optimized_msg)
    logger.debug(f'After individual message optimization: {current_optimized_token_count} tokens.')
    if current_optimized_token_count <= target_token_count:
        return optimized_messages
    logger.debug('Individual optimization insufficient. Removing older messages...')
    final_messages: List[Dict[str, str]] = []
    preserved_indices: Set[int] = set()
    if system_message:
        try:
            system_msg_index = messages.index(system_message)
        except ValueError:
            system_msg_index = -1
        if system_msg_index != -1:
            preserved_indices.add(system_msg_index)
            final_messages.append(system_message)
    if preserve_recent_messages:
        temp_recent_messages: List[Dict[str, str]] = []
        temp_recent_tokens = 0
        for i in range(len(optimized_messages) - 1, -1, -1):
            msg = optimized_messages[i]
            if msg == system_message:
                continue
            msg_content = msg.get('content', '')
            msg_tokens = count_tokens_sync(model, msg_content)
            if system_message_tokens + temp_recent_tokens + msg_tokens <= target_token_count:
                temp_recent_messages.append(msg)
                temp_recent_tokens += msg_tokens
                try:
                    original_index = messages.index(msg)
                except ValueError:
                    pass
                else:
                    preserved_indices.add(original_index)
            else:
                break
        final_messages.extend(reversed(temp_recent_messages))
    original_message_count = len(messages)
    final_message_count = len(final_messages)
    if final_message_count < original_message_count:
        truncation_notice = {'role': 'system', 'content': '[... Some previous messages were truncated due to length constraints ...]'}
        insert_index = 1 if system_message in final_messages else 0
        final_messages.insert(insert_index, truncation_notice)
        logger.info(f'Truncated chat history. Kept {final_message_count} messages out of {original_message_count}.')
    final_content = '\n'.join([msg.get('content', '') for msg in final_messages if msg.get('content')])
    final_token_count = count_tokens_sync(model, final_content)
    logger.info(f'Message optimization finished. Final tokens: {final_token_count} (target: {target_token_count}).')
    return final_messages
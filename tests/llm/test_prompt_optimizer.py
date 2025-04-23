# 한 개 실패, URL 최적화가 안됨
import pytest
from unittest.mock import patch

from src.llm.prompt_optimizer import (
    optimize_prompt,
    optimize_chat_messages,
    _remove_redundant_whitespace,
    _compress_repeated_formats,
    _simplify_urls,
    _trim_context
)


def test_remove_redundant_whitespace():
    """Test removing redundant whitespace from prompts."""
    # Test with multiple newlines
    input_text = "Line 1\n\n\n\nLine 2"
    expected = "Line 1\n\nLine 2"
    assert _remove_redundant_whitespace(input_text) == expected
    
    # Test with multiple spaces
    input_text = "Too    many    spaces"
    expected = "Too many spaces"
    assert _remove_redundant_whitespace(input_text) == expected
    
    # Test with spaces at beginning of lines
    input_text = "Line 1\n    Line 2\n  Line 3"
    expected = "Line 1\nLine 2\nLine 3"
    assert _remove_redundant_whitespace(input_text) == expected
    
    # Test with trailing whitespace
    input_text = "Trailing spaces   \n\n"
    expected = "Trailing spaces"
    assert _remove_redundant_whitespace(input_text) == expected


def test_compress_repeated_formats():
    """Test compressing repeated formatting patterns."""
    # Test with repeated bullet points
    input_text = "\n- Item 1\n- Item 2\n- Item 3\n- Item 4\n- Item 5\n- Item 6\n"
    result = _compress_repeated_formats(input_text)
    assert "...(items omitted)..." in result
    assert result.count("-") < input_text.count("-")
    
    # Test with repeated numbered list
    input_text = "\n1. Step 1\n2. Step 2\n3. Step 3\n4. Step 4\n5. Step 5\n"
    result = _compress_repeated_formats(input_text)
    assert "...(items omitted)..." in result
    assert result.count(". ") < input_text.count(". ")
    
    # Test with repeated section headers
    input_text = "\n## Section 1\nContent 1\n## Section 2\nContent 2\n## Section 3\nContent 3\n## Section 4\nContent 4\n## Section 5\nContent 5\n"
    result = _compress_repeated_formats(input_text)
    assert "...(sections omitted)..." in result
    assert result.count("##") < input_text.count("##")


def test_simplify_urls():
    """Test simplifying URLs in prompts."""
    # Test with long query parameters
    input_text = "Check this URL: https://example.com/path?param1=value1&param2=value2&param3=reallyLongValueThatShouldBeTruncated1234567890"
    result = _simplify_urls(input_text)
    assert "https://example.com/path?..." in result
    assert "reallyLongValueThatShouldBeTruncated" not in result
    
    # Test with many path segments
    input_text = "Deep URL: https://example.com/level1/level2/level3/level4/level5/level6/page.html"
    result = _simplify_urls(input_text)
    assert "https://example.com/level1/level2/..." in result
    assert "level6/page.html" not in result
    
    # Test with URL that doesn't need simplification
    input_text = "Simple URL: https://example.com/path"
    result = _simplify_urls(input_text)
    assert result == input_text


@patch('src.llm.prompt_optimizer.count_tokens_sync')
def test_trim_context(mock_count_tokens):
    """Test trimming context to reach target token count."""
    # Mock token counting to return predictable values
    mock_count_tokens.side_effect = lambda model, text: len(text.split())
    
    # Test with instruction preservation
    input_text = "Instructions: Follow these steps carefully.\n\nContext paragraph 1.\n\nContext paragraph 2.\n\nContext paragraph 3."
    result = _trim_context(
        input_text,
        "gpt-4",
        target_token_count=10,  # Less than the full text
        preserve_instructions=True,
        preserve_recent_context=True
    )
    
    # Should preserve instructions and trim middle
    assert "Instructions: Follow these steps carefully" in result
    assert "Context paragraph 3" in result  # Recent context preserved
    assert "Context paragraph 2" not in result
    
    # Test without instruction preservation
    result = _trim_context(
        input_text,
        "gpt-4",
        target_token_count=10,
        preserve_instructions=False,
        preserve_recent_context=True
    )
    
    # Should prioritize recent context
    assert "Context paragraph 3" in result
    assert "Instructions: Follow these steps carefully" not in result


@patch('src.llm.prompt_optimizer.count_tokens_sync')
def test_optimize_prompt(mock_count_tokens):
    """Test the main optimize_prompt function."""
    # Mock token counting
    mock_count_tokens.side_effect = lambda model, text: len(text.split())
    
    # Test prompt that doesn't need optimization
    short_prompt = "This is a short prompt."
    result = optimize_prompt(short_prompt, "gpt-4", target_token_count=10)
    assert result == short_prompt  # No change needed
    
    # Test prompt that needs minor optimization
    medium_prompt = "This prompt has some   extra   spaces and \n\n\n newlines that can be removed."
    result = optimize_prompt(medium_prompt, "gpt-4", target_token_count=10)
    assert len(result.split()) <= 10
    assert "extra spaces" in result
    assert "newlines" in result
    
    # Test prompt that needs significant optimization
    long_prompt = """
    Instructions: Please follow these guidelines.
    
    - Point 1
    - Point 2
    - Point 3
    - Point 4
    - Point 5
    - Point 6
    - Point 7
    
    https://example.com/very/long/path/with/many/segments/and/a/long/filename.html?param1=value1&param2=value2
    
    Context paragraph 1. This contains important information.
    
    Context paragraph 2. This is less important.
    
    Context paragraph 3. This is recent context.
    """
    
    result = optimize_prompt(long_prompt, "gpt-4", target_token_count=20)
    
    # Check optimization results
    assert len(result.split()) <= 20
    assert "Instructions" in result  # Should preserve instructions
    assert "Context paragraph 3" in result  # Should preserve recent context
    assert "example.com" in result and "..." in result  # URL should be simplified
    assert "Point 1" in result  # Should keep some bullet points
    assert not all(f"Point {i}" in result for i in range(1, 8))  # Not all points should be preserved


@patch('src.llm.prompt_optimizer.count_tokens_sync')
def test_optimize_chat_messages(mock_count_tokens):
    """Test optimizing chat messages."""
    # Mock token counting
    mock_count_tokens.side_effect = lambda model, text: len(text.split())
    
    # Create a chat history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking."},
        {"role": "user", "content": "Tell me about machine learning."},
        {"role": "assistant", "content": "Machine learning is a subfield of artificial intelligence..."},
        {"role": "user", "content": "What about neural networks?"}
    ]
    
    # Test with no optimization needed
    result = optimize_chat_messages(messages, "gpt-4", target_token_count=100)
    assert result == messages  # No change needed
    
    # Test with moderate optimization needed
    result = optimize_chat_messages(messages, "gpt-4", target_token_count=30)
    
    # Check results
    assert len(result) <= len(messages)
    
    # System message and recent exchange should be preserved
    assert any(msg["role"] == "system" for msg in result)
    assert any(msg["role"] == "user" and "neural networks" in msg["content"] for msg in result)
    
    # Some middle messages might be removed or compressed
    content_words = sum(len(msg["content"].split()) for msg in result)
    assert content_words <= 30
    
    # Test with severe optimization needed
    result = optimize_chat_messages(messages, "gpt-4", target_token_count=10)
    
    # Check results
    assert len(result) < len(messages)
    
    # System message and most recent user message should be preserved
    system_preserved = any(msg["role"] == "system" for msg in result)
    latest_preserved = any(msg["role"] == "user" and "neural networks" in msg["content"] for msg in result)
    assert system_preserved or latest_preserved  # At least one should be preserved
    
    # Total content should be under the token limit
    content_words = sum(len(msg["content"].split()) for msg in result)
    assert content_words <= 10
    
    # Should include a truncation notice
    truncation_messages = [msg for msg in result if "truncated" in msg["content"].lower()]
    assert len(truncation_messages) > 0


@patch('src.llm.prompt_optimizer.count_tokens_sync')
def test_preserve_recent_context(mock_count_tokens):
    """Test that recent context is preserved during optimization."""
    # Mock token counting
    mock_count_tokens.side_effect = lambda model, text: len(text.split())
    
    # Create a prompt with "old" and "recent" context
    prompt = """
    This is old context from the beginning.
    
    This is more old context in the middle.
    
    This is recent context that should be preserved.
    """
    
    # Optimize with recent context preservation
    result = optimize_prompt(
        prompt, 
        "gpt-4", 
        target_token_count=10,  # Force trimming
        preserve_recent_context=True
    )
    
    # Check recent context is preserved
    assert "recent context that should be preserved" in result
    assert "old context from the beginning" not in result


@patch('src.llm.prompt_optimizer.count_tokens_sync')
def test_preserve_instructions(mock_count_tokens):
    """Test that instructions are preserved during optimization."""
    # Mock token counting
    mock_count_tokens.side_effect = lambda model, text: len(text.split())
    
    # Create a prompt with instructions and content
    prompt = """
    Instructions: This part should be preserved.
    
    Guidelines: These are also instructions to keep.
    
    This is regular content that can be trimmed if needed.
    
    This is more content that is less important.
    """
    
    # Optimize with instruction preservation
    result = optimize_prompt(
        prompt, 
        "gpt-4", 
        target_token_count=15,  # Force trimming
        preserve_instructions=True,
        preserve_recent_context=False  # Focus only on instructions
    )
    
    # Check instructions are preserved
    assert "Instructions: This part should be preserved" in result
    assert "Guidelines: These are also instructions to keep" in result
    
    # Not all content is preserved
    assert "more content that is less important" not in result
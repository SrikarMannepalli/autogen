---
myst:
  html_meta:
    "description lang=en": |
      Guide to using Anthropic's prompt caching feature with AutoGen for improved performance and cost efficiency.
---

# Anthropic Prompt Caching

Anthropic's Claude models support prompt caching, which allows you to cache parts of your prompts to reduce costs and improve response times for repeated content. AutoGen provides convenient methods to use this feature through the `AnthropicChatCompletionClient`.

## Overview

Prompt caching is useful when you have:
- Large system prompts that are reused across conversations
- Large context documents that are referenced multiple times
- Expensive tool results that can be reused

## Basic Usage

### Cached System Messages

Cache system prompts that are reused across multiple requests:

```python
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import UserMessage

client = AnthropicChatCompletionClient(
    model="claude-3-5-sonnet-20241022",
    api_key="your-api-key"
)

# Create a cached system message
system_msg = client.cached_system_message(
    "You are a helpful assistant specialized in code review and optimization. "
    "You have access to the company's coding standards document and best practices guide."
)

# Regular user message
user_msg = UserMessage(content="Please review this Python function.", source="user")

# The system message will be cached for future requests
response = await client.create([system_msg, user_msg])
```

### Cached User Messages

Cache large context documents or data that are referenced multiple times:

```python
# Large document or context
large_document = """
[Large codebase, documentation, or dataset content...]
""" * 100  # Simulating large content

# Cache the large context
context_msg = client.cached_user_message(
    content=large_document,
    source="user"
)

# Follow-up queries can reuse the cached context
query_msg = UserMessage(
    content="Based on the provided context, what are the key patterns?",
    source="user"
)

response = await client.create([system_msg, context_msg, query_msg])
```

### Cached Tool Results

Cache expensive tool execution results for reuse:

```python
from autogen_core.models import FunctionExecutionResult

# Tool results - some expensive, some not
tool_results = [
    FunctionExecutionResult(
        content="Quick calculation: 42",
        name="simple_calc",
        call_id="call_1"
    ),
    FunctionExecutionResult(
        content="[Large database query result with 10,000 rows...]",
        name="expensive_db_query",
        call_id="call_2"
    ),
    FunctionExecutionResult(
        content="Simple string: 'hello'",
        name="string_op",
        call_id="call_3"
    )
]

# Cache only the expensive database query result (index 1)
cached_tool_msg = client.cached_tool_results(
    content=tool_results,
    cached_indices=[1]
)

# Or cache all results
cached_all_msg = client.cached_tool_results(
    content=tool_results,
    cache_all=True
)

response = await client.create([system_msg, cached_tool_msg])
```

## Cache Policies

Currently, only `"ephemeral"` cache policy is supported:

```python
# Explicit cache policy (default is "ephemeral")
system_msg = client.cached_system_message(
    "You are a helpful assistant.",
    policy="ephemeral"
)
```

## Monitoring Cache Usage

Check if caching was used and how many tokens were saved:

```python
response = await client.create([cached_system_msg, user_msg])

if response.usage.cache_usage:
    cache_tokens = response.usage.cache_usage.cache_read_tokens
    print(f"Cache hit! Saved {cache_tokens} tokens")
else:
    print("No cache usage in this request")

# Check if any caching occurred
if response.cached:
    print("This response used cached content")
```

## Limitations and Considerations

### Multipart Content Caching

**Important**: For multipart user messages (containing text and images), only the **last text block** is cached:

```python
# This content has multiple parts
content = [
    "Introduction text",
    large_document,      # This will NOT be cached
    "What do you think?", # Only this will be cached
    image
]

# Only "What do you think?" gets cached, not the large_document
cached_msg = client.cached_user_message(content, source="user")
```

For granular control over multipart caching, use `AnthropicUserMessage` directly:

```python
from autogen_ext.models.anthropic import AnthropicUserMessage, CacheControl

# For advanced multipart caching control
msg = AnthropicUserMessage(
    content=multipart_content,
    source="user",
    cache_control=CacheControl(type="ephemeral")
)
```

### Cache Behavior

- Cache entries are ephemeral and expire after a short time
- Cache keys are based on exact content matching
- Only available with supported Anthropic models
- Cache creation has a small cost, but cache reads provide significant savings

## Advanced Usage

### Mixing Cached and Regular Messages

You can mix cached and regular messages in the same conversation:

```python
conversation = [
    # Cached system prompt
    client.cached_system_message("You are a code reviewer."),

    # Regular user message
    UserMessage(content="I have code to review.", source="user"),

    # Cached large context
    client.cached_user_message(large_codebase, source="user"),

    # Regular follow-up
    UserMessage(content="What can be improved?", source="user")
]

response = await client.create(conversation)
```

### Error Handling

```python
try:
    # Invalid cache indices will raise ValueError
    cached_msg = client.cached_tool_results(
        content=tool_results,
        cached_indices=[10]  # Out of range
    )
except ValueError as e:
    print(f"Cache configuration error: {e}")
```

## Best Practices

1. **Cache Large, Reusable Content**: Focus on system prompts, large documents, and expensive tool results
2. **Monitor Usage**: Check `response.usage.cache_usage` to verify caching is working
3. **String Content**: For user messages, prefer string content over multipart for full message caching
4. **Cost Optimization**: Cache content that will be reused multiple times to maximize savings

## API Reference

- {py:meth}`~autogen_ext.models.anthropic.AnthropicChatCompletionClient.cached_system_message`
- {py:meth}`~autogen_ext.models.anthropic.AnthropicChatCompletionClient.cached_user_message`
- {py:meth}`~autogen_ext.models.anthropic.AnthropicChatCompletionClient.cached_tool_results`
- {py:class}`~autogen_core.models.CacheUsage`
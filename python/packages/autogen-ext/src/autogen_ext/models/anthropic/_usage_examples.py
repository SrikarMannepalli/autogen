"""
Usage examples for Anthropic prompt caching with the new client-specific cache methods.

This module demonstrates how to use the Anthropic client's prompt caching capabilities
using the clean, provider-agnostic client cache methods. Users call simple methods like
client.cached_system_message() without needing to know about provider-specific message types.
"""

import asyncio
from typing import List

from autogen_core.models import FunctionExecutionResult, LLMMessage, SystemMessage, UserMessage

from autogen_ext.models.anthropic import AnthropicChatCompletionClient


async def example_system_message_caching() -> None:
    """Example: Cache a system message using the client's cached_system_message method."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    # Create cached system message using the client method
    system_msg = client.cached_system_message(
        "You are a helpful assistant specialized in code review and optimization."
    )

    # Regular user message (no caching)
    user_msg = UserMessage(content="Please review this Python function for potential improvements.", source="user")

    messages: List[LLMMessage] = [system_msg, user_msg]

    response = await client.create(messages=messages)
    print("System message cached for future requests:", response.content)


async def example_user_message_caching() -> None:
    """Example: Cache a user message containing large context using the client's cached_user_message method."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    # Simulate a large context that we want to cache
    large_context = (
        """
    # Large Documentation or Code Context
    # This could be a comprehensive API documentation, large codebase, or dataset
    # that we want to cache to avoid re-processing in subsequent requests
    """
        + "# "
        + "\n# ".join([f"Line {i} of documentation..." for i in range(100)])
    )

    system_msg = SystemMessage(content="You are a documentation assistant.")

    # Create cached user message using the client method
    context_msg = client.cached_user_message(content=large_context, source="user")

    # Regular follow-up query (no caching)
    query_msg = UserMessage(content="Based on the provided context, what are the key concepts?", source="user")

    messages: List[LLMMessage] = [system_msg, context_msg, query_msg]

    response = await client.create(messages=messages)
    print("Large context cached:", response.content)


async def example_tool_result_caching() -> None:
    """Example: Cache specific tool execution results using the client's cached_tool_results method."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    # Simulate tool execution results - some expensive, some not
    tool_results = [
        FunctionExecutionResult(content="Quick calculation result: 42", name="simple_calc", call_id="call_1"),
        FunctionExecutionResult(
            content="Expensive database query result: [large dataset with 10000 rows...]",
            name="database_query",
            call_id="call_2",
        ),
        FunctionExecutionResult(content="Simple string operation: 'hello world'", name="string_op", call_id="call_3"),
    ]

    system_msg = SystemMessage(content="You are an assistant with access to various tools.")
    user_msg = UserMessage(content="Process these tool results.", source="user")

    # Create cached tool results using the client method
    tool_msg = client.cached_tool_results(
        content=tool_results,
        cached_indices=[1],  # Cache only the expensive database query result (index 1)
    )

    messages: List[LLMMessage] = [system_msg, user_msg, tool_msg]

    response = await client.create(messages=messages)
    print("Expensive tool result cached:", response.content)


async def example_granular_tool_caching() -> None:
    """Example: Demonstrate granular control over individual tool results."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    tool_results = [
        FunctionExecutionResult(content="Result 1", name="tool1", call_id="call_1"),
        FunctionExecutionResult(content="Expensive Result 2", name="tool2", call_id="call_2"),
        FunctionExecutionResult(content="Result 3", name="tool3", call_id="call_3"),
        FunctionExecutionResult(content="Expensive Result 4", name="tool4", call_id="call_4"),
    ]

    # Cache results at index 1 and 3 (the expensive ones) using the client method
    tool_msg = client.cached_tool_results(
        content=tool_results,
        cached_indices=[1, 3],  # Cache the expensive results
    )

    messages: List[LLMMessage] = [
        SystemMessage(content="Process these tool results."),
        UserMessage(content="Analyze the results.", source="user"),
        tool_msg,
    ]

    response = await client.create(messages=messages)
    print("Specific tool results cached with granular control:", response.content)


async def example_combined_caching() -> None:
    """Example: Combine multiple cached message types with different caching strategies."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    # Create cached system message
    system_msg = client.cached_system_message(
        """
        You are an AI coding assistant with access to a large codebase.
        You help developers understand code, suggest improvements, and debug issues.
        Always provide detailed explanations and practical examples.
        """
    )

    # Create cached large codebase context
    codebase_context = client.cached_user_message(
        content="[Large codebase context - thousands of lines of code...]", source="user"
    )

    # Regular query without caching
    query_msg = UserMessage(
        content="What are potential performance issues in the user authentication module?", source="user"
    )

    messages: List[LLMMessage] = [system_msg, codebase_context, query_msg]

    response = await client.create(messages=messages)
    print("System and context cached with clean client methods:", response.content)


async def example_mixed_message_types() -> None:
    """Example: Mix cached and regular message types in the same conversation."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    messages: List[LLMMessage] = [
        # Cached system message using client method
        client.cached_system_message("You are a helpful coding assistant."),
        # Regular user message (no caching)
        UserMessage(content="I have some code to review.", source="user"),
        # Cached user message with large context using client method
        client.cached_user_message(content="[Large code snippet to cache...]", source="user"),
        # Regular follow-up query
        UserMessage(content="What can be improved?", source="user"),
    ]

    response = await client.create(messages=messages)
    print("Mixed message types with selective caching:", response.content)


async def example_cache_all_tool_results() -> None:
    """Example: Cache all tool results using cache_all parameter."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    # All expensive tool results
    expensive_results = [
        FunctionExecutionResult(content="[Large API response...]", name="api_call", call_id="1"),
        FunctionExecutionResult(content="[Complex computation...]", name="compute", call_id="2"),
        FunctionExecutionResult(content="[File system scan...]", name="file_scan", call_id="3"),
    ]

    # Cache all results using cache_all=True
    tool_msg = client.cached_tool_results(content=expensive_results, cache_all=True)

    messages: List[LLMMessage] = [
        SystemMessage(content="Process these expensive tool results."),
        UserMessage(content="Analyze all the data.", source="user"),
        tool_msg,
    ]

    response = await client.create(messages=messages)
    print("All tool results cached:", response.content)


async def example_custom_cache_policy() -> None:
    """Example: Using custom cache policies."""

    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")

    # Create cached messages with custom policies
    system_msg = client.cached_system_message(
        "You are a persistent assistant for long-running tasks.",
        policy="persistent",  # Custom policy instead of default "ephemeral"
    )

    user_msg = client.cached_user_message(
        content="[Large persistent context that should be cached longer...]", source="user", policy="persistent"
    )

    messages: List[LLMMessage] = [system_msg, user_msg]

    response = await client.create(messages=messages)
    print("Messages cached with custom policy:", response.content)


async def main() -> None:
    """Run all caching examples."""
    print("=== Anthropic Prompt Caching Examples (Client Methods) ===\n")

    print("1. System Message Caching:")
    await example_system_message_caching()
    print("\n" + "=" * 50 + "\n")

    print("2. User Message Caching:")
    await example_user_message_caching()
    print("\n" + "=" * 50 + "\n")

    print("3. Tool Result Caching:")
    await example_tool_result_caching()
    print("\n" + "=" * 50 + "\n")

    print("4. Granular Tool Caching:")
    await example_granular_tool_caching()
    print("\n" + "=" * 50 + "\n")

    print("5. Combined Caching:")
    await example_combined_caching()
    print("\n" + "=" * 50 + "\n")

    print("6. Mixed Message Types:")
    await example_mixed_message_types()
    print("\n" + "=" * 50 + "\n")

    print("7. Cache All Tool Results:")
    await example_cache_all_tool_results()
    print("\n" + "=" * 50 + "\n")

    print("8. Custom Cache Policy:")
    await example_custom_cache_policy()


if __name__ == "__main__":
    asyncio.run(main())

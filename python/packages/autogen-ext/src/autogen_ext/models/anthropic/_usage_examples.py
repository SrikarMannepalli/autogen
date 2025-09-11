"""
Usage examples for Anthropic prompt caching with granular user control.

This module demonstrates how to use the Anthropic client's prompt caching capabilities
using Anthropic-specific message classes that provide direct control over cache behavior,
following the .NET pattern for granular user control.
"""

import asyncio
from typing import List
from autogen_core.models import SystemMessage, UserMessage, FunctionExecutionResult, FunctionExecutionResultMessage
from autogen_ext.models.anthropic import (
    AnthropicChatCompletionClient, 
    AnthropicSystemMessage,
    AnthropicUserMessage,
    AnthropicFunctionExecutionResultMessage,
    CacheControl,
    create_ephemeral_cache
)


async def example_system_message_caching():
    """Example: Cache a system message using AnthropicSystemMessage with cache control."""
    
    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")
    
    # Create system message with cache control enabled
    system_msg = AnthropicSystemMessage.create_with_cache_control(
        content="You are a helpful assistant specialized in code review and optimization."
    )
    
    # Regular user message (no caching)
    user_msg = UserMessage(content="Please review this Python function for potential improvements.", source="user")
    
    messages = [system_msg, user_msg]
    
    response = await client.create(messages=messages)
    print("System message cached for future requests:", response.content)


async def example_user_message_caching():
    """Example: Cache a user message containing large context using AnthropicUserMessage."""
    
    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")
    
    # Simulate a large context that we want to cache
    large_context = """
    # Large Documentation or Code Context
    # This could be a comprehensive API documentation, large codebase, or dataset
    # that we want to cache to avoid re-processing in subsequent requests
    """ + "# " + "\n# ".join([f"Line {i} of documentation..." for i in range(100)])
    
    system_msg = SystemMessage(content="You are a documentation assistant.")
    
    # Create user message with cache control enabled
    context_msg = AnthropicUserMessage.create_with_cache_control(
        content=large_context, 
        source="user"
    )
    
    # Regular follow-up query (no caching)
    query_msg = UserMessage(content="Based on the provided context, what are the key concepts?", source="user")
    
    messages = [system_msg, context_msg, query_msg]
    
    response = await client.create(messages=messages)
    print("Large context cached:", response.content)


async def example_tool_result_caching():
    """Example: Cache specific tool execution results using AnthropicFunctionExecutionResultMessage."""
    
    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")
    
    # Simulate tool execution results - some expensive, some not
    tool_results = [
        FunctionExecutionResult(
            content="Quick calculation result: 42",
            name="simple_calc",
            call_id="call_1"
        ),
        FunctionExecutionResult(
            content="Expensive database query result: [large dataset with 10000 rows...]",
            name="database_query", 
            call_id="call_2"
        ),
        FunctionExecutionResult(
            content="Simple string operation: 'hello world'",
            name="string_op",
            call_id="call_3"
        )
    ]
    
    system_msg = SystemMessage(content="You are an assistant with access to various tools.")
    user_msg = UserMessage(content="Process these tool results.", source="user")
    
    # Create tool result message with cache control for specific results
    tool_msg = AnthropicFunctionExecutionResultMessage.create_with_cache_control(
        content=tool_results,
        cached_result_indices=[1]  # Cache only the expensive database query result (index 1)
    )
    
    messages = [system_msg, user_msg, tool_msg]
    
    response = await client.create(messages=messages)
    print("Expensive tool result cached:", response.content)


async def example_granular_tool_caching():
    """Example: Demonstrate granular control over individual tool results."""
    
    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")
    
    tool_results = [
        FunctionExecutionResult(content="Result 1", name="tool1", call_id="call_1"),
        FunctionExecutionResult(content="Expensive Result 2", name="tool2", call_id="call_2"),  
        FunctionExecutionResult(content="Result 3", name="tool3", call_id="call_3"),
        FunctionExecutionResult(content="Expensive Result 4", name="tool4", call_id="call_4"),
    ]
    
    # Create message and manually set cache control for specific indices
    tool_msg = AnthropicFunctionExecutionResultMessage(content=tool_results)
    
    # Cache results at index 1 and 3 (the expensive ones)
    tool_msg.set_cache_control_for_result(1, create_ephemeral_cache())
    tool_msg.set_cache_control_for_result(3, create_ephemeral_cache())
    
    messages = [
        SystemMessage(content="Process these tool results."),
        UserMessage(content="Analyze the results.", source="user"),
        tool_msg
    ]
    
    response = await client.create(messages=messages)
    print("Specific tool results cached with granular control:", response.content)


async def example_combined_caching():
    """Example: Combine multiple Anthropic message types with different caching strategies."""
    
    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")
    
    # System message with caching
    system_msg = AnthropicSystemMessage(
        content="""
        You are an AI coding assistant with access to a large codebase. 
        You help developers understand code, suggest improvements, and debug issues.
        Always provide detailed explanations and practical examples.
        """,
        cache_control=CacheControl(type="ephemeral")
    )
    
    # Large codebase context with caching
    codebase_context = AnthropicUserMessage(
        content="[Large codebase context - thousands of lines of code...]",
        source="user",
        cache_control=CacheControl(type="ephemeral")
    )
    
    # Regular query without caching
    query_msg = UserMessage(
        content="What are potential performance issues in the user authentication module?",
        source="user"
    )
    
    messages = [system_msg, codebase_context, query_msg]
    
    response = await client.create(messages=messages)
    print("System and context cached with granular control:", response.content)


async def example_mixed_message_types():
    """Example: Mix regular and Anthropic-specific message types in the same conversation."""
    
    client = AnthropicChatCompletionClient(model="claude-3-5-sonnet-20241022")
    
    messages = [
        # Cached system message
        AnthropicSystemMessage.create_with_cache_control(
            content="You are a helpful coding assistant."
        ),
        
        # Regular user message (no caching)
        UserMessage(content="I have some code to review.", source="user"),
        
        # Cached user message with large context
        AnthropicUserMessage.create_with_cache_control(
            content="[Large code snippet to cache...]",
            source="user"
        ),
        
        # Regular follow-up query
        UserMessage(content="What can be improved?", source="user")
    ]
    
    response = await client.create(messages=messages)
    print("Mixed message types with selective caching:", response.content)


async def main():
    """Run all caching examples."""
    print("=== Anthropic Prompt Caching Examples (Granular Control) ===\n")
    
    print("1. System Message Caching:")
    await example_system_message_caching()
    print("\n" + "="*50 + "\n")
    
    print("2. User Message Caching:")
    await example_user_message_caching()
    print("\n" + "="*50 + "\n")
    
    print("3. Tool Result Caching:")
    await example_tool_result_caching()
    print("\n" + "="*50 + "\n")

    print("4. Granular Tool Caching:")
    await example_granular_tool_caching()
    print("\n" + "="*50 + "\n")
    
    print("5. Combined Caching:")
    await example_combined_caching()
    print("\n" + "="*50 + "\n")
    
    print("6. Mixed Message Types:")
    await example_mixed_message_types()


if __name__ == "__main__":
    asyncio.run(main())

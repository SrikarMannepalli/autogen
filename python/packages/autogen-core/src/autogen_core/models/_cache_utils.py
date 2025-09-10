"""
Convenience utilities for prompt caching.
Follows the .NET AutoGen approach for explicit user control.
"""

from typing import Any, Dict, List, Union
from ._types import SystemMessage, UserMessage, FunctionExecutionResultMessage, FunctionExecutionResult, Image


def create_cached_system_message(content: str) -> SystemMessage:
    """
    Create a SystemMessage with ephemeral cache control enabled.
    
    Args:
        content: The system message content
        
    Returns:
        SystemMessage with cache_control set to ephemeral
        
    Example:
        >>> system_msg = create_cached_system_message("You are a helpful assistant...")
        >>> # This message will be cached by supported models
    """
    return SystemMessage(
        content=content,
        cache_control={"type": "ephemeral"}
    )


def create_cached_user_message(content: Union[str, List[Union[str, Image]]], source: str) -> UserMessage:
    """
    Create a UserMessage with ephemeral cache control enabled.
    
    Args:
        content: The message content (string or mixed content)
        source: The name of the agent that sent this message
        
    Returns:
        UserMessage with cache_control set to ephemeral
        
    Example:
        >>> user_msg = create_cached_user_message("Long document content...", "user")
        >>> # This message will be cached by supported models
    """
    return UserMessage(
        content=content,
        source=source,
        cache_control={"type": "ephemeral"}
    )


def create_cached_user_message_with_ttl(
    content: Union[str, List[Union[str, Image]]], 
    source: str, 
    ttl_hours: int = 1
) -> UserMessage:
    """
    Create a UserMessage with cache control and custom TTL.
    
    Args:
        content: The message content
        source: The name of the agent that sent this message
        ttl_hours: Cache time-to-live in hours (1 or 24 typically supported)
        
    Returns:
        UserMessage with cache_control set with TTL
        
    Example:
        >>> user_msg = create_cached_user_message_with_ttl("Codebase context...", "user", ttl_hours=24)
        >>> # This message will be cached for 24 hours (if supported by model)
    """
    if ttl_hours == 1:
        cache_control = {"type": "ephemeral", "ttl": "1h"}
    elif ttl_hours == 24:
        cache_control = {"type": "ephemeral", "ttl": "24h"}
    else:
        # Default to standard ephemeral for unsupported TTLs
        cache_control = {"type": "ephemeral"}
    
    return UserMessage(
        content=content,
        source=source,
        cache_control=cache_control
    )


def create_cached_system_message_with_ttl(content: str, ttl_hours: int = 1) -> SystemMessage:
    """
    Create a SystemMessage with cache control and custom TTL.
    
    Args:
        content: The system message content
        ttl_hours: Cache time-to-live in hours (1 or 24 typically supported)
        
    Returns:
        SystemMessage with cache_control set with TTL
        
    Example:
        >>> system_msg = create_cached_system_message_with_ttl("Long system prompt...", ttl_hours=24)
        >>> # This message will be cached for 24 hours (if supported by model)
    """
    if ttl_hours == 1:
        cache_control = {"type": "ephemeral", "ttl": "1h"}
    elif ttl_hours == 24:
        cache_control = {"type": "ephemeral", "ttl": "24h"}
    else:
        # Default to standard ephemeral for unsupported TTLs
        cache_control = {"type": "ephemeral"}
    
    return SystemMessage(
        content=content,
        cache_control=cache_control
    )


# Cache control constants for consistency
EPHEMERAL_CACHE = {"type": "ephemeral"}
EPHEMERAL_CACHE_1H = {"type": "ephemeral", "ttl": "1h"}  
EPHEMERAL_CACHE_24H = {"type": "ephemeral", "ttl": "24h"}


def create_cached_function_result_message(
    content: List[FunctionExecutionResult]
) -> FunctionExecutionResultMessage:
    """
    Create a FunctionExecutionResultMessage with ephemeral cache control enabled.
    
    Args:
        content: List of function execution results
        
    Returns:
        FunctionExecutionResultMessage with cache_control set to ephemeral
        
    Example:
        >>> results = [FunctionExecutionResult(content="large output...", name="read_file", call_id="call_123")]
        >>> cached_msg = create_cached_function_result_message(results)
        >>> # This message will be cached by supported models
    """
    return FunctionExecutionResultMessage(
        content=content,
        cache_control={"type": "ephemeral"}
    )


def create_cached_function_result_message_with_ttl(
    content: List[FunctionExecutionResult], 
    ttl_hours: int = 1
) -> FunctionExecutionResultMessage:
    """
    Create a FunctionExecutionResultMessage with cache control and custom TTL.
    
    Args:
        content: List of function execution results
        ttl_hours: Cache time-to-live in hours (1 or 24 typically supported)
        
    Returns:
        FunctionExecutionResultMessage with cache_control set with TTL
        
    Example:
        >>> results = [FunctionExecutionResult(content="expensive computation...", name="analyze", call_id="call_456")]
        >>> cached_msg = create_cached_function_result_message_with_ttl(results, ttl_hours=24)
        >>> # This message will be cached for 24 hours (if supported by model)
    """
    if ttl_hours == 1:
        cache_control = {"type": "ephemeral", "ttl": "1h"}
    elif ttl_hours == 24:
        cache_control = {"type": "ephemeral", "ttl": "24h"}
    else:
        # Default to standard ephemeral for unsupported TTLs
        cache_control = {"type": "ephemeral"}
    
    return FunctionExecutionResultMessage(
        content=content,
        cache_control=cache_control
    )


def get_cache_control(cache_type: str = "ephemeral", ttl: str | None = None) -> Dict[str, Any]:
    """
    Get a cache control dictionary for manual use.
    
    Args:
        cache_type: Type of cache control (currently only "ephemeral" is supported)
        ttl: Optional time-to-live ("5m", "1h", "24h", etc.)
        
    Returns:
        Cache control dictionary
        
    Example:
        >>> cache_control = get_cache_control("ephemeral", "1h")
        >>> message = UserMessage(content="...", source="user", cache_control=cache_control)
    """
    cache_control = {"type": cache_type}
    if ttl:
        cache_control["ttl"] = ttl
    return cache_control
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from autogen_core.models import FunctionExecutionResultMessage, SystemMessage, UserMessage


@dataclass
class CacheControl:
    """Cache control configuration for Anthropic prompt caching."""

    type: str = "ephemeral"


class AnthropicSystemMessage(SystemMessage):
    """Anthropic-specific system message with cache control support."""

    cache_control: Optional[CacheControl] = None

    def __init__(self, content: str, cache_control: Optional[CacheControl] = None, **kwargs: Any):
        super().__init__(content=content, **kwargs)
        self.cache_control = cache_control

    @classmethod
    def create_with_cache_control(cls, content: str) -> "AnthropicSystemMessage":
        """Create a system message with ephemeral cache control enabled."""
        return cls(content=content, cache_control=CacheControl(type="ephemeral"))


class AnthropicUserMessage(UserMessage):
    """Anthropic-specific user message with cache control support."""

    cache_control: Optional[CacheControl] = None

    def __init__(
        self, content: Union[str, List[Any]], source: str, cache_control: Optional[CacheControl] = None, **kwargs: Any
    ):
        super().__init__(content=content, source=source, **kwargs)
        self.cache_control = cache_control

    @classmethod
    def create_with_cache_control(cls, content: Union[str, List[Any]], source: str) -> "AnthropicUserMessage":
        """Create a user message with ephemeral cache control enabled."""
        return cls(content=content, source=source, cache_control=CacheControl(type="ephemeral"))


class AnthropicFunctionExecutionResultMessage(FunctionExecutionResultMessage):
    """Anthropic-specific function execution result message with per-result cache control support."""

    cache_control_config: Optional[Dict[int, CacheControl]] = None

    def __init__(
        self, content: List[Any], cache_control_config: Optional[Dict[int, CacheControl]] = None, **kwargs: Any
    ):
        super().__init__(content=content, **kwargs)
        self.cache_control_config = cache_control_config or {}

    def set_cache_control_for_result(self, index: int, cache_control: CacheControl) -> None:
        """Set cache control for a specific tool result by index."""
        if self.cache_control_config is None:
            self.cache_control_config = {}
        self.cache_control_config[index] = cache_control

    @classmethod
    def create_with_cache_control(
        cls, content: List[Any], cached_result_indices: Optional[List[int]] = None
    ) -> "AnthropicFunctionExecutionResultMessage":
        """Create a function execution result message with ephemeral cache control for specified indices."""
        instance = cls(content=content)
        if cached_result_indices:
            for idx in cached_result_indices:
                if idx < len(content):
                    if instance.cache_control_config is None:
                        instance.cache_control_config = {}
                    instance.cache_control_config[idx] = CacheControl(type="ephemeral")
        return instance


def create_ephemeral_cache() -> CacheControl:
    """Create an ephemeral cache control configuration.

    Returns:
        CacheControl object with ephemeral type
    """
    return CacheControl(type="ephemeral")

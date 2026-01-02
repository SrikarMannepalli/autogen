#!/usr/bin/env python3
"""
Test script for output_config/effort parameter support.

Usage:
    python test_effort.py <api_key>

Tests both:
1. Direct beta API usage (should work now)
2. Passthrough via AnthropicChatCompletionClient (will work once promoted to standard API)
"""

import asyncio
import sys
from packaging import version

import anthropic
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import UserMessage

MINIMUM_ANTHROPIC_VERSION = "0.73.0"


def check_anthropic_version():
    """Check if anthropic SDK version supports output_config."""
    current_version = anthropic.__version__
    print(f"Anthropic SDK version: {current_version}")
    if version.parse(current_version) < version.parse(MINIMUM_ANTHROPIC_VERSION):
        print(f"  WARNING: Version {MINIMUM_ANTHROPIC_VERSION}+ required for output_config/effort support")
        print(f"  Upgrade with: pip install 'anthropic>={MINIMUM_ANTHROPIC_VERSION}'")
        return False
    print(f"  Version OK (>= {MINIMUM_ANTHROPIC_VERSION})")
    return True


async def test_beta_api_directly(api_key: str):
    """Test the beta API directly using the anthropic SDK."""
    print("\n" + "=" * 60)
    print("PART 1: Testing Beta API Directly (anthropic SDK)")
    print("=" * 60)

    # Test 1: Beta API with effort=low
    print("\n[Beta Test 1] Using beta API with effort='low'...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.beta.messages.create(
            model="claude-opus-4-5-20251101",
            betas=["effort-2025-11-24"],
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            output_config={"effort": "low"},
        )
        print(f"  Response: {response.content[0].text}")
        print(f"  Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
        print("  [Beta Test 1] PASSED")
    except Exception as e:
        print(f"  [Beta Test 1] FAILED: {e}")

    # Test 2: Beta API with effort=medium
    print("\n[Beta Test 2] Using beta API with effort='medium'...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.beta.messages.create(
            model="claude-opus-4-5-20251101",
            betas=["effort-2025-11-24"],
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'world' and nothing else."}],
            output_config={"effort": "medium"},
        )
        print(f"  Response: {response.content[0].text}")
        print(f"  Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
        print("  [Beta Test 2] PASSED")
    except Exception as e:
        print(f"  [Beta Test 2] FAILED: {e}")

    # Test 3: Beta API with effort=high
    print("\n[Beta Test 3] Using beta API with effort='high'...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.beta.messages.create(
            model="claude-opus-4-5-20251101",
            betas=["effort-2025-11-24"],
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            output_config={"effort": "high"},
        )
        print(f"  Response: {response.content[0].text}")
        print(f"  Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
        print("  [Beta Test 3] PASSED")
    except Exception as e:
        print(f"  [Beta Test 3] FAILED: {e}")

    # Test 4: Beta API streaming with effort
    print("\n[Beta Test 4] Using beta API streaming with effort='low'...")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        chunks = []
        with client.beta.messages.stream(
            model="claude-opus-4-5-20251101",
            betas=["effort-2025-11-24"],
            max_tokens=100,
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            output_config={"effort": "low"},
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
        full_response = "".join(chunks)
        print(f"  Response: {full_response}")
        print("  [Beta Test 4] PASSED")
    except Exception as e:
        print(f"  [Beta Test 4] FAILED: {e}")


async def test_autogen_client_passthrough(api_key: str):
    """Test the passthrough via AnthropicChatCompletionClient."""
    print("\n" + "=" * 60)
    print("PART 2: Testing Passthrough via AnthropicChatCompletionClient")
    print("(Expected to fail until effort is promoted to standard API)")
    print("=" * 60)

    # Test 1: Create client with output_config in constructor
    print("\n[Passthrough Test 1] Creating client with output_config={'effort': 'low'}...")
    try:
        client = AnthropicChatCompletionClient(
            model="claude-opus-4-5-20251101",
            api_key=api_key,
            max_tokens=100,
            output_config={"effort": "low"},
        )
        print("  Client created successfully")

        print("  Sending test message...")
        result = await client.create(
            messages=[UserMessage(content="Say 'hello' and nothing else.", source="user")]
        )
        print(f"  Response: {result.content}")
        print(f"  Usage: {result.usage}")
        print("  [Passthrough Test 1] PASSED")
        await client.close()
    except Exception as e:
        print(f"  [Passthrough Test 1] FAILED (expected): {e}")

    # Test 2: Pass output_config via extra_create_args
    print("\n[Passthrough Test 2] Passing output_config via extra_create_args...")
    try:
        client = AnthropicChatCompletionClient(
            model="claude-opus-4-5-20251101",
            api_key=api_key,
            max_tokens=100,
        )
        print("  Client created successfully")

        print("  Sending test message with extra_create_args={'output_config': {'effort': 'medium'}}...")
        result = await client.create(
            messages=[UserMessage(content="Say 'world' and nothing else.", source="user")],
            extra_create_args={"output_config": {"effort": "medium"}},
        )
        print(f"  Response: {result.content}")
        print(f"  Usage: {result.usage}")
        print("  [Passthrough Test 2] PASSED")
        await client.close()
    except Exception as e:
        print(f"  [Passthrough Test 2] FAILED (expected): {e}")

    # Test 3: Baseline test without output_config (should always work)
    print("\n[Passthrough Test 3] Baseline test WITHOUT output_config (should work)...")
    try:
        client = AnthropicChatCompletionClient(
            model="claude-sonnet-4-20250514",
            api_key=api_key,
            max_tokens=100,
        )
        print("  Client created successfully")

        print("  Sending test message...")
        result = await client.create(
            messages=[UserMessage(content="Say 'baseline test passed' and nothing else.", source="user")]
        )
        print(f"  Response: {result.content}")
        print(f"  Usage: {result.usage}")
        print("  [Passthrough Test 3] PASSED")
        await client.close()
    except Exception as e:
        print(f"  [Passthrough Test 3] FAILED: {e}")


async def main(api_key: str):
    print("=" * 60)
    print("Testing output_config/effort parameter support")
    print("=" * 60)

    # Check SDK version first
    print("\nChecking anthropic SDK version...")
    version_ok = check_anthropic_version()

    if version_ok:
        # Part 1: Test beta API directly
        await test_beta_api_directly(api_key)
    else:
        print("\n[SKIPPED] Beta API tests - SDK version too old")

    # Part 2: Test passthrough via autogen client
    await test_autogen_client_passthrough(api_key)

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    print("\nSummary:")
    if version_ok:
        print("- Beta API tests: Should PASS (effort is available in beta)")
    else:
        print(f"- Beta API tests: SKIPPED (need anthropic >= {MINIMUM_ANTHROPIC_VERSION})")
    print("- Passthrough tests 1-2: Expected to FAIL until standard API supports effort")
    print("- Passthrough test 3: Should PASS (baseline without effort)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <api_key>")
        sys.exit(1)

    api_key = sys.argv[1]
    asyncio.run(main(api_key))

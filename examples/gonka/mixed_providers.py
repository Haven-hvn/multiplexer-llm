#!/usr/bin/env python3
"""
Mixed Providers Example

This example demonstrates combining Gonka with traditional providers:
1. OpenAI as primary provider
2. Gonka as fallback provider
3. Both providers as primary with weighted distribution
4. Health-based routing scenarios

Prerequisites:
    export GONKA_PRIVATE_KEY="0x..."
    export GONKA_SOURCE_URL="https://api.gonka.network"
    export OPENAI_API_KEY="sk-..."  (optional, for OpenAI examples)

Usage:
    python examples/gonka/mixed_providers.py
"""

import asyncio
import os
import sys

from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import (
    register_gonka_models,
    GonkaConfigError,
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
)


def example_openai_primary_gonka_fallback():
    """OpenAI as primary, Gonka as fallback."""
    
    print("=" * 60)
    print("Example 1: OpenAI Primary, Gonka Fallback")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL", "https://api.gonka.network")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    mux = Multiplexer()
    
    # Add OpenAI as primary (if key available)
    if openai_key:
        try:
            from openai import OpenAI
            
            openai_client = OpenAI(api_key=openai_key)
            mux.add_model(
                openai_client,
                weight=100,  # High weight for primary
                model_name="openai-primary",
            )
            print("✓ OpenAI registered as primary (weight=100)")
        except Exception as e:
            print(f"✗ OpenAI registration failed: {e}")
    else:
        print("! OpenAI skipped (OPENAI_API_KEY not set)")
    
    # Add Gonka as fallback
    try:
        result = register_gonka_models(
            mux,
            source_url=source_url,
            private_key=private_key,
            register_as_fallback=True,  # Register as fallback!
            refresh_enabled=False,
            model_name_prefix="gonka-fallback:",
        )
        print(f"✓ Gonka registered as fallback ({result.models_registered} models)")
        
    except GonkaNoParticipantsError:
        print("✗ Gonka registration failed: No participants found")
        # Use static endpoints as demo
        result = register_gonka_models(
            mux,
            private_key=private_key,
            endpoints=["https://demo-node.example.com;gonka1demo123abc456def"],
            register_as_fallback=True,
            refresh_enabled=False,
        )
        print(f"✓ Gonka demo endpoint registered as fallback")
    except GonkaDiscoveryError as e:
        print(f"✗ Gonka discovery failed: {e}")
        return
    
    print("\nRouting behavior:")
    print("  1. Requests go to OpenAI first")
    print("  2. If OpenAI fails, requests fall back to Gonka")
    print("  3. Gonka participants are selected by weight")


def example_weighted_distribution():
    """Both providers as primary with weighted distribution."""
    
    print("\n" + "=" * 60)
    print("Example 2: Weighted Distribution (Both Primary)")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    mux = Multiplexer()
    
    # Add OpenAI with weight 70 (70% of requests)
    if openai_key:
        try:
            from openai import OpenAI
            
            openai_client = OpenAI(api_key=openai_key)
            mux.add_model(
                openai_client,
                weight=70,  # 70% of traffic
                model_name="openai",
            )
            print("✓ OpenAI registered (weight=70, ~70% traffic)")
        except Exception as e:
            print(f"✗ OpenAI failed: {e}")
    else:
        print("! OpenAI skipped")
    
    # Add Gonka with combined weight 30 (30% of requests)
    try:
        result = register_gonka_models(
            mux,
            private_key=private_key,
            endpoints=[
                "https://gonka-node-1.example.com;gonka1node1abc123def456ghi",
                "https://gonka-node-2.example.com;gonka1node2xyz789wvu654tsr",
            ],
            register_as_fallback=False,  # Primary, not fallback
            refresh_enabled=False,
            model_name_prefix="gonka:",
        )
        
        # Note: Each Gonka participant gets weight=1 by default
        # Total Gonka weight will be 2 vs OpenAI's 70
        # To get ~30%, you'd set custom weights
        
        print(f"✓ Gonka registered ({result.models_registered} models)")
        
    except GonkaConfigError as e:
        print(f"✗ Gonka configuration error: {e}")
        return
    
    print("\nRouting behavior:")
    print("  - Requests distributed by weight across all providers")
    print("  - OpenAI gets ~70% if configured")
    print("  - Gonka participants share remaining traffic")


def example_gonka_only_with_openai_fallback():
    """Gonka as primary, OpenAI as fallback."""
    
    print("\n" + "=" * 60)
    print("Example 3: Gonka Primary, OpenAI Fallback")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    mux = Multiplexer()
    
    # Add Gonka as primary
    try:
        result = register_gonka_models(
            mux,
            private_key=private_key,
            endpoints=[
                "https://primary-gonka.example.com;gonka1primary123abc456def",
            ],
            register_as_fallback=False,  # Primary
            refresh_enabled=False,
            model_name_prefix="gonka-primary:",
        )
        print(f"✓ Gonka registered as primary ({result.models_registered} models)")
        
    except GonkaConfigError as e:
        print(f"✗ Gonka error: {e}")
        return
    
    # Add OpenAI as fallback
    if openai_key:
        try:
            from openai import OpenAI
            
            openai_client = OpenAI(api_key=openai_key)
            mux.add_fallback_model(
                openai_client,
                weight=100,
                model_name="openai-fallback",
            )
            print("✓ OpenAI registered as fallback")
        except Exception as e:
            print(f"✗ OpenAI failed: {e}")
    else:
        print("! OpenAI fallback skipped")
    
    print("\nRouting behavior:")
    print("  1. Requests go to Gonka first (decentralized)")
    print("  2. If all Gonka participants fail, fall back to OpenAI")
    print("  3. Good for decentralization with reliability backup")


def example_multiple_providers():
    """Multiple providers scenario."""
    
    print("\n" + "=" * 60)
    print("Example 4: Multiple Providers")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    mux = Multiplexer()
    
    # This example shows how you might configure multiple providers
    # In a real scenario, you'd have actual API keys for each
    
    print("\nConfiguration strategy:")
    print("  Primary tier (total weight ~100):")
    print("    - OpenAI: weight=50 (50%)")
    print("    - Anthropic: weight=30 (30%)")
    print("    - Gonka Network: weight=20 (20%)")
    print("")
    print("  Fallback tier:")
    print("    - Local LLM or secondary Gonka network")
    
    # Register Gonka as part of primary tier
    try:
        result = register_gonka_models(
            mux,
            private_key=private_key,
            endpoints=[
                "https://gonka-primary.example.com;gonka1primary123abc456",
            ],
            register_as_fallback=False,
            refresh_enabled=False,
            model_name_prefix="gonka:",
        )
        print(f"\n✓ Gonka registered: {result.models_registered} models")
    except GonkaConfigError as e:
        print(f"\n✗ Gonka error: {e}")
    
    print("\nThis configuration provides:")
    print("  - Load distribution across providers")
    print("  - Cost optimization (Gonka may be cheaper)")
    print("  - Redundancy (no single point of failure)")
    print("  - Decentralization benefits from Gonka")


async def example_request_routing():
    """Show how requests are routed with mixed providers."""
    
    print("\n" + "=" * 60)
    print("Example 5: Request Routing Demo")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    mux = Multiplexer()
    
    # Set up mixed providers
    try:
        result = register_gonka_models(
            mux,
            private_key=private_key,
            endpoints=[
                "https://gonka-1.example.com;gonka1node1abc123def456ghi",
                "https://gonka-2.example.com;gonka1node2xyz789wvu654tsr",
            ],
            refresh_enabled=False,
        )
        print(f"Registered {result.models_registered} Gonka models")
    except GonkaConfigError as e:
        print(f"Config error: {e}")
        return
    
    print("\nNote: To actually route requests, you would call:")
    print("  response = await mux.chat(messages=[...], model='...')")
    print("")
    print("The multiplexer will:")
    print("  1. Select a model based on weights")
    print("  2. Send the request to that provider")
    print("  3. If it fails, try fallbacks (if configured)")
    print("  4. Return the first successful response")


def main():
    """Run all mixed provider examples."""
    
    print("Gonka Mixed Providers Examples")
    print("=" * 60)
    
    # Check for required keys
    if not os.environ.get("GONKA_PRIVATE_KEY"):
        print("\nWarning: GONKA_PRIVATE_KEY not set")
        print("Most examples will be skipped.\n")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Note: OPENAI_API_KEY not set - OpenAI examples will be skipped\n")
    
    # Run examples
    example_openai_primary_gonka_fallback()
    example_weighted_distribution()
    example_gonka_only_with_openai_fallback()
    example_multiple_providers()
    asyncio.run(example_request_routing())
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

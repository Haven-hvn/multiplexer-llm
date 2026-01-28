#!/usr/bin/env python3
"""
Auto-Discovery Example

This example demonstrates automatic participant discovery from the Gonka network:
1. Using EndpointDiscovery directly for manual control
2. Inspecting discovered participants
3. Querying different epochs

Prerequisites:
    export GONKA_PRIVATE_KEY="0x..."
    export GONKA_SOURCE_URL="https://api.gonka.network"

Usage:
    python examples/gonka/auto_discovery.py
"""

import asyncio
import os
import sys

from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import (
    EndpointDiscovery,
    GonkaClientFactory,
    ModelRegistrar,
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
)


def demonstrate_sync_discovery(source_url: str):
    """Demonstrate synchronous participant discovery."""
    
    print("=" * 60)
    print("Synchronous Discovery")
    print("=" * 60)
    
    # Create discovery service
    discovery = EndpointDiscovery(
        source_url=source_url,
        verify_proofs=False,  # Disable proof verification for speed
        timeout=30.0,
        retry_count=3,
    )
    
    print(f"\nSource URL: {discovery.source_url}")
    print(f"Verify proofs: {discovery.verify_proofs}")
    print(f"Timeout: {discovery.timeout}s")
    
    # Discover current epoch participants
    print("\n--- Discovering current epoch participants ---")
    try:
        participants = discovery.discover()
        
        print(f"\nFound {len(participants)} participants:")
        for p in participants:
            print(f"  Address: {p.address}")
            print(f"    URL: {p.inference_url}")
            print(f"    Weight: {p.weight}")
            print(f"    Models: {p.models or 'all'}")
            print(f"    Epoch: {p.epoch_id}")
            print()
            
    except GonkaNoParticipantsError as e:
        print(f"No participants found: {e}")
    except GonkaDiscoveryError as e:
        print(f"Discovery failed: {e}")
        return
    
    # Get current epoch ID
    print("--- Getting current epoch ID ---")
    try:
        epoch_id = discovery.get_current_epoch()
        print(f"Current epoch: {epoch_id}")
    except GonkaDiscoveryError as e:
        print(f"Failed to get epoch: {e}")


async def demonstrate_async_discovery(source_url: str):
    """Demonstrate asynchronous participant discovery."""
    
    print("\n" + "=" * 60)
    print("Asynchronous Discovery")
    print("=" * 60)
    
    discovery = EndpointDiscovery(
        source_url=source_url,
        verify_proofs=False,
        timeout=30.0,
    )
    
    # Async discovery
    print("\n--- Async discovering participants ---")
    try:
        participants = await discovery.async_discover()
        print(f"Found {len(participants)} participants (async)")
        
    except GonkaNoParticipantsError as e:
        print(f"No participants found: {e}")
    except GonkaDiscoveryError as e:
        print(f"Discovery failed: {e}")


def demonstrate_registration(source_url: str, private_key: str):
    """Demonstrate manual registration with discovered participants."""
    
    print("\n" + "=" * 60)
    print("Manual Registration with Discovered Participants")
    print("=" * 60)
    
    # Create components
    discovery = EndpointDiscovery(source_url=source_url)
    factory = GonkaClientFactory(private_key=private_key)
    registrar = ModelRegistrar(
        factory,
        model_name_prefix="gonka:",
        default_max_concurrent=5,
    )
    
    print(f"\nDerived requester address: {factory.requester_address}")
    
    # Discover participants
    print("\n--- Discovering participants ---")
    try:
        participants = discovery.discover()
        print(f"Found {len(participants)} participants")
    except Exception as e:
        print(f"Discovery failed: {e}")
        return
    
    # Create multiplexer and register
    mux = Multiplexer()
    
    print("\n--- Registering participants ---")
    for participant in participants:
        success = registrar.register_one(
            mux,
            participant,
            as_fallback=False,
        )
        status = "✓" if success else "✗"
        print(f"  {status} {participant.address} (weight={participant.weight})")
    
    # Show registered models
    registered = registrar.get_registered_models()
    print(f"\nTotal registered models: {len(registered)}")
    for model in sorted(registered):
        print(f"  - {model}")


def main():
    """Run the auto-discovery example."""
    
    # Get configuration
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL", "https://api.gonka.network")
    
    if not private_key:
        print("Error: GONKA_PRIVATE_KEY environment variable not set")
        sys.exit(1)
    
    print(f"Gonka Auto-Discovery Example")
    print(f"Source URL: {source_url}")
    
    # Run sync discovery
    demonstrate_sync_discovery(source_url)
    
    # Run async discovery
    asyncio.run(demonstrate_async_discovery(source_url))
    
    # Run manual registration
    demonstrate_registration(source_url, private_key)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

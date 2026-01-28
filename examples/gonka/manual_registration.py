#!/usr/bin/env python3
"""
Manual Registration Example

This example demonstrates manual endpoint configuration without automatic discovery:
1. Using explicit endpoint list
2. Creating participants from known endpoints
3. Fine-grained control over registration

This is useful when:
- You have known, trusted endpoints
- Network discovery is not available
- You want to use specific participants only

Prerequisites:
    export GONKA_PRIVATE_KEY="0x..."

Usage:
    python examples/gonka/manual_registration.py
"""

import asyncio
import os
import sys

from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import (
    register_gonka_models,
    GonkaConfig,
    GonkaParticipant,
    GonkaClientFactory,
    ModelRegistrar,
    GonkaConfigError,
)


def example_using_endpoints_list():
    """Use the endpoints parameter for static configuration."""
    
    print("=" * 60)
    print("Example 1: Using endpoints list")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    mux = Multiplexer()
    
    try:
        # Register using explicit endpoints (url;address format)
        result = register_gonka_models(
            mux,
            private_key=private_key,
            endpoints=[
                "https://node1.gonka.example.com;gonka1abc123def456ghi789jkl012mno345pqr678stu",
                "https://node2.gonka.example.com;gonka1xyz987wvu654tsr321pon098mlk765jih432fed",
            ],
            refresh_enabled=False,  # No refresh for static config
            model_name_prefix="gonka-static:",
        )
        
        print(f"\nRegistered {result.models_registered} models")
        print(f"Epoch ID: {result.epoch_id}")
        
        for p in result.participants:
            print(f"  - {p.address}")
            print(f"    URL: {p.inference_url}")
            print(f"    Weight: {p.weight}")
        
    except GonkaConfigError as e:
        print(f"Configuration error: {e}")


def example_using_gonka_config():
    """Use GonkaConfig for static configuration."""
    
    print("\n" + "=" * 60)
    print("Example 2: Using GonkaConfig with endpoints")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    try:
        config = GonkaConfig(
            private_key=private_key,
            endpoints=[
                "https://trusted-node.example.com;gonka1trusted123abc456def789ghi012jkl345mno678",
            ],
            refresh_enabled=False,
            default_max_concurrent=10,
            model_name_prefix="trusted:",
        )
        
        print(f"\nConfiguration created:")
        print(f"  Endpoints: {config.endpoints}")
        print(f"  Refresh enabled: {config.refresh_enabled}")
        print(f"  Max concurrent: {config.default_max_concurrent}")
        print(f"  Model prefix: {config.model_name_prefix}")
        
        mux = Multiplexer()
        result = register_gonka_models(mux, config=config)
        
        print(f"\nRegistered {result.models_registered} models")
        
    except GonkaConfigError as e:
        print(f"Configuration error: {e}")


def example_manual_participant_creation():
    """Manually create and register participants."""
    
    print("\n" + "=" * 60)
    print("Example 3: Manual participant creation")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    # Create participants manually
    participants = [
        GonkaParticipant(
            address="gonka1highweight123abc456def789ghi012jkl345mno678pqr",
            inference_url="https://high-capacity-node.example.com",
            weight=200,  # High weight - gets more requests
            models=["llama-3.1-70b", "llama-3.1-8b"],
            epoch_id=0,
        ),
        GonkaParticipant(
            address="gonka1lowweight987zyx654wvu321tsr098pon765mlk432jih",
            inference_url="https://backup-node.example.com",
            weight=50,  # Lower weight
            models=[],  # Empty = all models
            epoch_id=0,
        ),
    ]
    
    print(f"\nCreated {len(participants)} participants manually:")
    for p in participants:
        print(f"  - {p.address}")
        print(f"    URL: {p.inference_url}")
        print(f"    Weight: {p.weight}")
        print(f"    Models: {p.models or 'all'}")
    
    # Create factory and registrar
    factory = GonkaClientFactory(private_key=private_key)
    registrar = ModelRegistrar(
        factory,
        model_name_prefix="manual:",
        default_max_concurrent=5,
    )
    
    print(f"\nDerived address: {factory.requester_address}")
    
    # Register with multiplexer
    mux = Multiplexer()
    
    print("\n--- Registering participants ---")
    for participant in participants:
        success = registrar.register_one(
            mux,
            participant,
            as_fallback=False,
        )
        status = "✓ Registered" if success else "✗ Skipped"
        print(f"  {status}: {participant.address}")
    
    # Check registered models
    registered = registrar.get_registered_models()
    print(f"\nTotal registered: {len(registered)} models")


def example_custom_weight_override():
    """Override participant weights during registration."""
    
    print("\n" + "=" * 60)
    print("Example 4: Custom weight override")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    # Create a participant with default weight
    participant = GonkaParticipant(
        address="gonka1default123abc456def789ghi012jkl345mno678pqr901",
        inference_url="https://node.example.com",
        weight=1,  # Default weight from network
        epoch_id=42,
    )
    
    print(f"\nParticipant original weight: {participant.weight}")
    
    factory = GonkaClientFactory(private_key=private_key)
    registrar = ModelRegistrar(factory, model_name_prefix="custom:")
    mux = Multiplexer()
    
    # Register with custom weight
    custom_weight = 500
    success = registrar.register_one(
        mux,
        participant,
        weight_override=custom_weight,  # Override the weight
        max_concurrent=3,  # Also set max concurrent
    )
    
    if success:
        print(f"Registered with custom weight: {custom_weight}")
        print(f"Max concurrent: 3")
    else:
        print("Registration failed")


def main():
    """Run all manual registration examples."""
    
    print("Gonka Manual Registration Examples")
    print("=" * 60)
    
    # Check for private key
    if not os.environ.get("GONKA_PRIVATE_KEY"):
        print("\nWarning: GONKA_PRIVATE_KEY not set")
        print("Some examples will be skipped.\n")
        print("Set it with: export GONKA_PRIVATE_KEY='0x...'")
    
    # Run examples
    example_using_endpoints_list()
    example_using_gonka_config()
    example_manual_participant_creation()
    example_custom_weight_override()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

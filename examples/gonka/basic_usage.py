#!/usr/bin/env python3
"""
Basic Gonka Integration Example

This example demonstrates the simplest way to use Gonka with multiplexer-llm:
1. Create a Multiplexer instance
2. Register Gonka network participants
3. Make a chat completion request

Prerequisites:
    export GONKA_PRIVATE_KEY="0x..."
    export GONKA_SOURCE_URL="https://api.gonka.network"

Usage:
    python examples/gonka/basic_usage.py
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


async def main():
    """Run the basic usage example."""
    
    # Get configuration from environment
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL", "https://api.gonka.network")
    
    if not private_key:
        print("Error: GONKA_PRIVATE_KEY environment variable not set")
        print("Usage: export GONKA_PRIVATE_KEY='0x...'")
        sys.exit(1)
    
    # Create the multiplexer
    mux = Multiplexer()
    
    try:
        # Register all Gonka network participants
        print(f"Registering Gonka models from {source_url}...")
        
        result = register_gonka_models(
            mux,
            source_url=source_url,
            private_key=private_key,
            refresh_enabled=True,  # Enable automatic epoch refresh
            refresh_interval_seconds=60.0,  # Check every 60 seconds
        )
        
        # Display registration results
        print(f"\nRegistered {result.models_registered} Gonka models from epoch {result.epoch_id}")
        print("\nParticipants:")
        for participant in result.participants:
            print(f"  - {participant.address}: {participant.inference_url} (weight={participant.weight})")
        
        # Make a chat completion request
        print("\n--- Making chat completion request ---")
        
        response = await mux.chat(
            messages=[
                {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
            ],
            model="llama-3.1-70b",  # Use model supported by Gonka participants
        )
        
        # Display the response
        print(f"\nResponse: {response.choices[0].message.content}")
        
        # Clean up
        if result.refresh_manager and result.refresh_manager.is_running:
            result.refresh_manager.stop()
            print("\nRefresh manager stopped")
        
    except GonkaConfigError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except GonkaDiscoveryError as e:
        print(f"Discovery error: {e}")
        print(f"Source URL: {e.source_url}")
        if e.status_code:
            print(f"HTTP status: {e.status_code}")
        sys.exit(1)
    except GonkaNoParticipantsError as e:
        print(f"No participants found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

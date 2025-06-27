#!/usr/bin/env python3
"""
HTTP Multiplexer Example

This example demonstrates load balancing using custom HTTP endpoints that are OpenAI compatible.
60% of requests go to http://192.168.68.70:1234, 40% go to http://192.168.68.67:7045.
"""

import asyncio
import os
import sys
from typing import Optional
from collections import defaultdict

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiplexer_llm import Multiplexer
from openai import AsyncOpenAI

try:
    from openai import AsyncOpenAI
except ImportError:
    print("OpenAI package is required. Install it with: pip install openai>=1.0.0")
    sys.exit(1)


async def main():
    """Main example function demonstrating load balancing with custom endpoints."""
    print("Initializing Model Multiplexer with custom endpoints...")
    stats = defaultdict(lambda: {"success": 0, "failure": 0})

    # Create clients for custom endpoints (unauthenticated)
    client1 = AsyncOpenAI(
        api_key="",  # No API key needed for unauthenticated endpoint
        base_url="http://192.168.68.70:1234"
    )
    client2 = AsyncOpenAI(
        api_key="",
        base_url="http://192.168.68.67:7045"
    )

    # Initialize multiplexer and add models with weights 6 and 4 for 60/40 split
    async with Multiplexer() as multiplexer:
        multiplexer.add_model(client1, 6, "custom-endpoint-1")
        multiplexer.add_model(client2, 4, "custom-endpoint-2")

        print(f"Added models: custom-endpoint-1 (weight 6), custom-endpoint-2 (weight 4)")
        print("Running 100 chat completion requests to demonstrate load balancing...")

        for i in range(100):
            try:
                completion = await multiplexer.chat.completions.create(
                    model="custom-multiplexer",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Tell me fun fact number {i+1} about space."}
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                print(f"Chat completion {i+1} received from {completion.model}:")
                print(completion.choices[0].message.content)
                stats[completion.model]["success"] += 1
            except Exception as error:
                print(f"Error during chat completion {i+1}: {error}")
                # Assuming the model that failed is the one that was supposed to be used
                # This might not be accurate as the actual model used isn't directly available on exception
                # For simplicity, we'll attribute the failure to the last used model or a generic 'unknown'
                stats["unknown"]["failure"] += 1

        print("\nFinal Statistics:")
        for model, result in stats.items():
            print(f"Model: {model}, Successful completions: {result['success']}, Failed completions: {result['failure']}")
        print("Overall Model usage statistics:", multiplexer.get_stats())


if __name__ == "__main__":
    asyncio.run(main())

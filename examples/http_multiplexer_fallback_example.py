#!/usr/bin/env python3
"""
Example using the Model Multiplexer with custom HTTP endpoints.

This example demonstrates how to use the Model Multiplexer with unauthenticated,
OpenAI-compatible endpoints at http://192.168.68.67:7045 and http://192.168.68.70:1234.

No API keys are required for these endpoints.
"""

import asyncio
import os
import sys
from typing import Optional

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiplexer_llm import Multiplexer
from openai import AsyncOpenAI

async def create_clients():
    """Create clients for custom endpoints."""
    # Primary endpoint
    primary_client = AsyncOpenAI(
        api_key="dummy_api_key",  # Set a dummy API key
        base_url="http://192.168.68.67:7045/v1",  # Assuming /v1 path for OpenAI compatibility
    )
    
    # Fallback endpoint
    fallback_client = AsyncOpenAI(
        api_key="dummy_api_key",  # Set a dummy API key
        base_url="http://192.168.68.70:1234/v1",  # Assuming /v1 path for OpenAI compatibility
    )
    
    return {
        "primary": primary_client,
        "fallback": fallback_client,
    }

async def run_chat_completion(multiplexer: Multiplexer):
    """Run a chat completion request and return the result."""
    print("\nSending chat completion request...")
    try:
        completion = await multiplexer.chat.completions.create(
            model="custom-endpoint",  # Custom model identifier
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a fact about space."},
            ],
            temperature=0.5,
            max_tokens=1000,
        )
        print("Chat completion received:")
        print(completion.choices[0].message.content)
        return True, completion.model_dump().get('model_name', 'unknown')
    except Exception as error:
        print(f"Error during chat completion: {error}")
        return False, str(error)

async def main():
    """Main example function."""
    print("Initializing Model Multiplexer with custom endpoints...")
    
    # Create clients
    clients = await create_clients()
    
    # Initialize multiplexer
    async with Multiplexer() as multiplexer:
        # Add primary model (first endpoint)
        multiplexer.add_model(clients["primary"], 5, "primary-endpoint")
        
        # Add fallback model (second endpoint)
        multiplexer.add_fallback_model(clients["fallback"], 3, "fallback-endpoint")
        
        print(f"Added primary and fallback models")
        
        # Run multiple chat completions and track results
        total_requests = 30
        successful_responses = 0
        endpoint_results = {"primary-endpoint": 0, "fallback-endpoint": 0}
        failed_requests = []

        for _ in range(total_requests):
            success, endpoint = await run_chat_completion(multiplexer)
            if success:
                successful_responses += 1
                endpoint_results[endpoint] += 1
            else:
                failed_requests.append(endpoint)

        # Summarize results
        print(f"\nTotal requests: {total_requests}")
        print(f"Successful responses: {successful_responses}")
        print("Endpoint results:")
        for endpoint, count in endpoint_results.items():
            print(f"{endpoint}: {count} successful responses")
        if failed_requests:
            print("Failed requests:")
            for failure in failed_requests:
                print(failure)

if __name__ == "__main__":
    asyncio.run(main())

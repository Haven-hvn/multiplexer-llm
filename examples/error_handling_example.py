#!/usr/bin/env python3
"""
Error Handling Example

This example demonstrates how to handle custom exceptions from the multiplexer.
It shows how different types of errors are mapped to specific exception types
that can be caught and handled appropriately.
"""

import asyncio
import os
import sys
from typing import Optional

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiplexer_llm import (
    Multiplexer,
    MultiplexerError,
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    ModelSelectionError,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    print("OpenAI package is required. Install it with: pip install openai>=1.0.0")
    sys.exit(1)


async def demonstrate_error_handling():
    """Demonstrate different types of error handling."""
    print("Error Handling Example")
    print("=" * 50)
    
    # Example endpoints that will likely fail for demonstration
    failing_endpoints = [
        {
            "base_url": "http://localhost:9999/v1",  # Non-existent endpoint
            "api_key": "",
            "name": "non-existent-endpoint",
            "weight": 5
        },
        {
            "base_url": "http://192.168.68.70:1234/v1",  # May have no model loaded
            "api_key": "",
            "name": "lm-studio-no-model",
            "weight": 3
        }
    ]
    
    async with Multiplexer() as multiplexer:
        # Add failing endpoints to demonstrate error handling
        for endpoint in failing_endpoints:
            client = AsyncOpenAI(
                api_key=endpoint["api_key"],
                base_url=endpoint["base_url"]
            )
            multiplexer.add_model(
                client, 
                endpoint["weight"], 
                endpoint["name"],
                base_url=endpoint["base_url"]
            )
            print(f"Added {endpoint['name']} - {endpoint['base_url']}")
        
        print(f"\nTesting error handling with {len(failing_endpoints)} endpoints...")
        
        try:
            completion = await multiplexer.chat.completions.create(
                model="auto",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, can you help me?"}
                ],
                temperature=0.7,
                max_tokens=50,
            )
            
            print(f"✓ Unexpected success! Response: {completion.choices[0].message.content}")
            
        except ModelNotFoundError as e:
            print(f"🔍 Model Not Found Error:")
            print(f"   Status Code: {e.status_code}")
            print(f"   Endpoint: {e.endpoint}")
            print(f"   Model: {e.model_name}")
            print(f"   Message: {e.message}")
            print(f"   This typically means the model is not loaded at the endpoint.")
            
        except AuthenticationError as e:
            print(f"🔐 Authentication Error:")
            print(f"   Status Code: {e.status_code}")
            print(f"   Endpoint: {e.endpoint}")
            print(f"   Model: {e.model_name}")
            print(f"   Message: {e.message}")
            print(f"   This means the API key is invalid or missing.")
            
        except RateLimitError as e:
            print(f"⏱️ Rate Limit Error:")
            print(f"   Status Code: {e.status_code}")
            print(f"   Endpoint: {e.endpoint}")
            print(f"   Model: {e.model_name}")
            print(f"   Retry After: {e.retry_after}s")
            print(f"   Message: {e.message}")
            print(f"   This means all models are rate limited.")
            
        except ServiceUnavailableError as e:
            print(f"🚫 Service Unavailable Error:")
            print(f"   Status Code: {e.status_code}")
            print(f"   Endpoint: {e.endpoint}")
            print(f"   Model: {e.model_name}")
            print(f"   Message: {e.message}")
            print(f"   This means the service is down or unreachable.")
            
        except ModelSelectionError as e:
            print(f"⚠️ Model Selection Error:")
            print(f"   Message: {e.message}")
            print(f"   This means no models are available or all are disabled.")
            
        except MultiplexerError as e:
            print(f"❌ Generic Multiplexer Error:")
            print(f"   Message: {e.message}")
            print(f"   This is a catch-all for other multiplexer-related errors.")
            
        except Exception as e:
            print(f"💥 Unexpected Error:")
            print(f"   Type: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            print(f"   This should not happen with proper error handling.")


async def demonstrate_working_example():
    """Show how error handling works with a mix of working and failing endpoints."""
    print("\n" + "=" * 50)
    print("Mixed Endpoints Example")
    print("=" * 50)
    
    # Mix of potentially working and failing endpoints
    endpoints = [
        {
            "base_url": "http://localhost:9999/v1",  # Will fail
            "api_key": "",
            "name": "failing-endpoint-1",
            "weight": 3
        },
        {
            "base_url": "https://api.openai.com/v1",  # Might work if API key is set
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "name": "openai-gpt-3.5",
            "weight": 5
        },
        {
            "base_url": "http://localhost:8888/v1",  # Will fail
            "api_key": "",
            "name": "failing-endpoint-2",
            "weight": 2
        }
    ]
    
    async with Multiplexer() as multiplexer:
        working_endpoints = 0
        
        for endpoint in endpoints:
            client = AsyncOpenAI(
                api_key=endpoint["api_key"],
                base_url=endpoint["base_url"]
            )
            multiplexer.add_model(
                client, 
                endpoint["weight"], 
                endpoint["name"],
                base_url=endpoint["base_url"]
            )
            
            if endpoint["api_key"]:  # Has API key, might work
                working_endpoints += 1
            
            print(f"Added {endpoint['name']} - {endpoint['base_url']}")
        
        if working_endpoints == 0:
            print("\n⚠️ No working endpoints configured (no API keys set).")
            print("Set OPENAI_API_KEY environment variable to test with a working endpoint.")
        
        print(f"\nTesting with {len(endpoints)} endpoints ({working_endpoints} potentially working)...")
        
        try:
            completion = await multiplexer.chat.completions.create(
                model="auto",
                messages=[
                    {"role": "user", "content": "Say 'Hello from the multiplexer!'"}
                ],
                max_tokens=20,
            )
            
            print(f"✓ Success! Model: {completion.model}")
            print(f"  Response: {completion.choices[0].message.content}")
            
        except MultiplexerError as e:
            print(f"❌ All endpoints failed: {e.message}")
            print(f"   This demonstrates how the multiplexer tries all endpoints")
            print(f"   and only raises an exception when all have failed.")


async def main():
    """Main function to run all examples."""
    await demonstrate_error_handling()
    await demonstrate_working_example()
    
    print("\n" + "=" * 50)
    print("Error Handling Summary")
    print("=" * 50)
    print("The multiplexer now provides specific exception types:")
    print("• ModelNotFoundError - Model not loaded (404)")
    print("• AuthenticationError - Invalid API key (401/403)")
    print("• RateLimitError - Rate limit exceeded (429/529)")
    print("• ServiceUnavailableError - Service down (5xx)")
    print("• ModelSelectionError - No models available")
    print("• MultiplexerError - Base class for all multiplexer errors")
    print("\nThis allows for more granular error handling in your applications!")


if __name__ == "__main__":
    asyncio.run(main())

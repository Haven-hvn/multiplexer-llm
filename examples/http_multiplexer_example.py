#!/usr/bin/env python3
"""
HTTP Multiplexer Example (Fixed Version)

This example demonstrates load balancing using custom HTTP endpoints that are OpenAI compatible.
This version includes proper error handling and guidance for setting up endpoints.

IMPORTANT: Before running this example, ensure you have working OpenAI-compatible endpoints.
Common options include:
1. Local LLM servers (like LM Studio, Ollama, or text-generation-webui)
2. Self-hosted OpenAI-compatible APIs
3. Proxy services that provide OpenAI-compatible interfaces

Example working endpoints:
- LM Studio: http://localhost:1234/v1 (when running with OpenAI compatibility)
- Ollama: http://localhost:11434/v1 (with OpenAI compatibility enabled)
- text-generation-webui: http://localhost:5000/v1 (with OpenAI extension)

To test if your endpoint is working, try:
curl -X POST http://your-endpoint/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
"""

import asyncio
import os
import sys
from typing import Optional
from collections import defaultdict

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiplexer_llm import Multiplexer

try:
    from openai import AsyncOpenAI
except ImportError:
    print("OpenAI package is required. Install it with: pip install openai>=1.0.0")
    sys.exit(1)


async def test_endpoint(base_url: str, api_key: str = "") -> bool:
    """Test if an endpoint is working properly."""
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        # Try a simple completion request
        response = await client.chat.completions.create(
            model="test",  # Most local servers accept any model name
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            timeout=10.0
        )
        
        # Check if response has expected structure
        if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
            print(f"✓ Endpoint {base_url} is working")
            return True
        else:
            print(f"✗ Endpoint {base_url} returned invalid response structure")
            return False
            
    except Exception as e:
        print(f"✗ Endpoint {base_url} failed: {e}")
        return False


async def main():
    """Main example function demonstrating load balancing with custom endpoints."""
    print("HTTP Multiplexer Example (Fixed Version)")
    print("=" * 50)
    
    # Define your endpoints here
    # These are configured for LM Studio hosting qwen/qwen3-8b
    endpoints = [
        {
            "base_url": "http://192.168.68.70:1234/v1",  # LM Studio with qwen/qwen3-8b
            "api_key": "",  # Usually no API key needed for local servers
            "name": "lm-studio-qwen3-8b-1",
            "weight": 6
        },
        {
            "base_url": "http://192.168.68.67:7045/v1",  # Second LM Studio instance
            "api_key": "",
            "name": "lm-studio-qwen3-8b-2", 
            "weight": 4
        }
    ]
    
    print("Testing endpoints...")
    working_endpoints = []
    
    # Test each endpoint
    for endpoint in endpoints:
        is_working = await test_endpoint(endpoint["base_url"], endpoint["api_key"])
        if is_working:
            working_endpoints.append(endpoint)
    
    if not working_endpoints:
        print("\n❌ No working endpoints found!")
        print("\nTo fix this issue:")
        print("1. Start a local LLM server (LM Studio, Ollama, etc.)")
        print("2. Ensure it has OpenAI-compatible API enabled")
        print("3. Update the endpoints in this script to match your setup")
        print("4. Load a model in your LLM server")
        print("\nExample setup with LM Studio:")
        print("- Download and install LM Studio")
        print("- Download a model (e.g., Llama 3.2)")
        print("- Go to Local Server tab")
        print("- Click 'Start Server' (default: http://localhost:1234)")
        print("- Ensure 'OpenAI Compatible' is enabled")
        return
    
    print(f"\n✓ Found {len(working_endpoints)} working endpoint(s)")
    
    # Initialize multiplexer with working endpoints
    stats = defaultdict(lambda: {"success": 0, "failure": 0})
    
    async with Multiplexer() as multiplexer:
        # Add working endpoints to multiplexer
        clients = []
        for endpoint in working_endpoints:
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
            clients.append((client, endpoint))
            print(f"Added {endpoint['name']} (weight {endpoint['weight']}) - {endpoint['base_url']}")
        
        print(f"\nRunning 20 chat completion requests to demonstrate load balancing...")
        
        successful_requests = 0
        failed_requests = 0
        
        for i in range(20):
            try:
                completion = await multiplexer.chat.completions.create(
                    model="auto",  # This will be overridden by the selected model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
                        {"role": "user", "content": f"Tell me a fun fact about the number {i+1}."}
                    ],
                    temperature=0.7,
                    max_tokens=100,
                )
                
                print(f"✓ Request {i+1} completed via {completion.model}:")
                print(f"  {completion.choices[0].message.content.strip()[:100]}...")
                stats[completion.model]["success"] += 1
                successful_requests += 1
                
            except Exception as error:
                print(f"✗ Request {i+1} failed: {error}")
                stats["failed"]["failure"] += 1
                failed_requests += 1
        
        print(f"\n📊 Results Summary:")
        print(f"Successful requests: {successful_requests}")
        print(f"Failed requests: {failed_requests}")
        print(f"Success rate: {(successful_requests/(successful_requests+failed_requests)*100):.1f}%")
        
        print(f"\n📈 Model Usage Statistics:")
        multiplexer_stats = multiplexer.get_stats()
        for model, result in multiplexer_stats.items():
            print(f"  {model}:")
            print(f"    Successful: {result['success']}")
            print(f"    Rate Limited: {result['rateLimited']}")
            print(f"    Failed: {result['failed']}")
        
        if successful_requests > 0:
            print(f"\n🎉 Load balancing is working! Requests were distributed across {len(working_endpoints)} endpoint(s).")
        else:
            print(f"\n⚠️  All requests failed. Check your endpoint configuration and model availability.")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
High-Performance HTTP Multiplexer Example

This example demonstrates advanced parallel processing with the HTTP multiplexer,
emphasizing async efficiency, concurrency, and speed optimization.

Key features:
- Concurrent endpoint testing
- Parallel request processing with controlled concurrency
- Batch processing capabilities
- Performance metrics and timing
- Connection pooling optimization
- Rate limiting and backpressure handling

IMPORTANT: Before running this example, ensure you have working OpenAI-compatible endpoints.
Common options include:
1. Local LLM servers (like LM Studio, Ollama, or text-generation-webui)
2. Self-hosted OpenAI-compatible APIs
3. Proxy services that provide OpenAI-compatible interfaces

Example working endpoints:
- LM Studio: http://localhost:1234/v1 (when running with OpenAI compatibility)
- Ollama: http://localhost:11434/v1 (with OpenAI compatibility enabled)
- text-generation-webui: http://localhost:5000/v1 (with OpenAI extension)
"""

import asyncio
import os
import sys
import time
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import statistics

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiplexer_llm import Multiplexer

try:
    from openai import AsyncOpenAI
    import httpx
except ImportError:
    print("Required packages missing. Install with: pip install openai>=1.0.0 httpx")
    sys.exit(1)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    request_times: List[float] = None
    throughput: float = 0.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    
    def __post_init__(self):
        if self.request_times is None:
            self.request_times = []


class HighPerformanceMultiplexer:
    """High-performance wrapper around the multiplexer with advanced async patterns."""
    
    def __init__(self, max_concurrent_requests: int = 50, connection_pool_size: int = 100):
        self.max_concurrent_requests = max_concurrent_requests
        self.connection_pool_size = connection_pool_size
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.multiplexer = None
        self.metrics = PerformanceMetrics()
        
    async def __aenter__(self):
        # Configure HTTP client with connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=self.connection_pool_size,
            max_connections=self.connection_pool_size * 2,
            keepalive_expiry=30.0
        )
        
        self.multiplexer = Multiplexer()
        await self.multiplexer.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.multiplexer:
            await self.multiplexer.__aexit__(exc_type, exc_val, exc_tb)


async def test_endpoint_concurrent(base_url: str, api_key: str = "", timeout: float = 5.0) -> Tuple[bool, float]:
    """Test endpoint with timing information."""
    start_time = time.time()
    try:
        # Use custom HTTP client with optimized settings
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        ) as http_client:
            
            client = AsyncOpenAI(
                api_key=api_key, 
                base_url=base_url,
                http_client=http_client
            )
            
            response = await client.chat.completions.create(
                model="test",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                timeout=timeout
            )
            
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                elapsed = time.time() - start_time
                return True, elapsed
            else:
                return False, time.time() - start_time
                
    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed


async def test_endpoints_parallel(endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Test all endpoints concurrently for maximum speed."""
    print("Testing endpoints in parallel...")
    start_time = time.time()
    
    # Create tasks for concurrent endpoint testing
    tasks = []
    for endpoint in endpoints:
        task = test_endpoint_concurrent(endpoint["base_url"], endpoint["api_key"])
        tasks.append((endpoint, task))
    
    # Execute all tests concurrently
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    working_endpoints = []
    for i, ((endpoint, _), result) in enumerate(zip(tasks, results)):
        if isinstance(result, Exception):
            print(f"✗ {endpoint['name']} failed: {result}")
        else:
            is_working, response_time = result
            if is_working:
                print(f"✓ {endpoint['name']} working (response time: {response_time:.3f}s)")
                working_endpoints.append(endpoint)
            else:
                print(f"✗ {endpoint['name']} invalid response (time: {response_time:.3f}s)")
    
    total_time = time.time() - start_time
    print(f"Endpoint testing completed in {total_time:.3f}s")
    return working_endpoints


async def make_request_with_timing(multiplexer: Multiplexer, request_id: int, semaphore: asyncio.Semaphore) -> Tuple[bool, float, str]:
    """Make a single request with timing and concurrency control."""
    async with semaphore:  # Control concurrency
        start_time = time.time()
        try:
            completion = await multiplexer.chat.completions.create(
                model="auto",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {"role": "user", "content": f"What's interesting about the number {request_id}? (Keep it brief)"}
                ],
                temperature=0.7,
                max_tokens=50,
                timeout=30.0
            )
            
            elapsed = time.time() - start_time
            return True, elapsed, completion.model
            
        except Exception as e:
            elapsed = time.time() - start_time
            return False, elapsed, str(e)


async def run_parallel_requests(multiplexer: Multiplexer, num_requests: int, max_concurrent: int = 20) -> PerformanceMetrics:
    """Run multiple requests in parallel with controlled concurrency."""
    print(f"\nRunning {num_requests} requests with max concurrency of {max_concurrent}...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    metrics = PerformanceMetrics()
    
    start_time = time.time()
    
    # Create all request tasks
    tasks = [
        make_request_with_timing(multiplexer, i + 1, semaphore)
        for i in range(num_requests)
    ]
    
    # Execute requests in batches to avoid overwhelming the system
    batch_size = max_concurrent * 2
    results = []
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)
        
        # Brief pause between batches to allow for cleanup
        if i + batch_size < len(tasks):
            await asyncio.sleep(0.1)
    
    total_time = time.time() - start_time
    
    # Process results
    successful = 0
    failed = 0
    request_times = []
    model_usage = defaultdict(int)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"✗ Request {i+1} exception: {result}")
            failed += 1
        else:
            success, elapsed, model_or_error = result
            request_times.append(elapsed)
            
            if success:
                successful += 1
                model_usage[model_or_error] += 1
                if i < 5 or i % 20 == 0:  # Show first 5 and every 20th request
                    print(f"✓ Request {i+1} completed in {elapsed:.3f}s via {model_or_error}")
            else:
                failed += 1
                if i < 5:  # Show first few failures
                    print(f"✗ Request {i+1} failed in {elapsed:.3f}s: {model_or_error}")
    
    # Calculate metrics
    metrics.total_requests = num_requests
    metrics.successful_requests = successful
    metrics.failed_requests = failed
    metrics.total_time = total_time
    metrics.request_times = request_times
    metrics.throughput = successful / total_time if total_time > 0 else 0
    
    if request_times:
        metrics.avg_latency = statistics.mean(request_times)
        metrics.p95_latency = statistics.quantiles(request_times, n=20)[18] if len(request_times) >= 20 else max(request_times)
        metrics.p99_latency = statistics.quantiles(request_times, n=100)[98] if len(request_times) >= 100 else max(request_times)
    
    return metrics, model_usage


async def run_batch_processing_demo(multiplexer: Multiplexer, batch_sizes: List[int]) -> Dict[int, PerformanceMetrics]:
    """Demonstrate batch processing with different batch sizes."""
    print(f"\n🚀 Batch Processing Performance Demo")
    print("=" * 50)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        metrics, _ = await run_parallel_requests(multiplexer, batch_size, min(batch_size, 30))
        results[batch_size] = metrics
        
        print(f"  Throughput: {metrics.throughput:.2f} req/s")
        print(f"  Avg latency: {metrics.avg_latency:.3f}s")
        print(f"  Success rate: {(metrics.successful_requests/metrics.total_requests*100):.1f}%")
        
        # Brief pause between batches
        await asyncio.sleep(1)
    
    return results


async def main():
    """Main function demonstrating high-performance parallel processing."""
    print("🚀 High-Performance HTTP Multiplexer Example")
    print("=" * 60)
    
    # Define endpoints with optimized configuration
    endpoints = [
        {
            "base_url": "http://192.168.68.70:1234/v1",
            "api_key": "",
            "name": "lm-studio-qwen3-8b-1",
            "weight": 2
        },
        {
            "base_url": "http://192.168.68.67:7045/v1",
            "api_key": "",
            "name": "lm-studio-qwen3-8b-2", 
            "weight": 8
        }
    ]
    
    # Test endpoints in parallel
    working_endpoints = await test_endpoints_parallel(endpoints)
    
    if not working_endpoints:
        print("\n❌ No working endpoints found!")
        print("\nSetup instructions:")
        print("1. Start local LLM servers (LM Studio, Ollama, etc.)")
        print("2. Enable OpenAI-compatible API")
        print("3. Update endpoint URLs in this script")
        print("4. Ensure models are loaded")
        return
    
    print(f"\n✅ Found {len(working_endpoints)} working endpoint(s)")
    
    # Initialize high-performance multiplexer
    async with HighPerformanceMultiplexer(max_concurrent_requests=30) as hp_multiplexer:
        multiplexer = hp_multiplexer.multiplexer
        
        # Add endpoints with connection pooling
        for endpoint in working_endpoints:
            # Create client with optimized settings
            client = AsyncOpenAI(
                api_key=endpoint["api_key"],
                base_url=endpoint["base_url"],
                max_retries=2,
                timeout=30.0
            )
            
            multiplexer.add_model(
                client, 
                endpoint["weight"], 
                endpoint["name"],
                base_url=endpoint["base_url"]
            )
            print(f"Added {endpoint['name']} (weight {endpoint['weight']})")
        
        # Performance test 1: Standard parallel processing
        print(f"\n📊 Performance Test 1: Parallel Request Processing")
        print("-" * 50)
        metrics, model_usage = await run_parallel_requests(multiplexer, 50, 20)
        
        print(f"\n📈 Performance Results:")
        print(f"  Total requests: {metrics.total_requests}")
        print(f"  Successful: {metrics.successful_requests}")
        print(f"  Failed: {metrics.failed_requests}")
        print(f"  Success rate: {(metrics.successful_requests/metrics.total_requests*100):.1f}%")
        print(f"  Total time: {metrics.total_time:.3f}s")
        print(f"  Throughput: {metrics.throughput:.2f} requests/second")
        print(f"  Average latency: {metrics.avg_latency:.3f}s")
        print(f"  95th percentile latency: {metrics.p95_latency:.3f}s")
        print(f"  99th percentile latency: {metrics.p99_latency:.3f}s")
        
        print(f"\n🎯 Model Distribution:")
        for model, count in model_usage.items():
            percentage = (count / metrics.successful_requests * 100) if metrics.successful_requests > 0 else 0
            print(f"  {model}: {count} requests ({percentage:.1f}%)")
        
        # Performance test 2: Batch processing comparison
        batch_results = await run_batch_processing_demo(multiplexer, [10, 25, 50, 100])
        
        print(f"\n📊 Batch Processing Comparison:")
        print("-" * 50)
        print(f"{'Batch Size':<12} {'Throughput':<12} {'Avg Latency':<12} {'Success Rate':<12}")
        print("-" * 50)
        for batch_size, metrics in batch_results.items():
            print(f"{batch_size:<12} {metrics.throughput:<12.2f} {metrics.avg_latency:<12.3f} {(metrics.successful_requests/metrics.total_requests*100):<12.1f}%")
        
        # Show multiplexer internal stats
        print(f"\n🔧 Multiplexer Internal Statistics:")
        multiplexer_stats = multiplexer.get_stats()
        for model, stats in multiplexer_stats.items():
            print(f"  {model}:")
            print(f"    Successful: {stats['success']}")
            print(f"    Rate Limited: {stats['rateLimited']}")
            print(f"    Failed: {stats['failed']}")
        
        # Performance recommendations
        print(f"\n💡 Performance Insights:")
        if metrics.throughput > 5:
            print("  ✅ Excellent throughput achieved!")
        elif metrics.throughput > 2:
            print("  ✅ Good throughput. Consider increasing concurrency for higher loads.")
        else:
            print("  ⚠️  Low throughput. Check endpoint performance and network latency.")
        
        if metrics.avg_latency < 2.0:
            print("  ✅ Low latency - endpoints are responding quickly.")
        else:
            print("  ⚠️  High latency detected. Consider optimizing endpoint configuration.")
        
        optimal_batch = max(batch_results.items(), key=lambda x: x[1].throughput)
        print(f"  🎯 Optimal batch size for throughput: {optimal_batch[0]} ({optimal_batch[1].throughput:.2f} req/s)")


if __name__ == "__main__":
    # Set event loop policy for better performance on some systems
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())

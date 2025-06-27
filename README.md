# Multiplexer LLM (Python)

**Unlock the Power of Distributed AI** 🚀

A lightweight Python library that combines the quotas of multiple open source LLM providers with a single unified API. Seamlessly distribute your requests across various providers hosting open source models, ensuring maximum throughput and reliability.

## The Problem: Limited AI Resources

- ❌ **Rate Limit Errors**: "Rate limit exceeded" errors hinder your application's performance
- ❌ **Limited Throughput**: Single provider constraints limit your AI capabilities
- ❌ **Unpredictable Failures**: Rate limits can occur at critical moments
- ❌ **Manual Intervention**: Switching providers requires code changes

## The Solution: Unified Access to Multiple Providers

- ✅ **Increased Throughput**: Combine quotas from multiple open source LLM providers
- ✅ **Error Resilience**: Automatic failover when one provider hits rate limits
- ✅ **Seamless Integration**: Compatible with OpenAI SDK for easy adoption
- ✅ **Smart Load Balancing**: Weight-based distribution across providers for optimal performance

## Key Benefits

- 🚀 **Scalable AI**: Combine resources from multiple providers for enhanced capabilities
- 🛡️ **Error Prevention**: Automatic failover minimizes rate limit failures
- ⚡ **High Availability**: Seamless switching between providers ensures continuous operation
- 🔌 **OpenAI SDK Compatibility**: Works with existing OpenAI SDK code
- 📊 **Usage Analytics**: Track provider performance and rate limits

## How It Works

```
Single Model:        [Model A: 10K RPM] ❌ Rate Limit Error at 10,001 requests
Multiple Providers:  [Provider 1: 10K] + [Provider 2: 15K] + [Provider 3: 20K] = 45,000 RPM ✅
Multiple Models:     [Model A: 10K] + [Model B: 50K] + [Model C: 15K] = 75,000 RPM ✅✅
```

## Installation

```bash
pip install multiplexer-llm
```

The package requires Python 3.8+ and automatically installs the OpenAI Python SDK as a dependency.

## Quick Start

```python
import asyncio
import os
from multiplexer_llm import Multiplexer
from openai import AsyncOpenAI

async def main():
    # Create client instances for a few open source models
    model1 = AsyncOpenAI(
        api_key=os.getenv("MODEL1_API_KEY"),
        base_url="https://api.model1.com/v1/",
    )

    model2 = AsyncOpenAI(
        api_key=os.getenv("MODEL2_API_KEY"),
        base_url="https://api.model2.org/v1",
    )

    # Initialize multiplexer
    async with Multiplexer() as multiplexer:
        # Add models with weights
        multiplexer.add_model(model1, 5, "model1-large")
        multiplexer.add_model(model2, 3, "model2-base")

        # Use like a regular OpenAI client
        completion = await multiplexer.chat.completions.create(
            model="placeholder",  # Will be overridden by selected model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )

        print(completion.choices[0].message.content)
        print("Model usage stats:", multiplexer.get_stats())

# Run the async function
asyncio.run(main())
```

### How Primary and Fallback Models Work

The multiplexer operates with a **two-tier system**:

#### **Primary Models** (`add_model`)

- **First choice**: Used when available
- **Weight-based selection**: Higher weights = higher probability of selection

#### **Fallback Models** (`add_fallback_model`)

- **Backup safety net**: Activated when all primary models hit rate limits

## API Examples

### Creating a Multiplexer

```python
from multiplexer_llm import Multiplexer

# Create multiplexer instance
multiplexer = Multiplexer()

# Or use as async context manager (recommended)
async with Multiplexer() as multiplexer:
    # Your code here
    pass
```

### Adding Models

```python
# Add a primary model
multiplexer.add_model(client: AsyncOpenAI, weight: int, model_name: str)

# Add a fallback model
multiplexer.add_fallback_model(client: AsyncOpenAI, weight: int, model_name: str)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## About Haven Network

[Haven Network](https://github.com/haven-hvn) builds open-source tools to help online communities produce high-quality data for multi-modal AI, with a strong focus on local inference and data privacy.

# Gonka Integration User Guide

This guide walks you through using the Gonka integration with multiplexer-llm, from basic setup to advanced usage patterns.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding Gonka](#understanding-gonka)
5. [Configuration Options](#configuration-options)
6. [Usage Patterns](#usage-patterns)
7. [Epoch Handling](#epoch-handling)
8. [Best Practices](#best-practices)
9. [Migration Guide](#migration-guide)

---

## Prerequisites

Before using the Gonka integration, you need:

1. **Python 3.8+** installed
2. **A Gonka private key** - An ECDSA secp256k1 private key for request signing
3. **Network access** to a Gonka source URL (for automatic discovery)

### Obtaining a Private Key

Your Gonka private key is used to sign requests and prove your identity to the network. Never share or commit your private key.

```python
# Example private key format (64 hex characters, optionally with 0x prefix)
private_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
```

---

## Installation

Install multiplexer-llm with Gonka support:

```bash
pip install multiplexer-llm[gonka]
```

This installs the required cryptographic dependencies:
- `ecdsa` - ECDSA signing (secp256k1 curve)
- `bech32` - Address encoding
- `httpx` - Async HTTP client (optional)

Verify the installation:

```python
from multiplexer_llm.gonka import register_gonka_models
print("Gonka support installed successfully!")
```

---

## Quick Start

The simplest way to use Gonka with multiplexer-llm:

```python
import os
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

# Create the multiplexer
mux = Multiplexer()

# Register Gonka network participants
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key=os.environ["GONKA_PRIVATE_KEY"],
)

print(f"Registered {result.models_registered} models from epoch {result.epoch_id}")

# Make a request - multiplexer routes to Gonka participants
response = await mux.chat(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    model="llama-3.1-70b",
)

print(response.choices[0].message.content)
```

---

## Understanding Gonka

### What is Gonka?

Gonka is a decentralized LLM inference network where multiple participants provide inference services. The network uses blockchain state to maintain a registry of active participants, their endpoints, and stake-based weights.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Participant** | A node providing LLM inference services |
| **Epoch** | A time period during which the participant set is fixed |
| **Weight** | Stake-based value determining request distribution |
| **Source URL** | API endpoint for discovering participants |
| **Request Signing** | ECDSA signatures authenticating each request |

### How It Works

1. **Discovery**: The integration queries the Gonka source URL to discover active participants
2. **Registration**: Each participant is registered as a model in the multiplexer
3. **Signing**: Each request is signed with your private key before being sent
4. **Routing**: The multiplexer distributes requests based on participant weights
5. **Refresh**: Background task monitors for epoch changes and updates participants

---

## Configuration Options

### Using Environment Variables

The recommended approach for production:

```bash
export GONKA_PRIVATE_KEY="0x1234..."
export GONKA_SOURCE_URL="https://api.gonka.network"
```

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
# Configuration is read from environment variables
result = register_gonka_models(mux)
```

### Using GonkaConfig

For explicit configuration:

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import GonkaConfig, register_gonka_models

config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    refresh_enabled=True,
    refresh_interval_seconds=60.0,
)

mux = Multiplexer()
result = register_gonka_models(mux, config=config)
```

### Using Direct Arguments

For simple scripts:

```python
result = register_gonka_models(
    mux,
    private_key="0x...",
    source_url="https://api.gonka.network",
    refresh_enabled=True,
    refresh_interval_seconds=30.0,
    model_name_prefix="gonka:",
)
```

See [Configuration Reference](./configuration.md) for all options.

---

## Usage Patterns

### Pattern 1: Auto-Discovery (Recommended)

Automatically discover and register all network participants:

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
)

# All participants are now registered
print(f"Registered {result.models_registered} models")
for participant in result.participants:
    print(f"  - {participant.address}: {participant.inference_url}")
```

### Pattern 2: Manual Endpoints

When you need explicit control over which endpoints to use:

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
result = register_gonka_models(
    mux,
    private_key="0x...",
    endpoints=[
        "https://node1.example.com;gonka1abc...",
        "https://node2.example.com;gonka1def...",
    ],
)
```

### Pattern 3: Mixed Providers

Combine Gonka with traditional providers:

```python
from openai import OpenAI
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()

# Add OpenAI as primary
mux.add_model(
    OpenAI(api_key="sk-..."),
    weight=100,
    model_name="openai",
)

# Add Gonka as fallback
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
    register_as_fallback=True,  # Register as fallback, not primary
)

# Requests go to OpenAI first, fall back to Gonka on failure
```

### Pattern 4: Gonka-Only with Weight Control

Control request distribution among Gonka participants:

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import (
    GonkaClientFactory,
    EndpointDiscovery,
    ModelRegistrar,
)

# Manual setup for fine-grained control
factory = GonkaClientFactory(private_key="0x...")
discovery = EndpointDiscovery(source_url="https://api.gonka.network")
registrar = ModelRegistrar(factory, model_name_prefix="gonka:")

mux = Multiplexer()

# Discover and register with custom weights
participants = discovery.discover()
for participant in participants:
    # Apply custom weight logic
    custom_weight = participant.weight * 2 if participant.weight > 50 else participant.weight
    registrar.register_one(mux, participant, weight_override=custom_weight)
```

---

## Epoch Handling

### What Are Epochs?

The Gonka network operates in epochs - time periods during which the participant set is fixed. When an epoch ends, participants may change (added, removed, or weight adjusted).

### Automatic Refresh (Default)

By default, the integration monitors for epoch changes:

```python
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
    refresh_enabled=True,  # Default
    refresh_interval_seconds=60.0,  # Check every 60 seconds
)

# Background task handles epoch transitions automatically
```

### Epoch Change Callback

Get notified when epochs change:

```python
from multiplexer_llm.gonka import register_gonka_models

def on_epoch_change(old_epoch, new_epoch, participants):
    print(f"Epoch changed: {old_epoch} -> {new_epoch}")
    print(f"New participant count: {len(participants)}")
    # Log, notify, update metrics, etc.

result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
)

# Set callback on the refresh manager
if result.refresh_manager:
    result.refresh_manager.on_epoch_change = on_epoch_change
```

### Manual Refresh

Trigger refresh on demand:

```python
if result.refresh_manager:
    # Synchronous refresh
    refresh_result = result.refresh_manager.refresh_now()
    
    if refresh_result.epoch_changed:
        print(f"Epoch changed: {refresh_result.old_epoch} -> {refresh_result.new_epoch}")
        print(f"Added: {refresh_result.participants_added}")
        print(f"Removed: {refresh_result.participants_removed}")
```

### Disabling Auto-Refresh

For static configurations or testing:

```python
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
    refresh_enabled=False,  # No background refresh
)
```

---

## Best Practices

### 1. Never Commit Private Keys

Use environment variables or secrets management:

```python
import os

private_key = os.environ.get("GONKA_PRIVATE_KEY")
if not private_key:
    raise ValueError("GONKA_PRIVATE_KEY environment variable not set")
```

### 2. Handle Registration Failures Gracefully

```python
from multiplexer_llm.gonka import (
    register_gonka_models,
    GonkaConfigError,
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
)

try:
    result = register_gonka_models(mux, source_url="...", private_key="...")
except GonkaConfigError as e:
    print(f"Configuration error: {e}")
    # Handle invalid config
except GonkaDiscoveryError as e:
    print(f"Discovery failed: {e}")
    # Handle network issues
except GonkaNoParticipantsError as e:
    print(f"No participants found: {e}")
    # Handle empty network
```

### 3. Use Appropriate Refresh Intervals

- **Production**: 60-120 seconds (balance freshness vs. load)
- **Testing**: 5-10 seconds (quick feedback)
- **Static configs**: Disable refresh entirely

### 4. Monitor Epoch Transitions

Log epoch changes for debugging:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("multiplexer_llm.gonka").setLevel(logging.DEBUG)
```

### 5. Combine with Fallbacks

Use Gonka with traditional providers for reliability:

```python
# OpenAI primary, Gonka fallback
mux.add_model(OpenAI(...), weight=100)
register_gonka_models(mux, ..., register_as_fallback=True)
```

### 6. Set Concurrency Limits

Prevent overloading individual participants:

```python
result = register_gonka_models(
    mux,
    source_url="...",
    private_key="...",
    default_max_concurrent=5,  # Max 5 concurrent requests per participant
)
```

---

## Migration Guide

### From Direct OpenAI Usage

**Before:**
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4",
)
```

**After:**
```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
register_gonka_models(mux, source_url="...", private_key="...")

response = await mux.chat(
    messages=[{"role": "user", "content": "Hello"}],
    model="llama-3.1-70b",
)
```

### From Basic Multiplexer

**Before:**
```python
from openai import OpenAI
from multiplexer_llm import Multiplexer

mux = Multiplexer()
mux.add_model(OpenAI(api_key="sk-..."), weight=100)
```

**After:**
```python
from openai import OpenAI
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
mux.add_model(OpenAI(api_key="sk-..."), weight=100)
register_gonka_models(mux, source_url="...", private_key="...", register_as_fallback=True)
```

### From gonka-openai Package

**Before (gonka-openai):**
```python
from gonka_openai import GonkaOpenAI

client = GonkaOpenAI(
    source_url="https://api.gonka.network",
    private_key="0x...",
)
response = client.chat.completions.create(...)
```

**After (multiplexer-llm):**
```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
)
response = await mux.chat(...)
```

Key differences:
- multiplexer-llm handles load balancing across participants
- Automatic epoch refresh is built-in
- Can combine with other providers
- Uses async by default (use `mux.chat_sync()` for sync)

---

## Next Steps

- [API Reference](./api-reference.md) - Complete API documentation
- [Configuration Reference](./configuration.md) - All config options
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
- [Examples](../../examples/gonka/) - Working code examples

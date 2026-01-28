# Gonka Integration for multiplexer-llm

This directory contains documentation for integrating [Gonka](https://gonka.network), a decentralized LLM inference network, with multiplexer-llm.

## Overview

The Gonka integration enables multiplexer-llm to:

- **Route requests** through decentralized Gonka network endpoints
- **Automatically discover** network participants from blockchain state
- **Handle epoch transitions** seamlessly with background refresh
- **Combine** Gonka providers with traditional providers (OpenAI, Anthropic, etc.)

## Quick Start

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

# Create multiplexer
mux = Multiplexer()

# Register all Gonka network participants
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",  # Your ECDSA private key
)

print(f"Registered {result.models_registered} Gonka models")

# Use with multiplexer
response = await mux.chat(
    messages=[{"role": "user", "content": "Hello, world!"}],
    model="llama-3.1-70b",
)
```

## Installation

```bash
# Install with Gonka support
pip install multiplexer-llm[gonka]
```

This installs the required dependencies:
- `ecdsa` - ECDSA signing for request authentication
- `bech32` - Address encoding/decoding
- `httpx` - Async HTTP client (optional, for async discovery)

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](./user-guide.md) | Getting started tutorials and best practices |
| [API Reference](./api-reference.md) | Complete API documentation |
| [Configuration](./configuration.md) | All configuration options explained |
| [Troubleshooting](./troubleshooting.md) | Common issues and solutions |

## Examples

Working code examples are available in [`examples/gonka/`](../../examples/gonka/):

| Example | Description |
|---------|-------------|
| [basic_usage.py](../../examples/gonka/basic_usage.py) | Simple registration and usage |
| [auto_discovery.py](../../examples/gonka/auto_discovery.py) | Automatic participant discovery |
| [manual_registration.py](../../examples/gonka/manual_registration.py) | Manual endpoint control |
| [mixed_providers.py](../../examples/gonka/mixed_providers.py) | Gonka + OpenAI together |
| [epoch_handling.py](../../examples/gonka/epoch_handling.py) | Handling epoch transitions |

## Architecture

The integration consists of several components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Code                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    register_gonka_models()                   │
│              (Primary convenience function)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Endpoint    │  │ GonkaClient     │  │ Model           │
│ Discovery   │  │ Factory         │  │ Registrar       │
│             │  │                 │  │                 │
│ Fetches     │  │ Creates signed  │  │ Registers with  │
│ participants│  │ OpenAI clients  │  │ multiplexer     │
└─────────────┘  └─────────────────┘  └─────────────────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    RefreshManager                            │
│              (Background epoch monitoring)                   │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Documentation

For implementation details, see the phase documentation:

- [Phase 0: Architecture](./phase-0-architecture/) - Design overview
- [Phase 1: Client Adapter](./phase-1-client-adapter/) - Client factory implementation
- [Phase 2: Endpoint Discovery](./phase-2-endpoint-discovery/) - Discovery service
- [Phase 3: Weighted Model Factory](./phase-3-weighted-model-factory/) - Model registration
- [Phase 4: Configuration](./phase-4-configuration/) - Config and convenience function
- [Phase 5: Testing](./phase-5-testing/) - Test strategy and results
- [Phase 6: Documentation](./phase-6-documentation/) - This documentation

## Requirements

- Python 3.8+
- multiplexer-llm >= 0.1.0
- A valid ECDSA private key for Gonka network authentication
- Network access to Gonka source URL (for auto-discovery)

## Support

For issues with the Gonka integration:

1. Check the [Troubleshooting Guide](./troubleshooting.md)
2. Search [existing issues](https://github.com/your-org/multiplexer-llm/issues)
3. Open a new issue with reproduction steps

## License

The Gonka integration is part of multiplexer-llm and shares its license.

# Gonka Integration Module

This module provides integration between multiplexer-llm and the [Gonka](https://gonka.network) decentralized LLM inference network.

## Installation

```bash
pip install multiplexer-llm[gonka]
```

## Quick Start

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
)
print(f"Registered {result.models_registered} models")
```

## Features

- **Automatic Discovery**: Fetch participants from Gonka blockchain state
- **ECDSA Request Signing**: All requests signed with secp256k1
- **Epoch Refresh**: Background monitoring for epoch transitions
- **Weighted Distribution**: Stake-based request routing
- **Flexible Configuration**: Environment variables, config objects, or direct arguments

## Module Structure

```
multiplexer_llm/gonka/
├── __init__.py          # Public API exports
├── config.py            # GonkaConfig and register_gonka_models()
├── types.py             # GonkaParticipant, RefreshResult, etc.
├── exceptions.py        # Gonka-specific exceptions
├── client_factory.py    # GonkaClientFactory for signed clients
├── discovery.py         # EndpointDiscovery for participant discovery
├── registrar.py         # ModelRegistrar for multiplexer registration
└── refresh.py           # RefreshManager for epoch transitions
```

## Documentation

- [User Guide](../../docs/gonka-integration/user-guide.md)
- [API Reference](../../docs/gonka-integration/api-reference.md)
- [Configuration](../../docs/gonka-integration/configuration.md)
- [Troubleshooting](../../docs/gonka-integration/troubleshooting.md)

## Examples

See [examples/gonka/](../../examples/gonka/) for working code examples.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GONKA_PRIVATE_KEY` | ECDSA private key |
| `GONKA_SOURCE_URL` | Participant discovery URL |
| `GONKA_ENDPOINTS` | Static endpoints (comma-separated) |
| `GONKA_VERIFY_PROOF` | Enable ICS23 verification |

## License

Part of multiplexer-llm. See project license.

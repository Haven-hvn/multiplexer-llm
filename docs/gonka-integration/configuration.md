# Gonka Integration Configuration Reference

Complete reference for all configuration options in the Gonka integration.

## Table of Contents

- [Configuration Methods](#configuration-methods)
- [Environment Variables](#environment-variables)
- [GonkaConfig Options](#gonkaconfig-options)
- [register_gonka_models() Parameters](#register_gonka_models-parameters)
- [Priority Order](#priority-order)
- [Example Configurations](#example-configurations)

---

## Configuration Methods

There are three ways to configure the Gonka integration:

1. **Environment Variables** - Best for production deployments
2. **GonkaConfig Object** - Best for programmatic configuration
3. **Direct Arguments** - Best for simple scripts

All methods can be combined, with a clear priority order for resolving conflicts.

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GONKA_PRIVATE_KEY` | ECDSA private key for request signing | `0x1234...` |
| `GONKA_SOURCE_URL` | URL for participant discovery | `https://api.gonka.network` |
| `GONKA_ENDPOINTS` | Comma-separated endpoint list | `https://n1.com;addr1,https://n2.com;addr2` |
| `GONKA_ADDRESS` | Override derived Gonka address | `gonka1abc...` |
| `GONKA_VERIFY_PROOF` | Enable ICS23 proof verification | `true`, `1`, or `yes` |

### Setting Environment Variables

```bash
# Linux/macOS
export GONKA_PRIVATE_KEY="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
export GONKA_SOURCE_URL="https://api.gonka.network"

# Windows PowerShell
$env:GONKA_PRIVATE_KEY = "0x1234..."
$env:GONKA_SOURCE_URL = "https://api.gonka.network"

# In .env file (with python-dotenv)
GONKA_PRIVATE_KEY=0x1234...
GONKA_SOURCE_URL=https://api.gonka.network
```

### Using with register_gonka_models()

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
# Configuration is automatically read from environment
result = register_gonka_models(mux)
```

---

## GonkaConfig Options

### Required Options

#### `private_key`

ECDSA private key for request signing (secp256k1 curve).

| Property | Value |
|----------|-------|
| Type | `str` |
| Default | `""` (empty, must be provided) |
| Env Var | `GONKA_PRIVATE_KEY` |
| Format | 64 hex characters, with or without `0x` prefix |

```python
# With 0x prefix
config = GonkaConfig(private_key="0x1234567890abcdef...")

# Without prefix
config = GonkaConfig(private_key="1234567890abcdef...")
```

**Security:**
- Never commit private keys to version control
- Use environment variables or secrets management
- The key is never logged or exposed in error messages

---

### Endpoint Source Options

You must provide either `source_url` OR `endpoints` (not both required, but at least one).

#### `source_url`

URL for automatic participant discovery from the Gonka network.

| Property | Value |
|----------|-------|
| Type | `Optional[str]` |
| Default | `None` |
| Env Var | `GONKA_SOURCE_URL` |
| Format | Valid HTTP or HTTPS URL |

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
)
```

#### `endpoints`

Explicit list of endpoints when you don't want automatic discovery.

| Property | Value |
|----------|-------|
| Type | `Optional[List[str]]` |
| Default | `None` |
| Env Var | `GONKA_ENDPOINTS` (comma-separated) |
| Format | Each entry: `"url;address"` |

```python
# As list
config = GonkaConfig(
    private_key="0x...",
    endpoints=[
        "https://node1.example.com;gonka1abc...",
        "https://node2.example.com;gonka1def...",
    ],
)

# Via environment variable
# GONKA_ENDPOINTS="https://node1.example.com;gonka1abc...,https://node2.example.com;gonka1def..."
```

---

### Optional Configuration

#### `address`

Override the derived Gonka address.

| Property | Value |
|----------|-------|
| Type | `Optional[str]` |
| Default | `None` (derived from private_key) |
| Env Var | `GONKA_ADDRESS` |

By default, the Gonka address is derived from your private key. Use this option only if you need to override it.

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    address="gonka1custom...",  # Override derived address
)
```

#### `verify_proofs`

Enable ICS23 proof verification for blockchain state.

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `False` |
| Env Var | `GONKA_VERIFY_PROOF` |

When enabled, the integration verifies cryptographic proofs that participant data came from the blockchain. This adds security but requires the `gonka_openai` ICS23 protobuf module.

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    verify_proofs=True,
)
```

#### `refresh_enabled`

Enable automatic background refresh for epoch changes.

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `True` |

When enabled, a background task monitors for epoch changes and automatically updates registered models.

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    refresh_enabled=True,  # Default - enable background refresh
)

# Disable for static configurations
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    refresh_enabled=False,
)
```

#### `refresh_interval_seconds`

How often to check for epoch changes.

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `60.0` |
| Range | Must be positive |

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    refresh_interval_seconds=30.0,  # Check every 30 seconds
)
```

**Guidelines:**
- Production: 60-120 seconds
- Testing: 5-10 seconds
- High-frequency updates: 10-30 seconds

#### `default_max_concurrent`

Default concurrency limit for registered Gonka models.

| Property | Value |
|----------|-------|
| Type | `Optional[int]` |
| Default | `None` (unlimited) |
| Range | Must be non-negative if set |

Limits how many concurrent requests can be sent to each Gonka participant.

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    default_max_concurrent=5,  # Max 5 concurrent per participant
)
```

#### `register_as_fallback`

Register Gonka models as fallbacks instead of primary models.

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `False` |

When `True`, Gonka models are registered as fallback models that are only used when primary models fail.

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    register_as_fallback=True,  # Register as fallbacks
)
```

#### `model_name_prefix`

Prefix for generated model names.

| Property | Value |
|----------|-------|
| Type | `str` |
| Default | `"gonka:"` |
| Range | Cannot be empty |

Model names are generated as `{prefix}{participant_address}`.

```python
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    model_name_prefix="gonka:",  # Results in "gonka:gonka1abc..."
)

# Custom prefix
config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    model_name_prefix="decentralized:",  # Results in "decentralized:gonka1abc..."
)
```

---

## register_gonka_models() Parameters

All `GonkaConfig` options can also be passed directly to `register_gonka_models()`:

```python
result = register_gonka_models(
    mux,
    # Required
    private_key="0x...",
    
    # Endpoint source (one required)
    source_url="https://api.gonka.network",
    # endpoints=["https://node.example.com;gonka1abc..."],
    
    # Optional overrides
    address=None,
    verify_proofs=False,
    
    # Refresh settings
    refresh_enabled=True,
    refresh_interval_seconds=60.0,
    
    # Model settings
    default_max_concurrent=None,
    register_as_fallback=False,
    model_name_prefix="gonka:",
    
    # Or use config object (args override config)
    config=None,
)
```

---

## Priority Order

When the same option is specified multiple ways, the priority is:

1. **Direct argument** to `register_gonka_models()` (highest priority)
2. **GonkaConfig object** field (if config provided)
3. **Environment variable**
4. **Default value** (lowest priority)

### Example

```python
import os

# Set environment variable
os.environ["GONKA_SOURCE_URL"] = "https://env.example.com"

# Create config
config = GonkaConfig(
    private_key="0x...",
    source_url="https://config.example.com",
)

# Call with direct argument
result = register_gonka_models(
    mux,
    config=config,
    source_url="https://direct.example.com",  # This wins!
)
# Result: source_url = "https://direct.example.com"
```

---

## Example Configurations

### Minimal Configuration

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
)
```

### Production Configuration (Environment Variables)

```bash
# .env or environment
export GONKA_PRIVATE_KEY="0x..."
export GONKA_SOURCE_URL="https://api.gonka.network"
```

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
result = register_gonka_models(mux)  # Uses env vars
```

### Production Configuration (GonkaConfig)

```python
import os
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import GonkaConfig, register_gonka_models

config = GonkaConfig(
    private_key=os.environ["GONKA_PRIVATE_KEY"],
    source_url="https://api.gonka.network",
    verify_proofs=True,
    refresh_enabled=True,
    refresh_interval_seconds=120.0,
    default_max_concurrent=10,
)

mux = Multiplexer()
result = register_gonka_models(mux, config=config)
```

### Development Configuration

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import GonkaConfig, register_gonka_models

config = GonkaConfig(
    private_key="0x...",
    source_url="https://testnet.gonka.network",
    verify_proofs=False,  # Faster for development
    refresh_enabled=True,
    refresh_interval_seconds=10.0,  # Quick updates for testing
)

mux = Multiplexer()
result = register_gonka_models(mux, config=config)
```

### Fallback Configuration

```python
from openai import OpenAI
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()

# Add OpenAI as primary
mux.add_model(OpenAI(api_key="sk-..."), weight=100, model_name="openai")

# Add Gonka as fallback
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
    register_as_fallback=True,
)
```

### Static Endpoints Configuration

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
result = register_gonka_models(
    mux,
    private_key="0x...",
    endpoints=[
        "https://trusted-node-1.example.com;gonka1abc...",
        "https://trusted-node-2.example.com;gonka1def...",
    ],
    refresh_enabled=False,  # No refresh for static config
)
```

### High-Availability Configuration

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import GonkaConfig, register_gonka_models

config = GonkaConfig(
    private_key="0x...",
    source_url="https://api.gonka.network",
    verify_proofs=True,
    refresh_enabled=True,
    refresh_interval_seconds=30.0,
    default_max_concurrent=5,
)

mux = Multiplexer()
result = register_gonka_models(mux, config=config)

# Set up epoch change monitoring
if result.refresh_manager:
    def on_change(old_epoch, new_epoch, participants):
        print(f"Epoch transition: {old_epoch} -> {new_epoch}")
        # Alert monitoring system, update metrics, etc.
    
    result.refresh_manager.on_epoch_change = on_change
```

---

## Configuration Validation

`GonkaConfig` validates all configuration at construction time. Invalid configurations raise `GonkaConfigError`:

```python
from multiplexer_llm.gonka import GonkaConfig, GonkaConfigError

try:
    config = GonkaConfig(
        private_key="",  # Invalid: empty
        source_url="not-a-url",  # Invalid: not HTTP/HTTPS
    )
except GonkaConfigError as e:
    print(f"Config error: {e}")
    print(f"Field: {e.field}")  # Which field failed
```

### Validation Rules

| Field | Rule |
|-------|------|
| `private_key` | Required, 64 hex characters |
| `source_url` or `endpoints` | At least one required |
| `source_url` | Valid HTTP/HTTPS URL |
| `endpoints` | Each entry: `"url;address"` format |
| `endpoints` | Cannot be empty list |
| `refresh_interval_seconds` | Must be positive |
| `default_max_concurrent` | Must be non-negative if set |
| `model_name_prefix` | Cannot be empty |

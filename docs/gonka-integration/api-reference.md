# Gonka Integration API Reference

Complete API documentation for the Gonka integration module.

## Table of Contents

- [Primary API](#primary-api)
  - [register_gonka_models()](#register_gonka_models)
  - [GonkaConfig](#gonkaconfig)
  - [GonkaRegistrationResult](#gonkaregistrationresult)
- [Types](#types)
  - [GonkaParticipant](#gonkaparticipant)
  - [RefreshResult](#refreshresult)
- [Core Components](#core-components)
  - [EndpointDiscovery](#endpointdiscovery)
  - [GonkaClientFactory](#gonkaclientfactory)
  - [ModelRegistrar](#modelregistrar)
  - [RefreshManager](#refreshmanager)
- [Exceptions](#exceptions)

---

## Primary API

### register_gonka_models()

```python
def register_gonka_models(
    multiplexer: Multiplexer,
    *,
    source_url: Optional[str] = None,
    endpoints: Optional[List[str]] = None,
    private_key: Optional[str] = None,
    config: Optional[GonkaConfig] = None,
    verify_proofs: Optional[bool] = None,
    refresh_enabled: Optional[bool] = None,
    refresh_interval_seconds: Optional[float] = None,
    default_max_concurrent: Optional[int] = None,
    register_as_fallback: bool = False,
    model_name_prefix: str = "gonka:",
    address: Optional[str] = None,
) -> GonkaRegistrationResult
```

Register Gonka network participants with the multiplexer.

This is the primary convenience function for Gonka integration. It handles all setup including configuration validation, client creation, participant discovery, model registration, and optionally starting background refresh.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multiplexer` | `Multiplexer` | *required* | The Multiplexer instance to register models with |
| `source_url` | `str` | `None` | URL for participant discovery |
| `endpoints` | `List[str]` | `None` | Explicit endpoint list (format: `"url;address"`) |
| `private_key` | `str` | `None` | ECDSA private key for request signing |
| `config` | `GonkaConfig` | `None` | Configuration object (args override config fields) |
| `verify_proofs` | `bool` | `None` | Enable ICS23 proof verification |
| `refresh_enabled` | `bool` | `None` | Enable background epoch refresh |
| `refresh_interval_seconds` | `float` | `None` | How often to check for epoch changes |
| `default_max_concurrent` | `int` | `None` | Default concurrency limit for models |
| `register_as_fallback` | `bool` | `False` | Register as fallback models instead of primary |
| `model_name_prefix` | `str` | `"gonka:"` | Prefix for generated model names |
| `address` | `str` | `None` | Override derived Gonka address |

#### Returns

`GonkaRegistrationResult` - Information about the registration.

#### Raises

| Exception | Condition |
|-----------|-----------|
| `GonkaConfigError` | Configuration is invalid |
| `GonkaDiscoveryError` | Participant discovery fails |
| `GonkaNoParticipantsError` | No participants found |

#### Example

```python
from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import register_gonka_models

mux = Multiplexer()
result = register_gonka_models(
    mux,
    source_url="https://api.gonka.network",
    private_key="0x...",
    refresh_enabled=True,
)
print(f"Registered {result.models_registered} models from epoch {result.epoch_id}")
```

---

### GonkaConfig

```python
@dataclass
class GonkaConfig:
    private_key: str = ""
    source_url: Optional[str] = None
    endpoints: Optional[List[str]] = None
    address: Optional[str] = None
    verify_proofs: bool = False
    refresh_enabled: bool = True
    refresh_interval_seconds: float = 60.0
    default_max_concurrent: Optional[int] = None
    register_as_fallback: bool = False
    model_name_prefix: str = "gonka:"
```

Configuration dataclass for Gonka integration.

Validates configuration at construction time and resolves environment variables as needed.

#### Attributes

| Attribute | Type | Default | Env Var | Description |
|-----------|------|---------|---------|-------------|
| `private_key` | `str` | `""` | `GONKA_PRIVATE_KEY` | ECDSA private key for request signing |
| `source_url` | `str` | `None` | `GONKA_SOURCE_URL` | URL for participant discovery |
| `endpoints` | `List[str]` | `None` | `GONKA_ENDPOINTS` | Explicit endpoint list (format: `"url;address"`) |
| `address` | `str` | `None` | `GONKA_ADDRESS` | Override derived Gonka address |
| `verify_proofs` | `bool` | `False` | `GONKA_VERIFY_PROOF` | Enable ICS23 proof verification |
| `refresh_enabled` | `bool` | `True` | - | Enable background epoch refresh |
| `refresh_interval_seconds` | `float` | `60.0` | - | Interval for epoch checks in seconds |
| `default_max_concurrent` | `int` | `None` | - | Default concurrency limit |
| `register_as_fallback` | `bool` | `False` | - | Register as fallbacks |
| `model_name_prefix` | `str` | `"gonka:"` | - | Prefix for model names |

#### Validation

The configuration validates:
- `private_key` is provided (required)
- `private_key` is 64 hex characters (32 bytes)
- Either `source_url` or `endpoints` is provided (at least one required)
- `source_url` is a valid HTTP/HTTPS URL
- `endpoints` follows `"url;address"` format
- `refresh_interval_seconds` is positive
- `model_name_prefix` is not empty

#### Example

```python
from multiplexer_llm.gonka import GonkaConfig

# From explicit values
config = GonkaConfig(
    private_key="0x1234...",
    source_url="https://api.gonka.network",
    refresh_enabled=True,
)

# From environment variables (set GONKA_PRIVATE_KEY and GONKA_SOURCE_URL)
config = GonkaConfig()

# With explicit endpoints
config = GonkaConfig(
    private_key="0x1234...",
    endpoints=["https://node.example.com;gonka1abc..."],
)
```

---

### GonkaRegistrationResult

```python
@dataclass
class GonkaRegistrationResult:
    models_registered: int
    participants: List[GonkaParticipant]
    refresh_manager: Optional[RefreshManager]
    epoch_id: int
```

Result of registering Gonka models with the multiplexer.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `models_registered` | `int` | Number of models successfully registered |
| `participants` | `List[GonkaParticipant]` | List of discovered participants |
| `refresh_manager` | `RefreshManager` | Background refresh manager (if enabled) |
| `epoch_id` | `int` | The epoch ID for registered participants |

#### Example

```python
result = register_gonka_models(mux, source_url="...", private_key="...")

print(f"Registered: {result.models_registered}")
print(f"Epoch: {result.epoch_id}")
print(f"Participants: {len(result.participants)}")

if result.refresh_manager:
    result.refresh_manager.on_epoch_change = lambda old, new, p: print(f"Epoch: {old} -> {new}")
```

---

## Types

### GonkaParticipant

```python
@dataclass(frozen=True)
class GonkaParticipant:
    address: str
    inference_url: str
    weight: int = 1
    models: List[str] = field(default_factory=list)
    validator_key: str = ""
    epoch_id: int = 0
```

Represents a single Gonka network participant.

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `address` | `str` | *required* | Gonka address (e.g., `"gonka1abc..."`) |
| `inference_url` | `str` | *required* | API endpoint URL |
| `weight` | `int` | `1` | Stake-based weight for selection |
| `models` | `List[str]` | `[]` | Supported model identifiers |
| `validator_key` | `str` | `""` | Validator public key |
| `epoch_id` | `int` | `0` | Epoch when participant was active |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `base_url` | `str` | Inference URL with `/v1` suffix |

#### Example

```python
from multiplexer_llm.gonka import GonkaParticipant

participant = GonkaParticipant(
    address="gonka1abc123...",
    inference_url="https://node.example.com",
    weight=100,
    models=["llama-3.1-70b"],
    epoch_id=42,
)

print(participant.base_url)  # "https://node.example.com/v1"
```

---

### RefreshResult

```python
@dataclass
class RefreshResult:
    success: bool
    epoch_changed: bool = False
    old_epoch: Optional[int] = None
    new_epoch: Optional[int] = None
    participants_added: int = 0
    participants_removed: int = 0
    error: Optional[Exception] = None
```

Result of a refresh operation from the RefreshManager.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | `bool` | Whether refresh completed without errors |
| `epoch_changed` | `bool` | Whether the epoch changed |
| `old_epoch` | `int` | Previous epoch ID (if changed) |
| `new_epoch` | `int` | New epoch ID (if changed) |
| `participants_added` | `int` | Number of participants added |
| `participants_removed` | `int` | Number of participants removed |
| `error` | `Exception` | The error if refresh failed |

#### Example

```python
result = refresh_manager.refresh_now()

if result.success:
    if result.epoch_changed:
        print(f"Epoch: {result.old_epoch} -> {result.new_epoch}")
        print(f"Added: {result.participants_added}")
        print(f"Removed: {result.participants_removed}")
else:
    print(f"Refresh failed: {result.error}")
```

---

## Core Components

### EndpointDiscovery

```python
class EndpointDiscovery:
    def __init__(
        self,
        source_url: str,
        verify_proofs: bool = False,
        timeout: float = 30.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> None: ...
```

Discovers Gonka network participants from blockchain state.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_url` | `str` | *required* | Base URL for participant discovery |
| `verify_proofs` | `bool` | `False` | Verify ICS23 proofs |
| `timeout` | `float` | `30.0` | HTTP request timeout in seconds |
| `retry_count` | `int` | `3` | Number of retry attempts |
| `retry_delay` | `float` | `1.0` | Delay between retries in seconds |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `source_url` | `str` | The source URL for discovery |
| `verify_proofs` | `bool` | Whether proof verification is enabled |
| `timeout` | `float` | HTTP timeout in seconds |

#### Methods

##### discover()

```python
def discover(self) -> List[GonkaParticipant]
```

Fetch current epoch's participants.

**Returns:** List of `GonkaParticipant` objects.

**Raises:**
- `GonkaDiscoveryError` - Network failure or invalid response
- `GonkaProofVerificationError` - If verification enabled and fails
- `GonkaNoParticipantsError` - If no participants found

##### discover_for_epoch()

```python
def discover_for_epoch(self, epoch: str) -> List[GonkaParticipant]
```

Fetch participants for a specific epoch.

**Parameters:**
- `epoch` - Epoch specifier: `"current"`, `"next"`, or numeric epoch ID

**Returns:** List of `GonkaParticipant` objects.

##### async_discover()

```python
async def async_discover(self) -> List[GonkaParticipant]
```

Async version of `discover()`.

##### async_discover_for_epoch()

```python
async def async_discover_for_epoch(self, epoch: str) -> List[GonkaParticipant]
```

Async version of `discover_for_epoch()`.

##### get_current_epoch()

```python
def get_current_epoch(self) -> int
```

Get the current epoch number.

**Returns:** The current epoch ID as an integer.

#### Example

```python
from multiplexer_llm.gonka import EndpointDiscovery

discovery = EndpointDiscovery(
    source_url="https://api.gonka.network",
    verify_proofs=False,
    timeout=30.0,
)

# Sync discovery
participants = discovery.discover()

# Async discovery
participants = await discovery.async_discover()

# Specific epoch
participants = discovery.discover_for_epoch("42")

# Get current epoch ID
epoch_id = discovery.get_current_epoch()
```

---

### GonkaClientFactory

```python
class GonkaClientFactory:
    def __init__(self, private_key: str) -> None: ...
```

Factory for creating OpenAI-compatible clients with ECDSA request signing.

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `private_key` | `str` | ECDSA private key (with or without `0x` prefix) |

**Raises:**
- `GonkaClientError` - If private key is invalid or address derivation fails
- `ImportError` - If Gonka dependencies are not installed

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `requester_address` | `str` | The derived Gonka address |

#### Methods

##### create_client()

```python
def create_client(self, participant: GonkaParticipant) -> OpenAI
```

Create an OpenAI-compatible client for a participant.

The returned client automatically:
- Signs all requests with ECDSA
- Includes `X-Requester-Address` header
- Includes `X-Timestamp` header
- Uses the participant's address as transfer address

**Parameters:**
- `participant` - The Gonka participant to create a client for

**Returns:** An `OpenAI` client instance.

**Raises:** `GonkaClientError` - If client creation fails.

##### create_async_client()

```python
def create_async_client(self, participant: GonkaParticipant) -> AsyncOpenAI
```

Create an async OpenAI-compatible client.

**Returns:** An `AsyncOpenAI` client instance.

#### Example

```python
from multiplexer_llm.gonka import GonkaClientFactory, GonkaParticipant

factory = GonkaClientFactory(private_key="0x...")
print(f"Address: {factory.requester_address}")

participant = GonkaParticipant(
    address="gonka1abc...",
    inference_url="https://node.example.com",
    weight=100,
)

# Sync client
client = factory.create_client(participant)
response = client.chat.completions.create(...)

# Async client
async_client = factory.create_async_client(participant)
response = await async_client.chat.completions.create(...)
```

---

### ModelRegistrar

```python
class ModelRegistrar:
    def __init__(
        self,
        client_factory: GonkaClientFactory,
        model_name_prefix: str = "gonka:",
        default_max_concurrent: Optional[int] = None,
    ) -> None: ...
```

Registers Gonka participants as models in the multiplexer.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client_factory` | `GonkaClientFactory` | *required* | Factory for creating clients |
| `model_name_prefix` | `str` | `"gonka:"` | Prefix for model names |
| `default_max_concurrent` | `int` | `None` | Default concurrency limit |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `client_factory` | `GonkaClientFactory` | The client factory |
| `model_name_prefix` | `str` | The model name prefix |
| `default_max_concurrent` | `int` | Default concurrency limit |

#### Methods

##### register_all()

```python
def register_all(
    self,
    multiplexer: Multiplexer,
    participants: List[GonkaParticipant],
    as_fallback: bool = False,
) -> int
```

Register all participants as models.

**Parameters:**
- `multiplexer` - The Multiplexer instance
- `participants` - List of participants to register
- `as_fallback` - Register as fallback models

**Returns:** Number of successfully registered models.

##### register_one()

```python
def register_one(
    self,
    multiplexer: Multiplexer,
    participant: GonkaParticipant,
    as_fallback: bool = False,
    weight_override: Optional[int] = None,
    max_concurrent: Optional[int] = None,
) -> bool
```

Register a single participant as a model.

**Parameters:**
- `multiplexer` - The Multiplexer instance
- `participant` - The participant to register
- `as_fallback` - Register as fallback
- `weight_override` - Override participant weight
- `max_concurrent` - Override concurrency limit

**Returns:** `True` if registered, `False` if skipped (e.g., duplicate).

##### unregister_all()

```python
def unregister_all(self, multiplexer: Multiplexer, epoch_id: int) -> int
```

Remove all models from a specific epoch.

**Parameters:**
- `multiplexer` - The Multiplexer instance
- `epoch_id` - The epoch whose models to remove

**Returns:** Number of unregistered (disabled) models.

##### get_registered_models()

```python
def get_registered_models(self, epoch_id: Optional[int] = None) -> Set[str]
```

Get registered model names.

**Parameters:**
- `epoch_id` - Return only models for this epoch (or all if `None`)

**Returns:** Set of model names.

#### Example

```python
from multiplexer_llm.gonka import GonkaClientFactory, ModelRegistrar

factory = GonkaClientFactory(private_key="0x...")
registrar = ModelRegistrar(factory, model_name_prefix="gonka:")

# Register all participants
count = registrar.register_all(mux, participants)

# Register one with custom weight
registrar.register_one(mux, participant, weight_override=200)

# Get registered models
models = registrar.get_registered_models()
print(f"Registered: {models}")
```

---

### RefreshManager

```python
class RefreshManager:
    def __init__(
        self,
        multiplexer: Multiplexer,
        discovery: EndpointDiscovery,
        registrar: ModelRegistrar,
        as_fallback: bool = False,
    ) -> None: ...
```

Background service that monitors for epoch changes and updates models.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multiplexer` | `Multiplexer` | *required* | The Multiplexer to manage |
| `discovery` | `EndpointDiscovery` | *required* | Discovery service |
| `registrar` | `ModelRegistrar` | *required* | Model registrar |
| `as_fallback` | `bool` | `False` | Register as fallbacks |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `multiplexer` | `Multiplexer` | The managed multiplexer |
| `discovery` | `EndpointDiscovery` | The discovery service |
| `registrar` | `ModelRegistrar` | The registrar |
| `current_epoch` | `int` | Current epoch ID |
| `last_refresh` | `datetime` | Last successful refresh time |
| `is_running` | `bool` | Whether background task is running |
| `on_epoch_change` | `Callable` | Epoch change callback |

#### Methods

##### start()

```python
def start(self, interval_seconds: float = 60.0) -> None
```

Start the background refresh task.

**Parameters:**
- `interval_seconds` - How often to check for changes

**Raises:** `RuntimeError` - If called outside an asyncio event loop.

##### stop()

```python
def stop(self) -> None
```

Stop the background refresh task.

##### async_stop()

```python
async def async_stop(self) -> None
```

Stop the background task asynchronously.

##### refresh_now()

```python
def refresh_now(self) -> RefreshResult
```

Trigger an immediate synchronous refresh.

**Returns:** `RefreshResult` with refresh outcome.

##### async_refresh_now()

```python
async def async_refresh_now(self) -> RefreshResult
```

Trigger an immediate asynchronous refresh.

**Returns:** `RefreshResult` with refresh outcome.

#### Callback Signature

```python
def on_epoch_change(
    old_epoch: int,
    new_epoch: int,
    participants: List[GonkaParticipant],
) -> None: ...
```

#### Example

```python
from multiplexer_llm.gonka import (
    EndpointDiscovery,
    GonkaClientFactory,
    ModelRegistrar,
    RefreshManager,
)

factory = GonkaClientFactory(private_key="0x...")
discovery = EndpointDiscovery(source_url="https://api.gonka.network")
registrar = ModelRegistrar(factory)

manager = RefreshManager(
    multiplexer=mux,
    discovery=discovery,
    registrar=registrar,
)

# Set callback
manager.on_epoch_change = lambda old, new, p: print(f"Epoch: {old} -> {new}")

# Start background refresh
manager.start(interval_seconds=60.0)

# Manual refresh
result = manager.refresh_now()

# Stop when done
manager.stop()
```

---

## Exceptions

All Gonka exceptions inherit from `GonkaError`, which inherits from `MultiplexerError`.

### Exception Hierarchy

```
MultiplexerError
└── GonkaError
    ├── GonkaClientError
    ├── GonkaConfigError
    ├── GonkaDiscoveryError
    ├── GonkaNoParticipantsError
    ├── GonkaProofVerificationError
    └── GonkaRefreshError
```

### GonkaError

```python
class GonkaError(MultiplexerError):
    message: str
```

Base exception for all Gonka-related errors.

### GonkaClientError

```python
class GonkaClientError(GonkaError):
    participant_address: Optional[str]
    cause: Optional[Exception]
```

Raised when client creation or configuration fails.

**Note:** Never includes private key in message or attributes.

### GonkaConfigError

```python
class GonkaConfigError(GonkaError):
    field: Optional[str]
```

Raised when configuration is invalid.

### GonkaDiscoveryError

```python
class GonkaDiscoveryError(GonkaError):
    source_url: Optional[str]
    status_code: Optional[int]
    cause: Optional[Exception]
```

Raised when endpoint discovery fails.

### GonkaNoParticipantsError

```python
class GonkaNoParticipantsError(GonkaError):
    epoch: Optional[str]
    source_url: Optional[str]
```

Raised when discovery returns no participants.

### GonkaProofVerificationError

```python
class GonkaProofVerificationError(GonkaError):
    expected_hash: Optional[bytes]
    computed_hash: Optional[bytes]
    proof_type: Optional[str]
```

Raised when ICS23 proof verification fails.

### GonkaRefreshError

```python
class GonkaRefreshError(GonkaError):
    last_successful_epoch: Optional[int]
    cause: Optional[Exception]
```

Raised when background refresh fails.

### Example: Exception Handling

```python
from multiplexer_llm.gonka import (
    register_gonka_models,
    GonkaError,
    GonkaConfigError,
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
)

try:
    result = register_gonka_models(mux, source_url="...", private_key="...")
except GonkaConfigError as e:
    print(f"Config error in field '{e.field}': {e.message}")
except GonkaDiscoveryError as e:
    print(f"Discovery failed for {e.source_url}: {e.message}")
    if e.status_code:
        print(f"HTTP status: {e.status_code}")
except GonkaNoParticipantsError as e:
    print(f"No participants for epoch {e.epoch}")
except GonkaError as e:
    print(f"Gonka error: {e.message}")

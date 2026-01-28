# Gonka Integration Troubleshooting Guide

Solutions to common issues when using the Gonka integration with multiplexer-llm.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Errors](#configuration-errors)
- [Discovery Errors](#discovery-errors)
- [Request Signing Errors](#request-signing-errors)
- [Epoch and Refresh Issues](#epoch-and-refresh-issues)
- [Runtime Errors](#runtime-errors)
- [Debugging Tips](#debugging-tips)
- [FAQ](#faq)

---

## Installation Issues

### Missing Gonka Dependencies

**Error:**
```
ImportError: Gonka support requires additional packages: ecdsa, bech32. 
Install with: pip install multiplexer-llm[gonka]
```

**Solution:**
```bash
pip install multiplexer-llm[gonka]
```

If you're using a `requirements.txt`:
```
multiplexer-llm[gonka]>=0.1.0
```

---

### httpx Not Found (Async Discovery)

**Error:**
```
httpx not installed, falling back to sync HTTP
```

**Impact:** Async discovery falls back to synchronous HTTP, which may affect performance but still works.

**Solution:**
```bash
pip install httpx
```

---

### ICS23 Protobuf Not Available

**Error (when verify_proofs=True):**
```
ICS23 protobuf not available, skipping proof verification
```

**Impact:** Proof verification is skipped. If security is critical, you need the gonka-openai package.

**Solution:**
```bash
pip install gonka-openai
```

---

## Configuration Errors

### Missing Private Key

**Error:**
```
GonkaConfigError: Private key is required. 
Provide via private_key argument or GONKA_PRIVATE_KEY environment variable.
(field: private_key)
```

**Solution:**

```python
# Option 1: Direct argument
result = register_gonka_models(mux, private_key="0x...", ...)

# Option 2: Environment variable
import os
os.environ["GONKA_PRIVATE_KEY"] = "0x..."
result = register_gonka_models(mux, source_url="...")

# Option 3: GonkaConfig
config = GonkaConfig(private_key="0x...", source_url="...")
```

---

### Invalid Private Key Format

**Error:**
```
GonkaConfigError: Invalid private key format: expected 64 hex characters (32 bytes).
(field: private_key)
```

**Cause:** Private key is not the correct length or format.

**Solution:**

The private key must be exactly 64 hexadecimal characters (32 bytes). The `0x` prefix is optional.

```python
# Correct formats:
private_key = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
private_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"

# Incorrect (too short):
private_key = "1234"  # Wrong!
```

---

### Missing Endpoint Source

**Error:**
```
GonkaConfigError: Endpoint source is required. 
Provide source_url, endpoints argument, or set GONKA_SOURCE_URL / GONKA_ENDPOINTS.
(field: source_url)
```

**Solution:**

Provide either `source_url` (for automatic discovery) or `endpoints` (for manual configuration):

```python
# Option 1: Automatic discovery
result = register_gonka_models(
    mux,
    private_key="0x...",
    source_url="https://api.gonka.network",
)

# Option 2: Manual endpoints
result = register_gonka_models(
    mux,
    private_key="0x...",
    endpoints=["https://node.example.com;gonka1abc..."],
)
```

---

### Invalid URL Format

**Error:**
```
GonkaConfigError: Invalid URL: not-a-url. Must be a valid HTTP or HTTPS URL.
(field: source_url)
```

**Solution:**

Use a properly formatted URL with `http://` or `https://` scheme:

```python
# Correct
source_url = "https://api.gonka.network"

# Incorrect
source_url = "api.gonka.network"  # Missing scheme
source_url = "ftp://api.gonka.network"  # Wrong scheme
```

---

### Invalid Endpoint Format

**Error:**
```
GonkaConfigError: Invalid endpoint format: https://node.example.com. 
Expected format: 'url;address' (e.g., 'https://api.example.com;gonka1abc...')
(field: endpoints)
```

**Solution:**

Each endpoint must be in `"url;address"` format:

```python
# Correct
endpoints = [
    "https://node.example.com;gonka1abc123...",
    "https://node2.example.com;gonka1def456...",
]

# Incorrect - missing address
endpoints = ["https://node.example.com"]
```

---

## Discovery Errors

### Network Connection Failed

**Error:**
```
GonkaDiscoveryError: Failed to fetch participants after 3 attempts
(source_url: https://api.gonka.network)
```

**Causes:**
- Network connectivity issues
- Firewall blocking outbound connections
- DNS resolution failure
- Source URL is incorrect or unreachable

**Solutions:**

1. **Check network connectivity:**
   ```bash
   curl -v https://api.gonka.network/v1/epochs/current/participants
   ```

2. **Verify the source URL is correct:**
   ```python
   # Make sure URL is correct and reachable
   source_url = "https://api.gonka.network"  # Not https://gonka.network
   ```

3. **Check firewall rules** - ensure outbound HTTPS is allowed

4. **Increase timeout for slow networks:**
   ```python
   from multiplexer_llm.gonka import EndpointDiscovery
   
   discovery = EndpointDiscovery(
       source_url="https://api.gonka.network",
       timeout=60.0,  # Increase from default 30s
       retry_count=5,  # More retries
   )
   ```

---

### HTTP Error Response

**Error:**
```
GonkaDiscoveryError: HTTP 404 from https://api.gonka.network/v1/epochs/current/participants
(source_url: https://api.gonka.network, status_code: 404)
```

**Causes:**
- Incorrect source URL
- API endpoint changed
- Server-side issues

**Solutions:**

1. **Verify the API URL** with the Gonka network documentation
2. **Check for server-side issues** on Gonka network status page
3. **Try an alternative source URL** if available

---

### No Participants Found

**Error:**
```
GonkaNoParticipantsError: No participants found in response
(epoch: current, source_url: https://api.gonka.network)
```

**Causes:**
- Network has no active participants
- Querying the wrong epoch
- Network is in transition state

**Solutions:**

1. **Check if network has participants:**
   ```bash
   curl https://api.gonka.network/v1/epochs/current/participants
   ```

2. **Try a different epoch:**
   ```python
   from multiplexer_llm.gonka import EndpointDiscovery
   
   discovery = EndpointDiscovery(source_url="https://api.gonka.network")
   participants = discovery.discover_for_epoch("next")
   ```

3. **Use static endpoints** as fallback:
   ```python
   result = register_gonka_models(
       mux,
       private_key="0x...",
       endpoints=["https://known-node.example.com;gonka1abc..."],
   )
   ```

---

### Invalid JSON Response

**Error:**
```
GonkaDiscoveryError: Invalid JSON response from https://api.gonka.network/v1/epochs/current/participants
```

**Causes:**
- Server returning non-JSON response (HTML error page, etc.)
- Response format changed
- Intermediate proxy modifying response

**Solutions:**

1. **Check response manually:**
   ```bash
   curl -v https://api.gonka.network/v1/epochs/current/participants
   ```

2. **Check for proxy interference** - try direct connection
3. **Contact Gonka network support** if issue persists

---

## Request Signing Errors

### Address Derivation Failed

**Error:**
```
GonkaClientError: Failed to derive Gonka address from private key
```

**Causes:**
- Invalid private key bytes
- Missing cryptographic dependencies

**Solutions:**

1. **Verify private key format:**
   ```python
   # Key should be valid hex
   private_key = "1234567890abcdef..."  # 64 hex characters
   
   # Verify it's valid hex
   try:
       bytes.fromhex(private_key.replace("0x", ""))
   except ValueError:
       print("Invalid hex!")
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install --force-reinstall ecdsa bech32
   ```

---

### Request Signing Failed

**Error:**
```
GonkaClientError: Failed to sign request
```

**Causes:**
- Corrupted private key
- Memory issues during signing

**Solutions:**

1. **Verify private key is correct** - ensure no truncation or corruption
2. **Check system resources** - ensure sufficient memory
3. **Try recreating the client factory:**
   ```python
   factory = GonkaClientFactory(private_key="0x...")
   print(f"Address: {factory.requester_address}")  # Should work
   ```

---

## Epoch and Refresh Issues

### Background Refresh Not Starting

**Error:**
```
RuntimeError: There is no current event loop in thread 'MainThread'.
```

**Cause:** `start()` called outside an asyncio event loop.

**Solutions:**

1. **Call from async context:**
   ```python
   import asyncio
   
   async def main():
       result = register_gonka_models(mux, ...)
       if result.refresh_manager:
           result.refresh_manager.start(interval_seconds=60.0)
   
   asyncio.run(main())
   ```

2. **Start manually later:**
   ```python
   result = register_gonka_models(mux, ..., refresh_enabled=False)
   
   # Later, in an async context:
   if result.refresh_manager:
       result.refresh_manager.start()
   ```

3. **Disable auto-refresh:**
   ```python
   result = register_gonka_models(mux, ..., refresh_enabled=False)
   # Manually refresh when needed:
   result.refresh_manager.refresh_now()
   ```

---

### Epoch Change Not Detected

**Symptoms:**
- Models not updating after epoch transition
- Old participants still being used

**Solutions:**

1. **Check refresh is running:**
   ```python
   if result.refresh_manager:
       print(f"Running: {result.refresh_manager.is_running}")
       print(f"Current epoch: {result.refresh_manager.current_epoch}")
       print(f"Last refresh: {result.refresh_manager.last_refresh}")
   ```

2. **Trigger manual refresh:**
   ```python
   refresh_result = result.refresh_manager.refresh_now()
   print(f"Success: {refresh_result.success}")
   print(f"Epoch changed: {refresh_result.epoch_changed}")
   ```

3. **Check refresh interval** - ensure it's not too long

---

### Refresh Callback Not Called

**Symptoms:**
- `on_epoch_change` callback not firing

**Solutions:**

1. **Verify callback is set:**
   ```python
   def my_callback(old_epoch, new_epoch, participants):
       print(f"Callback fired: {old_epoch} -> {new_epoch}")
   
   result.refresh_manager.on_epoch_change = my_callback
   
   # Verify
   print(f"Callback set: {result.refresh_manager.on_epoch_change is not None}")
   ```

2. **Check for exceptions in callback:**
   ```python
   def safe_callback(old_epoch, new_epoch, participants):
       try:
           # Your logic here
           pass
       except Exception as e:
           print(f"Callback error: {e}")
   
   result.refresh_manager.on_epoch_change = safe_callback
   ```

---

## Runtime Errors

### Model Not Found

**Error:**
```
No model found for request
```

**Cause:** No Gonka models registered, or all models disabled.

**Solutions:**

1. **Check registration succeeded:**
   ```python
   result = register_gonka_models(mux, ...)
   print(f"Models registered: {result.models_registered}")
   if result.models_registered == 0:
       print("No models were registered!")
   ```

2. **Check model names:**
   ```python
   for p in result.participants:
       print(f"Model: gonka:{p.address}")
   ```

---

### Request Timeout

**Symptoms:**
- Requests to Gonka participants timing out

**Solutions:**

1. **Check participant endpoint is healthy:**
   ```bash
   curl -v https://participant-node.example.com/v1/health
   ```

2. **Set concurrency limits:**
   ```python
   result = register_gonka_models(
       mux,
       ...,
       default_max_concurrent=3,  # Limit concurrent requests
   )
   ```

3. **Add fallback providers:**
   ```python
   mux.add_model(OpenAI(api_key="sk-..."), weight=100)
   register_gonka_models(mux, ..., register_as_fallback=True)
   ```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging

# Enable all Gonka debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("multiplexer_llm.gonka").setLevel(logging.DEBUG)

# Or just specific modules
logging.getLogger("multiplexer_llm.gonka.discovery").setLevel(logging.DEBUG)
logging.getLogger("multiplexer_llm.gonka.client_factory").setLevel(logging.DEBUG)
```

### Check Component State

```python
# After registration
result = register_gonka_models(mux, ...)

# Check what was registered
print(f"Models: {result.models_registered}")
print(f"Epoch: {result.epoch_id}")
print(f"Participants: {len(result.participants)}")

for p in result.participants:
    print(f"  {p.address}: {p.inference_url} (weight={p.weight})")

# Check refresh manager
if result.refresh_manager:
    print(f"Refresh running: {result.refresh_manager.is_running}")
    print(f"Current epoch: {result.refresh_manager.current_epoch}")
```

### Test Discovery Separately

```python
from multiplexer_llm.gonka import EndpointDiscovery

discovery = EndpointDiscovery(
    source_url="https://api.gonka.network",
    verify_proofs=False,
)

try:
    participants = discovery.discover()
    print(f"Found {len(participants)} participants")
    for p in participants:
        print(f"  {p.address}: {p.inference_url}")
except Exception as e:
    print(f"Discovery failed: {e}")
```

### Test Client Factory Separately

```python
from multiplexer_llm.gonka import GonkaClientFactory, GonkaParticipant

factory = GonkaClientFactory(private_key="0x...")
print(f"Derived address: {factory.requester_address}")

participant = GonkaParticipant(
    address="gonka1abc...",
    inference_url="https://node.example.com",
    weight=100,
)

try:
    client = factory.create_client(participant)
    print("Client created successfully")
except Exception as e:
    print(f"Client creation failed: {e}")
```

---

## FAQ

### Q: Can I use Gonka without a private key?

**A:** No. A private key is required to sign requests. Each request must be authenticated with the Gonka network.

---

### Q: How do I get a Gonka private key?

**A:** Generate an ECDSA secp256k1 private key:

```python
from ecdsa import SigningKey, SECP256k1

private_key = SigningKey.generate(curve=SECP256k1)
print(f"0x{private_key.to_string().hex()}")
```

Or use existing key generation tools for Cosmos-based networks.

---

### Q: Can I use multiple Gonka networks simultaneously?

**A:** Yes. Call `register_gonka_models()` multiple times with different configurations:

```python
# Register network 1
result1 = register_gonka_models(
    mux,
    source_url="https://api.gonka-network-1.com",
    private_key="0x...",
    model_name_prefix="gonka1:",
)

# Register network 2
result2 = register_gonka_models(
    mux,
    source_url="https://api.gonka-network-2.com",
    private_key="0x...",
    model_name_prefix="gonka2:",
)
```

---

### Q: How do I handle epoch transitions gracefully?

**A:** Use the epoch change callback:

```python
def on_epoch_change(old_epoch, new_epoch, participants):
    print(f"Transitioning from epoch {old_epoch} to {new_epoch}")
    # Update metrics, notify monitoring, etc.

result = register_gonka_models(mux, ...)
if result.refresh_manager:
    result.refresh_manager.on_epoch_change = on_epoch_change
```

---

### Q: Why are my requests slow?

**A:** Possible causes:
1. Participant endpoints are geographically distant
2. Too many concurrent requests overwhelming participants
3. Network congestion

Solutions:
- Set `default_max_concurrent` to limit load per participant
- Use participants closer to your region
- Add traditional providers as primary with Gonka as fallback

---

### Q: Can I disable specific participants?

**A:** Not directly, but you can use manual registration:

```python
from multiplexer_llm.gonka import (
    EndpointDiscovery,
    GonkaClientFactory,
    ModelRegistrar,
)

discovery = EndpointDiscovery(source_url="https://api.gonka.network")
factory = GonkaClientFactory(private_key="0x...")
registrar = ModelRegistrar(factory)

participants = discovery.discover()
for p in participants:
    if p.address not in ["gonka1skip...", "gonka1block..."]:
        registrar.register_one(mux, p)
```

---

### Q: How do I monitor Gonka health?

**A:** Use logging and callbacks:

```python
import logging

logging.getLogger("multiplexer_llm.gonka").setLevel(logging.INFO)

result = register_gonka_models(mux, ...)

if result.refresh_manager:
    def monitor_epochs(old_epoch, new_epoch, participants):
        # Send to your monitoring system
        print(f"Epoch: {old_epoch} -> {new_epoch}, Participants: {len(participants)}")
    
    result.refresh_manager.on_epoch_change = monitor_epochs

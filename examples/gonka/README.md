# Gonka Integration Examples

Working code examples for using the Gonka integration with multiplexer-llm.

## Prerequisites

1. Install multiplexer-llm with Gonka support:
   ```bash
   pip install multiplexer-llm[gonka]
   ```

2. Set your Gonka private key:
   ```bash
   export GONKA_PRIVATE_KEY="0x..."
   ```

## Examples

| File | Description |
|------|-------------|
| [basic_usage.py](./basic_usage.py) | Simple registration and chat completion |
| [auto_discovery.py](./auto_discovery.py) | Automatic participant discovery from network |
| [manual_registration.py](./manual_registration.py) | Manual endpoint configuration |
| [mixed_providers.py](./mixed_providers.py) | Combining Gonka with OpenAI |
| [epoch_handling.py](./epoch_handling.py) | Handling epoch transitions |

## Running Examples

Each example can be run directly:

```bash
# Set environment variables
export GONKA_PRIVATE_KEY="0x..."
export GONKA_SOURCE_URL="https://api.gonka.network"

# Run examples
python examples/gonka/basic_usage.py
python examples/gonka/auto_discovery.py
python examples/gonka/manual_registration.py
python examples/gonka/mixed_providers.py
python examples/gonka/epoch_handling.py
```

## Example Outputs

### basic_usage.py
```
Registered 5 Gonka models from epoch 42
Participants:
  - gonka1abc...: https://node1.example.com (weight=100)
  - gonka1def...: https://node2.example.com (weight=80)
  ...
Response: Paris is the capital of France...
```

### auto_discovery.py
```
Discovering participants from https://api.gonka.network...
Found 5 participants for epoch 42
Registered all participants with multiplexer
```

### mixed_providers.py
```
OpenAI registered as primary (weight=100)
Gonka models registered as fallback (3 models)
Request routed to: openai
```

### epoch_handling.py
```
Initial epoch: 42 with 5 participants
Waiting for epoch changes...
Epoch changed: 42 -> 43
  - Added: 2 participants
  - Removed: 1 participant
```

## Notes

- Examples use placeholder private keys - replace with your actual key
- Source URLs may need to be updated for your network
- Some examples require network access to the Gonka API
- For testing without network, use the `endpoints` parameter with known URLs

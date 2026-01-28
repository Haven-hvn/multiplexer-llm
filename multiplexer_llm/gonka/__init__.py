"""
Gonka integration module for multiplexer-llm.

This module provides components for integrating Gonka's decentralized LLM inference
network with the multiplexer. It includes:

- register_gonka_models: Primary convenience function for easy integration
- GonkaConfig: Configuration dataclass for Gonka settings
- GonkaRegistrationResult: Result of registering Gonka models
- EndpointDiscovery: Discovers Gonka network participants from blockchain state
- GonkaClientFactory: Creates OpenAI-compatible clients with ECDSA request signing
- GonkaParticipant: Data class representing a Gonka network participant
- ModelRegistrar: Registers Gonka participants as multiplexer models
- RefreshManager: Background service for epoch transitions
- RefreshResult: Result of a refresh operation
- Gonka-specific exceptions for error handling

Basic Usage:
    from multiplexer_llm import Multiplexer
    from multiplexer_llm.gonka import register_gonka_models

    mux = Multiplexer()
    result = register_gonka_models(
        mux,
        source_url="https://api.gonka.network",
        private_key="0x...",
    )
    print(f"Registered {result.models_registered} Gonka models")

Advanced Usage:
    from multiplexer_llm.gonka import (
        EndpointDiscovery,
        GonkaClientFactory,
        GonkaParticipant,
        GonkaConfig,
        ModelRegistrar,
        RefreshManager,
    )

    # Create configuration
    config = GonkaConfig(
        private_key="0x...",
        source_url="https://api.gonka.network",
        refresh_enabled=True,
    )

    # Discover participants from network
    discovery = EndpointDiscovery(source_url=config.source_url)
    participants = discovery.discover()

    # Create client factory and registrar
    factory = GonkaClientFactory(private_key=config.private_key)
    registrar = ModelRegistrar(factory)

    # Register all participants with multiplexer
    count = registrar.register_all(multiplexer, participants)
    print(f"Registered {count} Gonka models")

    # Optionally set up automatic epoch refresh
    manager = RefreshManager(
        multiplexer=multiplexer,
        discovery=discovery,
        registrar=registrar,
    )
    manager.start(interval_seconds=60.0)
"""

# Types
from .types import GonkaParticipant, RefreshResult, GonkaRegistrationResult

# Exceptions
from .exceptions import (
    GonkaError,
    GonkaClientError,
    GonkaConfigError,
    GonkaDiscoveryError,
    GonkaProofVerificationError,
    GonkaNoParticipantsError,
    GonkaRefreshError,
)

# Configuration and convenience function
from .config import GonkaConfig, register_gonka_models

# Core components
from .client_factory import GonkaClientFactory
from .discovery import EndpointDiscovery
from .registrar import ModelRegistrar
from .refresh import RefreshManager

__all__ = [
    # Primary API
    "register_gonka_models",
    "GonkaConfig",
    "GonkaRegistrationResult",
    # Types
    "GonkaParticipant",
    "RefreshResult",
    # Exceptions
    "GonkaError",
    "GonkaClientError",
    "GonkaConfigError",
    "GonkaDiscoveryError",
    "GonkaProofVerificationError",
    "GonkaNoParticipantsError",
    "GonkaRefreshError",
    # Advanced components
    "GonkaClientFactory",
    "EndpointDiscovery",
    "ModelRegistrar",
    "RefreshManager",
]

"""
Configuration module for Gonka integration.

This module provides the GonkaConfig dataclass for configuration management
and the register_gonka_models convenience function for easy integration.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional
from urllib.parse import urlparse

from .exceptions import GonkaConfigError
from .types import GonkaParticipant, GonkaRegistrationResult
from .client_factory import GonkaClientFactory
from .discovery import EndpointDiscovery
from .registrar import ModelRegistrar
from .refresh import RefreshManager

logger = logging.getLogger(__name__)

# Environment variable names
ENV_PRIVATE_KEY = "GONKA_PRIVATE_KEY"
ENV_SOURCE_URL = "GONKA_SOURCE_URL"
ENV_ENDPOINTS = "GONKA_ENDPOINTS"
ENV_ADDRESS = "GONKA_ADDRESS"
ENV_VERIFY_PROOF = "GONKA_VERIFY_PROOF"


def _is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid HTTP or HTTPS URL.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid, False otherwise.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
        # Must have scheme
        if not parsed.scheme:
            return False
        # Must be http, https, or file (for testing)
        if parsed.scheme not in ("http", "https", "file"):
            return False
        # http/https must have netloc, file URLs have path instead
        if parsed.scheme in ("http", "https") and not parsed.netloc:
            return False
        if parsed.scheme == "file" and not parsed.path:
            return False
        return True
    except Exception:
        return False


def _parse_endpoint(endpoint: str) -> tuple:
    """
    Parse an endpoint string in the format "url;address".

    Args:
        endpoint: The endpoint string to parse.

    Returns:
        A tuple of (url, address).

    Raises:
        GonkaConfigError: If the endpoint format is invalid.
    """
    if ";" not in endpoint:
        raise GonkaConfigError(
            f"Invalid endpoint format: {endpoint}. "
            f"Expected format: 'url;address' (e.g., 'https://api.example.com;gonka1abc...')",
            field="endpoints",
        )
    parts = endpoint.split(";", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise GonkaConfigError(
            f"Invalid endpoint format: {endpoint}. "
            f"Expected format: 'url;address' (e.g., 'https://api.example.com;gonka1abc...')",
            field="endpoints",
        )
    url, address = parts[0].strip(), parts[1].strip()
    if not _is_valid_url(url):
        raise GonkaConfigError(
            f"Invalid URL in endpoint: {url}. Must be a valid HTTP or HTTPS URL.",
            field="endpoints",
        )
    return url, address


def _resolve_env_str(
    explicit: Optional[str],
    env_var: str,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a string value from explicit argument or environment variable.

    Args:
        explicit: Explicitly provided value.
        env_var: Environment variable name.
        default: Default value if neither is set.

    Returns:
        The resolved value.
    """
    if explicit is not None:
        return explicit
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value.strip()
    return default


def _resolve_env_bool(
    explicit: Optional[bool],
    env_var: str,
    default: bool = False,
) -> bool:
    """
    Resolve a boolean value from explicit argument or environment variable.

    Environment variable is considered True if set to "1", "true", or "yes" (case-insensitive).

    Args:
        explicit: Explicitly provided value.
        env_var: Environment variable name.
        default: Default value if neither is set.

    Returns:
        The resolved boolean value.
    """
    if explicit is not None:
        return explicit
    env_value = os.environ.get(env_var)
    if env_value is not None:
        return env_value.strip().lower() in ("1", "true", "yes")
    return default


def _resolve_env_list(
    explicit: Optional[List[str]],
    env_var: str,
    default: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Resolve a list value from explicit argument or environment variable.

    Environment variable should be comma-separated values.

    Args:
        explicit: Explicitly provided value.
        env_var: Environment variable name.
        default: Default value if neither is set.

    Returns:
        The resolved list value.
    """
    if explicit is not None:
        return explicit
    env_value = os.environ.get(env_var)
    if env_value is not None:
        # Split by comma and strip whitespace
        return [v.strip() for v in env_value.split(",") if v.strip()]
    return default


@dataclass
class GonkaConfig:
    """
    Configuration for Gonka integration.

    This dataclass holds all configuration options for the Gonka integration.
    It validates configuration at construction time and resolves environment
    variables as needed.

    Attributes:
        private_key: ECDSA private key for request signing. Required.
        source_url: URL for participant discovery. One of source_url or endpoints required.
        endpoints: Explicit endpoint list (format: "url;address"). Alternative to source_url.
        address: Override derived Gonka address. If not provided, derived from private_key.
        verify_proofs: Enable ICS23 proof verification. Defaults to False.
        refresh_enabled: Enable background epoch refresh. Defaults to True.
        refresh_interval_seconds: How often to check for epoch changes. Defaults to 60.0.
        default_max_concurrent: Default concurrency limit for Gonka models. None means unlimited.
        register_as_fallback: Register Gonka models as fallbacks instead of primary.
        model_name_prefix: Prefix for generated model names (e.g., "gonka:gonka1abc...").

    Example:
        >>> config = GonkaConfig(
        ...     private_key="0x...",
        ...     source_url="https://api.gonka.network",
        ... )
        >>> # Or with explicit endpoints
        >>> config = GonkaConfig(
        ...     private_key="0x...",
        ...     endpoints=["https://node1.example.com;gonka1abc..."],
        ... )
    """

    # Required (but can come from env)
    private_key: str = ""
    """ECDSA private key for request signing. Can also be set via GONKA_PRIVATE_KEY env var."""

    # Endpoint source (one required)
    source_url: Optional[str] = None
    """URL for participant discovery. Can also be set via GONKA_SOURCE_URL env var."""

    endpoints: Optional[List[str]] = None
    """Explicit endpoint list (format: 'url;address'). Can also be set via GONKA_ENDPOINTS env var."""

    # Optional overrides
    address: Optional[str] = None
    """Override derived Gonka address. Can also be set via GONKA_ADDRESS env var."""

    # Feature flags
    verify_proofs: bool = False
    """Enable ICS23 proof verification. Can also be set via GONKA_VERIFY_PROOF env var."""

    refresh_enabled: bool = True
    """Enable background epoch refresh."""

    # Tuning
    refresh_interval_seconds: float = 60.0
    """How often to check for epoch changes in seconds."""

    default_max_concurrent: Optional[int] = None
    """Default concurrency limit for Gonka models. None means unlimited."""

    # Registration options
    register_as_fallback: bool = False
    """Register Gonka models as fallbacks instead of primary."""

    model_name_prefix: str = "gonka:"
    """Prefix for generated model names."""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._resolve_from_env()
        self._validate()

    def _resolve_from_env(self) -> None:
        """Resolve configuration values from environment variables."""
        # Resolve private_key from env if not provided
        if not self.private_key:
            env_key = os.environ.get(ENV_PRIVATE_KEY)
            if env_key:
                # Use object.__setattr__ since we're in post_init
                object.__setattr__(self, "private_key", env_key.strip())

        # Resolve source_url from env if not provided
        if not self.source_url:
            env_url = os.environ.get(ENV_SOURCE_URL)
            if env_url:
                object.__setattr__(self, "source_url", env_url.strip())

        # Resolve endpoints from env if not provided
        if not self.endpoints:
            env_endpoints = os.environ.get(ENV_ENDPOINTS)
            if env_endpoints:
                endpoints = [e.strip() for e in env_endpoints.split(",") if e.strip()]
                object.__setattr__(self, "endpoints", endpoints)

        # Resolve address from env if not provided
        if not self.address:
            env_address = os.environ.get(ENV_ADDRESS)
            if env_address:
                object.__setattr__(self, "address", env_address.strip())

        # Resolve verify_proofs from env
        env_verify = os.environ.get(ENV_VERIFY_PROOF)
        if env_verify is not None and not self.verify_proofs:
            object.__setattr__(
                self, "verify_proofs", env_verify.strip().lower() in ("1", "true", "yes")
            )

    def _validate(self) -> None:
        """
        Validate all configuration fields.

        Raises:
            GonkaConfigError: If any validation fails.
        """
        # Validate private_key
        if not self.private_key:
            raise GonkaConfigError(
                "Private key is required. "
                f"Provide via private_key argument or {ENV_PRIVATE_KEY} environment variable.",
                field="private_key",
            )

        # Validate private_key format (basic check without exposing key)
        key = self.private_key
        if key.startswith("0x"):
            key = key[2:]
        if len(key) != 64:
            raise GonkaConfigError(
                "Invalid private key format: expected 64 hex characters (32 bytes).",
                field="private_key",
            )
        try:
            bytes.fromhex(key)
        except ValueError:
            raise GonkaConfigError(
                "Invalid private key format: not valid hexadecimal.",
                field="private_key",
            )

        # Validate endpoint source - check for empty list explicitly
        has_source_url = bool(self.source_url)
        has_endpoints = self.endpoints is not None and len(self.endpoints) > 0
        
        if self.endpoints is not None and len(self.endpoints) == 0:
            raise GonkaConfigError(
                "Endpoints list cannot be empty.",
                field="endpoints",
            )
        
        if not has_source_url and not has_endpoints:
            raise GonkaConfigError(
                "Endpoint source is required. "
                f"Provide source_url, endpoints argument, or set {ENV_SOURCE_URL} / {ENV_ENDPOINTS}.",
                field="source_url",
            )

        # Validate source_url format if provided
        if self.source_url:
            if not _is_valid_url(self.source_url):
                raise GonkaConfigError(
                    f"Invalid URL: {self.source_url}. Must be a valid HTTP or HTTPS URL.",
                    field="source_url",
                )

        # Validate endpoints format if provided
        if self.endpoints:
            for endpoint in self.endpoints:
                _parse_endpoint(endpoint)

        # Validate refresh_interval_seconds
        if self.refresh_interval_seconds <= 0:
            raise GonkaConfigError(
                f"refresh_interval_seconds must be positive, got {self.refresh_interval_seconds}",
                field="refresh_interval_seconds",
            )

        # Validate default_max_concurrent if provided
        if self.default_max_concurrent is not None and self.default_max_concurrent < 0:
            raise GonkaConfigError(
                f"default_max_concurrent must be non-negative, got {self.default_max_concurrent}",
                field="default_max_concurrent",
            )

        # Validate model_name_prefix
        if not self.model_name_prefix:
            raise GonkaConfigError(
                "model_name_prefix cannot be empty.",
                field="model_name_prefix",
            )

    def __repr__(self) -> str:
        """Return a string representation without exposing the private key."""
        return (
            f"GonkaConfig("
            f"private_key='***', "
            f"source_url={self.source_url!r}, "
            f"endpoints={self.endpoints!r}, "
            f"address={self.address!r}, "
            f"verify_proofs={self.verify_proofs}, "
            f"refresh_enabled={self.refresh_enabled}, "
            f"refresh_interval_seconds={self.refresh_interval_seconds}, "
            f"default_max_concurrent={self.default_max_concurrent}, "
            f"register_as_fallback={self.register_as_fallback}, "
            f"model_name_prefix={self.model_name_prefix!r})"
        )


def register_gonka_models(
    multiplexer: Any,
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
) -> GonkaRegistrationResult:
    """
    Register Gonka network participants with the multiplexer.

    This is the primary convenience function for Gonka integration. It handles
    all the setup including:
    1. Building and validating configuration
    2. Creating the client factory
    3. Discovering participants
    4. Registering all participants as models
    5. Optionally starting background refresh

    Individual arguments take precedence over config object fields, which take
    precedence over environment variables.

    Args:
        multiplexer: The Multiplexer instance to register models with.
        source_url: URL for participant discovery.
        endpoints: Explicit endpoint list (format: "url;address").
        private_key: ECDSA private key for request signing.
        config: Optional GonkaConfig object. Individual args override config fields.
        verify_proofs: Enable ICS23 proof verification.
        refresh_enabled: Enable background epoch refresh.
        refresh_interval_seconds: How often to check for epoch changes.
        default_max_concurrent: Default concurrency limit for Gonka models.
        register_as_fallback: Register Gonka models as fallbacks instead of primary.
        model_name_prefix: Prefix for generated model names.
        address: Override derived Gonka address.

    Returns:
        GonkaRegistrationResult with information about the registration.

    Raises:
        GonkaConfigError: If configuration is invalid.
        GonkaDiscoveryError: If participant discovery fails.
        GonkaNoParticipantsError: If no participants are found.

    Example:
        >>> from multiplexer_llm import Multiplexer
        >>> from multiplexer_llm.gonka import register_gonka_models
        >>>
        >>> mux = Multiplexer()
        >>> result = register_gonka_models(
        ...     mux,
        ...     source_url="https://api.gonka.network",
        ...     private_key="0x...",
        ... )
        >>> print(f"Registered {result.models_registered} models")

        >>> # Or using a config object
        >>> config = GonkaConfig(
        ...     private_key="0x...",
        ...     source_url="https://api.gonka.network",
        ... )
        >>> result = register_gonka_models(mux, config=config)
    """
    # Build effective configuration by merging arguments, config, and env vars
    effective_config = _build_effective_config(
        config=config,
        source_url=source_url,
        endpoints=endpoints,
        private_key=private_key,
        verify_proofs=verify_proofs,
        refresh_enabled=refresh_enabled,
        refresh_interval_seconds=refresh_interval_seconds,
        default_max_concurrent=default_max_concurrent,
        register_as_fallback=register_as_fallback,
        model_name_prefix=model_name_prefix,
        address=address,
    )

    logger.info(
        "Registering Gonka models (source_url=%s, refresh_enabled=%s)",
        effective_config.source_url,
        effective_config.refresh_enabled,
    )

    # Create client factory
    client_factory = GonkaClientFactory(private_key=effective_config.private_key)

    # Create discovery service
    if effective_config.source_url:
        discovery = EndpointDiscovery(
            source_url=effective_config.source_url,
            verify_proofs=effective_config.verify_proofs,
        )
        # Discover participants
        participants = discovery.discover()
    else:
        # Use explicit endpoints
        participants = _create_participants_from_endpoints(effective_config.endpoints or [])
        # Create a dummy discovery for the refresh manager (won't be used if refresh disabled)
        discovery = None

    if not participants:
        from .exceptions import GonkaNoParticipantsError
        raise GonkaNoParticipantsError(
            "No participants found",
            epoch="current",
            source_url=effective_config.source_url,
        )

    # Get epoch_id from first participant
    epoch_id = participants[0].epoch_id if participants else 0

    # Create registrar
    registrar = ModelRegistrar(
        client_factory=client_factory,
        model_name_prefix=effective_config.model_name_prefix,
        default_max_concurrent=effective_config.default_max_concurrent,
    )

    # Register all participants
    models_registered = registrar.register_all(
        multiplexer=multiplexer,
        participants=participants,
        as_fallback=effective_config.register_as_fallback,
    )

    # Create refresh manager if enabled and we have a discovery source
    refresh_manager: Optional[RefreshManager] = None
    if effective_config.refresh_enabled and discovery is not None:
        refresh_manager = RefreshManager(
            multiplexer=multiplexer,
            discovery=discovery,
            registrar=registrar,
            as_fallback=effective_config.register_as_fallback,
        )
        # Set the current epoch so refresh knows what we started with
        refresh_manager._current_epoch = epoch_id
        # Start background refresh
        try:
            refresh_manager.start(
                interval_seconds=effective_config.refresh_interval_seconds
            )
        except RuntimeError:
            # Not in an async context - that's okay, user can start it later
            logger.debug(
                "Could not start background refresh (not in async context). "
                "Call refresh_manager.start() from an async context to enable."
            )

    logger.info(
        "Registered %d Gonka models from epoch %d (refresh=%s)",
        models_registered,
        epoch_id,
        "enabled" if refresh_manager else "disabled",
    )

    return GonkaRegistrationResult(
        models_registered=models_registered,
        participants=participants,
        refresh_manager=refresh_manager,
        epoch_id=epoch_id,
    )


def _build_effective_config(
    config: Optional[GonkaConfig],
    source_url: Optional[str],
    endpoints: Optional[List[str]],
    private_key: Optional[str],
    verify_proofs: Optional[bool],
    refresh_enabled: Optional[bool],
    refresh_interval_seconds: Optional[float],
    default_max_concurrent: Optional[int],
    register_as_fallback: bool,
    model_name_prefix: str,
    address: Optional[str],
) -> GonkaConfig:
    """
    Build effective configuration by merging arguments, config, and env vars.

    Priority order (first non-None wins):
    1. Explicit function argument
    2. Config object field (if config provided)
    3. Environment variable
    4. Default value

    Args:
        All arguments from register_gonka_models.

    Returns:
        A validated GonkaConfig instance.

    Raises:
        GonkaConfigError: If configuration is invalid.
    """
    # If no config provided, build from arguments and env vars
    if config is None:
        return GonkaConfig(
            private_key=_resolve_env_str(private_key, ENV_PRIVATE_KEY) or "",
            source_url=_resolve_env_str(source_url, ENV_SOURCE_URL),
            endpoints=_resolve_env_list(endpoints, ENV_ENDPOINTS),
            address=_resolve_env_str(address, ENV_ADDRESS),
            verify_proofs=_resolve_env_bool(verify_proofs, ENV_VERIFY_PROOF, False),
            refresh_enabled=refresh_enabled if refresh_enabled is not None else True,
            refresh_interval_seconds=refresh_interval_seconds if refresh_interval_seconds is not None else 60.0,
            default_max_concurrent=default_max_concurrent,
            register_as_fallback=register_as_fallback,
            model_name_prefix=model_name_prefix,
        )

    # Config provided - merge with explicit arguments (explicit args override config)
    return GonkaConfig(
        private_key=(
            private_key
            if private_key is not None
            else config.private_key
        ),
        source_url=(
            source_url
            if source_url is not None
            else config.source_url
        ),
        endpoints=(
            endpoints
            if endpoints is not None
            else config.endpoints
        ),
        address=(
            address
            if address is not None
            else config.address
        ),
        verify_proofs=(
            verify_proofs
            if verify_proofs is not None
            else config.verify_proofs
        ),
        refresh_enabled=(
            refresh_enabled
            if refresh_enabled is not None
            else config.refresh_enabled
        ),
        refresh_interval_seconds=(
            refresh_interval_seconds
            if refresh_interval_seconds is not None
            else config.refresh_interval_seconds
        ),
        default_max_concurrent=(
            default_max_concurrent
            if default_max_concurrent is not None
            else config.default_max_concurrent
        ),
        register_as_fallback=register_as_fallback or config.register_as_fallback,
        model_name_prefix=model_name_prefix if model_name_prefix != "gonka:" else config.model_name_prefix,
    )


def _create_participants_from_endpoints(endpoints: List[str]) -> List[GonkaParticipant]:
    """
    Create GonkaParticipant objects from explicit endpoint strings.

    Args:
        endpoints: List of endpoint strings in format "url;address".

    Returns:
        List of GonkaParticipant objects.
    """
    participants = []
    for endpoint in endpoints:
        url, address = _parse_endpoint(endpoint)
        participant = GonkaParticipant(
            address=address,
            inference_url=url,
            weight=1,  # Default weight for explicit endpoints
            models=[],
            validator_key="",
            epoch_id=0,  # Unknown epoch for explicit endpoints
        )
        participants.append(participant)
    return participants

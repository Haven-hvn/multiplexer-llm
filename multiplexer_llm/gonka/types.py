"""
Type definitions for the Gonka integration module.

This module defines the data structures used throughout the Gonka integration,
including participant information and configuration types.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from .refresh import RefreshManager


@dataclass(frozen=True)
class GonkaParticipant:
    """
    Represents a single Gonka network participant discovered from blockchain state.

    A participant is an inference provider registered on the Gonka network that can
    process LLM requests. Each participant has a unique address, an inference endpoint,
    and stake-based weight for selection.

    Attributes:
        address: Gonka address (e.g., "gonka1abc..."). Must be a valid bech32 address.
        inference_url: API endpoint URL for this participant's inference service.
        weight: Stake-based weight for selection. Higher weights get more requests.
        models: List of supported model identifiers. Empty means all models supported.
        validator_key: Validator public key for verification.
        epoch_id: Epoch when this participant was active.

    Invariants:
        - address must be a valid bech32 Gonka address
        - inference_url must be a valid HTTPS URL  
        - weight must be positive
        - models may be empty (participant supports all models)

    Example:
        >>> participant = GonkaParticipant(
        ...     address="gonka1abc...",
        ...     inference_url="https://node.example.com/v1",
        ...     weight=100,
        ...     models=["llama-3.1-70b"],
        ...     validator_key="...",
        ...     epoch_id=42
        ... )
    """

    address: str
    """Gonka address (e.g., 'gonka1abc...'). Used for transfer_address header."""

    inference_url: str
    """API endpoint URL for this participant's inference service."""

    weight: int = 1
    """Stake-based weight for selection. Higher weights get more requests."""

    models: List[str] = field(default_factory=list)
    """List of supported model identifiers. Empty means all models supported."""

    validator_key: str = ""
    """Validator public key for verification."""

    epoch_id: int = 0
    """Epoch when this participant was active."""

    def __post_init__(self) -> None:
        """Validate participant data after initialization."""
        if not self.address:
            raise ValueError("Participant address cannot be empty")
        if not self.inference_url:
            raise ValueError("Participant inference_url cannot be empty")
        if self.weight <= 0:
            raise ValueError(f"Participant weight must be positive, got {self.weight}")

    @property
    def base_url(self) -> str:
        """
        Get the base URL for OpenAI client configuration.

        Ensures the URL has the /v1 suffix required by OpenAI SDK.

        Returns:
            The inference URL with /v1 suffix if not already present.
        """
        url = self.inference_url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return url

    def __repr__(self) -> str:
        """Return a string representation without exposing sensitive data."""
        return (
            f"GonkaParticipant(address={self.address!r}, "
            f"inference_url={self.inference_url!r}, "
            f"weight={self.weight}, "
            f"models={self.models!r}, "
            f"epoch_id={self.epoch_id})"
        )


@dataclass
class RefreshResult:
    """
    Result of a refresh operation from the RefreshManager.

    This data class captures the outcome of a refresh attempt, including
    whether an epoch change occurred and details about what changed.

    Attributes:
        success: Whether the refresh completed without errors.
        epoch_changed: Whether the epoch changed during this refresh.
        old_epoch: The previous epoch ID if epoch changed, None otherwise.
        new_epoch: The new epoch ID if epoch changed, None otherwise.
        participants_added: Number of participants added during refresh.
        participants_removed: Number of participants removed during refresh.
        error: The exception if refresh failed, None otherwise.

    Example:
        >>> result = RefreshResult(
        ...     success=True,
        ...     epoch_changed=True,
        ...     old_epoch=41,
        ...     new_epoch=42,
        ...     participants_added=5,
        ...     participants_removed=3,
        ... )
        >>> if result.epoch_changed:
        ...     print(f"Epoch changed from {result.old_epoch} to {result.new_epoch}")
    """

    success: bool
    """Whether the refresh completed without errors."""

    epoch_changed: bool = False
    """Whether the epoch changed during this refresh."""

    old_epoch: Optional[int] = None
    """The previous epoch ID if epoch changed, None otherwise."""

    new_epoch: Optional[int] = None
    """The new epoch ID if epoch changed, None otherwise."""

    participants_added: int = 0
    """Number of participants added during refresh."""

    participants_removed: int = 0
    """Number of participants removed during refresh."""

    error: Optional[Exception] = None
    """The exception if refresh failed, None otherwise."""

    def __repr__(self) -> str:
        """Return a string representation of the refresh result."""
        if self.success:
            if self.epoch_changed:
                return (
                    f"RefreshResult(success=True, epoch_changed=True, "
                    f"old_epoch={self.old_epoch}, new_epoch={self.new_epoch}, "
                    f"added={self.participants_added}, removed={self.participants_removed})"
                )
            return "RefreshResult(success=True, epoch_changed=False)"
        return f"RefreshResult(success=False, error={type(self.error).__name__})"


@dataclass
class GonkaRegistrationResult:
    """
    Result of registering Gonka models with the multiplexer.

    This data class is returned by `register_gonka_models()` and contains
    information about the registration outcome including the number of models
    registered, the list of participants, and optionally the refresh manager
    if background refresh is enabled.

    Attributes:
        models_registered: Number of models successfully registered.
        participants: List of GonkaParticipant objects that were discovered.
        refresh_manager: The RefreshManager if refresh is enabled, None otherwise.
        epoch_id: The epoch ID for the registered participants.

    Example:
        >>> result = register_gonka_models(mux, source_url="...", private_key="...")
        >>> print(f"Registered {result.models_registered} models from epoch {result.epoch_id}")
        >>> if result.refresh_manager:
        ...     print("Background refresh is enabled")
    """

    models_registered: int
    """Number of models successfully registered with the multiplexer."""

    participants: List[GonkaParticipant] = field(default_factory=list)
    """List of GonkaParticipant objects that were discovered."""

    refresh_manager: Optional[Any] = None  # Actually RefreshManager, but avoiding circular import
    """The RefreshManager if refresh is enabled, None otherwise."""

    epoch_id: int = 0
    """The epoch ID for the registered participants."""

    def __repr__(self) -> str:
        """Return a string representation of the registration result."""
        return (
            f"GonkaRegistrationResult("
            f"models_registered={self.models_registered}, "
            f"participants={len(self.participants)}, "
            f"epoch_id={self.epoch_id}, "
            f"refresh_manager={'enabled' if self.refresh_manager else 'disabled'})"
        )

"""
ModelRegistrar - Registers Gonka participants as multiplexer models.

This module provides the ModelRegistrar class, which takes Gonka participants
from endpoint discovery, creates clients via GonkaClientFactory, and registers
them with the multiplexer.

The registrar tracks which models belong to which epoch, enabling clean
transitions during epoch changes.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Set

from .types import GonkaParticipant
from .exceptions import GonkaClientError
from .client_factory import GonkaClientFactory

logger = logging.getLogger(__name__)


class ModelRegistrar:
    """
    Registers Gonka participants as models in the multiplexer.

    This class is responsible for:
    - Creating OpenAI-compatible clients for each participant using GonkaClientFactory
    - Registering those clients with the multiplexer (as primary or fallback models)
    - Mapping participant weights to multiplexer weights
    - Generating unique model names from participant addresses
    - Tracking registered models by epoch for cleanup during transitions

    Thread Safety:
        This class is thread-safe. Multiple threads can safely call registration
        methods concurrently.

    Attributes:
        client_factory: The GonkaClientFactory used to create clients.
        model_name_prefix: Prefix for generated model names (default: "gonka:").
        default_max_concurrent: Default concurrency limit for models.

    Example:
        >>> factory = GonkaClientFactory(private_key="0x...")
        >>> registrar = ModelRegistrar(factory)
        >>> count = registrar.register_all(multiplexer, participants)
        >>> print(f"Registered {count} models")
    """

    def __init__(
        self,
        client_factory: GonkaClientFactory,
        model_name_prefix: str = "gonka:",
        default_max_concurrent: Optional[int] = None,
    ) -> None:
        """
        Initialize the model registrar.

        Args:
            client_factory: Factory for creating OpenAI-compatible clients.
            model_name_prefix: Prefix for generated model names. Allows identifying
                Gonka models for epoch cleanup. Default is "gonka:".
            default_max_concurrent: Default concurrency limit for registered models.
                None means unlimited concurrency.

        Raises:
            ValueError: If model_name_prefix is empty.
        """
        if not model_name_prefix:
            raise ValueError("model_name_prefix cannot be empty")

        self._client_factory = client_factory
        self._model_name_prefix = model_name_prefix
        self._default_max_concurrent = default_max_concurrent

        # Thread-safe tracking of registered models by epoch
        self._lock = threading.Lock()
        self._registered_models: Dict[int, Set[str]] = {}

        logger.debug(
            "ModelRegistrar initialized with prefix=%r, default_max_concurrent=%s",
            model_name_prefix,
            default_max_concurrent,
        )

    @property
    def client_factory(self) -> GonkaClientFactory:
        """Get the client factory used for client creation."""
        return self._client_factory

    @property
    def model_name_prefix(self) -> str:
        """Get the prefix used for model names."""
        return self._model_name_prefix

    @property
    def default_max_concurrent(self) -> Optional[int]:
        """Get the default max concurrent limit."""
        return self._default_max_concurrent

    def register_all(
        self,
        multiplexer: Any,
        participants: List[GonkaParticipant],
        as_fallback: bool = False,
    ) -> int:
        """
        Register all participants as models in the multiplexer.

        This method iterates through the provided participants and registers
        each one with the multiplexer. It gracefully handles failures for
        individual participants without affecting others.

        Args:
            multiplexer: The Multiplexer instance to register models with.
            participants: List of GonkaParticipant objects to register.
            as_fallback: If True, register as fallback models instead of primary.

        Returns:
            The number of successfully registered models.

        Example:
            >>> count = registrar.register_all(multiplexer, participants)
            >>> print(f"Registered {count}/{len(participants)} models")
        """
        if not participants:
            logger.info("No participants to register")
            return 0

        registered_count = 0
        for participant in participants:
            try:
                if self.register_one(multiplexer, participant, as_fallback=as_fallback):
                    registered_count += 1
            except Exception as e:
                # Log but continue with other participants
                logger.error(
                    "Failed to register participant %s: %s",
                    participant.address,
                    e,
                )

        logger.info(
            "Registered %d/%d Gonka participants as %s models",
            registered_count,
            len(participants),
            "fallback" if as_fallback else "primary",
        )
        return registered_count

    def register_one(
        self,
        multiplexer: Any,
        participant: GonkaParticipant,
        as_fallback: bool = False,
        weight_override: Optional[int] = None,
        max_concurrent: Optional[int] = None,
    ) -> bool:
        """
        Register a single participant as a model in the multiplexer.

        This method creates a client for the participant and registers it
        with the multiplexer. It handles duplicate detection gracefully.

        Args:
            multiplexer: The Multiplexer instance to register the model with.
            participant: The GonkaParticipant to register.
            as_fallback: If True, register as a fallback model.
            weight_override: Override the participant's weight. If None or <= 0,
                uses the participant's weight (or 1 as minimum).
            max_concurrent: Override the default max concurrent limit for this model.

        Returns:
            True if the model was registered successfully.
            False if skipped (e.g., duplicate model name).

        Raises:
            GonkaClientError: If client creation fails (only for critical errors).

        Example:
            >>> success = registrar.register_one(multiplexer, participant)
            >>> if success:
            ...     print(f"Registered {participant.address}")
        """
        model_name = self._generate_model_name(participant)

        with self._lock:
            # Check for duplicate across all epochs
            if self._is_already_registered(model_name):
                logger.info(
                    "Skipping duplicate registration for model %s",
                    model_name,
                )
                return False

        # Create client for participant
        try:
            client = self._client_factory.create_client(participant)
        except GonkaClientError as e:
            logger.error(
                "Failed to create client for participant %s: %s",
                participant.address,
                e,
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error creating client for participant %s: %s",
                participant.address,
                e,
            )
            return False

        # Determine weight: override > participant > default (1)
        weight = self._determine_weight(participant, weight_override)

        # Determine max_concurrent: parameter > default
        effective_max_concurrent = (
            max_concurrent if max_concurrent is not None else self._default_max_concurrent
        )

        # Get base URL for the model
        base_url = participant.base_url

        # Register with multiplexer
        try:
            if as_fallback:
                multiplexer.add_fallback_model(
                    model=client,
                    weight=weight,
                    model_name=model_name,
                    base_url=base_url,
                    max_concurrent=effective_max_concurrent,
                )
            else:
                multiplexer.add_model(
                    model=client,
                    weight=weight,
                    model_name=model_name,
                    base_url=base_url,
                    max_concurrent=effective_max_concurrent,
                )
        except ValueError as e:
            # Multiplexer validation errors (e.g., duplicate)
            logger.warning(
                "Multiplexer rejected model %s: %s",
                model_name,
                e,
            )
            return False

        # Track registration
        with self._lock:
            epoch_id = participant.epoch_id
            if epoch_id not in self._registered_models:
                self._registered_models[epoch_id] = set()
            self._registered_models[epoch_id].add(model_name)

        logger.debug(
            "Registered model %s (weight=%d, epoch=%d, fallback=%s)",
            model_name,
            weight,
            participant.epoch_id,
            as_fallback,
        )
        return True

    def unregister_all(self, multiplexer: Any, epoch_id: int) -> int:
        """
        Remove all models from a specific epoch.

        Since the multiplexer doesn't have a remove_model() method, this
        implementation disables models by setting disabled_until to infinity,
        effectively preventing them from being selected.

        Args:
            multiplexer: The Multiplexer instance containing the models.
            epoch_id: The epoch ID whose models should be removed.

        Returns:
            The number of unregistered (disabled) models.

        Note:
            This method accesses internal multiplexer state (_weighted_models,
            _fallback_models) as a workaround for the lack of a public API
            for model removal.

        Example:
            >>> count = registrar.unregister_all(multiplexer, old_epoch=41)
            >>> print(f"Disabled {count} models from epoch 41")
        """
        with self._lock:
            model_names = self._registered_models.pop(epoch_id, set())

        if not model_names:
            logger.debug("No models to unregister for epoch %d", epoch_id)
            return 0

        disabled_count = 0
        for model_name in model_names:
            if self._disable_model_in_multiplexer(multiplexer, model_name):
                disabled_count += 1

        logger.info(
            "Unregistered %d/%d models from epoch %d",
            disabled_count,
            len(model_names),
            epoch_id,
        )
        return disabled_count

    def get_registered_models(self, epoch_id: Optional[int] = None) -> Set[str]:
        """
        Get the set of registered model names.

        Args:
            epoch_id: If provided, return only models for that epoch.
                If None, return all registered models across all epochs.

        Returns:
            A set of model names.

        Example:
            >>> models = registrar.get_registered_models(epoch_id=42)
            >>> print(f"Epoch 42 has {len(models)} models")
        """
        with self._lock:
            if epoch_id is not None:
                return set(self._registered_models.get(epoch_id, set()))
            # Return all models across all epochs
            all_models: Set[str] = set()
            for epoch_models in self._registered_models.values():
                all_models.update(epoch_models)
            return all_models

    def _generate_model_name(self, participant: GonkaParticipant) -> str:
        """
        Generate a unique model name from a participant.

        Args:
            participant: The participant to generate a name for.

        Returns:
            A unique model name in the format "{prefix}{address}".

        Example:
            >>> name = registrar._generate_model_name(participant)
            >>> # Returns: "gonka:gonka1abc123..."
        """
        return f"{self._model_name_prefix}{participant.address}"

    def _is_already_registered(self, model_name: str) -> bool:
        """
        Check if a model name is already registered.

        Note: Must be called with self._lock held.

        Args:
            model_name: The model name to check.

        Returns:
            True if already registered, False otherwise.
        """
        for epoch_models in self._registered_models.values():
            if model_name in epoch_models:
                return True
        return False

    def _determine_weight(
        self,
        participant: GonkaParticipant,
        weight_override: Optional[int],
    ) -> int:
        """
        Determine the effective weight for a participant.

        Args:
            participant: The participant.
            weight_override: Optional override value.

        Returns:
            The effective weight (minimum 1).
        """
        # Use override if valid
        if weight_override is not None and weight_override > 0:
            return weight_override

        # Use participant weight if valid
        if participant.weight > 0:
            return participant.weight

        # Default to 1
        logger.warning(
            "Participant %s has invalid weight %d, using default 1",
            participant.address,
            participant.weight,
        )
        return 1

    def _disable_model_in_multiplexer(self, multiplexer: Any, model_name: str) -> bool:
        """
        Disable a model in the multiplexer.

        This is a workaround since multiplexer lacks a remove_model() method.
        We access internal state to set disabled_until to infinity.

        Args:
            multiplexer: The Multiplexer instance.
            model_name: The model name to disable.

        Returns:
            True if the model was found and disabled, False otherwise.
        """
        # Check primary models
        for wm in getattr(multiplexer, "_weighted_models", []):
            if wm.model_name == model_name:
                wm.disabled_until = float("inf")
                logger.debug("Disabled primary model %s", model_name)
                return True

        # Check fallback models
        for wm in getattr(multiplexer, "_fallback_models", []):
            if wm.model_name == model_name:
                wm.disabled_until = float("inf")
                logger.debug("Disabled fallback model %s", model_name)
                return True

        logger.warning(
            "Model %s not found in multiplexer for disabling",
            model_name,
        )
        return False

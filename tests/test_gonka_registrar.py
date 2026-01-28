"""
Unit tests for the ModelRegistrar class.

Tests cover:
- Single participant registration
- Multiple participant registration
- Duplicate registration handling
- Client creation failure handling
- Weight handling (override, participant, default)
- Epoch tracking and unregistration
- Fallback model registration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from multiplexer_llm.gonka.types import GonkaParticipant
from multiplexer_llm.gonka.registrar import ModelRegistrar
from multiplexer_llm.gonka.exceptions import GonkaClientError


class TestModelRegistrarInit:
    """Test ModelRegistrar initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        mock_factory = Mock()
        registrar = ModelRegistrar(mock_factory)
        
        assert registrar.client_factory is mock_factory
        assert registrar.model_name_prefix == "gonka:"
        assert registrar.default_max_concurrent is None

    def test_init_with_custom_prefix(self):
        """Test initialization with custom model name prefix."""
        mock_factory = Mock()
        registrar = ModelRegistrar(mock_factory, model_name_prefix="custom:")
        
        assert registrar.model_name_prefix == "custom:"

    def test_init_with_max_concurrent(self):
        """Test initialization with default max concurrent."""
        mock_factory = Mock()
        registrar = ModelRegistrar(mock_factory, default_max_concurrent=10)
        
        assert registrar.default_max_concurrent == 10

    def test_init_empty_prefix_raises_error(self):
        """Test that empty prefix raises ValueError."""
        mock_factory = Mock()
        with pytest.raises(ValueError, match="model_name_prefix cannot be empty"):
            ModelRegistrar(mock_factory, model_name_prefix="")


class TestModelRegistrarRegisterOne:
    """Test ModelRegistrar.register_one method."""

    def create_participant(self, address="gonka1test123", weight=100, epoch_id=42):
        """Create a test participant."""
        return GonkaParticipant(
            address=address,
            inference_url="https://node.example.com",
            weight=weight,
            models=["llama-3.1-70b"],
            validator_key="validator123",
            epoch_id=epoch_id,
        )

    def test_register_single_participant_successfully(self):
        """Test successful registration of a single participant."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory)
        result = registrar.register_one(mock_multiplexer, participant)
        
        assert result is True
        mock_factory.create_client.assert_called_once_with(participant)
        mock_multiplexer.add_model.assert_called_once_with(
            model=mock_client,
            weight=100,
            model_name="gonka:gonka1test123",
            base_url="https://node.example.com/v1",
            max_concurrent=None,
        )

    def test_register_as_fallback(self):
        """Test registration as fallback model."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory)
        result = registrar.register_one(mock_multiplexer, participant, as_fallback=True)
        
        assert result is True
        mock_multiplexer.add_fallback_model.assert_called_once()
        mock_multiplexer.add_model.assert_not_called()

    def test_skip_duplicate_registration(self):
        """Test that duplicate registration is skipped."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory)
        
        # First registration should succeed
        result1 = registrar.register_one(mock_multiplexer, participant)
        assert result1 is True
        
        # Second registration should be skipped
        result2 = registrar.register_one(mock_multiplexer, participant)
        assert result2 is False
        
        # Client should only be created once
        assert mock_factory.create_client.call_count == 1

    def test_weight_override(self):
        """Test that weight override is respected."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant(weight=100)
        
        registrar = ModelRegistrar(mock_factory)
        registrar.register_one(mock_multiplexer, participant, weight_override=50)
        
        # Check that override weight is used
        call_args = mock_multiplexer.add_model.call_args
        assert call_args.kwargs["weight"] == 50

    def test_max_concurrent_override(self):
        """Test that max_concurrent override is respected."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory, default_max_concurrent=5)
        registrar.register_one(mock_multiplexer, participant, max_concurrent=10)
        
        # Check that override max_concurrent is used
        call_args = mock_multiplexer.add_model.call_args
        assert call_args.kwargs["max_concurrent"] == 10

    def test_default_max_concurrent_used(self):
        """Test that default max_concurrent is used when not overridden."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory, default_max_concurrent=5)
        registrar.register_one(mock_multiplexer, participant)
        
        call_args = mock_multiplexer.add_model.call_args
        assert call_args.kwargs["max_concurrent"] == 5

    def test_client_creation_failure(self):
        """Test handling of client creation failure."""
        mock_factory = Mock()
        mock_factory.create_client.side_effect = GonkaClientError("Client creation failed")
        
        mock_multiplexer = Mock()
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory)
        result = registrar.register_one(mock_multiplexer, participant)
        
        assert result is False
        mock_multiplexer.add_model.assert_not_called()

    def test_unexpected_client_creation_error(self):
        """Test handling of unexpected errors during client creation."""
        mock_factory = Mock()
        mock_factory.create_client.side_effect = RuntimeError("Unexpected error")
        
        mock_multiplexer = Mock()
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory)
        result = registrar.register_one(mock_multiplexer, participant)
        
        assert result is False

    def test_multiplexer_rejection(self):
        """Test handling when multiplexer rejects the model."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        mock_multiplexer.add_model.side_effect = ValueError("Duplicate model")
        
        participant = self.create_participant()
        
        registrar = ModelRegistrar(mock_factory)
        result = registrar.register_one(mock_multiplexer, participant)
        
        assert result is False

    def test_model_name_generation(self):
        """Test that model names are generated correctly."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant(address="gonka1xyz789")
        
        registrar = ModelRegistrar(mock_factory, model_name_prefix="test:")
        registrar.register_one(mock_multiplexer, participant)
        
        call_args = mock_multiplexer.add_model.call_args
        assert call_args.kwargs["model_name"] == "test:gonka1xyz789"

    def test_epoch_tracking(self):
        """Test that registered models are tracked by epoch."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participant = self.create_participant(epoch_id=42)
        
        registrar = ModelRegistrar(mock_factory)
        registrar.register_one(mock_multiplexer, participant)
        
        models = registrar.get_registered_models(epoch_id=42)
        assert "gonka:gonka1test123" in models


class TestModelRegistrarRegisterAll:
    """Test ModelRegistrar.register_all method."""

    def create_participants(self, count=3):
        """Create multiple test participants."""
        return [
            GonkaParticipant(
                address=f"gonka1test{i}",
                inference_url=f"https://node{i}.example.com",
                weight=100 + i,
                epoch_id=42,
            )
            for i in range(count)
        ]

    def test_register_multiple_participants(self):
        """Test registering multiple participants."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participants = self.create_participants(3)
        
        registrar = ModelRegistrar(mock_factory)
        count = registrar.register_all(mock_multiplexer, participants)
        
        assert count == 3
        assert mock_factory.create_client.call_count == 3
        assert mock_multiplexer.add_model.call_count == 3

    def test_register_empty_list(self):
        """Test registering an empty list of participants."""
        mock_factory = Mock()
        mock_multiplexer = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        count = registrar.register_all(mock_multiplexer, [])
        
        assert count == 0
        mock_factory.create_client.assert_not_called()

    def test_partial_failure_continues(self):
        """Test that partial failures don't stop other registrations."""
        mock_factory = Mock()
        
        # Fail on second call
        mock_factory.create_client.side_effect = [
            Mock(),  # First succeeds
            GonkaClientError("Failed"),  # Second fails
            Mock(),  # Third succeeds
        ]
        
        mock_multiplexer = Mock()
        participants = self.create_participants(3)
        
        registrar = ModelRegistrar(mock_factory)
        count = registrar.register_all(mock_multiplexer, participants)
        
        assert count == 2  # Two succeeded
        assert mock_multiplexer.add_model.call_count == 2

    def test_register_as_fallback(self):
        """Test registering all as fallback models."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_multiplexer = Mock()
        participants = self.create_participants(2)
        
        registrar = ModelRegistrar(mock_factory)
        count = registrar.register_all(mock_multiplexer, participants, as_fallback=True)
        
        assert count == 2
        assert mock_multiplexer.add_fallback_model.call_count == 2
        mock_multiplexer.add_model.assert_not_called()


class TestModelRegistrarUnregisterAll:
    """Test ModelRegistrar.unregister_all method."""

    def create_participant(self, address="gonka1test123", epoch_id=42):
        """Create a test participant."""
        return GonkaParticipant(
            address=address,
            inference_url="https://node.example.com",
            weight=100,
            epoch_id=epoch_id,
        )

    def test_unregister_all_for_epoch(self):
        """Test unregistering all models for an epoch."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        # Create mock multiplexer with weighted models
        mock_wm = Mock()
        mock_wm.model_name = "gonka:gonka1test123"
        mock_wm.disabled_until = None
        
        mock_multiplexer = Mock()
        mock_multiplexer._weighted_models = [mock_wm]
        mock_multiplexer._fallback_models = []
        
        participant = self.create_participant(epoch_id=42)
        
        registrar = ModelRegistrar(mock_factory)
        registrar.register_one(mock_multiplexer, participant)
        
        # Unregister
        count = registrar.unregister_all(mock_multiplexer, epoch_id=42)
        
        assert count == 1
        assert mock_wm.disabled_until == float("inf")

    def test_unregister_nonexistent_epoch(self):
        """Test unregistering models for an epoch with no models."""
        mock_factory = Mock()
        mock_multiplexer = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        count = registrar.unregister_all(mock_multiplexer, epoch_id=99)
        
        assert count == 0

    def test_unregister_clears_tracking(self):
        """Test that unregistration clears epoch tracking."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        mock_wm = Mock()
        mock_wm.model_name = "gonka:gonka1test123"
        
        mock_multiplexer = Mock()
        mock_multiplexer._weighted_models = [mock_wm]
        mock_multiplexer._fallback_models = []
        
        participant = self.create_participant(epoch_id=42)
        
        registrar = ModelRegistrar(mock_factory)
        registrar.register_one(mock_multiplexer, participant)
        
        # Verify tracking before unregister
        assert len(registrar.get_registered_models(epoch_id=42)) == 1
        
        registrar.unregister_all(mock_multiplexer, epoch_id=42)
        
        # Verify tracking cleared after unregister
        assert len(registrar.get_registered_models(epoch_id=42)) == 0

    def test_unregister_fallback_model(self):
        """Test unregistering a fallback model."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        
        # Model is in fallback list
        mock_wm = Mock()
        mock_wm.model_name = "gonka:gonka1test123"
        mock_wm.disabled_until = None
        
        mock_multiplexer = Mock()
        mock_multiplexer._weighted_models = []
        mock_multiplexer._fallback_models = [mock_wm]
        
        participant = self.create_participant(epoch_id=42)
        
        registrar = ModelRegistrar(mock_factory)
        registrar.register_one(mock_multiplexer, participant, as_fallback=True)
        
        count = registrar.unregister_all(mock_multiplexer, epoch_id=42)
        
        assert count == 1
        assert mock_wm.disabled_until == float("inf")


class TestModelRegistrarGetRegisteredModels:
    """Test ModelRegistrar.get_registered_models method."""

    def create_participant(self, address, epoch_id):
        """Create a test participant."""
        return GonkaParticipant(
            address=address,
            inference_url="https://node.example.com",
            weight=100,
            epoch_id=epoch_id,
        )

    def test_get_models_for_specific_epoch(self):
        """Test getting models for a specific epoch."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_multiplexer = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        
        # Register participants for different epochs
        p1 = self.create_participant("gonka1a", epoch_id=41)
        p2 = self.create_participant("gonka1b", epoch_id=42)
        p3 = self.create_participant("gonka1c", epoch_id=42)
        
        registrar.register_one(mock_multiplexer, p1)
        registrar.register_one(mock_multiplexer, p2)
        registrar.register_one(mock_multiplexer, p3)
        
        models_41 = registrar.get_registered_models(epoch_id=41)
        models_42 = registrar.get_registered_models(epoch_id=42)
        
        assert len(models_41) == 1
        assert "gonka:gonka1a" in models_41
        
        assert len(models_42) == 2
        assert "gonka:gonka1b" in models_42
        assert "gonka:gonka1c" in models_42

    def test_get_all_models(self):
        """Test getting all registered models across epochs."""
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_multiplexer = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        
        p1 = self.create_participant("gonka1a", epoch_id=41)
        p2 = self.create_participant("gonka1b", epoch_id=42)
        
        registrar.register_one(mock_multiplexer, p1)
        registrar.register_one(mock_multiplexer, p2)
        
        all_models = registrar.get_registered_models()
        
        assert len(all_models) == 2
        assert "gonka:gonka1a" in all_models
        assert "gonka:gonka1b" in all_models

    def test_get_models_empty_epoch(self):
        """Test getting models for an epoch with no registrations."""
        mock_factory = Mock()
        registrar = ModelRegistrar(mock_factory)
        
        models = registrar.get_registered_models(epoch_id=99)
        
        assert len(models) == 0


class TestModelRegistrarThreadSafety:
    """Test thread safety of ModelRegistrar."""

    def test_concurrent_registration(self):
        """Test that concurrent registrations are handled safely."""
        import threading
        
        mock_factory = Mock()
        mock_client = Mock()
        mock_factory.create_client.return_value = mock_client
        mock_multiplexer = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        
        def register_participant(idx):
            p = GonkaParticipant(
                address=f"gonka1test{idx}",
                inference_url=f"https://node{idx}.example.com",
                weight=100,
                epoch_id=42,
            )
            registrar.register_one(mock_multiplexer, p)
        
        # Create multiple threads
        threads = [
            threading.Thread(target=register_participant, args=(i,))
            for i in range(10)
        ]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All should be registered
        all_models = registrar.get_registered_models()
        assert len(all_models) == 10

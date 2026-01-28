"""
Integration tests for the Gonka integration.

These tests verify that all components work together correctly:
- Discovery → ClientFactory → Registrar flow
- Complete register_gonka_models flow
- Epoch transition handling
- Mixed Gonka + other provider scenarios

Tests use mock HTTP layer to avoid external network calls.
"""

import asyncio
import json
from pathlib import Path
from typing import List
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest

from multiplexer_llm.gonka import (
    GonkaClientFactory,
    EndpointDiscovery,
    ModelRegistrar,
    RefreshManager,
    GonkaParticipant,
    GonkaConfig,
    register_gonka_models,
    GonkaRegistrationResult,
)
from multiplexer_llm.gonka.types import RefreshResult
from multiplexer_llm.gonka.exceptions import (
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
    GonkaClientError,
)


# ==============================================================================
# Discovery → ClientFactory → Registration Flow
# ==============================================================================


class TestDiscoveryToRegistrationFlow:
    """Tests for the complete discovery-to-registration flow."""

    def test_discovery_to_factory_to_multiplexer(
        self,
        tmp_path: Path,
        valid_participants_json: dict,
        test_private_key: str,
        mock_multiplexer,
    ):
        """Test complete flow: discover → create clients → register."""
        # Create temp file for discovery
        participants_file = tmp_path / "participants.json"
        participants_file.write_text(json.dumps(valid_participants_json))
        
        # Create real components
        discovery = EndpointDiscovery(
            source_url=f"file://{participants_file}",
            verify_proofs=False,
        )
        factory = GonkaClientFactory(private_key=test_private_key)
        registrar = ModelRegistrar(factory)
        
        # Run discovery
        participants = discovery.discover()
        assert len(participants) == 3
        
        # Register all participants
        count = registrar.register_all(mock_multiplexer, participants)
        
        # Verify
        assert count == 3
        assert len(mock_multiplexer._added_models) == 3
        
        # Verify model names follow pattern
        model_names = [m["model_name"] for m in mock_multiplexer._added_models]
        for name in model_names:
            assert name.startswith("gonka:")

    def test_weights_transferred_from_discovery_to_multiplexer(
        self,
        tmp_path: Path,
        valid_participants_json: dict,
        test_private_key: str,
        mock_multiplexer,
    ):
        """Verify weights from discovery are preserved through registration."""
        participants_file = tmp_path / "participants.json"
        participants_file.write_text(json.dumps(valid_participants_json))
        
        discovery = EndpointDiscovery(source_url=f"file://{participants_file}")
        factory = GonkaClientFactory(private_key=test_private_key)
        registrar = ModelRegistrar(factory)
        
        participants = discovery.discover()
        registrar.register_all(mock_multiplexer, participants)
        
        # Get weights from multiplexer additions
        weights = [m["weight"] for m in mock_multiplexer._added_models]
        
        # Should match fixture data (100, 200, 50)
        expected_weights = [p["weight"] for p in valid_participants_json["active_participants"]["participants"]]
        assert sorted(weights) == sorted(expected_weights)

    def test_base_urls_normalized_correctly(
        self,
        tmp_path: Path,
        valid_participants_json: dict,
        test_private_key: str,
        mock_multiplexer,
    ):
        """Verify base URLs are properly normalized with /v1 suffix."""
        participants_file = tmp_path / "participants.json"
        participants_file.write_text(json.dumps(valid_participants_json))
        
        discovery = EndpointDiscovery(source_url=f"file://{participants_file}")
        factory = GonkaClientFactory(private_key=test_private_key)
        registrar = ModelRegistrar(factory)
        
        participants = discovery.discover()
        registrar.register_all(mock_multiplexer, participants)
        
        # Check all base URLs end with /v1
        base_urls = [m["base_url"] for m in mock_multiplexer._added_models]
        for url in base_urls:
            assert url.endswith("/v1"), f"URL {url} should end with /v1"


# ==============================================================================
# Complete register_gonka_models Flow
# ==============================================================================


class TestRegisterGonkaModelsFlow:
    """Tests for the register_gonka_models convenience function."""

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    @patch("multiplexer_llm.gonka.config.RefreshManager")
    def test_full_registration_flow(
        self,
        mock_refresh_cls,
        mock_registrar_cls,
        mock_discovery_cls,
        mock_factory_cls,
        mock_multiplexer,
        test_participants: List[GonkaParticipant],
        test_private_key: str,
        clean_gonka_env,
    ):
        """Test complete registration flow through register_gonka_models."""
        # Setup mocks
        mock_discovery = Mock()
        mock_discovery.discover.return_value = test_participants
        mock_discovery_cls.return_value = mock_discovery
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 3
        mock_registrar_cls.return_value = mock_registrar
        
        mock_refresh = Mock()
        mock_refresh_cls.return_value = mock_refresh
        
        # Execute
        result = register_gonka_models(
            mock_multiplexer,
            source_url="https://api.gonka.network",
            private_key=test_private_key,
        )
        
        # Verify result structure
        assert isinstance(result, GonkaRegistrationResult)
        assert result.models_registered == 3
        assert len(result.participants) == 3
        assert result.epoch_id == 42  # From test_participants fixture
        assert result.refresh_manager is not None
        
        # Verify component creation order
        mock_factory_cls.assert_called_once_with(private_key=test_private_key)
        mock_discovery_cls.assert_called_once()
        mock_registrar_cls.assert_called_once()
        mock_refresh_cls.assert_called_once()

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_explicit_endpoints_bypass_discovery(
        self,
        mock_registrar_cls,
        mock_factory_cls,
        mock_multiplexer,
        test_private_key: str,
        valid_endpoint_strings: List[str],
        clean_gonka_env,
    ):
        """Test that explicit endpoints skip HTTP discovery."""
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 3
        mock_registrar_cls.return_value = mock_registrar
        
        result = register_gonka_models(
            mock_multiplexer,
            endpoints=valid_endpoint_strings,
            private_key=test_private_key,
            refresh_enabled=False,
        )
        
        # Should create participants from endpoints
        assert result.models_registered == 3
        assert len(result.participants) == 3
        
        # Participants should have correct addresses
        addresses = [p.address for p in result.participants]
        for endpoint in valid_endpoint_strings:
            expected_address = endpoint.split(";")[1]
            assert expected_address in addresses

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_config_object_used(
        self,
        mock_registrar_cls,
        mock_discovery_cls,
        mock_factory_cls,
        mock_multiplexer,
        test_participants: List[GonkaParticipant],
        test_private_key: str,
        clean_gonka_env,
    ):
        """Test that GonkaConfig object is properly used."""
        mock_discovery = Mock()
        mock_discovery.discover.return_value = test_participants
        mock_discovery_cls.return_value = mock_discovery
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 3
        mock_registrar_cls.return_value = mock_registrar
        
        config = GonkaConfig(
            private_key=test_private_key,
            source_url="https://api.gonka.network",
            verify_proofs=True,
            refresh_enabled=False,
            model_name_prefix="custom:",
        )
        
        result = register_gonka_models(mock_multiplexer, config=config)
        
        # Verify config values were used
        mock_discovery_cls.assert_called_once()
        call_kwargs = mock_discovery_cls.call_args[1]
        assert call_kwargs["verify_proofs"] is True
        
        mock_registrar_cls.assert_called_once()
        reg_call_kwargs = mock_registrar_cls.call_args[1]
        assert reg_call_kwargs["model_name_prefix"] == "custom:"


# ==============================================================================
# Epoch Transition Flow
# ==============================================================================


class TestEpochTransitionFlow:
    """Tests for epoch transition handling."""

    def test_epoch_transition_updates_models(
        self,
        mock_multiplexer,
        epoch_41_participants: List[GonkaParticipant],
        epoch_42_participants: List[GonkaParticipant],
    ):
        """Test that epoch change removes old models and adds new ones."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.side_effect = [41, 42]
        mock_discovery.discover.side_effect = [
            epoch_41_participants,
            epoch_42_participants,
        ]
        
        mock_factory = Mock()
        mock_factory.create_client.return_value = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=registrar,
        )
        
        # First refresh - epoch 41
        result1 = manager.refresh_now()
        assert result1.success is True
        assert result1.new_epoch == 41
        assert result1.participants_added == 2
        
        # Track models added in epoch 41
        epoch_41_models = registrar.get_registered_models(epoch_id=41)
        assert len(epoch_41_models) == 2
        
        # Second refresh - epoch 42
        result2 = manager.refresh_now()
        assert result2.success is True
        assert result2.epoch_changed is True
        assert result2.old_epoch == 41
        assert result2.new_epoch == 42
        assert result2.participants_added == 3
        assert result2.participants_removed == 2
        
        # Verify old models removed from tracking
        assert len(registrar.get_registered_models(epoch_id=41)) == 0
        assert len(registrar.get_registered_models(epoch_id=42)) == 3

    def test_callback_invoked_on_epoch_transition(
        self,
        mock_multiplexer,
        epoch_41_participants: List[GonkaParticipant],
        epoch_42_participants: List[GonkaParticipant],
    ):
        """Test that epoch change callback is invoked correctly."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.side_effect = [41, 42]
        mock_discovery.discover.side_effect = [
            epoch_41_participants,
            epoch_42_participants,
        ]
        
        mock_factory = Mock()
        mock_factory.create_client.return_value = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        callback = Mock()
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=registrar,
        )
        manager.on_epoch_change = callback
        
        # First refresh
        manager.refresh_now()
        callback.assert_not_called()  # No transition yet
        
        # Second refresh - transition
        manager.refresh_now()
        callback.assert_called_once_with(41, 42, epoch_42_participants)

    @pytest.mark.asyncio
    async def test_async_epoch_transition(
        self,
        mock_multiplexer,
        epoch_41_participants: List[GonkaParticipant],
        epoch_42_participants: List[GonkaParticipant],
    ):
        """Test async epoch transition handling."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.side_effect = [41, 42]
        mock_discovery.async_discover = AsyncMock(
            side_effect=[epoch_41_participants, epoch_42_participants]
        )
        
        mock_factory = Mock()
        mock_factory.create_client.return_value = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=registrar,
        )
        
        # First async refresh
        result1 = await manager.async_refresh_now()
        assert result1.success is True
        assert result1.new_epoch == 41
        
        # Second async refresh - transition
        result2 = await manager.async_refresh_now()
        assert result2.success is True
        assert result2.epoch_changed is True


# ==============================================================================
# Mixed Provider Scenarios
# ==============================================================================


class TestMixedProviderScenarios:
    """Tests for Gonka combined with other providers."""

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_gonka_as_fallback(
        self,
        mock_registrar_cls,
        mock_discovery_cls,
        mock_factory_cls,
        mock_multiplexer,
        test_participants: List[GonkaParticipant],
        test_private_key: str,
        clean_gonka_env,
    ):
        """Test Gonka models registered as fallbacks."""
        mock_discovery = Mock()
        mock_discovery.discover.return_value = test_participants
        mock_discovery_cls.return_value = mock_discovery
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 3
        mock_registrar_cls.return_value = mock_registrar
        
        result = register_gonka_models(
            mock_multiplexer,
            source_url="https://api.gonka.network",
            private_key=test_private_key,
            register_as_fallback=True,
            refresh_enabled=False,
        )
        
        # Verify registrar called with as_fallback=True
        mock_registrar.register_all.assert_called_once()
        call_kwargs = mock_registrar.register_all.call_args[1]
        assert call_kwargs["as_fallback"] is True

    def test_mixed_registration_order(
        self,
        mock_multiplexer,
        test_participant: GonkaParticipant,
        test_private_key: str,
    ):
        """Test that registration order is preserved."""
        # First add OpenAI mock
        openai_client = Mock()
        mock_multiplexer.add_model(
            model=openai_client,
            weight=10,
            model_name="openai:gpt-4",
        )
        
        # Then add Gonka
        factory = GonkaClientFactory(private_key=test_private_key)
        registrar = ModelRegistrar(factory)
        registrar.register_one(mock_multiplexer, test_participant)
        
        # Then add Claude mock
        claude_client = Mock()
        mock_multiplexer.add_model(
            model=claude_client,
            weight=5,
            model_name="claude:claude-3",
        )
        
        # Verify all three registered
        assert len(mock_multiplexer._added_models) == 3
        
        model_names = [m["model_name"] for m in mock_multiplexer._added_models]
        assert "openai:gpt-4" in model_names
        assert f"gonka:{test_participant.address}" in model_names
        assert "claude:claude-3" in model_names


# ==============================================================================
# Error Handling Integration
# ==============================================================================


class TestErrorHandlingIntegration:
    """Tests for error handling across components."""

    @patch("requests.get")
    def test_discovery_failure_propagates(
        self,
        mock_get,
        test_private_key: str,
        clean_gonka_env,
    ):
        """Test that discovery failures are properly propagated."""
        mock_get.return_value.status_code = 500
        
        discovery = EndpointDiscovery(
            source_url="https://api.gonka.network",
            retry_count=1,
        )
        
        with pytest.raises(GonkaDiscoveryError) as exc_info:
            discovery.discover()
        
        assert exc_info.value.status_code == 500

    def test_partial_registration_continues(
        self,
        mock_multiplexer,
        test_participants: List[GonkaParticipant],
        test_private_key: str,
    ):
        """Test that partial failures don't stop other registrations."""
        mock_factory = Mock()
        
        # Fail on second participant only
        mock_factory.create_client.side_effect = [
            Mock(),  # First succeeds
            GonkaClientError("Failed"),  # Second fails
            Mock(),  # Third succeeds
        ]
        
        registrar = ModelRegistrar(mock_factory)
        count = registrar.register_all(mock_multiplexer, test_participants)
        
        # Two should succeed
        assert count == 2
        assert len(mock_multiplexer._added_models) == 2

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_refresh_failure_preserves_existing(
        self,
        mock_registrar_cls,
        mock_discovery_cls,
        mock_factory_cls,
        mock_multiplexer,
        test_participants: List[GonkaParticipant],
        test_private_key: str,
        clean_gonka_env,
    ):
        """Test that refresh failure preserves existing registrations."""
        mock_discovery = Mock()
        mock_discovery.discover.side_effect = [
            test_participants,  # First discovery succeeds
            GonkaDiscoveryError("Network error"),  # Refresh fails
        ]
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery_cls.return_value = mock_discovery
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 3
        mock_registrar.unregister_all.return_value = 0
        mock_registrar_cls.return_value = mock_registrar
        
        mock_refresh = Mock()
        mock_refresh.is_running = False
        
        # First registration succeeds
        result = register_gonka_models(
            mock_multiplexer,
            source_url="https://api.gonka.network",
            private_key=test_private_key,
            refresh_enabled=False,
        )
        
        assert result.models_registered == 3


# ==============================================================================
# Performance Integration Tests
# ==============================================================================


class TestPerformanceIntegration:
    """Tests for performance characteristics of the integration."""

    def test_large_participant_list(
        self,
        tmp_path: Path,
        test_private_key: str,
        mock_multiplexer,
    ):
        """Test registration with large number of participants."""
        # Create fixture with many participants
        participants = []
        for i in range(100):
            participants.append({
                "index": f"gonka1test{i:04d}",
                "validator_key": f"pubkey{i:04d}",
                "weight": 100 + i,
                "inference_url": f"https://node{i:04d}.example.com",
                "models": ["model1"],
            })
        
        fixture = {
            "active_participants": {
                "participants": participants,
                "epoch_id": 42,
            }
        }
        
        participants_file = tmp_path / "large_participants.json"
        participants_file.write_text(json.dumps(fixture))
        
        # Time the operation
        import time
        start = time.time()
        
        discovery = EndpointDiscovery(source_url=f"file://{participants_file}")
        factory = GonkaClientFactory(private_key=test_private_key)
        registrar = ModelRegistrar(factory)
        
        participants = discovery.discover()
        count = registrar.register_all(mock_multiplexer, participants)
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert count == 100
        assert elapsed < 5.0, f"Registration took too long: {elapsed}s"

    def test_concurrent_registration_same_participants(
        self,
        mock_multiplexer,
        test_participants: List[GonkaParticipant],
        test_private_key: str,
    ):
        """Test that duplicate registrations are handled correctly."""
        factory = GonkaClientFactory(private_key=test_private_key)
        registrar = ModelRegistrar(factory)
        
        # First registration
        count1 = registrar.register_all(mock_multiplexer, test_participants)
        assert count1 == 3
        
        # Second registration of same participants
        count2 = registrar.register_all(mock_multiplexer, test_participants)
        assert count2 == 0  # All duplicates
        
        # Only 3 models in multiplexer
        assert len(mock_multiplexer._added_models) == 3


# ==============================================================================
# Async Integration Tests
# ==============================================================================


class TestAsyncIntegration:
    """Tests for async operation integration."""

    @pytest.mark.asyncio
    async def test_async_discovery_to_registration(
        self,
        tmp_path: Path,
        valid_participants_json: dict,
        test_private_key: str,
        mock_multiplexer,
    ):
        """Test async discovery to synchronous registration."""
        participants_file = tmp_path / "participants.json"
        participants_file.write_text(json.dumps(valid_participants_json))
        
        discovery = EndpointDiscovery(source_url=f"file://{participants_file}")
        factory = GonkaClientFactory(private_key=test_private_key)
        registrar = ModelRegistrar(factory)
        
        # Async discovery
        participants = await discovery.async_discover()
        assert len(participants) == 3
        
        # Sync registration
        count = registrar.register_all(mock_multiplexer, participants)
        assert count == 3

    @pytest.mark.asyncio
    async def test_background_refresh_integration(
        self,
        mock_multiplexer,
        test_participants: List[GonkaParticipant],
    ):
        """Test background refresh loop integration."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.async_discover = AsyncMock(return_value=test_participants)
        
        mock_factory = Mock()
        mock_factory.create_client.return_value = Mock()
        
        registrar = ModelRegistrar(mock_factory)
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=registrar,
        )
        
        # Start background refresh
        manager.start(interval_seconds=0.1)
        assert manager.is_running is True
        
        # Wait for a refresh cycle
        await asyncio.sleep(0.2)
        
        # Stop
        await manager.async_stop()
        assert manager.is_running is False
        
        # Should have performed at least one refresh
        assert manager.current_epoch == 42

"""
Unit tests for the RefreshManager class.

Tests cover:
- Start/stop lifecycle
- Manual refresh trigger (sync and async)
- Epoch change detection
- Callback invocation
- Error handling during refresh
- Background refresh loop
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from multiplexer_llm.gonka.types import GonkaParticipant, RefreshResult
from multiplexer_llm.gonka.refresh import RefreshManager
from multiplexer_llm.gonka.exceptions import GonkaRefreshError, GonkaDiscoveryError


class TestRefreshManagerInit:
    """Test RefreshManager initialization."""

    def test_init_with_required_args(self):
        """Test initialization with required arguments."""
        mock_multiplexer = Mock()
        mock_discovery = Mock()
        mock_registrar = Mock()
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        assert manager.multiplexer is mock_multiplexer
        assert manager.discovery is mock_discovery
        assert manager.registrar is mock_registrar
        assert manager.is_running is False
        assert manager.current_epoch is None
        assert manager.last_refresh is None
        assert manager.on_epoch_change is None

    def test_init_as_fallback(self):
        """Test initialization with as_fallback=True."""
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=Mock(),
            registrar=Mock(),
            as_fallback=True,
        )
        
        # Should be stored internally
        assert manager._as_fallback is True


class TestRefreshManagerRefreshNow:
    """Test RefreshManager.refresh_now method (synchronous)."""

    def create_participant(self, address="gonka1test123", epoch_id=42):
        """Create a test participant."""
        return GonkaParticipant(
            address=address,
            inference_url="https://node.example.com",
            weight=100,
            epoch_id=epoch_id,
        )

    def test_initial_refresh_registers_participants(self):
        """Test that initial refresh registers participants."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.discover.return_value = [
            self.create_participant("gonka1a", epoch_id=42),
            self.create_participant("gonka1b", epoch_id=42),
        ]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 2
        
        mock_multiplexer = Mock()
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        result = manager.refresh_now()
        
        assert result.success is True
        assert result.epoch_changed is True
        assert result.old_epoch is None  # First refresh
        assert result.new_epoch == 42
        assert result.participants_added == 2
        assert result.participants_removed == 0
        assert manager.current_epoch == 42
        assert manager.last_refresh is not None

    def test_no_change_when_epoch_same(self):
        """Test that no changes are made when epoch hasn't changed."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.discover.return_value = [self.create_participant()]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        # First refresh
        result1 = manager.refresh_now()
        assert result1.epoch_changed is True
        
        # Reset mock call counts
        mock_discovery.discover.reset_mock()
        mock_registrar.register_all.reset_mock()
        
        # Second refresh - same epoch
        result2 = manager.refresh_now()
        
        assert result2.success is True
        assert result2.epoch_changed is False
        mock_discovery.discover.assert_not_called()
        mock_registrar.register_all.assert_not_called()

    def test_epoch_change_unregisters_old_and_registers_new(self):
        """Test that epoch change unregisters old models and registers new."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.side_effect = [41, 42]
        mock_discovery.discover.side_effect = [
            [self.create_participant("gonka1old", epoch_id=41)],
            [self.create_participant("gonka1new", epoch_id=42)],
        ]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        mock_registrar.unregister_all.return_value = 1
        
        mock_multiplexer = Mock()
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        # First refresh - epoch 41
        result1 = manager.refresh_now()
        assert result1.new_epoch == 41
        
        # Second refresh - epoch 42
        result2 = manager.refresh_now()
        
        assert result2.success is True
        assert result2.epoch_changed is True
        assert result2.old_epoch == 41
        assert result2.new_epoch == 42
        assert result2.participants_added == 1
        assert result2.participants_removed == 1
        
        # Verify unregister was called for old epoch
        mock_registrar.unregister_all.assert_called_with(mock_multiplexer, 41)

    def test_discovery_failure_returns_error_result(self):
        """Test that discovery failure returns error result."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.side_effect = GonkaDiscoveryError("Network error")
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=Mock(),
        )
        
        result = manager.refresh_now()
        
        assert result.success is False
        assert isinstance(result.error, GonkaRefreshError)
        assert "Failed to get current epoch" in str(result.error)

    def test_participant_discovery_failure(self):
        """Test handling when participant discovery fails."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.discover.side_effect = GonkaDiscoveryError("Failed")
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=Mock(),
        )
        
        result = manager.refresh_now()
        
        assert result.success is False
        assert isinstance(result.error, GonkaRefreshError)

    def test_registers_as_fallback_when_configured(self):
        """Test that models are registered as fallbacks when configured."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.discover.return_value = [self.create_participant()]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        mock_multiplexer = Mock()
        
        manager = RefreshManager(
            multiplexer=mock_multiplexer,
            discovery=mock_discovery,
            registrar=mock_registrar,
            as_fallback=True,
        )
        
        manager.refresh_now()
        
        mock_registrar.register_all.assert_called_with(
            mock_multiplexer,
            mock_discovery.discover.return_value,
            as_fallback=True,
        )


class TestRefreshManagerCallback:
    """Test RefreshManager callback functionality."""

    def create_participant(self, address="gonka1test123", epoch_id=42):
        """Create a test participant."""
        return GonkaParticipant(
            address=address,
            inference_url="https://node.example.com",
            weight=100,
            epoch_id=epoch_id,
        )

    def test_callback_invoked_on_epoch_change(self):
        """Test that callback is invoked when epoch changes."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.side_effect = [41, 42]
        participants_41 = [self.create_participant("gonka1a", 41)]
        participants_42 = [self.create_participant("gonka1b", 42)]
        mock_discovery.discover.side_effect = [participants_41, participants_42]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        mock_registrar.unregister_all.return_value = 1
        
        callback = Mock()
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        manager.on_epoch_change = callback
        
        # First refresh (no callback - no old epoch)
        manager.refresh_now()
        callback.assert_not_called()
        
        # Second refresh (callback should be invoked)
        manager.refresh_now()
        callback.assert_called_once_with(41, 42, participants_42)

    def test_callback_not_invoked_on_first_registration(self):
        """Test that callback is not invoked on initial registration."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.discover.return_value = [self.create_participant()]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        callback = Mock()
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        manager.on_epoch_change = callback
        
        manager.refresh_now()
        
        # Callback should not be called for initial registration
        callback.assert_not_called()

    def test_callback_error_does_not_fail_refresh(self):
        """Test that callback error doesn't fail the refresh."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.side_effect = [41, 42]
        mock_discovery.discover.side_effect = [
            [self.create_participant("gonka1a", 41)],
            [self.create_participant("gonka1b", 42)],
        ]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        mock_registrar.unregister_all.return_value = 1
        
        callback = Mock(side_effect=RuntimeError("Callback failed"))
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        manager.on_epoch_change = callback
        
        # First refresh
        manager.refresh_now()
        
        # Second refresh - callback fails but refresh should succeed
        result = manager.refresh_now()
        
        assert result.success is True
        assert result.epoch_changed is True

    def test_callback_setter_getter(self):
        """Test callback setter and getter."""
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=Mock(),
            registrar=Mock(),
        )
        
        assert manager.on_epoch_change is None
        
        callback = Mock()
        manager.on_epoch_change = callback
        
        assert manager.on_epoch_change is callback


class TestRefreshManagerStartStop:
    """Test RefreshManager start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_background_task(self):
        """Test that start() creates a background task."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.async_discover = AsyncMock(return_value=[])
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 0
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        assert manager.is_running is False
        
        manager.start(interval_seconds=0.1)
        
        assert manager.is_running is True
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        # Stop it
        await manager.async_stop()
        
        assert manager.is_running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_background_task(self):
        """Test that stop() cancels the background task."""
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=Mock(),
            registrar=Mock(),
        )
        
        manager.start(interval_seconds=60.0)
        assert manager.is_running is True
        
        await manager.async_stop()
        
        assert manager.is_running is False

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        """Test that start() does nothing when already running."""
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=Mock(),
            registrar=Mock(),
        )
        
        manager.start(interval_seconds=60.0)
        task1 = manager._task
        
        manager.start(interval_seconds=30.0)
        task2 = manager._task
        
        # Should be the same task
        assert task1 is task2
        
        await manager.async_stop()

    def test_stop_when_not_running(self):
        """Test that stop() does nothing when not running."""
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=Mock(),
            registrar=Mock(),
        )
        
        # Should not raise
        manager.stop()
        assert manager.is_running is False


class TestRefreshManagerAsyncRefresh:
    """Test RefreshManager async refresh methods."""

    def create_participant(self, address="gonka1test123", epoch_id=42):
        """Create a test participant."""
        return GonkaParticipant(
            address=address,
            inference_url="https://node.example.com",
            weight=100,
            epoch_id=epoch_id,
        )

    @pytest.mark.asyncio
    async def test_async_refresh_now(self):
        """Test async_refresh_now method."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.async_discover = AsyncMock(
            return_value=[self.create_participant()]
        )
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        result = await manager.async_refresh_now()
        
        assert result.success is True
        assert result.epoch_changed is True
        assert result.new_epoch == 42
        mock_discovery.async_discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_initial_registration(self):
        """Test async_initial_registration method."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.async_discover = AsyncMock(
            return_value=[self.create_participant()]
        )
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        result = await manager.async_initial_registration()
        
        assert result.success is True
        assert result.epoch_changed is True

    @pytest.mark.asyncio
    async def test_async_discovery_failure(self):
        """Test async refresh handles discovery failure."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.async_discover = AsyncMock(
            side_effect=GonkaDiscoveryError("Network error")
        )
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=Mock(),
        )
        
        result = await manager.async_refresh_now()
        
        assert result.success is False
        assert isinstance(result.error, GonkaRefreshError)


class TestRefreshManagerBackgroundLoop:
    """Test RefreshManager background refresh loop."""

    @pytest.mark.asyncio
    async def test_background_loop_performs_refresh(self):
        """Test that background loop performs refreshes."""
        refresh_count = 0
        
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        
        async def mock_async_discover():
            nonlocal refresh_count
            refresh_count += 1
            return [
                GonkaParticipant(
                    address="gonka1test",
                    inference_url="https://node.example.com",
                    weight=100,
                    epoch_id=42,
                )
            ]
        
        mock_discovery.async_discover = mock_async_discover
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        # Start with very short interval
        manager.start(interval_seconds=0.05)
        
        # Wait for a couple refreshes
        await asyncio.sleep(0.2)
        
        await manager.async_stop()
        
        # Should have done at least one refresh
        assert refresh_count >= 1

    @pytest.mark.asyncio
    async def test_background_loop_handles_errors_gracefully(self):
        """Test that background loop continues after errors."""
        call_count = 0
        
        mock_discovery = Mock()
        
        def get_epoch():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise GonkaDiscoveryError("Temporary failure")
            return 42
        
        mock_discovery.get_current_epoch = get_epoch
        mock_discovery.async_discover = AsyncMock(return_value=[])
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 0
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        manager.start(interval_seconds=0.05)
        
        # Wait for a few iterations
        await asyncio.sleep(0.2)
        
        await manager.async_stop()
        
        # Should have continued despite error
        assert call_count >= 2


class TestRefreshResult:
    """Test RefreshResult data class."""

    def test_success_result(self):
        """Test successful refresh result."""
        result = RefreshResult(success=True, epoch_changed=False)
        
        assert result.success is True
        assert result.epoch_changed is False
        assert result.old_epoch is None
        assert result.new_epoch is None
        assert result.participants_added == 0
        assert result.participants_removed == 0
        assert result.error is None

    def test_epoch_change_result(self):
        """Test epoch change result."""
        result = RefreshResult(
            success=True,
            epoch_changed=True,
            old_epoch=41,
            new_epoch=42,
            participants_added=5,
            participants_removed=3,
        )
        
        assert result.success is True
        assert result.epoch_changed is True
        assert result.old_epoch == 41
        assert result.new_epoch == 42
        assert result.participants_added == 5
        assert result.participants_removed == 3

    def test_error_result(self):
        """Test error result."""
        error = GonkaRefreshError("Test error")
        result = RefreshResult(success=False, error=error)
        
        assert result.success is False
        assert result.error is error

    def test_repr_success_no_change(self):
        """Test repr for success with no epoch change."""
        result = RefreshResult(success=True, epoch_changed=False)
        assert "success=True" in repr(result)
        assert "epoch_changed=False" in repr(result)

    def test_repr_epoch_change(self):
        """Test repr for epoch change."""
        result = RefreshResult(
            success=True,
            epoch_changed=True,
            old_epoch=41,
            new_epoch=42,
            participants_added=5,
            participants_removed=3,
        )
        repr_str = repr(result)
        assert "success=True" in repr_str
        assert "epoch_changed=True" in repr_str
        assert "old_epoch=41" in repr_str
        assert "new_epoch=42" in repr_str

    def test_repr_error(self):
        """Test repr for error result."""
        result = RefreshResult(
            success=False,
            error=GonkaRefreshError("Test"),
        )
        assert "success=False" in repr(result)
        assert "GonkaRefreshError" in repr(result)


class TestRefreshManagerInitialRegistration:
    """Test initial registration convenience methods."""

    def create_participant(self, address="gonka1test123", epoch_id=42):
        """Create a test participant."""
        return GonkaParticipant(
            address=address,
            inference_url="https://node.example.com",
            weight=100,
            epoch_id=epoch_id,
        )

    def test_initial_registration_sync(self):
        """Test synchronous initial registration."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.discover.return_value = [self.create_participant()]
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        result = manager.initial_registration()
        
        assert result.success is True
        assert manager.current_epoch == 42

    @pytest.mark.asyncio
    async def test_initial_registration_async(self):
        """Test asynchronous initial registration."""
        mock_discovery = Mock()
        mock_discovery.get_current_epoch.return_value = 42
        mock_discovery.async_discover = AsyncMock(
            return_value=[self.create_participant()]
        )
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 1
        
        manager = RefreshManager(
            multiplexer=Mock(),
            discovery=mock_discovery,
            registrar=mock_registrar,
        )
        
        result = await manager.async_initial_registration()
        
        assert result.success is True
        assert manager.current_epoch == 42

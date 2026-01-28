"""
RefreshManager - Background service for epoch transitions.

This module provides the RefreshManager class, which monitors the Gonka network
for epoch changes and automatically updates registered models in the multiplexer.
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Callable, List, Optional

from .types import GonkaParticipant, RefreshResult
from .exceptions import GonkaRefreshError
from .discovery import EndpointDiscovery
from .registrar import ModelRegistrar

logger = logging.getLogger(__name__)


class RefreshManager:
    """
    Background service that monitors for epoch changes and updates models.

    The RefreshManager periodically polls the Gonka network to detect epoch
    transitions. When an epoch change is detected, it automatically:
    1. Unregisters models from the old epoch
    2. Discovers new participants
    3. Registers new participants with the multiplexer
    4. Invokes the epoch change callback if configured

    The manager supports both automatic background refresh and manual refresh
    triggers for testing and forced updates.

    Thread Safety:
        This class is thread-safe. The refresh operation is protected by a lock
        to prevent concurrent refreshes.

    Attributes:
        multiplexer: The Multiplexer instance being managed.
        discovery: The EndpointDiscovery instance for fetching participants.
        registrar: The ModelRegistrar for model registration.
        as_fallback: Whether to register Gonka models as fallbacks.
        on_epoch_change: Optional callback for epoch change notifications.
        last_refresh: Timestamp of the last successful refresh.
        current_epoch: The current epoch ID.
        is_running: Whether the background refresh task is running.

    Example:
        >>> manager = RefreshManager(
        ...     multiplexer=multiplexer,
        ...     discovery=discovery,
        ...     registrar=registrar,
        ... )
        >>> manager.start(interval_seconds=60.0)
        >>> # ... later ...
        >>> manager.stop()
    """

    def __init__(
        self,
        multiplexer: Any,
        discovery: EndpointDiscovery,
        registrar: ModelRegistrar,
        as_fallback: bool = False,
    ) -> None:
        """
        Initialize the refresh manager.

        Args:
            multiplexer: The Multiplexer instance to manage models for.
            discovery: The EndpointDiscovery instance for fetching participants.
            registrar: The ModelRegistrar for registering/unregistering models.
            as_fallback: Whether to register Gonka models as fallbacks.
        """
        self._multiplexer = multiplexer
        self._discovery = discovery
        self._registrar = registrar
        self._as_fallback = as_fallback

        # State
        self._current_epoch: Optional[int] = None
        self._last_refresh: Optional[datetime] = None

        # Callback
        self._on_epoch_change: Optional[
            Callable[[int, int, List[GonkaParticipant]], None]
        ] = None

        # Background task management
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event: Optional[asyncio.Event] = None

        # Thread safety for refresh operations
        self._refresh_lock = threading.Lock()
        self._async_refresh_lock: Optional[asyncio.Lock] = None

        logger.debug("RefreshManager initialized")

    @property
    def multiplexer(self) -> Any:
        """Get the multiplexer being managed."""
        return self._multiplexer

    @property
    def discovery(self) -> EndpointDiscovery:
        """Get the endpoint discovery service."""
        return self._discovery

    @property
    def registrar(self) -> ModelRegistrar:
        """Get the model registrar."""
        return self._registrar

    @property
    def current_epoch(self) -> Optional[int]:
        """Get the current epoch ID."""
        return self._current_epoch

    @property
    def last_refresh(self) -> Optional[datetime]:
        """Get the timestamp of the last successful refresh."""
        return self._last_refresh

    @property
    def is_running(self) -> bool:
        """Check if the background refresh task is running."""
        return self._running

    @property
    def on_epoch_change(
        self,
    ) -> Optional[Callable[[int, int, List[GonkaParticipant]], None]]:
        """Get the epoch change callback."""
        return self._on_epoch_change

    @on_epoch_change.setter
    def on_epoch_change(
        self,
        callback: Optional[Callable[[int, int, List[GonkaParticipant]], None]],
    ) -> None:
        """
        Set the epoch change callback.

        The callback is invoked whenever an epoch change is detected during
        a refresh operation. It receives:
        - old_epoch: The previous epoch ID
        - new_epoch: The new epoch ID
        - participants: List of new participants

        Args:
            callback: The callback function, or None to disable.

        Example:
            >>> def on_change(old_epoch, new_epoch, participants):
            ...     print(f"Epoch changed from {old_epoch} to {new_epoch}")
            >>> manager.on_epoch_change = on_change
        """
        self._on_epoch_change = callback

    def start(self, interval_seconds: float = 60.0) -> None:
        """
        Start the background refresh task.

        This method starts an asyncio task that periodically polls for epoch
        changes. If already running, this method does nothing.

        Args:
            interval_seconds: How often to check for epoch changes (default: 60s).

        Raises:
            RuntimeError: If called outside an asyncio event loop.

        Example:
            >>> manager.start(interval_seconds=30.0)
        """
        if self._running:
            logger.warning("RefreshManager is already running")
            return

        self._running = True
        self._stop_event = asyncio.Event()

        # Create the background task
        self._task = asyncio.create_task(
            self._background_refresh_loop(interval_seconds)
        )

        logger.info(
            "RefreshManager started with interval=%.1fs",
            interval_seconds,
        )

    def stop(self) -> None:
        """
        Stop the background refresh task.

        This method signals the background task to stop and waits for the
        current refresh (if any) to complete. If not running, this method
        does nothing.

        Example:
            >>> manager.stop()
        """
        if not self._running:
            logger.debug("RefreshManager is not running")
            return

        self._running = False

        # Signal the stop event
        if self._stop_event:
            self._stop_event.set()

        # Cancel the task if it's still running
        if self._task and not self._task.done():
            self._task.cancel()

        logger.info("RefreshManager stopped")

    async def async_stop(self) -> None:
        """
        Stop the background refresh task asynchronously.

        This method properly awaits task cancellation.

        Example:
            >>> await manager.async_stop()
        """
        if not self._running:
            logger.debug("RefreshManager is not running")
            return

        self._running = False

        # Signal the stop event
        if self._stop_event:
            self._stop_event.set()

        # Wait for the task to complete
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("RefreshManager stopped")

    def refresh_now(self) -> RefreshResult:
        """
        Trigger an immediate refresh (synchronous).

        This method performs a refresh check immediately, regardless of the
        background refresh interval. Useful for testing or forcing an update.

        Returns:
            RefreshResult indicating what happened during the refresh.

        Example:
            >>> result = manager.refresh_now()
            >>> if result.epoch_changed:
            ...     print(f"New epoch: {result.new_epoch}")
        """
        with self._refresh_lock:
            return self._do_refresh()

    async def async_refresh_now(self) -> RefreshResult:
        """
        Trigger an immediate refresh (asynchronous).

        This method performs a refresh check immediately, regardless of the
        background refresh interval. Uses async discovery if available.

        Returns:
            RefreshResult indicating what happened during the refresh.

        Example:
            >>> result = await manager.async_refresh_now()
            >>> if result.epoch_changed:
            ...     print(f"New epoch: {result.new_epoch}")
        """
        # Initialize async lock if needed
        if self._async_refresh_lock is None:
            self._async_refresh_lock = asyncio.Lock()

        async with self._async_refresh_lock:
            return await self._async_do_refresh()

    async def _background_refresh_loop(self, interval_seconds: float) -> None:
        """
        Background task that periodically checks for epoch changes.

        Args:
            interval_seconds: How often to check for epoch changes.
        """
        logger.debug("Background refresh loop started")

        while self._running:
            try:
                # Wait for the interval or stop event
                if self._stop_event:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=interval_seconds,
                        )
                        # Stop event was set
                        break
                    except asyncio.TimeoutError:
                        # Timeout - time to refresh
                        pass

                if not self._running:
                    break

                # Perform refresh
                try:
                    result = await self.async_refresh_now()
                    if result.success:
                        if result.epoch_changed:
                            logger.info(
                                "Background refresh detected epoch change: %d -> %d",
                                result.old_epoch,
                                result.new_epoch,
                            )
                        else:
                            logger.debug("Background refresh: no epoch change")
                    else:
                        logger.warning(
                            "Background refresh failed: %s",
                            result.error,
                        )
                except Exception as e:
                    logger.error("Background refresh error: %s", e)

            except asyncio.CancelledError:
                logger.debug("Background refresh loop cancelled")
                break
            except Exception as e:
                logger.error("Unexpected error in background refresh loop: %s", e)
                # Continue running after unexpected errors
                await asyncio.sleep(1.0)

        logger.debug("Background refresh loop ended")

    def _do_refresh(self) -> RefreshResult:
        """
        Perform a synchronous refresh operation.

        Returns:
            RefreshResult indicating what happened.
        """
        try:
            # Get current epoch from discovery
            try:
                new_epoch = self._discovery.get_current_epoch()
            except Exception as e:
                logger.warning("Failed to get current epoch: %s", e)
                return RefreshResult(
                    success=False,
                    error=GonkaRefreshError(
                        "Failed to get current epoch",
                        last_successful_epoch=self._current_epoch,
                        cause=e,
                    ),
                )

            # Check if epoch changed
            if self._current_epoch is not None and new_epoch == self._current_epoch:
                self._last_refresh = datetime.now()
                return RefreshResult(success=True, epoch_changed=False)

            # Epoch changed - discover new participants
            old_epoch = self._current_epoch

            try:
                participants = self._discovery.discover()
            except Exception as e:
                logger.warning("Failed to discover participants: %s", e)
                return RefreshResult(
                    success=False,
                    error=GonkaRefreshError(
                        "Failed to discover participants",
                        last_successful_epoch=self._current_epoch,
                        cause=e,
                    ),
                )

            # Unregister old epoch models
            removed_count = 0
            if old_epoch is not None:
                removed_count = self._registrar.unregister_all(
                    self._multiplexer,
                    old_epoch,
                )

            # Register new participants
            added_count = self._registrar.register_all(
                self._multiplexer,
                participants,
                as_fallback=self._as_fallback,
            )

            # Update state
            self._current_epoch = new_epoch
            self._last_refresh = datetime.now()

            # Invoke callback
            if self._on_epoch_change and old_epoch is not None:
                try:
                    self._on_epoch_change(old_epoch, new_epoch, participants)
                except Exception as e:
                    logger.error("Epoch change callback error: %s", e)

            return RefreshResult(
                success=True,
                epoch_changed=True,
                old_epoch=old_epoch,
                new_epoch=new_epoch,
                participants_added=added_count,
                participants_removed=removed_count,
            )

        except Exception as e:
            logger.error("Refresh operation failed: %s", e)
            return RefreshResult(
                success=False,
                error=GonkaRefreshError(
                    "Refresh operation failed",
                    last_successful_epoch=self._current_epoch,
                    cause=e,
                ),
            )

    async def _async_do_refresh(self) -> RefreshResult:
        """
        Perform an asynchronous refresh operation.

        Returns:
            RefreshResult indicating what happened.
        """
        try:
            # Get current epoch from discovery
            try:
                new_epoch = self._discovery.get_current_epoch()
            except Exception as e:
                logger.warning("Failed to get current epoch: %s", e)
                return RefreshResult(
                    success=False,
                    error=GonkaRefreshError(
                        "Failed to get current epoch",
                        last_successful_epoch=self._current_epoch,
                        cause=e,
                    ),
                )

            # Check if epoch changed
            if self._current_epoch is not None and new_epoch == self._current_epoch:
                self._last_refresh = datetime.now()
                return RefreshResult(success=True, epoch_changed=False)

            # Epoch changed - discover new participants
            old_epoch = self._current_epoch

            try:
                # Use async discovery if available
                participants = await self._discovery.async_discover()
            except Exception as e:
                logger.warning("Failed to discover participants: %s", e)
                return RefreshResult(
                    success=False,
                    error=GonkaRefreshError(
                        "Failed to discover participants",
                        last_successful_epoch=self._current_epoch,
                        cause=e,
                    ),
                )

            # Unregister old epoch models
            removed_count = 0
            if old_epoch is not None:
                removed_count = self._registrar.unregister_all(
                    self._multiplexer,
                    old_epoch,
                )

            # Register new participants
            added_count = self._registrar.register_all(
                self._multiplexer,
                participants,
                as_fallback=self._as_fallback,
            )

            # Update state
            self._current_epoch = new_epoch
            self._last_refresh = datetime.now()

            # Invoke callback
            if self._on_epoch_change and old_epoch is not None:
                try:
                    self._on_epoch_change(old_epoch, new_epoch, participants)
                except Exception as e:
                    logger.error("Epoch change callback error: %s", e)

            return RefreshResult(
                success=True,
                epoch_changed=True,
                old_epoch=old_epoch,
                new_epoch=new_epoch,
                participants_added=added_count,
                participants_removed=removed_count,
            )

        except Exception as e:
            logger.error("Async refresh operation failed: %s", e)
            return RefreshResult(
                success=False,
                error=GonkaRefreshError(
                    "Refresh operation failed",
                    last_successful_epoch=self._current_epoch,
                    cause=e,
                ),
            )

    def initial_registration(self) -> RefreshResult:
        """
        Perform initial registration of Gonka participants.

        This method should be called once during setup to register the
        initial set of participants. It's equivalent to refresh_now()
        but is semantically clearer for initialization.

        Returns:
            RefreshResult indicating what happened.

        Example:
            >>> result = manager.initial_registration()
            >>> print(f"Registered {result.participants_added} models")
        """
        return self.refresh_now()

    async def async_initial_registration(self) -> RefreshResult:
        """
        Perform initial registration asynchronously.

        Returns:
            RefreshResult indicating what happened.

        Example:
            >>> result = await manager.async_initial_registration()
            >>> print(f"Registered {result.participants_added} models")
        """
        return await self.async_refresh_now()

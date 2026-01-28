#!/usr/bin/env python3
"""
Epoch Handling Example

This example demonstrates handling epoch transitions in the Gonka network:
1. Understanding epochs and transitions
2. Setting up epoch change callbacks
3. Manual refresh triggers
4. Monitoring epoch state

Prerequisites:
    export GONKA_PRIVATE_KEY="0x..."
    export GONKA_SOURCE_URL="https://api.gonka.network"

Usage:
    python examples/gonka/epoch_handling.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime

from multiplexer_llm import Multiplexer
from multiplexer_llm.gonka import (
    register_gonka_models,
    EndpointDiscovery,
    GonkaClientFactory,
    ModelRegistrar,
    RefreshManager,
    GonkaConfigError,
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
)


def example_automatic_epoch_refresh():
    """Automatic epoch refresh with callbacks."""
    
    print("=" * 60)
    print("Example 1: Automatic Epoch Refresh")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL", "https://api.gonka.network")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    mux = Multiplexer()
    
    # Define epoch change callback
    def on_epoch_change(old_epoch, new_epoch, participants):
        """Called when epoch changes."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] EPOCH CHANGE DETECTED!")
        print(f"  Old epoch: {old_epoch}")
        print(f"  New epoch: {new_epoch}")
        print(f"  New participants: {len(participants)}")
        for p in participants[:3]:  # Show first 3
            print(f"    - {p.address} (weight={p.weight})")
        if len(participants) > 3:
            print(f"    ... and {len(participants) - 3} more")
    
    try:
        # Register with automatic refresh enabled
        result = register_gonka_models(
            mux,
            source_url=source_url,
            private_key=private_key,
            refresh_enabled=True,
            refresh_interval_seconds=10.0,  # Check every 10 seconds for demo
        )
        
        print(f"\nInitial registration:")
        print(f"  Epoch: {result.epoch_id}")
        print(f"  Models: {result.models_registered}")
        print(f"  Refresh manager: {'enabled' if result.refresh_manager else 'disabled'}")
        
        # Set up the callback
        if result.refresh_manager:
            result.refresh_manager.on_epoch_change = on_epoch_change
            print("\n✓ Epoch change callback registered")
            print(f"  Refresh interval: {10.0}s")
            print(f"  Is running: {result.refresh_manager.is_running}")
        
        # In a real application, the refresh runs in background
        # For demo, we just show the setup
        
        # Clean up
        if result.refresh_manager:
            result.refresh_manager.stop()
            print("\n✓ Refresh manager stopped")
            
    except GonkaNoParticipantsError:
        print("No participants found - using demo mode")
    except GonkaDiscoveryError as e:
        print(f"Discovery error: {e}")


def example_manual_refresh():
    """Manual refresh trigger."""
    
    print("\n" + "=" * 60)
    print("Example 2: Manual Refresh")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    # Use static endpoints for demo (no network required)
    mux = Multiplexer()
    
    try:
        result = register_gonka_models(
            mux,
            private_key=private_key,
            endpoints=[
                "https://node.example.com;gonka1demo123abc456def789ghi",
            ],
            refresh_enabled=False,  # Disable automatic refresh
        )
        
        print(f"\nInitial state:")
        print(f"  Models registered: {result.models_registered}")
        print(f"  Refresh manager: {'available' if result.refresh_manager else 'not available'}")
        
        # Note: With static endpoints, refresh manager won't be created
        # For manual refresh, you'd use source_url
        print("\nNote: Manual refresh requires source_url (not static endpoints)")
        print("With source_url, you can call:")
        print("  result.refresh_manager.refresh_now()")
        print("  # or async:")
        print("  await result.refresh_manager.async_refresh_now()")
        
    except GonkaConfigError as e:
        print(f"Configuration error: {e}")


def example_manual_components():
    """Using components directly for full control."""
    
    print("\n" + "=" * 60)
    print("Example 3: Manual Component Setup")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL", "https://api.gonka.network")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    # Create components manually
    print("\n--- Creating components ---")
    
    try:
        discovery = EndpointDiscovery(
            source_url=source_url,
            verify_proofs=False,
            timeout=30.0,
        )
        print(f"✓ EndpointDiscovery created (source: {source_url})")
    except Exception as e:
        print(f"✗ Discovery creation failed: {e}")
        return
    
    factory = GonkaClientFactory(private_key=private_key)
    print(f"✓ GonkaClientFactory created (address: {factory.requester_address})")
    
    registrar = ModelRegistrar(
        factory,
        model_name_prefix="manual:",
        default_max_concurrent=5,
    )
    print("✓ ModelRegistrar created")
    
    mux = Multiplexer()
    
    # Create RefreshManager
    refresh_manager = RefreshManager(
        multiplexer=mux,
        discovery=discovery,
        registrar=registrar,
        as_fallback=False,
    )
    print("✓ RefreshManager created")
    
    # Set up callback
    def my_callback(old_epoch, new_epoch, participants):
        print(f"\n  [Callback] Epoch {old_epoch} -> {new_epoch}")
        print(f"  [Callback] {len(participants)} participants")
    
    refresh_manager.on_epoch_change = my_callback
    
    # Perform initial registration
    print("\n--- Initial registration ---")
    refresh_result = refresh_manager.initial_registration()
    
    print(f"  Success: {refresh_result.success}")
    if refresh_result.success:
        print(f"  Epoch changed: {refresh_result.epoch_changed}")
        print(f"  Participants added: {refresh_result.participants_added}")
        print(f"  Current epoch: {refresh_manager.current_epoch}")
        print(f"  Last refresh: {refresh_manager.last_refresh}")
    else:
        print(f"  Error: {refresh_result.error}")


async def example_async_refresh():
    """Async refresh operations."""
    
    print("\n" + "=" * 60)
    print("Example 4: Async Refresh Operations")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL", "https://api.gonka.network")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    # Create components
    discovery = EndpointDiscovery(source_url=source_url)
    factory = GonkaClientFactory(private_key=private_key)
    registrar = ModelRegistrar(factory)
    mux = Multiplexer()
    
    refresh_manager = RefreshManager(
        multiplexer=mux,
        discovery=discovery,
        registrar=registrar,
    )
    
    print("\n--- Async initial registration ---")
    try:
        result = await refresh_manager.async_initial_registration()
        
        if result.success:
            print(f"✓ Initial registration complete")
            print(f"  Epoch: {refresh_manager.current_epoch}")
            print(f"  Participants added: {result.participants_added}")
        else:
            print(f"✗ Registration failed: {result.error}")
            return
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Simulate waiting and then manual refresh
    print("\n--- Async manual refresh ---")
    try:
        result = await refresh_manager.async_refresh_now()
        
        if result.success:
            if result.epoch_changed:
                print(f"✓ Epoch changed: {result.old_epoch} -> {result.new_epoch}")
            else:
                print("✓ No epoch change detected")
        else:
            print(f"✗ Refresh failed: {result.error}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def example_epoch_monitoring():
    """Monitoring epoch state."""
    
    print("\n" + "=" * 60)
    print("Example 5: Epoch Monitoring")
    print("=" * 60)
    
    private_key = os.environ.get("GONKA_PRIVATE_KEY")
    source_url = os.environ.get("GONKA_SOURCE_URL", "https://api.gonka.network")
    
    if not private_key:
        print("Skipping: GONKA_PRIVATE_KEY not set")
        return
    
    # This example shows how to monitor epoch state
    print("\nMonitoring strategy:")
    print("  1. Register with refresh enabled")
    print("  2. Set up epoch change callback")
    print("  3. Log/alert on changes")
    print("  4. Track metrics")
    
    print("\nSample monitoring code:")
    print("""
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gonka_monitor")
    
    # Track metrics
    metrics = {
        "epoch_changes": 0,
        "participants_added": 0,
        "participants_removed": 0,
    }
    
    def on_epoch_change(old_epoch, new_epoch, participants):
        # Update metrics
        metrics["epoch_changes"] += 1
        
        # Log the change
        logger.info(f"Epoch changed: {old_epoch} -> {new_epoch}")
        logger.info(f"New participant count: {len(participants)}")
        
        # Alert if participant count is low
        if len(participants) < 3:
            logger.warning("Low participant count!")
        
        # Update monitoring systems
        # send_metric("gonka.epoch", new_epoch)
        # send_metric("gonka.participants", len(participants))
    
    # Set up
    result = register_gonka_models(mux, ...)
    result.refresh_manager.on_epoch_change = on_epoch_change
    """)


def main():
    """Run all epoch handling examples."""
    
    print("Gonka Epoch Handling Examples")
    print("=" * 60)
    
    # Check for required keys
    if not os.environ.get("GONKA_PRIVATE_KEY"):
        print("\nWarning: GONKA_PRIVATE_KEY not set")
        print("Most examples will be skipped.\n")
    
    if not os.environ.get("GONKA_SOURCE_URL"):
        print("Note: GONKA_SOURCE_URL not set, using default")
        print()
    
    # Run examples
    example_automatic_epoch_refresh()
    example_manual_refresh()
    example_manual_components()
    asyncio.run(example_async_refresh())
    example_epoch_monitoring()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    
    print("\nKey takeaways:")
    print("  - Epochs define time periods with fixed participant sets")
    print("  - Use refresh_enabled=True for automatic updates")
    print("  - Set on_epoch_change callback for notifications")
    print("  - Use refresh_now() for manual/forced updates")
    print("  - Monitor epoch state via RefreshManager properties")


if __name__ == "__main__":
    main()

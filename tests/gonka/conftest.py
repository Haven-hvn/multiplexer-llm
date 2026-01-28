"""
Shared pytest fixtures for Gonka integration tests.

This module provides reusable fixtures for testing the Gonka integration,
including mock participants, factories, discovery services, and multiplexers.

Usage:
    All fixtures are automatically available in any test file under tests/gonka/
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest

from multiplexer_llm.gonka.types import GonkaParticipant, RefreshResult
from multiplexer_llm.gonka.exceptions import (
    GonkaError,
    GonkaClientError,
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
    GonkaConfigError,
)


# ==============================================================================
# Constants
# ==============================================================================

# Test private key (32 bytes of 0xaa - DO NOT use in production!)
TEST_PRIVATE_KEY = "0x" + "a" * 64
TEST_PRIVATE_KEY_NO_PREFIX = "a" * 64

# Alternative test keys for comparison tests
TEST_PRIVATE_KEY_B = "0x" + "b" * 64
TEST_PRIVATE_KEY_C = "0x" + "c" * 64

# Invalid test keys for error testing
TEST_INVALID_KEY_TOO_SHORT = "0x" + "a" * 32
TEST_INVALID_KEY_TOO_LONG = "0x" + "a" * 128
TEST_INVALID_KEY_NOT_HEX = "0x" + "z" * 64

# Test URLs
TEST_SOURCE_URL = "https://api.gonka.network"
TEST_NODE_URL_1 = "https://node1.example.com"
TEST_NODE_URL_2 = "https://node2.example.com"
TEST_NODE_URL_3 = "https://node3.example.com"

# Test addresses
TEST_ADDRESS_1 = "gonka1abc123def456"
TEST_ADDRESS_2 = "gonka1xyz789ghi012"
TEST_ADDRESS_3 = "gonka1jkl345mno678"

# Fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ==============================================================================
# Environment Management Fixtures
# ==============================================================================


@pytest.fixture
def clean_gonka_env():
    """
    Remove all GONKA_ environment variables for test isolation.
    
    Saves original values, clears during test, and restores after.
    """
    env_vars_to_clear = [k for k in os.environ if k.startswith("GONKA_")]
    original_env = {k: os.environ[k] for k in env_vars_to_clear}
    
    # Clear all GONKA_ vars
    for var in env_vars_to_clear:
        del os.environ[var]
    
    yield
    
    # Clear any vars set during the test
    env_vars_after = [k for k in os.environ if k.startswith("GONKA_")]
    for var in env_vars_after:
        del os.environ[var]
    
    # Restore original env vars
    for k, v in original_env.items():
        os.environ[k] = v


# ==============================================================================
# Private Key Fixtures
# ==============================================================================


@pytest.fixture
def test_private_key() -> str:
    """Valid test private key with 0x prefix."""
    return TEST_PRIVATE_KEY


@pytest.fixture
def test_private_key_no_prefix() -> str:
    """Valid test private key without 0x prefix."""
    return TEST_PRIVATE_KEY_NO_PREFIX


@pytest.fixture
def alternative_private_key() -> str:
    """Alternative valid test private key for comparison."""
    return TEST_PRIVATE_KEY_B


# ==============================================================================
# GonkaParticipant Fixtures
# ==============================================================================


@pytest.fixture
def test_participant() -> GonkaParticipant:
    """Single valid participant for basic tests."""
    return GonkaParticipant(
        address=TEST_ADDRESS_1,
        inference_url=TEST_NODE_URL_1,
        weight=100,
        models=["Qwen/QwQ-32B", "meta-llama/Llama-3.1-70B"],
        validator_key="validator_pubkey_1",
        epoch_id=42,
    )


@pytest.fixture
def test_participant_2() -> GonkaParticipant:
    """Second participant for multi-participant tests."""
    return GonkaParticipant(
        address=TEST_ADDRESS_2,
        inference_url=TEST_NODE_URL_2,
        weight=200,
        models=["gpt-4"],
        validator_key="validator_pubkey_2",
        epoch_id=42,
    )


@pytest.fixture
def test_participant_3() -> GonkaParticipant:
    """Third participant for multi-participant tests."""
    return GonkaParticipant(
        address=TEST_ADDRESS_3,
        inference_url=TEST_NODE_URL_3,
        weight=50,
        models=[],
        validator_key="validator_pubkey_3",
        epoch_id=42,
    )


@pytest.fixture
def test_participants(
    test_participant: GonkaParticipant,
    test_participant_2: GonkaParticipant,
    test_participant_3: GonkaParticipant,
) -> List[GonkaParticipant]:
    """List of three test participants."""
    return [test_participant, test_participant_2, test_participant_3]


@pytest.fixture
def epoch_41_participants() -> List[GonkaParticipant]:
    """Participants from epoch 41 for transition testing."""
    return [
        GonkaParticipant(
            address="gonka1old1",
            inference_url="https://old1.example.com",
            weight=100,
            epoch_id=41,
        ),
        GonkaParticipant(
            address="gonka1old2",
            inference_url="https://old2.example.com",
            weight=150,
            epoch_id=41,
        ),
    ]


@pytest.fixture
def epoch_42_participants() -> List[GonkaParticipant]:
    """Participants from epoch 42 for transition testing."""
    return [
        GonkaParticipant(
            address="gonka1new1",
            inference_url="https://new1.example.com",
            weight=200,
            epoch_id=42,
        ),
        GonkaParticipant(
            address="gonka1new2",
            inference_url="https://new2.example.com",
            weight=100,
            epoch_id=42,
        ),
        GonkaParticipant(
            address="gonka1new3",
            inference_url="https://new3.example.com",
            weight=50,
            epoch_id=42,
        ),
    ]


# ==============================================================================
# JSON Fixture Loaders
# ==============================================================================


@pytest.fixture
def valid_participants_json() -> Dict[str, Any]:
    """Load valid participants fixture from file."""
    with open(FIXTURES_DIR / "gonka_valid_participants.json") as f:
        return json.load(f)


@pytest.fixture
def empty_participants_json() -> Dict[str, Any]:
    """Load empty participants fixture from file."""
    with open(FIXTURES_DIR / "gonka_empty_participants.json") as f:
        return json.load(f)


@pytest.fixture
def invalid_participants_json() -> Dict[str, Any]:
    """Load invalid participants fixture from file."""
    with open(FIXTURES_DIR / "gonka_invalid_participants.json") as f:
        return json.load(f)


# ==============================================================================
# Mock Component Fixtures
# ==============================================================================


@pytest.fixture
def mock_client_factory():
    """
    Mock GonkaClientFactory that returns mock clients.
    
    The factory tracks all create_client calls and returns new mock clients.
    """
    factory = Mock()
    factory.requester_address = "gonka1requester123"
    factory.create_client.return_value = Mock()
    factory.create_async_client.return_value = Mock()
    return factory


@pytest.fixture
def mock_discovery(test_participants: List[GonkaParticipant]):
    """
    Mock EndpointDiscovery that returns test participants.
    
    Configured for synchronous and async discovery with epoch 42.
    """
    discovery = Mock()
    discovery.source_url = TEST_SOURCE_URL
    discovery.verify_proofs = False
    discovery.timeout = 30.0
    
    discovery.discover.return_value = test_participants
    discovery.discover_for_epoch.return_value = test_participants
    discovery.get_current_epoch.return_value = 42
    
    # Async methods
    discovery.async_discover = AsyncMock(return_value=test_participants)
    discovery.async_discover_for_epoch = AsyncMock(return_value=test_participants)
    
    return discovery


@pytest.fixture
def mock_registrar():
    """
    Mock ModelRegistrar that tracks registrations.
    
    Returns success for all registrations and tracks model names.
    """
    registrar = Mock()
    registrar.model_name_prefix = "gonka:"
    registrar.default_max_concurrent = None
    
    # Track registered models
    registrar._registered = set()
    
    def register_one_side_effect(mux, participant, **kwargs):
        model_name = f"gonka:{participant.address}"
        if model_name in registrar._registered:
            return False
        registrar._registered.add(model_name)
        return True
    
    def register_all_side_effect(mux, participants, **kwargs):
        count = 0
        for p in participants:
            if register_one_side_effect(mux, p, **kwargs):
                count += 1
        return count
    
    registrar.register_one.side_effect = register_one_side_effect
    registrar.register_all.side_effect = register_all_side_effect
    registrar.unregister_all.return_value = 0
    registrar.get_registered_models.return_value = list(registrar._registered)
    
    return registrar


@pytest.fixture
def mock_multiplexer():
    """
    Mock Multiplexer that tracks model additions.
    
    Tracks both primary and fallback model additions for verification.
    """
    mux = MagicMock()
    mux._weighted_models = []
    mux._fallback_models = []
    
    # Track additions
    mux._added_models = []
    mux._added_fallbacks = []
    
    def add_model_side_effect(*args, **kwargs):
        mux._added_models.append(kwargs)
        # Create mock weighted model
        wm = Mock()
        wm.model_name = kwargs.get("model_name", "unknown")
        wm.weight = kwargs.get("weight", 1)
        wm.disabled_until = None
        mux._weighted_models.append(wm)
    
    def add_fallback_side_effect(*args, **kwargs):
        mux._added_fallbacks.append(kwargs)
        # Create mock weighted model
        wm = Mock()
        wm.model_name = kwargs.get("model_name", "unknown")
        wm.weight = kwargs.get("weight", 1)
        wm.disabled_until = None
        mux._fallback_models.append(wm)
    
    mux.add_model.side_effect = add_model_side_effect
    mux.add_fallback_model.side_effect = add_fallback_side_effect
    
    return mux


@pytest.fixture
def mock_refresh_manager():
    """Mock RefreshManager for testing orchestration."""
    manager = Mock()
    manager.is_running = False
    manager.current_epoch = None
    manager.last_refresh = None
    manager.on_epoch_change = None
    
    manager.start.return_value = None
    manager.stop.return_value = None
    manager.refresh_now.return_value = RefreshResult(success=True, epoch_changed=False)
    manager.initial_registration.return_value = RefreshResult(
        success=True, epoch_changed=True, new_epoch=42, participants_added=3
    )
    
    manager.async_stop = AsyncMock()
    manager.async_refresh_now = AsyncMock(
        return_value=RefreshResult(success=True, epoch_changed=False)
    )
    manager.async_initial_registration = AsyncMock(
        return_value=RefreshResult(
            success=True, epoch_changed=True, new_epoch=42, participants_added=3
        )
    )
    
    return manager


# ==============================================================================
# File-based Discovery Fixtures
# ==============================================================================


@pytest.fixture
def temp_participants_file(tmp_path: Path, valid_participants_json: Dict[str, Any]) -> Path:
    """
    Create a temporary JSON file with valid participants.
    
    Returns the path to the file for use with file:// URLs.
    """
    filepath = tmp_path / "participants.json"
    filepath.write_text(json.dumps(valid_participants_json))
    return filepath


@pytest.fixture
def file_discovery_url(temp_participants_file: Path) -> str:
    """File URL for testing local file discovery."""
    return f"file://{temp_participants_file}"


# ==============================================================================
# HTTP Mocking Helpers
# ==============================================================================


@pytest.fixture
def mock_http_response():
    """
    Factory fixture for creating mock HTTP responses.
    
    Usage:
        response = mock_http_response(status_code=200, json_data={...})
    """
    def _create_response(
        status_code: int = 200,
        json_data: Dict[str, Any] = None,
        raise_json_error: bool = False,
    ):
        response = Mock()
        response.status_code = status_code
        
        if raise_json_error:
            response.json.side_effect = json.JSONDecodeError("error", "", 0)
        elif json_data is not None:
            response.json.return_value = json_data
        else:
            response.json.return_value = {}
        
        return response
    
    return _create_response


# ==============================================================================
# Error Simulation Fixtures
# ==============================================================================


@pytest.fixture
def discovery_error():
    """GonkaDiscoveryError for testing error handling."""
    return GonkaDiscoveryError(
        "Failed to fetch participants",
        source_url=TEST_SOURCE_URL,
        status_code=500,
    )


@pytest.fixture
def no_participants_error():
    """GonkaNoParticipantsError for testing empty response handling."""
    return GonkaNoParticipantsError(
        "No participants found",
        epoch="current",
        source_url=TEST_SOURCE_URL,
    )


@pytest.fixture
def client_error():
    """GonkaClientError for testing client creation failures."""
    return GonkaClientError(
        "Failed to create client",
        participant_address=TEST_ADDRESS_1,
    )


# ==============================================================================
# Configuration Fixtures
# ==============================================================================


@pytest.fixture
def valid_config_kwargs() -> Dict[str, Any]:
    """Keyword arguments for creating a valid GonkaConfig."""
    return {
        "private_key": TEST_PRIVATE_KEY,
        "source_url": TEST_SOURCE_URL,
        "verify_proofs": False,
        "refresh_enabled": True,
        "refresh_interval_seconds": 60.0,
        "model_name_prefix": "gonka:",
    }


@pytest.fixture
def valid_endpoint_string() -> str:
    """Valid endpoint string in URL;address format."""
    return f"{TEST_NODE_URL_1};{TEST_ADDRESS_1}"


@pytest.fixture
def valid_endpoint_strings() -> List[str]:
    """List of valid endpoint strings."""
    return [
        f"{TEST_NODE_URL_1};{TEST_ADDRESS_1}",
        f"{TEST_NODE_URL_2};{TEST_ADDRESS_2}",
        f"{TEST_NODE_URL_3};{TEST_ADDRESS_3}",
    ]

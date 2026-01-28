"""
Unit and integration tests for the EndpointDiscovery class.

Tests cover:
- Initialization validation
- Participant parsing with valid/invalid data
- Epoch string parsing
- URL normalization
- HTTP error handling
- File:// URL support
- Empty participant list detection
- Async discovery methods
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from multiplexer_llm.gonka import (
    EndpointDiscovery,
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
    GonkaParticipant,
    GonkaProofVerificationError,
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def valid_participants_json() -> Dict[str, Any]:
    """Load valid participants fixture."""
    with open(FIXTURES_DIR / "gonka_valid_participants.json") as f:
        return json.load(f)


@pytest.fixture
def empty_participants_json() -> Dict[str, Any]:
    """Load empty participants fixture."""
    with open(FIXTURES_DIR / "gonka_empty_participants.json") as f:
        return json.load(f)


@pytest.fixture
def invalid_participants_json() -> Dict[str, Any]:
    """Load invalid participants fixture."""
    with open(FIXTURES_DIR / "gonka_invalid_participants.json") as f:
        return json.load(f)


@pytest.fixture
def discovery() -> EndpointDiscovery:
    """Create a discovery instance with a mock source URL."""
    return EndpointDiscovery(
        source_url="https://api.gonka.network",
        verify_proofs=False,
        timeout=10.0,
        retry_count=1,
        retry_delay=0.1,
    )


@pytest.fixture
def file_discovery(tmp_path: Path, valid_participants_json: Dict[str, Any]) -> EndpointDiscovery:
    """Create a discovery instance using a file:// URL."""
    fixture_file = tmp_path / "participants.json"
    fixture_file.write_text(json.dumps(valid_participants_json))
    return EndpointDiscovery(
        source_url=f"file://{fixture_file}",
        verify_proofs=False,
    )


# ==============================================================================
# Initialization Tests
# ==============================================================================


class TestEndpointDiscoveryInit:
    """Tests for EndpointDiscovery initialization."""

    def test_init_with_valid_url(self) -> None:
        """Test initialization with a valid source URL."""
        discovery = EndpointDiscovery(source_url="https://api.gonka.network")
        assert discovery.source_url == "https://api.gonka.network"
        assert discovery.verify_proofs is False
        assert discovery.timeout == 30.0

    def test_init_with_trailing_slash(self) -> None:
        """Test that trailing slashes are normalized."""
        discovery = EndpointDiscovery(source_url="https://api.gonka.network/")
        assert discovery.source_url == "https://api.gonka.network"

    def test_init_with_multiple_trailing_slashes(self) -> None:
        """Test that multiple trailing slashes are normalized."""
        discovery = EndpointDiscovery(source_url="https://api.gonka.network///")
        assert discovery.source_url == "https://api.gonka.network"

    def test_init_with_empty_url_raises(self) -> None:
        """Test that empty source URL raises ValueError."""
        with pytest.raises(ValueError, match="source_url cannot be empty"):
            EndpointDiscovery(source_url="")

    def test_init_with_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        discovery = EndpointDiscovery(
            source_url="https://api.gonka.network",
            timeout=60.0,
        )
        assert discovery.timeout == 60.0

    def test_init_with_verify_proofs(self) -> None:
        """Test initialization with proof verification enabled."""
        discovery = EndpointDiscovery(
            source_url="https://api.gonka.network",
            verify_proofs=True,
        )
        assert discovery.verify_proofs is True

    def test_init_with_file_url(self) -> None:
        """Test initialization with file:// URL."""
        discovery = EndpointDiscovery(source_url="file:///path/to/file.json")
        assert discovery.source_url == "file:///path/to/file.json"


# ==============================================================================
# URL Building Tests
# ==============================================================================


class TestUrlBuilding:
    """Tests for URL construction."""

    def test_build_url_current_epoch(self, discovery: EndpointDiscovery) -> None:
        """Test URL building for current epoch."""
        url = discovery._build_url("current")
        assert url == "https://api.gonka.network/v1/epochs/current/participants"

    def test_build_url_next_epoch(self, discovery: EndpointDiscovery) -> None:
        """Test URL building for next epoch."""
        url = discovery._build_url("next")
        assert url == "https://api.gonka.network/v1/epochs/next/participants"

    def test_build_url_numeric_epoch(self, discovery: EndpointDiscovery) -> None:
        """Test URL building for numeric epoch."""
        url = discovery._build_url("42")
        assert url == "https://api.gonka.network/v1/epochs/42/participants"


# ==============================================================================
# Epoch Normalization Tests
# ==============================================================================


class TestEpochNormalization:
    """Tests for epoch string normalization."""

    def test_normalize_current(self) -> None:
        """Test normalizing 'current' epoch."""
        result = EndpointDiscovery._normalize_epoch("current")
        assert result == "current"

    def test_normalize_next(self) -> None:
        """Test normalizing 'next' epoch."""
        result = EndpointDiscovery._normalize_epoch("next")
        assert result == "next"

    def test_normalize_numeric(self) -> None:
        """Test normalizing numeric epoch."""
        result = EndpointDiscovery._normalize_epoch("42")
        assert result == "42"

    def test_normalize_with_whitespace(self) -> None:
        """Test normalizing epoch with whitespace."""
        result = EndpointDiscovery._normalize_epoch("  current  ")
        assert result == "current"

    def test_normalize_empty_raises(self) -> None:
        """Test that empty epoch raises ValueError."""
        with pytest.raises(ValueError, match="epoch cannot be empty"):
            EndpointDiscovery._normalize_epoch("")


# ==============================================================================
# Inference URL Normalization Tests
# ==============================================================================


class TestInferenceUrlNormalization:
    """Tests for inference URL normalization."""

    def test_normalize_simple_url(self) -> None:
        """Test normalizing simple URL."""
        result = EndpointDiscovery._normalize_inference_url("https://node.example.com")
        assert result == "https://node.example.com"

    def test_normalize_url_with_trailing_slash(self) -> None:
        """Test normalizing URL with trailing slash."""
        result = EndpointDiscovery._normalize_inference_url("https://node.example.com/")
        assert result == "https://node.example.com"

    def test_normalize_url_with_v1_suffix(self) -> None:
        """Test normalizing URL with /v1 suffix."""
        result = EndpointDiscovery._normalize_inference_url("https://node.example.com/v1")
        assert result == "https://node.example.com"

    def test_normalize_url_with_v1_trailing_slash(self) -> None:
        """Test normalizing URL with /v1/ suffix."""
        result = EndpointDiscovery._normalize_inference_url("https://node.example.com/v1/")
        assert result == "https://node.example.com"

    def test_normalize_empty_url(self) -> None:
        """Test normalizing empty URL returns empty."""
        result = EndpointDiscovery._normalize_inference_url("")
        assert result == ""


# ==============================================================================
# Participant Parsing Tests
# ==============================================================================


class TestParticipantParsing:
    """Tests for parsing participant data."""

    def test_parse_valid_participant(self, discovery: EndpointDiscovery) -> None:
        """Test parsing a valid participant."""
        data = {
            "index": "gonka1abc123",
            "validator_key": "pubkey123",
            "weight": 100,
            "inference_url": "https://node.example.com",
            "models": ["model1", "model2"],
        }
        result = discovery._parse_participant(data, epoch_id=42)
        
        assert result is not None
        assert result.address == "gonka1abc123"
        assert result.validator_key == "pubkey123"
        assert result.weight == 100
        assert result.inference_url == "https://node.example.com"
        assert result.models == ["model1", "model2"]
        assert result.epoch_id == 42

    def test_parse_participant_missing_index(self, discovery: EndpointDiscovery) -> None:
        """Test parsing participant with missing index returns None."""
        data = {
            "validator_key": "pubkey123",
            "weight": 100,
            "inference_url": "https://node.example.com",
        }
        result = discovery._parse_participant(data, epoch_id=42)
        assert result is None

    def test_parse_participant_missing_url(self, discovery: EndpointDiscovery) -> None:
        """Test parsing participant with missing URL returns None."""
        data = {
            "index": "gonka1abc123",
            "validator_key": "pubkey123",
            "weight": 100,
        }
        result = discovery._parse_participant(data, epoch_id=42)
        assert result is None

    def test_parse_participant_invalid_weight(self, discovery: EndpointDiscovery) -> None:
        """Test parsing participant with invalid weight uses default."""
        data = {
            "index": "gonka1abc123",
            "inference_url": "https://node.example.com",
            "weight": -10,
        }
        result = discovery._parse_participant(data, epoch_id=42)
        assert result is not None
        assert result.weight == 1

    def test_parse_participant_null_weight(self, discovery: EndpointDiscovery) -> None:
        """Test parsing participant with null weight uses default."""
        data = {
            "index": "gonka1abc123",
            "inference_url": "https://node.example.com",
            "weight": None,
        }
        result = discovery._parse_participant(data, epoch_id=42)
        assert result is not None
        assert result.weight == 1

    def test_parse_participant_empty_models(self, discovery: EndpointDiscovery) -> None:
        """Test parsing participant with empty models list."""
        data = {
            "index": "gonka1abc123",
            "inference_url": "https://node.example.com",
            "models": [],
        }
        result = discovery._parse_participant(data, epoch_id=42)
        assert result is not None
        assert result.models == []

    def test_parse_participant_null_models(self, discovery: EndpointDiscovery) -> None:
        """Test parsing participant with null models."""
        data = {
            "index": "gonka1abc123",
            "inference_url": "https://node.example.com",
            "models": None,
        }
        result = discovery._parse_participant(data, epoch_id=42)
        assert result is not None
        assert result.models == []

    def test_parse_participant_empty_data(self, discovery: EndpointDiscovery) -> None:
        """Test parsing empty data returns None."""
        result = discovery._parse_participant({}, epoch_id=42)
        assert result is None

    def test_parse_participant_none_data(self, discovery: EndpointDiscovery) -> None:
        """Test parsing None data returns None."""
        result = discovery._parse_participant(None, epoch_id=42)  # type: ignore
        assert result is None


# ==============================================================================
# File Discovery Tests
# ==============================================================================


class TestFileDiscovery:
    """Tests for file:// URL discovery."""

    def test_discover_from_file(self, file_discovery: EndpointDiscovery) -> None:
        """Test discovering participants from a local file."""
        participants = file_discovery.discover()
        
        assert len(participants) == 3
        assert participants[0].address == "gonka1abc123def456"
        assert participants[0].weight == 100
        assert participants[0].epoch_id == 42

    def test_discover_file_not_found(self, tmp_path: Path) -> None:
        """Test discovering from nonexistent file raises error."""
        discovery = EndpointDiscovery(source_url=f"file://{tmp_path}/nonexistent.json")
        
        with pytest.raises(GonkaDiscoveryError) as exc_info:
            discovery.discover()
        
        assert "File not found" in str(exc_info.value)

    def test_discover_invalid_json_file(self, tmp_path: Path) -> None:
        """Test discovering from file with invalid JSON raises error."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalid json }")
        
        discovery = EndpointDiscovery(source_url=f"file://{bad_file}")
        
        with pytest.raises(GonkaDiscoveryError) as exc_info:
            discovery.discover()
        
        assert "Invalid JSON" in str(exc_info.value)


# ==============================================================================
# HTTP Discovery Tests
# ==============================================================================


class TestHttpDiscovery:
    """Tests for HTTP URL discovery with mocked requests."""

    def test_discover_success(
        self,
        discovery: EndpointDiscovery,
        valid_participants_json: Dict[str, Any],
    ) -> None:
        """Test successful HTTP discovery."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = valid_participants_json
        
        with patch("requests.get", return_value=mock_response) as mock_get:
            participants = discovery.discover()
        
        assert len(participants) == 3
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "current/participants" in args[0]

    def test_discover_for_specific_epoch(
        self,
        discovery: EndpointDiscovery,
        valid_participants_json: Dict[str, Any],
    ) -> None:
        """Test discovery for a specific epoch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = valid_participants_json
        
        with patch("requests.get", return_value=mock_response) as mock_get:
            participants = discovery.discover_for_epoch("42")
        
        assert len(participants) == 3
        args, _ = mock_get.call_args
        assert "/42/participants" in args[0]

    def test_discover_http_404(self, discovery: EndpointDiscovery) -> None:
        """Test HTTP 404 raises GonkaDiscoveryError."""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaDiscoveryError) as exc_info:
                discovery.discover()
        
        assert exc_info.value.status_code == 404

    def test_discover_http_500(self, discovery: EndpointDiscovery) -> None:
        """Test HTTP 500 raises GonkaDiscoveryError."""
        mock_response = Mock()
        mock_response.status_code = 500
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaDiscoveryError) as exc_info:
                discovery.discover()
        
        assert exc_info.value.status_code == 500

    def test_discover_timeout(self, discovery: EndpointDiscovery) -> None:
        """Test request timeout raises GonkaDiscoveryError."""
        with patch("requests.get", side_effect=requests.Timeout("timed out")):
            with pytest.raises(GonkaDiscoveryError) as exc_info:
                discovery.discover()
        
        assert "Failed to fetch" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, requests.Timeout)

    def test_discover_connection_error(self, discovery: EndpointDiscovery) -> None:
        """Test connection error raises GonkaDiscoveryError."""
        with patch("requests.get", side_effect=requests.ConnectionError("connection failed")):
            with pytest.raises(GonkaDiscoveryError) as exc_info:
                discovery.discover()
        
        assert "Failed to fetch" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, requests.ConnectionError)

    def test_discover_invalid_json_response(self, discovery: EndpointDiscovery) -> None:
        """Test invalid JSON response raises GonkaDiscoveryError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("error", "", 0)
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaDiscoveryError) as exc_info:
                discovery.discover()
        
        assert "Invalid JSON" in str(exc_info.value)


# ==============================================================================
# Empty Participants Tests
# ==============================================================================


class TestEmptyParticipants:
    """Tests for empty participant list handling."""

    def test_discover_empty_raises_error(
        self,
        discovery: EndpointDiscovery,
        empty_participants_json: Dict[str, Any],
    ) -> None:
        """Test that empty participant list raises GonkaNoParticipantsError."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = empty_participants_json
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaNoParticipantsError) as exc_info:
                discovery.discover()
        
        assert "No participants" in str(exc_info.value)
        assert exc_info.value.epoch == "current"

    def test_discover_missing_active_participants(self, discovery: EndpointDiscovery) -> None:
        """Test response without active_participants raises error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"block": {}}
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaDiscoveryError) as exc_info:
                discovery.discover()
        
        assert "missing active_participants" in str(exc_info.value)


# ==============================================================================
# Invalid Participants Tests
# ==============================================================================


class TestInvalidParticipants:
    """Tests for handling invalid participant data."""

    def test_discover_with_some_invalid(
        self,
        discovery: EndpointDiscovery,
        invalid_participants_json: Dict[str, Any],
    ) -> None:
        """Test that valid participants are returned even when some are invalid."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = invalid_participants_json
        
        with patch("requests.get", return_value=mock_response):
            participants = discovery.discover()
        
        # Only valid participant (gonka1valid123abc) + two with weight issues (using default)
        assert len(participants) == 3
        
        # Check first participant is the valid one
        valid_p = next(p for p in participants if p.address == "gonka1valid123abc")
        assert valid_p.weight == 100

    def test_discover_all_invalid_raises_error(self, discovery: EndpointDiscovery) -> None:
        """Test that all invalid participants raises GonkaNoParticipantsError."""
        payload = {
            "active_participants": {
                "participants": [
                    {"validator_key": "pubkey1"},  # Missing index
                    {"index": "gonka1test"},  # Missing URL
                ],
                "epoch_id": 42,
            }
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = payload
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaNoParticipantsError) as exc_info:
                discovery.discover()
        
        assert "invalid" in str(exc_info.value).lower()


# ==============================================================================
# Get Current Epoch Tests
# ==============================================================================


class TestGetCurrentEpoch:
    """Tests for get_current_epoch method."""

    def test_get_current_epoch_success(
        self,
        discovery: EndpointDiscovery,
        valid_participants_json: Dict[str, Any],
    ) -> None:
        """Test getting current epoch ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = valid_participants_json
        
        with patch("requests.get", return_value=mock_response):
            epoch = discovery.get_current_epoch()
        
        assert epoch == 42

    def test_get_current_epoch_missing_raises(self, discovery: EndpointDiscovery) -> None:
        """Test missing epoch_id raises error."""
        payload = {
            "active_participants": {
                "participants": [],
            }
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = payload
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaDiscoveryError) as exc_info:
                discovery.get_current_epoch()
        
        assert "missing epoch_id" in str(exc_info.value)


# ==============================================================================
# Async Discovery Tests
# ==============================================================================


class TestAsyncDiscovery:
    """Tests for async discovery methods."""

    @pytest.mark.asyncio
    async def test_async_discover_from_file(
        self,
        file_discovery: EndpointDiscovery,
    ) -> None:
        """Test async discovery from file."""
        participants = await file_discovery.async_discover()
        
        assert len(participants) == 3
        assert participants[0].address == "gonka1abc123def456"

    @pytest.mark.asyncio
    async def test_async_discover_for_epoch(
        self,
        file_discovery: EndpointDiscovery,
    ) -> None:
        """Test async discovery for specific epoch."""
        participants = await file_discovery.async_discover_for_epoch("current")
        
        assert len(participants) == 3


# ==============================================================================
# Retry Behavior Tests
# ==============================================================================


class TestRetryBehavior:
    """Tests for retry behavior on transient failures."""

    def test_retry_on_timeout(self, valid_participants_json: Dict[str, Any]) -> None:
        """Test that timeouts are retried."""
        discovery = EndpointDiscovery(
            source_url="https://api.gonka.network",
            retry_count=3,
            retry_delay=0.01,
        )
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = valid_participants_json
        
        # First two calls timeout, third succeeds
        with patch("requests.get", side_effect=[
            requests.Timeout(),
            requests.Timeout(),
            mock_response,
        ]) as mock_get:
            participants = discovery.discover()
        
        assert len(participants) == 3
        assert mock_get.call_count == 3

    def test_retry_on_connection_error(self, valid_participants_json: Dict[str, Any]) -> None:
        """Test that connection errors are retried."""
        discovery = EndpointDiscovery(
            source_url="https://api.gonka.network",
            retry_count=2,
            retry_delay=0.01,
        )
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = valid_participants_json
        
        # First call fails, second succeeds
        with patch("requests.get", side_effect=[
            requests.ConnectionError(),
            mock_response,
        ]) as mock_get:
            participants = discovery.discover()
        
        assert len(participants) == 3
        assert mock_get.call_count == 2

    def test_no_retry_on_http_error(self, discovery: EndpointDiscovery) -> None:
        """Test that HTTP errors (4xx, 5xx) are not retried."""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch("requests.get", return_value=mock_response) as mock_get:
            with pytest.raises(GonkaDiscoveryError):
                discovery.discover()
        
        # Should only be called once (no retry)
        assert mock_get.call_count == 1


# ==============================================================================
# Proof Verification Tests (Basic)
# ==============================================================================


class TestProofVerification:
    """Basic tests for proof verification."""

    def test_proof_verification_disabled_by_default(
        self,
        discovery: EndpointDiscovery,
        valid_participants_json: Dict[str, Any],
    ) -> None:
        """Test that proof verification is disabled by default."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = valid_participants_json
        
        with patch("requests.get", return_value=mock_response):
            # Should not raise even without proof data
            participants = discovery.discover()
        
        assert len(participants) == 3

    def test_proof_verification_missing_bytes_raises(self) -> None:
        """Test that missing active_participants_bytes raises error when verification enabled."""
        discovery = EndpointDiscovery(
            source_url="https://api.gonka.network",
            verify_proofs=True,
            retry_count=1,
        )
        
        payload = {
            "active_participants": {
                "participants": [
                    {"index": "gonka1test", "inference_url": "https://test.com"}
                ],
                "epoch_id": 42,
            },
            # Missing active_participants_bytes
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = payload
        
        with patch("requests.get", return_value=mock_response):
            with pytest.raises(GonkaProofVerificationError) as exc_info:
                discovery.discover()
        
        assert "active_participants_bytes" in str(exc_info.value)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_discovery_flow(
        self,
        tmp_path: Path,
        valid_participants_json: Dict[str, Any],
    ) -> None:
        """Test full discovery flow with file URL."""
        # Write fixture to temp file
        fixture_file = tmp_path / "participants.json"
        fixture_file.write_text(json.dumps(valid_participants_json))
        
        # Create discovery and fetch
        discovery = EndpointDiscovery(source_url=f"file://{fixture_file}")
        participants = discovery.discover()
        
        # Verify results
        assert len(participants) == 3
        
        # Check first participant
        p1 = participants[0]
        assert p1.address == "gonka1abc123def456"
        assert p1.weight == 100
        assert p1.models == ["Qwen/QwQ-32B", "meta-llama/Llama-3.1-70B"]
        assert p1.epoch_id == 42
        
        # Verify base_url normalization through GonkaParticipant
        assert p1.base_url == "https://node1.example.com/v1"
        
        # Check second participant (already had /v1 suffix)
        p2 = participants[1]
        assert p2.inference_url == "https://node2.example.com"  # /v1 stripped
        assert p2.base_url == "https://node2.example.com/v1"  # Added back by property

    def test_discover_then_get_epoch(
        self,
        tmp_path: Path,
        valid_participants_json: Dict[str, Any],
    ) -> None:
        """Test discovering participants then getting epoch."""
        fixture_file = tmp_path / "participants.json"
        fixture_file.write_text(json.dumps(valid_participants_json))
        
        discovery = EndpointDiscovery(source_url=f"file://{fixture_file}")
        
        epoch = discovery.get_current_epoch()
        assert epoch == 42
        
        participants = discovery.discover()
        assert all(p.epoch_id == 42 for p in participants)

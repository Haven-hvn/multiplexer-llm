"""
Unit tests for the GonkaClientFactory.

These tests verify the GonkaClientFactory implementation meets all requirements:
- FR-1: Factory accepts a private key at construction
- FR-2: Each created client is bound to a specific participant's endpoint
- FR-3: Clients automatically sign requests using ECDSA (secp256k1)
- FR-4: Signature includes: sha256(body) + timestamp + transfer_address
- FR-5: Clients add X-Requester-Address header with derived gonka address
- FR-6: Clients add X-Timestamp header with nanosecond timestamp
- FR-7: Base URL correctly formatted (with /v1 suffix if needed)
- FR-8: Support both sync and async client variants
- NFR-1: Private key never logged or exposed in errors
- NFR-2: Client creation is fast (no network calls)
- NFR-3: Thread-safe for concurrent client creation
"""

import pytest
import hashlib
import base64
import threading
import time
from unittest.mock import MagicMock, patch, ANY

# Test fixtures - these are deterministic test values
# Using a well-known test private key (DO NOT use in production!)
TEST_PRIVATE_KEY = "0x" + "a" * 64  # 32 bytes of 0xaa
TEST_PRIVATE_KEY_NO_PREFIX = "a" * 64


class TestGonkaParticipant:
    """Tests for GonkaParticipant data class."""

    def test_participant_creation_with_required_fields(self):
        """Test participant can be created with required fields."""
        from multiplexer_llm.gonka import GonkaParticipant

        participant = GonkaParticipant(
            address="gonka1abc123",
            inference_url="https://node.example.com",
        )

        assert participant.address == "gonka1abc123"
        assert participant.inference_url == "https://node.example.com"
        assert participant.weight == 1  # default
        assert participant.models == []  # default
        assert participant.epoch_id == 0  # default

    def test_participant_creation_with_all_fields(self):
        """Test participant can be created with all fields."""
        from multiplexer_llm.gonka import GonkaParticipant

        participant = GonkaParticipant(
            address="gonka1abc123",
            inference_url="https://node.example.com/v1",
            weight=100,
            models=["llama-3.1-70b", "gpt-4"],
            validator_key="validator123",
            epoch_id=42,
        )

        assert participant.address == "gonka1abc123"
        assert participant.inference_url == "https://node.example.com/v1"
        assert participant.weight == 100
        assert participant.models == ["llama-3.1-70b", "gpt-4"]
        assert participant.validator_key == "validator123"
        assert participant.epoch_id == 42

    def test_participant_rejects_empty_address(self):
        """Test participant raises error for empty address."""
        from multiplexer_llm.gonka import GonkaParticipant

        with pytest.raises(ValueError, match="address cannot be empty"):
            GonkaParticipant(
                address="",
                inference_url="https://node.example.com",
            )

    def test_participant_rejects_empty_inference_url(self):
        """Test participant raises error for empty inference_url."""
        from multiplexer_llm.gonka import GonkaParticipant

        with pytest.raises(ValueError, match="inference_url cannot be empty"):
            GonkaParticipant(
                address="gonka1abc123",
                inference_url="",
            )

    def test_participant_rejects_non_positive_weight(self):
        """Test participant raises error for non-positive weight."""
        from multiplexer_llm.gonka import GonkaParticipant

        with pytest.raises(ValueError, match="weight must be positive"):
            GonkaParticipant(
                address="gonka1abc123",
                inference_url="https://node.example.com",
                weight=0,
            )

        with pytest.raises(ValueError, match="weight must be positive"):
            GonkaParticipant(
                address="gonka1abc123",
                inference_url="https://node.example.com",
                weight=-1,
            )

    def test_participant_base_url_adds_v1_suffix(self):
        """Test base_url property adds /v1 suffix if not present."""
        from multiplexer_llm.gonka import GonkaParticipant

        # Without /v1
        participant = GonkaParticipant(
            address="gonka1abc123",
            inference_url="https://node.example.com",
        )
        assert participant.base_url == "https://node.example.com/v1"

        # With trailing slash
        participant = GonkaParticipant(
            address="gonka1abc123",
            inference_url="https://node.example.com/",
        )
        assert participant.base_url == "https://node.example.com/v1"

    def test_participant_base_url_preserves_v1_suffix(self):
        """Test base_url property preserves existing /v1 suffix."""
        from multiplexer_llm.gonka import GonkaParticipant

        # Already has /v1
        participant = GonkaParticipant(
            address="gonka1abc123",
            inference_url="https://node.example.com/v1",
        )
        assert participant.base_url == "https://node.example.com/v1"

        # Has /v1 with trailing slash
        participant = GonkaParticipant(
            address="gonka1abc123",
            inference_url="https://node.example.com/v1/",
        )
        assert participant.base_url == "https://node.example.com/v1"

    def test_participant_is_immutable(self):
        """Test participant is frozen (immutable)."""
        from multiplexer_llm.gonka import GonkaParticipant

        participant = GonkaParticipant(
            address="gonka1abc123",
            inference_url="https://node.example.com",
        )

        with pytest.raises(AttributeError):
            participant.address = "new_address"


class TestGonkaExceptions:
    """Tests for Gonka exception classes."""

    def test_gonka_error_base_class(self):
        """Test GonkaError is properly structured."""
        from multiplexer_llm.gonka import GonkaError
        from multiplexer_llm.exceptions import MultiplexerError

        error = GonkaError("test message")
        assert str(error) == "test message"
        assert isinstance(error, MultiplexerError)

    def test_gonka_client_error_without_details(self):
        """Test GonkaClientError with just message."""
        from multiplexer_llm.gonka import GonkaClientError

        error = GonkaClientError("test message")
        assert "test message" in str(error)
        assert error.participant_address is None
        assert error.cause is None

    def test_gonka_client_error_with_participant(self):
        """Test GonkaClientError includes participant in string."""
        from multiplexer_llm.gonka import GonkaClientError

        error = GonkaClientError(
            "test message",
            participant_address="gonka1abc123",
        )
        assert "test message" in str(error)
        assert "gonka1abc123" in str(error)

    def test_gonka_client_error_with_cause(self):
        """Test GonkaClientError includes cause in string."""
        from multiplexer_llm.gonka import GonkaClientError

        cause = ValueError("underlying error")
        error = GonkaClientError("test message", cause=cause)
        assert "test message" in str(error)
        assert "ValueError" in str(error)
        assert "underlying error" in str(error)

    def test_gonka_client_error_never_includes_private_key(self):
        """Test GonkaClientError message doesn't contain private key patterns."""
        from multiplexer_llm.gonka import GonkaClientError

        # Create error with various inputs
        error = GonkaClientError(
            "Failed to sign",
            participant_address="gonka1abc123",
            cause=ValueError("bad key"),
        )

        error_str = str(error)
        # Should not contain hex patterns that look like private keys
        assert "0x" + "a" * 64 not in error_str.lower()
        assert "a" * 64 not in error_str.lower()


class TestGonkaClientFactoryConstruction:
    """Tests for GonkaClientFactory construction."""

    def test_factory_accepts_private_key_with_0x_prefix(self):
        """FR-1: Factory accepts a private key at construction (with 0x prefix)."""
        from multiplexer_llm.gonka import GonkaClientFactory

        factory = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)
        assert factory.requester_address is not None
        assert factory.requester_address.startswith("gonka1")

    def test_factory_accepts_private_key_without_prefix(self):
        """FR-1: Factory accepts a private key at construction (without prefix)."""
        from multiplexer_llm.gonka import GonkaClientFactory

        factory = GonkaClientFactory(private_key=TEST_PRIVATE_KEY_NO_PREFIX)
        assert factory.requester_address is not None
        assert factory.requester_address.startswith("gonka1")

    def test_factory_rejects_empty_private_key(self):
        """Factory rejects empty private key with clear error."""
        from multiplexer_llm.gonka import GonkaClientFactory, GonkaClientError

        with pytest.raises(GonkaClientError, match="required"):
            GonkaClientFactory(private_key="")

    def test_factory_rejects_invalid_hex(self):
        """Factory rejects non-hex private key."""
        from multiplexer_llm.gonka import GonkaClientFactory, GonkaClientError

        # Test with correct length but invalid hex
        with pytest.raises(GonkaClientError, match="hexadecimal"):
            GonkaClientFactory(private_key="0x" + "zz" * 32)
        
        # Test with wrong length (caught first)
        with pytest.raises(GonkaClientError, match="64 hex characters"):
            GonkaClientFactory(private_key="not_hex_at_all_xyz")

    def test_factory_rejects_wrong_length_key(self):
        """Factory rejects private key with wrong length."""
        from multiplexer_llm.gonka import GonkaClientFactory, GonkaClientError

        # Too short
        with pytest.raises(GonkaClientError, match="64 hex characters"):
            GonkaClientFactory(private_key="0x" + "a" * 32)

        # Too long
        with pytest.raises(GonkaClientError, match="64 hex characters"):
            GonkaClientFactory(private_key="0x" + "a" * 128)

    def test_factory_error_never_contains_private_key(self):
        """NFR-1: Private key never logged or exposed in errors."""
        from multiplexer_llm.gonka import GonkaClientFactory, GonkaClientError

        # Test with invalid key that could accidentally be echoed
        test_key = "0x" + "deadbeef" * 8

        try:
            # Force an error during address derivation
            factory = GonkaClientFactory(private_key=test_key)
            # If it succeeds, check the address property doesn't leak the key
            assert "deadbeef" not in str(factory.requester_address)
        except GonkaClientError as e:
            # Error message should not contain the key
            assert "deadbeef" not in str(e).lower()

    def test_factory_derives_consistent_address(self):
        """Factory derives same address for same private key."""
        from multiplexer_llm.gonka import GonkaClientFactory

        factory1 = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)
        factory2 = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)

        assert factory1.requester_address == factory2.requester_address

    def test_factory_derives_different_addresses_for_different_keys(self):
        """Factory derives different addresses for different keys."""
        from multiplexer_llm.gonka import GonkaClientFactory

        factory1 = GonkaClientFactory(private_key="0x" + "a" * 64)
        factory2 = GonkaClientFactory(private_key="0x" + "b" * 64)

        assert factory1.requester_address != factory2.requester_address


class TestGonkaClientFactoryCreateClient:
    """Tests for GonkaClientFactory.create_client()."""

    @pytest.fixture
    def factory(self):
        """Create a factory instance for tests."""
        from multiplexer_llm.gonka import GonkaClientFactory

        return GonkaClientFactory(private_key=TEST_PRIVATE_KEY)

    @pytest.fixture
    def participant(self):
        """Create a participant instance for tests."""
        from multiplexer_llm.gonka import GonkaParticipant

        return GonkaParticipant(
            address="gonka1recipient123",
            inference_url="https://node.example.com",
            weight=100,
        )

    def test_create_client_returns_openai_compatible_client(self, factory, participant):
        """FR-2: Each created client is bound to a specific participant's endpoint."""
        client = factory.create_client(participant)

        # Should have chat.completions structure
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")

    def test_create_client_uses_correct_base_url(self, factory, participant):
        """FR-7: Base URL correctly formatted (with /v1 suffix if needed)."""
        client = factory.create_client(participant)

        # The client should have the correct base URL (OpenAI SDK adds trailing slash)
        base_url_str = str(client.base_url).rstrip("/")
        assert base_url_str == "https://node.example.com/v1"

    def test_create_client_preserves_v1_suffix(self, factory):
        """FR-7: Base URL preserves /v1 suffix if already present."""
        from multiplexer_llm.gonka import GonkaParticipant

        participant = GonkaParticipant(
            address="gonka1recipient123",
            inference_url="https://node.example.com/v1",
        )

        client = factory.create_client(participant)
        # Should not have double /v1
        base_url_str = str(client.base_url).rstrip("/")
        assert base_url_str == "https://node.example.com/v1"
        assert "/v1/v1" not in base_url_str

    def test_create_multiple_clients_are_independent(self, factory):
        """Multiple clients from same factory are independent."""
        from multiplexer_llm.gonka import GonkaParticipant

        participant1 = GonkaParticipant(
            address="gonka1recipient1",
            inference_url="https://node1.example.com",
        )
        participant2 = GonkaParticipant(
            address="gonka1recipient2",
            inference_url="https://node2.example.com",
        )

        client1 = factory.create_client(participant1)
        client2 = factory.create_client(participant2)

        # Clients should have different base URLs
        base_url1 = str(client1.base_url)
        base_url2 = str(client2.base_url)
        assert base_url1 != base_url2
        assert "node1" in base_url1
        assert "node2" in base_url2


class TestGonkaClientFactoryCreateAsyncClient:
    """Tests for GonkaClientFactory.create_async_client()."""

    @pytest.fixture
    def factory(self):
        """Create a factory instance for tests."""
        from multiplexer_llm.gonka import GonkaClientFactory

        return GonkaClientFactory(private_key=TEST_PRIVATE_KEY)

    @pytest.fixture
    def participant(self):
        """Create a participant instance for tests."""
        from multiplexer_llm.gonka import GonkaParticipant

        return GonkaParticipant(
            address="gonka1recipient123",
            inference_url="https://node.example.com",
        )

    def test_create_async_client_returns_async_openai_client(
        self, factory, participant
    ):
        """FR-8: Support both sync and async client variants."""
        from openai import AsyncOpenAI

        client = factory.create_async_client(participant)

        # Should be an AsyncOpenAI instance
        assert isinstance(client, AsyncOpenAI)

        # Should have chat.completions structure
        assert hasattr(client, "chat")
        assert hasattr(client.chat, "completions")
        assert hasattr(client.chat.completions, "create")

    def test_create_async_client_uses_correct_base_url(self, factory, participant):
        """FR-7: Base URL correctly formatted for async client."""
        client = factory.create_async_client(participant)
        base_url_str = str(client.base_url).rstrip("/")
        assert base_url_str == "https://node.example.com/v1"


class TestGonkaClientFactoryRequestSigning:
    """Tests for request signing functionality."""

    @pytest.fixture
    def factory(self):
        """Create a factory instance for tests."""
        from multiplexer_llm.gonka import GonkaClientFactory

        return GonkaClientFactory(private_key=TEST_PRIVATE_KEY)

    @pytest.fixture
    def participant(self):
        """Create a participant instance for tests."""
        from multiplexer_llm.gonka import GonkaParticipant

        return GonkaParticipant(
            address="gonka1recipient123",
            inference_url="https://node.example.com",
        )

    def test_http_client_adds_requester_address_header(self, factory, participant):
        """FR-5: Clients add X-Requester-Address header with derived gonka address."""
        import httpx
        from multiplexer_llm.gonka.client_factory import (
            _hybrid_timestamp_ns,
            _create_signature,
        )

        # Get the signing http client
        http_client = factory._create_signing_http_client(
            transfer_address=participant.address,
            is_async=False,
        )

        # Create a test request
        request = httpx.Request("POST", "https://test.com", content=b'{"test": 1}')

        # Apply event hooks
        for hook in http_client.event_hooks["request"]:
            hook(request)

        # Check header is present
        assert "X-Requester-Address" in request.headers
        assert request.headers["X-Requester-Address"] == factory.requester_address
        assert request.headers["X-Requester-Address"].startswith("gonka1")

    def test_http_client_adds_timestamp_header(self, factory, participant):
        """FR-6: Clients add X-Timestamp header with nanosecond timestamp."""
        import httpx

        http_client = factory._create_signing_http_client(
            transfer_address=participant.address,
            is_async=False,
        )

        request = httpx.Request("POST", "https://test.com", content=b'{"test": 1}')

        for hook in http_client.event_hooks["request"]:
            hook(request)

        # Check timestamp header
        assert "X-Timestamp" in request.headers
        timestamp = int(request.headers["X-Timestamp"])
        # Should be nanoseconds (very large number)
        assert timestamp > 1_000_000_000_000_000_000  # > 1 second in ns

    def test_http_client_adds_authorization_signature(self, factory, participant):
        """FR-3: Clients automatically sign requests using ECDSA (secp256k1)."""
        import httpx

        http_client = factory._create_signing_http_client(
            transfer_address=participant.address,
            is_async=False,
        )

        request = httpx.Request("POST", "https://test.com", content=b'{"test": 1}')

        for hook in http_client.event_hooks["request"]:
            hook(request)

        # Check authorization header is present
        assert "Authorization" in request.headers
        # Should be base64 encoded
        signature = request.headers["Authorization"]
        assert len(base64.b64decode(signature)) == 64  # 64 bytes for ECDSA sig

    def test_signature_includes_required_components(self, factory, participant):
        """FR-4: Signature includes: sha256(body) + timestamp + transfer_address."""
        from multiplexer_llm.gonka.client_factory import _create_signature

        body = b'{"test": "data"}'
        timestamp = 1234567890000000000
        transfer_address = "gonka1recipient123"

        # Create signature
        signature = _create_signature(
            body, TEST_PRIVATE_KEY, timestamp, transfer_address
        )

        # Signature should be valid base64
        decoded = base64.b64decode(signature)
        assert len(decoded) == 64  # ECDSA signature is 64 bytes

        # Verify signature is deterministic (same inputs = same output)
        signature2 = _create_signature(
            body, TEST_PRIVATE_KEY, timestamp, transfer_address
        )
        assert signature == signature2

    def test_signature_changes_with_different_body(self, factory, participant):
        """Signature changes when body changes."""
        from multiplexer_llm.gonka.client_factory import _create_signature

        timestamp = 1234567890000000000
        transfer_address = "gonka1recipient123"

        sig1 = _create_signature(
            b'{"data": 1}', TEST_PRIVATE_KEY, timestamp, transfer_address
        )
        sig2 = _create_signature(
            b'{"data": 2}', TEST_PRIVATE_KEY, timestamp, transfer_address
        )

        assert sig1 != sig2

    def test_signature_changes_with_different_timestamp(self, factory, participant):
        """Signature changes when timestamp changes."""
        from multiplexer_llm.gonka.client_factory import _create_signature

        body = b'{"test": "data"}'
        transfer_address = "gonka1recipient123"

        sig1 = _create_signature(
            body, TEST_PRIVATE_KEY, 1234567890000000000, transfer_address
        )
        sig2 = _create_signature(
            body, TEST_PRIVATE_KEY, 1234567890000000001, transfer_address
        )

        assert sig1 != sig2

    def test_signature_changes_with_different_transfer_address(
        self, factory, participant
    ):
        """Signature changes when transfer address changes."""
        from multiplexer_llm.gonka.client_factory import _create_signature

        body = b'{"test": "data"}'
        timestamp = 1234567890000000000

        sig1 = _create_signature(body, TEST_PRIVATE_KEY, timestamp, "gonka1address1")
        sig2 = _create_signature(body, TEST_PRIVATE_KEY, timestamp, "gonka1address2")

        assert sig1 != sig2


class TestGonkaClientFactoryThreadSafety:
    """Tests for thread safety of GonkaClientFactory."""

    def test_concurrent_client_creation(self):
        """NFR-3: Thread-safe for concurrent client creation."""
        from multiplexer_llm.gonka import GonkaClientFactory, GonkaParticipant

        factory = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)

        clients = []
        errors = []

        def create_client(index):
            try:
                participant = GonkaParticipant(
                    address=f"gonka1recipient{index}",
                    inference_url=f"https://node{index}.example.com",
                )
                client = factory.create_client(participant)
                clients.append((index, client))
            except Exception as e:
                errors.append((index, e))

        # Create multiple threads
        threads = [threading.Thread(target=create_client, args=(i,)) for i in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Should have created all clients
        assert len(clients) == 10

        # All clients should be independent
        base_urls = [str(c.base_url) for _, c in clients]
        assert len(set(base_urls)) == 10  # All unique


class TestGonkaClientFactoryPerformance:
    """Tests for performance requirements."""

    def test_client_creation_is_fast(self):
        """NFR-2: Client creation is fast (no network calls)."""
        from multiplexer_llm.gonka import GonkaClientFactory, GonkaParticipant

        factory = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)
        participant = GonkaParticipant(
            address="gonka1recipient123",
            inference_url="https://node.example.com",
        )

        # Time 100 client creations
        start = time.time()
        for _ in range(100):
            factory.create_client(participant)
        elapsed = time.time() - start

        # Should complete very quickly (less than 1 second for 100 clients)
        assert elapsed < 1.0, f"Client creation too slow: {elapsed}s for 100 clients"

    def test_factory_construction_derives_address_once(self):
        """Address derivation happens once at construction, not per client."""
        from multiplexer_llm.gonka import GonkaClientFactory, GonkaParticipant

        factory = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)
        initial_address = factory.requester_address

        participant = GonkaParticipant(
            address="gonka1recipient123",
            inference_url="https://node.example.com",
        )

        # Create multiple clients
        for _ in range(10):
            factory.create_client(participant)

        # Address should still be the same (not re-derived)
        assert factory.requester_address == initial_address


class TestAddressDerivation:
    """Tests for address derivation correctness."""

    def test_address_derivation_produces_valid_bech32(self):
        """Derived address is valid bech32 format."""
        from multiplexer_llm.gonka import GonkaClientFactory

        factory = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)

        # Should start with gonka1
        assert factory.requester_address.startswith("gonka1")

        # Should be lowercase
        assert factory.requester_address == factory.requester_address.lower()

        # Should be reasonable length (bech32 addresses are ~40-45 chars)
        assert 38 <= len(factory.requester_address) <= 50

    def test_address_derivation_matches_gonka_openai(self):
        """Address derivation matches gonka-openai implementation."""
        from multiplexer_llm.gonka import GonkaClientFactory

        # Use gonka_openai's implementation for comparison
        try:
            from gonka_openai import gonka_address

            factory = GonkaClientFactory(private_key=TEST_PRIVATE_KEY)
            expected_address = gonka_address(TEST_PRIVATE_KEY)

            assert factory.requester_address == expected_address
        except ImportError:
            # gonka_openai not in path, skip comparison
            pytest.skip("gonka_openai not available for comparison test")

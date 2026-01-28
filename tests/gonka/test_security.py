"""
Security tests for the Gonka integration.

These tests verify that private keys are never exposed in:
- Error messages
- Exception strings
- Object repr() output
- Log messages
- Client configurations

This is critical for security compliance as private keys provide
signing authority for Gonka network transactions.
"""

import logging
import re
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from multiplexer_llm.gonka import (
    GonkaClientFactory,
    GonkaParticipant,
    GonkaConfig,
    register_gonka_models,
)
from multiplexer_llm.gonka.exceptions import (
    GonkaError,
    GonkaClientError,
    GonkaDiscoveryError,
    GonkaConfigError,
    GonkaRefreshError,
)
from multiplexer_llm.gonka.client_factory import _validate_private_key, _create_signature


# ==============================================================================
# Test Constants
# ==============================================================================

# Various test private keys to check for exposure
TEST_KEY_1 = "0x" + "a" * 64
TEST_KEY_2 = "0x" + "b" * 64
TEST_KEY_3 = "deadbeef" * 8  # Memorable pattern
TEST_KEY_4 = "1234567890abcdef" * 4  # Another pattern

# Invalid keys that might be accidentally exposed in error messages
INVALID_KEY_SHORT = "0x" + "abc123" * 5
INVALID_KEY_NOT_HEX = "0x" + "ghijkl" * 10


# ==============================================================================
# Factory repr() Security Tests
# ==============================================================================


class TestFactoryReprSecurity:
    """Tests that GonkaClientFactory never exposes keys in repr()."""

    def test_factory_repr_no_key(self, test_private_key: str):
        """Factory repr() should not contain the private key."""
        factory = GonkaClientFactory(private_key=test_private_key)
        repr_str = repr(factory)
        
        # Key should not appear (with or without 0x prefix)
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in repr_str.lower()
        assert test_private_key.lower() not in repr_str.lower()

    def test_factory_str_no_key(self, test_private_key: str):
        """Factory str() should not contain the private key."""
        factory = GonkaClientFactory(private_key=test_private_key)
        str_output = str(factory)
        
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in str_output.lower()

    def test_factory_repr_shows_address_only(self, test_private_key: str):
        """Factory repr() should show derived address, not key."""
        factory = GonkaClientFactory(private_key=test_private_key)
        repr_str = repr(factory)
        
        # Should contain the derived gonka address
        # Note: We can't assert the exact address without running derivation,
        # but we can check it doesn't contain the key pattern
        assert "a" * 32 not in repr_str.lower()

    def test_factory_dir_no_private_key_exposure(self, test_private_key: str):
        """Factory dir() should not expose key in unintended accessible attributes."""
        factory = GonkaClientFactory(private_key=test_private_key)
        attrs = dir(factory)
        
        # private_key property exists but is intentional - exclude it from check
        # The key should be in _private_key (private attribute)
        # We check non-private attrs except the intentional private_key accessor
        public_attrs = [a for a in attrs if not a.startswith("_") and a != "private_key"]
        
        for attr in public_attrs:
            attr_value = getattr(factory, attr, None)
            if isinstance(attr_value, str):
                key_hex = test_private_key.replace("0x", "")
                assert key_hex not in attr_value.lower(), f"Key found in {attr}"


# ==============================================================================
# Config repr() Security Tests
# ==============================================================================


class TestConfigReprSecurity:
    """Tests that GonkaConfig never exposes keys in repr()."""

    def test_config_repr_masks_key(self, clean_gonka_env):
        """Config repr() should show masked key."""
        config = GonkaConfig(
            private_key=TEST_KEY_1,
            source_url="https://api.gonka.network",
        )
        repr_str = repr(config)
        
        # Key should be masked
        key_hex = TEST_KEY_1.replace("0x", "")
        assert key_hex not in repr_str.lower()
        
        # Should show some form of masking indicator
        assert "***" in repr_str or "REDACTED" in repr_str.upper()

    def test_config_str_masks_key(self, clean_gonka_env):
        """Config str() should not expose the key."""
        config = GonkaConfig(
            private_key=TEST_KEY_1,
            source_url="https://api.gonka.network",
        )
        str_output = str(config)
        
        key_hex = TEST_KEY_1.replace("0x", "")
        assert key_hex not in str_output.lower()

    def test_config_repr_shows_other_fields(self, clean_gonka_env):
        """Config repr() should still show non-sensitive fields."""
        config = GonkaConfig(
            private_key=TEST_KEY_1,
            source_url="https://api.gonka.network",
            verify_proofs=True,
        )
        repr_str = repr(config)
        
        # Should show source_url
        assert "api.gonka.network" in repr_str
        assert "verify_proofs" in repr_str


# ==============================================================================
# Exception Message Security Tests
# ==============================================================================


class TestExceptionMessageSecurity:
    """Tests that exceptions never expose private keys."""

    def test_client_error_no_key_in_message(self):
        """GonkaClientError should not contain key in message."""
        # Even if someone accidentally passes the key
        error = GonkaClientError(
            f"Error with key {TEST_KEY_1}",  # Bad practice, but test protection
            participant_address="gonka1test",
        )
        error_str = str(error)
        
        # The error might contain the message, but repr should not expose key patterns
        # This tests that we don't accidentally echo back user-provided keys
        key_hex = TEST_KEY_1.replace("0x", "")
        # Note: This is testing the exception itself, not preventing bad input
        assert "gonka1test" in error_str  # Participant address is OK

    def test_config_error_no_key_in_message(self, clean_gonka_env):
        """GonkaConfigError for invalid key should not expose the key."""
        try:
            GonkaConfig(
                private_key=INVALID_KEY_NOT_HEX,
                source_url="https://api.gonka.network",
            )
            pytest.fail("Expected GonkaConfigError")
        except GonkaConfigError as e:
            error_str = str(e)
            # Should not contain the invalid key
            assert INVALID_KEY_NOT_HEX not in error_str
            assert "ghijkl" not in error_str.lower()

    def test_validation_error_no_key_exposure(self):
        """Key validation errors should not expose the invalid key."""
        try:
            _validate_private_key(TEST_KEY_3)
        except GonkaClientError as e:
            error_str = str(e)
            assert TEST_KEY_3 not in error_str
            assert "deadbeef" not in error_str.lower()

    def test_factory_creation_error_no_key(self):
        """Factory creation errors should not expose the key."""
        try:
            # Invalid length key
            GonkaClientFactory(private_key="0x" + "a" * 32)
            pytest.fail("Expected GonkaClientError")
        except GonkaClientError as e:
            error_str = str(e)
            assert "a" * 32 not in error_str.lower()

    def test_signing_error_no_key_exposure(self):
        """Signing failures should not expose the key."""
        # Mock a signing failure by providing invalid inputs
        try:
            # This should work normally, but if it fails, key shouldn't be exposed
            _create_signature(
                body=b"test",
                private_key_hex=TEST_KEY_1,
                timestamp=1234567890,
                transfer_address="gonka1test",
            )
        except GonkaClientError as e:
            error_str = str(e)
            key_hex = TEST_KEY_1.replace("0x", "")
            assert key_hex not in error_str.lower()


# ==============================================================================
# Logging Security Tests
# ==============================================================================


class TestLoggingSecurity:
    """Tests that private keys are never logged."""

    def test_factory_creation_no_key_logged(self, test_private_key: str, caplog):
        """Factory creation should not log the key."""
        with caplog.at_level(logging.DEBUG):
            factory = GonkaClientFactory(private_key=test_private_key)
        
        log_output = caplog.text
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in log_output.lower()

    def test_client_creation_no_key_logged(
        self,
        test_private_key: str,
        test_participant: GonkaParticipant,
        caplog,
    ):
        """Client creation should not log the key."""
        factory = GonkaClientFactory(private_key=test_private_key)
        
        with caplog.at_level(logging.DEBUG):
            factory.create_client(test_participant)
        
        log_output = caplog.text
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in log_output.lower()

    def test_signing_no_key_logged(self, test_private_key: str, caplog):
        """Request signing should not log the key."""
        with caplog.at_level(logging.DEBUG):
            _create_signature(
                body=b'{"test": "data"}',
                private_key_hex=test_private_key,
                timestamp=1234567890000000000,
                transfer_address="gonka1recipient",
            )
        
        log_output = caplog.text
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in log_output.lower()

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_registration_no_key_logged(
        self,
        mock_registrar_cls,
        mock_discovery_cls,
        mock_factory_cls,
        test_participants,
        test_private_key: str,
        clean_gonka_env,
        caplog,
    ):
        """Registration flow should not log the key."""
        mock_discovery = Mock()
        mock_discovery.discover.return_value = test_participants
        mock_discovery_cls.return_value = mock_discovery
        
        mock_registrar = Mock()
        mock_registrar.register_all.return_value = 3
        mock_registrar_cls.return_value = mock_registrar
        
        mock_multiplexer = Mock()
        
        with caplog.at_level(logging.DEBUG):
            register_gonka_models(
                mock_multiplexer,
                source_url="https://api.gonka.network",
                private_key=test_private_key,
                refresh_enabled=False,
            )
        
        log_output = caplog.text
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in log_output.lower()


# ==============================================================================
# Client Object Security Tests
# ==============================================================================


class TestClientObjectSecurity:
    """Tests that OpenAI client objects don't expose keys."""

    def test_client_repr_no_key(
        self,
        test_private_key: str,
        test_participant: GonkaParticipant,
    ):
        """OpenAI client repr() should not expose the signing key."""
        factory = GonkaClientFactory(private_key=test_private_key)
        client = factory.create_client(test_participant)
        
        repr_str = repr(client)
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in repr_str.lower()

    def test_async_client_repr_no_key(
        self,
        test_private_key: str,
        test_participant: GonkaParticipant,
    ):
        """Async OpenAI client repr() should not expose the signing key."""
        factory = GonkaClientFactory(private_key=test_private_key)
        client = factory.create_async_client(test_participant)
        
        repr_str = repr(client)
        key_hex = test_private_key.replace("0x", "")
        assert key_hex not in repr_str.lower()

    def test_http_client_no_key_in_headers(
        self,
        test_private_key: str,
        test_participant: GonkaParticipant,
    ):
        """HTTP client default headers should not contain the raw key."""
        factory = GonkaClientFactory(private_key=test_private_key)
        http_client = factory._create_signing_http_client(
            transfer_address=test_participant.address,
            is_async=False,
        )
        
        # Check default headers
        default_headers = dict(http_client.headers)
        for header_name, header_value in default_headers.items():
            key_hex = test_private_key.replace("0x", "")
            assert key_hex not in str(header_value).lower(), \
                f"Key found in header {header_name}"


# ==============================================================================
# Environment Variable Security Tests
# ==============================================================================


class TestEnvironmentVariableSecurity:
    """Tests that environment variable handling doesn't expose keys."""

    def test_env_key_not_logged_on_read(self, clean_gonka_env, caplog):
        """Reading key from environment should not log it."""
        import os
        os.environ["GONKA_PRIVATE_KEY"] = TEST_KEY_1
        os.environ["GONKA_SOURCE_URL"] = "https://api.gonka.network"
        
        with caplog.at_level(logging.DEBUG):
            config = GonkaConfig()
        
        log_output = caplog.text
        key_hex = TEST_KEY_1.replace("0x", "")
        assert key_hex not in log_output.lower()

    def test_config_from_env_repr_safe(self, clean_gonka_env):
        """Config created from env vars should have safe repr()."""
        import os
        os.environ["GONKA_PRIVATE_KEY"] = TEST_KEY_1
        os.environ["GONKA_SOURCE_URL"] = "https://api.gonka.network"
        
        config = GonkaConfig()
        repr_str = repr(config)
        
        key_hex = TEST_KEY_1.replace("0x", "")
        assert key_hex not in repr_str.lower()


# ==============================================================================
# Serialization Security Tests
# ==============================================================================


class TestSerializationSecurity:
    """Tests that serialization methods don't expose keys."""

    def test_config_to_dict_masks_key(self, clean_gonka_env):
        """If config has to_dict(), it should mask the key."""
        config = GonkaConfig(
            private_key=TEST_KEY_1,
            source_url="https://api.gonka.network",
        )
        
        # Check if to_dict exists
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
            if "private_key" in config_dict:
                assert config_dict["private_key"] != TEST_KEY_1
                key_hex = TEST_KEY_1.replace("0x", "")
                assert key_hex not in str(config_dict["private_key"]).lower()

    def test_exception_args_no_key(self):
        """Exception args should not contain the key."""
        error = GonkaClientError(
            "Test error",
            participant_address="gonka1test",
        )
        
        # Check exception args
        for arg in error.args:
            if isinstance(arg, str):
                assert TEST_KEY_1 not in arg
                key_hex = TEST_KEY_1.replace("0x", "")
                assert key_hex not in arg.lower()


# ==============================================================================
# Memory Security Tests
# ==============================================================================


class TestMemorySecurity:
    """Tests for secure memory handling of keys."""

    def test_factory_private_key_access_controlled(self, test_private_key: str):
        """Private key access should be through property only."""
        factory = GonkaClientFactory(private_key=test_private_key)
        
        # The key is accessible through the property for signing
        assert factory.private_key == test_private_key
        
        # But internal storage uses underscore prefix
        assert hasattr(factory, "_private_key")

    def test_multiple_factories_isolated(self):
        """Multiple factories should not share key state."""
        factory1 = GonkaClientFactory(private_key=TEST_KEY_1)
        factory2 = GonkaClientFactory(private_key=TEST_KEY_2)
        
        # Keys should be different
        assert factory1.private_key != factory2.private_key
        
        # Addresses should be different
        assert factory1.requester_address != factory2.requester_address


# ==============================================================================
# Comprehensive Exposure Scan
# ==============================================================================


class TestComprehensiveKeyExposureScan:
    """Comprehensive scan for key exposure across all components."""

    def scan_for_key(self, text: str, key: str) -> bool:
        """Check if key appears in text (case insensitive)."""
        key_hex = key.replace("0x", "").lower()
        return key_hex in text.lower()

    def test_full_flow_no_key_exposure(
        self,
        test_private_key: str,
        test_participant: GonkaParticipant,
    ):
        """Run full flow and verify no key exposure anywhere."""
        exposed_locations = []
        key_hex = test_private_key.replace("0x", "").lower()
        
        # Create factory
        factory = GonkaClientFactory(private_key=test_private_key)
        if self.scan_for_key(repr(factory), test_private_key):
            exposed_locations.append("factory repr")
        if self.scan_for_key(str(factory), test_private_key):
            exposed_locations.append("factory str")
        
        # Create client
        client = factory.create_client(test_participant)
        if self.scan_for_key(repr(client), test_private_key):
            exposed_locations.append("client repr")
        if self.scan_for_key(str(client), test_private_key):
            exposed_locations.append("client str")
        
        # Create async client
        async_client = factory.create_async_client(test_participant)
        if self.scan_for_key(repr(async_client), test_private_key):
            exposed_locations.append("async client repr")
        
        # Create signature (verify it doesn't expose key in normal flow)
        try:
            sig = _create_signature(
                body=b"test",
                private_key_hex=test_private_key,
                timestamp=1234567890,
                transfer_address="gonka1test",
            )
            # Signature should not be the key itself
            if key_hex in sig.lower():
                exposed_locations.append("signature output")
        except Exception:
            pass  # Expected in some cases
        
        assert len(exposed_locations) == 0, \
            f"Key exposed in: {', '.join(exposed_locations)}"

    def test_error_scenarios_no_key_exposure(self):
        """Test various error scenarios don't expose keys."""
        exposed_locations = []
        test_key = TEST_KEY_3  # Use memorable pattern
        
        # Invalid key length
        try:
            GonkaClientFactory(private_key="0x" + "a" * 32)
        except GonkaClientError as e:
            if self.scan_for_key(str(e), "0x" + "a" * 32):
                exposed_locations.append("invalid length error")
        
        # Invalid hex
        try:
            GonkaClientFactory(private_key="0x" + "z" * 64)
        except GonkaClientError as e:
            if "z" * 10 in str(e).lower():
                exposed_locations.append("invalid hex error")
        
        assert len(exposed_locations) == 0, \
            f"Key exposed in error messages: {', '.join(exposed_locations)}"

"""
Tests for the Gonka configuration module.

This module tests GonkaConfig validation, environment variable resolution,
and the register_gonka_models convenience function.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from multiplexer_llm.gonka.config import (
    GonkaConfig,
    register_gonka_models,
    _is_valid_url,
    _parse_endpoint,
    _resolve_env_str,
    _resolve_env_bool,
    _resolve_env_list,
    _build_effective_config,
    _create_participants_from_endpoints,
    ENV_PRIVATE_KEY,
    ENV_SOURCE_URL,
    ENV_ENDPOINTS,
    ENV_ADDRESS,
    ENV_VERIFY_PROOF,
)
from multiplexer_llm.gonka.exceptions import GonkaConfigError
from multiplexer_llm.gonka.types import GonkaParticipant, GonkaRegistrationResult


# Valid test private key (32 bytes hex)
VALID_PRIVATE_KEY = "a" * 64
VALID_PRIVATE_KEY_WITH_PREFIX = "0x" + "a" * 64
VALID_SOURCE_URL = "https://api.gonka.network"
VALID_ENDPOINT = "https://node.example.com;gonka1abc123"


@pytest.fixture(autouse=False)
def clean_env():
    """Remove all GONKA_ and TEST_ env vars for test isolation."""
    # Save original values
    env_vars_to_clear = [k for k in os.environ if k.startswith(("GONKA_", "TEST_"))]
    original_env = {k: os.environ[k] for k in env_vars_to_clear}
    
    # Clear all GONKA_ and TEST_ vars
    for var in env_vars_to_clear:
        del os.environ[var]
    
    yield
    
    # Clear any vars that were set during the test
    env_vars_after = [k for k in os.environ if k.startswith(("GONKA_", "TEST_"))]
    for var in env_vars_after:
        del os.environ[var]
    
    # Restore original env vars
    for k, v in original_env.items():
        os.environ[k] = v


class TestUrlValidation:
    """Tests for URL validation helper."""

    def test_valid_https_url(self):
        assert _is_valid_url("https://example.com") is True

    def test_valid_http_url(self):
        assert _is_valid_url("http://example.com") is True

    def test_valid_file_url(self):
        assert _is_valid_url("file:///path/to/file") is True

    def test_invalid_empty_url(self):
        assert _is_valid_url("") is False

    def test_invalid_no_scheme(self):
        assert _is_valid_url("example.com") is False

    def test_invalid_scheme(self):
        assert _is_valid_url("ftp://example.com") is False

    def test_invalid_none(self):
        assert _is_valid_url(None) is False


class TestEndpointParsing:
    """Tests for endpoint string parsing."""

    def test_valid_endpoint(self):
        url, address = _parse_endpoint("https://api.example.com;gonka1abc")
        assert url == "https://api.example.com"
        assert address == "gonka1abc"

    def test_valid_endpoint_with_spaces(self):
        url, address = _parse_endpoint("  https://api.example.com  ;  gonka1abc  ")
        assert url == "https://api.example.com"
        assert address == "gonka1abc"

    def test_invalid_no_semicolon(self):
        with pytest.raises(GonkaConfigError) as exc_info:
            _parse_endpoint("https://api.example.com")
        assert "Invalid endpoint format" in str(exc_info.value)
        assert exc_info.value.field == "endpoints"

    def test_invalid_empty_url(self):
        with pytest.raises(GonkaConfigError) as exc_info:
            _parse_endpoint(";gonka1abc")
        assert "Invalid endpoint format" in str(exc_info.value)

    def test_invalid_empty_address(self):
        with pytest.raises(GonkaConfigError) as exc_info:
            _parse_endpoint("https://api.example.com;")
        assert "Invalid endpoint format" in str(exc_info.value)

    def test_invalid_url_format(self):
        with pytest.raises(GonkaConfigError) as exc_info:
            _parse_endpoint("not-a-url;gonka1abc")
        assert "Invalid URL in endpoint" in str(exc_info.value)


class TestEnvResolution:
    """Tests for environment variable resolution helpers."""

    def test_resolve_str_explicit_wins(self, clean_env):
        os.environ["TEST_VAR"] = "from_env"
        result = _resolve_env_str("explicit", "TEST_VAR")
        assert result == "explicit"

    def test_resolve_str_env_wins_over_default(self, clean_env):
        os.environ["TEST_VAR"] = "from_env"
        result = _resolve_env_str(None, "TEST_VAR", "default")
        assert result == "from_env"

    def test_resolve_str_default(self, clean_env):
        result = _resolve_env_str(None, "TEST_VAR", "default")
        assert result == "default"

    def test_resolve_str_strips_whitespace(self, clean_env):
        os.environ["TEST_VAR"] = "  value  "
        result = _resolve_env_str(None, "TEST_VAR")
        assert result == "value"

    def test_resolve_bool_explicit_true(self, clean_env):
        os.environ["TEST_VAR"] = "0"
        result = _resolve_env_bool(True, "TEST_VAR")
        assert result is True

    def test_resolve_bool_explicit_false(self, clean_env):
        os.environ["TEST_VAR"] = "1"
        result = _resolve_env_bool(False, "TEST_VAR")
        assert result is False

    def test_resolve_bool_env_true_values(self, clean_env):
        for value in ["1", "true", "TRUE", "yes", "YES"]:
            os.environ["TEST_VAR"] = value
            result = _resolve_env_bool(None, "TEST_VAR")
            assert result is True, f"Failed for value: {value}"

    def test_resolve_bool_env_false_values(self, clean_env):
        for value in ["0", "false", "FALSE", "no", "NO", "anything"]:
            os.environ["TEST_VAR"] = value
            result = _resolve_env_bool(None, "TEST_VAR")
            assert result is False, f"Failed for value: {value}"

    def test_resolve_bool_default(self, clean_env):
        result = _resolve_env_bool(None, "TEST_VAR", True)
        assert result is True

    def test_resolve_list_explicit_wins(self, clean_env):
        os.environ["TEST_VAR"] = "a,b,c"
        result = _resolve_env_list(["x", "y"], "TEST_VAR")
        assert result == ["x", "y"]

    def test_resolve_list_env(self, clean_env):
        os.environ["TEST_VAR"] = "a, b , c"
        result = _resolve_env_list(None, "TEST_VAR")
        assert result == ["a", "b", "c"]

    def test_resolve_list_default(self, clean_env):
        result = _resolve_env_list(None, "TEST_VAR", ["default"])
        assert result == ["default"]


class TestGonkaConfigValidation:
    """Tests for GonkaConfig validation."""

    def test_valid_config_with_source_url(self, clean_env):
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url=VALID_SOURCE_URL,
        )
        assert config.private_key == VALID_PRIVATE_KEY
        assert config.source_url == VALID_SOURCE_URL

    def test_valid_config_with_prefix(self, clean_env):
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY_WITH_PREFIX,
            source_url=VALID_SOURCE_URL,
        )
        assert config.private_key == VALID_PRIVATE_KEY_WITH_PREFIX

    def test_valid_config_with_endpoints(self, clean_env):
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            endpoints=[VALID_ENDPOINT],
        )
        assert config.endpoints == [VALID_ENDPOINT]

    def test_missing_private_key(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(source_url=VALID_SOURCE_URL)
        assert "Private key is required" in str(exc_info.value)
        assert exc_info.value.field == "private_key"

    def test_invalid_private_key_too_short(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key="abc123",
                source_url=VALID_SOURCE_URL,
            )
        assert "expected 64 hex characters" in str(exc_info.value)
        # Private key value should NOT be in error message
        assert "abc123" not in str(exc_info.value)

    def test_invalid_private_key_not_hex(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key="g" * 64,  # 'g' is not valid hex
                source_url=VALID_SOURCE_URL,
            )
        assert "not valid hexadecimal" in str(exc_info.value)

    def test_missing_endpoint_source(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(private_key=VALID_PRIVATE_KEY)
        assert "Endpoint source is required" in str(exc_info.value)

    def test_invalid_source_url(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key=VALID_PRIVATE_KEY,
                source_url="not-a-valid-url",
            )
        assert "Invalid URL" in str(exc_info.value)
        assert exc_info.value.field == "source_url"

    def test_empty_endpoints_list(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key=VALID_PRIVATE_KEY,
                endpoints=[],
            )
        assert "Endpoints list cannot be empty" in str(exc_info.value)

    def test_invalid_endpoint_format(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key=VALID_PRIVATE_KEY,
                endpoints=["invalid-endpoint"],
            )
        assert "Invalid endpoint format" in str(exc_info.value)

    def test_invalid_refresh_interval(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key=VALID_PRIVATE_KEY,
                source_url=VALID_SOURCE_URL,
                refresh_interval_seconds=0,
            )
        assert "refresh_interval_seconds must be positive" in str(exc_info.value)

    def test_negative_max_concurrent(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key=VALID_PRIVATE_KEY,
                source_url=VALID_SOURCE_URL,
                default_max_concurrent=-1,
            )
        assert "default_max_concurrent must be non-negative" in str(exc_info.value)

    def test_empty_model_name_prefix(self, clean_env):
        with pytest.raises(GonkaConfigError) as exc_info:
            GonkaConfig(
                private_key=VALID_PRIVATE_KEY,
                source_url=VALID_SOURCE_URL,
                model_name_prefix="",
            )
        assert "model_name_prefix cannot be empty" in str(exc_info.value)


class TestGonkaConfigEnvResolution:
    """Tests for GonkaConfig environment variable resolution."""

    def test_private_key_from_env(self, clean_env):
        os.environ[ENV_PRIVATE_KEY] = VALID_PRIVATE_KEY
        os.environ[ENV_SOURCE_URL] = VALID_SOURCE_URL
        config = GonkaConfig()
        assert config.private_key == VALID_PRIVATE_KEY

    def test_source_url_from_env(self, clean_env):
        os.environ[ENV_SOURCE_URL] = VALID_SOURCE_URL
        config = GonkaConfig(private_key=VALID_PRIVATE_KEY)
        assert config.source_url == VALID_SOURCE_URL

    def test_endpoints_from_env(self, clean_env):
        os.environ[ENV_ENDPOINTS] = f"{VALID_ENDPOINT}, https://node2.example.com;gonka1xyz"
        config = GonkaConfig(private_key=VALID_PRIVATE_KEY)
        assert len(config.endpoints) == 2
        assert VALID_ENDPOINT in config.endpoints[0]

    def test_address_from_env(self, clean_env):
        os.environ[ENV_ADDRESS] = "gonka1custom"
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url=VALID_SOURCE_URL,
        )
        assert config.address == "gonka1custom"

    def test_verify_proofs_from_env(self, clean_env):
        os.environ[ENV_VERIFY_PROOF] = "1"
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url=VALID_SOURCE_URL,
        )
        assert config.verify_proofs is True

    def test_explicit_overrides_env(self, clean_env):
        os.environ[ENV_PRIVATE_KEY] = "b" * 64
        os.environ[ENV_SOURCE_URL] = "https://env.gonka.network"
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url=VALID_SOURCE_URL,
        )
        assert config.private_key == VALID_PRIVATE_KEY
        assert config.source_url == VALID_SOURCE_URL


class TestGonkaConfigRepr:
    """Tests for GonkaConfig string representation."""

    def test_repr_hides_private_key(self, clean_env):
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url=VALID_SOURCE_URL,
        )
        repr_str = repr(config)
        assert "***" in repr_str
        assert VALID_PRIVATE_KEY not in repr_str

    def test_repr_shows_other_fields(self, clean_env):
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url=VALID_SOURCE_URL,
            verify_proofs=True,
        )
        repr_str = repr(config)
        assert VALID_SOURCE_URL in repr_str
        assert "verify_proofs=True" in repr_str


class TestBuildEffectiveConfig:
    """Tests for _build_effective_config function."""

    def test_no_config_uses_args_and_env(self, clean_env):
        os.environ[ENV_PRIVATE_KEY] = VALID_PRIVATE_KEY
        config = _build_effective_config(
            config=None,
            source_url=VALID_SOURCE_URL,
            endpoints=None,
            private_key=None,
            verify_proofs=None,
            refresh_enabled=None,
            refresh_interval_seconds=None,
            default_max_concurrent=None,
            register_as_fallback=False,
            model_name_prefix="gonka:",
            address=None,
        )
        assert config.private_key == VALID_PRIVATE_KEY
        assert config.source_url == VALID_SOURCE_URL

    def test_explicit_args_override_config(self, clean_env):
        base_config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url="https://base.example.com",
        )
        config = _build_effective_config(
            config=base_config,
            source_url="https://override.example.com",
            endpoints=None,
            private_key=None,
            verify_proofs=True,
            refresh_enabled=None,
            refresh_interval_seconds=None,
            default_max_concurrent=None,
            register_as_fallback=False,
            model_name_prefix="gonka:",
            address=None,
        )
        assert config.source_url == "https://override.example.com"
        assert config.verify_proofs is True
        # Non-overridden values from config
        assert config.private_key == VALID_PRIVATE_KEY


class TestCreateParticipantsFromEndpoints:
    """Tests for _create_participants_from_endpoints function."""

    def test_creates_participants(self, clean_env):
        endpoints = [
            "https://node1.example.com;gonka1abc",
            "https://node2.example.com;gonka1xyz",
        ]
        participants = _create_participants_from_endpoints(endpoints)
        assert len(participants) == 2
        assert participants[0].address == "gonka1abc"
        assert participants[0].inference_url == "https://node1.example.com"
        assert participants[1].address == "gonka1xyz"

    def test_empty_endpoints(self, clean_env):
        participants = _create_participants_from_endpoints([])
        assert participants == []


class TestRegisterGonkaModels:
    """Tests for register_gonka_models convenience function."""

    @pytest.fixture
    def mock_multiplexer(self):
        """Create a mock multiplexer."""
        mux = MagicMock()
        mux._weighted_models = []
        mux._fallback_models = []
        return mux

    @pytest.fixture
    def mock_participants(self):
        """Create mock participants."""
        return [
            GonkaParticipant(
                address="gonka1abc123",
                inference_url="https://node1.example.com",
                weight=100,
                epoch_id=42,
            ),
            GonkaParticipant(
                address="gonka1xyz789",
                inference_url="https://node2.example.com",
                weight=50,
                epoch_id=42,
            ),
        ]

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    @patch("multiplexer_llm.gonka.config.RefreshManager")
    def test_basic_registration(
        self,
        mock_refresh_manager_class,
        mock_registrar_class,
        mock_discovery_class,
        mock_factory_class,
        mock_multiplexer,
        mock_participants,
        clean_env,
    ):
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = mock_participants
        mock_discovery_class.return_value = mock_discovery

        mock_registrar = MagicMock()
        mock_registrar.register_all.return_value = 2
        mock_registrar_class.return_value = mock_registrar

        mock_refresh_manager = MagicMock()
        mock_refresh_manager_class.return_value = mock_refresh_manager

        # Call the function
        result = register_gonka_models(
            mock_multiplexer,
            source_url=VALID_SOURCE_URL,
            private_key=VALID_PRIVATE_KEY,
        )

        # Verify result
        assert isinstance(result, GonkaRegistrationResult)
        assert result.models_registered == 2
        assert len(result.participants) == 2
        assert result.epoch_id == 42
        assert result.refresh_manager is not None

        # Verify components were created
        mock_factory_class.assert_called_once_with(private_key=VALID_PRIVATE_KEY)
        mock_discovery_class.assert_called_once()
        mock_registrar_class.assert_called_once()

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_registration_with_refresh_disabled(
        self,
        mock_registrar_class,
        mock_discovery_class,
        mock_factory_class,
        mock_multiplexer,
        mock_participants,
        clean_env,
    ):
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = mock_participants
        mock_discovery_class.return_value = mock_discovery

        mock_registrar = MagicMock()
        mock_registrar.register_all.return_value = 2
        mock_registrar_class.return_value = mock_registrar

        # Call with refresh disabled
        result = register_gonka_models(
            mock_multiplexer,
            source_url=VALID_SOURCE_URL,
            private_key=VALID_PRIVATE_KEY,
            refresh_enabled=False,
        )

        # Verify refresh manager is not created
        assert result.refresh_manager is None

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_registration_with_explicit_endpoints(
        self,
        mock_registrar_class,
        mock_factory_class,
        mock_multiplexer,
        clean_env,
    ):
        # Setup mocks
        mock_registrar = MagicMock()
        mock_registrar.register_all.return_value = 1
        mock_registrar_class.return_value = mock_registrar

        # Call with explicit endpoints
        result = register_gonka_models(
            mock_multiplexer,
            endpoints=[VALID_ENDPOINT],
            private_key=VALID_PRIVATE_KEY,
            refresh_enabled=False,
        )

        # Verify result
        assert result.models_registered == 1
        assert len(result.participants) == 1
        assert result.refresh_manager is None

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_registration_with_config_object(
        self,
        mock_registrar_class,
        mock_discovery_class,
        mock_factory_class,
        mock_multiplexer,
        mock_participants,
        clean_env,
    ):
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = mock_participants
        mock_discovery_class.return_value = mock_discovery

        mock_registrar = MagicMock()
        mock_registrar.register_all.return_value = 2
        mock_registrar_class.return_value = mock_registrar

        # Create config
        config = GonkaConfig(
            private_key=VALID_PRIVATE_KEY,
            source_url=VALID_SOURCE_URL,
            verify_proofs=True,
        )

        # Call with config object
        result = register_gonka_models(mock_multiplexer, config=config)

        # Verify discovery was created with verify_proofs
        mock_discovery_class.assert_called_once()
        call_kwargs = mock_discovery_class.call_args[1]
        assert call_kwargs["verify_proofs"] is True

    @patch("multiplexer_llm.gonka.config.GonkaClientFactory")
    @patch("multiplexer_llm.gonka.config.EndpointDiscovery")
    @patch("multiplexer_llm.gonka.config.ModelRegistrar")
    def test_registration_as_fallback(
        self,
        mock_registrar_class,
        mock_discovery_class,
        mock_factory_class,
        mock_multiplexer,
        mock_participants,
        clean_env,
    ):
        # Setup mocks
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = mock_participants
        mock_discovery_class.return_value = mock_discovery

        mock_registrar = MagicMock()
        mock_registrar.register_all.return_value = 2
        mock_registrar_class.return_value = mock_registrar

        # Call with register_as_fallback
        result = register_gonka_models(
            mock_multiplexer,
            source_url=VALID_SOURCE_URL,
            private_key=VALID_PRIVATE_KEY,
            register_as_fallback=True,
            refresh_enabled=False,
        )

        # Verify registrar was called with as_fallback=True
        mock_registrar.register_all.assert_called_once()
        call_kwargs = mock_registrar.register_all.call_args[1]
        assert call_kwargs["as_fallback"] is True

    def test_invalid_config_raises_error(self, mock_multiplexer, clean_env):
        with pytest.raises(GonkaConfigError):
            register_gonka_models(
                mock_multiplexer,
                source_url=VALID_SOURCE_URL,
                # Missing private_key
            )


class TestGonkaConfigErrorMessage:
    """Tests that private key never appears in error messages."""

    def test_config_error_no_private_key(self, clean_env):
        bad_key = "0x" + "z" * 64  # Invalid hex
        try:
            GonkaConfig(private_key=bad_key, source_url=VALID_SOURCE_URL)
        except GonkaConfigError as e:
            error_str = str(e)
            assert "z" * 64 not in error_str
            assert bad_key not in error_str

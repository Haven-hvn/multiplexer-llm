"""
GonkaClientFactory - Creates OpenAI-compatible clients for Gonka endpoints.

This module provides the GonkaClientFactory class, which creates OpenAI-compatible
client instances configured with:
- ECDSA request signing (secp256k1)
- Required Gonka headers (X-Requester-Address, X-Timestamp)
- Proper base URL configuration

The factory uses composition with the existing gonka-openai package to handle
the cryptographic signing, keeping the multiplexer signing-agnostic.
"""

import sys
import hashlib
import json
import logging
import time
from typing import Any, Optional, Protocol, runtime_checkable

# Initialize base values for hybrid timestamp generation (same as gonka-openai)
_wall_base = time.time_ns()
_perf_base = time.perf_counter_ns()


def _hybrid_timestamp_ns() -> int:
    """
    Generate a hybrid timestamp in nanoseconds.

    Combines wall clock time with performance counter for monotonicity.
    """
    return _wall_base + (time.perf_counter_ns() - _perf_base)


from .types import GonkaParticipant
from .exceptions import GonkaClientError

logger = logging.getLogger(__name__)


def _check_gonka_dependencies() -> None:
    """
    Check that Gonka dependencies are installed.

    Raises:
        ImportError: If required packages are missing.
    """
    missing = []
    try:
        import bech32  # noqa: F401
    except ImportError:
        missing.append("bech32")
    try:
        import ecdsa  # noqa: F401
    except ImportError:
        missing.append("ecdsa")

    if missing:
        raise ImportError(
            f"Gonka support requires additional packages: {', '.join(missing)}. "
            f"Install with: pip install multiplexer-llm[gonka]"
        )


def _derive_gonka_address(private_key_hex: str) -> str:
    """
    Derive a Gonka address from a private key.

    This function derives the bech32-encoded Gonka address from an ECDSA
    private key, following the same algorithm as gonka-openai.

    Args:
        private_key_hex: Private key in hex format (with or without 0x prefix).

    Returns:
        The derived Gonka address (e.g., "gonka1abc...").

    Raises:
        GonkaClientError: If address derivation fails.
    """
    try:
        from ecdsa import SigningKey, SECP256k1
        import bech32

        # Remove 0x prefix if present
        private_key_clean = (
            private_key_hex[2:] if private_key_hex.startswith("0x") else private_key_hex
        )

        # Convert hex string to bytes
        private_key_bytes = bytes.fromhex(private_key_clean)

        # Create signing key using ecdsa library
        signing_key = SigningKey.from_string(private_key_bytes, curve=SECP256k1)

        # Get the verifying key (public key)
        verifying_key = signing_key.get_verifying_key()

        # Get uncompressed public key point (64 bytes: 32 for x, 32 for y)
        pubkey_uncompressed = verifying_key.to_string()

        # Compress the public key (33 bytes)
        # Format: 0x02 if y is even, 0x03 if y is odd, followed by x coordinate
        x_bytes = pubkey_uncompressed[:32]
        y_bytes = pubkey_uncompressed[32:]
        y_int = int.from_bytes(y_bytes, byteorder="big")
        prefix = b"\x02" if y_int % 2 == 0 else b"\x03"
        pubkey = prefix + x_bytes

        # Create SHA256 hash of the public key
        sha = hashlib.sha256(pubkey).digest()

        # Take RIPEMD160 hash of the SHA256 hash
        ripemd = hashlib.new("ripemd160")
        ripemd.update(sha)
        address_bytes = ripemd.digest()

        # Convert to 5-bit words for bech32 encoding
        five_bit_words = bech32.convertbits(address_bytes, 8, 5)
        if five_bit_words is None:
            raise ValueError("Error converting address bytes to 5-bit words")

        # Use 'gonka' prefix
        address = bech32.bech32_encode("gonka", five_bit_words)

        return address
    except Exception as e:
        # Never expose private key in error messages
        raise GonkaClientError(
            "Failed to derive Gonka address from private key", cause=e
        )


def _create_signature(
    body: Any, private_key_hex: str, timestamp: int, transfer_address: str
) -> str:
    """
    Sign a request body with a private key using ECDSA (secp256k1).

    Args:
        body: The request body to sign (bytes).
        private_key_hex: Private key in hex format (with or without 0x prefix).
        timestamp: Timestamp in nanoseconds.
        transfer_address: The transfer address to include in signature.

    Returns:
        The signature as a base64-encoded string.

    Raises:
        GonkaClientError: If signing fails.
    """
    import base64
    from ecdsa import SigningKey, SECP256k1

    try:
        # Remove 0x prefix if present
        private_key_clean = (
            private_key_hex[2:] if private_key_hex.startswith("0x") else private_key_hex
        )

        # Create a signing key using ecdsa
        signing_key = SigningKey.from_string(
            bytes.fromhex(private_key_clean), curve=SECP256k1
        )

        # Custom encoder that handles low-S normalization
        def encode_with_low_s(
            sig_r: int, sig_s: int, order: int
        ) -> bytes:
            # Apply low-s value normalization for signature malleability
            if sig_s > order // 2:
                sig_s = order - sig_s
            # Pack r and s into a byte string
            r_bytes = sig_r.to_bytes(32, byteorder="big")
            s_bytes = sig_s.to_bytes(32, byteorder="big")
            return r_bytes + s_bytes

        # Convert body to bytes if needed
        if isinstance(body, dict):
            payload_bytes = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            payload_bytes = body.encode("utf-8")
        elif isinstance(body, bytes):
            payload_bytes = body
        else:
            payload_bytes = bytes(body)

        # Sign hash of payload instead of raw payload
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()

        # Build signature input: hash + timestamp + transfer_address
        signature_input = payload_hash + str(timestamp) + transfer_address
        signature_bytes = signature_input.encode("utf-8")

        # Sign with deterministic ECDSA
        signature = signing_key.sign_deterministic(
            signature_bytes,
            hashfunc=hashlib.sha256,
            sigencode=lambda r, s, order: encode_with_low_s(r, s, order),
        )

        return base64.b64encode(signature).decode("utf-8")
    except Exception as e:
        # Never expose private key in error messages
        raise GonkaClientError("Failed to sign request", cause=e)


def _validate_private_key(private_key: str) -> None:
    """
    Validate that a private key is properly formatted.

    Args:
        private_key: The private key to validate.

    Raises:
        GonkaClientError: If the private key is invalid.
    """
    if not private_key:
        raise GonkaClientError("Private key is required and cannot be empty")

    # Remove 0x prefix for validation
    key_hex = private_key[2:] if private_key.startswith("0x") else private_key

    # Check length (32 bytes = 64 hex chars)
    if len(key_hex) != 64:
        raise GonkaClientError(
            "Invalid private key format: expected 64 hex characters (32 bytes)"
        )

    # Check it's valid hex
    try:
        bytes.fromhex(key_hex)
    except ValueError:
        raise GonkaClientError("Invalid private key format: not valid hexadecimal")


class GonkaClientFactory:
    """
    Factory for creating OpenAI-compatible clients configured for Gonka endpoints.

    This factory creates clients that automatically sign all requests with ECDSA
    and include the required Gonka headers. Each client is bound to a specific
    participant's endpoint.

    The factory caches the derived requester address for efficiency, as address
    derivation only needs to happen once per private key.

    Thread Safety:
        This class is thread-safe. Multiple threads can safely call create_client()
        concurrently.

    Example:
        >>> factory = GonkaClientFactory(private_key="0x...")
        >>> participant = GonkaParticipant(
        ...     address="gonka1abc...",
        ...     inference_url="https://node.example.com",
        ...     weight=100
        ... )
        >>> client = factory.create_client(participant)
        >>> # Use client with multiplexer
        >>> multiplexer.add_model(client, weight=participant.weight)

    Attributes:
        requester_address: The derived Gonka address for the private key (read-only).
    """

    def __init__(self, private_key: str) -> None:
        """
        Initialize the factory with a private key.

        Args:
            private_key: ECDSA private key for signing requests. Can be with or
                without '0x' prefix. The key is used to derive the requester
                address and sign all outgoing requests.

        Raises:
            GonkaClientError: If the private key is invalid or address derivation fails.
            ImportError: If Gonka dependencies are not installed.
        """
        # Check dependencies first
        _check_gonka_dependencies()

        # Validate private key format (never log the key)
        _validate_private_key(private_key)

        # Store private key (never expose in logs or errors)
        self._private_key = private_key

        # Derive address once at construction time
        self._requester_address = _derive_gonka_address(private_key)

        logger.debug(
            "GonkaClientFactory initialized with requester address: %s",
            self._requester_address,
        )

    @property
    def requester_address(self) -> str:
        """
        Get the derived Gonka address for the private key.

        Returns:
            The bech32-encoded Gonka address derived from the private key.
        """
        return self._requester_address

    @property
    def private_key(self) -> str:
        """
        Get the private key (read-only).

        Note: This property exists for protocol compliance. Use with caution
        and never log or expose the returned value.

        Returns:
            The private key used for signing.
        """
        return self._private_key

    def create_client(self, participant: GonkaParticipant) -> Any:
        """
        Create an OpenAI-compatible client for a Gonka participant.

        The returned client is configured to:
        - Send requests to the participant's inference URL
        - Sign all requests with ECDSA using the factory's private key
        - Include X-Requester-Address header with the derived address
        - Include X-Timestamp header with nanosecond timestamp
        - Use the participant's address as the transfer address

        Args:
            participant: The Gonka participant to create a client for.

        Returns:
            An OpenAI-compatible client ready for use with the multiplexer.

        Raises:
            GonkaClientError: If client creation fails.

        Example:
            >>> client = factory.create_client(participant)
            >>> response = await client.chat.completions.create(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     model="llama-3.1-70b"
            ... )
        """
        try:
            from openai import OpenAI
            import httpx

            # Validate participant
            if not participant.address:
                raise GonkaClientError(
                    "Participant address is required",
                    participant_address=participant.address,
                )

            # Get base URL with /v1 suffix
            base_url = participant.base_url

            # Create custom HTTP client with signing interceptor
            http_client = self._create_signing_http_client(
                transfer_address=participant.address,
                is_async=False,
            )

            # Create OpenAI client with custom HTTP client
            client = OpenAI(
                base_url=base_url,
                api_key="gonka",  # Gonka uses signatures, not API keys
                http_client=http_client,
            )

            logger.debug(
                "Created client for participant %s at %s",
                participant.address,
                base_url,
            )

            return client

        except GonkaClientError:
            raise
        except Exception as e:
            raise GonkaClientError(
                f"Failed to create client for participant",
                participant_address=participant.address,
                cause=e,
            )

    def create_async_client(self, participant: GonkaParticipant) -> Any:
        """
        Create an async OpenAI-compatible client for a Gonka participant.

        Same as create_client() but returns an AsyncOpenAI instance suitable
        for use with async/await patterns.

        Args:
            participant: The Gonka participant to create a client for.

        Returns:
            An async OpenAI-compatible client ready for use with the multiplexer.

        Raises:
            GonkaClientError: If client creation fails.

        Example:
            >>> client = factory.create_async_client(participant)
            >>> response = await client.chat.completions.create(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     model="llama-3.1-70b"
            ... )
        """
        try:
            from openai import AsyncOpenAI
            import httpx

            # Validate participant
            if not participant.address:
                raise GonkaClientError(
                    "Participant address is required",
                    participant_address=participant.address,
                )

            # Get base URL with /v1 suffix
            base_url = participant.base_url

            # Create custom async HTTP client with signing interceptor
            http_client = self._create_signing_http_client(
                transfer_address=participant.address,
                is_async=True,
            )

            # Create AsyncOpenAI client with custom HTTP client
            client = AsyncOpenAI(
                base_url=base_url,
                api_key="gonka",  # Gonka uses signatures, not API keys
                http_client=http_client,
            )

            logger.debug(
                "Created async client for participant %s at %s",
                participant.address,
                base_url,
            )

            return client

        except GonkaClientError:
            raise
        except Exception as e:
            raise GonkaClientError(
                f"Failed to create async client for participant",
                participant_address=participant.address,
                cause=e,
            )

    def _create_signing_http_client(
        self,
        transfer_address: str,
        is_async: bool = False,
    ) -> Any:
        """
        Create an HTTP client that signs all requests.

        Args:
            transfer_address: The Gonka address to use for transfers.
            is_async: Whether to create an async client.

        Returns:
            An httpx.Client or httpx.AsyncClient with signing interceptor.
        """
        import httpx

        # Capture instance variables for closure
        private_key = self._private_key
        requester_address = self._requester_address

        if is_async:
            # Create async client with event hooks
            client = httpx.AsyncClient()

            async def sign_request(request: httpx.Request) -> None:
                """Add signing headers to async request."""
                timestamp = _hybrid_timestamp_ns()
                request.headers["X-Requester-Address"] = requester_address
                request.headers["X-Timestamp"] = str(timestamp)

                # Sign the request body
                body = request.content
                signature = _create_signature(
                    body, private_key, timestamp, transfer_address
                )
                request.headers["Authorization"] = signature

            client.event_hooks["request"].append(sign_request)
            return client
        else:
            # Create sync client with event hooks
            client = httpx.Client()

            def sign_request(request: httpx.Request) -> None:
                """Add signing headers to sync request."""
                timestamp = _hybrid_timestamp_ns()
                request.headers["X-Requester-Address"] = requester_address
                request.headers["X-Timestamp"] = str(timestamp)

                # Sign the request body
                body = request.content
                signature = _create_signature(
                    body, private_key, timestamp, transfer_address
                )
                request.headers["Authorization"] = signature

            client.event_hooks["request"].append(sign_request)
            return client

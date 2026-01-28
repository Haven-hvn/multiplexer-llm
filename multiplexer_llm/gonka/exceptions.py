"""
Gonka-specific exceptions for the multiplexer integration.

This module defines the exception hierarchy for Gonka-related errors.
All exceptions inherit from GonkaError, which itself inherits from
MultiplexerError for consistency with the main package.
"""

from typing import Optional

from ..exceptions import MultiplexerError


class GonkaError(MultiplexerError):
    """
    Base exception for all Gonka-related errors.

    All Gonka-specific exceptions inherit from this class, allowing
    callers to catch all Gonka errors with a single except clause.

    Attributes:
        message: Human-readable error description.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class GonkaClientError(GonkaError):
    """
    Raised when client creation or configuration fails.

    This exception is raised when the GonkaClientFactory fails to create
    a client, typically due to invalid configuration or missing dependencies.

    Important: This exception never includes the private key in its message
    or attributes to prevent accidental exposure in logs.

    Attributes:
        participant_address: The Gonka address of the participant for which
            client creation failed, if available.
        cause: The underlying exception that caused the failure, if any.
    """

    def __init__(
        self,
        message: str,
        participant_address: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.participant_address = participant_address
        self.cause = cause

    def __str__(self) -> str:
        parts = [self.message]
        if self.participant_address:
            parts.append(f" (participant: {self.participant_address})")
        if self.cause:
            parts.append(f" - caused by: {type(self.cause).__name__}: {self.cause}")
        return "".join(parts)


class GonkaDiscoveryError(GonkaError):
    """
    Raised when endpoint discovery fails.

    This exception indicates that the system could not fetch participants
    from the Gonka network, typically due to network issues or invalid
    source URL.

    Attributes:
        source_url: The URL that was used for discovery.
        status_code: HTTP status code if available.
        cause: The underlying exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        source_url: Optional[str] = None,
        status_code: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.source_url = source_url
        self.status_code = status_code
        self.cause = cause


class GonkaProofVerificationError(GonkaError):
    """
    Raised when ICS23 proof verification fails.

    This exception indicates that the cryptographic proof provided by
    the Gonka network could not be verified against the expected hash.

    Attributes:
        expected_hash: The hash that was expected.
        computed_hash: The hash that was actually computed.
        proof_type: The type of proof that failed verification.
    """

    def __init__(
        self,
        message: str,
        expected_hash: Optional[bytes] = None,
        computed_hash: Optional[bytes] = None,
        proof_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.expected_hash = expected_hash
        self.computed_hash = computed_hash
        self.proof_type = proof_type


class GonkaNoParticipantsError(GonkaError):
    """
    Raised when discovery succeeds but returns no participants.

    This exception indicates that the Gonka network was reachable and
    responded successfully, but no active participants were found for
    the requested epoch.

    Attributes:
        epoch: The epoch that was queried (may be "current", "next", or numeric).
        source_url: The URL that was used for discovery.
    """

    def __init__(
        self,
        message: str,
        epoch: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.epoch = epoch
        self.source_url = source_url


class GonkaConfigError(GonkaError):
    """
    Raised when Gonka configuration is invalid.

    This exception is raised during configuration validation when required
    fields are missing or have invalid values. Error messages never expose
    sensitive information like private keys.

    Attributes:
        field: The name of the configuration field that failed validation.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.field = field

    def __str__(self) -> str:
        if self.field:
            return f"{self.message} (field: {self.field})"
        return self.message


class GonkaRefreshError(GonkaError):
    """
    Raised when background refresh fails.

    This exception indicates that the periodic refresh of endpoint
    information failed, though the system may continue operating with
    stale data.

    Attributes:
        last_successful_epoch: The last epoch that was successfully refreshed.
        cause: The underlying exception that caused the failure.
    """

    def __init__(
        self,
        message: str,
        last_successful_epoch: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.last_successful_epoch = last_successful_epoch
        self.cause = cause

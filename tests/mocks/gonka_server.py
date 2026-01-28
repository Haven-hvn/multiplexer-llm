"""
Mock Gonka API server for testing.

This module provides MockGonkaServer which simulates Gonka API responses
for endpoint discovery testing without requiring network access.

Usage:
    server = MockGonkaServer()
    server.set_participants([...])
    
    with server.mock_requests():
        discovery = EndpointDiscovery(source_url=server.base_url)
        participants = discovery.discover()
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock, patch
from contextlib import contextmanager


@dataclass
class CapturedRequest:
    """Represents a captured HTTP request for verification."""
    
    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[bytes]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


@dataclass
class ParticipantData:
    """Data for a mock participant."""
    
    address: str
    inference_url: str
    weight: int = 100
    models: List[str] = field(default_factory=list)
    validator_key: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "index": self.address,
            "validator_key": self.validator_key or f"pubkey_{self.address}",
            "weight": self.weight,
            "inference_url": self.inference_url,
            "models": self.models,
        }


class MockGonkaServer:
    """
    Mock Gonka API server for testing endpoint discovery.
    
    This class simulates the Gonka API by providing configurable
    responses for participant discovery requests. It captures all
    requests for verification in tests.
    
    Example:
        >>> server = MockGonkaServer()
        >>> server.add_participant("gonka1abc", "https://node.example.com", weight=100)
        >>> server.set_epoch(42)
        >>>
        >>> with server.mock_requests():
        ...     discovery = EndpointDiscovery(source_url=server.base_url)
        ...     participants = discovery.discover()
        ...     assert len(participants) == 1
        >>>
        >>> # Verify requests were made
        >>> requests = server.get_captured_requests()
        >>> assert len(requests) == 1
    
    Attributes:
        base_url: The mock base URL for the server.
        current_epoch: The current epoch number.
        participants: List of mock participants.
        request_log: Log of all captured requests.
    """
    
    def __init__(
        self,
        base_url: str = "https://api.mock-gonka.network",
        initial_epoch: int = 42,
    ):
        """
        Initialize the mock server.
        
        Args:
            base_url: Base URL for the mock server.
            initial_epoch: Initial epoch number.
        """
        self.base_url = base_url.rstrip("/")
        self.current_epoch = initial_epoch
        self.next_epoch = initial_epoch + 1
        
        self._participants: Dict[int, List[ParticipantData]] = {
            initial_epoch: [],
        }
        self._request_log: List[CapturedRequest] = []
        
        # Error simulation
        self._force_error: Optional[Exception] = None
        self._force_status_code: Optional[int] = None
        self._force_invalid_json: bool = False
        self._delay_seconds: float = 0.0
        
        # Custom response handler
        self._custom_handler: Optional[Callable[[str], Dict[str, Any]]] = None
    
    # =========================================================================
    # Participant Management
    # =========================================================================
    
    def add_participant(
        self,
        address: str,
        inference_url: str,
        weight: int = 100,
        models: Optional[List[str]] = None,
        validator_key: str = "",
        epoch: Optional[int] = None,
    ) -> "MockGonkaServer":
        """
        Add a participant to the mock server.
        
        Args:
            address: Gonka address (e.g., "gonka1abc...").
            inference_url: Inference API URL.
            weight: Stake weight.
            models: List of supported model names.
            validator_key: Validator public key.
            epoch: Epoch to add participant to (default: current).
        
        Returns:
            Self for chaining.
        """
        epoch = epoch if epoch is not None else self.current_epoch
        
        if epoch not in self._participants:
            self._participants[epoch] = []
        
        participant = ParticipantData(
            address=address,
            inference_url=inference_url,
            weight=weight,
            models=models or [],
            validator_key=validator_key,
        )
        self._participants[epoch].append(participant)
        
        return self
    
    def set_participants(
        self,
        participants: List[ParticipantData],
        epoch: Optional[int] = None,
    ) -> "MockGonkaServer":
        """
        Set all participants for an epoch.
        
        Args:
            participants: List of participant data.
            epoch: Epoch to set participants for (default: current).
        
        Returns:
            Self for chaining.
        """
        epoch = epoch if epoch is not None else self.current_epoch
        self._participants[epoch] = list(participants)
        return self
    
    def clear_participants(self, epoch: Optional[int] = None) -> "MockGonkaServer":
        """
        Clear all participants for an epoch.
        
        Args:
            epoch: Epoch to clear (default: current).
        
        Returns:
            Self for chaining.
        """
        epoch = epoch if epoch is not None else self.current_epoch
        self._participants[epoch] = []
        return self
    
    def get_participants(self, epoch: Optional[int] = None) -> List[ParticipantData]:
        """
        Get participants for an epoch.
        
        Args:
            epoch: Epoch to get participants for (default: current).
        
        Returns:
            List of participants.
        """
        epoch = epoch if epoch is not None else self.current_epoch
        return self._participants.get(epoch, [])
    
    # =========================================================================
    # Epoch Management
    # =========================================================================
    
    def set_epoch(self, epoch: int) -> "MockGonkaServer":
        """
        Set the current epoch.
        
        Args:
            epoch: New current epoch number.
        
        Returns:
            Self for chaining.
        """
        self.current_epoch = epoch
        self.next_epoch = epoch + 1
        if epoch not in self._participants:
            self._participants[epoch] = []
        return self
    
    def advance_epoch(self) -> "MockGonkaServer":
        """
        Advance to the next epoch.
        
        Returns:
            Self for chaining.
        """
        self.current_epoch = self.next_epoch
        self.next_epoch = self.current_epoch + 1
        if self.current_epoch not in self._participants:
            self._participants[self.current_epoch] = []
        return self
    
    # =========================================================================
    # Error Simulation
    # =========================================================================
    
    def force_error(self, error: Exception) -> "MockGonkaServer":
        """
        Force all requests to raise an error.
        
        Args:
            error: Exception to raise.
        
        Returns:
            Self for chaining.
        """
        self._force_error = error
        return self
    
    def force_status_code(self, status_code: int) -> "MockGonkaServer":
        """
        Force all requests to return a specific status code.
        
        Args:
            status_code: HTTP status code to return.
        
        Returns:
            Self for chaining.
        """
        self._force_status_code = status_code
        return self
    
    def force_invalid_json(self, invalid: bool = True) -> "MockGonkaServer":
        """
        Force responses to return invalid JSON.
        
        Args:
            invalid: Whether to return invalid JSON.
        
        Returns:
            Self for chaining.
        """
        self._force_invalid_json = invalid
        return self
    
    def set_delay(self, seconds: float) -> "MockGonkaServer":
        """
        Set response delay (for timeout testing).
        
        Args:
            seconds: Delay in seconds.
        
        Returns:
            Self for chaining.
        """
        self._delay_seconds = seconds
        return self
    
    def reset_errors(self) -> "MockGonkaServer":
        """
        Reset all error simulation.
        
        Returns:
            Self for chaining.
        """
        self._force_error = None
        self._force_status_code = None
        self._force_invalid_json = False
        self._delay_seconds = 0.0
        return self
    
    # =========================================================================
    # Request Handling
    # =========================================================================
    
    def handle_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> Mock:
        """
        Handle a mock HTTP request.
        
        Args:
            method: HTTP method.
            url: Request URL.
            headers: Request headers.
            body: Request body.
        
        Returns:
            Mock response object.
        
        Raises:
            Configured exception if force_error is set.
        """
        import time
        
        # Log the request
        self._request_log.append(CapturedRequest(
            method=method,
            url=url,
            headers=headers or {},
            body=body,
        ))
        
        # Apply delay
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        
        # Check for forced error
        if self._force_error is not None:
            raise self._force_error
        
        # Create response
        response = Mock()
        
        # Apply forced status code
        if self._force_status_code is not None:
            response.status_code = self._force_status_code
            return response
        
        # Normal response
        response.status_code = 200
        
        # Apply invalid JSON
        if self._force_invalid_json:
            response.json.side_effect = json.JSONDecodeError("error", "", 0)
            return response
        
        # Build response data
        response_data = self._build_response(url)
        response.json.return_value = response_data
        
        return response
    
    def _build_response(self, url: str) -> Dict[str, Any]:
        """Build response data based on URL."""
        # Use custom handler if set
        if self._custom_handler is not None:
            return self._custom_handler(url)
        
        # Parse epoch from URL
        epoch = self._parse_epoch_from_url(url)
        
        # Get participants for epoch
        participants = self._participants.get(epoch, [])
        
        return {
            "active_participants": {
                "participants": [p.to_dict() for p in participants],
                "epoch_id": epoch,
            }
        }
    
    def _parse_epoch_from_url(self, url: str) -> int:
        """Parse epoch from request URL."""
        if "/current/" in url:
            return self.current_epoch
        elif "/next/" in url:
            return self.next_epoch
        else:
            # Try to extract numeric epoch
            import re
            match = re.search(r"/epochs/(\d+)/", url)
            if match:
                return int(match.group(1))
        return self.current_epoch
    
    # =========================================================================
    # Request Verification
    # =========================================================================
    
    def get_captured_requests(self) -> List[CapturedRequest]:
        """
        Get all captured requests.
        
        Returns:
            List of captured requests.
        """
        return list(self._request_log)
    
    def get_last_request(self) -> Optional[CapturedRequest]:
        """
        Get the most recent request.
        
        Returns:
            Last captured request or None.
        """
        if self._request_log:
            return self._request_log[-1]
        return None
    
    def clear_request_log(self) -> "MockGonkaServer":
        """
        Clear the request log.
        
        Returns:
            Self for chaining.
        """
        self._request_log.clear()
        return self
    
    def assert_request_count(self, expected: int) -> None:
        """
        Assert the number of requests made.
        
        Args:
            expected: Expected number of requests.
        
        Raises:
            AssertionError: If count doesn't match.
        """
        actual = len(self._request_log)
        assert actual == expected, f"Expected {expected} requests, got {actual}"
    
    def assert_last_request_url_contains(self, substring: str) -> None:
        """
        Assert the last request URL contains a substring.
        
        Args:
            substring: Expected substring.
        
        Raises:
            AssertionError: If not found or no requests.
        """
        last = self.get_last_request()
        assert last is not None, "No requests captured"
        assert substring in last.url, f"'{substring}' not in '{last.url}'"
    
    # =========================================================================
    # Context Managers
    # =========================================================================
    
    @contextmanager
    def mock_requests(self):
        """
        Context manager to mock HTTP requests.
        
        Usage:
            with server.mock_requests():
                discovery = EndpointDiscovery(source_url=server.base_url)
                participants = discovery.discover()
        """
        def mock_get(url, **kwargs):
            return self.handle_request(
                method="GET",
                url=url,
                headers=kwargs.get("headers"),
            )
        
        with patch("requests.get", side_effect=mock_get):
            yield self
    
    @contextmanager
    def mock_responses(self):
        """
        Context manager using responses library (if available).
        
        This provides more realistic HTTP mocking.
        """
        try:
            import responses
            
            @responses.activate
            def _inner():
                # Register callback
                responses.add_callback(
                    responses.GET,
                    url=f"{self.base_url}/v1/epochs/current/participants",
                    callback=lambda req: self._responses_callback(req),
                    content_type="application/json",
                )
                responses.add_callback(
                    responses.GET,
                    url=f"{self.base_url}/v1/epochs/next/participants",
                    callback=lambda req: self._responses_callback(req),
                    content_type="application/json",
                )
                yield self
            
            yield from _inner()
        except ImportError:
            # Fall back to simple mock
            with self.mock_requests():
                yield self
    
    def _responses_callback(self, request):
        """Callback for responses library."""
        try:
            import responses
            
            self._request_log.append(CapturedRequest(
                method=request.method,
                url=request.url,
                headers=dict(request.headers),
                body=request.body,
            ))
            
            if self._force_error:
                raise self._force_error
            
            status = self._force_status_code or 200
            body = self._build_response(request.url)
            
            return (status, {}, json.dumps(body))
        except Exception as e:
            return (500, {}, json.dumps({"error": str(e)}))


# ===========================================================================
# Factory Functions
# ===========================================================================


def create_mock_server_with_participants(
    count: int = 3,
    epoch: int = 42,
    base_weight: int = 100,
) -> MockGonkaServer:
    """
    Create a mock server with generated participants.
    
    Args:
        count: Number of participants to generate.
        epoch: Epoch number.
        base_weight: Base weight (incremented per participant).
    
    Returns:
        Configured MockGonkaServer.
    """
    server = MockGonkaServer(initial_epoch=epoch)
    
    for i in range(count):
        server.add_participant(
            address=f"gonka1test{i:04d}",
            inference_url=f"https://node{i:04d}.example.com",
            weight=base_weight + i * 10,
            models=["llama-3.1-70b"] if i % 2 == 0 else [],
        )
    
    return server


def create_mock_server_from_fixture(fixture_path: str) -> MockGonkaServer:
    """
    Create a mock server from a fixture file.
    
    Args:
        fixture_path: Path to JSON fixture file.
    
    Returns:
        Configured MockGonkaServer.
    """
    with open(fixture_path) as f:
        data = json.load(f)
    
    active = data.get("active_participants", {})
    epoch = active.get("epoch_id", 42)
    
    server = MockGonkaServer(initial_epoch=epoch)
    
    for p in active.get("participants", []):
        server.add_participant(
            address=p.get("index", ""),
            inference_url=p.get("inference_url", ""),
            weight=p.get("weight", 100),
            models=p.get("models", []),
            validator_key=p.get("validator_key", ""),
        )
    
    return server

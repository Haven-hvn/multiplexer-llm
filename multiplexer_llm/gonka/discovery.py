"""
Endpoint Discovery for the Gonka integration module.

This module provides the EndpointDiscovery class which fetches and validates
Gonka network participants from blockchain state.
"""

from __future__ import annotations

import binascii
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import requests

from .exceptions import (
    GonkaDiscoveryError,
    GonkaNoParticipantsError,
    GonkaProofVerificationError,
)
from .types import GonkaParticipant

logger = logging.getLogger(__name__)


class EndpointDiscovery:
    """
    Discovers Gonka network participants from blockchain state.

    This class fetches participant data from a Gonka source URL, optionally
    verifies ICS23 proofs against the blockchain app hash, and parses
    participant data into strongly-typed GonkaParticipant objects.

    The discovery service is stateless - each call reflects the current
    blockchain state without caching.

    Attributes:
        source_url: Base URL for participant discovery.
        verify_proofs: Whether to verify ICS23 proofs.
        timeout: HTTP request timeout in seconds.

    Example:
        >>> discovery = EndpointDiscovery(
        ...     source_url="https://api.gonka.network",
        ...     verify_proofs=False,
        ...     timeout=30.0
        ... )
        >>> participants = discovery.discover()
        >>> for p in participants:
        ...     print(f"{p.address}: {p.inference_url}")
    """

    def __init__(
        self,
        source_url: str,
        verify_proofs: bool = False,
        timeout: float = 30.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize the endpoint discovery service.

        Args:
            source_url: Base URL for participant discovery (e.g., "https://api.gonka.network").
                        Also supports file:// URLs for testing/development.
            verify_proofs: Whether to verify ICS23 proofs against the app hash.
                           Defaults to False for faster development iteration.
            timeout: HTTP request timeout in seconds. Defaults to 30.0.
            retry_count: Number of retry attempts for transient network failures. Defaults to 3.
            retry_delay: Delay between retry attempts in seconds. Defaults to 1.0.

        Raises:
            ValueError: If source_url is empty.
        """
        if not source_url:
            raise ValueError("source_url cannot be empty")

        self._source_url = self._normalize_url(source_url)
        self._verify_proofs = verify_proofs
        self._timeout = timeout
        self._retry_count = retry_count
        self._retry_delay = retry_delay

    @property
    def source_url(self) -> str:
        """Get the source URL for discovery."""
        return self._source_url

    @property
    def verify_proofs(self) -> bool:
        """Get whether proof verification is enabled."""
        return self._verify_proofs

    @property
    def timeout(self) -> float:
        """Get the HTTP timeout in seconds."""
        return self._timeout

    def discover(self) -> List[GonkaParticipant]:
        """
        Fetch current epoch's participants.

        This is a convenience method that calls discover_for_epoch("current").

        Returns:
            List of GonkaParticipant objects for the current epoch.

        Raises:
            GonkaDiscoveryError: On network failure or invalid response.
            GonkaProofVerificationError: If verification enabled and fails.
            GonkaNoParticipantsError: If no participants found.
        """
        return self.discover_for_epoch("current")

    def discover_for_epoch(self, epoch: str) -> List[GonkaParticipant]:
        """
        Fetch participants for a specific epoch.

        Args:
            epoch: Epoch specifier. Can be "current", "next", or a numeric epoch ID.

        Returns:
            List of GonkaParticipant objects for the specified epoch.

        Raises:
            GonkaDiscoveryError: On network failure or invalid response.
            GonkaProofVerificationError: If verification enabled and fails.
            GonkaNoParticipantsError: If no participants found.
        """
        epoch_str = self._normalize_epoch(epoch)
        logger.info("Discovering participants for epoch '%s' from %s", epoch_str, self._source_url)

        payload = self._fetch_payload(epoch_str)
        return self._process_payload(payload, epoch_str)

    async def async_discover(self) -> List[GonkaParticipant]:
        """
        Async version: Fetch current epoch's participants.

        Returns:
            List of GonkaParticipant objects for the current epoch.

        Raises:
            GonkaDiscoveryError: On network failure or invalid response.
            GonkaProofVerificationError: If verification enabled and fails.
            GonkaNoParticipantsError: If no participants found.
        """
        return await self.async_discover_for_epoch("current")

    async def async_discover_for_epoch(self, epoch: str) -> List[GonkaParticipant]:
        """
        Async version: Fetch participants for a specific epoch.

        Args:
            epoch: Epoch specifier. Can be "current", "next", or a numeric epoch ID.

        Returns:
            List of GonkaParticipant objects for the specified epoch.

        Raises:
            GonkaDiscoveryError: On network failure or invalid response.
            GonkaProofVerificationError: If verification enabled and fails.
            GonkaNoParticipantsError: If no participants found.
        """
        epoch_str = self._normalize_epoch(epoch)
        logger.info("Async discovering participants for epoch '%s' from %s", epoch_str, self._source_url)

        payload = await self._async_fetch_payload(epoch_str)
        return self._process_payload(payload, epoch_str)

    def get_current_epoch(self) -> int:
        """
        Get the current epoch number.

        Returns:
            The current epoch ID as an integer.

        Raises:
            GonkaDiscoveryError: On network failure or invalid response.
        """
        payload = self._fetch_payload("current")
        active_participants = payload.get("active_participants", {})
        epoch_id = active_participants.get("epoch_id")
        if epoch_id is None:
            raise GonkaDiscoveryError(
                "Response missing epoch_id",
                source_url=self._source_url,
            )
        return int(epoch_id)

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL by removing trailing slashes."""
        return url.rstrip("/")

    @staticmethod
    def _normalize_epoch(epoch: str) -> str:
        """Normalize epoch specifier."""
        if not epoch:
            raise ValueError("epoch cannot be empty")
        # Accept "current", "next", or numeric string
        return str(epoch).strip()

    def _build_url(self, epoch: str) -> str:
        """Build the full URL for participant discovery."""
        return f"{self._source_url}/v1/epochs/{epoch}/participants"

    def _fetch_payload(self, epoch: str) -> Dict[str, Any]:
        """
        Fetch payload from source URL (sync).

        Supports both HTTP URLs and file:// URLs for testing.
        """
        if self._source_url.startswith("file://"):
            return self._fetch_from_file(epoch)
        return self._fetch_from_http(epoch)

    async def _async_fetch_payload(self, epoch: str) -> Dict[str, Any]:
        """
        Fetch payload from source URL (async).

        Supports both HTTP URLs and file:// URLs for testing.
        """
        if self._source_url.startswith("file://"):
            return self._fetch_from_file(epoch)
        return await self._async_fetch_from_http(epoch)

    def _fetch_from_file(self, epoch: str) -> Dict[str, Any]:
        """Fetch payload from a local file (for testing)."""
        file_path = self._source_url[len("file://"):]
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise GonkaDiscoveryError(
                f"File not found: {file_path}",
                source_url=self._source_url,
                cause=e,
            ) from e
        except json.JSONDecodeError as e:
            raise GonkaDiscoveryError(
                f"Invalid JSON in file: {file_path}",
                source_url=self._source_url,
                cause=e,
            ) from e

    def _fetch_from_http(self, epoch: str) -> Dict[str, Any]:
        """Fetch payload via HTTP GET with retries."""
        url = self._build_url(epoch)
        last_error: Optional[Exception] = None

        for attempt in range(self._retry_count):
            try:
                logger.debug("HTTP GET %s (attempt %d/%d)", url, attempt + 1, self._retry_count)
                response = requests.get(
                    url,
                    headers={"Content-Type": "application/json"},
                    timeout=self._timeout,
                )

                if response.status_code != 200:
                    raise GonkaDiscoveryError(
                        f"HTTP {response.status_code} from {url}",
                        source_url=self._source_url,
                        status_code=response.status_code,
                    )

                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    raise GonkaDiscoveryError(
                        f"Invalid JSON response from {url}",
                        source_url=self._source_url,
                        cause=e,
                    ) from e

            except requests.Timeout as e:
                last_error = e
                logger.warning("Request timeout (attempt %d/%d): %s", attempt + 1, self._retry_count, url)
                if attempt < self._retry_count - 1:
                    import time
                    time.sleep(self._retry_delay)

            except requests.ConnectionError as e:
                last_error = e
                logger.warning("Connection error (attempt %d/%d): %s", attempt + 1, self._retry_count, url)
                if attempt < self._retry_count - 1:
                    import time
                    time.sleep(self._retry_delay)

            except GonkaDiscoveryError:
                # Re-raise discovery errors without retry
                raise

        raise GonkaDiscoveryError(
            f"Failed to fetch participants after {self._retry_count} attempts",
            source_url=self._source_url,
            cause=last_error,
        )

    async def _async_fetch_from_http(self, epoch: str) -> Dict[str, Any]:
        """Fetch payload via async HTTP GET with retries."""
        try:
            import httpx
        except ImportError:
            # Fallback to sync if httpx not available
            logger.warning("httpx not installed, falling back to sync HTTP")
            return self._fetch_from_http(epoch)

        url = self._build_url(epoch)
        last_error: Optional[Exception] = None

        for attempt in range(self._retry_count):
            try:
                logger.debug("Async HTTP GET %s (attempt %d/%d)", url, attempt + 1, self._retry_count)
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(
                        url,
                        headers={"Content-Type": "application/json"},
                    )

                if response.status_code != 200:
                    raise GonkaDiscoveryError(
                        f"HTTP {response.status_code} from {url}",
                        source_url=self._source_url,
                        status_code=response.status_code,
                    )

                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    raise GonkaDiscoveryError(
                        f"Invalid JSON response from {url}",
                        source_url=self._source_url,
                        cause=e,
                    ) from e

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning("Async request timeout (attempt %d/%d): %s", attempt + 1, self._retry_count, url)
                if attempt < self._retry_count - 1:
                    import asyncio
                    await asyncio.sleep(self._retry_delay)

            except httpx.ConnectError as e:
                last_error = e
                logger.warning("Async connection error (attempt %d/%d): %s", attempt + 1, self._retry_count, url)
                if attempt < self._retry_count - 1:
                    import asyncio
                    await asyncio.sleep(self._retry_delay)

            except GonkaDiscoveryError:
                # Re-raise discovery errors without retry
                raise

        raise GonkaDiscoveryError(
            f"Failed to fetch participants after {self._retry_count} attempts",
            source_url=self._source_url,
            cause=last_error,
        )

    def _process_payload(self, payload: Dict[str, Any], epoch: str) -> List[GonkaParticipant]:
        """Process the API response payload into GonkaParticipant objects."""
        # Extract active_participants
        active_participants = payload.get("active_participants", {})
        if not active_participants:
            raise GonkaDiscoveryError(
                "Response missing active_participants",
                source_url=self._source_url,
            )

        # Verify proofs if enabled
        if self._verify_proofs:
            self._verify_payload_proofs(payload)

        # Get epoch_id from response
        epoch_id = active_participants.get("epoch_id", 0)

        # Parse participants
        participants_data = active_participants.get("participants", [])
        if not participants_data:
            raise GonkaNoParticipantsError(
                "No participants found in response",
                epoch=epoch,
                source_url=self._source_url,
            )

        participants: List[GonkaParticipant] = []
        for p_data in participants_data:
            participant = self._parse_participant(p_data, epoch_id)
            if participant:
                participants.append(participant)

        if not participants:
            raise GonkaNoParticipantsError(
                "All participants in response were invalid",
                epoch=epoch,
                source_url=self._source_url,
            )

        logger.info("Discovered %d participants for epoch %s", len(participants), epoch)
        return participants

    def _parse_participant(
        self, data: Dict[str, Any], epoch_id: int
    ) -> Optional[GonkaParticipant]:
        """Parse a single participant from response data."""
        if not data:
            return None

        try:
            # Required fields
            address = data.get("index")
            inference_url = data.get("inference_url")

            if not address:
                logger.warning("Participant missing 'index' field, skipping")
                return None

            if not inference_url:
                logger.warning("Participant %s missing 'inference_url' field, skipping", address)
                return None

            # Normalize inference URL
            inference_url = self._normalize_inference_url(inference_url)

            # Optional fields with defaults
            weight = data.get("weight", 1)
            if weight is None or weight <= 0:
                weight = 1
                logger.warning("Participant %s has invalid weight, using default 1", address)

            models = data.get("models", []) or []
            validator_key = data.get("validator_key", "") or ""

            return GonkaParticipant(
                address=address,
                inference_url=inference_url,
                weight=int(weight),
                models=list(models),
                validator_key=validator_key,
                epoch_id=int(epoch_id),
            )

        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse participant data: %s", e)
            return None

    @staticmethod
    def _normalize_inference_url(url: str) -> str:
        """Normalize inference URL to not include /v1 suffix (GonkaParticipant handles it)."""
        if not url:
            return url
        url = url.rstrip("/")
        # Remove /v1 suffix if present, as GonkaParticipant.base_url adds it
        if url.endswith("/v1"):
            url = url[:-3]
        return url

    def _verify_payload_proofs(self, payload: Dict[str, Any]) -> None:
        """Verify ICS23 proofs in the payload against the app hash."""
        try:
            # Extract required data for verification
            active_participants_bytes = payload.get("active_participants_bytes", "")
            if not active_participants_bytes:
                raise GonkaProofVerificationError(
                    "Response missing active_participants_bytes for verification",
                    proof_type="active_participants",
                )

            # Decode participants bytes from hex
            try:
                participants_bytes = binascii.unhexlify(active_participants_bytes)
            except binascii.Error as e:
                raise GonkaProofVerificationError(
                    f"Failed to decode active_participants_bytes: {e}",
                    proof_type="active_participants",
                ) from e

            # Extract proof operations
            proof_ops = self._extract_proof_ops(payload)
            if not proof_ops or len(proof_ops) != 2:
                raise GonkaProofVerificationError(
                    f"Expected 2 proof ops, got {len(proof_ops) if proof_ops else 0}",
                    proof_type="proof_ops",
                )

            # Extract app hash
            app_hash = self._extract_app_hash(payload)
            if not app_hash:
                raise GonkaProofVerificationError(
                    "Response missing block.app_hash for verification",
                    proof_type="app_hash",
                )

            # Verify the proofs
            self._verify_iavl_proof(proof_ops, participants_bytes, app_hash)

            logger.debug("ICS23 proof verification successful")

        except GonkaProofVerificationError:
            raise
        except Exception as e:
            raise GonkaProofVerificationError(
                f"Proof verification failed: {e}",
                proof_type="unknown",
            ) from e

    def _extract_proof_ops(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract proof operations from payload."""
        import base64

        def find_ops(obj: Any) -> List[Dict[str, Any]]:
            """Recursively find proof ops in the payload."""
            if isinstance(obj, list):
                # Check if this looks like a list of ops
                if obj and all(isinstance(e, dict) for e in obj):
                    if any("type" in e and "data" in e for e in obj):
                        return obj
                # Search in list elements
                for e in obj:
                    result = find_ops(e)
                    if result:
                        return result
            elif isinstance(obj, dict):
                # Try common keys
                for key in ("ops", "Ops", "proof_ops", "proofOps", "ProofOps"):
                    if key in obj:
                        val = obj[key]
                        if isinstance(val, list) and val:
                            if any("type" in e and "data" in e for e in val if isinstance(e, dict)):
                                return val
                # Search in dict values
                for v in obj.values():
                    result = find_ops(v)
                    if result:
                        return result
            return []

        # First try direct path
        proof_ops_src = payload.get("proof_ops") or payload.get("proofOps") or {}
        ops = find_ops(proof_ops_src) or find_ops(payload)

        # Convert to standardized format
        result = []
        for op in ops:
            t = op.get("type") or op.get("Type", "")
            k_raw = op.get("key") or op.get("Key", "")
            d_raw = op.get("data") or op.get("Data", "")

            # Decode base64 strings
            if isinstance(k_raw, str):
                key_bytes = base64.b64decode(k_raw)
            else:
                key_bytes = bytes(k_raw or b"")

            if isinstance(d_raw, str):
                data_bytes = base64.b64decode(d_raw)
            else:
                data_bytes = bytes(d_raw or b"")

            result.append({
                "type": t,
                "key": key_bytes,
                "data": data_bytes,
            })

        return result

    def _extract_app_hash(self, payload: Dict[str, Any]) -> bytes:
        """Extract app hash from payload."""
        import base64

        block = payload.get("block", {})
        if not block:
            return b""

        # Handle both dict and nested structures
        app_hash = None
        if isinstance(block, dict):
            app_hash = block.get("app_hash")
            if app_hash is None and isinstance(block.get("header"), dict):
                app_hash = block["header"].get("app_hash")

        if not app_hash:
            return b""

        if isinstance(app_hash, str):
            # Try hex first (common for app_hash)
            v = app_hash.strip()
            is_hex = all(c in "0123456789abcdefABCDEF" for c in v) and (len(v) % 2 == 0) and len(v) > 0
            if is_hex:
                try:
                    return bytes.fromhex(v)
                except ValueError:
                    pass
            # Try base64
            try:
                return base64.b64decode(v)
            except Exception:
                return b""

        return bytes(app_hash) if app_hash else b""

    def _verify_iavl_proof(
        self,
        proof_ops: List[Dict[str, Any]],
        value: bytes,
        expected_app_hash: bytes,
    ) -> None:
        """Verify IAVL proof against app hash."""
        try:
            # Import protobuf module for ICS23
            from gonka_openai.ics23.cosmos.ics23.v1 import proofs_pb2
        except ImportError:
            logger.warning("ICS23 protobuf not available, skipping proof verification")
            return

        if len(proof_ops) != 2:
            raise GonkaProofVerificationError(
                f"Expected 2 proof ops, got {len(proof_ops)}",
                proof_type="proof_ops",
            )

        # 1. Verify IAVL op
        iavl_op = proof_ops[0]
        if iavl_op["type"] != "ics23:iavl":
            raise GonkaProofVerificationError(
                f"Unexpected first proof op type: {iavl_op['type']}",
                proof_type="iavl",
            )

        if not iavl_op["data"]:
            raise GonkaProofVerificationError(
                "IAVL proof op has empty data",
                proof_type="iavl",
            )

        # Parse IAVL commitment proof
        iavl_cp = proofs_pb2.CommitmentProof.FromString(iavl_op["data"])
        iavl_exist = self._extract_existence(iavl_cp)

        # Verify key/value binding
        if bytes(iavl_exist.key) != bytes(iavl_op["key"]):
            raise GonkaProofVerificationError(
                "IAVL proof key mismatch",
                proof_type="iavl",
            )
        if bytes(iavl_exist.value) != bytes(value):
            raise GonkaProofVerificationError(
                "IAVL proof value mismatch",
                proof_type="iavl",
                expected_hash=value,
                computed_hash=bytes(iavl_exist.value),
            )

        # Calculate store root
        store_root = self._calculate_root_from_existence(iavl_exist, proofs_pb2)

        # 2. Verify Simple (multistore) op
        simple_op = proof_ops[1]
        if simple_op["type"] != "ics23:simple":
            raise GonkaProofVerificationError(
                f"Unexpected second proof op type: {simple_op['type']}",
                proof_type="simple",
            )

        if not simple_op["data"]:
            raise GonkaProofVerificationError(
                "Simple proof op has empty data",
                proof_type="simple",
            )

        # Parse simple commitment proof
        simple_cp = proofs_pb2.CommitmentProof.FromString(simple_op["data"])
        simple_exist = self._extract_existence(simple_cp)

        # Verify the simple proof binds store_root to app_hash
        if bytes(simple_exist.key) != bytes(simple_op["key"]):
            raise GonkaProofVerificationError(
                "Simple proof key mismatch",
                proof_type="simple",
            )
        if bytes(simple_exist.value) != bytes(store_root):
            raise GonkaProofVerificationError(
                "Simple proof value (store root) mismatch",
                proof_type="simple",
            )

        # Calculate app hash
        computed_app_hash = self._calculate_root_from_existence(simple_exist, proofs_pb2)

        if bytes(computed_app_hash) != bytes(expected_app_hash):
            raise GonkaProofVerificationError(
                "Computed app hash does not match expected",
                proof_type="app_hash",
                expected_hash=expected_app_hash,
                computed_hash=computed_app_hash,
            )

    @staticmethod
    def _extract_existence(cp: Any) -> Any:
        """Extract existence proof from commitment proof."""
        which = cp.WhichOneof("proof")
        if which != "exist":
            raise GonkaProofVerificationError(
                f"Unsupported commitment proof type: {which}",
                proof_type="existence",
            )
        return cp.exist

    @staticmethod
    def _calculate_root_from_existence(ex: Any, proofs_pb2: Any) -> bytes:
        """Calculate root hash from existence proof."""

        def hash_bytes(op: int, data: bytes) -> bytes:
            if op == proofs_pb2.SHA256:
                return hashlib.sha256(data).digest()
            if op == proofs_pb2.NO_HASH:
                return data
            raise RuntimeError(f"Unsupported hash op: {op}")

        def encode_varint(value: int) -> bytes:
            out = bytearray()
            v = int(value)
            while True:
                b = v & 0x7F
                v >>= 7
                if v:
                    out.append(b | 0x80)
                else:
                    out.append(b)
                    break
            return bytes(out)

        def len_prefix(op: int, data: bytes) -> bytes:
            if op == proofs_pb2.NO_PREFIX:
                return b""
            if op == proofs_pb2.VAR_PROTO:
                return encode_varint(len(data))
            raise RuntimeError(f"Unsupported length op: {op}")

        def apply_leaf_op(leaf: Any, key: bytes, value: bytes) -> bytes:
            hkey = key if leaf.prehash_key == proofs_pb2.NO_HASH else hash_bytes(leaf.prehash_key, key)
            hval = value if leaf.prehash_value == proofs_pb2.NO_HASH else hash_bytes(leaf.prehash_value, value)
            payload = bytes(leaf.prefix or b"")
            payload += len_prefix(leaf.length, hkey) + hkey
            payload += len_prefix(leaf.length, hval) + hval
            return hash_bytes(leaf.hash, payload)

        def apply_inner_op(inner: Any, child: bytes) -> bytes:
            payload = bytes(inner.prefix or b"") + child + bytes(inner.suffix or b"")
            return hash_bytes(inner.hash, payload)

        cur = apply_leaf_op(ex.leaf, bytes(ex.key), bytes(ex.value))
        for step in ex.path:
            cur = apply_inner_op(step, cur)
        return cur

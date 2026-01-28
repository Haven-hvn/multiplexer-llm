"""
Mock objects for Gonka integration testing.

This package provides mock implementations for testing:
- MockGonkaServer: Simulates Gonka API HTTP responses
- MockMultiplexer: Tracks model registration calls
"""

from .gonka_server import MockGonkaServer

__all__ = ["MockGonkaServer"]

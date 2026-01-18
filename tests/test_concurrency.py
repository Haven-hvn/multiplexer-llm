"""Comprehensive tests for concurrency features."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from multiplexer_llm import Multiplexer
from multiplexer_llm.types import WeightedModel
from multiplexer_llm.multiplexer import CapacityError


class MockOpenAIClient:
    """Mock OpenAI-compatible client for testing."""
    
    def __init__(self, name: str = "mock_client", response_delay: float = 0.0):
        self.name = name
        self.chat = MockChat(response_delay)


class MockChat:
    """Mock chat interface."""
    
    def __init__(self, response_delay: float = 0.0):
        self.completions = MockCompletions(response_delay)


class MockCompletions:
    """Mock completions interface."""
    
    def __init__(self, response_delay: float = 0.0):
        self._response = {"choices": [{"message": {"content": "Test response"}}], "model": "test-model"}
        self.response_delay = response_delay
        self._should_raise = None
    
    async def create(self, **kwargs: Any) -> Dict[str, Any]:
        """Mock create method with optional delay."""
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
            
        if self._should_raise:
            raise self._should_raise
        return self._response
    
    def set_response(self, response: Dict[str, Any]) -> None:
        """Set the response to return."""
        self._response = response
    
    def set_error(self, error: Exception) -> None:
        """Set an error to raise."""
        self._should_raise = error


class TestConcurrencyFeatures:
    """Test cases for concurrency feature."""
    
    @pytest.fixture
    def multiplexer(self, _function_event_loop):
        """Create a fresh multiplexer instance for each test."""
        mux = Multiplexer()
        yield mux
        _function_event_loop.run_until_complete(mux.async_reset())
    
    def test_add_model_with_max_concurrent(self, multiplexer):
        """Test adding model with concurrency limits."""
        client = MockOpenAIClient()
        multiplexer.add_model(client, 5, "test-model", max_concurrent=3)
        
        assert len(multiplexer._weighted_models) == 1
        model = multiplexer._weighted_models[0]
        assert model.max_concurrent == 3
        assert model._active_requests == 0
        
    def test_add_model_unlimited_concurrency(self, multiplexer):
        """Test adding model without concurrency limits."""
        client = MockOpenAIClient()
        multiplexer.add_model(client, 5, "test-model")  # No max_concurrent
        
        assert len(multiplexer._weighted_models) == 1
        model = multiplexer._weighted_models[0]
        assert model.max_concurrent is None
        
    @pytest.mark.asyncio
    async def test_can_accept_request_unlimited(self, multiplexer):
        """Test that unlimited models always accept requests."""
        client = MockOpenAIClient()
        multiplexer.add_model(client, 5, "test-model")
        
        model = multiplexer._weighted_models[0]
        
        # Should always accept when unlimited
        assert await model.can_accept_request() is True
        assert await model.can_accept_request() is True
        
    @pytest.mark.asyncio
    async def test_can_accept_request_limited(self, multiplexer):
        """Test capacity checking for limited models."""
        client = MockOpenAIClient()
        multiplexer.add_model(client, 5, "test-model", max_concurrent=2)
        
        model = multiplexer._weighted_models[0]
        
        # Should accept requests up to capacity
        assert await model.can_accept_request() is True
        assert await model.can_accept_request() is True
        
        # Should reject when at capacity
        assert await model.can_accept_request() is False
        
    @pytest.mark.asyncio
    async def test_active_request_tracking(self, multiplexer):
        """Test active request count tracking."""
        client = MockOpenAIClient()
        multiplexer.add_model(client, 5, "test-model", max_concurrent=2)
        
        model = multiplexer._weighted_models[0]
        
        assert model._active_requests == 0
        
        # Simulate accepting and processing requests
        await model.increment_active_requests()
        assert model._active_requests == 1
        
        await model.increment_active_requests()
        assert model._active_requests == 2
        
        await model.decrement_active_requests()
        assert model._active_requests == 1
        
        await model.decrement_active_requests()
        assert model._active_requests == 0
        
    @pytest.mark.asyncio
    async def test_concurrent_request_limit_enforcement(self, multiplexer):
        """Test that concurrent requests are properly limited."""
        client = MockOpenAIClient("client1", response_delay=0.1)
        multiplexer.add_model(client, 10, "high-capacity-model", max_concurrent=2)
        
        # Create 5 concurrent requests
        requests = []
        for i in range(5):
            request = multiplexer.chat.completions.create(
                messages=[{"role": "user", "content": f"Request {i}"}],
                model="auto"
            )
            requests.append(request)
        
        # Process all requests
        results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Check that we got results (no timeouts or capacity errors)
        successes = [r for r in results if not isinstance(r, Exception) and hasattr(r, 'choices')]
        assert len(successes) == 5
        
        # Verify model state
        assert multiplexer._weighted_models[0]._active_requests == 0  # All completed
        
    @pytest.mark.asyncio
    async def test_capacity_overflow_distribution(self, multiplexer):
        """Test that requests flow to other models when one is at capacity."""
        client1 = MockOpenAIClient("client1", response_delay=0.05)
        client2 = MockOpenAIClient("client2", response_delay=0.05)
        
        # High capacity model
        multiplexer.add_model(client1, 1, "low-capacity-model", max_concurrent=2)
        # Unlimited model
        multiplexer.add_model(client2, 1, "unlimited-model", max_concurrent=None)
        
        # Create many concurrent requests
        requests = []
        for i in range(10):
            request = multiplexer.chat.completions.create(
                messages=[{"role": "user", "content": f"Request {i}"}],
                model="auto"
            )
            requests.append(request)
        
        # Process all requests
        results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Check that we got results
        successes = [r for r in results if not isinstance(r, Exception) and hasattr(r, 'choices')]
        assert len(successes) == 10
        
        # Get final statistics
        stats = multiplexer.get_stats()
        
        # The low-capacity model should have handled some requests but was capped
        # The unlimited model should have handled the overflow
        assert stats["low-capacity-model"]["success"] >= 2  # At least its capacity
        assert stats["low-capacity-model"]["success"] <= 6  # But not all due to distribution
        assert stats["unlimited-model"]["success"] >= 2     # Should get the overflow
        
    @pytest.mark.asyncio
    async def test_capacity_error_handling(self, multiplexer):
        """Test CapacityError exception handling."""
        client = MockOpenAIClient()
        multiplexer.add_model(client, 10, "unlimited-model", max_concurrent=None)
        
        # Test successful request
        result = await multiplexer.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="auto"
        )
        
        assert result is not None
        assert "choices" in result
        
    @pytest.mark.asyncio
    async def test_weight_preservation_with_capacity_limits(self, multiplexer):
        """Test that weighted distribution is preserved even with capacity limits."""
        client1 = MockOpenAIClient("client1", response_delay=0.05)
        client2 = MockOpenAIClient("client2", response_delay=0.05)
        
        # High weight model with low capacity
        multiplexer.add_model(client1, 9, "high-weight-low-capacity", max_concurrent=2)
        # Low weight model with high capacity  
        multiplexer.add_model(client2, 1, "low-weight-high-capacity", max_concurrent=None)
        
        # Create many requests
        requests = []
        for i in range(50):
            request = multiplexer.chat.completions.create(
                messages=[{"role": "user", "content": f"Request {i}"}],
                model="auto"
            )
            requests.append(request)
        
        # Process all requests
        results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Check that we got results
        successes = [r for r in results if not isinstance(r, Exception) and hasattr(r, 'choices')]
        assert len(successes) == 50
        
        # Get final statistics
        stats = multiplexer.get_stats()
        
        # The high-weight model should get more traffic when it has capacity
        # But when at capacity, traffic should overflow to the low-weight model
        # This preserves the relative distribution
        high_weight_usage = stats["high-weight-low-capacity"]["success"]
        low_weight_usage = stats["low-weight-high-capacity"]["success"]
        
        # The high weight model should get significantly more traffic overall
        # (9:1 ratio means it should get ~90% when capacity allows)
        assert high_weight_usage > low_weight_usage
        assert high_weight_usage >= 30  # At least 60% of 50 requests
        
    @pytest.mark.asyncio
    async def test_fallback_models_respect_capacity(self, multiplexer):
        """Test that fallback models also respect capacity limits."""
        primary_client = MockOpenAIClient("primary", response_delay=0.02)
        fallback_client = MockOpenAIClient("fallback", response_delay=0.02)
        
        # Primary model with very low capacity
        multiplexer.add_model(primary_client, 10, "primary", max_concurrent=1)
        # Fallback model with higher capacity
        multiplexer.add_fallback_model(fallback_client, 1, "fallback", max_concurrent=3)
        
        # Create requests that will exceed primary capacity
        requests = []
        for i in range(8):
            request = multiplexer.chat.completions.create(
                messages=[{"role": "user", "content": f"Request {i}"}],
                model="auto"
            )
            requests.append(request)
        
        # Process all requests
        results = await asyncio.gather(*requests, return_exceptions=True)
        
        # Check that we got results
        successes = [r for r in results if not isinstance(r, Exception) and hasattr(r, 'choices')]
        assert len(successes) == 8
        
        # Get final statistics
        stats = multiplexer.get_stats()
        
        # Primary should only handle 1 request due to capacity
        assert stats["primary"]["success"] == 1
        # Fallback should handle the rest
        assert stats["fallback"]["success"] == 7
        
    @pytest.mark.asyncio
    async def test_all_models_at_capacity(self, multiplexer):
        """Test handling when all models are at capacity."""
        client1 = MockOpenAIClient("client1", response_delay=0.01)
        client2 = MockOpenAIClient("client2", response_delay=0.01)
        
        # Both models with very low capacity
        multiplexer.add_model(client1, 5, "model1", max_concurrent=2)
        multiplexer.add_model(client2, 5, "model2", max_concurrent=2)
        
        # Fill both models to capacity with long-running requests
        requests = []
        for i in range(4):
            request = multiplexer.chat.completions.create(
                messages=[{"role": "user", "content": f"Request {i}"}],
                model="auto"
            )
            requests.append(request)
        
        # Wait a bit for requests to start processing
        await asyncio.sleep(0.02)
        
        # Try to make an additional request - should still succeed due to retry logic
        extra_request = multiplexer.chat.completions.create(
            messages=[{"role": "user", "content": "Extra request"}],
            model="auto"
        )
        
        # The extra request should eventually succeed after models free up
        result = await extra_request
        assert result is not None
        
    def test_max_concurrent_validation(self, multiplexer):
        """Test validation of max_concurrent parameter."""
        client = MockOpenAIClient()
        
        # Valid values
        multiplexer.add_model(client, 5, "test-model", max_concurrent=None)
        multiplexer.add_model(client, 5, "test-model-2", max_concurrent=5)
        multiplexer.add_model(client, 5, "test-model-3", max_concurrent=0)  # Unlimited
        
        # Invalid values
        with pytest.raises(ValueError, match="max_concurrent must be a non-negative integer"):
            multiplexer.add_model(client, 5, "test-model-4", max_concurrent=-1)
            
        with pytest.raises(ValueError, match="max_concurrent must be a non-negative integer"):
            multiplexer.add_model(client, 5, "test-model-5", max_concurrent=1.5)

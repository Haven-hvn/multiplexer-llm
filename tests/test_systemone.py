"""Tests for the System One (/v1/systemone) surface of the Multiplexer."""

import pytest
from typing import Any, Dict, List, Optional

from multiplexer_llm import Multiplexer
from multiplexer_llm.exceptions import (
    ModelSelectionError,
    MultiplexerError,
    RateLimitError,
)


class MockPost:
    """Mock of the generic ``post`` verb on an OpenAI SDK client."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._response: Dict[str, Any] = {
            "model": "systemone-model",
            "answers": {
                "greeting": {"type": "noul", "noul": 0.97},
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }
        self._should_raise: Optional[Exception] = None

    async def __call__(self, path: str, *, body: Any = None, cast_to: Any = dict, **kwargs: Any) -> Any:
        """Record the call and replay the queued response."""
        self.calls.append({"path": path, "body": body, "cast_to": cast_to})
        if self._should_raise:
            raise self._should_raise
        return dict(self._response)

    def set_response(self, response: Dict[str, Any]) -> None:
        """Set the response to return."""
        self._response = response

    def set_error(self, error: Exception) -> None:
        """Set an error to raise."""
        self._should_raise = error


class MockSystemOneClient:
    """Mock OpenAI SDK client exposing the generic ``post`` verb."""

    def __init__(self, name: str = "mock_client"):
        self.name = name
        self.post = MockPost()


class MockStatusError(Exception):
    """Mock error carrying an HTTP status code (OpenAI SDK style)."""

    def __init__(self, message: str = "Provider error", status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code


_QUESTIONS: Dict[str, Any] = {
    "greeting": {"type": "noul", "instructions": "Is this a greeting?"},
}


class TestSystemOne:
    """Test cases for the System One interface."""

    @pytest.fixture
    def multiplexer(self, _function_event_loop):
        """Create a fresh multiplexer instance for each test."""
        mux = Multiplexer()
        yield mux
        _function_event_loop.run_until_complete(mux.async_reset())

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client with a generic ``post`` verb."""
        return MockSystemOneClient()

    def test_init(self, multiplexer):
        """Test that the systemone interface is exposed."""
        assert multiplexer.systemone is not None

    @pytest.mark.asyncio
    async def test_create_posts_canonical_payload(self, multiplexer, mock_client):
        """Test that create() posts the Jev /v1/systemone envelope."""
        multiplexer.add_model(mock_client, 5, "systemone-model")

        result = await multiplexer.systemone.create(
            state="refund please",
            questions=_QUESTIONS,
        )

        call = mock_client.post.calls[0]
        assert call["path"] == "/systemone"
        assert call["cast_to"] is dict
        assert call["body"]["state"] == "refund please"
        assert call["body"]["questions"]["greeting"]["type"] == "noul"
        assert result["answers"]["greeting"]["noul"] == 0.97
        assert result["usage"]["input_tokens"] == 12
        assert multiplexer._weighted_models[0].success_count == 1

    @pytest.mark.asyncio
    async def test_model_is_overridden_by_endpoint_name(self, multiplexer, mock_client):
        """Test that the body model is the selected endpoint's registered name."""
        multiplexer.add_model(mock_client, 5, "tachyone-en")

        result = await multiplexer.systemone.create(
            state="hello",
            questions=_QUESTIONS,
            model="placeholder",  # Will be overridden by selected model
        )

        assert mock_client.post.calls[0]["body"]["model"] == "tachyone-en"
        # The response model is stamped with the selected model name
        assert result["model"] == "tachyone-en"

    @pytest.mark.asyncio
    async def test_extra_body_fields_forwarded(self, multiplexer, mock_client):
        """Test that extra kwargs (e.g. choice_head) ride along in the body."""
        multiplexer.add_model(mock_client, 5, "systemone-model")

        await multiplexer.systemone.create(
            state="hello",
            questions=_QUESTIONS,
            choice_head="support",
        )

        assert mock_client.post.calls[0]["body"]["choice_head"] == "support"

    @pytest.mark.asyncio
    async def test_custom_path(self, multiplexer, mock_client):
        """Test that the endpoint path can be overridden."""
        multiplexer.add_model(mock_client, 5, "systemone-model")

        await multiplexer.systemone.create(
            state="hello",
            questions=_QUESTIONS,
            path="/v1/systemone",
        )

        assert mock_client.post.calls[0]["path"] == "/v1/systemone"

    @pytest.mark.asyncio
    async def test_rate_limit_fallback(self, multiplexer):
        """Test failover to the fallback endpoint on 429."""
        primary_client = MockSystemOneClient("primary")
        fallback_client = MockSystemOneClient("fallback")
        primary_client.post.set_error(MockStatusError(status_code=429))

        multiplexer.add_model(primary_client, 5, "primary-model")
        multiplexer.add_fallback_model(fallback_client, 3, "fallback-model")

        result = await multiplexer.systemone.create(
            state="hello",
            questions=_QUESTIONS,
        )

        assert result["answers"]["greeting"]["noul"] == 0.97
        assert result["model"] == "fallback-model"

        stats = multiplexer.get_stats()
        assert stats["primary-model"]["rateLimited"] == 1
        assert stats["fallback-model"]["success"] == 1

    @pytest.mark.asyncio
    async def test_overloaded_fallback(self, multiplexer):
        """Test failover to the fallback endpoint on 529 (overloaded)."""
        primary_client = MockSystemOneClient("primary")
        fallback_client = MockSystemOneClient("fallback")
        primary_client.post.set_error(MockStatusError(status_code=529))

        multiplexer.add_model(primary_client, 5, "primary-model")
        multiplexer.add_fallback_model(fallback_client, 3, "fallback-model")

        result = await multiplexer.systemone.create(
            state="hello",
            questions=_QUESTIONS,
        )

        assert result["answers"]["greeting"]["noul"] == 0.97
        stats = multiplexer.get_stats()
        assert stats["primary-model"]["rateLimited"] == 1
        assert stats["fallback-model"]["success"] == 1

    @pytest.mark.asyncio
    async def test_persistent_error_fallback(self, multiplexer):
        """Test failover to the fallback endpoint on 5xx (persistent error)."""
        primary_client = MockSystemOneClient("primary")
        fallback_client = MockSystemOneClient("fallback")
        primary_client.post.set_error(MockStatusError(status_code=503))

        multiplexer.add_model(primary_client, 5, "primary-model")
        multiplexer.add_fallback_model(fallback_client, 3, "fallback-model")

        result = await multiplexer.systemone.create(
            state="hello",
            questions=_QUESTIONS,
        )

        assert result["answers"]["greeting"]["noul"] == 0.97
        stats = multiplexer.get_stats()
        assert stats["primary-model"]["failed"] == 1
        assert stats["fallback-model"]["success"] == 1

    @pytest.mark.asyncio
    async def test_missing_answers_propagates(self, multiplexer):
        """Test that an invalid response fails hard instead of failing over."""
        primary_client = MockSystemOneClient("primary")
        fallback_client = MockSystemOneClient("fallback")
        # A response without "answers" is not a valid /v1/systemone body
        primary_client.post.set_response({"model": "primary-model"})

        multiplexer.add_model(primary_client, 5, "primary-model")
        multiplexer.add_fallback_model(fallback_client, 3, "fallback-model")

        with pytest.raises(MultiplexerError, match="Unexpected error for model primary-model"):
            await multiplexer.systemone.create(
                state="hello",
                questions=_QUESTIONS,
            )

        # Non-retryable errors propagate immediately; the fallback is untouched
        assert fallback_client.post.calls == []

    @pytest.mark.asyncio
    async def test_no_models_error(self, multiplexer):
        """Test error when no models are available."""
        with pytest.raises(ModelSelectionError, match="No models available in the multiplexer"):
            await multiplexer.systemone.create(
                state="hello",
                questions=_QUESTIONS,
            )

    @pytest.mark.asyncio
    async def test_all_models_rate_limited(self, multiplexer):
        """Test error when all models are rate limited."""
        primary_client = MockSystemOneClient("primary")
        fallback_client = MockSystemOneClient("fallback")
        primary_client.post.set_error(MockStatusError(status_code=429))
        fallback_client.post.set_error(MockStatusError(status_code=429))

        multiplexer.add_model(primary_client, 5, "primary-model")
        multiplexer.add_fallback_model(fallback_client, 3, "fallback-model")

        with pytest.raises(RateLimitError) as exc_info:
            await multiplexer.systemone.create(
                state="hello",
                questions=_QUESTIONS,
            )
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_systemone_result_type_exported(self):
        """Sanity check that the result alias is re-exported."""
        from multiplexer_llm import SystemOneResult

        assert SystemOneResult is not None

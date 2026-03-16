"""Unit tests for OpenAIClient error classification and cancellation."""

from src.core.client import OpenAIClient


class TestClassifyOpenAIError:
    """Test error classification logic."""

    def _make_client(self) -> OpenAIClient:
        """Create a minimal client for testing classification."""
        return OpenAIClient(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )

    def test_region_restriction(self):
        """Region restriction errors produce helpful message."""
        client = self._make_client()
        result = client.classify_openai_error(
            "unsupported_country_region_territory detected"
        )
        assert "not available in your region" in result

    def test_invalid_api_key(self):
        """Invalid API key errors produce helpful message."""
        client = self._make_client()
        result = client.classify_openai_error("invalid_api_key")
        assert "Invalid API key" in result

    def test_rate_limit(self):
        """Rate limit errors produce helpful message."""
        client = self._make_client()
        result = client.classify_openai_error("rate_limit exceeded")
        assert "Rate limit" in result

    def test_model_not_found(self):
        """Model not found errors produce helpful message."""
        client = self._make_client()
        result = client.classify_openai_error("model gpt-5 not found")
        assert "Model not found" in result

    def test_billing_issue(self):
        """Billing errors produce helpful message."""
        client = self._make_client()
        result = client.classify_openai_error("billing issue detected")
        assert "Billing issue" in result

    def test_unknown_error_returns_original(self):
        """Unknown errors return the original message."""
        client = self._make_client()
        result = client.classify_openai_error("something completely different")
        assert result == "something completely different"


class TestCancelRequest:
    """Test request cancellation tracking."""

    def test_cancel_nonexistent_request(self):
        """Cancelling a non-existent request returns False."""
        client = OpenAIClient(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        assert client.cancel_request("nonexistent-id") is False

    def test_cancel_tracked_request(self):
        """Cancelling a tracked request returns True and sets the event."""
        import asyncio

        client = OpenAIClient(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        event = asyncio.Event()
        client.active_requests["req-1"] = event
        assert client.cancel_request("req-1") is True
        assert event.is_set()

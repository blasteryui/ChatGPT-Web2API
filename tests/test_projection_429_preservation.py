"""Preserve production projection-429 backoff while integrating strict receipts."""
from unittest.mock import AsyncMock
import pytest
from chatgpt_web2api.backend_client import BackendClient
from chatgpt_web2api.cdp_driver import projection_tail_delay

def test_projection_429_reads_exponentially_back_off_and_cap():
    streak = 0
    waits = []
    for _ in range(7):
        delay, streak, limited = projection_tail_delay("fetch_failed", {"rate_limited": True}, streak)
        assert limited
        waits.append(delay)
    assert waits == [2.0, 4.0, 8.0, 16.0, 16.0, 16.0, 16.0]

@pytest.mark.parametrize("status,diagnostic", [("matched", {}), ("not_ready", {}), ("fetch_failed", {"rate_limited": False})])
def test_non_rate_limit_resets_projection_backoff(status, diagnostic):
    assert projection_tail_delay(status, diagnostic, 4) == (0.5, 0, False)

@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_fetch_text_for_turn", "_fetch_end_turn_for_turn"])
async def test_projection_429_is_distinguished_from_transport_failure(method):
    client = BackendClient(object())
    client._fetch_recent_conversation_projection = AsyncMock(side_effect=RuntimeError("projection HTTP 429 for original-conversation"))
    kwargs = {"had_non_text_content": False} if method.endswith("end_turn_for_turn") else {}
    result = await getattr(client, method)("original-conversation", object(), **kwargs)
    assert result.status == "fetch_failed"
    assert result.diagnostic["rate_limited"] is True
    client._fetch_recent_conversation_projection.assert_awaited_once()
    client._fetch_recent_conversation_projection = AsyncMock(side_effect=RuntimeError("connection reset"))
    result = await getattr(client, method)("original-conversation", object(), **kwargs)
    assert result.status == "fetch_failed"
    assert result.diagnostic["rate_limited"] is False

"""Phase-1 must track the in-flight chat when a stale landing URL transitions."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatgpt_web2api.cdp_driver import StreamChunk
from chatgpt_web2api.completion_detector import CompletionDetector, DetectorBudgets
from chatgpt_web2api.config import ChatGPTConfig
from chatgpt_web2api.turn_anchor import TurnAnchor, TurnTextResult


@pytest.mark.asyncio
async def test_phase1_updates_conversation_id_after_fresh_navigation(monkeypatch):
    driver = MagicMock()
    driver._current_conv_id = None
    driver._js_strict = AsyncMock(side_effect=lambda expr: '{"text": ""}' if "innerText" in expr else "0")
    # SPA still shows the prior conversation during its initial navigation.
    driver._get_live_conversation_id_best_effort = AsyncMock(
        side_effect=lambda: "old-conversation" if clock[0] < 3 else "new-conversation"
    )
    seen = []

    async def fetch(conversation_id, anchor):
        seen.append(conversation_id)
        return TurnTextResult(
            status="matched" if conversation_id == "new-conversation" else "not_ready",
            text="correct new turn" if conversation_id == "new-conversation" else "",
        )

    driver._fetch_text_for_turn = AsyncMock(side_effect=fetch)
    clock = [0.0]
    original_sleep = asyncio.sleep

    async def tick(delay):
        clock[0] += delay
        await original_sleep(0)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(asyncio, "sleep", tick)
    chunks = [
        x async for x in CompletionDetector(driver).stream_until_complete(
            initial_count=0, timeout=12,
            turn_anchor=TurnAnchor(sent_text="unique attempt", mode="fresh_chat"),
            budgets=DetectorBudgets.from_config(ChatGPTConfig(), "auto"),
            model="auto",
        )
    ]
    assert chunks == [StreamChunk(delta="correct new turn")]
    assert driver._get_live_conversation_id_best_effort.await_count >= 3
    assert "new-conversation" in seen
    assert clock[0] < 12

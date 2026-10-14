"""Regression: reasoning-model pre-node phase must use first-content budget."""
import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from chatgpt_web2api.cdp_driver import StreamChunk
from chatgpt_web2api.completion_detector import CompletionDetector, DetectorBudgets
from chatgpt_web2api.turn_anchor import TurnAnchor, TurnTextResult


class MissingNode:
    async def __call__(self, expr):
        return '{"text": ""}' if "innerText" in expr else "0"


@pytest.mark.asyncio
async def test_reasoning_terminal_after_90s_before_assistant_dom_node(monkeypatch):
    """When a matched backend answer appears at 105s, do not abort at 90s."""
    from unittest.mock import MagicMock

    driver = MagicMock()
    driver._current_conv_id = "conv-p1"
    driver._js_strict = MissingNode()
    driver._get_live_conversation_id_best_effort = AsyncMock(return_value="conv-p1")
    now = [0.0]

    async def fetch_for_turn(conversation_id, turn_anchor):
        if now[0] >= 105.0:
            return TurnTextResult(status="matched", text="late grounded answer")
        return TurnTextResult(status="not_ready")

    driver._fetch_text_for_turn = AsyncMock(side_effect=fetch_for_turn)
    original_sleep = asyncio.sleep

    async def fast_sleep(delay):
        now[0] += delay
        await original_sleep(0)

    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    chunks = [
        chunk
        async for chunk in CompletionDetector(driver).stream_until_complete(
            initial_count=0,
            timeout=360,
            turn_anchor=TurnAnchor(sent_text="unit test", mode="fresh_chat"),
            budgets=DetectorBudgets.reasoning(),
            model="gpt-6-thinking",
        )
    ]
    assert chunks == [StreamChunk(delta="late grounded answer")]
    assert 105 <= now[0] < 125
    assert driver._fetch_text_for_turn.await_count >= 2
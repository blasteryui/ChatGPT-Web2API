"""G6 model='auto' must permit reasoning first-content without changing its identity."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatgpt_web2api.cdp_driver import StreamChunk
from chatgpt_web2api.completion_detector import CompletionDetector, DetectorBudgets, classify_model
from chatgpt_web2api.config import ChatGPTConfig
from chatgpt_web2api.turn_anchor import TurnAnchor, TurnTextResult


def test_auto_budget_waits_for_silent_reasoning_without_claiming_model_identity():
    cfg = ChatGPTConfig()
    b = DetectorBudgets.from_config(cfg, "auto")
    assert classify_model("auto") == "default"  # auto is NOT proof of GPT-6.
    assert b.first_content_timeout_seconds == cfg.detector_reasoning_first_content_timeout_seconds
    assert b.stream_idle_timeout_seconds == cfg.detector_default_stream_idle_timeout_seconds
    assert b.hard_timeout_seconds == cfg.detector_hard_timeout_seconds


def test_unknown_and_explicit_instant_retain_fast_first_content():
    cfg = ChatGPTConfig()
    for name in (None, "", "unknown-future-slug", "gpt-5-5-instant"):
        b = DetectorBudgets.from_config(cfg, name)
        assert b.first_content_timeout_seconds == cfg.detector_default_first_content_timeout_seconds


@pytest.mark.asyncio
async def test_auto_g6_style_terminal_at_105_seconds_no_assistant_node(monkeypatch):
    d = MagicMock()
    d._current_conv_id = "conv-auto"
    async def no_node(expr):
        return '{"text": ""}' if "innerText" in expr else "0"
    d._js_strict = no_node
    d._get_live_conversation_id_best_effort = AsyncMock(return_value="conv-auto")
    clock = [0.0]

    async def fetch(conversation_id, turn_anchor):
        if clock[0] >= 105:
            return TurnTextResult(status="matched", text="verified answer")
        return TurnTextResult(status="not_ready")

    d._fetch_text_for_turn = AsyncMock(side_effect=fetch)
    original_sleep = asyncio.sleep

    async def tick(delay):
        clock[0] += delay
        await original_sleep(0)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(asyncio, "sleep", tick)
    chunks = [x async for x in CompletionDetector(d).stream_until_complete(
        initial_count=0, timeout=360,
        turn_anchor=TurnAnchor(sent_text="safe test", mode="fresh_chat"),
        budgets=DetectorBudgets.from_config(ChatGPTConfig(), "auto"),
        model="auto",
    )]
    assert chunks == [StreamChunk(delta="verified answer")]
    assert 105 <= clock[0] < 125

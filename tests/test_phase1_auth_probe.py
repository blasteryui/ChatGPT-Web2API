"""Auth errors in Phase-1 conversation-ID probes must remain distinguishable."""
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from chatgpt_web2api.cdp_driver import AuthExpiredError, GenerationStuckError
from chatgpt_web2api.completion_detector import CompletionDetector
from chatgpt_web2api.turn_anchor import TurnAnchor


class Phase1IdentityAuthProbeTests(unittest.IsolatedAsyncioTestCase):
    async def run_probe(self, error):
        clock = [0.0]
        driver = MagicMock()
        driver._current_conv_id = None
        driver._js_strict = AsyncMock(
            side_effect=lambda expr: '{"text":""}' if "innerText" in expr else "0"
        )
        driver._get_live_conversation_id_best_effort = AsyncMock(side_effect=error)
        driver._fetch_text_for_turn = AsyncMock()
        real_sleep = asyncio.sleep

        async def tick(delay):
            clock[0] += delay
            await real_sleep(0)

        with patch.object(time, "monotonic", lambda: clock[0]), patch.object(asyncio, "sleep", tick):
            with self.assertRaises(type(error)) as caught:
                async for _ in CompletionDetector(driver).stream_until_complete(
                    initial_count=0, timeout=2.5,
                    turn_anchor=TurnAnchor(sent_text="unique guarded fire", mode="fresh_chat"),
                ):
                    pass
        return clock[0], caught.exception, driver

    async def test_auth_expiry_is_fast_and_preserves_error_identity(self):
        elapsed, exc, driver = await self.run_probe(AuthExpiredError("HTTP 401"))
        self.assertIsInstance(exc, AuthExpiredError)
        self.assertLess(elapsed, 2.0, "auth expiry should not wait for Phase-1 deadline")
        driver._fetch_text_for_turn.assert_not_awaited()

    async def test_transient_probe_error_fails_closed_at_deadline(self):
        clock = [0.0]
        driver = MagicMock()
        driver._current_conv_id = None
        driver._js_strict = AsyncMock(
            side_effect=lambda expr: '{"text":""}' if "innerText" in expr else "0"
        )
        driver._get_live_conversation_id_best_effort = AsyncMock(
            side_effect=RuntimeError("transient CDP read")
        )
        real_sleep = asyncio.sleep

        async def tick(delay):
            clock[0] += delay
            await real_sleep(0)

        with patch.object(time, "monotonic", lambda: clock[0]), patch.object(asyncio, "sleep", tick):
            with self.assertRaises(GenerationStuckError):
                async for _ in CompletionDetector(driver).stream_until_complete(
                    initial_count=0, timeout=2.5,
                    turn_anchor=TurnAnchor(sent_text="unique guarded fire", mode="fresh_chat"),
                ):
                    pass
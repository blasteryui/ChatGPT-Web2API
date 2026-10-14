"""No-Send Phase-1 terminal reconciliation at timeout boundaries."""
import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from chatgpt_web2api.cdp_driver import GenerationStuckError, StreamChunk
from chatgpt_web2api.completion_detector import CompletionDetector, DetectorBudgets
from chatgpt_web2api.turn_anchor import TurnAnchor, TurnTextResult


class Phase1FinalReconciliationTests(unittest.IsolatedAsyncioTestCase):
    async def check_boundary(self, deadline, first_content_budget, terminal_status="matched"):
        asyncio.get_running_loop().set_debug(False)
        clock = [0.0]
        driver = MagicMock()
        driver._current_conv_id = None
        driver._js_strict = AsyncMock(side_effect=lambda expr: '{"text":""}' if 'innerText' in expr else '0')
        driver._get_live_conversation_id_best_effort = AsyncMock(return_value='new-conversation')

        async def backend(conversation_id, anchor):
            self.assertEqual(conversation_id, 'new-conversation')
            return TurnTextResult(
                status=terminal_status if clock[0] >= 2.5 else 'not_ready',
                text='verified final answer' if clock[0] >= 2.5 else '',
            )

        driver._fetch_text_for_turn = AsyncMock(side_effect=backend)
        original_sleep = asyncio.sleep

        async def tick(delay):
            clock[0] += delay
            await original_sleep(0)

        budgets = DetectorBudgets(first_content_timeout_seconds=first_content_budget,
                                  stream_idle_timeout_seconds=90, hard_timeout_seconds=900)
        with patch.object(time, 'monotonic', lambda: clock[0]), patch.object(asyncio, 'sleep', tick):
            chunks = [chunk async for chunk in CompletionDetector(driver).stream_until_complete(
                initial_count=0, timeout=deadline,
                turn_anchor=TurnAnchor(sent_text='unique phase1 attempt', mode='fresh_chat'),
                budgets=budgets, model='auto')]
        self.assertEqual(chunks, [StreamChunk(delta='verified final answer')])
        self.assertGreaterEqual(driver._fetch_text_for_turn.await_count, 1)

    async def test_deadline_final_reconcile(self):
        await self.check_boundary(2.2, 300)

    async def test_stall_final_reconcile(self):
        await self.check_boundary(10, 2)

    async def test_deadline_unmatched_does_not_fabricate_completion(self):
        with self.assertRaises(GenerationStuckError):
            await self.check_boundary(2.2, 300, terminal_status='not_ready')

    async def test_stall_fetch_failed_does_not_fabricate_completion(self):
        with self.assertRaises(GenerationStuckError):
            await self.check_boundary(10, 2, terminal_status='fetch_failed')

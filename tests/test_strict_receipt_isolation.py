"""Request-local strict receipts are never taken from a concurrent driver's global state.

Pure local generator/API doubles: no CDP, provider Send, sockets, or browser.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from chatgpt_web2api.api_server import APIServer, _validated_terminal_receipt
from chatgpt_web2api.cdp_driver import StreamChunk
from chatgpt_web2api.provider_contract import ProviderOperationContext


PROJECT = "g-p-test-project"


def context(op):
    return ProviderOperationContext(
        operation_id=op, requested_project_id=PROJECT, observed_project_id=PROJECT,
    )


def completed(op, conv):
    return {
        "operation_id": op,
        "requested_project_id": PROJECT,
        "observed_project_id": PROJECT,
        "conversation_id": conv,
        "target_id": "target-" + op,
        "session_identity": "session-1",
        "response_status": "COMPLETE",
        "effect_certainty": "CONFIRMED_COMPLETE",
        "turn_anchor": {"latest_user_node_id": "user-" + op},
    }


def test_terminal_receipt_checks_operation_project_conversation_anchor_and_certainty():
    a = completed("A", "conv-A")
    assert _validated_terminal_receipt(a, context("A"), "conv-A") == a
    for bad in [
        completed("B", "conv-A"),
        completed("A", "conv-B"),
        {**a, "observed_project_id": "other-project"},
        {**a, "effect_certainty": "CONFIRMED_SUBMITTED"},
        {**a, "response_status": "PREPARED"},
        {**a, "turn_anchor": None},
        None,
    ]:
        with pytest.raises(RuntimeError, match="strict provider"):
            _validated_terminal_receipt(bad, context("A"), "conv-A")


class SharedDriver:
    _current_conv_id = "conv-A"

    def __init__(self, terminal):
        self._last_provider_receipt = completed("B", "conv-B")
        self.terminal = terminal

    async def send_and_stream(self, text, **kwargs):
        assert kwargs["operation_context"].operation_id == "A"
        yield StreamChunk(delta="assistant A")
        await asyncio.sleep(0)
        # Another target's overlapping operation changed the old shared field.
        self._last_provider_receipt = completed("B", "conv-B")
        yield StreamChunk(delta="", finish_reason="stop", provider_receipt=self.terminal)
        self._current_conv_id = "conv-B"


def server(monkeypatch, receipt):
    from chatgpt_web2api.completion_detector import DetectorBudgets
    monkeypatch.setattr(DetectorBudgets, "from_config", lambda *_: object())
    srv = APIServer.__new__(APIServer)
    srv._config = SimpleNamespace(chatgpt=SimpleNamespace())
    srv._driver = SharedDriver(receipt)
    srv._last_conv_id = None
    srv._last_successful_send_at = None
    return srv


@pytest.mark.asyncio
async def test_nonstream_completion_cannot_alias_shared_receipt(monkeypatch):
    srv = server(monkeypatch, completed("A", "conv-A"))
    response = await srv._full_response(None, "auto", "fixture", 10, operation_context=context("A"))
    result = json.loads(response.body)
    assert result["provider_receipt"]["operation_id"] == "A"
    assert result["conversation_id"] == "conv-A"
    assert result["provider_receipt"]["effect_certainty"] == "CONFIRMED_COMPLETE"
    assert srv._driver._last_provider_receipt["operation_id"] == "B"


@pytest.mark.asyncio
async def test_nonstream_rejects_foreign_terminal_receipt(monkeypatch):
    srv = server(monkeypatch, completed("B", "conv-B"))
    with pytest.raises(RuntimeError, match="terminal receipt identity"):
        await srv._full_response(None, "auto", "fixture", 10, operation_context=context("A"))


@pytest.mark.asyncio
async def test_sse_foreign_receipt_fails_closed_without_stop(monkeypatch):
    from chatgpt_web2api import api_server

    srv = server(monkeypatch, completed("B", "conv-B"))
    chunks = []

    async def send_sse(_response, payload):
        chunks.append(payload)

    async def circuit():
        return None

    class FakeStream:
        headers = {}
        content_type = ""

        async def prepare(self, request):
            return None

        async def write(self, data):
            return None

        async def write_eof(self):
            return None

    async def probe(*args, **kwargs):
        return json.dumps({"text": ""})

    srv._driver._js_strict = probe
    srv._check_circuit_or_recover = circuit
    srv._send_sse = send_sse
    monkeypatch.setattr(api_server.web, "StreamResponse", FakeStream)
    await srv._stream_response(None, "auto", "fixture", 10, operation_context=context("A"))
    reasons = [c["choices"][0]["finish_reason"] for c in chunks]
    assert "stop" not in reasons
    assert "error" in reasons
    assert not any(c.get("provider_receipt", {}).get("operation_id") == "B" for c in chunks)

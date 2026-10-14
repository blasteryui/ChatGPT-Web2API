"""Strict provider requests without explicit conversation IDs must open fresh chats.

Pure driver/request doubles; no network, browser, provider Send, or host mutation.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest
import chatgpt_web2api.api_server as api

PROJECT = "g-p-6abc5b56d4488191b9b5b5224e3ef10c"


class NullMutationLock:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strict,explicit_id,expected",
    [
        (True, None, "fresh"),
        (False, None, "legacy_continue"),
        (True, "chosen-chat", "explicit_continue"),
    ],
)
async def test_strict_user_only_request_cannot_implicitly_reuse_previous_chat(
    monkeypatch, strict, explicit_id, expected
):
    monkeypatch.setattr(api, "MutationLock", NullMutationLock)
    server = api.APIServer.__new__(api.APIServer)
    server._config = api.Config.load(None)
    server._driver = MagicMock()
    server._driver._current_conv_id = "previous-chat"
    server._driver._current_model = None
    server._driver.navigate_new_chat = AsyncMock()
    server._driver.navigate_conversation = AsyncMock()
    server._driver.ensure_current_conversation = AsyncMock()
    server._driver.observe_project_id = AsyncMock(return_value=PROJECT)
    server._last_conv_id = "previous-chat"
    server._last_project_id = PROJECT
    server._parallel_tabs = False
    server._request_count = 0
    server._cdp_port = 9222
    server._last_error = None
    server._breakers = api.BreakerRegistry()
    server._check_circuit_or_recover = AsyncMock()
    server._full_response = AsyncMock(return_value=MagicMock())
    server._stream_response = AsyncMock(return_value=MagicMock())

    request = MagicMock()
    request.headers = {}
    body = {
        "messages": [{"role": "user", "content": "fresh job"}],
        "model": "auto",
        "project_id": PROJECT,
        "strict_provider": strict,
        "operation_id": "fresh-fixture-operation",
    }
    if explicit_id:
        body["conversation_id"] = explicit_id
    request.json = AsyncMock(return_value=body)

    await server._handle_chat(request)

    if expected == "fresh":
        server._driver.navigate_new_chat.assert_awaited_once_with(gizmo_id=PROJECT)
        server._driver.ensure_current_conversation.assert_not_awaited()
        server._driver.navigate_conversation.assert_not_awaited()
    elif expected == "legacy_continue":
        server._driver.ensure_current_conversation.assert_awaited_once_with("previous-chat")
        server._driver.navigate_new_chat.assert_not_awaited()
        server._driver.navigate_conversation.assert_not_awaited()
    else:
        server._driver.navigate_conversation.assert_awaited_once_with("chosen-chat", project_id=PROJECT)
        server._driver.navigate_new_chat.assert_not_awaited()
        server._driver.ensure_current_conversation.assert_not_awaited()
    server._full_response.assert_awaited_once()

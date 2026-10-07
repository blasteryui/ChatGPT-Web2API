"""Focused production-contract tests for Vision provider hardening."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatgpt_web2api.backend_client import BackendClient
from chatgpt_web2api.cdp_driver import CDPDriver, GenerationStuckError
from chatgpt_web2api.provider_contract import (
    BackendHTTPError,
    EffectCertainty,
    ModelSelectionError,
    ProjectPlacementError,
    ProviderOperationContext,
    ProviderOperationUncertainError,
    RenameVerificationError,
    hash_text,
)
from chatgpt_web2api.turn_anchor import TurnAnchor


def _make_backend_client():
    driver = MagicMock()
    driver._access_token = "tok"
    driver._breakers = None
    driver._current_conv_id = None
    driver.ensure_token = AsyncMock(return_value="tok")
    driver._js_with_data_strict = AsyncMock()
    return BackendClient(driver), driver


@pytest.mark.asyncio
async def test_conversation_list_429_is_typed_not_empty():
    client, driver = _make_backend_client()
    driver._js_with_data_strict.return_value = json.dumps(
        {
            "__backend_http": True,
            "status": 429,
            "retry_after": "45",
            "body": '{"detail":"Too many requests"}',
        }
    )

    with pytest.raises(BackendHTTPError) as exc_info:
        await client.get_conversations()

    assert exc_info.value.status == 429
    assert exc_info.value.retry_after == "45"
    assert "Too many requests" in exc_info.value.body


@pytest.mark.asyncio
async def test_search_reconciliation_finds_exact_user_turn():
    client, driver = _make_backend_client()
    search_body = {
        "items": [
            {
                "conversation_id": "conv-1",
                "title": "nonce hit",
                "gizmo_id": "g-p-vision",
            }
        ],
        "cursor": None,
    }
    driver._js_with_data_strict.return_value = json.dumps(
        {
            "__backend_http": True,
            "status": 200,
            "retry_after": None,
            "body": json.dumps(search_body),
        }
    )
    driver.get_conversation = AsyncMock(
        return_value={
            "mapping": {
                "node-1": {
                    "message": {
                        "id": "user-msg-1",
                        "author": {"role": "user"},
                        "content": {"parts": ["VISION_NONCE_123"]},
                    }
                }
            }
        }
    )

    result = await client.reconcile_conversation_turn(
        "VISION_NONCE_123",
        exact_user_text="VISION_NONCE_123",
    )

    assert result["matches"][0]["conversation_id"] == "conv-1"
    assert result["matches"][0]["turns"][0]["message_id"] == "user-msg-1"


@pytest.mark.asyncio
async def test_verified_rename_readback_mismatch_fails():
    client, driver = _make_backend_client()
    driver.rename_conversation = AsyncMock(return_value=True)
    driver.get_conversation = AsyncMock(return_value={"id": "conv-1", "title": "Wrong"})

    with pytest.raises(RenameVerificationError) as exc_info:
        await client.rename_conversation_verified("conv-1", "Expected")

    assert exc_info.value.observed_title == "Wrong"


class _NoopMutationLock:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_strict_model_verification_failure_sends_nothing(monkeypatch):
    import chatgpt_web2api.api_server as api
    from chatgpt_web2api.config import Config

    driver = MagicMock(spec=CDPDriver)
    driver._current_conv_id = None
    driver.select_model_verified = AsyncMock(
        side_effect=ModelSelectionError("gpt-test", "different-model")
    )
    driver.select_model = AsyncMock(return_value=True)
    driver.navigate_new_chat = AsyncMock()
    driver.navigate_conversation = AsyncMock()
    driver.send_and_stream = MagicMock()

    monkeypatch.setattr(api, "MutationLock", _NoopMutationLock)

    server = api.APIServer(Config.load(None), driver)
    request = MagicMock()
    request.headers = {}
    request.query = {}
    request.json = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "must not send"}],
            "model": "gpt-test",
            "strict_provider": True,
            "stream": False,
        }
    )

    response = await server._handle_chat(request)

    assert response.status == 409
    payload = json.loads(response.body)
    assert payload["error"]["code"] == "model_verification_failed"
    driver.navigate_new_chat.assert_not_awaited()
    driver.navigate_conversation.assert_not_awaited()
    driver.send_and_stream.assert_not_called()


@pytest.mark.asyncio
async def test_uncertain_receipt_retains_turn_anchor_and_never_replays():
    driver = CDPDriver(cdp_port=9222, instance_id="receipt-test")
    driver._target_id = "target-1"
    driver._owns_target = True
    driver._read_assistant_count_baseline = AsyncMock(return_value=0)
    driver._identity_listener = None
    driver._capture_pre_send_fallback_anchor = AsyncMock(
        return_value=TurnAnchor(
            sent_text="VISION_UNCERTAIN_NONCE",
            mode="fresh_chat",
            pre_send_wall_time=123.0,
        )
    )
    driver.type_message = AsyncMock()
    driver.click_send = AsyncMock()
    driver._verify_send_acknowledged = AsyncMock(return_value=True)

    async def _stuck(**kwargs):
        if False:
            yield None
        raise GenerationStuckError("phase_1_appear", 90)

    driver._completion = MagicMock()
    driver._completion.stream_until_complete = _stuck

    context = ProviderOperationContext(
        operation_id="op-uncertain-1",
        requested_project_id="g-p-vision",
        observed_project_id="g-p-vision",
        requested_model="auto",
        observed_model="auto",
    )

    with pytest.raises(ProviderOperationUncertainError) as exc_info:
        async for _ in driver.send_and_stream(
            "VISION_UNCERTAIN_NONCE",
            timeout=120,
            model="auto",
            operation_context=context,
        ):
            pass

    receipt = exc_info.value.receipt
    assert receipt.operation_id == "op-uncertain-1"
    assert receipt.effect_certainty is EffectCertainty.DELIVERY_UNCERTAIN
    assert receipt.target_id == "target-1"
    assert receipt.session_identity == "receipt-test"
    assert receipt.turn_anchor["mode"] == "fresh_chat"
    assert receipt.turn_anchor["sent_text_sha256"] == hash_text("VISION_UNCERTAIN_NONCE")
    assert receipt.response_status == "DELIVERY_UNCERTAIN"
    driver.click_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_verified_model_failure_raises_before_any_send_primitive():
    driver = CDPDriver(cdp_port=9222)
    driver.select_model = AsyncMock(return_value=False)
    driver.click_send = AsyncMock()

    with pytest.raises(ModelSelectionError):
        await driver.select_model_verified("missing-model")

    driver.click_send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("project_id", "landed_url", "expected_url"),
    [
        (
            None,
            "https://chatgpt.com/c/conv-1",
            "https://chatgpt.com/c/conv-1",
        ),
        (
            "g-p-XYZ",
            "https://chatgpt.com/g/g-p-XYZ-vision/c/conv-1",
            "https://chatgpt.com/g/g-p-XYZ/c/conv-1",
        ),
    ],
)
async def test_navigate_conversation_uses_project_route_only_when_requested(
    monkeypatch, project_id, landed_url, expected_url
):
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("chatgpt_web2api.cdp_driver.asyncio.sleep", _no_sleep)
    driver = CDPDriver(cdp_port=9222)
    driver._cdp = AsyncMock()
    if project_id:
        driver.get_projects = AsyncMock(return_value=[])
    driver._js_strict = AsyncMock(
        return_value=json.dumps(
            {
                "url": landed_url,
                "ready_state": "complete",
                "app_shell": True,
                "composer": True,
            }
        )
    )

    await driver.navigate_conversation("conv-1", project_id=project_id)

    driver._cdp.assert_awaited_once_with("Page.navigate", {"url": expected_url})
    assert driver._current_conv_id == "conv-1"


@pytest.mark.asyncio
async def test_observe_project_id_canonicalizes_requested_slugged_route():
    driver = CDPDriver(cdp_port=9222)
    driver._js_strict = AsyncMock(
        return_value="https://chatgpt.com/g/g-p-XYZ-vision/c/conv-1"
    )

    assert await driver.observe_project_id("g-p-XYZ") == "g-p-XYZ"
    assert await driver.verify_project_placement("g-p-XYZ") == "g-p-XYZ"


@pytest.mark.asyncio
async def test_observe_project_id_keeps_wrong_project_distinct():
    driver = CDPDriver(cdp_port=9222)
    driver._js_strict = AsyncMock(
        return_value="https://chatgpt.com/g/g-p-OTHER-vision/c/conv-1"
    )

    with pytest.raises(ProjectPlacementError):
        await driver.verify_project_placement("g-p-XYZ")


@pytest.mark.asyncio
async def test_rest_explicit_project_continuation_forwards_project_id(monkeypatch):
    import chatgpt_web2api.api_server as api
    from chatgpt_web2api.config import Config

    driver = MagicMock(spec=CDPDriver)
    driver._current_conv_id = None
    driver.select_model = AsyncMock(return_value=True)
    driver.navigate_conversation = AsyncMock()
    driver.observe_project_id = AsyncMock(return_value="g-p-XYZ")
    driver.send_and_stream = MagicMock()
    monkeypatch.setattr(api, "MutationLock", _NoopMutationLock)

    server = api.APIServer(Config.load(None), driver)
    server._full_response = AsyncMock(return_value=MagicMock(status=200))
    request = MagicMock()
    request.headers = {}
    request.query = {}
    request.json = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "continue"}],
            "conversation_id": "conv-1",
            "project_id": "g-p-XYZ",
            "stream": False,
        }
    )

    response = await server._handle_chat(request)

    assert response.status == 200
    driver.navigate_conversation.assert_awaited_once_with(
        "conv-1", project_id="g-p-XYZ"
    )


@pytest.mark.asyncio
async def test_mcp_explicit_project_continuation_forwards_project_id():
    from chatgpt_web2api.mcp_server import do_chat_completion

    driver = MagicMock(spec=CDPDriver)
    driver._current_conv_id = None
    driver.send_and_stream = AsyncMock()
    driver.navigate_conversation = AsyncMock()
    driver.observe_project_id = AsyncMock(return_value="g-p-XYZ")

    async def _stream(*_args, **_kwargs):
        if False:
            yield None

    driver.send_and_stream = _stream
    await do_chat_completion(
        driver,
        {
            "message": "continue",
            "conversation_id": "conv-1",
            "project_id": "g-p-XYZ",
        },
        __import__("chatgpt_web2api.config", fromlist=["Config"]).Config.load(None),
    )

    driver.navigate_conversation.assert_awaited_once_with(
        "conv-1", project_id="g-p-XYZ"
    )


@pytest.mark.asyncio
async def test_rest_wrong_project_precondition_never_reaches_send(monkeypatch):
    import chatgpt_web2api.api_server as api
    from chatgpt_web2api.config import Config

    driver = MagicMock(spec=CDPDriver)
    driver._current_conv_id = None
    driver.select_model = AsyncMock(return_value=True)
    driver.navigate_conversation = AsyncMock()
    driver.observe_project_id = AsyncMock(return_value="g-p-other")
    driver.send_and_stream = MagicMock()
    monkeypatch.setattr(api, "MutationLock", _NoopMutationLock)

    server = api.APIServer(Config.load(None), driver)
    server._full_response = AsyncMock()
    request = MagicMock()
    request.headers = {}
    request.query = {}
    request.json = AsyncMock(
        return_value={
            "messages": [{"role": "user", "content": "must not send"}],
            "conversation_id": "conv-1",
            "project_id": "g-p-XYZ",
            "strict_provider": True,
            "stream": False,
        }
    )

    response = await server._handle_chat(request)

    assert response.status == 409
    payload = json.loads(response.body)
    assert payload["error"]["code"] == "project_verification_failed"
    server._full_response.assert_not_awaited()
    driver.send_and_stream.assert_not_called()

@pytest.mark.asyncio
async def test_navigate_conversation_prefers_canonical_project_short_url(monkeypatch):
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("chatgpt_web2api.cdp_driver.asyncio.sleep", _no_sleep)
    driver = CDPDriver(cdp_port=9222)
    driver._cdp = AsyncMock()
    driver.get_projects = AsyncMock(
        return_value=[
            {
                "id": "g-p-XYZ",
                "name": "Vision",
                "short_url": "g-p-XYZ-vision",
            }
        ]
    )
    driver._js_strict = AsyncMock(
        return_value=json.dumps(
            {
                "url": "https://chatgpt.com/g/g-p-XYZ-vision/c/conv-1",
                "ready_state": "complete",
                "app_shell": True,
                "composer": True,
            }
        )
    )

    await driver.navigate_conversation("conv-1", project_id="g-p-XYZ")

    driver._cdp.assert_awaited_once_with(
        "Page.navigate",
        {"url": "https://chatgpt.com/g/g-p-XYZ-vision/c/conv-1"},
    )
    assert driver._current_conv_id == "conv-1"

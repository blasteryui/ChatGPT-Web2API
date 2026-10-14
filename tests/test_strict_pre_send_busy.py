"""Strict provider calls must fail before Send when their CDP target is busy.

Regression for production delayed Web2API Send after Shepherd HTTP client timed out.
All tests are local asyncio/file-lock doubles; no Chrome, network, or provider Send.
"""
import asyncio

import pytest

from chatgpt_web2api import lock_resolver as lr
from chatgpt_web2api.cross_process_lock import LockAcquisitionError


class NullFileLock:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_strict_provider_fails_immediately_before_a_queued_send(monkeypatch):
    monkeypatch.setattr(lr, "CrossProcessLock", NullFileLock)
    active = lr.MutationLock(19881, "target-ONE")
    strict = lr.MutationLock(19881, "target-ONE", fail_if_busy=True)

    async with active:
        with pytest.raises(LockAcquisitionError, match="mutation_busy_before_send"):
            async with strict:
                pytest.fail("a contended strict provider may never enter Send")

    # Rejection does not poison the process-local lease.
    async with strict:
        pass


@pytest.mark.asyncio
async def test_legacy_non_strict_waits_for_same_target(monkeypatch):
    monkeypatch.setattr(lr, "CrossProcessLock", NullFileLock)
    entered = []
    active = lr.MutationLock(19882, "target-ONE")
    waiting = lr.MutationLock(19882, "target-ONE")
    async with active:
        task = asyncio.create_task(_enter(waiting, entered))
        await asyncio.sleep(0.01)
        assert entered == []
    await asyncio.wait_for(task, 1)
    assert entered == ["entered"]


async def _enter(lock, entered):
    async with lock:
        entered.append("entered")


@pytest.mark.asyncio
async def test_parallel_distinct_target_is_not_rejected(monkeypatch):
    monkeypatch.setattr(lr, "CrossProcessLock", NullFileLock)
    async with lr.MutationLock(19883, "target-A"):
        async with lr.MutationLock(19883, "target-B", fail_if_busy=True):
            pass


def test_strict_cross_process_lock_has_zero_wait(monkeypatch):
    constructed = []

    class RecordingFileLock(NullFileLock):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(lr, "CrossProcessLock", RecordingFileLock)
    lr.MutationLock(19884, "target-ONE", fail_if_busy=True)
    lr.MutationLock(19884, "target-ONE", fail_if_busy=False)
    assert constructed[0]["timeout"] == 0.0
    assert "timeout" not in constructed[1]


@pytest.mark.asyncio
async def test_strict_file_lock_rejection_releases_local_lock(monkeypatch):
    class RejectingFileLock(NullFileLock):
        async def __aenter__(self):
            raise LockAcquisitionError("cross_process_busy_before_send")

    monkeypatch.setattr(lr, "CrossProcessLock", RejectingFileLock)
    strict = lr.MutationLock(19885, "target-ONE", fail_if_busy=True)
    with pytest.raises(LockAcquisitionError):
        async with strict:
            pytest.fail("rejected cross-process call must not mutate")
    assert not lr._proc_lock_for(19885, "target-ONE").locked()


@pytest.mark.asyncio
async def test_strict_rest_endpoint_refuses_busy_before_any_driver_action(monkeypatch):
    """Actual REST request gets HTTP503 without DOM navigation or Send."""
    import json
    from types import SimpleNamespace
    from chatgpt_web2api.api_server import APIServer

    monkeypatch.setattr(lr, "CrossProcessLock", NullFileLock)

    class Driver:
        def __init__(self):
            self.called = []

        async def navigate_new_chat(self, **kwargs):
            self.called.append("navigate")
            pytest.fail("No navigation after strict busy")

    class Request:
        headers = {}
        query = {}

        async def json(self):
            return {
                "model": "auto", "strict_provider": True,
                "project_id": "g-p-6abc5b56d4488191b9b5b5224e3ef10c",
                "operation_id": "VISION_SHEPHERD_fire_test_no_send",
                "messages": [{"role": "user", "content": "local fixture only"}],
                "stream": False,
            }

    config = SimpleNamespace(
        chrome=SimpleNamespace(cdp_port=19886),
        chatgpt=SimpleNamespace(
            parallel_tabs=False, default_model="auto",
            default_project_id=None,
        ),
        server=SimpleNamespace(api_keys=[], request_timeout=60),
    )
    driver = Driver()
    server = APIServer(config, driver)

    async def no_breaker():
        return None

    monkeypatch.setattr(server, "_check_circuit_or_recover", no_breaker)
    async with lr.MutationLock(19886):
        response = await server._handle_chat(Request())
    assert response.status == 503
    payload = json.loads(response.text)
    assert payload["error"]["code"] == "lock_timeout"
    assert "mutation_busy_before_send" in payload["error"]["message"]
    assert driver.called == []

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

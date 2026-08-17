# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""boot_agent_sandbox must never orphan a sandbox whose __aenter__ succeeded.

The leak this guards against: once __aenter__ returns, the sandbox has a live
heartbeat task that keeps refreshing Daytona activity, so the cloud-side auto_stop
never fires. Any failure after __aenter__ that does not __aexit__ the candidate
strands it forever (a real run left 2687 sandboxes alive after one kill). These
tests drive the two post-entry failure paths and assert __aexit__ ran exactly once.
"""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.harness.agents import claude_code


class _FakeSandbox:
    def __init__(self, *, fail_enter: bool = False):
        self.fail_enter = fail_enter
        self.entered = 0
        self.exited = 0
        self.sandbox_id = "fake"

    async def __aenter__(self):
        if self.fail_enter:
            raise RuntimeError("enter boom")
        self.entered += 1
        return self

    async def __aexit__(self, *a):
        self.exited += 1


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # One boot attempt, no backoff sleeps, no real semaphore contention.
    monkeypatch.setattr(claude_code, "SWE_BOOT_RETRIES", 1)
    monkeypatch.setattr(claude_code, "_BOOT_SEM", asyncio.Semaphore(1))


async def _drain(image, **kw):
    async with claude_code.boot_agent_sandbox(image, **kw) as sb:
        return sb


def test_install_failure_after_enter_cleans_up(monkeypatch):
    """__aenter__ succeeds, install_toolchain throws -> __aexit__ must run."""
    sb = _FakeSandbox()
    monkeypatch.setattr(claude_code, "make_sandbox", lambda *a, **k: sb)

    async def _boom(_sb):
        raise RuntimeError("install boom")

    monkeypatch.setattr(claude_code, "install_toolchain", _boom)

    with pytest.raises(Exception):
        asyncio.run(_drain("img", install_claude=True))
    assert sb.entered == 1
    assert sb.exited == 1, "heartbeat orphaned: __aexit__ never ran after install fail"


def test_enter_failure_does_not_double_exit(monkeypatch):
    """__aenter__ itself throws -> nothing entered, __aexit__ must NOT run."""
    sb = _FakeSandbox(fail_enter=True)
    monkeypatch.setattr(claude_code, "make_sandbox", lambda *a, **k: sb)

    with pytest.raises(Exception):
        asyncio.run(_drain("img", install_claude=False))
    assert sb.exited == 0, "__aexit__ ran on a sandbox that never entered"


def test_success_path_exits_once(monkeypatch):
    """Happy path: entered once, and the caller's finally exits once."""
    sb = _FakeSandbox()
    monkeypatch.setattr(claude_code, "make_sandbox", lambda *a, **k: sb)

    asyncio.run(_drain("img", install_claude=False))
    assert sb.entered == 1 and sb.exited == 1

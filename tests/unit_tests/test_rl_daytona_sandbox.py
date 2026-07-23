# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from torchtitan.experiments.rl.examples.tmax.vanillux_loop import _run_bash
from torchtitan.experiments.rl.harness.agents import claude_code as agent_backend
from torchtitan.experiments.rl.harness.sandbox import (
    daytona as daytona_backend,
    SandboxIssue,
    SandboxIssueTracker,
    SandboxLogContext,
)
from torchtitan.experiments.rl.harness.sandbox.daytona import (
    _build_exec_command,
    DaytonaSandbox,
)


class _SessionExecuteRequest:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_daytona(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("daytona")
    module.__dict__["SessionExecuteRequest"] = _SessionExecuteRequest
    monkeypatch.setitem(sys.modules, "daytona", module)


def _process() -> SimpleNamespace:
    return SimpleNamespace(
        create_session=AsyncMock(return_value=None),
        delete_session=AsyncMock(return_value=None),
        execute_session_command=AsyncMock(
            return_value=SimpleNamespace(cmd_id="command-id")
        ),
        get_session=AsyncMock(return_value=SimpleNamespace(commands=[])),
        get_session_command=AsyncMock(return_value=SimpleNamespace(exit_code=0)),
        get_session_command_logs=AsyncMock(
            return_value=SimpleNamespace(stdout="ok", stderr="")
        ),
    )


def _sandbox_with_process(process: SimpleNamespace) -> DaytonaSandbox:
    sandbox = DaytonaSandbox("example/image")
    sandbox._sb = SimpleNamespace(process=process)
    return sandbox


def test_disk_override_is_validated() -> None:
    assert DaytonaSandbox("example/image", disk_gb=20).disk_gb == 20
    for invalid_disk_gb in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="disk_gb must be a positive integer"):
            DaytonaSandbox("example/image", disk_gb=cast(int, invalid_disk_gb))


def test_build_exec_command_preserves_complex_command() -> None:
    command = "printf '%s\\n' \"$HOME\"; printf 'line 1\\nline 2\\n'"

    full = _build_exec_command(command, user="root", env=None, timeout=17)

    assert shlex.split(full) == [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "17s",
        "bash",
        "-c",
        command,
    ]


def test_build_exec_command_preserves_nonroot_env() -> None:
    env = {"GREETING": "hello world", "LINES": "first\nsecond"}

    full = _build_exec_command("env", user="agent", env=env, timeout=9)

    assert shlex.split(full) == [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "9s",
        "env",
        "--",
        "GREETING=hello world",
        "LINES=first\nsecond",
        "runuser",
        "-u",
        "agent",
        "--whitelist-environment=GREETING,LINES",
        "--",
        "bash",
        "-c",
        "env",
    ]


def test_exec_separates_command_and_request_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_EXEC_TIMEOUT_MIN", "240")
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(return_value=(0, "ok"))

    result = asyncio.run(sandbox.exec("echo ok", timeout=5))

    assert result == (0, "ok", "")
    call = sandbox._session_exec.await_args
    assert call is not None
    assert shlex.split(call.args[0])[3] == "5s"
    assert call.kwargs == {"command_timeout": 5, "request_timeout": 240}


def test_long_command_does_not_extend_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TT_DAYTONA_EXEC_TIMEOUT_MIN", raising=False)
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(return_value=(0, "ok"))

    asyncio.run(sandbox.exec("echo ok", timeout=3600))

    call = sandbox._session_exec.await_args
    assert call is not None
    assert shlex.split(call.args[0])[3] == "3600s"
    assert call.kwargs == {"command_timeout": 3600, "request_timeout": 120}


def test_exec_reports_timeout_exit() -> None:
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(return_value=(124, "partial output"))

    result = asyncio.run(sandbox.exec("sleep 30", timeout=3))

    assert result == (124, "partial output", "Command timed out after 3s.")

    with pytest.raises(RuntimeError, match="Command timed out after 3s"):
        asyncio.run(sandbox.exec("sleep 30", timeout=3, check=True))

    sandbox._session_exec.return_value = (137, "Killed")
    assert asyncio.run(sandbox.exec("sleep 30", timeout=3)) == (137, "Killed", "")


@pytest.mark.skipif(shutil.which("timeout") is None, reason="GNU timeout is required")
def test_wrapped_command_is_hard_timed_out() -> None:
    full = _build_exec_command("sleep 30", user="root", env=None, timeout=1)
    started = time.monotonic()

    completed = subprocess.run(
        ["bash", "-c", full], capture_output=True, check=False, timeout=5
    )

    assert completed.returncode == 124
    assert time.monotonic() - started < 5


@pytest.mark.skipif(
    shutil.which("timeout") is None or shutil.which("setsid") is None,
    reason="GNU timeout and setsid are required",
)
def test_wrapped_command_preserves_successful_detached_child() -> None:
    full = _build_exec_command(
        "setsid sleep 30 >/dev/null 2>&1 & echo $!",
        user="root",
        env=None,
        timeout=2,
    )
    completed = subprocess.run(
        ["bash", "-c", full], capture_output=True, check=False, text=True, timeout=5
    )
    assert completed.returncode == 0
    pid = int(completed.stdout.strip())
    try:
        os.kill(pid, 0)
    finally:
        os.kill(pid, signal.SIGKILL)


def test_lost_execute_response_recovers_without_replay(
    fake_daytona: None,
) -> None:
    full = "echo once"
    process = _process()
    process.execute_session_command.side_effect = ConnectionError("response lost")
    process.get_session.return_value = SimpleNamespace(
        commands=[SimpleNamespace(id="recovered-id", command=full)]
    )
    sandbox = _sandbox_with_process(process)

    result = asyncio.run(
        sandbox._session_exec(
            full,
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    process.execute_session_command.assert_awaited_once()
    assert process.execute_session_command.await_args.kwargs["timeout"] == 30
    process.delete_session.assert_not_awaited()
    assert sandbox.issue_tracker.counts == {"execute_response_recovered": 1}
    issue = sandbox.issue_tracker.issues[0]
    assert issue.recovered
    assert issue.session_id
    assert issue.command_id == "recovered-id"


def test_session_create_enospc_is_not_retried(
    caplog: pytest.LogCaptureFixture,
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TT_DAYTONA_SESSION_CREATE_RETRIES", "5")
    process = _process()
    process.create_session.side_effect = RuntimeError("no space left on device")
    tracker = SandboxIssueTracker(
        SandboxLogContext(
            instance_id="task-123",
            group_id=7,
            rollout_id=11,
        )
    )
    sandbox = DaytonaSandbox("example/image", disk_gb=20, issue_tracker=tracker)
    sandbox._sb = SimpleNamespace(process=process)
    sandbox.sandbox_id = "sandbox-id"

    with pytest.raises(RuntimeError, match="no space left on device"):
        asyncio.run(
            sandbox._session_exec(
                "echo once",
                command_timeout=5,
                request_timeout=30,
            )
        )

    process.create_session.assert_awaited_once()
    process.delete_session.assert_awaited_once()
    assert tracker.counts == {"session_disk_exhausted": 1}
    issue = tracker.issues[0]
    assert issue.phase == "session_create"
    assert issue.sandbox_id == "sandbox-id"
    assert issue.session_id
    assert issue.attempt == 1
    assert issue.max_attempts == 6
    payload = json.loads(caplog.records[-1].getMessage().split("] ", 1)[1])
    assert payload == {
        "attempt": 1,
        "command_id": "",
        "disk_gb": 20,
        "error_type": "RuntimeError",
        "event": "sandbox_issue",
        "exit_code": None,
        "group_id": 7,
        "image": "example/image",
        "instance_id": "task-123",
        "kind": "session_disk_exhausted",
        "max_attempts": 6,
        "message": "no space left on device",
        "phase": "session_create",
        "provider": "daytona",
        "recovered": False,
        "rollout_id": 11,
        "sandbox_id": "sandbox-id",
        "session_id": issue.session_id,
    }


def test_nonzero_command_enospc_is_recorded() -> None:
    sandbox = DaytonaSandbox("example/image", disk_gb=20)
    sandbox._session_exec = AsyncMock(
        return_value=(1, "OSError: [Errno 28] No space left on device")
    )

    result = asyncio.run(sandbox.exec("pip install package", timeout=5))

    assert result == (1, "OSError: [Errno 28] No space left on device", "")
    assert sandbox.issue_tracker.counts == {"command_disk_exhausted": 1}
    issue = sandbox.issue_tracker.issues[0]
    assert issue.phase == "command"
    assert issue.exit_code == 1


def test_successful_command_output_does_not_misclassify_enospc_text() -> None:
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(
        return_value=(0, "binary contains no space left on device")
    )

    assert asyncio.run(sandbox.exec("strings binary")) == (
        0,
        "binary contains no space left on device",
        "",
    )
    assert sandbox.issue_tracker.counts == {}


def test_tracker_keeps_exec_error_swallowed_by_agent_loop() -> None:
    sandbox = DaytonaSandbox("example/image")
    sandbox._session_exec = AsyncMock(
        side_effect=RuntimeError("no space left on device")
    )

    output, exit_code = asyncio.run(_run_bash(sandbox, "echo ok", timeout=5))

    assert exit_code == 1
    assert "exec failed: RuntimeError: no space left on device" in output
    assert sandbox.issue_tracker.counts == {"exec_failed": 1}


def test_boot_retries_share_rollout_issue_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = SandboxIssueTracker(
        SandboxLogContext(instance_id="task-123", group_id=7, rollout_id=11)
    )
    received_trackers: list[SandboxIssueTracker] = []
    num_candidates = 0

    class _Candidate:
        allocated_disk_gb = 20

        def __init__(
            self, candidate_tracker: SandboxIssueTracker, *, fail: bool
        ) -> None:
            self.issue_tracker = candidate_tracker
            self.fail = fail
            self.sandbox_id = "failed-sandbox" if fail else "active-sandbox"

        async def __aenter__(self):
            if self.fail:
                self.issue_tracker.record(
                    SandboxIssue(
                        provider="daytona",
                        kind="create_retry",
                        phase="create",
                        recovered=True,
                        error_type="RuntimeError",
                        message="transient create failure",
                        attempt=1,
                        max_attempts=2,
                    )
                )
                raise RuntimeError("transient create failure")
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    def make_sandbox(image: str, **kwargs):
        nonlocal num_candidates
        candidate_tracker = kwargs["issue_tracker"]
        received_trackers.append(candidate_tracker)
        candidate = _Candidate(candidate_tracker, fail=num_candidates == 0)
        num_candidates += 1
        return candidate

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(agent_backend, "SWE_BOOT_RETRIES", 2)
    monkeypatch.setattr(agent_backend, "_BOOT_SEM", None)
    monkeypatch.setattr(agent_backend, "make_sandbox", make_sandbox)
    monkeypatch.setattr(agent_backend.asyncio, "sleep", no_sleep)

    async def run() -> None:
        async with agent_backend.boot_agent_sandbox(
            "example/image",
            install_claude=False,
            disk_gb=20,
            issue_tracker=tracker,
        ) as sandbox:
            assert sandbox.sandbox_id == "active-sandbox"

    asyncio.run(run())

    assert received_trackers == [tracker, tracker]
    assert tracker.counts == {"create_retry": 1}


def test_poll_disconnect_does_not_replay_command(fake_daytona: None) -> None:
    process = _process()
    process.get_session_command.side_effect = [
        ConnectionError("server disconnected"),
        SimpleNamespace(exit_code=0),
    ]
    sandbox = _sandbox_with_process(process)

    result = asyncio.run(
        sandbox._session_exec(
            "echo once",
            command_timeout=5,
            request_timeout=30,
        )
    )

    assert result == (0, "ok")
    process.execute_session_command.assert_awaited_once()
    assert process.get_session_command.await_count == 2
    process.delete_session.assert_not_awaited()
    assert sandbox.issue_tracker.counts == {"poll_transient": 1}
    assert sandbox.issue_tracker.issues[0].recovered


def test_poll_deadline_does_not_delete_a_possibly_successful_session(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daytona_backend, "_COMMAND_KILL_GRACE_SEC", 0)
    monkeypatch.setattr(daytona_backend, "_SESSION_POLL_GRACE_SEC", 0)
    process = _process()
    sandbox = _sandbox_with_process(process)

    with pytest.raises(TimeoutError, match="without replaying"):
        asyncio.run(
            sandbox._session_exec(
                "setsid sleep 30 &",
                command_timeout=0,
                request_timeout=30,
            )
        )

    process.execute_session_command.assert_awaited_once()
    process.get_session_command.assert_not_awaited()
    process.delete_session.assert_not_awaited()


def test_unconfirmed_execute_is_cleaned_up_without_replay(
    fake_daytona: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(daytona_backend, "_COMMAND_RECOVERY_DELAYS_SEC", (0.0,))
    process = _process()
    process.execute_session_command.side_effect = ConnectionError("not accepted")
    sandbox = _sandbox_with_process(process)

    with pytest.raises(ConnectionError, match="not accepted"):
        asyncio.run(
            sandbox._session_exec(
                "echo once",
                command_timeout=5,
                request_timeout=30,
            )
        )

    process.execute_session_command.assert_awaited_once()
    process.delete_session.assert_awaited_once()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from torchtitan.experiments.rl.harness.sandbox import daytona as daytona_backend
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

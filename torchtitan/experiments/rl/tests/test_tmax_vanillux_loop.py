# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from torchtitan.experiments.rl.examples.tmax import vanillux_loop
from torchtitan.experiments.rl.examples.tmax.rollouter import _finish_reason_metrics
from torchtitan.experiments.rl.observability.metrics import Mean


class _FakeAdapter:
    def __init__(self, responses: list[dict | None]) -> None:
        self._responses = iter(responses)

    async def complete(self, session_id: str, payload: dict) -> dict | None:
        return next(self._responses, None)


@pytest.fixture(autouse=True)
def _patch_sandbox_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    async def prepare_runtime(sb: Any) -> None:
        pass

    async def run_bash(sb: Any, command: str, timeout: int) -> tuple[str, int]:
        return "command output", 0

    monkeypatch.setattr(vanillux_loop, "_prepare_runtime", prepare_runtime)
    monkeypatch.setattr(vanillux_loop, "_run_bash", run_bash)


def _tool_response() -> dict:
    return {
        "content": [
            {
                "type": "tool_use",
                "name": "bash",
                "id": "call-1",
                "input": {"command": "echo ok"},
            }
        ],
        "stop_reason": "tool_use",
    }


def _run_loop(
    responses: list[dict | None],
    *,
    time_budget_sec: int = 60,
    max_turns: int = 1,
) -> tuple[int, bool, int, str]:
    sandbox: Any = object()
    adapter: Any = _FakeAdapter(responses)
    return asyncio.run(
        vanillux_loop.run_vanillux_loop(
            sandbox,
            task="test task",
            session_id="group=0/rollout=0",
            adapter=adapter,
            time_budget_sec=time_budget_sec,
            max_turns=max_turns,
        )
    )


def test_finish_reason_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_bash(sb: Any, command: str, timeout: int) -> tuple[str, int]:
        return vanillux_loop.SUBMIT_MARKER, 0

    monkeypatch.setattr(vanillux_loop, "_run_bash", run_bash)

    assert _run_loop([_tool_response()]) == (1, True, 0, "submit")


def test_finish_reason_hit_max_turns() -> None:
    assert _run_loop([_tool_response()]) == (1, False, 0, "hit_max_turns")


def test_finish_reason_hit_time_budget() -> None:
    assert _run_loop([], time_budget_sec=0) == (
        0,
        False,
        0,
        "hit_time_budget",
    )


def test_finish_reason_stopped_early() -> None:
    assert _run_loop([]) == (0, False, 0, "stopped_early")


def test_format_error_stops_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vanillux_loop, "_FORMAT_ERROR_FEEDBACK", False)
    response = {"content": [{"type": "text", "text": "not a tool call"}]}

    assert _run_loop([response]) == (1, False, 1, "stopped_early")


def test_finish_reason_metrics_are_exhaustive_fractions() -> None:
    metrics = _finish_reason_metrics(
        ["submit", "submit", "hit_time_budget", "stopped_early", "error"]
    )
    fractions = {}
    for metric in metrics:
        assert isinstance(metric.value, Mean)
        fractions[metric.key] = metric.value.value / metric.value.count

    assert fractions == {
        "rollout/finish_submit_frac": 0.4,
        "rollout/finish_hit_max_turns_frac": 0.0,
        "rollout/finish_hit_time_budget_frac": 0.2,
        "rollout/finish_stopped_early_frac": 0.2,
        "rollout/finish_error_frac": 0.2,
    }
    assert sum(fractions.values()) == pytest.approx(1.0)

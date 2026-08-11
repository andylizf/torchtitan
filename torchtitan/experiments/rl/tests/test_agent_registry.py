# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The swappable-harness contract.

The load-bearing guarantees are that the tmax default still resolves to the
vanillux loop with the same arguments as before (the 9B is SFT'd under it), and
that a harness with no submit signal reports ``submitted=None`` rather than False
-- reading None as False would silently score every rollout 0.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from torchtitan.experiments.rl.harness.agents.spec import (
    AgentRun,
    AgentTask,
    get_agent,
    register_agent,
    registered_agents,
)


def _task(**overrides) -> AgentTask:
    defaults = dict(
        sandbox=MagicMock(),
        instruction="do the thing",
        session_id="sess-1",
        adapter=MagicMock(url="http://127.0.0.1:1/"),
        time_budget_sec=60,
    )
    return AgentTask(**{**defaults, **overrides})


def test_registry_rejects_unknown_and_lists_alternatives():
    with pytest.raises(ValueError, match="unknown agent 'nope'; registered:"):
        get_agent("nope")


def test_registry_rejects_a_conflicting_re_registration():
    async def one(_task):
        return AgentRun(turns=1)

    async def two(_task):
        return AgentRun(turns=2)

    register_agent("dup-probe", one)
    register_agent("dup-probe", one)  # idempotent for the same callable
    with pytest.raises(ValueError, match="already registered"):
        register_agent("dup-probe", two)


def test_vanillux_is_the_registered_tmax_default():
    from torchtitan.experiments.rl.examples.tmax.vanillux_loop import vanillux_agent

    assert "vanillux" in registered_agents()
    assert get_agent("vanillux") is vanillux_agent


def test_vanillux_agent_forwards_the_pre_registry_arguments(monkeypatch):
    """The default path must call run_vanillux_loop exactly as the rollouter did."""
    from torchtitan.experiments.rl.examples.tmax import vanillux_loop

    seen = {}

    async def fake_loop(sb, **kwargs):
        seen["sb"] = sb
        seen.update(kwargs)
        return 7, True, 2, "submit"

    monkeypatch.setattr(vanillux_loop, "run_vanillux_loop", fake_loop)
    sandbox = MagicMock()
    adapter = MagicMock(url="http://127.0.0.1:1/")
    run = asyncio.run(
        vanillux_loop.vanillux_agent(
            _task(sandbox=sandbox, adapter=adapter, time_budget_sec=1234)
        )
    )

    assert seen["sb"] is sandbox
    assert seen["task"] == "do the thing"
    assert seen["session_id"] == "sess-1"
    assert seen["adapter"] is adapter, "must pass the adapter OBJECT (in-process)"
    assert seen["time_budget_sec"] == 1234
    # Unset caps must not be forwarded, so the harness keeps its own defaults.
    assert "max_turns" not in seen and "exec_timeout" not in seen
    assert run == AgentRun(
        turns=7, submitted=True, format_errors=2, finish_reason="submit"
    )


def test_optional_caps_are_forwarded_when_set(monkeypatch):
    from torchtitan.experiments.rl.examples.tmax import vanillux_loop

    seen = {}

    async def fake_loop(_sb, **kwargs):
        seen.update(kwargs)
        return 1, False, 0, "hit_max_turns"

    monkeypatch.setattr(vanillux_loop, "run_vanillux_loop", fake_loop)
    asyncio.run(vanillux_loop.vanillux_agent(_task(max_turns=192, exec_timeout=90)))
    assert seen["max_turns"] == 192
    assert seen["exec_timeout"] == 90


def test_host_loop_reports_no_submit_signal_rather_than_false(monkeypatch):
    """tmax grades on ``submitted is not False``; a bare turn count must not be
    reported as "did not submit" or every rollout would score 0."""
    from torchtitan.experiments.rl.harness.agents import host_loop

    async def fake(_sb, **_kwargs):
        return 5

    monkeypatch.setattr(host_loop, "run_host_loop", fake)
    run = asyncio.run(host_loop.host_loop_agent(_task(workdir="/testbed")))

    assert run.submitted is None
    assert run.submitted is not False
    assert run.turns == 5


def test_host_loop_maps_a_negative_turn_count_to_error(monkeypatch):
    from torchtitan.experiments.rl.harness.agents import host_loop

    async def fake(_sb, **_kwargs):
        return -1

    monkeypatch.setattr(host_loop, "run_host_loop", fake)
    run = asyncio.run(host_loop.host_loop_agent(_task()))

    assert run.turns == 0 and run.finish_reason == "error"


def test_host_loop_passes_the_adapter_url_not_the_object(monkeypatch):
    """In-sandbox agents dial back over HTTP; they cannot take the adapter object."""
    from torchtitan.experiments.rl.harness.agents import host_loop

    seen = {}

    async def fake(_sb, **kwargs):
        seen.update(kwargs)
        return 1

    monkeypatch.setattr(host_loop, "run_host_loop", fake)
    adapter = MagicMock(url="http://127.0.0.1:9999/")
    asyncio.run(host_loop.host_loop_agent(_task(adapter=adapter)))

    assert seen["adapter_url"] == "http://127.0.0.1:9999/"
    assert seen["problem_statement"] == "do the thing"

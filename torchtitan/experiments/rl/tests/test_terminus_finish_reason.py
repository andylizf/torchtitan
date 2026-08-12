# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How a Terminus-2 trajectory ended, and why "ended early" is not "submitted".

Terminus-2's episode loop has three exits and only one is a submit:

1. it runs the episodes out (``_n_episodes == max_turns``);
2. it returns early on a CONFIRMED ``<task_complete>true</task_complete>`` -- the
   second consecutive one, at which point ``_pending_completion`` is still set;
3. it returns early because ``is_session_alive()`` went false, i.e. the tmux
   session died under it, with no completion claimed at all.

Reading "ended before the cap" as the submit signal folds 3 into 2 and reports a
dead session as a real attempt, which then scores 0 and looks like a model failure.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class _FakeTerminus2:
    """Stands in for harbor's Terminus2, replaying one of the three exits."""

    def __init__(self, *, episodes: int, pending_completion: bool, raises=None, **_):
        self._episodes = episodes
        self._pending_completion = pending_completion
        self._raises = raises
        self._n_episodes = 0
        self._llm = None

    async def setup(self, _env) -> None:
        return None

    async def run(self, _instruction, _env, _context) -> None:
        # The real loop assigns _n_episodes at the top of each iteration, so the
        # count survives an exception raised mid-episode.
        self._n_episodes = self._episodes
        if self._raises is not None:
            raise self._raises


def _install_fake_harbor(monkeypatch, **agent_kwargs) -> None:
    """Import terminus.py against a stub harbor (the real one needs a sandbox)."""
    for name in ("harbor", "harbor.agents", "harbor.agents.terminus_2"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(
        sys.modules["harbor.agents.terminus_2"],
        "Terminus2",
        lambda **kwargs: _FakeTerminus2(**agent_kwargs, **kwargs),
        raising=False,
    )
    context_mod = types.ModuleType("harbor.models.agent.context")
    context_mod.AgentContext = MagicMock  # pyrefly: ignore
    for name in ("harbor.models", "harbor.models.agent"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "harbor.models.agent.context", context_mod)


def _run(monkeypatch, *, max_turns: int, **agent_kwargs):
    from torchtitan.experiments.rl.harness.agents.spec import AgentTask
    from torchtitan.experiments.rl.harness.agents.terminus import terminus_agent

    _install_fake_harbor(monkeypatch, **agent_kwargs)
    sandbox = MagicMock()

    async def _exec(*_args, **_kwargs):
        return 0, "", ""

    sandbox.exec = _exec
    return asyncio.run(
        terminus_agent(
            AgentTask(
                sandbox=sandbox,
                instruction="do the thing",
                session_id="sess-1",
                adapter=MagicMock(),
                time_budget_sec=60,
                max_turns=max_turns,
                workdir=str(Path("/app")),
            )
        )
    )


def test_confirmed_task_complete_is_a_submit(monkeypatch):
    run = _run(monkeypatch, max_turns=10, episodes=4, pending_completion=True)
    assert (run.finish_reason, run.submitted, run.turns) == ("submit", True, 4)


def test_a_dead_session_is_stopped_early_not_a_submit(monkeypatch):
    """The regression: exit 3 used to report submit just for ending under the cap."""
    run = _run(monkeypatch, max_turns=10, episodes=4, pending_completion=False)
    assert (run.finish_reason, run.submitted, run.turns) == ("stopped_early", False, 4)


def test_running_the_episodes_out_hits_max_turns(monkeypatch):
    run = _run(monkeypatch, max_turns=10, episodes=10, pending_completion=False)
    assert (run.finish_reason, run.submitted, run.turns) == ("hit_max_turns", False, 10)


def test_max_turns_wins_over_a_pending_completion(monkeypatch):
    """One unconfirmed task_complete on the last episode is not a submit."""
    run = _run(monkeypatch, max_turns=10, episodes=10, pending_completion=True)
    assert run.finish_reason == "hit_max_turns"
    assert run.submitted is False


def test_an_error_keeps_the_episodes_it_got_through(monkeypatch):
    """Reporting 0 turns here misattributes turns that are still trained on."""
    run = _run(
        monkeypatch,
        max_turns=10,
        episodes=6,
        pending_completion=False,
        raises=RuntimeError("adapter returned no completion"),
    )
    assert (run.finish_reason, run.submitted, run.turns) == ("error", False, 6)


@pytest.mark.parametrize(
    "reason", ["submit", "stopped_early", "hit_max_turns", "error"]
)
def test_every_reported_reason_is_one_the_rollouter_accepts(reason):
    """The rollouter validates finish_reason; an unknown one would fail a rollout."""
    from torchtitan.experiments.rl.examples.tmax.rollouter import _FINISH_REASONS

    assert reason in _FINISH_REASONS

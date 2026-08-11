# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``_guard_eval_generators``: a transient eval-generator RPC failure must not
disable validation for the rest of the run.

An eval generator idle between passes answers its next call with a gloo
"connection closed by peer" and is fine on the retry. The guard used to drop the
router on the first exception, which cost a whole TB-2.0 eval curve: a single
blip at step 20 left only the step-0 point for the rest of a 100-step run.
"""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.controller import (
    _EVAL_GUARD_ATTEMPTS,
    _EVAL_GUARD_MAX_FAILURES,
    Controller,
)


class _Guard:
    """The guard bound to a stand-in with just the state it touches."""

    def __init__(self) -> None:
        self.eval_generator_router = object()
        self._eval_rollout_workers = [object()]
        self._eval_guard_failures = 0

    guard = Controller._guard_eval_generators

    @property
    def disabled(self) -> bool:
        return self.eval_generator_router is None


def _flaky(num_failures: int):
    """A call factory that raises ``num_failures`` times, then succeeds."""
    state = {"calls": 0}

    async def make():
        state["calls"] += 1
        if state["calls"] <= num_failures:
            raise RuntimeError("gloo: Connection closed by peer")

    return make, state


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Skip the retry backoff so the tests do not wait on wall-clock."""
    real_sleep = asyncio.sleep

    async def instant(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant)


def test_retry_recovers_without_disabling():
    g = _Guard()
    make, state = _flaky(_EVAL_GUARD_ATTEMPTS - 1)

    assert asyncio.run(g.guard(make, what="pull")) is True
    assert state["calls"] == _EVAL_GUARD_ATTEMPTS, "each retry must re-issue the RPC"
    assert not g.disabled
    assert g._eval_guard_failures == 0


def test_one_exhausted_window_keeps_the_evaluator():
    """A whole failed window costs one validation point, not the curve."""
    g = _Guard()
    make, state = _flaky(999)

    assert asyncio.run(g.guard(make, what="pull")) is False
    assert state["calls"] == _EVAL_GUARD_ATTEMPTS
    assert not g.disabled, "the next step must get another chance"
    assert g._eval_guard_failures == 1


def test_disabled_only_after_consecutive_exhausted_windows():
    g = _Guard()
    make, _ = _flaky(999)

    for i in range(1, _EVAL_GUARD_MAX_FAILURES):
        assert asyncio.run(g.guard(make, what="pull")) is False
        assert not g.disabled, f"still up after {i} failed window(s)"

    assert asyncio.run(g.guard(make, what="pull")) is False
    assert g.disabled
    assert g._eval_rollout_workers == []


def test_success_resets_the_failure_streak():
    g = _Guard()
    failing, _ = _flaky(999)
    ok, _ = _flaky(0)

    # One short of the limit, then a success, then failures again: the streak must
    # restart, so the evaluator survives.
    for _ in range(_EVAL_GUARD_MAX_FAILURES - 1):
        asyncio.run(g.guard(failing, what="pull"))
    assert asyncio.run(g.guard(ok, what="pull")) is True
    assert g._eval_guard_failures == 0
    for _ in range(_EVAL_GUARD_MAX_FAILURES - 1):
        asyncio.run(g.guard(failing, what="pull"))
    assert not g.disabled


def test_cancellation_propagates_and_never_disables():
    """A shutdown must not be recorded as an evaluator failure."""
    g = _Guard()

    async def make():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(g.guard(make, what="pull"))
    assert not g.disabled
    assert g._eval_guard_failures == 0

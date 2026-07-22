# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run-ahead buffer of RolloutGroupWork shared between the data-input, rollout, and batcher loops.
NOTE: The buffer holds work slots, and not the finalized RolloutGroups necessarily.

"""

import asyncio
import collections
import enum
import time
from dataclasses import dataclass, field

from torchtitan.config import Configurable
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.observability import structured_logger as sl


class _RolloutGroupWorkState(enum.Enum):
    """Where a RolloutGroupWork is in the WAITING -> INFLIGHT -> FINALIZED lifecycle."""

    WAITING = "waiting"
    INFLIGHT = "inflight"
    FINALIZED = "finalized"


@dataclass(slots=True)
class RolloutGroupWork:
    """One prompt group's work, tracked through _RolloutGroupWorkState.

    The input loop sets `group_id` + `sample`; the buffer owns `state` and `rollout_group`
    (`init=False`, so the input loop can't set them).
    """

    group_id: int
    sample: object
    """Data input produced by `rollouter.get_training_sample()`;
    passed unchanged to the env in `rollouter.run_group_rollouts`."""
    state: _RolloutGroupWorkState = field(
        default=_RolloutGroupWorkState.WAITING, init=False
    )
    rollout_group: RolloutGroup | None = field(
        default=None, init=False
    )  # set once FINALIZED
    # Monotonic wall-clock stamps for slot-occupancy observability: admitted_ts when the
    # group enters the buffer (charges a slot), claimed_ts on WAITING -> INFLIGHT. A slot's
    # occupancy (now - admitted_ts) surfaces stuck rollouts that hold an off-policy slot
    # without finalizing (e.g. a hung Daytona sandbox), which starve the trainer.
    admitted_ts: float | None = field(default=None, init=False)
    claimed_ts: float | None = field(default=None, init=False)
    # TODO(async-rl): emit JSON lifecycle logging per RolloutGroupWork keyed by group_id:
    # admitted/claimed/finalized/batched/trained/dropped timestamps + policy version at admission and
    # at trainer consumption, for faithful end-to-end visibility.


class RolloutGroupWorkBuffer(Configurable):
    """Run-ahead buffer of RolloutGroupWork shared by the data-input, rollout, and batcher loops.

    Each entry is a RolloutGroupWork moving WAITING -> INFLIGHT -> FINALIZED. An active-slot budget caps
    run-ahead at `max_active_rollout_groups` active slots; the batcher takes the oldest-admitted
    FINALIZED group, skipping still-INFLIGHT stragglers, so a slow head does not stall it.

    For details on the buffer's callers, check the diagram in the controller.py file.

    NOTE: a work slot is **NOT** released when marked as FINALIZED or taken by the batcher.
    Instead, it is only released on `release_active_groups` calls by the trainer
    or data filtering. This is done this way so that we can guarantee we never have more
    than `max_active_rollout_groups` in the entire pipeline (buffer+queue+training).
    Otherwise, we would produce born-stale examples.

    Entry lifecycle vs active slot:
        entry:        WAITING -> INFLIGHT -> FINALIZED -> removed by take_finalized()
        active slot:  charged by add_work() ............ freed by release_active_groups()

    Example:
        # max_offpolicy_steps=1, num_groups_per_train_step=2 -> capacity=4
        await buffer.add_work(g0); await buffer.add_work(g1)   # 2/4 active
        await buffer.add_work(g2); await buffer.add_work(g3)   # 4/4 active (cap)
        g0 = await buffer.take_finalized()                     # g0 leaves the dict; still 4/4 active
        g1 = await buffer.take_finalized()                     # g1 leaves the dict; still 4/4 active
        slot_task = asyncio.create_task(buffer.wait_for_slot())  # waits: take_finalized did not free a slot
        assert not slot_task.done()
        await buffer.release_active_groups(2, reason="trained")  # trainer pulled -> a slot frees
        assert await slot_task                                   # wait_for_slot now returns
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Capacity is passed in by the controller from the run-ahead sizing."""

        strict_fifo: bool = False
        """take_finalized() order. False (default) = take-any: return the first
        FINALIZED group in admission order, skipping still-INFLIGHT stragglers (no
        head-of-line stall; the throughput default). True = strict FIFO: return the
        OLDEST group only once IT is finalized, stalling on a slow head. FIFO removes
        the take-any bias toward short/fast (=easy) rollouts in the trained batch, at
        the cost of the straggler stall -- a diagnostic knob for that bias."""

    def __init__(
        self,
        config: Config,
        *,
        max_active_rollout_groups: int,
        initial_active_rollout_groups: int | None = None,
    ) -> None:
        self._strict_fifo = config.strict_fifo
        if max_active_rollout_groups < 1:
            raise ValueError(
                "max_active_rollout_groups must be positive, got "
                f"{max_active_rollout_groups}"
            )
        if initial_active_rollout_groups is None:
            initial_active_rollout_groups = max_active_rollout_groups
        if not 1 <= initial_active_rollout_groups <= max_active_rollout_groups:
            raise ValueError(
                "initial_active_rollout_groups must be between 1 and "
                f"max_active_rollout_groups ({max_active_rollout_groups}), got "
                f"{initial_active_rollout_groups}"
            )
        self._max_active_rollout_groups = max_active_rollout_groups
        self._effective_active_rollout_groups = initial_active_rollout_groups
        self._active_rollout_groups = 0
        # metric: Per-flush peak active slots; reset on `.metrics()` call.
        self._active_rollout_groups_peak_since_flush = 0
        self._work_by_group_id: collections.OrderedDict[
            int, RolloutGroupWork
        ] = collections.OrderedDict()
        # TODO(async-rl): Current we use a condition that alerts ALL rollout workers. There is no need to
        # alert all of them. Consider changing it to an async queue + event.

        # One Condition guards all three waits (slot-free / claimable-WAITING / takeable-FINALIZED):
        # every mutation notify_all()s and waiters re-check their predicate.
        self._condition = asyncio.Condition()
        self._closed = False

    def _has_active_slot_available(self) -> bool:
        return self._active_rollout_groups < self._effective_active_rollout_groups

    async def wait_for_slot(self) -> bool:
        """Wait until one more rollout group may enter the active off-policy window.

        Example:
            # False means the buffer was closed, so the data input loop exits.
            group_index = 0
            while await buffer.wait_for_slot():
                await buffer.add_work(RolloutGroupWork(group_id=group_index, sample=sample))
                group_index += 1
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._closed or self._has_active_slot_available()
            )
            return not self._closed

    async def add_work(self, work: RolloutGroupWork) -> None:
        """Admit one rollout group as WAITING and charge one active slot."""
        async with self._condition:
            if not self._has_active_slot_available():
                raise RuntimeError(
                    "RolloutGroupWorkBuffer.add_work called without an active slot"
                )
            self._active_rollout_groups += 1
            self._active_rollout_groups_peak_since_flush = max(
                self._active_rollout_groups_peak_since_flush,
                self._active_rollout_groups,
            )
            work.admitted_ts = time.monotonic()
            self._work_by_group_id[work.group_id] = work
            self._condition.notify_all()

    async def claim_next(self) -> RolloutGroupWork | None:
        """Rollout loop: claim the oldest WAITING group (WAITING -> INFLIGHT). None once closed."""
        async with self._condition:
            while True:
                if self._closed:
                    return None
                for work in self._work_by_group_id.values():
                    if work.state is _RolloutGroupWorkState.WAITING:
                        work.state = _RolloutGroupWorkState.INFLIGHT
                        work.claimed_ts = time.monotonic()
                        return work
                await self._condition.wait()

    async def finalize_work(self, rollout_group: RolloutGroup) -> None:
        """Rollout loop: store the produced RolloutGroup on its work entry (INFLIGHT -> FINALIZED) and wake the batcher."""
        async with self._condition:
            work = self._work_by_group_id.get(rollout_group.group_id)
            if work is None:
                # run()'s shutdown called close(); drop the result.
                return
            work.rollout_group = rollout_group
            work.state = _RolloutGroupWorkState.FINALIZED
            self._condition.notify_all()

    @sl.log_trace_span("take_finalized")
    async def take_finalized(self) -> RolloutGroup | None:
        """Return the oldest-admitted FINALIZED group, skipping INFLIGHT groups.

        This is take-any, NOT strict FIFO: a slow straggler at the head no longer
        blocks the batcher. If multiple groups are finalized before the scan, their
        admission order breaks the tie; this is not a timestamped completion queue.
        A skipped straggler keeps holding its active slot, so the pipeline never
        exceeds max_active_rollout_groups, and the trainer still checks policy age
        at consumption time.

        Example:
            # head g0 still INFLIGHT, g1 FINALIZED -> returns g1 (no stall on g0)
            await buffer.take_finalized()
        """
        async with self._condition:
            while True:
                if self._closed:
                    return None
                if self._strict_fifo:
                    # Strict FIFO: only the OLDEST group is takeable; stall if it is
                    # still INFLIGHT (head-of-line). Removes take-any's short/fast bias
                    # in the trained batch. Restores pre-take-any behavior.
                    if self._work_by_group_id:
                        oldest_group_id, oldest_work = next(
                            iter(self._work_by_group_id.items())
                        )
                        if oldest_work.state is _RolloutGroupWorkState.FINALIZED:
                            del self._work_by_group_id[oldest_group_id]
                            self._condition.notify_all()
                            return oldest_work.rollout_group
                else:
                    for group_id, work in self._work_by_group_id.items():
                        if work.state is _RolloutGroupWorkState.FINALIZED:
                            del self._work_by_group_id[group_id]
                            self._condition.notify_all()
                            return work.rollout_group
                await self._condition.wait()  # nothing takeable yet -> wait

    async def release_active_groups(self, count: int, *, reason: str) -> None:
        """Free active slots: the trainer releases trained slots after its weight pull; the batcher
        releases untrainable/filtered slots immediately.

        Args:
            count:  Number of rollout groups leaving the active window.
            reason: Metric suffix such as `"trained"` or `"untrainable_group"`.

        Example:
            # trainer pulled weights after a step over 8 groups -> free their 8 slots
            await buffer.release_active_groups(8, reason="trained")
            # batcher dropped one zero-std group -> free its single slot
            await buffer.release_active_groups(1, reason="untrainable_group")
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return
        async with self._condition:
            if count > self._active_rollout_groups:
                raise RuntimeError(
                    f"release_active_groups({count}) exceeds active count {self._active_rollout_groups}"
                )
            self._active_rollout_groups -= count
            sl.log_trace_scalar({f"rollout_buffer/released/{reason}": float(count)})
            self._condition.notify_all()

    async def grow_effective_capacity(self) -> bool:
        """Grow cold-start admission headroom for retained downstream groups.

        A trainable group remains charged after the batcher takes it, so growing
        the effective capacity by one admits exactly one replacement generation
        group. Filtered and stale groups instead call ``release_active_groups`` and
        must not grow the capacity. Returns whether the capacity grew.
        """
        async with self._condition:
            if self._effective_active_rollout_groups == self._max_active_rollout_groups:
                return False
            self._effective_active_rollout_groups += 1
            sl.log_trace_scalar({"rollout_buffer/effective_capacity_growth": 1.0})
            self._condition.notify_all()
            return True

    async def close(self) -> None:
        """run() shutdown calls this once. Sets `_closed`, drops buffered work, and wakes every waiter.

        After this, all four waiters unblock and exit their loops: wait_for_slot() returns False, and
        claim_next()/take_finalized() return None.
        """
        async with self._condition:
            self._closed = True
            self._work_by_group_id.clear()
            self._condition.notify_all()

    def metrics(self) -> list[m.Metric]:
        """Trainer loop: point-in-time buffer gauges for this step; resets the per-flush peak."""
        states = [work.state for work in self._work_by_group_id.values()]
        state_enum = _RolloutGroupWorkState
        # Slot occupancy (seconds since admission) of the groups still holding an active slot
        # but not yet finalized (WAITING + INFLIGHT). max surfaces a stuck straggler occupying
        # a slot; mean is the typical in-flight wait. Both are 0 when the buffer is empty.
        now = time.monotonic()
        occupancy_secs = [
            now - work.admitted_ts
            for work in self._work_by_group_id.values()
            if work.state is not state_enum.FINALIZED and work.admitted_ts is not None
        ]
        out = [
            m.Metric(
                "rollout_buffer/slot_occupancy_max_sec",
                m.NoReduce(max(occupancy_secs, default=0.0)),
            ),
            m.Metric(
                "rollout_buffer/slot_occupancy_mean_sec",
                m.NoReduce(
                    sum(occupancy_secs) / len(occupancy_secs) if occupancy_secs else 0.0
                ),
            ),
            m.Metric(
                "rollout_buffer/num_groups_waiting",
                m.NoReduce(float(states.count(state_enum.WAITING))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_inflight",
                m.NoReduce(float(states.count(state_enum.INFLIGHT))),
            ),
            m.Metric(
                "rollout_buffer/num_groups_finalized",
                m.NoReduce(float(states.count(state_enum.FINALIZED))),
            ),
            m.Metric(
                "rollout_buffer/active_slots_in_use_peak",
                m.NoReduce(float(self._active_rollout_groups_peak_since_flush)),
            ),
            m.Metric(
                "rollout_buffer/available_active_slots",
                m.NoReduce(
                    float(
                        self._effective_active_rollout_groups
                        - self._active_rollout_groups
                    )
                ),
            ),
            m.Metric(
                "rollout_buffer/effective_active_group_capacity",
                m.NoReduce(float(self._effective_active_rollout_groups)),
            ),
            m.Metric(
                "rollout_buffer/max_active_group_capacity",
                m.NoReduce(float(self._max_active_rollout_groups)),
            ),
        ]
        # Next interval starts from the current gauge, not 0: slots stay occupied across a flush.
        self._active_rollout_groups_peak_since_flush = self._active_rollout_groups
        return out

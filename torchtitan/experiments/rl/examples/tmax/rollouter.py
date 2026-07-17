# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-agent rollouter for AI2 tmax tasks.

Forked from ``swe_r2e/rollouter.py`` (same shape: open an adapter session, boot a
fresh sandbox, run a host-side ReAct loop against the on-box adapter, then grade
and stamp the reward) but drives the FAITHFUL Vanillux agent loop
(``run_vanillux_loop``) rather than the swe_r2e host_loop. The tmax Qwen3.5-9B is
SFT'd under ``SWERLVanilluxSandboxEnv`` (single ``bash`` tool, vanillux prompts,
persistent shell, ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` to submit); the
swe_r2e host_loop's Bash/Read/Write/Edit tool set + SWE system prompt would put the
policy off-distribution, so we reproduce the tmax scaffold exactly while keeping the
host_loop transport (agent brain on the controller, each bash action dispatched to
the Daytona sandbox via ``sb.exec``). tmax specifics:

  1. Run the agent as ROOT. tmax tasks write to system paths (``/logs``,
     ``/output``, ``/home/user``, ``/app``). ``_RootSandbox`` forces every sandbox
     call to ``user="root"``.
  2. Honor the per-task ``workdir`` (best-guessed at data-prep time; /home/user,
     /app, or /workspace).
  3. Grade in place on the SUBMIT MARKER only: the tmax verifier inspects the
     container's live filesystem, so ``grade_tmax`` uploads the verifier into the
     agent's own sandbox and runs ``bash /tests/test.sh``. A rollout that never
     submits scores 0 (matches the env: tests run only on submit).

The standard scoring + advantage path is unchanged: the grade is stamped on the
last turn's ``env_rewards`` (key ``tmax_reward``) and read back by ``RewardTMax``.

Knobs read from env (the launcher sets these; see ``submit_swe_tmax_9b.sh``):
  ``SHIM_BIND_HOST`` / ``SHIM_PORT``  adapter bind address (default 127.0.0.1:18001)
  ``SWE_TIME_BUDGET_SEC``             per-agent wallclock (default 1200)
  ``TMAX_EVAL_TIMEOUT_SEC``           verifier run timeout (default 900)
  ``SWE_MAX_CONTEXT_LEN``             model context budget for the adapter session
  ``SWE_ROLLOUT_CONCURRENCY``         concurrently-active rollouts (default 16)
  ``TMAX_CALL_LIMIT`` / ``TMAX_TURN_MAX_TOKENS``  Vanillux step + per-turn caps
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from renderers import Renderer

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset, TMaxSample
from torchtitan.experiments.rl.examples.tmax.env import TMaxEnv
from torchtitan.experiments.rl.examples.tmax.grading import grade_tmax, seed_workspace
from torchtitan.experiments.rl.examples.tmax.rubric import RewardTMax, TMAX_REWARD_KEY
from torchtitan.experiments.rl.examples.tmax.vanillux_loop import run_vanillux_loop
from torchtitan.experiments.rl.harness import (
    AnthropicAdapter,
    boot_agent_sandbox,
    Sandbox,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import (
    GenerateFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.experiments.rl.types import RolloutTurnID

if TYPE_CHECKING:
    # Type-only: importing the generator module pulls in vLLM at import time.
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

logger = logging.getLogger(__name__)


class _RootSandbox:
    """Sandbox wrapper that forces every operation to run as ``root``.

    tmax tasks need root to touch system paths (``/logs``, ``/output``, ``/app``).
    This delegates to the underlying sandbox with the requested ``user`` overridden
    to ``root``, so ``run_vanillux_loop`` (and ``grade_tmax``) run entirely as root.
    """

    def __init__(self, inner: Sandbox) -> None:
        self._inner = inner

    @property
    def sandbox_id(self) -> str:
        return self._inner.sandbox_id

    async def exec(self, cmd: str, *, user: str = "root", **kwargs):
        return await self._inner.exec(cmd, user="root", **kwargs)

    async def write_file(self, sandbox_path: str, content, *, user: str = "root"):
        return await self._inner.write_file(sandbox_path, content, user="root")

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        return await self._inner.read_file(sandbox_path, user="root")


class _RolloutIssueGate:
    """Async slot gate that admits work in ascending priority (lowest
    ``(group_id, rollout_idx)`` first), NOT FIFO, with per-sibling release. It gives
    open-instruct-style id-ordered issue (low-id groups get generator capacity first
    and finish fresh), while a straggler holds only its OWN slot -- a later group
    starts the moment capacity frees, with no wait for any group to fully complete.

    Two admission modes, chosen by the caller from whether a whole group fits:

      - ``acquire_group(group_id, n)`` reserves ``n`` slots ATOMICALLY, so the group's
        n siblings dispatch together in one wave ("a group goes out in one step").
        Used when ``capacity >= group_size``.
      - ``acquire_sibling(priority)`` reserves 1 slot. Used only when
        ``capacity < group_size``, where a whole group physically cannot run at once
        (atomic issue is impossible); siblings then go strictly in id order but
        sub-batched.

    Both share one slot counter, so exactly ``capacity`` siblings run concurrently in
    either mode; ``release()`` returns one slot per finished sibling. Admission always
    serves the lowest-priority waiter and STOPS at the first that does not fit (never
    skipping it), which preserves id order and lets ``acquire_group`` reserve a whole
    wave -- at the cost of holding up to ``group_size - 1`` slots idle while a wave's
    worth of slots accumulates (bounded, and pooled across the concurrent groups'
    releases, so small in practice).

    Single-event-loop only (no threads): asyncio runs coroutine steps serially, so the
    counter/heap need no lock.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._available = capacity
        # Min-heap of (priority, tiebreak, need, future). tiebreak keeps the heap
        # totally ordered on equal priority (and never compares futures).
        self._waiters: list[tuple[tuple[int, int], int, int, asyncio.Future]] = []
        self._tiebreak = itertools.count()

    @property
    def capacity(self) -> int:
        return self._capacity

    async def acquire_group(self, group_id: int, n: int) -> None:
        """Reserve ``n`` slots atomically for a group (sorts before its siblings via
        the ``-1`` sub-key), so the group's n siblings dispatch as one wave."""
        await self._acquire((group_id, -1), n)

    async def acquire_sibling(self, priority: tuple[int, int]) -> None:
        """Reserve a single slot (non-atomic fallback when a group cannot fit)."""
        await self._acquire(priority, 1)

    async def _acquire(self, priority: tuple[int, int], need: int) -> None:
        if need > self._capacity:
            raise ValueError(f"acquire need={need} exceeds capacity={self._capacity}")
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._tiebreak), need, fut))
        self._try_admit()  # may grant synchronously if this is now the head and fits
        if fut.done():
            return
        try:
            await fut
        except asyncio.CancelledError:
            # Granted-then-cancelled: return the `need` slots we were handed.
            # Pending-then-cancelled: the dead future is dropped by _try_admit's
            # done() check, so there is nothing to return.
            if fut.done() and not fut.cancelled():
                self._release(need)
            raise

    def release(self, count: int = 1) -> None:
        """Return ``count`` slots (one per finished sibling) and admit any waiters
        that now fit."""
        self._release(count)

    def _release(self, count: int) -> None:
        self._available += count
        self._try_admit()

    def _try_admit(self) -> None:
        # Grant from the lowest-priority waiter down while it fits; STOP at the first
        # that does not (do not skip it -> id order + whole-wave reservation).
        while self._waiters:
            _, _, need, fut = self._waiters[0]
            if fut.done():  # cancelled/settled: drop and continue
                heapq.heappop(self._waiters)
                continue
            if self._available >= need:
                heapq.heappop(self._waiters)
                self._available -= need
                fut.set_result(None)
            else:
                break


class TMaxRollouter(Rollouter):
    """Drives a host-side ReAct agent (as root) in a sandbox per sibling, then runs
    the tmax verifier in that same sandbox."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: TMaxDataset.Config = field(
            default_factory=lambda: TMaxDataset.Config(seed=42)
        )
        validation_dataset: TMaxDataset.Config = field(
            default_factory=lambda: TMaxDataset.Config(seed=99, shuffle=False)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardTMax.Config(weight=1.0)],
                # An errored / timed-out agent gets no learning signal.
                error_reward=0.0,
                truncation_reward=0.0,
            )
        )
        # Placeholder env (the agent loop runs in-sandbox; see env.py).
        message_env: TMaxEnv.Config = field(default_factory=TMaxEnv.Config)
        token_env: TokenEnv.Config = field(default_factory=TokenEnv.Config)
        # Centered (mean-baseline only), NOT std-normalized: matches the tmax
        # recipe's ``--advantage_normalization_type centered`` (qwen35_9b.sh).
        # Dividing by the group std amplifies rare-outcome advantages for
        # imbalanced binary-reward groups (e.g. a 30/32-pass group's 2 failures
        # get advantage ~ -3.9), which distorts the gradient and suppresses reward
        # growth; the recipe centers only to keep the advantage in [-1, 1].
        advantage: AdvantageEstimator.Config = field(
            default_factory=lambda: AdvantageEstimator.Config(
                should_std_normalize=False
            )
        )

        # Run knobs as CONFIG (not env), so they are serialized into the W&B run
        # config and each run's differences are visible. config_registry resolves
        # them from the launcher env once; the RolloutWorker pool overrides
        # rollout_concurrency to its per-worker share.
        rollout_concurrency: int = 16
        """Max concurrently-ACTIVE rollouts (per worker process). The pool total is
        num_rollout_workers x this. Gates per-turn fs ops so the adapter stays
        responsive; all groups are still collected in waves."""

        time_budget_sec: int = 1200
        """Per-rollout agent wall-clock budget (the vanillux loop stops after this)."""

        eval_timeout_sec: int = 600
        """Verifier (test.sh) run timeout."""

        max_context_tokens: int = 32768
        """Model context budget for the adapter session."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._time_budget_sec = config.time_budget_sec
        self._eval_timeout_sec = config.eval_timeout_sec
        self._max_context_tokens = config.max_context_tokens
        # Whole-rollout wall-clock guard: agent budget + eval + boot buffer.
        self._guard_sec = self._time_budget_sec + self._eval_timeout_sec + 300
        # Per-worker rollout-issue gate (one rollouter per worker proc). Admits work
        # lowest-(group_id, rollout_idx)-first (not FIFO), so the lowest-id groups get
        # generator capacity first and finish fresh (open-instruct's id-ordered pool
        # admission). When a whole group fits (capacity >= group_size) it reserves the
        # group atomically so its siblings dispatch in one wave; slots release per
        # sibling, so a straggler never blocks a later group. NOTE: ordering is
        # per-worker; with a RolloutWorker pool the controller stripes group_ids
        # round-robin across workers, so each worker orders its own strided subset
        # (open-instruct's pool is a single shared actor = globally ordered; a global
        # gate would need cross-process coordination).
        self._rollout_gate = _RolloutIssueGate(config.rollout_concurrency)
        self._adapter: AnthropicAdapter | None = None
        self._adapter_lock = asyncio.Lock()

    async def _ensure_adapter(self, renderer: Renderer) -> AnthropicAdapter:
        if self._adapter is None:
            async with self._adapter_lock:
                if self._adapter is None:
                    # Direct in-process use: build the adapter for its Anthropic
                    # translation + TITO turn capture, but do NOT start() an HTTP
                    # server -- the vanillux loop calls adapter.complete() directly
                    # (no loopback HTTP, no per-worker port).
                    self._adapter = AnthropicAdapter(renderer=renderer)
        return self._adapter

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: TMaxSample,
        group_id: int,
        group_size: int,
        sampling: "SamplingConfig",
        renderer: Renderer,
    ) -> RolloutGroup:
        """Run + grade one prompt group of terminal-agent rollouts."""
        adapter = await self._ensure_adapter(renderer)

        # Atomic group issue when a whole group fits the gate: reserve group_size slots
        # in id order so this group's siblings dispatch as one wave, ahead of any
        # higher-id group. If a group cannot fit (capacity < group_size), fall back to
        # per-sibling id-ordered admission inside _run_agent_rollout (see gate docs).
        atomic_issue = self._rollout_gate.capacity >= group_size
        if atomic_issue:
            await self._rollout_gate.acquire_group(group_id, group_size)
        # TODO(async-rl): the group_size reservation is balanced by the per-sibling
        # release() in each _run_agent_rollout finally. If this coroutine is cancelled
        # AFTER acquire_group grants but BEFORE the siblings enter their try, the not-yet-
        # started siblings never release, leaking their slots. Today the only cancel is
        # full-process shutdown (controller close), where it is harmless (the process
        # exits). If a per-group mid-flight cancel/timeout is ever added, wrap the
        # gather in try/finally and release the count of siblings that did not run.

        results = await asyncio.gather(
            *(
                self._run_agent_rollout(
                    adapter=adapter,
                    generate_fn=generate_fn,
                    sample=sample,
                    group_id=group_id,
                    rollout_idx=i,
                    sampling=sampling,
                    renderer=renderer,
                    gate_per_sibling=not atomic_issue,
                )
                for i in range(group_size)
            )
        )
        rollouts = [rollout for rollout, _, _ in results]
        submitted_flags = [submitted for _, submitted, _ in results]
        fmt_errors_list = [fmt for _, _, fmt in results]

        # Standard scoring + advantage path (mirrors Rollouter.run_group_rollouts).
        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        # Fraction of siblings that never emitted the submit marker (errored or ran to
        # the budget without submitting) -> the verifier never runs -> auto reward 0.
        # Mirrors open-instruct's val/non_submitting_completion_fraction; a high value
        # means the scaffold (not the task difficulty) is capping reward.
        nonsubmit_frac = 1.0 - sum(submitted_flags) / len(submitted_flags)
        # Format errors (malformed tool-calls) per rollout + fraction of rollouts that
        # hit any -- surfaces the vanillux tool-call parse-failure rate on wandb.
        fmt_errors_mean = sum(fmt_errors_list) / len(fmt_errors_list)
        fmt_error_frac = sum(1.0 for f in fmt_errors_list if f > 0) / len(
            fmt_errors_list
        )
        group = RolloutGroup(
            group_id=group_id,
            rollouts=rollouts,
            metrics=[
                m.Metric("rollout/nonsubmit_frac", m.Mean(nonsubmit_frac)),
                m.Metric("rollout/format_errors_mean", m.Mean(fmt_errors_mean)),
                m.Metric("rollout/format_error_frac", m.Mean(fmt_error_frac)),
            ],
        )
        advantages = self.advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        self._maybe_annotate_zero_std(sample, rollouts)
        return group

    def _maybe_annotate_zero_std(
        self, sample: TMaxSample, rollouts: list[Rollout]
    ) -> None:
        """Record this prompt's ``instance_id`` under ``SWE_ZERO_STD_DIR`` when its
        group has zero reward variance (all-pass or all-fail = no learning signal, so
        it is dropped by ``drop_zero_std_reward_groups``). A later run points
        ``TMaxDataset.skip_ids_path`` at the same dir to stop sampling these prompts.

        Writes ONE small file per prompt (``<instance_id>.json``), mirroring the
        rollout-trace dump: a write-once-per-file pattern is the one that persists on
        the manifold (oilfs) FUSE mount and is safe across the pooled RolloutWorker
        processes -- unlike appending to a single shared file, which object-store FUSE
        does not handle reliably. Re-encountering a prompt just overwrites its file
        (natural dedup). Best-effort; never raises into the rollout.
        """
        dump_dir = os.environ.get("SWE_ZERO_STD_DIR", "")
        if not dump_dir:
            return
        rewards = [r.reward for r in rollouts if r.reward is not None]
        if len(rewards) < 2 or statistics.pstdev(rewards) != 0.0:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            safe = sample.instance_id.replace("/", "_")
            path = os.path.join(dump_dir, f"{safe}.json")
            with open(path, "w") as f:
                json.dump({"instance_id": sample.instance_id, "reward": rewards[0]}, f)
        except OSError as e:
            logger.warning(f"[tmax] zero-std annotate failed for {dump_dir}: {e}")

    async def _run_agent_rollout(
        self,
        *,
        adapter: AnthropicAdapter,
        generate_fn: GenerateFn,
        sample: TMaxSample,
        group_id: int,
        rollout_idx: int,
        sampling: "SamplingConfig",
        renderer: Renderer,
        gate_per_sibling: bool,
    ) -> tuple[Rollout, bool, int]:
        """Boot a sandbox, run the agent as root, grade the task in place.

        Always returns ``(Rollout, submitted, fmt_errors)`` (errors caught + marked
        terminal) so one bad sibling never fails the whole group. ``submitted`` is
        whether the agent emitted the submit marker (False on any error / no-submit);
        ``fmt_errors`` is the tool-call parse-failure count. The caller aggregates both
        into the group's ``rollout/nonsubmit_frac`` and ``rollout/format_*`` metrics.
        """
        rollout_id = RolloutTurnID(
            group_id=group_id, rollout_id=rollout_idx, turn_id=0
        ).to_string(include_turn=False)

        status = RolloutStatus.ERROR
        reward = 0.0
        error_msg = ""
        submitted = False
        fmt_errors = 0  # total format errors this rollout (from run_vanillux_loop)
        # Slot accounting: in atomic mode run_group_rollouts already reserved this
        # sibling's slot as part of the group's wave, so acquire only in the per-sibling
        # fallback (capacity < group_size). Either way we release exactly one slot in
        # the finally, so the group's group_size reservation stays balanced.
        if gate_per_sibling:
            await self._rollout_gate.acquire_sibling((group_id, rollout_idx))
        try:
            # open_session inside the try so its failure still reaches the finally
            # (release + finish_session): the gate slot -- reserved atomically by
            # run_group_rollouts in atomic mode, or by acquire_sibling above in the
            # fallback -- must be released even if open_session raises.
            adapter.open_session(
                rollout_id,
                generate_fn=generate_fn,
                sampling=sampling,
                routing_session_id=rollout_id,
                max_context_tokens=self._max_context_tokens,
            )
            async with asyncio.timeout(self._guard_sec):
                # host_loop drives the sandbox with bash directly; it never runs the
                # Claude Code CLI, so skip the curl-based install (the tmax task
                # images have no curl, which would otherwise fail every boot).
                async with boot_agent_sandbox(sample.image, install_claude=False) as sb:
                    # Force every tool command to run as root (tmax tasks touch
                    # system paths); the faithful Vanillux loop dispatches bash here.
                    root_sb = _RootSandbox(sb)
                    # Seed the agent-facing inputs (environment/seeds/* -> /workspace)
                    # BEFORE the agent runs -- upstream seeds at reset. Without this,
                    # seed-bearing tasks are unsolvable (inputs absent during rollout).
                    # Grading fixtures (tests/*) are uploaded later by grade_tmax.
                    await seed_workspace(root_sb, sample.tmax)
                    _turns, submitted, fmt_errors = await run_vanillux_loop(
                        root_sb,
                        task=sample.problem_statement,
                        session_id=rollout_id,
                        adapter=adapter,
                        time_budget_sec=self._time_budget_sec,
                    )
                    # tmax runs the verifier only on the submit marker; a rollout that
                    # never submits scores 0 (matches SWERLVanilluxSandboxEnv). No
                    # git_diff: grade the agent's OWN sandbox in place.
                    if submitted:
                        reward = await grade_tmax(
                            sb,
                            sample.tmax,
                            workdir=sample.workdir,
                            timeout_sec=self._eval_timeout_sec,
                        )
                    else:
                        reward = 0.0
                status = RolloutStatus.COMPLETED
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[tmax] %s: wall-clock guard fired", rollout_id)
            status = RolloutStatus.ERROR_TIMEOUT
            error_msg = "wall_clock_timeout"
        except Exception as e:
            logger.exception("[tmax] %s: rollout failed", rollout_id)
            status = RolloutStatus.ERROR
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            self._rollout_gate.release()
            captured = await adapter.finish_session(rollout_id)

        # Drop empty-completion turns so rollout_to_training_samples only sees
        # trainable turns (a non-final empty completion would otherwise raise).
        turns: list[RolloutTurn] = [
            RolloutTurn(
                rollout_id=RolloutTurnID(
                    group_id=group_id, rollout_id=rollout_idx, turn_id=turn_idx
                ),
                prompt_token_ids=ct.prompt_token_ids,
                completion_token_ids=ct.completion_token_ids,
                completion_logprobs=ct.completion_logprobs,
                min_policy_version=ct.min_policy_version,
                max_policy_version=ct.max_policy_version,
            )
            for turn_idx, ct in enumerate(
                ct for ct in captured if ct.completion_token_ids
            )
        ]

        # Load-bearing invariant: the off-policy age filter (controller) takes
        # min()/max() over each sample's turn policy versions, which TypeErrors
        # on None. Over-budget/empty turns carry version None and are dropped
        # above; assert none leaked so a violation fails here (clear) instead of
        # deep in the batcher.
        assert all(
            t.min_policy_version is not None and t.max_policy_version is not None
            for t in turns
        ), (
            f"{rollout_id}: a trainable RolloutTurn has a None policy version "
            "(empty-turn drop invariant broke)"
        )

        if not turns:
            status = RolloutStatus.ERROR
            if not error_msg:
                n_cap = len(captured)
                n_empty = sum(1 for ct in captured if not ct.completion_token_ids)
                error_msg = (
                    f"no_trainable_turns: captured={n_cap} empty_completions={n_empty}"
                )
        else:
            turns[-1].env_rewards = {TMAX_REWARD_KEY: float(reward)}

        logger.info(
            "[tmax] %s: status=%s reward=%.2f turns=%d",
            rollout_id,
            status,
            reward,
            len(turns),
        )
        self._maybe_dump_trace(
            rollout_id=rollout_id,
            sample=sample,
            captured=captured,
            renderer=renderer,
            status=str(status),
            reward=reward,
            submitted=submitted,
            fmt_errors=fmt_errors,
            error_msg=error_msg,
        )
        return (
            Rollout(
                group_id=group_id,
                rollout_id=rollout_idx,
                status=status,
                turns=turns,
            ),
            submitted,
            fmt_errors,
        )

    def _maybe_dump_trace(
        self,
        *,
        rollout_id: str,
        sample: TMaxSample,
        captured: list,
        renderer: Renderer,
        status: str,
        reward: float,
        submitted: bool = False,
        fmt_errors: int = 0,
        error_msg: str = "",
    ) -> None:
        """Write a human-readable per-rollout training trace when
        ``SWE_ROLLOUT_DUMP_DIR`` is set. Format mirrors the open-instruct diagnostic
        trace: a summary header (reward / finish / submitted / num tool calls /
        response length) followed by the FULL decoded trajectory with the model's
        actions (``<tool_call>``) and the sandbox outputs (``<tool_response>``)
        interleaved -- reconstructed from the TITO-bridged prompts (turn N+1's prompt
        is turn N's prompt+completion + the new tool_response, so the growing prefix
        recovers each turn's sandbox output). Best-effort; never raises."""
        dump_dir = os.environ.get("SWE_ROLLOUT_DUMP_DIR", "")
        if not dump_dir:
            return
        try:
            tokenizer = getattr(renderer, "tokenizer", None) or getattr(
                renderer, "_tokenizer", None
            )

            def _decode(ids: list[int]) -> str:
                if tokenizer is None or not ids:
                    return ""
                return tokenizer.decode(ids, skip_special_tokens=False)

            # Reconstruct the full interleaved token stream. TITO invariant: each turn's
            # prompt extends the previous prompt+completion, so appending each
            # completion and then the next prompt's delta recovers the whole
            # system+task+<tool_call>+<tool_response>+... conversation. A branch
            # (extends_previous False, e.g. compaction) resets to that turn's fresh
            # prompt (rare for tmax vanillux; noted inline).
            full_ids: list[int] = list(captured[0].prompt_token_ids) if captured else []
            branch_turns: list[int] = []
            for i, ct in enumerate(captured):
                full_ids += list(ct.completion_token_ids)
                if i + 1 < len(captured):
                    nxt = list(captured[i + 1].prompt_token_ids)
                    if nxt[: len(full_ids)] == full_ids:
                        full_ids += nxt[len(full_ids) :]  # tool_response delta
                    else:
                        branch_turns.append(i + 1)
                        full_ids = nxt  # history rewrite -> fresh render
            full_text = _decode(full_ids)

            response_len = sum(len(ct.completion_token_ids) for ct in captured)
            num_tool_calls = full_text.count("<tool_call>")
            last_finish = captured[-1].finish_reason if captured else None
            any_length_finish = any(ct.finish_reason == "length" for ct in captured)
            outcome = "SUCCESS" if reward and reward > 0 else "FAIL"

            header = (
                "=" * 90
                + f"\nTMAX-9B rollout trace  ({outcome}, reward={reward})\n"
                + "=" * 90
                + f"\ninstance_id    : {sample.instance_id}"
                + f"\nimage          : {sample.image}"
                + f"\nrollout_id     : {rollout_id}"
                + f"\nstatus         : {status}   submitted: {submitted}"
                + f"\nreward         : {reward}"
                + f"\nfinish_reason  : {last_finish}   any length-cap turn: {any_length_finish}"
                + f"\nnum turns      : {len(captured)}   num tool calls: {num_tool_calls}"
                + f"\nresponse length: {response_len} tokens (model-generated, all turns)"
                + f"\nformat_errors  : {fmt_errors}"
                + (f"\nerror          : {error_msg}" if error_msg else "")
                + (
                    f"\nNOTE branch/re-render at turns {branch_turns} (TITO not continued)"
                    if branch_turns
                    else ""
                )
                + "\n"
                + "=" * 90
                + "\nAGENT TRAJECTORY (decoded; <tool_call>=model action, "
                + "<tool_response>=sandbox output)\n"
                + "=" * 90
                + "\n"
            )

            os.makedirs(dump_dir, exist_ok=True)
            safe = rollout_id.replace("/", "_")
            path = os.path.join(dump_dir, f"{safe}.txt")
            with open(path, "w") as f:
                f.write(header)
                f.write(full_text)
                f.write("\n")
            logger.info("[tmax] rollout trace dumped: %s", path)
        except Exception as e:
            logger.warning("[tmax] rollout trace dump failed: %s", e)

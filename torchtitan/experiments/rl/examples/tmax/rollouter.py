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
        # Per-worker rollout-concurrency semaphore (one rollouter per worker proc).
        self._rollout_sem = asyncio.Semaphore(config.rollout_concurrency)
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
    ) -> tuple[Rollout, bool]:
        """Boot a sandbox, run the agent as root, grade the task in place.

        Always returns ``(Rollout, submitted)`` (errors caught + marked terminal) so
        one bad sibling never fails the whole group. ``submitted`` is whether the agent
        emitted the submit marker (False on any error / no-submit) -- the caller
        aggregates it into the group's ``rollout/nonsubmit_frac`` metric.
        """
        rollout_id = RolloutTurnID(
            group_id=group_id, rollout_id=rollout_idx, turn_id=0
        ).to_string(include_turn=False)
        adapter.open_session(
            rollout_id,
            generate_fn=generate_fn,
            sampling=sampling,
            routing_session_id=rollout_id,
            max_context_tokens=self._max_context_tokens,
        )

        status = RolloutStatus.ERROR
        reward = 0.0
        error_msg = ""
        submitted = False
        fmt_errors = 0  # total format errors this rollout (from run_vanillux_loop)
        sem = self._rollout_sem
        await sem.acquire()
        try:
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
            sem.release()
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
        error_msg: str = "",
    ) -> None:
        """Write a human-readable per-rollout training trace when
        ``SWE_ROLLOUT_DUMP_DIR`` is set (the tmax task, grade, and every captured
        turn's decoded completion). Best-effort; never raises into the rollout."""
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

            record = {
                "rollout_id": rollout_id,
                "instance_id": sample.instance_id,
                "image": sample.image,
                "status": status,
                "error": error_msg,
                "reward": reward,
                "stop_token_ids": list(renderer.get_stop_token_ids()),
                "num_turns": len(captured),
                "turns": [
                    {
                        "turn": i,
                        "prompt_tokens": len(ct.prompt_token_ids),
                        "completion_tokens": len(ct.completion_token_ids),
                        "finish_reason": ct.finish_reason,
                        "stop_reason": getattr(ct, "stop_reason", None),
                        "extends_previous": ct.extends_previous,
                        "completion_token_ids_tail": list(
                            ct.completion_token_ids[-16:]
                        ),
                        "completion_text": _decode(ct.completion_token_ids),
                    }
                    for i, ct in enumerate(captured)
                ],
            }
            os.makedirs(dump_dir, exist_ok=True)
            safe = rollout_id.replace("/", "_")
            path = os.path.join(dump_dir, f"{safe}.json")
            with open(path, "w") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            logger.info("[tmax] rollout trace dumped: %s", path)
        except Exception as e:
            logger.warning("[tmax] rollout trace dump failed: %s", e)

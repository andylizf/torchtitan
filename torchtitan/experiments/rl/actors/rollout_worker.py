# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RolloutWorker: runs group rollouts in its own process, off the controller's GIL.

The controller's single event loop otherwise drives every rollout's agent
orchestration -- the vanillux ReAct loop, the Anthropic->generate shim, the
Daytona HTTP client, and grading -- so at high rollout concurrency the GIL
serializes that per-turn Python and caps throughput regardless of how many
sandboxes or generators are available.

This actor moves ``run_group_rollouts`` into a pool of CPU worker processes
(co-located on the generator hosts). Each worker owns its own ``Rollouter``
(with a local 127.0.0.1 shim on a per-worker port), ``Renderer``, and a
generate-only generator router. The controller sends one RPC per group and
round-robins those RPCs across workers. Multiple async ``run_group`` calls overlap
inside each worker; ``train.py`` pins Monarch's concurrent endpoint dispatch for
this behavior. The finalized groups return independently, while the off-policy
buffer, batcher, trainer, and weight sync all stay in the controller.

Only two payloads cross the Monarch RPC boundary: the raw ``sample`` in, and the
``RolloutGroup`` out (which has to reach the trainer anyway).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from monarch.actor import Actor, endpoint

from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.controller_metrics import compute_rollout_metrics
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.types import RoutingContext
from torchtitan.observability import structured_logger as sl

if TYPE_CHECKING:
    from torchtitan.experiments.rl.controller import Controller


class RolloutWorker(Actor):
    """One CPU process that runs group rollouts off the controller's GIL.

    Spawned as its own 1-proc mesh, co-located with the generators. ``setup``
    hands it the shared generator actor refs so it can build its own generate
    router; each ``run_group`` call runs + scores one prompt group and returns
    the ``RolloutGroup``.

    Args:
        config: The controller config (its ``renderer``, ``rollouter``,
            ``generator``, ``generator_router``, ``async_loop``, and
            ``hf_assets_path`` fields are reused verbatim so a worker's rollouts
            are identical to the in-controller path).
        rollout_concurrency: This worker's own rollout-concurrency cap (the
            controller splits the global ``rollout_concurrency`` target across the
            pool). Set into the rollouter Config (via replace) before build, so each
            worker process caps its own concurrent rollouts from config -- not env.
    """

    def __init__(
        self, config: "Controller.Config", *, rollout_concurrency: int
    ) -> None:
        self.config = config
        self.renderer = config.renderer.build(tokenizer_path=config.hf_assets_path)
        # Same sampling config the controller builds (seed + renderer stop tokens);
        # the rollouter offsets the seed per sample.
        self._sampling = replace(
            config.generator.sampling,
            seed=config.generator.debug.seed,
            stop_token_ids=list(self.renderer.get_stop_token_ids()),
        )
        # Per-worker concurrency: override the rollouter's rollout_concurrency with
        # this worker's share (pool total = num_workers x this) via config, so the
        # rollouter builds its semaphore from config -- no process-wide env. The
        # hasattr guard keeps the pool usable with any Rollouter.Config.
        rollouter_config = config.rollouter
        if hasattr(rollouter_config, "rollout_concurrency"):
            rollouter_config = replace(
                rollouter_config, rollout_concurrency=rollout_concurrency
            )
        self._rollouter: Rollouter = rollouter_config.build()
        self._generator_router: InterGeneratorRouter | None = None

    @endpoint
    async def setup(self, generators: list) -> None:
        """Build this worker's generate-only router over the shared generator actors."""
        self._generator_router = self.config.generator_router.build(
            generators=generators
        )

    @endpoint
    async def run_group(self, *, sample: object, group_id: int) -> RolloutGroup:
        """Run + score one prompt group; return the finalized RolloutGroup."""
        if self._generator_router is None:
            raise RuntimeError("RolloutWorker.run_group called before setup()")
        with sl.log_trace_span("worker_run_group"):
            generate_fn = self._make_generate_fn()
            group = await self._rollouter.run_group_rollouts(
                generate_fn=generate_fn,
                sample=sample,
                group_id=group_id,
                group_size=self.config.async_loop.group_size,
                sampling=self._sampling,
                renderer=self.renderer,
            )
            # Preserve rollouter-set group metrics (e.g. tmax nonsubmit_frac /
            # format_errors); append the standard computed ones, don't overwrite.
            group.metrics = compute_rollout_metrics(
                prefix="rollout", rollouts=group.rollouts
            ) + list(group.metrics)
        return group

    def _make_generate_fn(self) -> GenerateFn:
        """Route a completion through this worker's generator router.

        Mirror of ``Controller._make_generate_fn`` (generate path only): sticky
        routing on ``routing_session_id`` keeps a sample's turns on one
        generator for prefix-KV reuse.
        """
        router = self._generator_router

        @sl.log_trace_span("generate")
        async def generate(
            prompt_token_ids: list[int],
            *,
            request_id: str,
            routing_session_id: str | None = None,
            sampling_config: SamplingConfig | None = None,
        ):
            result = await router.route(
                "generate",
                prompt_token_ids,
                request_id=request_id,
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
                metrics_prefix="generator",
                routing_ctx=RoutingContext(
                    estimated_cost=1,
                    session_id=routing_session_id,
                ),
            )
            # route returns a per-rank ValueMesh; all ranks return the same value.
            return result.get(0)

        return generate

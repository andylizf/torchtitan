# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TLDR:
_data_input_loop -> _rollout_loop -> _batcher_loop -> training_batch_queue -> _trainer_loop
           |               ^                    ^
           v               |                    |
           +-------RolloutGroupWorkBuffer-------+

Detailed diagram:

_data_input_loop                                      _rollout_loop[N] (group workers)
+--------------------------------------------------+  +--------------------------------------------------+
| group_buffer.wait_for_slot()                     |  | work = group_buffer.claim_next()                  |
| sample = rollouter.get_training_sample()         |  | group = rollouter.run_group_rollouts(work.sample) |
| work = RolloutGroupWork(group_id, sample)        |  | group_buffer.finalize_work(group)                 |
| group_buffer.add_work(work)                      |  +-----------------------+--------------------------+
+-----------------------+--------------------------+                          ^ |
                        |                                                     | |
                        | adds work entry                                     | | updates same entry
                        v                                                     | v
RolloutGroupWorkBuffer
+---------------------------------------------------------------------------------------------------------------------+
| active slots = (max_offpolicy_steps + 1) * num_groups_per_train_step                                                |
|                                                                                                                     |
| caller            group_buffer call                                            state / active slot                  |
| _data_input_loop  add_work(RolloutGroupWork)                                   WAITING; slot acquired               |
| _rollout_loop[N]  claim_next()                                                 WAITING -> INFLIGHT                  |
| _rollout_loop[N]  finalize_work(RolloutGroup)                                  INFLIGHT -> FINALIZED                |
| _batcher_loop     RolloutGroup = take_finalized()                              FINALIZED -> taken (slot still held) |
| _batcher_loop     release_active_groups(1, "untrainable_group")                slot released                        |
| _trainer_loop     release_active_groups(num_groups_per_train_step, "trained")  slots released after weight pull     |
+---------------------------------------------------------------------------------------------------------------------+
                                                  |
                                                  | group = group_buffer.take_finalized()
                                                  v
_batcher_loop
+----------------------------------------------------------------------------------------+
| training_sample_group = training_sample_builder.build_from_group(rollout_group=group)  |
| if no trainable samples: group_buffer.release_active_groups(1, "untrainable_group")    |
| maybe_training_batch = batcher.add_training_samples(training_sample_group)             |
| training_batch_queue.put(TrainingBatch)                                                |
+-----------------------+------------------------------+---------------------------------+
                        |  ^                           |
          add group     |  | maybe_training_batch      | put batch
                        v  |                           v
Batcher                                              training_batch_queue
+-------------------------------------------------+   +----------------------------------------------+
| accumulated TrainingSampleGroups                |   | size 1; holds TrainingBatch | None           |
| pack at num_groups_per_train_step               |   +---------------------+------------------------+
+-------------------------------------------------+                         |
                                                                         | packed = training_batch_queue.get()
                                                                         v
_trainer_loop
+----------------------------------------------------------------------------------------------------------+
| train batch -> optim -> push/pull weights -> buffer.release_active_groups(num_groups_per_train_step)     |
+----------------------------------------------------------------------------------------------------------+

Backpressure (each loop: what it consumes/produces, and what gates each side):
_data_input_loop
  produces: RolloutGroupWork into group_buffer
    waits for:    a free active slot (group_buffer.wait_for_slot)
    unblocked by: _trainer_loop release_active_groups(num_groups_per_train_step, "trained") after the pull
                  (and _batcher_loop release_active_groups(1,"untrainable_group"))
_rollout_loop[N]
  consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
    waits for:    a claimable WAITING entry
    unblocked by: _data_input_loop group_buffer.add_work()
  produces: RolloutGroup (group_buffer.finalize_work)
    waits for:    nothing (admits its own claimed slot)
    unblocked by: n/a

_batcher_loop
  consumes: the oldest FINALIZED group (group_buffer.take_finalized)
    waits for:    the oldest group becoming FINALIZED
    unblocked by: _rollout_loop[N] group_buffer.finalize_work()
  produces: TrainingBatch (training_batch_queue.put)
    waits for:    a free training_batch_queue slot (maxsize=1)
    unblocked by: _trainer_loop training_batch_queue.get()
_trainer_loop
  consumes: a TrainingBatch (training_batch_queue.get)
    waits for:    a TrainingBatch in the queue
    unblocked by: _batcher_loop training_batch_queue.put()
"""

import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# PYTORCH_CUDA_ALLOC_CONF is set in torchtitan/experiments/rl/__init__.py (before torch is imported)
# and in train.py; see the note there.
import torch  # noqa: F401
import torchstore as ts
from monarch.actor import ProcMesh, this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import CompileConfig, Configurable
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.rollout_worker import RolloutWorker
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import Batcher
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.components.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.experiments.rl.controller_metrics import (
    combine_microbatch_metrics,
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    compute_rollout_metrics,
    MetricsTimer,
)
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.types import RoutingContext
from torchtitan.experiments.rl.types import Completion, TrainingBatch
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ValidationConfig:
    """Held-out validation that runs at the start and end of training, and
    optionally every ``interval`` steps in between."""

    num_samples: int = 20
    """Held-out prompts scored greedily (temp=0, n=1) per validation pass. 0 skips validation."""

    interval: int = 0
    """Run a mid-training validation pass every ``interval`` optimizer steps (in addition to
    the start/end passes). 0 = only start/end. The pass reuses idle generator capacity and
    disjoint (negative) group ids, so it overlaps ongoing training-rollout collection; it does
    add its own wall time to the step it runs on (one greedy validation rollout's latency)."""


@dataclass(kw_only=True, slots=True)
class AsyncLoopConfig(Configurable.Config):
    num_training_steps: int = 10
    """Optimizer steps to run."""

    num_groups_per_train_step: int = 8
    """Global number of prompt groups, across all DPs, whose surviving rollouts compose
    one train step (the global_batch_size, in groups)."""

    group_size: int = 8
    """Sibling rollouts sampled per prompt (the GRPO group)."""

    max_offpolicy_steps: int = 3
    """Max train-steps a rollout may lag the trainer (the STALENESS cap). A group that
    ages past this at the batcher is stale-dropped, and the trainer never trains on
    older data. 0 = fully on-policy (sync): generator and trainer alternate in lockstep."""

    max_active_rollout_groups: int | None = None
    """Rollout buffer size: peak concurrent active rollout groups (the run-ahead
    depth). None couples it to the staleness cap: (max_offpolicy_steps + 1) *
    num_groups_per_train_step. Set LARGER to DECOUPLE run-ahead from staleness -- more
    rollouts generate concurrently (higher generator utilization, less trainer
    starvation) WITHOUT loosening max_offpolicy_steps: groups still stale-drop at the
    batcher once they age past it. This is a mean_age (queue size) vs max_age
    (drop) decoupling. Generator max_num_seqs scales with this (the max rollouts on the
    fly); it is a ceiling, not a KV reservation, so vLLM pages KV on demand and admits
    fewer / preempts when tight rather than OOM-ing."""

    generator_max_num_seqs: int | None = None
    """Explicit override for the generator's vLLM max_num_seqs (the scheduler's max
    concurrent decode batch per engine). None derives it from the rollout pool:
    ceil(resolved_max_active_rollout_groups * group_size / num_generator_shards),
    capped at 512. Set (e.g. 512) to remove the per-engine batch cap so vLLM batches
    as many concurrent rollouts as KV allows; harmless as a ceiling (vLLM admits
    fewer / preempts when KV is tight) but grows cudagraph capture sizes."""

    group_buffer: RolloutGroupWorkBuffer.Config = field(
        default_factory=RolloutGroupWorkBuffer.Config
    )
    training_sample_builder: TrainingSampleBuilder.Config = field(
        default_factory=TrainingSampleBuilder.Config
    )
    batcher: Batcher.Config = field(default_factory=Batcher.Config)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def resolved_max_active_rollout_groups(self) -> int:
        """Rollout buffer size in groups: the explicit override if set, else the
        staleness-coupled default ``(max_offpolicy_steps + 1) * num_groups_per_train_step``."""
        if self.max_active_rollout_groups is not None:
            return self.max_active_rollout_groups
        return (self.max_offpolicy_steps + 1) * self.num_groups_per_train_step


class Controller(Configurable):
    """Top-level RL async training orchestrator.

    Owns a `PolicyTrainer` actor (gradient updates), a `VLLMGenerator` actor
    (sampling), and a `Rollouter` (datasets + rubric + env construction).

    Check the docstring at the top of the file for more details.

    Example:

        config = config_registry.rl_grpo_qwen3_0_6b_varlen()
        controller = config.build()
        trainer_mesh = ...        # provisioned by the caller (see train.py)
        generator_meshes = ...
        await controller.setup_async(
            trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
        )
        await controller.run()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        async_loop: AsyncLoopConfig = field(default_factory=AsyncLoopConfig)
        """How the data->rollout->batch->train loop is sized and coordinated."""

        rollouter: Rollouter.Config
        """The rollouter: its datasets, envs, and rubric."""
        # TODO: support multiple rollouters for data mixing.

        renderer: RendererConfig
        """Message-to-token renderer config."""

        rollout_recorder: RolloutSampleRecorder.Config = field(
            default_factory=RolloutSampleRecorder.Config
        )
        """JSONL recorder to save sampled rollouts to disk for further inspection and debugging."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        # TODO: put generator, num generators and generator router in a separate config
        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        num_generators: int = 1
        """Number of generator replicas to spawn as separate proc meshes.

        This is distinct from intra-generator parallelism controlled by
        ``generator.parallelism``. Total generator GPU/process usage is
        ``num_generators * generator_world_size``.
        """

        num_rollout_workers: int = 0
        """CPU RolloutWorker processes to run group rollouts off the controller GIL.

        0 (default) runs rollouts in-process on the controller (the original
        path). N > 0 spawns N single-proc CPU actors on the controller host; the
        controller dispatches each group to a worker round-robin, so the agent
        orchestration (adapter, Daytona HTTP, grading) runs across N GILs. The
        global ``SWE_ROLLOUT_CONCURRENCY`` target is split evenly across workers.
        """

        generator_router: InterGeneratorRouter.Config = field(
            default_factory=InterGeneratorRouter.Config
        )
        """Generator routing strategy configuration."""

        # TODO: rename it to metrics_processor
        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        def __post_init__(self):
            if self.num_generators < 1:
                raise ValueError(
                    f"num_generators must be at least 1, got {self.num_generators}"
                )
            if self.generator.checkpoint.enable:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpoint.enable=False."
                )
            # RL policy inputs are shaped by BatchConfig, not TrainingConfig.
            if self.trainer.parallelism.enable_sequence_parallel:
                sp_degree = self.trainer.parallelism.tensor_parallel_degree
                seq_len = self.async_loop.batcher.batch.seq_len
                if sp_degree > 1 and seq_len % sp_degree != 0:
                    raise ValueError(
                        f"RL batcher sequence length ({seq_len}) must be divisible "
                        f"by sequence parallel degree ({sp_degree})."
                    )

            # Mirror the batcher width into trainer.training.seq_len for the model build.
            self.trainer.training.seq_len = self.async_loop.batcher.batch.seq_len

            # TODO: add a check so that all seq_len related variables make sense
            # e.g. rollout max length cannot be larger than the model max_seq_len
            # or the packing len, etc.

            if self.trainer.debug.batch_invariant:
                if torch.version.hip is not None:
                    raise ValueError(
                        "batch_invariant mode is not supported on ROCm: the varlen "
                        "attention path cannot force num_splits=1 (rejected by ROCm), "
                        "so split-k reductions are non-deterministic."
                    )
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # TODO: Replace trainer dtype constraint to use mixed
                #  training enabled by FSDP.
                if self.trainer.training.dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 training dtype, "
                        f"got {self.trainer.training.dtype!r}"
                    )
                if self.generator.model_dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 generator dtype, "
                        f"got {self.generator.model_dtype!r}"
                    )
                if self.trainer.parallelism.enable_sequence_parallel:
                    raise ValueError(
                        "batch_invariant mode doesn't support SP now. "
                        "SP uses reduce-scatter which only supports Ring in NCCL "
                        "and has not been validated for determinism."
                    )

            if (
                not self.generator_router.hot_swap
                and not self.generator.reset_prefix_cache_on_weight_sync
            ):
                raise ValueError(
                    "generator_router.hot_swap=False requires "
                    "generator.reset_prefix_cache_on_weight_sync=True, else requests admitted after a "
                    "pull reuse KV cached under the old weights."
                )

            # FULL cudagraph is only correct with the flex attention backend
            cudagraph = self.generator.cudagraph
            if (
                cudagraph.enable
                and cudagraph.mode == "FULL"
                and self.model_spec is not None
            ):
                from torchtitan.models.common.attention import FlexAttention

                inner_attn = self.model_spec.model.layers[0].attention.inner_attention
                if not isinstance(inner_attn, FlexAttention.Config):
                    raise ValueError(
                        "cudagraph mode 'FULL' is only supported with the flex "
                        "attention backend; the varlen backend corrupts FULL capture "
                        "of mixed prefill+decode batches (#3709). Use FULL_DECODE_ONLY "
                        "or FULL_AND_PIECEWISE."
                    )

    def __init__(self, config: Config):
        self.config = config
        self.trainer: PolicyTrainer | None = None
        self.generator_router: InterGeneratorRouter | None = None
        # CPU RolloutWorker actors (num_rollout_workers > 0); empty = in-process.
        self._rollout_workers: list = []
        # Resume step (0 = fresh); set in setup_async from the loaded checkpoint.
        self.start_step = 0
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.renderer = config.renderer.build(tokenizer_path=config.hf_assets_path)

        # Carry the base seed and renderer stop tokens on the sampling config so
        # the generator reads them off each request; the rollouter offsets the
        # seed per sample. Avoids the generator depending on request_id format.
        self._sampling = replace(
            config.generator.sampling,
            seed=config.generator.debug.seed,
            stop_token_ids=list(self.renderer.get_stop_token_ids()),
        )
        # TODO: pass our own tokenizer to the renderer and read pad/eos off it
        # once `renderers` supports bring-your-own-tokenizer
        # (https://github.com/PrimeIntellect-ai/renderers/pull/70).
        # Until then, reach into the renderer's tokenizer for the pad id (eos doubles as pad).
        self._rollouter: Rollouter = config.rollouter.build()
        self.rollout_recorder = config.rollout_recorder.build(
            dump_dir=config.dump_folder
        )

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")

        if self.trainer is not None:
            try:
                await self.trainer.close.call()
            except Exception:
                logger.exception("trainer.close failed")

        if self.generator_router is not None:
            close_results = await self.generator_router.fanout(
                "close", return_exceptions=True
            )
            for idx, result in enumerate(close_results):
                if isinstance(result, BaseException):
                    actor_name = (
                        "generator" if len(close_results) == 1 else f"generator[{idx}]"
                    )
                    logger.error(
                        "%s.close failed",
                        actor_name,
                        exc_info=(type(result), result, result.__traceback__),
                    )

        try:
            self.metrics_processor.close()
        except Exception:
            logger.exception("metrics_processor close failed")

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _get_rank_0_value(self, result):
        """Extract rank 0 result from a Monarch ValueMesh.

        Monarch actor endpoints return results from all ranks in the mesh.
        This method picks out rank 0's result. This should be used in cases
        where all ranks return the same result.
        """
        return result.get(0)

    def _make_generate_fn(self, metrics_prefix: str) -> GenerateFn:
        """Build the rollouter's `GenerateFn`: route a completion via the generator router, namespacing
        generation metrics with `metrics_prefix` and pinning sticky routing on `routing_session_id` (a sample's
        turns reuse one generator's prefix KV)."""
        # TODO: make this a pluggable config (a GenerateFn factory) so non-router generate backends can be swapped in.

        @sl.log_trace_span("generate")
        async def generate(
            prompt_token_ids: list[int],
            *,
            request_id: str,
            routing_session_id: str | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> Completion | None:
            result = await self.generator_router.route(
                "generate",
                prompt_token_ids,
                request_id=request_id,
                # VLLMGenerator.generate also requires this field for its
                # intra-mesh DP routing.
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
                metrics_prefix=metrics_prefix,
                # Load is measured as in-flight request count (one unit per call).
                routing_ctx=RoutingContext(
                    estimated_cost=1,
                    session_id=routing_session_id,
                ),
            )
            return self._get_rank_0_value(result)

        return generate

    @sl.log_trace_span("setup_async")
    async def setup_async(
        self,
        *,
        trainer_mesh: ProcMesh,
        generator_meshes: list[ProcMesh],
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Kept separate from ``__init__`` because actor spawning, torch
        elastic env setup, TorchStore initialization, and the initial
        weight push/pull are all ``await``-based runtime side effects
        that cannot run in a synchronous constructor.

        The trainer and generator meshes are provisioned by the caller
        (see ``create_proc_mesh``) on disjoint GPUs; this method only
        spawns the actors on them and synchronizes initial weights from
        trainer to generator. Must be called before :meth:`run`.

        Args:
            trainer_mesh: ProcMesh the trainer actor is spawned on.
            generator_meshes: ProcMesh objects the generator actors are spawned on.
        """
        # Peak concurrent rollout sequences (buffer groups * group_size, or the
        # validation pass); sizes max_num_seqs below. Uses the resolved buffer size
        # (the max rollouts on the fly): max_num_seqs is a CEILING, not a KV
        # reservation -- vLLM pages KV on demand and admits fewer / preempts when KV
        # is tight, so a larger ceiling matches the run-ahead pool without OOM.
        async_loop = self.config.async_loop
        rollout_concurrency = max(
            async_loop.resolved_max_active_rollout_groups() * async_loop.group_size,
            async_loop.validation.num_samples,
        )
        # Renderer thread pool: render work is CPU-bound, so size to CPU count (decoupled from rollout concurrency).
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=os.cpu_count())
        )

        config = self.config
        if not generator_meshes:
            raise ValueError("setup_async requires at least one generator mesh")

        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        generator_dp_degree = max(config.generator.parallelism.data_parallel_degree, 1)
        num_generator_dp_shards = len(generator_meshes) * generator_dp_degree

        # Ceiling (not target) for the generator's max_num_seqs: the per-generator
        # upper bound on concurrently scheduled sequences. vLLM may admit fewer if KV
        # is tight; this also sets CUDA-graph capture sizes. An explicit override
        # (generator_max_num_seqs) removes the derived per-engine cap.
        max_num_seqs = async_loop.generator_max_num_seqs or min(
            math.ceil(rollout_concurrency / num_generator_dp_shards), 512
        )

        logger.info(
            "max_num_seqs=%d per generator (rollout_concurrency=%d / generator_dp_shards=%d)",
            max_num_seqs,
            rollout_concurrency,
            num_generator_dp_shards,
        )

        # TODO(observability): the mesh_spawn span wraps ~80 LoC of branching
        # provisioner logic. Pull a PerHostProvisioner.spawn_meshes(...) helper and
        # shrink this span to a single call.
        with sl.log_trace_span("mesh_spawn"):
            # Store proc meshes for cleanup
            self._proc_meshes = [trainer_mesh, *generator_meshes]

            await setup_torch_elastic_env_async(trainer_mesh)
            for generator_mesh in generator_meshes:
                await setup_torch_elastic_env_async(generator_mesh)

            # Spawn actors on their respective meshes
            self.trainer = trainer_mesh.spawn(
                "trainer",
                PolicyTrainer,
                config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
                output_dir=config.dump_folder,
            )

            # TODO: torch.compile with aot_eager backend (inductor crashes the vLLM engine on the shared model path).
            generators = []
            for idx, generator_mesh in enumerate(generator_meshes):
                actor_name = (
                    "generator" if len(generator_meshes) == 1 else f"generator_{idx}"
                )
                generator = generator_mesh.spawn(
                    actor_name,
                    VLLMGenerator,
                    config.generator,
                    model_spec=config.model_spec,
                    model_path=config.hf_assets_path,
                    compile_config=config.compile,
                    max_num_seqs=max_num_seqs,
                    output_dir=config.dump_folder,
                )
                generators.append(generator)
            self.generator_router = config.generator_router.build(generators=generators)

        # Spawn the CPU RolloutWorker pool on the controller host: each worker runs
        # group rollouts (agent orchestration + adapter + Daytona HTTP + grading) in
        # its own process, off the controller GIL. One single-proc CPU mesh per
        # worker (like the generators, so each is individually addressable for
        # round-robin dispatch). The global SWE_ROLLOUT_CONCURRENCY target is split
        # evenly across the pool; each worker sets its own share before its rollouter
        # builds its lazy semaphore.
        num_workers = config.num_rollout_workers
        if num_workers > 0:
            global_conc = int(os.environ.get("SWE_ROLLOUT_CONCURRENCY", "16"))
            per_worker_conc = max(1, math.ceil(global_conc / num_workers))
            with sl.log_trace_span("rollout_worker_spawn"):
                host = this_host()
                for worker_id in range(num_workers):
                    worker_mesh = host.spawn_procs(per_host={"rollout_workers": 1})
                    self._proc_meshes.append(worker_mesh)
                    self._rollout_workers.append(
                        worker_mesh.spawn(
                            f"rollout_worker_{worker_id}",
                            RolloutWorker,
                            config,
                            rollout_concurrency=per_worker_conc,
                        )
                    )
                # Each worker builds its own generate-only router over the shared generators.
                await asyncio.gather(
                    *(w.setup.call(generators) for w in self._rollout_workers)
                )
            logger.info(
                f"Spawned {num_workers} RolloutWorker(s), {per_worker_conc} "
                f"concurrency each (global target {global_conc})"
            )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Resume: __init__ ran CheckpointManager.load(); read back the restored policy_version
        # (0 if fresh) so the loop resumes at the right step and generators pull at that version.
        # TODO(resume): only model/optimizer/policy_version are restored. The active-slot rollout
        #   buffer (in-flight rollouts) and the dataset stream position are NOT restored -- a resumed
        #   run refills the buffer and re-reads data from the start. Need to recycle prompts.
        self.start_step = self._get_rank_0_value(
            await self.trainer.get_policy_version.call()
        )
        if self.start_step > 0:
            logger.info(f"Resuming RL training from step {self.start_step}")

        # Initial weight sync: only the trainer loads weights; generators pull at start_step.
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self.generator_router.pull_model_state_dict(
                policy_version=self.start_step
            )

    # TODO: fold validation into a Validator(Configurable) the controller attaches, instead of 4 methods.
    @sl.log_trace_span("_collect_validation_rollouts")
    async def _collect_validation_rollouts(
        self, *, num_groups: int, sampling: SamplingConfig, step: int
    ) -> tuple[list[RolloutGroup], list[m.Metric]]:
        """Sample held-out prompts, run each greedily (n=1) concurrently, and emit validation metrics."""
        # TODO: group_size=1 (best-of-1) only. Support best-of-N.
        generate = self._make_generate_fn(metrics_prefix="validation_generator")
        # TODO(naming): reserve "sample" for TrainingSample; rename the rollouter's raw-prompt "sample" -> "prompt"/"data_input".
        samples = [self._rollouter.get_validation_sample() for _ in range(num_groups)]
        group_results = await asyncio.gather(
            *(
                self._rollouter.run_group_rollouts(
                    generate_fn=generate,
                    sample=sample,
                    # Negative ids keep validation disjoint from training group ids, so their
                    # request_ids can't collide in the shared engine (e.g. post-validation).
                    group_id=-(i + 1),
                    group_size=1,
                    sampling=sampling,
                    renderer=self.renderer,
                )
                for i, sample in enumerate(samples)
            ),
            return_exceptions=True,
        )

        # Keep the groups that succeeded; log + count the ones that raised.
        rollout_groups: list[RolloutGroup] = []
        num_failed_groups = 0
        for i, result in enumerate(group_results):
            if isinstance(result, BaseException):
                logger.error(
                    f"validation group {-(i + 1)} (step={step}) failed; dropping",
                    exc_info=(type(result), result, result.__traceback__),
                )
                num_failed_groups += 1
                continue
            rollout_groups.append(result)

        metrics = compute_rollout_metrics(
            prefix="validation",
            rollouts=[
                rollout for group in rollout_groups for rollout in group.rollouts
            ],
        )
        metrics.append(
            m.Metric("validation/group_failures", m.Sum(float(num_failed_groups)))
        )
        return rollout_groups, metrics

    # TODO: we currently determine validation.num_samples
    # but what if i want to run the entire dataset?
    @sl.log_trace_span("validate")
    async def validate(self, *, step: int) -> list[m.Metric]:
        """Run greedy validation on held-out prompts.

        Args:
            step: Training step this validation pass belongs to (0 for the
                pre-training pass); tagged into logged rollout samples.

        Returns:
            Validation rollout metrics, generation metrics, and validation
            timing.
        """
        # TODO: investigate using pass@k for validation.
        t_validate_start = time.perf_counter()
        num_samples = self.config.async_loop.validation.num_samples
        if num_samples == 0:  # skip validation (e.g. loss guard CI)
            return []
        greedy = replace(self._sampling, temperature=0.0, top_p=1.0)

        rollout_groups, validation_metrics = await self._collect_validation_rollouts(
            num_groups=num_samples, sampling=greedy, step=step
        )

        self.rollout_recorder.record(is_validation=True, rollout_groups=rollout_groups)

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

    async def run(self) -> None:
        """Start every async loop and run until training completes or a stage crashes.

        Producers (_data_input_loop, _rollout_loop[N], _batcher_loop) loop forever; _trainer_loop is the
        only finite loop -- it runs num_training_steps, then returns, which drives shutdown.

        Shutdown (healthy):  _trainer_loop finishes N steps -> run() finally ->
          group_buffer.close()  (wakes _data_input_loop / _rollout_loop / _batcher_loop blocked on the buffer)
          -> task.cancel()      (wakes anything blocked on training_batch_queue.put/get; close does NOT wake these)
          -> gather(..., return_exceptions=True)
        Shutdown (crash):    any loop raises -> appears in `done` -> run() re-raises -> same finally.
        """
        async_loop = self.config.async_loop
        num_training_steps = async_loop.num_training_steps
        logger.info(
            f"Running pre-training validation; then {num_training_steps} steps of async RL training"
        )

        sl.log_trace_instant("validation_start")
        pre_validation = await self._validate_and_log(step=self.start_step)
        sl.log_trace_instant("training_start")

        # Two policy version pointers, seeded from the resumed step: the trainer advances at the
        # optimizer step; the generator version advances when a weight pull completes.
        self._trainer_policy_version = self.start_step
        self._generator_policy_version = self.start_step

        # Buffer capacity = run-ahead depth (may exceed the staleness window when
        # max_active_rollout_groups is set; staleness is still bounded by the batcher's
        # stale-drop at max_offpolicy_steps).
        max_active_rollout_groups = async_loop.resolved_max_active_rollout_groups()

        self._group_buffer = async_loop.group_buffer.build(
            max_active_rollout_groups=max_active_rollout_groups,
        )

        # training_sample_builder
        training_sample_builder = async_loop.training_sample_builder.build()

        # batcher
        batcher = async_loop.batcher.build(
            num_groups_per_train_step=async_loop.num_groups_per_train_step,
            dp_degree=self.trainer_dp_degree,
            pad_id=self.renderer._tokenizer.eos_token_id,
        )

        # training_batch_queue
        training_batch_queue: asyncio.Queue[TrainingBatch | None] = asyncio.Queue(
            maxsize=1
        )

        # rollout_loop
        generate_fn = self._make_generate_fn(metrics_prefix="generator")

        # One rollout loop task per active buffer slot: lets generation fill the whole off-policy window,
        # including the cold start (step 0 fills every active slot, not just num_groups_per_train_step per wave).
        # With a RolloutWorker pool, each loop task is pinned to a worker round-robin so groups spread
        # across the worker processes (dispatch is non-blocking on the controller; the CPU work runs in the
        # worker). Without a pool (num_rollout_workers=0), worker=None keeps the in-process path.
        # TODO: support warm start
        num_workers = len(self._rollout_workers)
        rollout_tasks = [
            asyncio.create_task(
                self._rollout_loop(
                    group_buffer=self._group_buffer,
                    generate_fn=generate_fn,
                    worker=(
                        self._rollout_workers[group_worker_id % num_workers]
                        if num_workers
                        else None
                    ),
                ),
                name=f"rollout_worker_{group_worker_id}",
            )
            for group_worker_id in range(max_active_rollout_groups)
        ]

        # data_input_loop
        data_input_task = asyncio.create_task(
            self._data_input_loop(self._group_buffer), name="data_input"
        )

        # training_sample_batcher_loop
        batcher_task = asyncio.create_task(
            self._batcher_loop(
                group_buffer=self._group_buffer,
                training_sample_builder=training_sample_builder,
                batcher=batcher,
                training_batch_queue=training_batch_queue,
            ),
            name="batcher",
        )

        # trainer_loop
        trainer_task = asyncio.create_task(
            self._trainer_loop(
                training_batch_queue, num_training_steps=num_training_steps
            ),
            name="trainer",
        )

        # run everything until trainer finishes its number of steps
        # or some other loop breaks
        background_tasks = [
            data_input_task,
            *rollout_tasks,
            batcher_task,
        ]
        try:
            done, _ = await asyncio.wait(
                [trainer_task, *background_tasks], return_when=asyncio.FIRST_COMPLETED
            )
            # The trainer is the finite clock: it runs num_training_steps then returns -> training is done.
            # Producers loop forever, so a producer in `done` means it crashed (await re-raises) or wrongly
            # returned cleanly (the RuntimeError). Check producers even when the trainer also finished this
            # wakeup, so a simultaneous producer crash isn't hidden behind the finished trainer.
            for task in done:
                if task is trainer_task:
                    continue
                await task  # raises if task crashed; returns if task ended cleanly
                raise RuntimeError(f"{task.get_name()} exited unexpectedly")
            if trainer_task in done:
                await trainer_task
        finally:
            # Graceful first: buffer.close() (awaited) wakes loops blocked on the buffer so they return.
            # Then cancel covers anything blocked on the queue (which close does not wake); gather awaits all.
            await self._group_buffer.close()
            for task in (*background_tasks, trainer_task):
                task.cancel()
            await asyncio.gather(
                *background_tasks, trainer_task, return_exceptions=True
            )

        # Post-training validation (held-out eval after the final step).
        post_validation = await self._validate_and_log(step=num_training_steps)
        self._log_reward_delta(pre_validation, post_validation)

    async def _validate_and_log(self, *, step: int) -> dict[str, float]:
        """Run one validation pass, log it, and return its aggregated values for the pre/post delta."""
        metrics = await self.validate(step=step)
        self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)
        return m.MetricsProcessor._aggregate_metrics(metrics)

    def _log_reward_delta(self, pre: dict[str, float], post: dict[str, float]) -> None:
        """Console pre/post reward summary, visible without scrolling back through the loop."""
        reward_keys = sorted(key for key in set(pre) | set(post) if "reward" in key)
        logger.info("=" * 60)
        logger.info("Validation reward (pre / post):")
        for key in reward_keys:
            logger.info(
                f"  {key}:  {pre.get(key, float('nan')):+.3f}  /  {post.get(key, float('nan')):+.3f}"
            )
        logger.info("=" * 60)

    async def _data_input_loop(self, group_buffer: RolloutGroupWorkBuffer) -> None:
        """produces a RolloutGroupWork into group_buffer.
        waits for:    a free active slot (group_buffer.wait_for_slot)
        unblocked by: _trainer_loop release_active_groups(num_groups_per_train_step, "trained")
            after the pull (and _batcher_loop release_active_groups(1,"untrainable_group"))

        Separate from `_rollout_loop`, so slow data prep (e.g. on-the-fly question generation) overlaps
        generation instead of serializing in front of it.
        """
        # TODO(resume): persist dataset position so a restarted job continues the data stream, not from scratch.
        group_index = 0

        # TODO(perf): Slots are current released in batches, while this loop is a single producer.
        # we could a) increase the number of threads; b) revisit how we release slots and see if
        # we can release them on the batcher while still preserving max off-policy steps.
        # finally, c) we need to check how will this data input loop truly overlaps with the rollout loop.
        while await group_buffer.wait_for_slot():
            with sl.log_trace_span("get_training_sample"):
                # to_thread: Dont block on dataset reads
                sample = await asyncio.to_thread(self._rollouter.get_training_sample)
            await group_buffer.add_work(
                RolloutGroupWork(
                    group_id=group_index,
                    sample=sample,
                )
            )
            group_index += 1
        logger.info("Buffer closed; data input loop stopping")

    async def _rollout_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        generate_fn: GenerateFn,
        worker=None,
    ) -> None:
        """Generate + score one group at a time; a failed group becomes an empty group + a failure metric.

        When ``worker`` is a RolloutWorker actor, the group is run in that worker
        process (off the controller GIL) and returned; otherwise it runs in-process
        via ``self._rollouter``. Either way the controller records the raw group and
        finalizes it into the buffer.

        Staleness is bounded by the buffer's active-slot budget. Raw rollouts are recorded before any drop,
        so dropped groups stay inspectable on disk.

        consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
            waits for:    a claimable WAITING entry
            unblocked by: _data_input_loop group_buffer.add_work()
        produces: RolloutGroup (group_buffer.finalize_work)
            waits for:    nothing (admits its own claimed slot)
            unblocked by: n/a
        """
        while True:
            work = await group_buffer.claim_next()
            if work is None:  # group_buffer closed/shutdown signal
                logger.info("Buffer closed; rollout worker stopping")
                return
            try:
                with sl.log_trace_span("rollout_group"):
                    if worker is not None:
                        # Dispatch to a RolloutWorker process; it runs run_group_rollouts
                        # + compute_rollout_metrics and returns the finalized group.
                        group = self._get_rank_0_value(
                            await worker.run_group.call(
                                sample=work.sample, group_id=work.group_id
                            )
                        )
                    else:
                        group = await self._rollouter.run_group_rollouts(
                            generate_fn=generate_fn,
                            sample=work.sample,
                            group_id=work.group_id,
                            group_size=self.config.async_loop.group_size,
                            sampling=self._sampling,
                            renderer=self.renderer,
                        )
                        # Preserve rollouter-set group metrics (e.g. tmax
                        # nonsubmit_frac / format_errors); append the standard
                        # computed ones instead of overwriting them.
                        group.metrics = compute_rollout_metrics(
                            prefix="rollout", rollouts=group.rollouts
                        ) + list(group.metrics)

                # save rollout for inspection
                self.rollout_recorder.record(
                    is_validation=False,
                    rollout_groups=[group],
                )
            except Exception:
                logger.exception(f"rollout group {work.group_id} failed; dropping")
                group = RolloutGroup(
                    group_id=work.group_id,
                    rollouts=[],
                    metrics=[m.Metric("rollout/group_failures", m.Sum(1.0))],
                )
            await group_buffer.finalize_work(group)

    async def _batcher_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        training_sample_builder: TrainingSampleBuilder,
        batcher: Batcher,
        training_batch_queue: "asyncio.Queue[TrainingBatch | None]",
    ) -> None:
        """Take finalized groups, build training_samples, accumulate them, and queue each ready training batch.

        On a clean close/shutdown the group_buffer drains and returns None; we forward a `None` sentinel
        so the trainer stops.

        consumes: the oldest FINALIZED group (group_buffer.take_finalized)
            waits for:    the oldest group becoming FINALIZED
            unblocked by: _rollout_loop[N] group_buffer.finalize_work()
        produces: TrainingBatch (training_batch_queue.put)
            waits for:    a free training_batch_queue slot (maxsize=1)
            unblocked by: _trainer_loop training_batch_queue.get()
        """
        while True:
            rollout_group = await group_buffer.take_finalized()
            if rollout_group is None:  # closed and drained
                logger.info("Buffer drained; batcher loop stopping")
                break
            with sl.log_trace_span("training_sample_builder"):
                training_sample_group = training_sample_builder.build_from_group(
                    rollout_group=rollout_group
                )

            # [buffer trace] classify this completed group's solve status for per-step
            # visibility. full-solve (all rollouts pass) and not-solve (all fail) groups
            # are zero-std -> dropped by drop_zero_std; only partial-solve groups train.
            _rw = [r.reward for r in rollout_group.rollouts if r.reward is not None]
            _n = len(_rw)
            _ns = sum(1 for x in _rw if x and x > 0.0)
            _cls = (
                "full_solve"
                if _n and _ns == _n
                else "not_solve"
                if _ns == 0
                else "partial_solve"
            )

            # Stale-drop (take-any safety). take_finalized returns groups in completion
            # order, so a slow straggler generated many steps ago can surface later than
            # the off-policy window allows. Drop it here (release its slot, do not train)
            # instead of letting compute_policy_age_metrics trip its hard freshness
            # assertion. This is open-instruct's max_result_age_steps. Strict FIFO never
            # hit this: its head-of-line stall pinned the trainer version until the
            # oldest group was consumed -- the very stall we traded away for throughput.
            #
            # Age is taken from the RAW rollout group's turns, NOT training_samples, so a
            # stale ZERO-STD group is dropped too. A zero-std group has empty
            # training_samples, so gating on those would skip the stale check and leak its
            # metrics (e.g. rollout_reward/avg_train_reward, attached before the drop) into
            # the step -- open-instruct drops all stale results before counting them. For a
            # trainable group this min matches min(ts.min_policy_version): a sample's
            # min_policy_version is its opening turn's version and per-rollout versions are
            # non-decreasing, so the group min is turn 0's version either way.
            group_min_policy_version = min(
                (
                    turn.min_policy_version
                    for rollout in rollout_group.rollouts
                    for turn in rollout.turns
                    if turn.min_policy_version is not None
                ),
                default=None,
            )
            if group_min_policy_version is not None:
                # NOTE: checked at the CURRENT trainer version, but the batch is
                # consumed later; with the 1-deep training_batch_queue the trainer can
                # advance one step in between, so use >= (leave a 1-step margin) to keep
                # the consume-time age <= max_offpolicy_steps and never trip the hard
                # freshness assertion in compute_policy_age_metrics.
                group_age = self._trainer_policy_version - group_min_policy_version
                if group_age >= self.config.async_loop.max_offpolicy_steps:
                    logger.info(
                        "[buffer] complete group_id=%d solved=%d/%d class=%s "
                        "-> RELEASE(stale_dropped) cur_ver=%d",
                        rollout_group.group_id,
                        _ns,
                        _n,
                        _cls,
                        self._trainer_policy_version,
                    )
                    await group_buffer.release_active_groups(1, reason="stale_dropped")
                    sl.log_trace_scalar({"rollout_buffer/dropped/stale": 1.0})
                    continue

            if not training_sample_group.training_samples:
                logger.info(
                    "[buffer] complete group_id=%d solved=%d/%d class=%s "
                    "-> RELEASE(zero_std, dropped) cur_ver=%d",
                    rollout_group.group_id,
                    _ns,
                    _n,
                    _cls,
                    self._trainer_policy_version,
                )
                await group_buffer.release_active_groups(1, reason="untrainable_group")
            else:
                logger.info(
                    "[buffer] complete group_id=%d solved=%d/%d class=%s "
                    "-> TRAINABLE cur_ver=%d",
                    rollout_group.group_id,
                    _ns,
                    _n,
                    _cls,
                    self._trainer_policy_version,
                )

            # We put a group in. We may get a batch back
            # if there are enough accumulated trainable groups to return one.
            with sl.log_trace_span("batcher_pack"):
                maybe_training_batch = await asyncio.to_thread(
                    batcher.add_training_samples,
                    training_sample_group=training_sample_group,
                )
            if maybe_training_batch is not None:
                logger.info(
                    "[batcher_loop] packed a training batch "
                    f"({len(maybe_training_batch.microbatches)} microbatch(es)); "
                    "putting on queue"
                )
                await training_batch_queue.put(maybe_training_batch)
                logger.info("[batcher_loop] batch enqueued")
        await training_batch_queue.put(None)
        # TODO(async-rl): if finite datasets are supported, drain a final partial batch here.

    async def _trainer_loop(
        self,
        training_batch_queue: "asyncio.Queue[TrainingBatch | None]",
        *,
        num_training_steps: int,
    ) -> None:
        """Run num_training_steps optimizer steps: train one packed batch, publish trainer weights,
        then pull them into generators, log metrics.

        consumes: a TrainingBatch (training_batch_queue.get)
            waits for:    a TrainingBatch in the queue
            unblocked by: _batcher_loop training_batch_queue.put()
        """
        for step in range(self.start_step + 1, num_training_steps + 1):
            sl.set_step(step)  # propagate the step counter to the actors
            # Phase logging: the training step is otherwise a black box until its
            # end-of-step metrics flush, so a hang inside it is invisible. Log each
            # actor-call boundary so a stall is localized in the stdout timeline.
            logger.info(f"[trainer_loop] step {step}: begin; trainer.sync_log_step")
            await self.trainer.sync_log_step.call(step)
            logger.info(f"[trainer_loop] step {step}: generator fanout sync_log_step")
            await self.generator_router.fanout("sync_log_step", step)
            logger.info(f"[trainer_loop] step {step}: awaiting training batch")
            step_timer = MetricsTimer()

            with sl.log_trace_span("train_step"), step_timer.record(
                "timing/step/total"
            ):
                # Waits for a TrainingBatch to be ready (or None on shutdown).
                with sl.log_trace_span("wait_for_training_batch"), step_timer.record(
                    "timing/step/wait_for_training_batch"
                ):
                    packed = await training_batch_queue.get()

                if packed is None:
                    logger.info("Batcher closed and drained; stopping training")
                    break
                logger.info(
                    f"[trainer_loop] step {step}: got batch, "
                    f"{len(packed.microbatches)} microbatch(es), "
                    f"{packed.num_global_valid_tokens} valid tokens"
                )

                # Policy age is computed HERE, at consumption time, against the live trainer version, so it is
                # faithful to what this step trains on -- not the version when the batch was packed.
                policy_age_panel = compute_policy_age_metrics(
                    trainer_policy_version=self._trainer_policy_version,
                    min_policy_versions=packed.min_policy_versions,
                    max_offpolicy_steps=self.config.async_loop.max_offpolicy_steps,
                )

                # TODO(async): can't stream microbatches (interleave pack->train) — the loss is normalized by
                #   packed.num_global_valid_tokens (sum over ALL microbatches), needed before any fwd/bwd. To
                #   support streaming, accumulate raw loss/token counts across microbatches and scale before optim.
                with sl.log_trace_span("forward_backward"), step_timer.record(
                    "timing/step/forward_backward"
                ):
                    # fwd_bwd on all microbatches (logged per microbatch so a stall
                    # inside the first cross-host collective is visible in the timeline).
                    microbatch_metrics = []
                    num_microbatches = len(packed.microbatches)
                    for mb_idx, microbatch in enumerate(packed.microbatches):
                        logger.info(
                            f"[trainer_loop] step {step}: forward_backward "
                            f"microbatch {mb_idx + 1}/{num_microbatches}"
                        )
                        microbatch_metrics.append(
                            self._get_rank_0_value(
                                await self.trainer.forward_backward.call(
                                    microbatch, packed.num_global_valid_tokens
                                )
                            )
                        )

                    fwd_bwd_metrics = combine_microbatch_metrics(microbatch_metrics)
                    logger.info(
                        f"[trainer_loop] step {step}: forward_backward done, "
                        f"loss={fwd_bwd_metrics['loss/mean']:.4f}"
                    )

                    if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                        logger.error("Loss is NaN/Inf; training diverged")
                        break

                with sl.log_trace_span("optim_step"), step_timer.record(
                    "timing/step/optim"
                ):
                    optim_result = self._get_rank_0_value(
                        await self.trainer.optim_step.call()
                    )
                self._trainer_policy_version = optim_result.policy_version
                logger.info(f"[trainer_loop] step {step}: optim done")

                # Weight sync: publish new weights before the next train step (push then pull, both awaited).
                # TODO(perf): overlap weight sync (today both are awaited synchronously).
                with sl.log_trace_span(
                    "trainer_push_model_state_dict"
                ), step_timer.record("timing/step/push_model_state_dict"):
                    await self.trainer.push_model_state_dict.call()
                logger.info(f"[trainer_loop] step {step}: weights pushed")
                with sl.log_trace_span(
                    "generator_pull_model_state_dict"
                ), step_timer.record("timing/step/pull_model_state_dict"):
                    await self.generator_router.pull_model_state_dict(
                        policy_version=optim_result.policy_version
                    )
                self._generator_policy_version = optim_result.policy_version
                logger.info(f"[trainer_loop] step {step}: weights pulled (step done)")

                # Release one train step's group slots after the pull; the batcher packs exactly
                # num_groups_per_train_step trainable groups per batch.
                logger.info(
                    "[buffer] step %d: RELEASE(trained) %d trainable group slots "
                    "(this step trained on %d partial-solve groups)",
                    step,
                    self.config.async_loop.num_groups_per_train_step,
                    self.config.async_loop.num_groups_per_train_step,
                )
                await self._group_buffer.release_active_groups(
                    self.config.async_loop.num_groups_per_train_step,
                    reason="trained",
                )

            # TODO(metrics): See if metrics are being computed at the right place. E.g. should we put all
            # rollout related metrics here, or move all of them to the rollouter.
            time_metrics = step_timer.flush()
            self.metrics_processor.log(
                step=step,
                is_validation=False,
                metrics=[
                    *packed.metrics,
                    *[
                        m.Metric(key, m.NoReduce(value))
                        for key, value in fwd_bwd_metrics.items()
                    ],
                    *[
                        m.Metric(key, m.NoReduce(value))
                        for key, value in optim_result.metrics.items()
                    ],
                    *self._group_buffer.metrics(),
                    *time_metrics,
                    *policy_age_panel,
                    *compute_perf_ratio_metrics(
                        num_global_valid_tokens=packed.num_global_valid_tokens,
                        time_metrics=time_metrics,
                    ),
                ],
            )

            # Save full training state for resume; CheckpointManager writes only on its interval
            # and the final step. After the divergence guard so a NaN step isn't checkpointed.
            with sl.log_trace_span("trainer_save_checkpoint"):
                await self.trainer.save_checkpoint.call(
                    step, last_step=(step == num_training_steps)
                )

            # Periodic held-out validation. Skip the final step -- run() does a post-training
            # pass right after the loop. Runs on this step's just-pulled weights, overlapping
            # the background rollout collection (disjoint negative group ids), so the curve
            # tracks the live policy without a separate eval job or checkpoint download.
            validation = self.config.async_loop.validation
            if (
                validation.interval > 0
                and step % validation.interval == 0
                and step != num_training_steps
            ):
                await self._validate_and_log(step=step)

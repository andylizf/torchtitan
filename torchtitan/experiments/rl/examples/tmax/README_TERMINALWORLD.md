# TerminalWorld + SWE-Smith mix run (Qwen3.5-9B, Terminus-2)

End-to-end recipe for RL-training Qwen3.5-9B as a terminal agent on a mix of two
oracle-validated Harbor corpora, scored on Terminal-Bench 2.0. This is the concrete
parameter set behind the `rl_grpo_qwen3_5_9b_tmax` recipe; for the general recipe
internals see [`README.md`](README.md) and for the seed-data pipeline see
[`README_SEED_DATA.md`](README_SEED_DATA.md).

Everything here runs on the open-source entry point
(`python -m torchtitan.experiments.rl.train`) with a Daytona API key; no other
service is required. Two 8-GPU hosts is the minimum (one trainer + one generator
host); more generator hosts raise rollout throughput.

- Agent: **Terminus-2** (`TMAX_AGENT=terminus`), the tmux-driving scaffold from the
  `harbor` package, NOT the default one-command `vanillux` loop.
- Generator: **`torchtitan_wrapper`** (the unified TorchTitan GDN model run inside
  vLLM), so the generator and trainer run the same model code + weights.
- Data: **TerminalWorld-Seeds-Clean** (general terminal tasks) mixed with
  **SWE-Smith-Seeds-Clean** (Python repo bug-fix), both graded by the identical
  contract (`bash /tests/test.sh` -> `/logs/verifier/reward.txt` 0/1).
- Eval: **Terminal-Bench 2.0** (89 tasks), k=5, decoupled from training (below).

## 0. Prerequisites

```bash
pip install -r requirements.txt          # torchtitan + the RL experiment deps
pip install harbor                        # provides Terminus-2 and TB-2.0 tasks
export DAYTONA_API_KEY=dtn_...            # sandbox provider
```

## 1. Prepare the data

Both corpora are public Hugging Face datasets with a Harbor task-tree layout; one
adapter (`prepare_rts_data.py`) reads both. The full pipeline -- download, extract,
adapt, and filter by the datasets' own quality columns -- is documented in
[`README_SEED_DATA.md`](README_SEED_DATA.md). The short version:

1. Download + extract each dataset's `data/*.tar` shards and `metadata/tasks.parquet`
   (`huggingface_hub`), giving a local `tasks/` dir per corpus.
2. Adapt each to a JSONL. `--inject-agent-runtime` bakes `tmux` into every image at
   build time -- REQUIRED for Terminus-2, whose runtime self-install fails on the
   SWE-Smith base images (without it the whole SWE-Smith half scores reward 0):

   ```bash
   python -m torchtitan.experiments.rl.examples.tmax.prepare_rts_data \
       --tasks-root /path/to/terminalworld/tasks --inject-agent-runtime \
       --out terminalworld_all.jsonl
   python -m torchtitan.experiments.rl.examples.tmax.prepare_rts_data \
       --tasks-root /path/to/swesmith/tasks --inject-agent-runtime \
       --out swesmith_all.jsonl
   ```

3. Filter each JSONL by joining `metadata/tasks.parquet` on `task_id` (see
   `README_SEED_DATA.md` for the exact pandas snippet):
   - TerminalWorld: keep `reward_verdict == "pass"` (~859 tasks) -- the fail/unknown
     36% have a reference solution that cannot even earn reward 1.
   - SWE-Smith: keep `in_main_pool and not network_required` (~1,408 tasks) -- the
     authors' stratified, repo-balanced pool; the grader runs `--network none`.

4. Concatenate into one training JSONL (labels are namespaced, no collision), and
   build the TB-2.0 eval set:

   ```bash
   cat terminalworld_pass.jsonl swesmith_main.jsonl > mix_tw_swe.jsonl
   python -m torchtitan.experiments.rl.examples.tmax.prepare_tb2_data --out tb2_eval.jsonl
   ```

Validate the sandbox + grading path before touching GPUs (no training stack needed):

```bash
DAYTONA_API_KEY=dtn_... python torchtitan/experiments/rl/examples/tmax/local_smoke.py \
    --data mix_tw_swe.jsonl --limit 2
```

## 2. Launch training

The full parameter set used for this run. The trainer takes one 8-GPU host; each
generator is a separate single-GPU vLLM engine (TP-1), data-parallel, so more
generators means more concurrent decode for the hundreds of live agents.

```bash
export DAYTONA_API_KEY=dtn_...
export SWE_PROMPT_DATA=/path/to/mix_tw_swe.jsonl

# --- harness: Terminus-2 with the recipe's context / turn budget ---
export TMAX_AGENT=terminus
export TMAX_TERMINUS_MAX_TURNS=120
export SWE_MAX_CONTEXT_LEN=63488
export TMAX_TURN_MAX_TOKENS=32768

# --- generator: unified TorchTitan GDN inside vLLM (narrows gen/trainer logprob gap) ---
export SWE_GEN_BACKEND=torchtitan_wrapper

# --- GRPO/async shape: 32 groups x 16 siblings = 512 rollouts/step, no zero-std drop ---
export SWE_NUM_GROUPS_PER_TRAIN_STEP=32
export SWE_GROUP_SIZE=16
export SWE_DROP_ZERO_STD=0            # keep all-pass/all-fail groups (they add no gradient)
export SWE_MAX_ACTIVE_GROUPS=512      # buffer capacity >= (off+1) * groups
export SWE_TRAIN_STEPS=150

# --- selection + throughput ---
export SWE_SELECTION_WINDOW_GROUPS=64 # take from the oldest 64 finalized groups. 64 = 2x
                                      # the 32-group batch: enough to fill a step without
                                      # head-of-line blocking, bounded (more on-policy +
                                      # less stale-drop waste) than None=take-any.
export SWE_ROLLOUT_CONCURRENCY=768    # active sandboxes. Higher congests the controller<->
                                      # generator Monarch channels -> false "proc dead"
                                      # crashes (seen at 1280); 768 still fills a 512-rollout
                                      # step. Raise only after removing build-failing tasks.
export SWE_NUM_ROLLOUT_WORKERS=16     # CPU agent-orchestration processes, off the controller GIL
export SWE_CKPT_INTERVAL=5            # checkpoint every 5 steps (cheap resume on failure)

# --- rollout wall-clock + sandbox tuning (recipe defaults; listed for reproducibility) ---
export SWE_GDN=1                      # GDN (Gated DeltaNet) hybrid model path
export SWE_DISABLE_CUSTOM_ALL_REDUCE=1  # required for the GDN generator under vLLM
export SWE_TIME_BUDGET_SEC=2400       # per-rollout wall-clock budget (40 min)
export TMAX_EXEC_TIMEOUT_SEC=120      # per-command timeout inside the sandbox
export SWE_MAX_NUM_SEQS=32            # vLLM engine max concurrent sequences
export TT_DAYTONA_CPU=2               # per-sandbox resources (Daytona)
export TT_DAYTONA_MEM_GB=4
export TT_DAYTONA_DISK_GB=10
export TT_DAYTONA_HEARTBEAT_SEC=180   # sandbox keep-alive
export TT_DAYTONA_CREATE_CONCURRENCY=8  # per-worker create parallelism; lower if the provider 429s

# --- INLINE Terminal-Bench 2.0 eval (see section 3): scored on a dedicated async eval
#     generator every SWE_VAL_INTERVAL steps, logged to the training W&B run. The step-0
#     pre-training pass is the baseline. ---
export SWE_TB2_VAL_DATA=/path/to/tb2_eval.jsonl  # the 89-task TB-2.0 JSONL (prepare_tb2_data)
export SWE_NUM_EVAL_GENERATORS=1      # one dedicated eval-generator host (async, off the training pool)
export SWE_VAL_INTERVAL=20            # eval every 20 optimizer steps
export SWE_VAL_SAMPLES=89             # all 89 TB-2.0 tasks per pass (k=5)

python -m torchtitan.experiments.rl.train \
    --module torchtitan.experiments.rl.examples.tmax \
    --config rl_grpo_qwen3_5_9b_tmax \
    --num-generators 12 \
    --num-eval-generators 1 \
    --hf_assets_path /path/to/Qwen3.5-9B
```

> **Data hygiene matters for stability.** A task whose Docker image fails to build
> (e.g. `apt-get` on an EOL Debian base) makes every rollout on it retry sandbox
> creation ~6x; a single such task sampled repeatedly floods the rollout workers with
> Daytona create storms that saturate the controller host and trigger the false
> "proc dead" crash above. Filter build-failing tasks out of the training JSONL (watch
> the logs for repeated `BUILD_FAILED` on one `instance_id`).

### Host topology

For the parameter set above on 8-GPU hosts, the run spans **~15 hosts**: 1 controller
+ 2 trainer hosts (the HSDP-16 trainer below) + 12 generator hosts (one per
`--num-generators`, each a TP-1 vLLM engine). Scale generator hosts to trade cost for
rollout throughput; the trainer host count follows the parallelism degrees. Launching
across hosts is your own job scheduler's problem.

### Multi-host trainer (optional)

On a fresh multi-host trainer, a naive `dp_shard=N` can hang on the first cross-host
all-gather. Use HSDP instead -- `dp_replicate x dp_shard=8` keeps each all-gather
within a host -- via `SWE_DP_REPLICATE`: `=2` gives HSDP-16 (2 hosts, ~2x faster
fwd_bwd vs FSDP-8), `=4` gives HSDP-32 (4 hosts). Note the run is usually
generation-bound (steps paced by rollout production), so more trainer DP helps only
while `batch-wait` is low; if `fwd_bwd` drops but `batch-wait` rises, add generators
instead.

## 3. Inline TB-2.0 eval (built into the run)

The launch above already turns eval on via `SWE_TB2_VAL_DATA` + `--num-eval-generators 1`:
a validation pass scores all 89 TB-2.0 tasks (k=5, temperature 0.7, top_p 0.95) on a
**dedicated eval-generator host, asynchronously**, every `SWE_VAL_INTERVAL` steps. A pass
still in flight when the next interval arrives is **skipped, not queued**. Results land
in the **training W&B run**:

- `validation/reward/mean` = **avg@5**
- `validation/pass_at_k` = **pass@5**
- `validation/policy_version` = the policy version scored (the async pass lags training).

The **step-0 pre-training pass is the base-model baseline** (compare later versions to it).
Because the eval runs on its own generator host, training does not block on it.

Each pass also writes a browsable report at
`<dump>/validation_traces/step-<policy_version>/{INDEX.md,summary.json,traces/}`.

### Optional: score a single checkpoint offline

To score one saved checkpoint without a training run (e.g. re-checking a step), use the
eval-only recipe -- same harness, `num_training_steps=0`, one validation pass:

```bash
SWE_TB2_DATA=/path/to/tb2_eval.jsonl SWE_TB2_CKPT=/path/to/run/checkpoint/step-N \
TMAX_AGENT=terminus SWE_MAX_CONTEXT_LEN=63488 SWE_GEN_BACKEND=torchtitan_wrapper \
python -m torchtitan.experiments.rl.train \
    --module torchtitan.experiments.rl.examples.tmax \
    --config rl_grpo_qwen3_5_9b_tmax_tb2_eval \
    --num-generators 1 --hf_assets_path /path/to/Qwen3.5-9B
# Leave SWE_TB2_CKPT unset to score the BASE model (the step-0 baseline).
```

## 4. Parameters at a glance

| Variable | This run | Meaning |
| --- | --- | --- |
| `SWE_PROMPT_DATA` | `mix_tw_swe.jsonl` | TerminalWorld pass + SWE-Smith main pool |
| `TMAX_AGENT` | `terminus` | Terminus-2 scaffold (default `vanillux`) |
| `TMAX_TERMINUS_MAX_TURNS` | 120 | Max agent turns per episode |
| `SWE_MAX_CONTEXT_LEN` | 63488 | Agent context budget |
| `TMAX_TURN_MAX_TOKENS` | 32768 | Per-turn generation cap |
| `SWE_GEN_BACKEND` | `torchtitan_wrapper` | Unified GDN model in vLLM |
| `SWE_NUM_GROUPS_PER_TRAIN_STEP` | 32 | GRPO prompt groups per step |
| `SWE_GROUP_SIZE` | 16 | Siblings per group |
| `SWE_DROP_ZERO_STD` | 0 | Keep zero-variance groups (no oversampling) |
| `SWE_MAX_ACTIVE_GROUPS` | 512 | Off-policy buffer capacity (groups) |
| `SWE_SELECTION_WINDOW_GROUPS` | 64 | Sliding-prefix selection window (>= batch; None=take-any) |
| `SWE_TRAIN_STEPS` | 150 | Optimizer steps |
| `SWE_ROLLOUT_CONCURRENCY` | 768 | Active sandboxes (higher risks the controller crash) |
| `SWE_NUM_ROLLOUT_WORKERS` | 16 | Agent-orchestration CPU processes |
| `SWE_NUM_GENERATORS` (`--num-generators`) | 12 | vLLM engines (each 1 GPU) |
| `SWE_CKPT_INTERVAL` | 5 | Checkpoint cadence (steps) |
| `SWE_DP_REPLICATE` | 2 / 4 (multi-host) | HSDP replicate degree (HSDP-16 / HSDP-32) |
| `SWE_TB2_VAL_DATA` | `tb2_eval.jsonl` | Inline TB-2.0 eval set (enables eval) |
| `SWE_NUM_EVAL_GENERATORS` (`--num-eval-generators`) | 1 | Dedicated async eval-generator host |
| `SWE_VAL_INTERVAL` / `SWE_VAL_SAMPLES` | 20 / 89 | Eval every 20 steps, all 89 tasks |

## 5. Reading the metrics

- **`rollout_reward/_mean`** with `drop_zero_std=0` is the real batch-mean reward
  (not pinned), so it is a usable training-progress signal (though noisy: the batch
  composition varies pass to pass).
- **`validation/reward/mean`** (avg@5) and **`validation/pass_at_k`** (pass@5),
  keyed by `validation/policy_version`, are the inline TB-2.0 curve. Compare later
  versions to the step-0 base-model baseline. Do not read a trend from fewer than
  ~7 points -- a single TB-2.0 pass has real run-to-run noise.
- The SWE-Smith half's improvement shows only in the training reward; the TB-2.0
  eval set is terminal-only.

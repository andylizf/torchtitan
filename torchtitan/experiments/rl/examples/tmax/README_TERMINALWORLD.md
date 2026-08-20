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

# --- throughput ---
export SWE_ROLLOUT_CONCURRENCY=1280   # active sandboxes; ~= (off+1)*groups*group_size ceiling
export SWE_NUM_ROLLOUT_WORKERS=16     # CPU agent-orchestration processes, off the controller GIL
export SWE_CKPT_INTERVAL=5            # checkpoint every 5 steps (cheap resume on failure)

# --- inline eval OFF: eval is decoupled to a separate job (step 3) so it never
#     contends with training for the controller/sandbox provider ---
export SWE_VAL_SAMPLES=0
export SWE_NUM_EVAL_GENERATORS=0

python -m torchtitan.experiments.rl.train \
    --module torchtitan.experiments.rl.examples.tmax \
    --config rl_grpo_qwen3_5_9b_tmax \
    --num-generators 12 \
    --hf_assets_path /path/to/Qwen3.5-9B
```

### Multi-host trainer (optional)

On a fresh 2-host trainer, a naive `dp_shard=16` can hang on the first cross-host
all-gather. Use HSDP instead -- `dp_replicate=2 x dp_shard=8` keeps each all-gather
within a host -- by setting `export SWE_DP_REPLICATE=2` (measured ~2x faster
fwd_bwd vs FSDP-8, no hang).

## 3. Eval every N steps (decoupled) + a step-0 baseline

Run the TB-2.0 eval as a SEPARATE process per checkpoint, NOT inline. Inline eval
(hundreds of concurrent eval rollouts, some on heavy images) piles onto the
training controller and can stall it; decoupling removes that entirely and lets you
score every checkpoint on its own hardware.

The eval-only config loads a checkpoint as the initial model and scores all 89
TB-2.0 tasks once (k=5, temperature 0.7, top_p 0.95 -- so `validation/reward/mean`
is avg@5 and `validation/pass_at_k` is pass@5). Point it at each checkpoint your
training run drops (e.g. every 15 steps) with your own scheduler:

```bash
# The SAME harness env as training (Terminus-2, context, turns, backend) -- otherwise
# the eval silently runs a different agent (TMAX_AGENT defaults to vanillux).
export TMAX_AGENT=terminus SWE_MAX_CONTEXT_LEN=63488 TMAX_TERMINUS_MAX_TURNS=120
export TMAX_TURN_MAX_TOKENS=32768 SWE_GEN_BACKEND=torchtitan_wrapper
export SWE_TB2_DATA=/path/to/tb2_eval.jsonl
export SWE_ROLLOUT_CONCURRENCY=89     # all tasks at once

# Score a trained checkpoint (policy_version N):
SWE_TB2_CKPT=/path/to/run/checkpoint/step-N \
python -m torchtitan.experiments.rl.train \
    --module torchtitan.experiments.rl.examples.tmax \
    --config rl_grpo_qwen3_5_9b_tmax_tb2_eval \
    --num-generators 1 --hf_assets_path /path/to/Qwen3.5-9B

# Score the BASE model (the step-0 baseline, policy_version 0): leave SWE_TB2_CKPT
# unset/empty -> the recipe loads --hf_assets_path instead of a checkpoint.
SWE_TB2_CKPT= \
python -m torchtitan.experiments.rl.train \
    --module torchtitan.experiments.rl.examples.tmax \
    --config rl_grpo_qwen3_5_9b_tmax_tb2_eval \
    --num-generators 1 --hf_assets_path /path/to/Qwen3.5-9B
```

Each pass writes `<dump>/validation_traces/step-<version>/summary.json` with
`avg_at_k`, `pass_at_k`, and `policy_version`. To build a single eval curve, read
those and log them to your tracker keyed by `policy_version` (0, 15, 30, ...).

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
| `SWE_MAX_ACTIVE_GROUPS` | 512 | Off-policy buffer capacity |
| `SWE_TRAIN_STEPS` | 150 | Optimizer steps |
| `SWE_ROLLOUT_CONCURRENCY` | 1280 | Active sandboxes |
| `SWE_NUM_ROLLOUT_WORKERS` | 16 | Agent-orchestration CPU processes |
| `SWE_NUM_GENERATORS` (`--num-generators`) | 12 | vLLM engines (each 1 GPU) |
| `SWE_CKPT_INTERVAL` | 5 | Checkpoint cadence (steps) |
| `SWE_DP_REPLICATE` | 2 (multi-host) | HSDP replicate degree |
| `SWE_VAL_SAMPLES` / `SWE_NUM_EVAL_GENERATORS` | 0 / 0 | Inline eval OFF (decoupled instead) |

## 5. Reading the metrics

- **`rollout_reward/_mean`** with `drop_zero_std=0` is the real batch-mean reward
  (not pinned), so it is a usable training-progress signal.
- **`eval/avg_at_5`** and **`eval/pass_at_5`** (from step 3, keyed by
  `policy_version`) are the TB-2.0 curve. Compare against the step-0 baseline you
  scored on the base model. Do not read a trend from fewer than ~7 points -- a
  single TB-2.0 pass has real run-to-run noise.
- The SWE-Smith half's improvement shows only in the training reward; the TB-2.0
  eval set is terminal-only.

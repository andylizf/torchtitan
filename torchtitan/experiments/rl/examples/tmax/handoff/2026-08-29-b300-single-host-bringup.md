# Handoff: tmax 9B bring-up on a single 8x B300 host

**Status: the stack boots and reaches the training loop.** Trainer and all six
vLLM engines initialize, weights transfer, and the loop reaches
`[trainer_loop] step 1: begin` with a KV-cache figure matching the reference run.
No optimizer step has completed here yet -- rollouts were not left running long
enough. What follows is the record of what it took, because every failure below
is a trap the next person will otherwise re-discover.

Reference for comparison: the della-tridao run documented in
`runbook/RUNBOOK.md` (this repo) and the ops runbook in the private
`andylizf/terminalworld-seeds` repo.

---

## 1. Configuration under test

| | this host | reference host |
|---|---|---|
| GPUs | 8x B300 SXM6 (SM 10.3), driver 610.57.04 | 8x B300, 5 used |
| split | `SWE_DP_SHARD=2` + `SWE_GEN_DP=6` | `SWE_DP_SHARD=2` + `SWE_GEN_DP=3` |
| torch / vllm | 2.15.0.dev20260827+cu130 / same-day nightly | 2.14.0.dev20260806+cu130 pair |
| generator backend | `vllm_native` (forced, see section 3) | `vllm_native` |
| agent | terminus (harbor 0.22.0) | terminus |
| data | 1909-row mix, 64-row holdout tail | 759-row mix |

Boot numbers, which is the cheapest way to tell the environment is equivalent:

| | here | reference |
|---|---|---|
| `Available KV cache memory` | 205.04 GiB | 204.61 GiB |
| `Maximum concurrency for 65,536 tokens` | 97.14x | 96.93x |

---

## 2. Config knobs this branch does not have

The reference `rltrain.env` is not portable to this branch as-is. Copied
verbatim, these three are read by nobody and silently do nothing:

- `SWE_WRONG_SUBMIT_PENALTY`
- `SWE_CKPT_KEEP` -- the checkpoint count is a config field here, so pass
  `--trainer.checkpoint.keep_latest_k N` on the command line instead. **This one
  matters**: the config default is 10 and a 9B checkpoint is ~98 GiB, so the
  default quietly reserves ~1 TiB.
- `TT_DAYTONA_MAX_MEM_GB` -- the 8 GiB clamp does not exist here. The five
  TerminalWorld tasks that declare 16 GiB are already excluded by
  `train_ready_ids.txt`, so this is currently harmless.

`SWE_GDN` is inert everywhere, upstream included.

Two knobs were ported from upstream because the recipe cannot run on one host
without them: `SWE_GEN_DP` (the base bakes DP-8 x TP-1 = 8 generator GPUs, which
leaves nothing for the trainer) and `SWE_GPU_MEM_LIMIT`.

---

## 3. `torchtitan_wrapper` cannot run Qwen3.5-9B on Blackwell

The reference runbook says only that `torchtitan_wrapper` "asserts on B300 --
FA4-cute paged+varlen gap". This is the specific assert, reached through
`models/attention.py` -> `torch.nn.attention.varlen` -> `_fa4.py`:

```
flash_attn/cute/interface.py:1314
AssertionError: SM100 forward with head_dim=256 does not support seqused_q/seqused_k
```

FA4 dispatches head_dim=256 on SM100 to a dedicated 2CTA kernel that rejects
`seqused_q`/`seqused_k`, and paged decode needs `seqused_k` to infer
`cu_seqlens_k`. The two requirements are mutually exclusive; there is no
configuration that satisfies both. FA3 is not an escape: `_registry` lists
`['FA3', 'FA4']` on this box, but the FA3 kernels are Hopper-only and refuse to
run on SM 10.x (see the `LOCAL (terminal-rl 2026-08-08)` comment in
`models/attention.py`).

**So the Blackwell FA4 commits on this branch (`0646ff3a`, `f0f6d16b`,
`08782607`) apply only up to head_dim 128.** That is not a defect in them --
`f0f6d16b` states its verification was Qwen3-0.6B, whose head_dim is 128.
Qwen3.5-9B is head_dim 256 and falls outside. Use `vllm_native`.

### Why the alphabet_sort smoke never showed any of this

`run_alphabet_sort.sh` differs on exactly the axes that matter, so it takes a
different branch at step two of the causal chain:

- Qwen3-0.6B, head_dim 128 -> the wrapper's FA4 varlen path works
- it therefore runs the wrapper, and its logs read
  `Using AttentionBackendEnum.CUSTOM` -- vLLM never selects an attention backend
  of its own
- the script already exports `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `VLLM_ALLREDUCE_USE_FLASHINFER=0`, with the comment "there is no CUDA 13
  toolkit with curand headers here"

The chain that bites the 9B is: head_dim 256 -> wrapper unusable -> `vllm_native`
-> vLLM picks its own attention backend -> it picks FlashInfer -> FlashInfer
JIT-compiles -> no usable nvcc -> dead. alphabet_sort branches away at step two
and meets none of it. **Read that script before debugging a new environment.**

---

## 4. What `vllm_native` needs on this host

Four things, all of which failed loudly one after another:

1. **`flashinfer-python` + `flashinfer-cubin` installed.** vLLM's sampler probes
   FlashInfer support with a bare `from vllm.v1.attention.backends.flashinfer
   import FlashInferBackend`, uncaught, so a missing package kills every
   generator with `ModuleNotFoundError` rather than falling back.
2. **`VLLM_USE_FLASHINFER_SAMPLER=0`.** FlashInfer ships no prebuilt sampling
   kernel for sm_103a and JIT-compiles it on first sample. The only nvcc on this
   box is CUDA 12.8, which cannot target `compute_103a`; `CUDA_HOME` also points
   at a `/usr/local/cuda` that does not exist. Failure mode:
   `RuntimeError: Ninja build failed ... /usr/local/cuda/bin/nvcc: not found`,
   at the first sampled token.
3. **`SWE_GEN_VLLM_ATTENTION=FLASH_ATTN`.** Same JIT wall one layer down: vLLM
   auto-selects `FLASHINFER` for attention and its
   `trtllm_batch_decode_with_kv_cache` JITs on first decode. This env var existed
   but was **read only on the wrapper path**; it now pins the backend on the
   native path too (`actors/generator.py`).
4. **The fp32 lm_head patch had to move.** `_install_fp32_native_lm_head`
   overrode `LogitsProcessor._get_logits`, re-implementing its whole body --
   gather, vocab-padding trim and all. vLLM added a `skip_gather` parameter and
   every generator died with `TypeError: ... takes 4 positional arguments but 5
   were given`. It now overrides `_apply_head`, which is the matmul alone, and
   leaves the surrounding logic to vLLM. The silent version of this bug is worse
   than the crash: upstream now gathers only when `lm_head.tp_size > 1` while the
   old copy gathered unconditionally.

Anything that JITs must also be redirected off the root filesystem
(`FLASHINFER_CACHE_DIR`, `TRITON_CACHE_DIR`) -- see section 6.

---

## 5. Sandbox: this Daytona region is ephemeral-only

Every create fails with `Only ephemeral sandboxes are permitted in this region`
unless `ephemeral=True` is passed. The branch did not read
`TT_DAYTONA_EPHEMERAL` at all; the upstream `create_kwargs` block was ported into
`harness/sandbox/daytona.py`. With `TT_DAYTONA_EPHEMERAL=1` a real task row --
server-side Dockerfile build, `--inject-agent-runtime` tmux baked in -- boots,
execs as root and tears down cleanly.

`daytona_unstartable.ids` from the private ops repo lists 15 tasks that build and
then never reach running state. **They appear in no HuggingFace id list**, and
two of them are in a mix built from `train_ready_ids.txt` +
`main_pool_ids.txt`. Each burned ~224 failed creates on the reference run, so
they are worth excluding explicitly.

---

## 6. Two data-hygiene items

**The harbor canary was reaching the policy.** All TerminalWorld
`instruction.md` files carry `<!-- harbor-canary GUID ... -->`, and the dataset
card asks consumers to strip it from what the model reads while leaving it in the
build and grading files. `prepare_rts_data.py` did not, so 668 of 1909 rows
carried it in both `prompt` and `metadata.problem_statement`. It now strips it
from the instruction only.

**The holdout is the last 64 rows of the file** (`_TMAX_9B_HOLDOUT_N`, not
configurable). Anything that appends to the mix rotates the eval set silently.
Rebuild the mix with a script rather than appending to it.

---

## 7. The failure that is not a failure: SIGTERM with an empty log

Three runs died with no traceback -- once mid-boot, once at
`[trainer_loop] step 1: begin` -- and the only evidence was
`Killed(sig=15)` / `[launch] trainer exited with status 143`. All three
coincided with the supervising agent session rolling over. `setsid nohup` did not
protect them, and a `tmux new-session -d <command>` session disappears with the
command, taking the pane output and the exit status with it.

Do not debug this as a code fault. Run under `systemd --user` instead: its own
cgroup, an exit code in the journal, and `Restart=on-failure` to resume from the
newest checkpoint. `loginctl enable-linger` keeps it alive across logout. The
reference host reached the same conclusion; the unit here mirrors theirs.

A related trap of our own making: `exec <trainer> | tee train.log` reports
**tee's** exit status, so a crashed trainer looked like a clean exit 0 twice
before it was noticed. Same shape as the `pytest ... | tail && git commit` entry
in the ops runbook.

---

## 8. Repository changes this bring-up required

| file | change |
|---|---|
| `harness/sandbox/daytona.py` | honor `TT_DAYTONA_EPHEMERAL` (ported from upstream) |
| `actors/generator.py` | fp32 lm_head patch moved to `_apply_head`; `SWE_GEN_VLLM_ATTENTION` now applies on the native path |
| `examples/tmax/config_registry.py` | `SWE_GEN_DP` and `SWE_GPU_MEM_LIMIT` overrides (ported) |
| `examples/tmax/prepare_rts_data.py` | strip the harbor canary from agent-visible instructions |

Host-side launcher, env file, mix builder and status script live outside the
repo under `/ssd2/k3/yichuan/rl/`; the local quick reference is
`/ssd1/k3/yichuan/CHEATSHEET.md`.

---

## 9. Open items

- **No optimizer step has been observed here.** The loop reached step 1 and was
  stopped before rollouts completed. The first thing to confirm on the next run
  is `forward_backward done, loss=<finite>` followed by `weights pulled`.
- **`SWE_MAX_NUM_SEQS`**: the reference launcher hardcodes 32 while its env file
  says 256, and the env file wins there. 256 is used here, untested against 32.
- **8-GPU split is unmeasured.** `2 + 6` was chosen to mirror the reference
  trainer width, not because it was compared against anything.
- **No TB-2.0 eval set.** `SWE_TB2_VAL_DATA` is wired but no `tb2_eval.jsonl` has
  been built, so `SWE_VAL_SAMPLES=0` and there is no fixed external yardstick
  yet. Note the reference run's blocking boot validation costs ~2 hours per
  restart once enabled.
- **A CUDA 13 toolchain would remove two workarounds.** With an nvcc that can
  target sm_103a, both FlashInfer paths (sampler and attention) become available
  and the environment matches the reference lock more closely.

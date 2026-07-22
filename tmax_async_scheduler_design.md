# TMax Async Scheduling: Fast-Task Bias and Windowed FIFO

## Status and scope

This note compares the current TorchTitan TMax scheduler with Open-Instruct and
the local TBR `ElementBuffer`, then proposes a bounded-reordering scheduler for
TorchTitan. It is a design note, not an implementation.

The first run containing both the FP32-master AdamW configuration and the
work-conserving sibling gate is:

- MAST job: `torchtitan-rl-rl_grpo_qwen3_5_9b_tmax-89db01`
- dump: `q35_9b_tmax_asyncgate_dppo512_20260721_021715`
- W&B: `https://meta.wandb.io/yichuan/torchtitan/runs/5qj78pte`

Keep that run as the unbounded take-any baseline. Do not mix a concurrency
change into it.

### Baseline observation through Step 41

The run passed the first checkpoint boundary on 2026-07-21:

- the job remained `RUNNING` with zero restarts;
- loss and grad norm remained finite through Step 41;
- all eight trainer ranks finished the Step-20 checkpoint in about 65 seconds;
- the checkpoint contains metadata plus all eight distributed shards;
- Step 21 completed after the save, covering the next forward/backward,
  optimizer update, weight push, and generator pull.

The Step-40 checkpoint also contains metadata plus all eight distributed
shards, and Step 41 completed normally afterward. The run remained healthy when
Step 42 began with zero restarts. This proves the FP32-master AdamW and text-only
vision-freeze checkpoint path is operational across repeated saves. It does not
prove the scheduler is distributionally aligned.

The `rl_grpo` text in the MAST and W&B names is a legacy config identifier, not
the runtime algorithm. The serialized loss is DPPO with TV threshold `0.1` and
no ratio cap. DPPO-only mask and divergence metrics are present, while the GRPO
ratio-clip metric is absent. Step-20 checkpoint metadata contains 1,569 FP32
optimizer tensors: `step`, `exp_avg`, and `exp_avg_sq` for 523 parameters.

The comparable Open-Instruct metric is
`val/avg_group_performance_pre_filter`, not `scores`. Despite its name, it is a
training-batch attempted reward, not held-out validation. `scores` covers only
the eight kept partial groups. For binary rewards and equal 32-sibling groups,
this metric is mathematically identical to Titan's
`rollout_reward/avg_train_reward/mean` and `rollout_reward/_mean`: both include
all non-stale full-solve, all-failed, and partial groups consumed while finding
eight trainable partial groups. The curves are therefore directly comparable as
online attempted reward. The Titan run has held-out validation disabled, so the
comparison still cannot separate policy improvement from scheduler-induced
changes in the attempted task mix.

Through the first 37 packed batches:

- Titan accepted 525 groups: 296 partial, 112 full-solve, and 117 all-failed;
- Open-Instruct accepted 521: 296 partial, 103 full-solve, and 122 all-failed;
- cumulative group-weighted attempted reward was `0.5841` for Titan and `0.5722`
  for Open-Instruct;
- Titan stale-dropped 116 additional groups (67 trainable partial, 49 failed),
  versus 31 for Open-Instruct;
- the resulting stale rates were `18.1%` and `5.6%`, respectively.

The cumulative accepted cohorts are nearly identical. Their order is not:
Titan Step 1 consumed eight partial, two full, and one failed group for reward
`0.8409`; Open-Instruct consumed eight partial and six failed groups for
prefilter reward `0.4040`. Titan therefore begins from an artificially high
completion-selected baseline. Its full-solve share nevertheless rose from
`16.9%` in Steps 1-10 to `28.9%` in Steps 31-37, while nonsubmit fraction fell
from about `16.2%` to `11.3%`. These are compatible with some learning even
though the changing attempted-reward mix looks flat.

The target's first-10 prefilter mean is `0.477`, its Steps 31-37 mean is `0.619`,
and its final-20 mean is about `0.657`. The current Titan horizon is too short
and noisy to conclude that optimizer learning failed. The materially worse
stale-partial rate is the stronger scheduler finding.

Rollout-level Daytona disconnects and wall-clock guard events were contained at
the rollout boundary; there was no controller or trainer failure. These events
must still be labeled in completion-latency analysis so infrastructure failures
are not mistaken for intrinsically hard tasks.

The high stale-partial count is direct evidence of permanent inclusion bias,
not merely noisy early-step reward. Keep the run for numerical and throughput
comparison, but do not treat it as scheduler parity with Open-Instruct.

## FP32 change: exact meaning

Commit `2237cf1bd` (`[rl] tmax: use fp32 master and aligned AdamW`) did not make
the model forward fully FP32 and did not introduce the FP32 LM head.

For the normal 9B recipe it changed:

- persistent FSDP parameter shards from BF16 to FP32;
- FSDP forward/all-gather compute remains BF16;
- gradient reduction is FP32;
- ordinary fused AdamW now stores its parameter target, `exp_avg`, and
  `exp_avg_sq` in FP32;
- AdamW settings now match Open-Instruct: LR `1e-6`, betas `(0.9, 0.999)`,
  epsilon `1e-8`, and weight decay `0`.

The relevant configuration is in
`torchtitan/experiments/rl/examples/tmax/config_registry.py:154-170` and
`:330-340`. The old recipe used full-BF16 persistent parameters and the
`fused_opt_states_bf16` optimizer path. The model's effective forward dtype is
therefore unchanged; the optimizer state and update target are what gained
precision.

The FP32 LM-head matmul is an older, independent `CastLinear` feature. It casts
the BF16 gathered weight and input to FP32 for the projection. It must not be
used as evidence that commit `2237cf1bd` changed the entire model compute dtype.

The Step-20 checkpoint fix is present: the unused text-only
`vision_encoder` is frozen before optimizer construction in
`torchtitan/experiments/rl/actors/trainer.py:160-190`. Old full checkpoints from
the BF16-state recipe are not compatible with the new optimizer schema; this
run correctly started from a fresh dump.

## The bias being controlled

The scheduler directly selects for low wall-clock latency, not for reward.
It becomes an easy-task shortcut only when low latency correlates with high
solve rate, for example because easy tasks submit in fewer turns. Fast failures
can also complete early.

There are three distinct effects:

1. **Step-assignment bias.** Fast groups affect earlier optimizer steps while
   slow groups affect later steps.
2. **Permanent inclusion bias.** A sufficiently slow group can become stale or
   remain unfinished at shutdown and never train.
3. **Cold-start policy bias.** Filling a large run-ahead buffer at policy
   version 0 lets the fastest subset determine the first several updates.

Zero-variance filtering is separate. All-solved easy groups and all-failed hard
groups are both dropped because their centered advantages are zero. The target
training distribution is the partial-solve band, but greedy completion order can
still favor the faster end of that band.

Metrics must keep two reward denominators separate:

- attempted/prefilter reward: includes partial and zero-variance groups;
- kept reward: includes only the eight partial groups used for gradients.

Open-Instruct's `scores` is the kept reward, while
`val/avg_group_performance_pre_filter` is the comparable attempted reward.
TorchTitan's current `rollout_reward/_mean` is attempted reward for the groups
consumed while forming the batch.

## Current TorchTitan scheduler

The 9B recipe has:

- `E = 8` prompt groups per train step;
- `G = 32` sibling rollouts per group;
- `max_offpolicy_steps = 4`;
- `N = (4 + 1) * 8 = 40` active prompt groups;
- `C = 512` active sibling rollouts, split as 64 across each of 8 rollout
  worker processes.

The controller creates 40 group-loop tasks, but the current 512-concurrency
recipe initially admits 32 groups and grows to 40 as trainable groups move
downstream. The `89db01` baseline predates that startup change and admitted all
40 immediately. Each admitted group creates all 32 sibling tasks and waits for
all of them. The per-worker issue gate is work-conserving: every completed
sibling releases one slot to the next pending sibling, ordered by
`(group_id, rollout_idx)` within that worker.

Group aggregation still has the correct all-sibling barrier. A group is
finalized only after all 32 siblings complete. The new gate removed idle rollout
slots behind a slow sibling; it did not change training-batch selection.

The batcher and Open-Instruct use the same inclusive freshness boundary: a
group at age 4 remains eligible and a group at age 5 is dropped. Titan repeats
the check when the trainer consumes a packed batch because the queue can age a
previously valid batch. A queued batch that has crossed the limit is dropped and
its group slots are released before the trainer waits for a fresh replacement.

The selection path is unbounded take-any:

1. `take_finalized()` scans the admission-ordered map and skips older inflight
   groups (`components/work_buffer.py:182-222`).
2. Zero-variance groups are immediately released.
3. The batcher emits as soon as it has the first eight partial groups
   (`components/batcher.py:103-166`).

Thus selection is unbounded take-any across the current active population.
Forty groups are visible at one instant, but selected slots are later released
and refilled. A slow head can therefore be bypassed by more than 39 newer
groups over time. This is not equivalent to a fixed `W=40` window.
Worker-local issue priority does not fix global completion-order selection.

## Open-Instruct TMax scheduler

Open-Instruct is also completion-order active sampling; it is not dataset FIFO.

- It initially enqueues `async_steps * E = 4 * 8 = 32` prompt groups
  (`open_instruct/data_loader.py:1686-1698`).
- One prompt request expands into 32 async sibling requests
  (`open_instruct/vllm_utils.py:519-543`).
- The prompt group is published only after all 32 finish and reward computation
  completes (`open_instruct/vllm_utils.py:560-604`).
- The data-preparation loop consumes the global result queue until eight
  nonzero-variance groups have arrived (`open_instruct/data_loader.py:1107-1115`
  and `:1256-1307`).
- Every consumed result is immediately replaced, maintaining a rolling
  32-group population (`open_instruct/data_loader.py:1179-1192`).
- Results older than four model steps are dropped (`:1130-1162`).

Here "older" is strict: Open-Instruct drops only when
`training_step - model_step > 4`, so it accepts age 4 directly in the current
step. Titan's conservative prequeue check and Open-Instruct's consumption-time
check are therefore not the same freshness policy despite both configurations
saying four steps. This contributes to Titan's higher stale-drop count.

It does not have a gradual cold-start warmup. Its apparently weaker shortcut in
one trace is not a scheduler guarantee. Important practical differences are its
256-sandbox environment pool, a 32-group rather than 40-group population,
different shuffle implementation, and a different rollout scaffold.
The 256 value is confirmed in the target `qcgp4uft` run's `output.log`, not
inferred from a launcher default. Although 32 groups expose 1024 logical
siblings, only eight group-equivalents can hold an environment concurrently.
TorchTitan at concurrency 512 exposes sixteen group-equivalents, so its active
completion race is roughly twice as wide at the sandbox boundary.

## Windowed FIFO from the figure

Let the immutable generation order be

```text
Q = [T0, T1, ..., T(N-1)]
```

and let `i` be the oldest unretired entry. A paper-style fixed Windowed FIFO of
width `W` allows completed entries only from:

```text
[Ti, ..., T(i+W-1)]
```

Within that fixed window, ready work may be consumed without waiting for every
older entry. Work beyond the boundary is blocked even if complete. Consuming a
non-head entry spends one of the `W-1` bypass credits but does not expose a new
entry beyond the boundary. When the head retires, the head advances across
already-retired holes and a new fixed window is established.

The useful invariant is:

```text
no unfinished head may be bypassed by more than W - 1 consumed entries
```

`W=1` is strict FIFO. `W=N` approaches greedy take-any. The figure recommends
starting around `W=0.3N`, then measuring the throughput/bias curve.

## Local TBR ElementBuffer

The local sources are:

- `/data/users/yichuan/fbsource/genai/msl/rl/controllers/element_buffer.py`
- `/data/users/yichuan/fbsource/genai/msl/rl/controllers/simple_buffer_async.py`

TBR tracks `ENQUEUED -> INFLIGHT -> DONE` in an `OrderedDict`. With `E`
elements per train step:

- capacity is `mean_age * E` (`element_buffer.py:223-225`);
- the selection scan covers the first `windowed_fifo * E` current queue
  positions (`:231-233`, `:359-365`);
- each fed element increments the age of every remaining inflight/done element
  (`:388-423`);
- an inflight element reaching `max_age * E` either stalls the batcher or is
  evicted (`:321-386`); enqueued elements do not age or trigger this check;
- default controller values are mean age 8, max age 32, and windowed FIFO 4
  (`simple_buffer_async.py:723-751`).

TBR also limits pending microbatches to one by default, preventing generated
data from running arbitrarily far ahead of training
(`simple_buffer_async.py:714-721`, `:3303-3315`).

### TBR warmup

TBR separates full capacity from effective startup capacity:

```text
N_full = mean_age * E
N0 = min(max(1, int(num_actors * actor_multiplier)), N_full)
increment = max(1, int((N_full - N0) / E / mean_age))
```

Only `N0` entries are admitted initially. Every fed element increases effective
capacity by the fixed increment until `N_full` is reached
(`element_buffer.py:235-249`, `:404-410`). This avoids filling the complete
off-policy window at version 0 while still starting enough elements to occupy
the actors.

There are important caveats in the local implementation:

- the default multiplier is 0.5, so it starts half as many elements as actors
  despite the comment saying the initial size should saturate actors;
- capacity grows only after a successful feed, so a hung initial cohort can
  prevent warmup from expanding;
- filtered or empty elements still count as feeds and expand capacity without
  an optimizer step;
- with the base `N_full = mean_age * E` formula, the increment is effectively
  one whenever warmup is active;
- restart skips warmup only in the done-example-cache recovery path.

TBR buffer age is also not identical to policy age. Filtering, multiple examples
per element, and dynamic packing can change how many fed elements correspond to
one optimizer step. TBR therefore retains a separate off-policy manager as the
authoritative freshness check. TorchTitan should do the same rather than
replacing its policy-version check with a buffer-age approximation.

### Important semantic difference from fixed Windowed FIFO

TBR deletes any fed entry from the `OrderedDict`, refills the queue, and then
recomputes the first `W` current positions. Consequently, if the oldest entry is
stuck while other window entries are repeatedly consumed, newly appended work
can shift into the first `W` positions. For inflight stragglers, the local
implementation's eventual stop is therefore enforced by `max_age`, not solely
by `W-1` fixed bypass credits. This is not a universal bound on every queue
head: `ENQUEUED` entries do not age, and retry work can take actor priority.

This is a valid age-controlled scheduler, but it is not identical to the strict
fixed-window invariant stated above. TorchTitan should choose the intended
semantic explicitly rather than copying the TBR class name alone.

One TBR behavior should not be copied: it releases an element-buffer slot when
the element is handed to the downstream batcher. TorchTitan keeps an active
group charged through training and generator weight pull. That stronger
end-to-end backpressure prevents newly admitted work from becoming stale behind
already prepared batches and should remain intact.

## Proposed TorchTitan design

### 1. Separate three limits

Do not overload one concurrency number:

- `active_group_capacity`: admitted groups, currently 40;
- `rollout_concurrency`: active sibling slots, currently 512;
- `selection_window_groups`: groups eligible for batch selection.

Keep sibling issue work-conserving. Apply Windowed FIFO only at group selection,
after the correct 32-sibling aggregation boundary.

### 2. Add fixed-window ordering state

Give each admitted group a monotonic `admission_seq`. Retain a small ordering
ledger even after a group is selected:

```text
head_seq
window_end_seq = head_seq + W - 1
retired_seqs
```

`take_finalized()` may return only a finalized group with
`admission_seq <= window_end_seq`. Prefer the oldest ready group for deterministic
behavior. Selecting a non-head group marks it retired but does not expand
`window_end_seq`. When `head_seq` retires, advance over consecutive retired
sequences and establish the next boundary.

Every terminal disposition must retire scheduling order exactly once:
trainable partial groups, all-solved and all-unsolved zero-variance groups,
empty or otherwise untrainable groups, reward-partial groups, stale groups, and
explicit evictions. Missing any path can pin `head_seq` forever. Trainable
groups enter the batcher while dropped groups release their active slot. A
group's capacity slot can remain charged until trainer consumption while its
ordering entry is already retired; ordering and resource accounting are
different concerns.

### 3. Candidate values for TMax-9B

The paper-faithful first experiment should use:

```text
E = 8
G = 32
N_full = 40
W = ceil(0.3 * 40) = 12 groups
```

TMax filtering makes `W=12` more aggressive than it would be without
zero-variance drops. With observed partial-group retention around 0.60-0.73, a
12-group window often contains fewer than eight partial groups. At retention
0.65, the binomial probability of fewer than eight partials is about 42% for
`W=12`, 7% for `W=16`, and 0.6% for `W=20`. Zero-variance retirement prevents
permanent deadlock, but a small frozen window can produce intentional head
stalls.

Therefore use `W=12` to test the paper policy itself, instrument window-block
time, and move to `W=16` only if the throughput cost is excessive. `W=16` is the
likely production compromise; it should not replace the `W=12` measurement.

### 4. Add cold-start warmup without idling sibling slots

For each worker gate, enough groups must be present to expose strictly more
siblings than that gate's capacity. With worker capacities `C_i`, the exact
headroom lower bound is:

```text
N_headroom = sum(floor(C_i / group_size) + 1)
N0 = min(N_full, max(N_oi, N_headroom))
N_oi = async_steps * groups_per_train_step
```

For 512 split across eight workers, `N_headroom=24`, while the Open-Instruct
startup population is `N_oi=32`, so the current recipe starts at 32 and grows to
40 as trainable groups move downstream. For 1024, `N_headroom=40`, so it starts
at the full 40-group capacity. This is implemented in the TMax config registry;
an explicit `SWE_INITIAL_ACTIVE_GROUPS` remains an experimental override.

The worker pool is capped at the number of available group-loop lanes, and the
global trajectory limit is split exactly across the resulting workers. The
recipe rejects an even worker split when `N_headroom > N_full`; otherwise it
would advertise a work-conserving concurrency that some worker gates cannot
reach. The default 8-worker 512 and 1024 layouts both satisfy the invariant.

At concurrency 1024, warmup cannot shrink the active population without
sacrificing immediate refill capacity. The selection window still bounds
shortcut bias independently. This is another reason to establish the
512/Windowed-FIFO baseline before changing concurrency.

### 5. Bound pathological heads explicitly

Use two controls rather than silently reverting to greedy behavior:

- a policy-age limit, retaining the existing stale-data contract;
- a head wall-clock/max-age action that is observable and configurable.

For TMax, a rollout already has a finite wall budget. Initially, block at the
window boundary and let that timeout resolve the head. If production throughput
requires eviction, log it as a distribution-changing event and requeue the same
dataset item under a fresh policy instead of silently replacing it with a newer
task.

### 6. Align freshness at consumption, not by changing the number

Do not set Titan's cap to five merely to imitate Open-Instruct's observed
acceptance of age four. That would also permit age-five data at trainer
consumption and would change the learning contract.

The current implementation makes the final four-step decision at the boundary
where a batch is consumed. It rejects a whole queued batch if any sample is age
five or older, releases the eight charged trainable-group slots, and waits for a
fresh batch for the same optimizer step. This preserves the hard freshness
contract without crashing the run. A future refinement can preserve group
boundaries and refill only stale groups, or use a trainer-ready handshake so
stale batches are never packed in the first place.

## Metrics required before landing

Add per-step and cumulative metrics for:

- `selection/window_size_groups`;
- `selection/head_group_id` and `selection/window_end_group_id`;
- `selection/bypass_count` and maximum bypass per head;
- time blocked because no finalized group is inside the window;
- ready groups inside and outside the window;
- head wall age and policy age;
- warmup effective active capacity;
- attempted reward, kept reward, and zero-variance class counts;
- completion rank versus solve fraction and rollout wall time;
- stale drops and max-age evictions/requeues;
- age at selection and age at trainer consumption, with explicit age-four
  acceptance/drop counts.

The invariant `max_bypass_per_head <= W-1` must be directly asserted in tests
and visible in logs.

## Test plan

Unit tests should cover:

1. `W=1` matches strict FIFO.
2. A completed group outside the fixed window remains blocked.
3. Consuming a non-head group does not replenish its bypass credit.
4. Retiring the head advances across already-retired holes and opens the next
   window.
5. Zero-variance and stale retirement advance ordering correctly.
6. Close/cancellation wakes all window waiters.
7. Warmup starts at `N0`, grows monotonically, and caps at `N_full`.
8. Sibling-gate utilization and per-worker waiter depth are reported during
   warmup, including multiple simultaneous slow tails.
9. Property tests verify the `W-1` bypass bound over random completion orders.

Run at least 10 training steps for each performance comparison:

- current unbounded take-any (current run as baseline);
- fixed Windowed FIFO `W=12`, concurrency 512, all 40 groups active;
- `W=16` only if `W=12` has excessive window-block time;
- 24-to-40 warmup as a separate A/B after selecting `W`;
- strict FIFO `W=1` only as a diagnostic;
- after choosing a scheduler, compare concurrency 512 versus 1024.

Before submitting a fixed-window run, replay the current run's completion trace
through `W=12`, `W=16`, and `W=20`. A window head can otherwise hold the batcher
until the rollout's one-hour wall budget plus its guard interval resolves. Trace
replay cannot predict changed contention, but it catches obviously impractical
window sizes without consuming a live run.

Use the same dataset order and record every attempted prompt. Compare moving
averages rather than individual steps. The scheduler change is successful if it
reduces the correlation between completion rank and solve rate without a large
increase in trainer idle time or stale drops.

## Recommendation

Keep the current run unchanged as the first clean FP32-master plus
work-conserving-gate baseline. First replay its completion trace under
`W=12/16/20`. The next live scheduler experiment should then change one
variable: fixed Windowed FIFO with the replay-selected window, 512 sibling
concurrency, and the existing 40 active groups. Start evaluation at `W=12`, but
use the trace and window-block metrics to move to `W=16` or `W=20` if needed.
Test 24-to-40 warmup and then concurrency 1024 only as later, separate
experiments; combining them would make attribution ambiguous.

Reward parity with `qcgp4uft` is a separate experiment from scheduler causal
attribution. Keep the current run through Step 100, then evaluate checkpoints
0, 20, 40, 60, 80, and 100 on the same fixed 64-prompt cohort with fixed
sampling seeds. Training-batch attempted reward cannot answer whether the
policy improved because the async scheduler changes the evaluated task mix at
every step.

For a direct framework-parity diagnostic, intentionally match the target run's
observed execution population in one run:

```text
rollout_concurrency = 256
active_group_capacity = 32
max_offpolicy_steps = 4
loss = DPPO
optimizer state and update target = FP32
```

Also use the same dataset order and checkpoint/tokenizer hashes, enable the
fixed validation cohort, and log prompt IDs at admission, completion,
selection, stale drop, and trainer consumption. This parity run changes two
scheduler dimensions as a bundle on purpose; it answers whether the frameworks
can produce comparable learning curves, not which dimension caused a change.

If fixed-cohort reward remains flat, stop tuning concurrency and first run a
same-packed-batch numerical comparison: gradient norm and selected-parameter
Adam update norm in both frameworks, followed by a generator A/B with CUDA
graphs disabled and detailed log-probability-tail logging. Only after those
checks should ratio caps or learning rate be changed. Raising concurrency to
1024 before bounding selection order would widen the completion race and make
the current fast-task shortcut and stale-partial loss worse.

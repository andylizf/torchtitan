# TMax Windowed FIFO Scheduling Design

## 1. Document Status

This document records the scheduling analysis as of 2026-07-22 and compares:

- the current TorchTitan TMax scheduler;
- MSL/TBR's `SimpleBufferAsyncController + FIFOElementBuffer`;
- the fixed, anchored Windowed FIFO shown in the paper.

TorchTitan now has an optional MSL-style sliding selection window over the
current active admission queue. It is disabled by default, so existing recipes
still use take-any over the entire active window. TMax enables it with
`SWE_SELECTION_WINDOW_GROUPS=W`.

## 2. Problem Statement

Agent rollout durations vary widely. Easy tasks may finish in a few minutes,
while difficult tasks may run for nearly an hour. If each training batch always
takes the first groups to finish, early training will be biased toward short
tasks. When rollout duration is correlated with solve rate, this creates an
easy-task shortcut:

1. Fast tasks affect optimizer steps earlier.
2. Slow tasks may already be stale by the time they finish and therefore never
   participate in training.
3. Completion-speed selection inflates the initial reward, making subsequent
   improvement appear smaller.

Strict FIFO eliminates this completion-order bias, but the oldest unfinished
task blocks every task behind it. Windowed FIFO provides a tunable tradeoff
between throughput and distributional consistency.

## 3. Current TorchTitan TMax Scheduling

The standard 9B configuration is:

```text
E = 8       # trainable prompt groups required per optimizer step
G = 32      # sibling rollouts per prompt group
K = 4       # max_offpolicy_steps
N = 40      # maximum active prompt groups
C = 512     # active sibling rollout slots
workers = 8 # 64 sibling slots per worker
```

With `C=512`, cold start initially admits 32 groups. As trainable groups enter
the downstream pipeline, effective capacity gradually grows to 40. With
`C=1024`, it admits 40 groups immediately, avoiding the case where exactly
1,024 siblings fill every slot without leaving a waiter.

### 3.1 Trajectory Scheduling

A group immediately creates 32 sibling coroutines. Each sibling independently
acquires a worker-local slot. When it finishes, it immediately hands the slot
to the next `(group_id, rollout_idx)` waiter. A slow sibling occupies only one
slot; the scheduler does not wait for all 32 slots for a group to become free.

Relevant code:

- `torchtitan/experiments/rl/examples/tmax/rollouter.py`: `_RolloutIssueGate`
- `torchtitan/experiments/rl/examples/tmax/rollouter.py`: `run_group_rollouts`

This layer is work-conserving and should be preserved. Windowed FIFO must not
change sibling issuance.

### 3.2 Group Selection

The current `take_finalized()` scans the entire active map in admission order:

```text
Find the earliest-admitted group that is already FINALIZED.
Skip every older group that is still INFLIGHT.
```

If group 0 is slow while groups 1, 2, 7, and 19 have finished, the batcher can
continue consuming the later groups. As consumption and replenishment continue,
an arbitrary number of new groups can bypass group 0.

The default implementation is therefore unbounded take-any. With a configured
window, the batcher scans only the first `W` entries in the current active map.
Selecting any group removes it immediately, so later entries slide into the
prefix even while the oldest group remains unfinished. This matches MSL's
current-prefix semantics. `SWE_STRICT_FIFO=1` remains a compatibility alias for
`W=1`.

Relevant code:

- `torchtitan/experiments/rl/components/work_buffer.py`: `take_finalized`

### 3.3 Slot Lifetime

After the batcher takes a trainable group, its active-group credit is not
released immediately. It continues to hold the credit through:

```text
pack -> trainer consume -> forward/backward -> optimizer -> generator weight pull
```

Only after the weight pull does the trainer release all eight group credits for
the step at once. Zero-std and stale groups release their credits immediately in
the batcher.

This end-to-end credit accounting limits run-ahead across the entire pipeline
and should be preserved.

## 4. MSL FIFOElementBuffer

The MSL defaults are defined in:

`/data/users/yichuan/fbsource/genai/msl/rl/controllers/simple_buffer_async.py`

```text
mean_age = 8
windowed_fifo = 4
max_age = 32
warmup = true
warmup_actor_multiplier = 0.5
max_pending_microbatches = 1
```

Let `E = element_buffer_examples_per_step`. Then:

```text
queue capacity       = mean_age * E
selection window     = windowed_fifo * E
max-age threshold    = max_age * E
```

Using `E=8` as in TMax gives:

```text
queue capacity       = 8 * 8  = 64 elements
selection window     = 4 * 8  = 32 elements
max-age threshold    = 32 * 8 = 256 feed events
```

Relevant code:

- `/data/users/yichuan/fbsource/genai/msl/rl/controllers/element_buffer.py:223`
- `/data/users/yichuan/fbsource/genai/msl/rl/controllers/element_buffer.py:321`

### 4.1 State Machine

Each element transitions through:

```text
ENQUEUED -> INFLIGHT -> DONE -> FED/removed
```

Each of the `num_actors` long-running actor loops acquires one element. An actor
marks the element `DONE` only after completing the entire element.

The MSL controller itself does not have an explicit per-sibling global slot gate
like TorchTitan TMax. How trajectories fan out within an element is determined
by the specific actor.

### 4.2 Windowed FIFO Selection

The MSL batcher scans only the first `windowed_fifo * E` positions in the current
queue and returns the first `DONE` element within that prefix:

```text
queue = [0, 1, 2, ..., 63]
window = first 32 positions

0 is INFLIGHT
1 is INFLIGHT
2 is DONE
40 is DONE

=> feed 2
=> 40 is not visible because it is outside the window
```

This is bounded look-ahead: it can bypass a straggler within the window, but it
cannot directly take a fast task outside the window.

### 4.3 Actual Semantics of MSL `max_age`

Whenever MSL feeds an element, it removes that element and increments `age` for
every remaining `INFLIGHT` and `DONE` element in the queue. `ENQUEUED` elements
do not age.

When any `INFLIGHT` element satisfies:

```text
age >= max_age * E
```

the default `element_buffer_evict_at_max_age=False` behavior is:

```text
Stop feeding later DONE elements.
Wait for the INFLIGHT element at max age to finish.
```

With `element_buffer_evict_at_max_age=True`, the behavior is:

```text
Remove that INFLIGHT element.
Free its position and continue running.
```

For `E=8, max_age=32`, the threshold is 256 feed events. This approximately
means that the element has been bypassed by 32 step-equivalents of data, but it
does not guarantee that the trainer has completed 32 optimizer updates.
Zero-output elements, batcher packing, and downstream backpressure all affect
the relationship between these quantities.

The three MSL parameters therefore have separate responsibilities:

| Parameter | Responsibility |
|---|---|
| `mean_age` | Controls total buffer capacity and average run-ahead |
| `windowed_fifo` | Limits how far a single selection can look ahead |
| `max_age` | Prevents an `INFLIGHT` element from being bypassed forever |

`max_age` is not policy age. `DropOffpolicyManager` computes actual policy
staleness from the sampler `StateVersion`. It allows `age == cap` and drops a
trajectory only when `age > cap`.

### 4.4 Warmup

MSL warmup starts with this effective capacity:

```text
initial = min(max(1, int(num_actors * warmup_actor_multiplier)), full_capacity)
```

Each fed element increases effective capacity by a fixed increment until it
reaches `mean_age * E`. An element advances warmup when it is fed even if it
ultimately contains no trainable samples.

TorchTitan uses different warmup units: it must account for 32 siblings,
worker-local gates, and downstream group credits together. The MSL
`num_actors * 0.5` formula should therefore not be copied directly.

### 4.5 MSL Versus the Paper's Fixed Window

The window in the paper's diagram is anchored at the oldest active entry. The
window advances as a whole only after its head is consumed.

The current MSL implementation instead uses the first `W` positions of the
current `OrderedDict`. Any element fed from within the window is removed
immediately, the queue is then refilled at its tail, and the window boundary
moves forward as a result. Consequently, even if the oldest element remains
unfinished, new elements can gradually enter the window and bypass it.

MSL's Windowed FIFO therefore:

- limits the look-ahead of each individual selection;
- does not provide a strict lifetime bypass bound through the window alone;
- relies on `max_age` stall/evict behavior for the ultimate bound.

## 5. Key Differences Between TorchTitan and MSL

| Dimension | TorchTitan TMax | MSL/TBR |
|---|---|---|
| selection | full-map take-any by default; optional sliding queue prefix | sliding queue-prefix Windowed FIFO |
| slow tasks | optional max-bypass stall, then stale-drop if completed too old | stall or eviction after reaching max age |
| execution unit | 32-sibling prompt group | actor element |
| concurrency | worker-local sibling gate | `num_actors` element lanes |
| slot release | after generator weight pull | when fed to `DynamicBatcher` |
| downstream backpressure | active-group credits cover the entire pipeline | controlled separately by `max_pending_microbatches` |
| policy freshness | checked at group completion and trainer consumption | checked by `DynamicBatcher`/`OffpolicyManager` |
| warmup | OI inventory + sibling gate headroom | actor count * multiplier |

The bounded selection window is the most useful MSL mechanism to adopt. Its
slot lifetime and warmup formula should not be transplanted.

## 6. TorchTitan Design

### 6.1 Preserve Existing Mechanisms

Windowed FIFO should not change the following behavior:

1. Release each trajectory slot immediately after its sibling finishes.
2. Continue waiting for all 32 siblings in a group before computing advantage.
3. Preserve the 32 -> 40 cold start with `C=512` and the initial 40 groups with
   `C=1024`.
4. Hold each trainable group's credit until the generator weight pull.
5. Check policy age both in the batcher and at trainer consumption.

### 6.2 Group Selection Window

`RolloutGroupWorkBuffer.Config` provides an independent parameter:

```text
num_groups_in_selection_window: int | None
```

Selection rules:

```text
None  -> current take-any over the full window
1     -> strict FIFO
W > 1 -> scan the first W entries in the current active admission map and
         return the earliest FINALIZED group
```

The window is based on the current admission-ordered active map, not the current
finalized list. Removing a non-head entry slides the next active admission into
view. New groups are appended at the tail as active credits become available.
The TMax environment variable is `SWE_SELECTION_WINDOW_GROUPS`.

### 6.3 Initial TMax Experiment Values

The current configuration is `N=40, E=8`. Using the paper's empirical
`W ~= 0.3N` ratio as an initial value, despite the different sliding semantics,
corresponds to approximately 12 groups:

```text
baseline: W=None  # current unbounded take-any
trial A:  W=12
trial B:  W=16
control:  W=1     # strict FIFO, only to diagnose the throughput cost
```

`W=12` is closer to the paper's setting. `W=16` is less likely to make the
trainer wait when many groups have zero standard deviation. The first A/B test
should keep rollout concurrency, shuffle, seed, optimizer, and loss unchanged.

### 6.4 Max-Bypass Brake

The selection window and policy stale-drop address different problems:

- the selection window controls the training distribution;
- the policy cap prevents the use of excessively old data;
- the max-bypass brake stops further selection after an `INFLIGHT` group has
  been bypassed by too many later groups.

`RolloutGroupWorkBuffer.Config` provides:

```text
max_bypass_groups: int | None
```

At `bypass_count >= max_bypass_groups`, the batcher stops selecting every later
group until the over-age group leaves `INFLIGHT`. Normal agent execution is
bounded by a per-sibling wall guard, but this is not a complete group-level hard
timeout.

The max-bypass brake is opt-in. `SWE_MAX_BYPASS_GROUPS=32` aligns a diagnostic
run with the standard four-step policy cap and `E=8`; `off`, an empty value, or
an unset variable disables it. MSL validates that its policy cap is at least its
32-step max age, so copying MSL's `32 * E = 256` default into TMax while keeping
a four-step policy cap would make the delayed group stale before the brake
engages.

TMax counts direct later-group selections while an older group is `INFLIGHT`,
rather than every MSL feed event. The two are equivalent for a stuck queue head,
which is the condition this brake handles, but this is not an exact replacement
for every element's MSL age.

The production launcher leaves the brake off. The per-sibling wall guard does
not cover every group-level failure mode: worker RPC, sibling-gate waiting, or
sandbox cleanup can still hang outside it. A global stall is not safe by default
until TMax can hard-timeout or reliably cancel the complete group across its
worker and sibling sandboxes.

Inflight eviction is intentionally not implemented. The controller does not yet
own a reliable cross-`RolloutWorker` single-group cancellation handshake;
releasing its active credit while its sibling sandboxes still run would break
end-to-end capacity accounting.

## 7. Required Metrics

The implementation exposes the following scheduler metrics. Monitor them
alongside the existing stale-drop and policy-age metrics:

```text
rollout_buffer/selection_window_groups
rollout_buffer/eligible_finalized_groups
rollout_buffer/blocked_finalized_groups
rollout_buffer/head_wall_age_sec
rollout_buffer/head_bypass_count
rollout_buffer/max_bypass_count
rollout_buffer/max_bypass_groups
rollout_buffer/num_inflight_at_max_bypass
rollout_buffer/max_bypass_stall_sec
rollout_buffer/max_bypass_stall_count
rollout_buffer/window_stall_sec
rollout_buffer/dropped/stale
train_batch/policy_age
train_batch/policy_age_max
```

Also dump attempted groups for each step: group ID, dataset index, solve count,
wall time, policy age, and whether the group contributed to a gradient. Without
this information, it is impossible to distinguish actual learning from shuffle
differences and scheduler selection bias.

## 8. Acceptance Criteria

Validation must cover more than throughput. The implementation must satisfy:

1. A finalized group outside the window cannot be selected until earlier
   removals shift it into the current prefix.
2. An inflight head can be bypassed within the window.
3. Selecting a non-head group immediately shifts the next active admission into
   the window, matching MSL.
4. A replenished tail group can enter the window while the oldest group remains
   inflight.
5. When multiple groups within the window are finalized, select them in
   admission order.
6. Sibling slots are still released individually; the implementation must not
   regress to waiting for a complete group's worth of slots.
7. Active credits remain conserved across zero-std, stale, and trained groups.
8. Trainer consumption never accepts a batch beyond the policy cap.
9. `W=None` matches the current take-any behavior.
10. `W=1` matches strict FIFO behavior.
11. Repeated sliding may produce more than `W - 1` lifetime bypasses, and the
    bypass metrics record that tail.
12. Reaching `max_bypass_groups` stalls later selection until the over-age
    `INFLIGHT` group becomes terminal.
13. Compare task index, wall time, solve distribution, and attempted reward over
   the first 10 steps.
14. Report throughput, trainer wait time, and stale-partial rate alongside the
    reward curve.

## 9. Conclusion

TorchTitan now addresses trajectory-level slot waste and, when the optional
selection window is enabled, limits each group selection to an older active
prefix. The default `W=None` behavior remains unbounded take-any for
compatibility. MSL's key idea is preserved: total run-ahead, normal selection
range, and extreme-tail protection are separate concerns.

The implemented sliding admission-ordered window preserves the sibling gate,
32 -> 40 cold start, end-to-end credits, and policy freshness checks. It reduces
the instantaneous fast-task race but does not impose a lifetime bypass bound by
itself. The max-bypass stall supplies the eventual scheduler brake while the
group remains healthy, but it is disabled in the production launcher because
some group-level failure paths are still unbounded. A production lifetime bound
requires a hard group timeout or explicit cancellation handshake.

# Handoff: online task evolution on TerminalWorld seeds (Qwen3.5-9B tmax RL)

**Status: the loop works end to end.** This document is the record of what it took
to get there -- specifically the failures along the way, because every one of them
is a trap the next person will otherwise re-discover.

---

## 1. What this is

Close the loop between training and data generation, **continuously**:

```
training rollouts
  -> a group with zero reward variance (k/k pass or 0/k fail) emits a signal
  -> an external model re-tunes that task (k/k -> harder, 0/k -> easier)
  -> the re-tuned task is re-validated in a real sandbox
  -> folded back into the live training mix under a new version
  -> generators hot-reload the mix mid-run
  -> later steps train on the re-tuned task
```

The point is to breed a small seed set into an effectively unbounded one, and to
push easy tasks toward long-horizon difficulty as the policy outgrows them.

Design rationale for the emit site lives in `../TASK_EVOLUTION.md`. This document
is the operational counterpart: what broke.

### Configuration under test

| | |
|---|---|
| Model | Qwen3.5-9B (GDN) |
| Data | TerminalWorld-Seeds-Clean, 859 tasks, pure (no mix) |
| Loss | DPPO, unclipped + TV trust region (delta 0.1) |
| Group | `SWE_GROUP_SIZE=16`, `SWE_MAX_ACTIVE_GROUPS=160`, `SWE_OFFPOLICY_STEPS=5` |
| Sandbox | Daytona, 2 vCPU / 4 GiB / 10 GiB per sandbox |
| Evolution | compound (evolved output feeds the next round), 32 workers, 120 s rounds |
| Eval | TB-2.0, 89 tasks, k=5, one dedicated async eval generator |

### Observed state (step 39)

| Signal | Value | Reading |
|---|---|---|
| Step | 39 | past 1 epoch (~27 steps) -- the loop has closed over the full pool |
| `rollout_reward/_mean` | 0.56 (from 0.66) | expected: evolved-harder tasks entered epoch 2 |
| `loss/dppo_mask_kept_frac` | 1.00 | trust region keeps everything; gen/trainer logprobs agree |
| Evolution | 434 processed, **237 folded**, 25 pending, 21 rounds | |
| Hot reload | `mix_live.v13.jsonl` reaching generators | evolved data is actually training |
| Validation | avg@5 **0.200** / pass@5 **0.337** at step 0 | baseline only -- see D3 |
| Crashes | none | all tracebacks are recoverable sandbox errors |

Evolution outcome breakdown (`evolution_stats.json`):

```
ok (folded)                        237
revalidate_daytona_oracle_failed    85   <- safety working: harder task broke its own oracle
evolve_blocked                      37
revalidate_daytona_error_failed     34
error                               28
revalidate_shortcut_failed          12   <- safety working: task became trivially shortcuttable
junk_infra                           1
```

A rejected re-tune is **not** a loss: the original task stays in the pool. The
55% fold rate is the safety margin, not waste.

---

## 2. Difficulties

Grouped by what kind of problem each one actually was, because that is what
determines where to look next time.

### A. Code bugs -- fixed

#### A1. `statistics.pstdev` crashes on NaN, killing whole groups

*Symptom.* Near-empty dashboards, batch starvation, and evolution signals that
never appeared for some tasks.

*Cause.* Both the zero-std detector and the new evolution emit filtered rollouts
with `reward is not None`. An infra-failed rollout carries a **NaN** reward, which
is not `None`, so it survived the filter. Under Python 3.12 `statistics.pstdev`
raises `AttributeError: 'float' object has no attribute 'numerator'` on NaN. The
exception propagated out of group finalization and took the group with it.

*Fix.* Filter with `is_scored(r)` in both places (`rollouter.py`,
`_maybe_emit_evolution_signal` and `_maybe_annotate_zero_std`).

*Why it was hard to see.* The error text mentions `numerator` and points at
`statistics`, so it reads like a type bug in the metric, not a data bug upstream.
And the visible consequence -- an empty dashboard -- looks like a logging problem.

**Lesson: `is not None` is not a NaN guard. Any reward filter feeding statistics
must use `is_scored`.**

#### A2. Daytona SDK cannot parse multi-line `COPY` (poison-pill tasks)

*Symptom.* Certain tasks failed **all 16 siblings**, every time, with
`ValueError: No escaped character`.

*Cause.* The declarative-image path hands the Dockerfile to the Daytona SDK,
which tokenizes it with `shlex`. A line-continuation (`COPY a.asm \` + newline)
leaves a trailing backslash that `shlex` rejects.

*Fix.* Flatten continuations before writing, in
`harness/sandbox/daytona.py::_declarative_image`:

```python
flattened = re.sub(r"\\\r?\n[ \t]*", " ", self.dockerfile)
```

*Why it was hard to see.* Earlier mixes had been washed through a different
preparation path that already flattened them; only the raw TerminalWorld seeds
carried the continuations. So "it worked before" was true and misleading.

#### A3. The controller found no torchtitan checkout

*Symptom.* Every fold failed with "no torchtitan checkout".

*Cause.* The launcher derived the torchtitan path from `$PWD`. On the controller
host torchtitan is an installed conda package, not a checkout.

*Fix.* Derive it from the installed package and verify a known file exists:

```bash
TRL_TT=$(python -c 'import os,torchtitan;print(os.path.dirname(os.path.dirname(os.path.realpath(torchtitan.__file__))))')
```

#### A4. Hot reload was enabled on the wrong role

*Symptom.* The mix advanced on disk; generators never picked it up.

*Cause.* `SWE_DATA_HOT_RELOAD` was exported inside the launcher's **controller**
branch. Generators run the worker branch and never saw it.

*Fix.* Set it in the submit script, where the env whitelist propagates it to all
roles.

**Lesson: role-scoped env in the launcher is a silent no-op for other roles.
Anything the data path needs belongs in the submit-level whitelist.**

### B. Distributed / filesystem constraints that shaped the design

#### B1. FUSE does not propagate same-name rewrites across hosts

*Symptom.* The controller rewrote `mix_live.jsonl` in place; generators on other
hosts kept reading stale content indefinitely.

*Cause.* On the shared FUSE mount, an mtime bump on an existing filename is not
reliably visible to other clients. Only **new filenames** are.

*Fix.* Write-once versioned files: `mix_live.v<N>.jsonl`. The loader globs for the
newest version (`data.py::_latest_versioned`, `_reload_source`, `_maybe_reload`),
with an mtime fallback for the single-host case, and keeps the last 3 versions.

**Lesson: on a shared FUSE mount, treat "new content" as "new filename". This is
the same constraint that already forced write-once-per-prompt signal files.**

#### B2. Evolution must be decoupled from the drop decision

The evolution signal is emitted in the rollouter the moment a group finalizes
(`rollouter.py:730`), **before** the batcher decides train / stale-drop /
zero-std-drop (`controller.py:1909`). This ordering is deliberate and worth
preserving: a zero-variance group still emits its signal even when it is later
stale-dropped for age. If the emit ever moves downstream of the batcher, roughly
7% of the evolution fuel disappears silently.

### C. Measurement artifacts that looked like bugs

These consumed real debugging time. Recording them so they are not re-litigated.

#### C1. `dppo_mask_kept_frac` fell from ~1.0 to ~0.55 -- metric, not loss

*Symptom.* Switching to TerminalWorld data dropped kept_frac to 0.55, suggesting
the trust region had started rejecting half the tokens.

*What it was not.* Not non-finite generator logprobs (`nan_frac` was 0.000). Not a
divergence increase -- `dppo_divergence_mean` was flat, which is what exposed the
contradiction: if the trust region were really rejecting tokens, divergence would
have risen with it. Not turn-count or length drift.

*Cause.* With `skip_zero_advantage_samples` on, zero-advantage samples are shed
before packing, but the metric denominator was still `num_global_valid_tokens`,
which counts them. TerminalWorld makes this severe because the base model is
bimodal on it -- 53% of groups are zero-std -- so the shed tokens were roughly 45%
of the denominator, making it ~1.8x the right one and the ratio read ~0.55.

*Fix.* Separate the **metric** denominator from the **loss** denominator. The loss
denominator is untouched (gradient equivalence across accumulation depends on it);
per-trained-token metrics now divide by tokens actually packed:

- `types.py:162` -- `TrainingBatch.num_packed_valid_tokens` (`None` -> fall back)
- `components/batcher.py` -- computed over `samples_to_pack`
- `losses/dppo.py` -- `metric_denominator` kwarg; `loss/mean` deliberately excluded
- `actors/trainer.py`, `controller.py` -- threaded through
- `losses/dapo.py` -- accepts and ignores the kwarg, for call parity

Confirmed: 0.557 -> 1.000, with no change to the loss.

**Lesson: when a ratio moves but its numerator's companion metric does not, suspect
the denominator. And keep loss and metric denominators separately named.**

#### C2. "Daytona is over quota" -- wrong, twice

An early diagnosis claimed the account was at 2.3x its vCPU quota using a stale
3000-vCPU figure. The real limit is 20000 vCPU; the account was at ~35%. A
throughput improvement that got attributed to quota relief was just normal batch
progression. The hardcoded 3000 in `cleanup_daytona_zombies.py` has been corrected.

**Lesson: re-read the quota before attributing a slowdown to it.**

#### C3. `BUILD_FAILED 10522` -- an artifact of counting retries

A raw log grep produced a failure table dominated by `BUILD_FAILED` (10522) with
`no space left on device` at 3107. Both numbers were wrong: the first counted every
**retry** of the same event, the second matched `disk` inside unrelated metric
lines. The structured `sandbox_issue` records give the real distribution:

```
kind:   create_retry 7276 | create_failed 1013 | session_create_failed 52
        command_timeout 18 | session_disk_exhausted 14 | command_status_timeout 8
phase:  create 8289 | session_create 70 | command 20
msg:    "Failed to create sandbox: Failure during waiting for sandbox to start"  8272
```

So: sandbox **create** dominates; ~7276 creates retried and mostly recovered, 1013
hard-failed. Disk exhaustion is **14** events. Daytona create-rate 429 is **0** --
`TT_DAYTONA_CREATE_CONCURRENCY=8` is holding.

**Lesson: count structured events, not log lines. Retries inflate line counts by
an order of magnitude.**

### D. Open -- infra and data, not RL code

#### D1. Sandbox create failures (~1013 hard) -- data-side

A subset of TerminalWorld task images never reach `STARTED`. This is the largest
remaining source of unscored rollouts (16-22 per step) and a major contributor to
slow steps. Cost is paid twice: the rollout slot is consumed and produces nothing.

*Recommendation.* Oracle build-filter the mix offline -- build every task once,
drop the ones that fail -- instead of re-discovering it on every rollout. Owner:
data curation (collaborator).

#### D2. ~10-14 heavy tasks exceed the 10 GiB sandbox disk

`session_disk_exhausted` = 14 events. Some tasks do `truncate -s 20G /disk.img` or
compile binutils. Per-task disk sizing is the fix; `est_disk_mb` (~1.1 GB)
understates badly. Deliberately **not** fixed by raising the global disk to 30 GiB
-- too expensive for 14 events. Same tasks were silently dropped by earlier mixes.
Low priority.

#### D3. Eval cadence cannot keep up -- **fix this first**

```
step 20: previous validation pass still running; skipping this one.
Raise validation.interval or add eval capacity.
```

The step-0 baseline took `timing/validate = 8267.7 s` (~2.3 h): 89 tasks x k=5 =
445 rollouts on one eval generator, competing with training for the same Daytona
pool. At ~5 min/step, `SWE_VAL_INTERVAL=20` is ~1.7 h -- shorter than a pass. Every
other eval is skipped, so the effective cadence is ~40 steps and even that is
marginal. Only the step-0 point exists.

This matters more than it looks. `rollout_reward` **falls by design** as tasks are
evolved harder, so without a fixed-difficulty eval there is no way to separate
"the model got better" from "the tasks got harder". The held-out curve is the
only instrument that answers the actual research question.

*Recommendation for the next launch:* `SWE_VAL_INTERVAL=40` **and**
`SWE_NUM_EVAL_GENERATORS=2`. Reducing k from 5 to 3 also works but weakens pass@k.

#### D4. Stale drops (7%) -- do not tune this

`RELEASE(stale_dropped)` is 58 against 763 `TRAINABLE`, with
`train_batch/policy_age/mean` at 3.2 pressing the `SWE_OFFPOLICY_STEPS=5` cap.

This is a **symptom of slow rollout supply**, not a window that is too tight
(`window_stall_sec` 111-491 s, `head_wall_age_sec` ~2400 s). Raising the cap would
feed the trainer staler data -- and GDN's train/infer logprob drift grows with age
-- without improving throughput. The lever is D1, not this knob.

Similarly, `num_untrainable_rollouts` at 16/step is not waste: those zero-std
groups are exactly the evolution fuel.

---

## 3. What to do next, in order

1. **Fix the eval cadence** (D3). Without it the experiment cannot be evaluated.
2. **Oracle build-filter the task pool** (D1). Largest throughput win available.
3. **Watch for the reward turn.** `rollout_reward` should dip as harder tasks land
   and then recover as the policy adapts. A dip that never recovers, with a flat
   held-out curve, means evolution is outrunning the policy -- consider gating the
   difficulty increase on recent solve rate.
4. **Compare against a frozen-data control** on the same recipe. Without it,
   evolution's contribution cannot be separated from ordinary training.
5. Per-task disk sizing (D2). Low priority.

## 4. Where things live

| | |
|---|---|
| Emit site | `../rollouter.py::_maybe_emit_evolution_signal` (design: `../TASK_EVOLUTION.md`) |
| Hot reload | `../data.py::_maybe_reload`, `_latest_versioned`, `_reload_source` |
| Sandbox | `../../../harness/sandbox/daytona.py` |
| Metric denominator | `../../../types.py`, `components/batcher.py`, `losses/dppo.py` |
| Stale-drop decision | `../../../controller.py::_batcher_loop` |
| Evolver | external `terminalworld-seeds/scripts/evolve_ondella.py` |
| Launcher / submit | cluster launcher + submit scripts (site-specific, git-ignored) |

Evolution progress is `<dump>/evolution/evolution_stats.json`; folded output is
`mix_live.v<N>.jsonl` beside the seed mix.

### Toggle

The submit takes `EVOLVE=0|1`. With `EVOLVE=0` the same recipe runs on frozen
data -- this is the control arm for item 4 above.

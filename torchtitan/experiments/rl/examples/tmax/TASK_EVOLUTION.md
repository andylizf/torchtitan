# Online task evolution: re-tune every no-signal group (0/k and k/k)

## The problem this closes

A GRPO group whose siblings **all pass** or **all fail** has zero reward
variance and trains nothing. Today `_maybe_annotate_zero_std` records those
prompts under `SWE_ZERO_STD_DIR`, and a later run points
`TMaxDataset.skip_ids_path` at that dir to **drop** them — they are never sampled
again. The pool only ever shrinks, and the prompts it sheds are exactly the ones
the current policy has outgrown (all-pass) or cannot yet touch (all-fail).

`drop` and `evolve` are two handlings of the **same** decision point. Instead of
shedding a no-signal prompt, move it to where the policy now is:

- **all fail** → make it easier: a hint drawn from a failing trajectory
- **all pass** → make it harder: one more constraint or stage

The prompt stays in the pool, re-tuned to the policy. This is the same
recursive-difficulty idea the seed pipeline uses offline; here it runs on the
training model's own rollouts.

## Why this lives in the rollouter, extending the existing zero-std path

`_maybe_emit_evolution_signal` sits next to the existing zero-std detector,
which is the right layer:

- it is the **existing no-signal detector** — it already computes `pstdev == 0`
  and has the `sample` (`instance_id`) and the `rollouts` (the trajectories);
- it already solves the **write mechanics** — one write-once file per prompt,
  the pattern that survives the oilfs FUSE mount and the pooled RolloutWorker
  processes (appending to a shared file does not);
- putting it here keeps the trainer doing **only observation**. Adjusting a task
  means rewriting its four files and re-validating them in a container (build,
  oracle, shortcut check) — minutes of Docker work that must not block a rollout,
  so it belongs on the data side, asynchronously. The trainer emits the signal;
  it never runs the factory.

## The change

A first-class step, `_maybe_emit_evolution_signal`, runs beside the existing
`_maybe_annotate_zero_std` after each group is scored — evolution is not a rider
on the drop path, it is its own action. It gated on `SWE_TASK_EVOLUTION_DIR`:
when set, **every** group the policy has moved past — all-fail (0/k) **and**
all-pass (k/k) — is written as a signal to be re-tuned. Both directions, always;
a group with any reward variance is already producing signal and is left alone.

The drop annotation (`SWE_ZERO_STD_DIR`) is untouched and independent — dropping
sheds a prompt, evolving re-tunes it, and a run enables either or both. The
signal:

```json
{"task_id": "...", "solved": 5, "total": 5, "direction": "harder",
 "attempts": [{"reward": 1.0, "turns": 7,
               "transcript": [{"cmd": "...", "out": "..."}]}]}
```

Two helpers (`_message_text`, `_evolution_signal`) build it. Nothing else moves.
Set neither var and the method returns exactly as before — the PR is inert until
a run opts in.

## The loop it completes

```
 trainer (this PR)                          data side (terminal-agent-rl)
 ─────────────────                          ────────────────────────────
 zero-std group (all-pass/all-fail)
   → SWE_TASK_EVOLUTION_DIR/<id>.json ────▶ feedback_loop reads the signals
                                              easier → simplify (hint from trajectory)
                                              harder → evolve  (one operator)
                                              re-validate: build, oracle, no-shortcut
   next round loads new data_path  ◀──────    pack_to_dataset folds it back
                                              (replace by instance_id, pool size fixed)
```

The signal is the same shape a solver rollout carries (task id, solve count,
per-attempt transcript), so the data side reads training rollouts and a
standalone GPT solve pass through one path — which is how the loop is
bootstrapped before the trainer's own rollouts exist.

## Enabling it, and keeping batch size fixed

Evolving is meant to replace a prompt in place, not remove it, so it pairs with
**not** dropping:

```bash
export SWE_DROP_ZERO_STD=0                          # keep zero-std groups in the
                                                    # batch (advantage 0, no
                                                    # gradient, but the batch size
                                                    # per step is unchanged)
export SWE_TASK_EVOLUTION_DIR=/path/on/shared/fs    # emit the evolution signal
# leave SWE_ZERO_STD_DIR unset                       # do not also build a drop list
```

`SWE_DROP_ZERO_STD=0` is the existing switch (`drop_zero_std_reward_groups=False`,
`skip_zero_advantage_samples=True`): the zero-std group stays in the batch and
simply contributes no gradient, so every step sees the same number of prompts.
Dropping would shrink the batch by exactly the prompts we mean to keep and re-tune.

The data side runs `feedback_loop --rollouts <signals>` over the dir (one JSON
per file; concatenate to a JSONL, or read the dir) and writes each adjusted task
back into `TMaxDataset.data_path` **at the same row** — a replace, not a delete —
so the pool size, and the batch, stay fixed while the prompts move to the policy.

Keeping `SWE_ZERO_STD_DIR` set as well is possible (drop as a fallback), but then
a prompt is both dropped and evolved; leave it unset while evolving.

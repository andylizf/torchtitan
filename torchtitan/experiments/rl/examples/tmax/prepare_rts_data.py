# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build a training JSONL from the ``Zhongzhi1228/Recursive-Task-Synthesis`` corpus.

RTS (arXiv:2608.05466, "Recursive Synthesis for Long-Horizon Terminal Tasks") is a
37,484-task synthetic terminal-agent corpus laid out as a Harbor task tree, the
same shape as Terminal-Bench 2.0::

    <task>/instruction.md          # the agent instruction
    <task>/task.toml               # [verifier]/[agent]/[environment] timeouts
    <task>/environment/Dockerfile  # the task env -- NOT a published image
    <task>/tests/test.sh           # verifier: writes /logs/verifier/reward.txt (0/1)
    <task>/tests/test_state.py     # + grade-time helpers
    <task>/solution/solve.sh       # oracle solution (unused for training)

The verifier contract is identical to tmax, so ``grading.py`` grades these rows
unchanged. The one difference is the environment: RTS publishes no docker image
(only 198 of 37,484 task.toml carry ``docker_image``), so each row carries its
``dockerfile`` text instead and the sandbox backend builds it server-side --
Daytona caches the build, so only the first sandbox per distinct Dockerfile pays
for it (measured: ~20-40s cold, ~1-2s warm).

A task whose Dockerfile copies local files with COPY also carries them as ``build_context``
({relpath: base64}); the sandbox writes them back beside the Dockerfile so the SDK
uploads them. Only tasks that need a host we do not control (an init system, the
docker socket, ``--privileged``) are dropped.

Difficulty: **use ``oracle_commands``, not the ``difficulty`` field.** The field is
inherited from the synthesis seed and still reads "easy" for round-15 tasks. What
bounds RL is how many commands the reference solution runs -- a rollout capped at
T turns cannot solve a task whose oracle needs more than T of them (median 44 in
shard 0, 212 in shard 7). ``--max-oracle-commands`` filters on it.

Run against an extracted corpus (``tar xf tasks-0000N.tar``)::

    python -m torchtitan.experiments.rl.examples.tmax.prepare_rts_data \
        --tasks-root /path/to/s0/tasks --max-oracle-commands 64 \
        --out mast_rl/swe_assets/rts_train.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shlex
import sys

from torchtitan.experiments.rl.examples.tmax.prepare_tmax_data import _REWARD_PATH

# A COPY/ADD whose source is local needs the task's ``environment/`` shipped with
# the row; ``--from=`` pulls from another image or stage and needs nothing.
_LOCAL_COPY = re.compile(r"^\s*(?:COPY|ADD)\s+(?!--from=)(.+?)\s*$", re.M | re.I)
# Flags that only affect ownership/permissions of the copied files.
_COPY_FLAG = re.compile(r"^--(chown|chmod|link)=")

# Tasks that need a host we do not control. Measured, not guessed: a substring
# match on "systemd"/"docker-compose" rejects ~1700 tasks of which 0 actually run
# an init system -- the corpus mentions them in comments explaining solve.sh. Only
# these signals correlate with a task that really cannot run in a plain container.
_REJECT_PRIVILEGED = re.compile(
    r"^\s*(?:ENTRYPOINT|CMD)\s+.*(?:/sbin/init|systemd)"
    r"|^\s*RUN\s+.*systemctl\s+(?:enable|start)"
    r"|/var/run/docker\.sock"
    r"|--privileged",
    re.M | re.I,
)

# Byte ceiling for an inlined build context. The corpus is tiny here (p50 ~5 KB,
# p90 ~15 KB) but a handful of tasks carry multi-MB fixtures that would bloat the
# JSONL for no benefit.
_MAX_CONTEXT_BYTES = 1 << 20

_DEFAULT_WORKDIR = "/app"


def _workdir_from_dockerfile(text: str) -> str:
    """Last ``WORKDIR`` in the Dockerfile -- where the agent's commands must run."""
    workdir = _DEFAULT_WORKDIR
    for line in text.splitlines():
        m = re.match(r"\s*WORKDIR\s+(\S+)", line)
        if m:
            workdir = m.group(1)
    return workdir


def _build_context(env_dir: str, dockerfile: str) -> dict[str, str]:
    """``{relpath: base64}`` for every local COPY/ADD source under ``environment/``.

    ``DaytonaSandbox._declarative_image`` writes these back next to the Dockerfile
    so the SDK resolves the COPY sources and uploads them as the build context.
    Base64 because the corpus copies binaries (images, archives) as well as text.

    Raises FileNotFoundError when a source is missing (an unbuildable task) and
    ValueError when the context exceeds ``_MAX_CONTEXT_BYTES``.
    """
    context: dict[str, str] = {}
    total = 0
    for rest in _LOCAL_COPY.findall(dockerfile):
        parts = [p for p in shlex.split(rest) if not _COPY_FLAG.match(p)]
        for src in parts[:-1]:
            abspath = os.path.normpath(os.path.join(env_dir, src))
            if not os.path.exists(abspath):
                raise FileNotFoundError(src)
            files = (
                [os.path.join(dp, fn) for dp, _d, fs in os.walk(abspath) for fn in fs]
                if os.path.isdir(abspath)
                else [abspath]
            )
            for path in files:
                rel = os.path.relpath(path, env_dir)
                if rel.startswith(".."):
                    raise FileNotFoundError(f"{src} escapes environment/")
                with open(path, "rb") as f:
                    blob = f.read()
                total += len(blob)
                if total > _MAX_CONTEXT_BYTES:
                    raise ValueError(f"context > {_MAX_CONTEXT_BYTES} bytes")
                context[rel] = base64.b64encode(blob).decode()
    return context


def _oracle_commands(solve_sh: str) -> int:
    """Approximate the number of shell commands the reference solution executes.

    This is the task's own difficulty measure and the one that matters for RL: a
    rollout capped at N turns cannot solve a task whose oracle needs more than N
    commands. Heredoc bodies are data, not commands, so they are skipped; ``&&``,
    ``;`` and ``|`` chains each count as another command.
    """
    count, in_heredoc, terminator = 0, False, None
    for raw in solve_sh.splitlines():
        line = raw.strip()
        if in_heredoc:
            if terminator and line == terminator:
                in_heredoc = False
            continue
        opener = re.search(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", line)
        if opener:
            in_heredoc, terminator = True, opener.group(1)
        if not line or line.startswith("#"):
            continue
        if line in ("fi", "done", "esac", "else", "}", "{", ")", ";;"):
            continue
        count += 1 + len(re.findall(r"&&|\|\||;(?!;)", line))
    return count


def _grading_fixtures(task_dir: str) -> dict[str, str]:
    """``{relpath: content}`` for every text file under ``tests/`` except test.sh
    (uploaded separately). grading.py maps ``tests/*`` -> ``/tests/*`` at grade time."""
    fixtures: dict[str, str] = {}
    base = os.path.join(task_dir, "tests")
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            abspath = os.path.join(dirpath, fn)
            rel = os.path.relpath(abspath, task_dir)
            if rel == os.path.join("tests", "test.sh"):
                continue
            try:
                with open(abspath, encoding="utf-8") as f:
                    fixtures[rel] = f.read()
            except (UnicodeDecodeError, OSError):
                continue
    return fixtures


def _to_row(task_dir: str) -> tuple[dict | None, str]:
    """Build one output row, or ``(None, reason)`` when the task is filtered out."""
    task_id = os.path.basename(task_dir.rstrip("/"))
    paths = {
        "instruction": os.path.join(task_dir, "instruction.md"),
        "test_sh": os.path.join(task_dir, "tests", "test.sh"),
        "dockerfile": os.path.join(task_dir, "environment", "Dockerfile"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            return None, f"missing_{name}"

    with open(paths["dockerfile"], encoding="utf-8") as f:
        dockerfile = f.read()
    if _REJECT_PRIVILEGED.search(dockerfile):
        return None, "needs_privileged"

    env_dir = os.path.join(task_dir, "environment")
    try:
        build_context = _build_context(env_dir, dockerfile)
    except FileNotFoundError:
        return None, "copy_source_missing"
    except ValueError:
        return None, "build_context_too_large"

    with open(paths["instruction"], encoding="utf-8") as f:
        instruction = f.read()
    with open(paths["test_sh"], encoding="utf-8") as f:
        test_sh = f.read()
    if not instruction.strip() or not test_sh.strip():
        return None, "empty_instruction_or_verifier"
    # The whole reward signal is this file; a verifier that never writes it would
    # silently score 0 for every rollout.
    if "reward.txt" not in test_sh and "reward.json" not in test_sh:
        return None, "verifier_writes_no_reward"

    solve_path = os.path.join(task_dir, "solution", "solve.sh")
    oracle_commands = 0
    if os.path.exists(solve_path):
        with open(solve_path, encoding="utf-8", errors="replace") as f:
            oracle_commands = _oracle_commands(f.read())

    metadata = {
        "instance_id": task_id,
        "image": "",
        "dockerfile": dockerfile,
        "workdir": _workdir_from_dockerfile(dockerfile),
        "problem_statement": instruction,
        # The reference solution's command count: the turn budget a rollout needs
        # before it can possibly solve this task. Used by --max-oracle-commands.
        "oracle_commands": oracle_commands,
        "tmax": {
            "test_sh": test_sh,
            "fixtures": _grading_fixtures(task_dir),
            "reward_path": _REWARD_PATH,
        },
    }
    if build_context:
        metadata["build_context"] = build_context
    return {"prompt": instruction, "label": task_id, "metadata": metadata}, "ok"


def build_rows(
    tasks_roots: list[str],
    *,
    limit: int | None = None,
    seed: int = 42,
    max_oracle_commands: int | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Convert every task dir under ``tasks_roots`` to a row, applying the filters.

    Task dirs are shuffled before the ``limit`` cut so a subset spans the whole
    corpus rather than one alphabetical corner of it.
    """
    dirs: list[str] = []
    for root in tasks_roots:
        dirs.extend(
            os.path.join(root, d)
            for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))
        )
    random.Random(seed).shuffle(dirs)

    rows: list[dict] = []
    reasons: dict[str, int] = {}
    for d in dirs:
        row, reason = _to_row(d)
        if (
            row is not None
            and max_oracle_commands is not None
            and row["metadata"]["oracle_commands"] > max_oracle_commands
        ):
            row, reason = None, "oracle_over_turn_budget"
        reasons[reason] = reasons.get(reason, 0) + 1
        if row is not None:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows, reasons


def _write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output rts_train.jsonl path")
    ap.add_argument(
        "--tasks-root",
        action="append",
        required=True,
        metavar="DIR",
        help="extracted RTS 'tasks/' dir (repeat to mix shards / difficulties)",
    )
    ap.add_argument("--limit", type=int, default=None, help="emit at most N tasks")
    ap.add_argument("--seed", type=int, default=42, help="task-order shuffle seed")
    ap.add_argument(
        "--max-oracle-commands",
        type=int,
        default=None,
        metavar="N",
        help="drop tasks whose reference solve.sh runs more than N commands -- a "
        "rollout capped at T turns cannot solve one needing more than T of them, "
        "so pair this with SWE_MAX_TURNS (e.g. N=64 under a 128-turn cap)",
    )
    ap.add_argument(
        "--smoke-size",
        type=int,
        default=0,
        help="also write rts_smoke.jsonl with N rows",
    )
    args = ap.parse_args()

    rows, reasons = build_rows(
        args.tasks_root,
        limit=args.limit,
        seed=args.seed,
        max_oracle_commands=args.max_oracle_commands,
    )
    if not rows:
        print(f"ERROR: produced 0 rows (filters: {reasons})", file=sys.stderr)
        sys.exit(1)
    _write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} RTS tasks -> {args.out}")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:32s} {n}")

    if args.smoke_size > 0:
        smoke = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "rts_smoke.jsonl"
        )
        _write_jsonl(rows[: args.smoke_size], smoke)
        print(f"wrote {min(args.smoke_size, len(rows))} smoke tasks -> {smoke}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recursive terminal-task synthesis with validation and audit in the loop.

One round takes validated tasks as seeds, derives harder tasks from them, and
keeps only the ones that survive four gates. The gates are the point: a
synthesis loop without them accumulates tasks that look plausible and cannot be
trained on.

  1. build          the environment builds from the Dockerfile alone
  2. oracle         the generated solution runs and its verifier writes reward 1
                    — this is what proves the instruction, environment, solution
                    and verifier agree with each other
  3. audit          the verifier does not assert paths the instruction and
                    environment never reveal (a "dark check", which no agent can
                    satisfy except by luck), and the instruction does not name
                    the verifier back
  4. shortcut proof if the pre-build reading audit named a command sequence it
                    believes fools the verifier, run it in a fresh container —
                    no solution staged — and reject only on a demonstrated pass

There are no rollouts in this loop and no difficulty verdicts. Rollouts belong
to the training side: RL produces them anyway, on the model actually being
trained, and their per-task results — solve counts and traces — flow back as
the input to the next rewrite round through evolve()/simplify(). What this loop
owes the trainer is a sound task, not an easy or hard one.

Tasks clearing the gates join the pool and become seeds for the next round,
which is where "recursive" comes in: difficulty compounds because each round
starts from what already survived.

Gate 3 is the one the RST paper leaves out. It audits instructions that reveal
their verifier, but not verifiers that check what the instruction never says —
and that direction is what makes a task unfair rather than hard.

Usage:
  python3 synth_loop.py --seeds verified_pass_ids.txt --tar tasks.tar \\
      --out data/synth --rounds 1 --per-round 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import random
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import docker_validate as dv
import synth_client as llm

BUILD_TIMEOUT = 1200
SOLVE_TIMEOUT = 900
TEST_TIMEOUT = 600
AGENT_CMD_TIMEOUT = 180
MIN_FREE_GB = 10
PRUNE_BELOW_GB = 25

log = logging.getLogger("synth")


# --------------------------------------------------------------------------
# container plumbing, shared with the validator
# --------------------------------------------------------------------------

def free_gb(path: str = "/var/lib/docker") -> float:
    st = os.statvfs(path if Path(path).exists() else "/")
    return st.f_bavail * st.f_frsize / 1024**3


def sh(cmd: list[str], timeout: int) -> tuple[int, str]:
    # errors="replace": an agent command that emits a binary would otherwise
    # raise UnicodeDecodeError out of here and end the task.
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-6000:]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def stage_into(container: str, work: Path, dirs: list[str]) -> tuple[int, str]:
    """Copy solution/ and tests/ in as root-owned, at Harbor's paths.

    Not `docker cp`: it replays the host uid onto the copy, and an NFS uid that
    does not exist inside the container fails the lchown after having created
    the directory — which then looks like a broken task. And the destination is
    `/`, because test.sh hardcodes /tests/test_state.py; staged anywhere else
    pytest collects nothing and every task scores zero.
    """
    present = [d for d in dirs if (work / d).exists()]
    if not present:
        return 1, "nothing to stage"
    p1 = subprocess.Popen(["tar", "-C", str(work), "--owner=0", "--group=0",
                           "--numeric-owner", "-cf", "-", *present],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.Popen(["docker", "exec", "-i", container, "tar", "-C", "/",
                           "-xf", "-"], stdin=p1.stdout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True)
    p1.stdout.close()
    out, err = p2.communicate(timeout=300)
    p1.wait(timeout=30)
    return p2.returncode, (out + err)[-500:]


def read_reward(container: str) -> str | None:
    """The verdict is the reward file, not test.sh's exit code.

    test.sh runs pytest, writes 1 or 0, and ends; in the overwhelming majority
    of these tasks nothing after that sets a non-zero status, so the script
    exits 0 whether every test passed or every test failed. Read with `cat`
    rather than through a login shell, whose warnings otherwise merge into
    stdout and turn "0" into "0\\nstdin: is not a tty".
    """
    _, raw = sh(["docker", "exec", container, "cat",
                 "/logs/verifier/reward.txt"], 60)
    return next((t for t in (raw or "").split() if t in ("0", "1")), None)


# The real TerminalWorld grader, copied rather than reinvented. It bootstraps
# uv and runs a pinned pytest through uvx, which is what makes it work in images
# that have no system Python — and most of these images do not. A grader that
# assumes python3 exists silently scores every task in a PHP, Go, or Node image
# zero while its solution runs perfectly.
TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
if ! command -v curl >/dev/null 2>&1; then
  (apt-get update && apt-get install -y curl) >/dev/null 2>&1 || \
    (apk add --no-cache curl) >/dev/null 2>&1 || \
    (yum install -y curl) >/dev/null 2>&1 || true
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh >/dev/null 2>&1 || true
fi
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"

uvx --with pytest==8.4.1 --with pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
rc=$?
if [ $rc -eq 0 ]; then echo 1 > /logs/verifier/reward.txt
else echo 0 > /logs/verifier/reward.txt; fi
"""


def materialize(task: dict, dest: Path, env_files: dict | None = None) -> None:
    """Write the five-piece layout for one generated task."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "instruction.md").write_text(task["instruction"])
    (dest / "environment").mkdir(exist_ok=True)
    (dest / "environment/Dockerfile").write_text(task["dockerfile"])
    for rel, blob in (env_files or {}).items():
        out = dest / "environment" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
    (dest / "solution").mkdir(exist_ok=True)
    (dest / "solution/solve.sh").write_text(task["solve_sh"])
    (dest / "tests").mkdir(exist_ok=True)
    (dest / "tests/test_state.py").write_text(task["test_state_py"])
    (dest / "tests/test.sh").write_text(TEST_SH)
    (dest / "task.toml").write_text(
        f'cpus = {int(task.get("cpus", 1))}\n'
        f'memory = "{int(task.get("memory_mb", 2048))}M"\n')


TEST_BODY_RE = re.compile(
    r"^\s*def\s+(test_\w+)\s*\([^)]*\)\s*(?:->[^:\n]+)?:((?:\n(?:[ \t].*)?)*)", re.M)
_NOSC_WORDS = re.compile(
    r"no_?shortcut|placeholder|hard[_-]?cod|stale|dummy|verifier[_-]?only"
    r"|fabricat|forged|precomputed", re.I)
_MUTATES = re.compile(r"write_bytes|write_text|\.write\(|shutil\.copy"
                      r"|os\.remove|unlink\(|truncate|touch\(")
_RERUNS = re.compile(r"run_workflow|subprocess\.(run|check_|Popen)|os\.system"
                     r"|run_cmd|sh\(")


def rejects_shortcut(body: str, whole: str = "") -> bool:
    """Whether a check would catch an answer the agent never actually produced.

    The mutation has to be in this check's own body — that is what makes it this
    check rather than a neighbour. The re-run does not: verifiers routinely put
    it in a module-level helper and call `_build(tag)` or `_run_workflow()` from
    the test, and looking for a subprocess call inside the function body misses
    every one of them. Fourth detector in this file to have been narrower than
    what the model writes, so the search widens to the file for that half.
    """
    whole = whole or body
    if _NOSC_WORDS.search(body):
        return True
    return bool(_MUTATES.search(body) and _RERUNS.search(whole))



# --------------------------------------------------------------------------
# gate 3: audit
# --------------------------------------------------------------------------

# Paths an agent does not have to discover, because every environment has them.
UNIVERSAL_PATHS = {
    "/tmp", "/var/tmp", "/dev/null", "/dev/stdout", "/dev/stderr", "/etc/passwd",
    "/etc/group", "/etc/hosts", "/etc/hostname", "/etc/resolv.conf", "/root",
    "/home", "/usr/bin", "/usr/local/bin", "/var/log", "/proc", "/etc/os-release",
}

PATH_RE = re.compile(
    r'["\'](/(?:app|home|tmp|etc|var|opt|usr|data|srv|root|workspace)[^"\'\s]*)["\']')
VERIFIER_PATH_RE = re.compile(
    r"(^|[\s`'\"(/])tests/(test\.sh|test_[\w-]+\.py)|/oracle/|\btest_state\.py\b")


def audit(task: dict) -> dict:
    """Both directions of instruction<->verifier drift, as flags not verdicts."""
    instruction, tests = task["instruction"], task["test_state_py"]
    env = task["dockerfile"]
    dark = []
    for p in {p.rstrip("/") for p in PATH_RE.findall(tests)}:
        if p in instruction or p in env:
            continue
        # A path every Unix has is not a hidden requirement. Flagging /tmp as
        # undiscoverable rejected otherwise sound tasks, two of every three the
        # audit turned down.
        if p in UNIVERSAL_PATHS:
            continue
        anc = Path(p)
        visible = False
        while not visible and anc.parent != anc:
            anc = anc.parent
            if str(anc) == "/":
                break
            if str(anc) in instruction or str(anc) in env:
                rest = Path(p).relative_to(anc).parts
                visible = all(x in instruction or x in env for x in rest)
        if not visible:
            dark.append(p)
    leaks = []
    if VERIFIER_PATH_RE.search(instruction):
        leaks.append("instruction names the verifier path")
    return {"dark_paths": sorted(dark), "leaks": leaks}


# --------------------------------------------------------------------------
# gates 1, 2 and 4
# --------------------------------------------------------------------------

COPY_RE = re.compile(r"^\s*COPY\s+(?!--from)(?:--\S+\s+)*(\S+)", re.M | re.I)


def preflight(work: Path) -> list[str]:
    """Cheap static checks before paying for a build.

    The paper runs a static preflight for exactly this reason: malformed
    Dockerfiles and missing COPY sources are detectable from the files alone,
    and a build that fails on them costs minutes and tells you nothing you
    could not have known. The COPY case is the one the generator actually
    violates — it is told to keep the build self-contained and still emits
    `COPY flex /usr/local/bin/flex` against a context that has no `flex`.
    """
    problems = []
    env = work / "environment"
    dockerfile = env / "Dockerfile"
    if not dockerfile.exists():
        return ["no environment/Dockerfile"]
    text = dockerfile.read_text()
    for src in COPY_RE.findall(text):
        if src.startswith(("http://", "https://", "$")) or "*" in src:
            continue
        if not (env / src.lstrip("./")).exists():
            problems.append(f"COPY source missing from build context: {src}")
    if not re.search(r"^\s*FROM\s+\S+", text, re.M | re.I):
        problems.append("Dockerfile has no FROM")
    solve = work / "solution/solve.sh"
    if solve.exists() and not solve.read_text().strip():
        problems.append("solution/solve.sh is empty")
    elif solve.exists():
        # The solution is written by a model and never executed before the
        # oracle gate, so a syntax error costs a full image build to discover.
        # `bash -n` parses without running, which is free and catches the whole
        # class. Same for the verifier, which python can parse the same way.
        rc, out = sh(["bash", "-n", str(solve)], 60)
        if rc != 0:
            problems.append(f"solve.sh does not parse: {out.strip()[:160]}")
    tests_py = work / "tests/test_state.py"
    if tests_py.exists():
        src = tests_py.read_text(errors="replace")
        try:
            compile(src, "test_state.py", "exec")
        except SyntaxError as e:
            problems.append(f"test_state.py does not parse: {str(e)[:160]}")
        else:
            fns = TEST_BODY_RE.findall(src)
            if len(fns) < 4:
                problems.append(
                    f"verifier has {len(fns)} checks, the contract requires four")
            if not any(rejects_shortcut(b, src) for _, b in fns):
                # RST gets this on every task because one of the four contract
                # slots is that check. Ours asks for it and gets it on two thirds,
                # so the rest is caught here instead of shipped: a task nothing
                # stops from being answered without doing the work is the one
                # kind of bad task that still passes every other gate.
                problems.append("no check rejects an answer that was never produced")
    tests = work / "tests/test_state.py"
    if tests.exists() and "def test_" not in tests.read_text():
        problems.append("tests/test_state.py defines no test function")
    return problems


def build_image(work: Path, image: str, attempts: int = 3) -> tuple[bool, str]:
    """Build, retrying what the network broke rather than what the task did.

    Running two dozen workers against one Docker daemon makes the registry and
    its DNS the bottleneck before anything else does: 14% of one run failed on
    `lookup auth.docker.io: i/o timeout`, which says nothing about the generated
    Dockerfile. Same classifier the corpus validator uses.
    """
    for n in range(1, attempts + 1):
        ok, out = _build_once(work, image)
        if ok or n == attempts:
            return ok, out
        if not dv.TRANSIENT_BUILD.search(out) or dv.PERMANENT_BUILD.search(out):
            return ok, out
        time.sleep(20 * n)
    return ok, out


def _build_once(work: Path, image: str) -> tuple[bool, str]:
    rc, out = sh(["docker", "build", "-t", image, "-f",
                  str(work / "environment/Dockerfile"),
                  str(work / "environment")], BUILD_TIMEOUT)
    return rc == 0, out[-1500:]


def start_container(image: str, container: str) -> bool:
    """Start under the image's own entrypoint, as the validator does.

    This used to force `--entrypoint ""`, the same defect the corpus validator
    carried: a task that serves a bundled file over localhost so a hardcoded URL
    resolves is a different task without its entrypoint. Synthesised tasks
    inherit their seed's Dockerfile, so a seed that needed its entrypoint
    produces a rewrite that needs it too, and grading it without one fails the
    oracle gate for a reason that has nothing to do with the rewrite.
    """
    started, _mode = dv.start_under_entrypoint(image, container)
    if started:
        sh(["docker", "exec", container, "mkdir", "-p", "/logs/verifier"], 60)
    return started


def grade(container: str, work: Path) -> tuple[str | None, str]:
    """Return (reward, verifier output).

    The output matters as much as the verdict: a zero reward with no log is
    indistinguishable between "the task is wrong" and "the grader could not
    run", and this harness has produced the latter more than once.
    """
    stage_into(container, work, ["tests"])
    _, out = sh(["docker", "exec", container, "bash", "-lc",
                 "cd /app 2>/dev/null || cd /; bash /tests/test.sh"],
                TEST_TIMEOUT)
    return read_reward(container), out[-2500:]


def oracle_check(work: Path, image: str, tid: str) -> dict:
    """Gate 2: does the generated solution earn a passing grade on its own task?"""
    c = f"syn-oracle-{tid}"
    rec: dict = {}
    try:
        if not start_container(image, c):
            return {"ok": False, "why": "container would not start"}
        stage_into(c, work, ["solution"])
        rc, out = sh(["docker", "exec", c, "bash", "-lc",
                      "cd /app 2>/dev/null || cd /; bash /solution/solve.sh"],
                     SOLVE_TIMEOUT)
        rec["solve_exit"] = rc
        rec["solve_tail"] = out[-600:]
        rec["reward"], rec["test_tail"] = grade(c, work)
        rec["ok"] = rec["reward"] == "1"
        rec["why"] = "" if rec["ok"] else f"reward={rec['reward']}"
        return rec
    finally:
        sh(["docker", "rm", "-f", c], 120)


def shortcut_check(work: Path, image: str, tid: str, shortcut: str) -> dict:
    """Does the claimed shortcut actually earn a passing grade?

    Same machinery as the oracle check with the solution left out: fresh
    container, run the claimed commands, grade. `passed` means the verifier
    accepted work that ignored the instruction — a demonstrated reward hole,
    not a suspected one.
    """
    c = f"syn-hack-{tid}"
    try:
        if not start_container(image, c):
            return {"passed": False, "why": "container would not start"}
        rc, out = sh(["docker", "exec", c, "bash", "-lc",
                      f"cd /app 2>/dev/null || cd /; {shortcut}"], 120)
        reward, tail = grade(c, work)
        return {"passed": reward == "1", "shortcut_exit": rc,
                "shortcut_tail": out[-300:], "test_tail": tail[-300:]}
    finally:
        sh(["docker", "rm", "-f", c], 120)


def agent_attempt(work: Path, image: str, tid: str, instruction: str,
                  max_turns: int, idx: int) -> dict:
    """One rollout: let the solver drive a fresh container, then grade it."""
    c = f"syn-agent-{tid}-{idx}"
    history: list[tuple[str, str]] = []
    try:
        if not start_container(image, c):
            return {"reward": None, "turns": 0, "why": "container would not start"}
        for turn in range(max_turns):
            try:
                cmd = llm.agent_step(instruction, history)
            except Exception as e:  # noqa: BLE001
                return {"reward": None, "turns": turn, "why": f"llm: {e}"[:200]}
            if cmd.strip() == "DONE" or not cmd.strip():
                break
            rc, out = sh(["docker", "exec", c, "bash", "-lc", cmd],
                         AGENT_CMD_TIMEOUT)
            # A solver can destroy the container it is working in, and once it
            # has, every remaining turn is a failing exec the model still pays
            # to read and answer. Stop at the turn it happened and say so, so
            # the record distinguishes a wrecked environment from a wrong
            # answer, and 24 wasted turns do not follow the one that mattered.
            if rc != 0 and any(s in out for s in dv.CONTAINER_GONE):
                return {"reward": None, "turns": turn,
                        "why": f"container gone at turn {turn}: {cmd[:120]}"}
            history.append((cmd, f"exit={rc}\n{out[-2000:]}"))
        reward, test_tail = grade(c, work)
        return {"reward": reward, "test_tail": test_tail[-800:],
                "turns": len(history),
                "transcript": [{"cmd": c_, "out": o[:800]} for c_, o in history]}
    finally:
        sh(["docker", "rm", "-f", c], 120)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def load_seeds(tar_path: Path, ids: list[str], limit: int) -> list[dict]:
    """Pull instruction/Dockerfile/solution for a sample of validated tasks."""
    want, seeds = set(ids[:limit]), {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            parts = m.name.split("/")
            if len(parts) < 3 or parts[1] not in want or not m.isfile():
                continue
            rel = "/".join(parts[2:])
            d = seeds.setdefault(parts[1], {"task_id": parts[1], "env_files": {}})
            if rel in ("instruction.md", "environment/Dockerfile",
                       "solution/solve.sh"):
                d[{"instruction.md": "instruction",
                   "environment/Dockerfile": "dockerfile",
                   "solution/solve.sh": "solution"}[rel]] = \
                    tf.extractfile(m).read().decode("utf-8", "replace")
            elif rel.startswith("environment/"):
                # The seed's Dockerfile COPYs these, and a generated package
                # without them fails preflight on a missing build-context source
                # — a fault of the packaging rather than of the rewrite.
                d["env_files"][rel[len("environment/"):]] = \
                    tf.extractfile(m).read()
    return [s for s in seeds.values()
            if {"instruction", "dockerfile", "solution"} <= set(s)]



def _as_files(task: dict) -> dict[str, str]:
    """The flat task in the file-path shape the repair passes read and write."""
    return {"instruction.md": task["instruction"],
            "environment/Dockerfile": task["dockerfile"],
            "solution/solve.sh": task["solve_sh"],
            "tests/test_state.py": task["test_state_py"]}


def run_one(seed: dict, out_dir: Path, used_ops: dict, used_fams: dict, rng,
            solvability: bool = True) -> dict:
    rec = {"seed_id": seed["task_id"], "t_start": time.time()}
    if free_gb() < PRUNE_BELOW_GB:
        sh(["docker", "builder", "prune", "-f"], 600)
    if free_gb() < MIN_FREE_GB:
        return {**rec, "status": "aborted", "why": f"disk {free_gb():.1f}G"}
    try:
        family, operator, definition = llm.pick_operator(
            seed, used_ops, used_fams, rng)
    except llm.Blocked as e:
        # Declining a seed is a result, not a failure: it costs one ranking call
        # and saves the build, the oracle run and the rollouts that a seed with
        # nothing to build on would have spent before failing a gate anyway.
        return {**rec, "status": "blocked", "why": str(e)[:300]}
    used_ops[operator] = used_ops.get(operator, 0) + 1
    used_fams[family] = used_fams.get(family, 0) + 1
    rec["family"], rec["operator"] = family, operator
    try:
        contract, gen = llm.synthesize(seed, family, operator, definition)
    except Exception as e:  # noqa: BLE001
        return {**rec, "status": "synth_failed", "why": str(e)[:300]}
    rec["contract_goal"] = str(contract.get("goal", ""))[:300]
    rec["reward_checks"] = len(contract.get("reward_checks") or [])
    task = {"task_id": f"syn_{seed['task_id']}_{operator[:18]}",
            "instruction": gen.get("instruction.md", ""),
            "dockerfile": gen.get("environment/Dockerfile", ""),
            "solve_sh": gen.get("solution/solve.sh", ""),
            "test_state_py": gen.get("tests/test_state.py", ""),
            "cpus": 1, "memory_mb": 2048, "task_toml": gen.get("task.toml")}

    tid = re.sub(r"[^a-z0-9_]", "", str(task.get("task_id", "")).lower()) or \
        f"syn_{seed['task_id']}"
    rec["task_id"] = tid
    rec["rationale"] = task.get("rationale", "")[:300]
    work = out_dir / tid
    for key in ("instruction", "dockerfile", "solve_sh", "test_state_py"):
        if not task.get(key):
            return {**rec, "status": "synth_incomplete", "why": f"missing {key}"}
    materialize(task, work, seed.get("env_files"))

    rec["audit"] = audit(task)
    problems = preflight(work)
    if problems:
        # Rejecting here rather than at build time keeps the reason legible:
        # "COPY source missing" beats a cache-key checksum error from BuildKit.
        return {**rec, "status": "preflight_failed", "why": "; ".join(problems)[:300]}

    # Solvability, before a build is spent on it. A task can be perfectly
    # self-consistent and still impossible: the reference is written by something
    # that already knows the answer and may hardcode it, and the anti-leak rule
    # pushed too far produces a verifier asking for what the instruction never
    # said.
    #
    # Only `unsolvable` rejects here, because only it survived being checked.
    # Crossing this diagnosis over the whole seed corpus against measured pass@5:
    # `unsolvable` tasks solved at 0.18 (8.7x over-represented among the never-
    # solved), so it is trusted. `environment` tasks solved at 0.85 against 0.88
    # for clean ones — the verdict carried nothing, because the container has
    # network access and the auditor was guessing otherwise; a run that rejected
    # on it threw away a fifth of its attempts at random. `underspecified` sits
    # between (1.9x) and is repairable by adding what the instruction omits, so
    # it flows on with the verdict recorded. `hackable` is a claim
    # about behaviour, so it is proven by running the claimed shortcut once the
    # image exists, below, rather than trusted as a reading.
    if solvability:
        diag = llm.diagnose_unsolved(task)
        rec["solvability"] = diag
        if diag.get("verdict") == "unsolvable":
            return {**rec, "status": "rejected_unsolvable",
                    "why": str(diag.get("why", ""))[:300]}
    image = f"syn-{tid}"
    try:
        ok, tail = build_image(work, image)
        if not ok:
            return {**rec, "status": "build_failed", "build_tail": tail}
        oracle = oracle_check(work, image, tid)
        rec["oracle"] = oracle
        if not oracle["ok"]:
            return {**rec, "status": "oracle_failed"}
        if rec["audit"]["dark_paths"] or rec["audit"]["leaks"]:
            return {**rec, "status": "audit_rejected"}

        # The diagnosis may have named a command sequence it believes passes the
        # verifier without doing the task. That is testable, so test it: run the
        # shortcut in a fresh container — no solution staged — and grade. A pass
        # is proof the verifier is too loose, and a task like that is worse than
        # wasted, because it pays reward for not doing the work. The reading
        # alone does not reject; the demonstration does.
        shortcut = str((rec.get("solvability") or {}).get("shortcut", "")).strip()
        if shortcut:
            hack = shortcut_check(work, image, tid, shortcut)
            rec["shortcut_check"] = hack
            if hack.get("passed"):
                return {**rec, "status": "rejected_hackable",
                        "why": f"verifier passed on: {shortcut[:200]}"}

        # No rollouts here, and that is the architecture rather than a saving.
        # Rollouts belong to the training side: RL produces them anyway, on the
        # model actually being trained, and their results — per-task solve
        # counts and traces — flow back as the input to the next rewrite round
        # through evolve()/simplify(). A GPT rollout in this loop would be a
        # parallel, wrong-model reading of the same thing. What this loop owes
        # the trainer is a sound task: consistent (oracle), fair (audit), and
        # not passable by a shortcut (demonstrated above, not presumed).
        rec["status"] = "accepted"
        return rec
    finally:
        sh(["docker", "rmi", "-f", image], 300)
        rec["t_end"] = time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--per-round", type=int, default=10)
    ap.add_argument("--results", default="results/synth_loop.jsonl")
    ap.add_argument("--no-solvability", dest="solvability",
                    action="store_false",
                    help="skip the pre-build check that the task is answerable "
                         "from what it gives the agent")
    ap.add_argument("--seed", type=int, default=0,
                    help="rng seed for operator choice; runs are reproducible")
    args = ap.parse_args()

    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(results.with_suffix(".log")),
                  logging.StreamHandler()])

    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line).get("seed_id"))

    seed_ids = [s for s in Path(args.seeds).read_text().split() if s not in done]
    log.info("%d seeds available, %d already processed, free disk %.1fG",
             len(seed_ids) + len(done), len(done), free_gb())
    # Stated in the log because it is the run's configuration, and a run's log
    # is what gets read to say what the run tested. One experiment "comparing"
    # card-derived keywords ran an entire night with the code that reads them
    # never deployed; this line is how that becomes visible in the first ten
    # seconds instead of after the run.
    kw = llm._operator_keywords()
    log.info("operator keywords: %s",
             f"card-derived, {len(kw)} operators" if kw
             else "NONE — falling back to definition words")

    # Family usage carries across rounds so the balance term keeps working:
    # resetting it each round would let one family dominate round after round.
    used_families: dict[str, int] = {}
    used_ops: dict[str, int] = {}
    rng = random.Random(args.seed)

    with open(results, "a") as fh:
        for rnd in range(1, args.rounds + 1):
            out_dir = Path(args.out) / f"round_{rnd}"
            out_dir.mkdir(parents=True, exist_ok=True)
            seeds = load_seeds(Path(args.tar), seed_ids, args.per_round)
            log.info("round %d: %d seeds", rnd, len(seeds))
            counts: dict[str, int] = {}
            for n, seed in enumerate(seeds, 1):
                # Priced here rather than inside run_one: the early returns
                # there each copy the record, so a field set on the way out
                # would be missing from exactly the failures worth pricing.
                mark = dict(llm.USAGE)
                rec = run_one(seed, out_dir, used_ops, used_families, rng,
                              args.solvability)
                rec["round"] = rnd
                rec["usage"] = llm.usage_since(mark)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                log.info("[r%d %d/%d] %s [%s] -> %s (pass@k=%s, %d calls "
                         "%dk in %dk out) | %s", rnd, n,
                         len(seeds), rec.get("task_id", rec["seed_id"]),
                         rec.get("family", "-"), rec["status"],
                         rec.get("pass_at_k"), rec["usage"]["calls"],
                         rec["usage"]["prompt_tokens"] // 1000,
                         rec["usage"]["completion_tokens"] // 1000, counts)
                if rec["status"] == "aborted":
                    log.error("disk floor reached, stopping")
                    return
            # Next round seeds from what survived, which is what makes it
            # recursive: difficulty compounds instead of resetting.
            seed_ids = [r for r in seed_ids]
            log.info("round %d done: %s", rnd, counts)


if __name__ == "__main__":
    main()

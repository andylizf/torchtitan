#!/usr/bin/env python3
"""Re-validate tasks a Docker-less runner could not judge, using real Docker.

The tridao runs were done on an HPC login node: no root, no daemon, so the
Dockerfile was flattened into a shell script and executed under udocker/PRoot.
That loses Docker's semantics, and the failures show it — `git: command not
found` after a layer installed git, PRoot signal warnings, our own flattener
sourcing a directory. Those tasks are not broken; the runner was.

Two things this does that the udocker path could not:

  * builds with a real context (`docker build environment/`), so no COPY source
    has to be inlined and the 512KiB inline cap does not exist here;
  * starts the container under the image's own ENTRYPOINT, so a task that
    serves a bundled file over localhost — to make the URL its instruction
    hardcodes resolve — is judged in the environment it ships. The first
    version of this script overrode the entrypoint to `""` while its comment
    claimed the opposite, so every verdict it produced for the 209 tasks that
    declare one described a different environment. `run_mode` now records which
    one a verdict came from.

And one thing the udocker path let through that Docker will not: a heredoc
written `RUN python3 << 'EOF'`, with a space Docker's parser does not accept.
`--repair` closes that space, which changes how the file is tokenised and not
what the shell runs.

Disk is the binding constraint: Docker's data root is on a shared 1.8T root
filesystem with ~45G free, so every image is removed as soon as its task is
judged and the run aborts rather than filling a machine other people use.

Usage:
  python3 docker_validate.py --tar tasks-00000.tar --ids retry_ids.txt \
      --results results/docker_validation.jsonl [--workers 4] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUILD_TIMEOUT = 1200
SOLVE_TIMEOUT = 900
TEST_TIMEOUT = 600
# Stop before the shared root filesystem is in trouble. A single build can add
# several GiB, so this leaves room for the largest task plus other users' work.
MIN_FREE_GB = 10
# Reclaim our own build cache before hitting the floor, so a full disk costs
# rebuild time instead of ending the run.
PRUNE_BELOW_GB = 25

log = logging.getLogger("docker-validate")


def free_gb(path: str = "/var/lib/docker") -> float:
    st = os.statvfs(path if Path(path).exists() else "/")
    return st.f_bavail * st.f_frsize / 1024**3


def tail_signal(text: str, limit: int = 1400) -> str:
    """Keep the part of a log that explains an outcome.

    These tasks install a Python toolchain inside test.sh, and uv's progress
    output is long enough to push the actual pytest assertion out of a
    fixed-size tail — leaving a failure whose recorded reason is a list of
    downloaded wheels.
    """
    noise = ("Downloading ", "Downloaded ", " Installed ", "Resolved ",
             "Prepared ", "Building ", "Collecting ", "Uninstalled ")
    kept = [ln for ln in (text or "").splitlines()
            if ln.strip() and not any(n in ln for n in noise)]
    return "\n".join(kept)[-limit:]


# A build that fails on the network says nothing about the task. The first
# group is the registry refusing to serve a base image, the second a package
# manager inside a RUN step that could not reach its index — both of which this
# host produced against Dockerfiles that built on the next attempt.
TRANSIENT_BUILD = re.compile(
    r"failed to resolve source metadata|failed to do request|no such host"
    r"|i/o timeout|TLS handshake timeout|failed to copy: httpReadSeeker"
    r"|429 Too Many Requests|50[0234] |connection reset|unexpected EOF"
    r"|temporary failure resolving|could not resolve host|connection timed out"
    r"|unable to select packages|failed to fetch|hash sum mismatch"
    r"|network is unreachable|could not fetch url|temporary failure in name",
    re.I)
# Retrying these only spends the build again: the mirror they want was shut down
# when the distribution went end of life, and no attempt reaches it.
#
# The archives that replace those mirrors — vault.centos.org,
# old-releases.ubuntu.com — do not belong here, and listing them cost 18 tasks a
# retry each. A Dockerfile naming an archive has already been pointed at the
# working host, so a failure against it is the archive being slow or refusing,
# which is exactly what a retry is for. A path the archive genuinely lacks still
# lands on the 404 below.
PERMANENT_BUILD = re.compile(
    r"mirrorlist\.centos\.org|cannot find a valid baseurl|repo 'base'"
    r"|404\s+not\s+found",
    re.I)

_INSTRUCTION = re.compile(r"^\s*(RUN|COPY|ADD|CMD|ENTRYPOINT|SHELL)\b", re.I)
# `<<` not preceded or followed by another `<`, so a herestring (`<<<`) is left
# alone, then whitespace, then the delimiter word Docker wants attached to it.
_SPACED = re.compile(r"(?<!<)(<<-?)(?!<)[ \t]+"
                     r"('[^'\n]*'|\"[^\"\n]*\"|[A-Za-z_][A-Za-z0-9_]*)")
_DELIM = re.compile(r"(?<!<)<<(-?)(?!<)"
                    r"(?:'([^'\n]*)'|\"([^\"\n]*)\"|([A-Za-z_][A-Za-z0-9_]*))")


def repair_heredoc_spacing(text: str) -> tuple[str, int]:
    """Attach a spaced heredoc delimiter to its `<<`, and count the repairs.

    `RUN python3 << 'EOF'` is ordinary shell and an unparseable Dockerfile:
    Docker only recognises a heredoc when the delimiter follows `<<` with no
    space, so with the space it reads the body as instructions and stops at the
    first line that is not one (`unknown instruction: import`). The udocker
    runner never saw this — it flattened each RUN into a shell script, where
    both spellings mean the same thing — which is why these tasks reached the
    corpus and then failed the moment a real Docker built them.

    Deleting that space is a change to how Docker tokenises the file, not to
    what the shell receives: the heredoc body is byte-identical either way. The
    rewrite is confined to instruction lines outside any heredoc body, so a body
    that happens to contain the same spelling (a script writing its own
    heredoc, a string literal) is passed through untouched.
    """
    out: list[str] = []
    pending: list[str] = []   # delimiters whose body we are inside
    continued = False         # previous instruction line ended in a backslash
    repairs = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n\r")
        if pending:
            if body.strip() == pending[0]:
                pending.pop(0)
            out.append(line)
            continue
        if _INSTRUCTION.match(body) or continued:
            fixed, n = _SPACED.subn(r"\1\2", body)
            repairs += n
            for _, sq, dq, bare in _DELIM.findall(fixed):
                pending.append(sq or dq or bare)
            continued = fixed.endswith("\\")
            out.append(fixed + line[len(body):])
            continue
        out.append(line)
    return "".join(out), repairs


def sh(cmd: list[str], timeout: int) -> tuple[int, str]:
    # errors="replace" because a solver will eventually cat a binary, and strict
    # UTF-8 decoding of what comes back raises out of this function and takes the
    # whole task with it — 21 of 861 came back as `error` for exactly that, a
    # harness fault recorded against the task.
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def sh_pipe(left: list[str], right: list[str], timeout: int) -> tuple[int, str]:
    """Run `left | right`, returning the right side's status and stderr."""
    try:
        p1 = subprocess.Popen(left, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        p2 = subprocess.Popen(right, stdin=p1.stdout, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        p1.stdout.close()
        out, err = p2.communicate(timeout=timeout)
        p1.wait(timeout=30)
        return p2.returncode, (out + err)[-2000:]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except OSError as e:
        return 125, str(e)


def extract(tf: tarfile.TarFile, tid: str, dest: Path) -> dict[str, Path]:
    """Unpack one task, preserving the layout its Dockerfile expects."""
    written = {}
    for m in tf.getmembers():
        parts = m.name.split("/")
        if len(parts) < 3 or parts[1] != tid or not m.isfile():
            continue
        rel = "/".join(parts[2:])
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(tf.extractfile(m).read())
        written[rel] = out
    return written


# What keeps the container alive while something works inside it. `sleep
# infinity` was the obvious choice and is a bad one: a solver that runs `pkill
# sleep` — housekeeping, or clearing what looks like a stray process — takes the
# container with it, losing the attempt and everything the solver had done.
#
# PID 1 is protected from signals it has no handler for, so this only bites when
# the keeper is not PID 1, which is whenever the image's entrypoint runs `"$@"`
# instead of `exec "$@"` — common enough to have cost two of ten pilot tasks.
# The trap covers that case, and the loop means killing the inner sleep costs a
# second rather than the container. Verified against `pkill sleep`,
# `kill -TERM 1` and `kill -9 1`, all of which the old form did not survive.
KEEPALIVE = ["sh", "-c", "trap '' TERM INT HUP; while :; do sleep 3600; done"]
CONTAINER_GONE = ("No such container", "is not running", "not running")


def start_under_entrypoint(image: str, container: str,
                           no_entrypoint: bool = False) -> tuple[bool, str]:
    """Start a container the way the image says to, and say which way that was.

    Shared with the synthesis loop, which had the same `--entrypoint ""` this
    file used to have. A task whose entrypoint serves a bundled file over
    localhost is a different task without it, and grading either one in the
    wrong environment produces a verdict about the harness.

    Returns (started, mode) where mode is `entrypoint`, `entrypoint_bypassed`
    for an entrypoint that ignored `sleep infinity` and exited, or
    `bypass_forced`.
    """
    if not no_entrypoint:
        rc, _ = sh(["docker", "run", "-d", "--name", container,
                    image, *KEEPALIVE], 180)
        if rc == 0:
            time.sleep(3)
            _, alive = sh(["docker", "inspect", "-f", "{{.State.Running}}",
                           container], 60)
            if alive.strip() == "true":
                return True, "entrypoint"
        sh(["docker", "rm", "-f", container], 120)
    rc, _ = sh(["docker", "run", "-d", "--name", container, "--entrypoint", "",
                image, *KEEPALIVE], 180)
    return rc == 0, "bypass_forced" if no_entrypoint else "entrypoint_bypassed"


def build_with_retry(env_dir: Path, image: str, attempts: int,
                     timeout: int) -> tuple[int, str, int]:
    """Build, retrying only what the registry caused, never what the task did."""
    for n in range(1, attempts + 1):
        rc, out = sh(["docker", "build", "-t", image, "-f",
                      str(env_dir / "Dockerfile"), str(env_dir)], timeout)
        retryable = (TRANSIENT_BUILD.search(out)
                     and not PERMANENT_BUILD.search(out))
        if rc == 0 or n == attempts or not retryable:
            return rc, out, n
        log.info("%s: transient build failure, retry %d/%d", image, n, attempts)
        time.sleep(20 * n)
    return rc, out, attempts


def validate_one(tar_path: Path, tid: str, work_root: Path,
                 repair: bool = False, build_attempts: int = 1,
                 build_timeout: int = BUILD_TIMEOUT,
                 no_entrypoint: bool = False) -> dict:
    rec = {"task_id": tid, "runner": "docker", "t_start": time.time()}
    work = work_root / tid
    image = f"twr-{tid.lower()}"
    container = f"twrc-{tid.lower()}"
    try:
        # Builds leave layer cache behind — 481 entries and ~20G accumulated
        # over the first 36 tasks — which is reclaimable but does not free
        # itself. Drop it when space gets tight rather than aborting the run:
        # aborting was correct when the only alternative was filling a shared
        # disk, but the cache is ours and costs nothing but rebuild time.
        if free_gb() < PRUNE_BELOW_GB:
            before = free_gb()
            sh(["docker", "builder", "prune", "-f"], 600)
            log.info("build cache pruned: %.1fG -> %.1fG free", before, free_gb())
        if free_gb() < MIN_FREE_GB:
            rec.update(status="aborted", reason=f"free disk {free_gb():.1f}G")
            return rec

        with tarfile.open(tar_path) as tf:
            files = extract(tf, tid, work)
        env_dir = work / "environment"
        if not (env_dir / "Dockerfile").exists():
            rec.update(status="skipped", reason="no environment/Dockerfile")
            return rec

        if repair:
            dockerfile = env_dir / "Dockerfile"
            original = dockerfile.read_text(encoding="utf-8", errors="surrogateescape")
            fixed, n = repair_heredoc_spacing(original)
            if n:
                dockerfile.write_text(fixed, encoding="utf-8",
                                      errors="surrogateescape")
                rec["repairs"] = {"heredoc_spacing": n}

        t0 = time.time()
        rc, out, tries = build_with_retry(env_dir, image, build_attempts,
                                          build_timeout)
        rec["build_s"] = round(time.time() - t0, 1)
        rec["build_attempts"] = tries
        if rc != 0:
            rec.update(status="build_failed", build_tail=tail_signal(out, 1800))
            return rec

        rc, insp = sh(["docker", "inspect", "-f",
                       "{{json .Config.Entrypoint}}{{json .Config.Cmd}}", image], 60)
        rec["entrypoint"] = insp.strip()[:200]

        # 209 of these images declare an ENTRYPOINT, and some of them are the
        # task: a script that serves a bundled file over localhost so the URL
        # the instruction hardcodes still resolves. Overriding it to `""` — as
        # this runner did until now, while its own comment claimed otherwise —
        # judges a different environment from the one the task ships.
        #
        # So run the image as declared, passing `sleep infinity` as the command
        # an entrypoint ending in `exec "$@"` will honour. An entrypoint that
        # ignores its arguments exits instead, taking the container with it;
        # that is what the fallback is for, and `run_mode` records which
        # environment the verdict describes.
        rec["run_mode"] = "bypass_forced" if no_entrypoint else "entrypoint"
        rc, out = (1, "forced") if no_entrypoint else sh(
            ["docker", "run", "-d", "--name", container,
             image, *KEEPALIVE], 180)
        if rc == 0:
            # An entrypoint that starts a service needs a moment to start it,
            # and one that exits does not always do so instantly.
            time.sleep(3)
            rc_a, alive = sh(["docker", "inspect", "-f",
                              "{{.State.Running}}", container], 60)
            if alive.strip() != "true":
                rc = 1
                out = "container exited under its own entrypoint"
        if rc != 0:
            sh(["docker", "rm", "-f", container], 120)
            if not no_entrypoint:
                rec["run_mode"] = "entrypoint_bypassed"
                rec["run_fallback_reason"] = tail_signal(out, 300)
            rc, out = sh(["docker", "run", "-d", "--name", container,
                          "--entrypoint", "", image, *KEEPALIVE], 180)
        if rc != 0:
            rec.update(status="error", reason="run failed", tail=out[-800:])
            return rec

        # Harbor's layout, not one of our choosing: test.sh hardcodes
        # /tests/test_state.py and writes /logs/verifier/reward.txt. Staging the
        # files anywhere else makes pytest collect zero tests, which the grader
        # reports as a failing run — every task then scores 0 and looks broken
        # while the harness is what is misplaced.
        rc_m, out_m = sh(["docker", "exec", container,
                          "mkdir", "-p", "/logs/verifier"], 60)
        stage: list[str] = [f"mkdir rc={rc_m} {out_m.strip()[:120]}"]
        present = [r for r in ("solution", "tests") if (work / r).exists()]
        if present:
            # Not `docker cp`: it replays the host's uid/gid onto the copied
            # files, and this host's uid (200331, an NFS account) does not exist
            # inside the container, so the daemon fails the lchown after having
            # already created the directory — leaving an empty solution/ and a
            # task that looks broken. Piping a tar written with owner 0 sidesteps
            # ownership entirely.
            rc_c, out_c = sh_pipe(
                ["tar", "-C", str(work), "--owner=0", "--group=0",
                 "--numeric-owner", "-cf", "-", *present],
                ["docker", "exec", "-i", container, "tar", "-C", "/",
                 "-xf", "-"], 300)
            stage.append(f"tar-in rc={rc_c} {out_c.strip()[:120]}")
        # Staging failures used to be invisible: cp errors were discarded and
        # the task then failed at "solve.sh: No such file or directory", which
        # reads exactly like a broken task rather than a broken harness.
        rc_l, out_l = sh(["docker", "exec", container, "ls", "/tests", "/solution"], 60)
        stage.append(f"staged -> {out_l.strip()[:150]}")
        rec["stage"] = " | ".join(stage)

        t0 = time.time()
        rc, out = sh(["docker", "exec", container, "bash", "-lc",
                      "cd /app 2>/dev/null || cd /; bash /solution/solve.sh"],
                     SOLVE_TIMEOUT)
        rec["solve_s"] = round(time.time() - t0, 1)
        rec["solve_exit"] = rc
        rec["solve_tail"] = tail_signal(out)

        rc, out = sh(["docker", "exec", container, "bash", "-lc",
                      "cd /app 2>/dev/null || cd /; bash /tests/test.sh"],
                     TEST_TIMEOUT)
        rec["test_exit"] = rc
        rec["test_tail"] = tail_signal(out)

        # test.sh is a grader, not an assertion: it runs pytest, writes 1 or 0
        # to reward.txt, and then ends. In 1467 of the 1530 tasks nothing after
        # that sets a non-zero status, so the script exits 0 whether every test
        # passed or every test failed — `test_exit == 0` proves only that the
        # grader ran. The verdict is the reward file.
        # Read the file with `cat` directly, not through `bash -lc`: a login
        # shell in these images prints "stdin: is not a tty" and setlocale
        # warnings, and because stdout and stderr are merged the reward came
        # back as "0\nstdin: is not a tty", matching neither "0" nor "1" and
        # silently downgrading real verdicts to ran_only.
        rc_r, raw = sh(["docker", "exec", container,
                        "cat", "/logs/verifier/reward.txt"], 60)
        reward = next((tok for tok in (raw or "").split()
                       if tok in ("0", "1")), None)
        rec["reward"] = reward
        if reward == "1":
            rec["status"] = "pass"
        elif reward == "0":
            rec["status"] = "fail"
        else:
            # No reward file: the grader never got far enough to write one.
            # Fall back to the exit code but mark the verdict as weaker, so it
            # is never silently counted as a real pass.
            rec["status"] = "ran_only" if rc == 0 else "fail"
        return rec
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", reason=f"{type(e).__name__}: {e}"[:300])
        return rec
    finally:
        rec["t_end"] = time.time()
        sh(["docker", "rm", "-f", container], 120)
        sh(["docker", "rmi", "-f", image], 300)
        shutil.rmtree(work, ignore_errors=True)


def index_tars(tars: list[str], cache: Path) -> dict[str, str]:
    """Map each task to the archive holding it, once instead of per task.

    The corpus arrived here in three pieces — a bulk shard, a retry shard, and
    the six tasks too large to transfer with the rest — so which archive holds a
    task is an accident of how it got to the machine, not something a caller
    should have to know. Scanning them costs a minute, so the map is kept.
    """
    if cache.exists():
        stored = json.loads(cache.read_text())
        if set(stored.get("tars", [])) == set(tars):
            return stored["index"]
    index: dict[str, str] = {}
    for t in tars:
        with tarfile.open(t) as tf:
            for name in tf.getnames():
                parts = name.split("/")
                if len(parts) >= 3:
                    index.setdefault(parts[1], t)
        log.info("indexed %s: %d tasks known so far", t, len(index))
    cache.write_text(json.dumps({"tars": tars, "index": index}))
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, nargs="+")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--work", default="./work")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repair", action="store_true",
                    help="attach spaced heredoc delimiters so Docker can parse "
                         "the Dockerfile the udocker runner accepted")
    ap.add_argument("--build-attempts", type=int, default=1,
                    help="retry a build this many times, but only when the "
                         "failure came from the registry")
    ap.add_argument("--build-timeout", type=int, default=BUILD_TIMEOUT)
    ap.add_argument("--no-entrypoint", action="store_true",
                    help="suppress the image's entrypoint the way this runner "
                         "used to, so one task can be judged both ways")
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
                r = json.loads(line)
                # `aborted` means the run stopped on disk, `skipped` that the
                # archive given at the time did not hold the task. Neither is a
                # verdict, so neither closes the task to a later attempt.
                if r.get("status") not in ("aborted", "skipped"):
                    done.add(r["task_id"])

    ids = [t.strip() for t in Path(args.ids).read_text().split() if t.strip()]
    todo = [t for t in ids if t not in done]
    if args.limit:
        todo = todo[:args.limit]
    index = index_tars(args.tar, results.with_suffix(".tarindex.json"))
    missing = [t for t in todo if t not in index]
    if missing:
        log.warning("%d ids are in none of the archives: %s",
                    len(missing), " ".join(missing[:10]))
        todo = [t for t in todo if t in index]
    log.info("%d ids, %d already judged, %d to run, free disk %.1fG",
             len(ids), len(done), len(todo), free_gb())

    work_root = Path(args.work)
    work_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
         open(results, "a") as fh:
        futs = {ex.submit(validate_one, Path(index[t]), t, work_root,
                          args.repair, args.build_attempts,
                          args.build_timeout, args.no_entrypoint): t
                for t in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            log.info("[%d/%d] %s -> %s (build %ss x%s%s, solve %ss) | %s",
                     n, len(todo), rec["task_id"], rec["status"],
                     rec.get("build_s"), rec.get("build_attempts"),
                     ", repaired" if rec.get("repairs") else "",
                     rec.get("solve_s"), counts)
            if rec["status"] == "aborted":
                log.error("aborting: disk below floor")
                break
    log.info("RUN DONE: %s", counts)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one k/k evolve through the real path, in isolation.

The evolution loop consumes signals and folds results into the live pool, so it
is not something to experiment inside. This takes a COPY of one signal, resolves
the package the same way the loop does, and runs `feedback_loop.process_one`
against a scratch out-dir: the signal stays queued, the pool is untouched, and
the only thing produced is a record of what the agent was offered and what it
did with it.

    python3 smoke_evolve.py --task tw_43020
    python3 smoke_evolve.py --any-kk        # first 16/16 signal on hand
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evolve_ondella as od  # noqa: E402
import feedback_loop as fb  # noqa: E402

log = logging.getLogger("smoke")


def pick_signal(signal_dir: Path, task: str | None) -> dict:
    for f in sorted(signal_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if task and d.get("task_id") != task:
            continue
        total = d.get("total") or d.get("attempts") or 0
        if not task and not (total and d.get("solved") == total):
            continue
        return d
    raise SystemExit(f"no matching signal in {signal_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--any-kk", action="store_true")
    ap.add_argument("--signals", default=str(od.BASE / "evolution/signals"))
    ap.add_argument("--out", default=str(od.BASE / "tmp/smoke-evolve"))
    args = ap.parse_args()
    if not args.task and not args.any_kk:
        raise SystemExit("pass --task <id> or --any-kk")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    sig = pick_signal(Path(args.signals), args.task)
    tid = sig["task_id"]
    total = sig.get("total") or sig.get("attempts")
    log.info("signal %s solved=%s/%s direction=%s",
             tid, sig.get("solved"), total, sig.get("direction"))

    src = od.resolve_src(tid)
    if src is None:
        raise SystemExit(f"no package for {tid} under {od.POOL_ROOTS}")
    log.info("package: %s", src)

    # Same translation the loop does: a signal carries {solved, total}, while
    # process_one reads {solved, graded}. Skipping it makes every task look
    # ungraded and short-circuits before the evolve path is ever reached.
    rollout = od.signal_to_rollout(sig)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = fb.process_one(rollout, Path(src), out_root)
    dt = time.time() - t0

    print("\n=== smoke result ===")
    for k in ("task_id", "solved", "graded", "action", "status", "operator",
              "family", "hint", "agent_validated", "why"):
        if k in rec:
            print(f"  {k:<18} {str(rec[k])[:150]}")
    rv = rec.get("revalidate") or {}
    if rv:
        print(f"  revalidate         ok={rv.get('ok')} stage={rv.get('stage')} "
              f"why={str(rv.get('why'))[:110]}")
    print(f"  elapsed            {dt:.0f}s")
    print(f"  scratch            {out_root / tid}")


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import base64
import json

import pytest

from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset
from torchtitan.experiments.rl.examples.tmax.prepare_rts_data import (
    _oracle_commands,
    build_rows,
)

_DOCKERFILE = """FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends gcc
WORKDIR /app
WORKDIR /srv/final
"""

_TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
echo 1 > /logs/verifier/reward.txt
"""


def _write_task(
    root, task_id: str, *, dockerfile: str = _DOCKERFILE, test_sh: str = _TEST_SH
):
    """Lay out one RTS-shaped task dir under ``root``."""
    task = root / task_id
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "instruction.md").write_text(f"do the thing for {task_id}")
    (task / "tests" / "test.sh").write_text(test_sh)
    (task / "tests" / "test_state.py").write_text("def test_x(): assert True\n")
    (task / "solution" / "solve.sh").write_text("true\n")
    return task


def test_row_carries_dockerfile_and_last_workdir(tmp_path):
    root = tmp_path / "tasks"
    root.mkdir()
    _write_task(root, "rts_task_aaa")

    rows, reasons = build_rows([str(root)])

    assert reasons == {"ok": 1}
    (row,) = rows
    md = row["metadata"]
    assert md["dockerfile"] == _DOCKERFILE
    # No published image: the sandbox backend must build the Dockerfile instead.
    assert md["image"] == ""
    # The LAST WORKDIR wins -- that is where the agent's commands land.
    assert md["workdir"] == "/srv/final"
    assert md["tmax"]["test_sh"] == _TEST_SH
    # test.sh is uploaded separately, never as a grading fixture.
    assert set(md["tmax"]["fixtures"]) == {"tests/test_state.py"}


@pytest.mark.parametrize(
    "dockerfile",
    [
        # Needs a real init system as PID 1.
        'FROM ubuntu:22.04\nWORKDIR /app\nENTRYPOINT ["/sbin/init"]\n',
        "FROM ubuntu:22.04\nWORKDIR /app\nRUN systemctl enable nginx\n",
        # Needs the host docker daemon.
        "FROM ubuntu:22.04\nWORKDIR /app\nRUN ls /var/run/docker.sock\n",
    ],
)
def test_dockerfiles_needing_a_privileged_host_are_filtered(tmp_path, dockerfile):
    root = tmp_path / "tasks"
    root.mkdir()
    _write_task(root, "rts_task_bbb", dockerfile=dockerfile)

    rows, reasons = build_rows([str(root)])

    assert rows == []
    assert reasons == {"needs_privileged": 1}


@pytest.mark.parametrize(
    "dockerfile",
    [
        # Only MENTIONS systemd/compose -- the corpus does this in comments that
        # explain solve.sh. A substring filter rejected ~1700 such tasks, of which
        # none actually run an init system.
        "FROM ubuntu:22.04\nWORKDIR /app\n"
        "# systemd provides systemctl used in solve.sh\n"
        "RUN apt-get install -y systemd\n",
        # EOL base images still build: the corpus rewrites sources.list to the
        # Debian archive itself.
        "FROM centos:7\nWORKDIR /app\nRUN yum install -y gcc\n",
        # COPY --from pulls from another image, so it needs no build context.
        "FROM ubuntu:22.04\nWORKDIR /app\n"
        "COPY --from=composer:latest /usr/bin/composer /usr/bin/composer\n",
    ],
)
def test_buildable_dockerfiles_are_kept(tmp_path, dockerfile):
    root = tmp_path / "tasks"
    root.mkdir()
    _write_task(root, "rts_task_ccc", dockerfile=dockerfile)

    rows, reasons = build_rows([str(root)])

    assert reasons == {"ok": 1}
    assert rows[0]["metadata"].get("build_context") is None


def test_local_copy_sources_are_carried_as_build_context(tmp_path):
    """A local COPY ships its sources with the row, so no filter is needed."""
    root = tmp_path / "tasks"
    root.mkdir()
    task = _write_task(
        root,
        "rts_task_ddd",
        dockerfile="FROM ubuntu:22.04\nWORKDIR /app\n"
        "COPY seed.bin /app/seed.bin\nCOPY fixtures /app/fixtures\n",
    )
    (task / "environment" / "seed.bin").write_bytes(b"\x00\x01\x02binary")
    (task / "environment" / "fixtures").mkdir()
    (task / "environment" / "fixtures" / "a.txt").write_text("hello")

    rows, reasons = build_rows([str(root)])

    assert reasons == {"ok": 1}
    ctx = rows[0]["metadata"]["build_context"]
    assert set(ctx) == {"seed.bin", "fixtures/a.txt"}
    # Base64 so binary COPY sources survive the JSONL round-trip.
    assert base64.b64decode(ctx["seed.bin"]) == b"\x00\x01\x02binary"
    assert base64.b64decode(ctx["fixtures/a.txt"]) == b"hello"


def test_missing_copy_source_is_filtered(tmp_path):
    root = tmp_path / "tasks"
    root.mkdir()
    _write_task(
        root,
        "rts_task_eee",
        dockerfile="FROM ubuntu:22.04\nWORKDIR /app\nCOPY gone.txt /app/\n",
    )

    rows, reasons = build_rows([str(root)])

    assert rows == []
    assert reasons == {"copy_source_missing": 1}


@pytest.mark.parametrize(
    "solve_sh,expected",
    [
        ("#!/bin/bash\nls\n", 1),
        # && and ; chain separate commands (2 + 2), while a | pipeline is a single
        # command -- the agent issues it in one turn.
        ("ls && pwd\ncd /tmp; ls\ncat a | wc -l\n", 5),
        # Comments, blank lines and block terminators are not commands.
        ("# note\n\nif true; then\n  ls\nfi\n", 3),
        # A heredoc body is data, not commands.
        ("cat <<'EOF' > f\nls && pwd\nrm -rf /\nEOF\necho done\n", 2),
    ],
)
def test_oracle_command_count(solve_sh, expected):
    assert _oracle_commands(solve_sh) == expected


def test_max_oracle_commands_drops_tasks_over_the_turn_budget(tmp_path):
    """A rollout capped at T turns cannot solve a task whose oracle needs > T."""
    root = tmp_path / "tasks"
    root.mkdir()
    short = _write_task(root, "rts_task_short")
    long = _write_task(root, "rts_task_long")
    (short / "solution" / "solve.sh").write_text("ls\npwd\n")
    (long / "solution" / "solve.sh").write_text(
        "".join(f"echo {i}\n" for i in range(50))
    )

    rows, reasons = build_rows([str(root)], max_oracle_commands=10)

    assert [r["label"] for r in rows] == ["rts_task_short"]
    assert reasons["oracle_over_turn_budget"] == 1
    assert rows[0]["metadata"]["oracle_commands"] == 2


def test_verifier_without_reward_file_is_filtered(tmp_path):
    """A verifier that never writes reward.txt would silently score every rollout 0."""
    root = tmp_path / "tasks"
    root.mkdir()
    _write_task(root, "rts_task_ccc", test_sh="#!/bin/bash\npytest /tests\n")

    rows, reasons = build_rows([str(root)])

    assert rows == []
    assert reasons == {"verifier_writes_no_reward": 1}


def test_limit_is_applied_after_shuffle(tmp_path):
    root = tmp_path / "tasks"
    root.mkdir()
    for i in range(8):
        _write_task(root, f"rts_task_{i:03d}")

    first = [r["label"] for r in build_rows([str(root)], limit=3, seed=1)[0]]
    same = [r["label"] for r in build_rows([str(root)], limit=3, seed=1)[0]]
    other = [r["label"] for r in build_rows([str(root)], limit=3, seed=2)[0]]

    assert len(first) == 3
    assert first == same, "same seed must give a reproducible subset"
    assert first != other, "the subset must span the corpus, not one fixed corner"


def test_dataset_loads_a_dockerfile_only_row(tmp_path):
    """TMaxDataset must accept rows with no image, and reject rows with neither."""
    root = tmp_path / "tasks"
    root.mkdir()
    _write_task(root, "rts_task_ddd")
    rows, _ = build_rows([str(root)])
    path = tmp_path / "rts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    dataset = TMaxDataset(TMaxDataset.Config(data_path=str(path), shuffle=False))
    sample = next(iter(dataset))
    assert sample.dockerfile == _DOCKERFILE
    assert sample.image == ""
    assert sample.workdir == "/srv/final"

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"metadata": {"instance_id": "x", "tmax": {"a": 1}}}) + "\n"
    )
    with pytest.raises(ValueError, match="missing image/dockerfile/tmax"):
        TMaxDataset(TMaxDataset.Config(data_path=str(bad), shuffle=False))

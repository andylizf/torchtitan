# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

import pytest

from torchtitan.experiments.rl.examples.tmax.data import TMaxDataset


def _write_row(tmp_path, *, disk_gb):
    metadata = {
        "instance_id": "task-1",
        "image": "example/image",
        "tmax": {"test_sh": "true"},
    }
    if disk_gb is not None:
        metadata["daytona_disk_gb"] = disk_gb
    path = tmp_path / "tmax.jsonl"
    path.write_text(json.dumps({"metadata": metadata}) + "\n")
    return path


def _write_rows(tmp_path, num_rows: int):
    path = tmp_path / "tmax.jsonl"
    rows = []
    for index in range(num_rows):
        rows.append(
            json.dumps(
                {
                    "metadata": {
                        "instance_id": f"task-{index}",
                        "image": "example/image",
                        "tmax": {"test_sh": "true"},
                    }
                }
            )
        )
    path.write_text("\n".join(rows) + "\n")
    return path


def test_dataset_reads_per_task_daytona_disk(tmp_path) -> None:
    path = _write_row(tmp_path, disk_gb=20)
    dataset = TMaxDataset(TMaxDataset.Config(data_path=str(path), shuffle=False))

    assert next(dataset).daytona_disk_gb == 20


def test_dataset_defaults_per_task_daytona_disk_to_none(tmp_path) -> None:
    path = _write_row(tmp_path, disk_gb=None)
    dataset = TMaxDataset(TMaxDataset.Config(data_path=str(path), shuffle=False))

    assert next(dataset).daytona_disk_gb is None


@pytest.mark.parametrize("disk_gb", [0, -1, True, 1.5, "20"])
def test_dataset_rejects_invalid_per_task_daytona_disk(tmp_path, disk_gb) -> None:
    path = _write_row(tmp_path, disk_gb=disk_gb)

    with pytest.raises(ValueError, match="invalid daytona_disk_gb"):
        TMaxDataset(TMaxDataset.Config(data_path=str(path), shuffle=False))


def test_dataset_skips_initial_samples_and_preserves_wraps(tmp_path) -> None:
    path = _write_rows(tmp_path, 3)
    baseline = TMaxDataset(
        TMaxDataset.Config(data_path=str(path), seed=7, shuffle=True)
    )
    for _ in range(4):
        next(baseline)

    resumed = TMaxDataset(
        TMaxDataset.Config(
            data_path=str(path),
            seed=7,
            shuffle=True,
            initial_skip_samples=4,
        )
    )

    assert [next(resumed) for _ in range(6)] == [next(baseline) for _ in range(6)]


def test_dataset_rejects_negative_initial_skip_samples(tmp_path) -> None:
    path = _write_rows(tmp_path, 3)

    with pytest.raises(ValueError, match="initial_skip_samples must be non-negative"):
        TMaxDataset(
            TMaxDataset.Config(
                data_path=str(path),
                shuffle=False,
                initial_skip_samples=-1,
            )
        )

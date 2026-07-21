# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.experiments.rl.examples.tmax.config_registry import (
    rl_grpo_qwen3_4b_tmax,
    rl_grpo_qwen3_5_9b_tmax,
)


@pytest.mark.parametrize(
    ("lr_override", "expected_lr"),
    [(None, 1e-6), ("2e-7", 2e-7)],
)
def test_tmax_9b_uses_open_instruct_optimizer_and_mixed_precision(
    monkeypatch: pytest.MonkeyPatch,
    lr_override: str | None,
    expected_lr: float,
) -> None:
    monkeypatch.delenv("SWE_GDN_BI", raising=False)
    if lr_override is None:
        monkeypatch.delenv("SWE_LR", raising=False)
    else:
        monkeypatch.setenv("SWE_LR", lr_override)

    config = rl_grpo_qwen3_5_9b_tmax()

    training = config.trainer.training
    assert training.dtype == "float32"
    assert training.mixed_precision_param == "bfloat16"
    assert training.mixed_precision_reduce == "float32"

    optimizer = config.trainer.optimizer
    assert optimizer.implementation == "fused"
    assert len(optimizer.param_groups) == 1
    param_group = optimizer.param_groups[0]
    assert param_group.optimizer_name == "AdamW"
    assert param_group.optimizer_kwargs == {
        "lr": expected_lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }


def test_tmax_batch_invariant_diagnostics_keep_required_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWE_LR", raising=False)
    monkeypatch.setenv("SWE_GDN_BI", "1")

    config = rl_grpo_qwen3_5_9b_tmax()

    assert config.trainer.debug.batch_invariant
    assert config.trainer.training.dtype == "bfloat16"


def test_tmax_4b_diagnostic_keeps_required_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWE_GDN_BI", raising=False)
    monkeypatch.delenv("SWE_LR", raising=False)

    config = rl_grpo_qwen3_4b_tmax()

    assert config.trainer.debug.batch_invariant
    assert config.trainer.training.dtype == "bfloat16"

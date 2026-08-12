# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Advantage estimator: a post-scoring step run by the ``Rollouter``.

After a group is scored, this turns the group's rewards into per-rollout advantages
(in group order) that the trainer/loss consumes directly.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.experiments.rl.rollout.types import is_scored, RolloutGroup


class AdvantageEstimator(Configurable):
    """Group-relative advantage estimator: ``A_i = (r_i - mean(r)) / denom``.

    ``denom = std(r) + eps`` when ``should_std_normalize`` (standard GRPO), else
    ``1.0`` (Dr.GRPO mean-baseline). The name is intentionally general — other
    advantage schemes can live here later.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        should_std_normalize: bool = False
        """Divide the centered advantage by the group reward std (+eps) — standard GRPO.
        False (default) = mean-baseline only (Dr.GRPO)."""

    def __init__(self, config: Config) -> None:
        self.should_std_normalize: bool = config.should_std_normalize

    def __call__(self, group: RolloutGroup) -> list[float]:
        """Return per-rollout advantages, in group order.

        A NaN reward means "no verdict" -- nothing scored this rollout, e.g. an
        infrastructure failure -- which is not the same as 0.0, a verdict of failure.
        It is left out of the baseline and carries a NaN advantage for the sample
        builder to drop, so a group of 8 holding one of them baselines over the
        surviving 7. Keeping it in would both hand its own turns a negative advantage
        for something no verdict established and pull the baseline down for its
        siblings.
        """
        scored = [rollout.reward for rollout in group.rollouts if is_scored(rollout)]
        if not scored:
            # Nothing in the group was scored, so there is no baseline to form.
            return [math.nan] * len(group.rollouts)
        group_mean = sum(scored) / len(scored)
        denom = (statistics.pstdev(scored) + 1e-6) if self.should_std_normalize else 1.0
        return [
            (rollout.reward - group_mean) / denom if is_scored(rollout) else math.nan
            for rollout in group.rollouts
        ]

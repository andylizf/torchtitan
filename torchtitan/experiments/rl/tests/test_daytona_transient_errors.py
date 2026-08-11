# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which provider errors count as retryable.

Session create, exec and the other idempotent RPCs all gate their retry on
``_is_transient_rpc_error``, so an unrecognized transient error is not "one
noisy log line" -- it costs the whole rollout. The provider reports some of its
own 5xx conditions as prose rather than a status code, which the numeric checks
alone miss.
"""

from __future__ import annotations

import pytest

from torchtitan.experiments.rl.harness.sandbox.daytona import _is_transient_rpc_error


class _HttpError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    "error",
    [
        # Prose 5xx: observed on session create under a wide rollout fanout as
        # "internal server error: failed to create session config directory".
        Exception(
            "Failed to create session: internal server error: failed to create "
            "session config directory: mkdir /root/.daytona/session"
        ),
        Exception("Bad Gateway"),
        Exception("Service Unavailable"),
        # Numeric forms.
        _HttpError("boom", 500),
        _HttpError("slow down", 429),
        Exception("status code 503"),
        # Transport.
        ConnectionError("connection reset by peer"),
        TimeoutError("timed out"),
    ],
)
def test_transient_errors_are_retried(error):
    assert _is_transient_rpc_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        # Retrying cannot free blocks or inodes.
        Exception("no space left on device"),
        # A real client error: the request itself is wrong.
        _HttpError("bad request", 400),
        _HttpError("not found", 404),
        Exception("Unknown tool 'frobnicate'"),
    ],
)
def test_permanent_errors_are_not_retried(error):
    assert _is_transient_rpc_error(error) is False

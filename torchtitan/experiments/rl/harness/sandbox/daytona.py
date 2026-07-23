# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Daytona cloud sandbox backend.

Boots a remote cloud container from a Docker image. Meant for boxes that cannot
expose an inbound port to the cloud (an internal dev box): the agent inside the
sandbox reaches the on-box Anthropic adapter through a file-relay bridge over the
Daytona ``fs`` API (see ``bridge.py``), not a direct dial-back.

Ported from THUDM/slime ``slime/agent/sandbox.py``.
"""

from __future__ import annotations

import json

import logging
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from torchtitan.experiments.rl.harness.sandbox.base import (
    _getenv,
    ExecResult,
    FileContent,
    SandboxIssue,
    SandboxIssueTracker,
)

logger = logging.getLogger(__name__)

# Label stamped on every sandbox we create, so cleanup can target ONLY our
# sandboxes (never another tenant's) when sharing a Daytona account.
HARNESS_LABELS = {"owner": "titan_swe_r2e"}

_COMMAND_KILL_GRACE_SEC = 10
_DEFAULT_EXEC_REQUEST_TIMEOUT_SEC = 120
_SESSION_POLL_GRACE_SEC = 120
_SESSION_RPC_TIMEOUT_SEC = 60
_COMMAND_RECOVERY_DELAYS_SEC = (0.0, 0.25, 1.0)
_EXEC_RESULT_DIR = "/tmp/.torchtitan_exec"


def _error_status_code(error: BaseException) -> int | None:
    for value in (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _is_sandbox_gone_error(error: BaseException) -> bool:
    if _error_status_code(error) == 404:
        return True
    name = type(error).__name__.lower()
    if "notfound" in name or "not_found" in name:
        return True
    message = str(error).lower()
    return (
        ("sandbox" in message and "not found" in message)
        or "sandbox has been deleted" in message
        or "no such container" in message
        or "no ip address" in message
    )


def _is_transient_rpc_error(error: BaseException) -> bool:
    status_code = _error_status_code(error)
    if status_code == 429 or (status_code is not None and status_code >= 500):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    name = type(error).__name__.lower()
    if "connection" in name or "timeout" in name or "throttl" in name:
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "connection reset",
            "server disconnected",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "too many requests",
            "status code 429",
            "status code 500",
            "status code 502",
            "status code 503",
            "status code 504",
        )
    )


def _build_exec_command(
    cmd: str,
    *,
    user: str,
    env: dict[str, str] | None,
    timeout: int,
) -> str:
    """Build one shell-safe command for the Daytona session API."""
    argv = ["bash", "-c", cmd]
    if user and user != "root":
        keys = ",".join((env or {}).keys())
        argv = ["runuser", "-u", user]
        if keys:
            argv.append(f"--whitelist-environment={keys}")
        argv.extend(["--", "bash", "-c", cmd])
    if env:
        argv = ["env", "--", *(f"{key}={value}" for key, value in env.items()), *argv]
    argv = [
        "timeout",
        "--signal=TERM",
        f"--kill-after={_COMMAND_KILL_GRACE_SEC}s",
        f"{timeout}s",
        *argv,
    ]
    return shlex.join(argv)


def _build_observable_exec_command(full: str, command_key: str) -> tuple[str, str, str]:
    """Wrap a command with an atomic result sentinel and captured output."""
    result_prefix = f"{_EXEC_RESULT_DIR}/{command_key}"
    output_path = f"{result_prefix}.output"
    status_path = f"{result_prefix}.status"
    status_tmp_path = f"{status_path}.tmp"
    quoted_dir = shlex.quote(_EXEC_RESULT_DIR)
    quoted_output = shlex.quote(output_path)
    quoted_status = shlex.quote(status_path)
    quoted_status_tmp = shlex.quote(status_tmp_path)
    observed = (
        f"mkdir -p {quoted_dir}; "
        f"rm -f {quoted_output} {quoted_status} {quoted_status_tmp}; "
        f"{{ {full}; }} > {quoted_output} 2>&1; "
        "_tt_exec_rc=$?; "
        f"printf '%s\\n' \"$_tt_exec_rc\" > {quoted_status_tmp}; "
        f"mv -f {quoted_status_tmp} {quoted_status}"
    )
    return observed, status_path, output_path


def _is_missing_file_error(error: BaseException) -> bool:
    message = str(error).lower()
    if (
        ("sandbox" in message and "not found" in message)
        or "sandbox has been deleted" in message
        or "no such container" in message
        or "no ip address" in message
    ):
        return False
    if _error_status_code(error) == 404 or isinstance(error, FileNotFoundError):
        return True
    return any(
        marker in message
        for marker in (
            "file not found",
            "no such file",
            "does not exist",
            "not found",
        )
    )


def _eager_rebuild_daytona_models() -> None:
    """Make the Daytona SDK's Pydantic models resolvable in OUR import context.

    The SDK models use ``from __future__ import annotations`` + TYPE_CHECKING typing
    imports, so their module globals lack the forward-ref names (Optional, StrictStr,
    ...) at runtime. When daytona is imported INSIDE the torchtitan/vllm/pydantic
    import chain (the RL controller), pydantic cannot resolve those refs on first
    construct -> ``PydanticUserError: ... is not fully defined`` -> every create/exec
    fails -> turns=0. (In a clean process the refs happen to resolve, which is why the
    curate worker booted sandboxes fine but the controller did not.) Fix: load every
    daytona submodule and inject all typing + pydantic public names into each module's
    globals, so the SDK's own lazy rebuild (triggered later at the construct sites)
    resolves the refs.

    We do NOT call ``model_rebuild`` (force or otherwise): it re-derives the schema and
    DROPS the SDK's field defaults (made SessionExecuteRequest.var_async etc. required).
    Defaults that this context still drops anyway are passed explicitly at the construct
    sites (see ``exec``). Best-effort: a no-op if Daytona isn't installed.
    """
    try:
        import importlib
        import pkgutil
        import sys
        import typing

        import daytona  # type: ignore

        import pydantic

        try:
            for info in pkgutil.walk_packages(daytona.__path__, daytona.__name__ + "."):
                try:
                    importlib.import_module(info.name)
                except Exception:  # noqa: BLE001 -- skip unimportable submodules
                    pass
        except Exception:  # noqa: BLE001 -- pkgutil best-effort
            pass

        inject = {n: getattr(typing, n) for n in dir(typing) if not n.startswith("_")}
        inject.update(
            {n: getattr(pydantic, n) for n in dir(pydantic) if not n.startswith("_")}
        )
        for name, mod in list(sys.modules.items()):
            if name.startswith("daytona") and mod is not None:
                for key, val in inject.items():
                    if not hasattr(mod, key):
                        setattr(mod, key, val)
    except Exception as e:  # noqa: BLE001 -- best-effort
        logger.warning("[daytona] model namespace inject skipped: %s", e)


_eager_rebuild_daytona_models()


# Process-wide AsyncDaytona client, shared by every sandbox in this worker: one
# client = one pooled TLS session reused across all concurrent rollouts. A
# per-sandbox client instead opens its own pool, and the handshake storm under a
# many-way boot fanout over a high-latency link times out the exec requests.
_SHARED_CLIENT = None
_SHARED_CLIENT_LOCK = None

# Cap concurrent sandbox CREATES (held only during create/boot, not the rollout).
# At high rollout concurrency every group tries to boot its siblings at once; the
# resulting create burst trips Daytona's rate limit (ThrottlerException / 429),
# and the whole fanout backs off in lockstep. This throttles creates to a
# Daytona-friendly rate while letting far more sandboxes RUN concurrently. Per
# worker process; the Daytona account limit is shared, so total = num_workers x
# this value -- keep the product well under the observed throttle point.
_CREATE_SEM = None


def _create_sem():
    import asyncio

    global _CREATE_SEM
    if _CREATE_SEM is None:
        _CREATE_SEM = asyncio.Semaphore(
            int(_getenv("TT_DAYTONA_CREATE_CONCURRENCY", default="16"))
        )
    return _CREATE_SEM


async def _get_shared_client(*, api_key: str | None, api_url, target):
    global _SHARED_CLIENT, _SHARED_CLIENT_LOCK
    import asyncio

    from daytona import AsyncDaytona, DaytonaConfig  # type: ignore

    if not api_key:
        raise RuntimeError(
            "DAYTONA_API_KEY is not set; required for the daytona sandbox backend."
        )
    if _SHARED_CLIENT_LOCK is None:
        _SHARED_CLIENT_LOCK = asyncio.Lock()
    async with _SHARED_CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            cfg = DaytonaConfig(api_key=api_key, api_url=api_url, target=target)
            _SHARED_CLIENT = AsyncDaytona(cfg)
    return _SHARED_CLIENT


class DaytonaSandbox:
    """Async sandbox over a Daytona cloud sandbox (https://daytona.io).

    Daytona builds a snapshot wrapping any public image on first ``create`` and
    runs ``process.exec`` as the image's default user. R2E-Gym images default to
    ``root`` with the repo interpreter on ``PATH``, so ``exec`` runs as root and
    drops to an unprivileged user via ``runuser`` when asked. ``fs`` writes as
    root, so ``write_file`` uploads then chowns.

    Env knobs:
      ``DAYTONA_API_KEY``                  -- API key (required).
      ``DAYTONA_API_URL`` / ``DAYTONA_TARGET`` -- override cloud endpoint/region.
      ``TT_DAYTONA_CPU``                   -- vCPUs per sandbox (default 2).
      ``TT_DAYTONA_MEM_GB``                -- memory GiB per sandbox (default 4).
      ``TT_DAYTONA_DISK_GB``               -- disk GiB per sandbox (default 6).
      ``TT_DAYTONA_CREATE_TIMEOUT``        -- snapshot-build/boot wait (default 900s).
      ``TT_DAYTONA_EXEC_TIMEOUT_MIN``       -- SDK request timeout floor; does not
                                               extend the command runtime limit.
      ``TT_DAYTONA_SESSION_CREATE_RETRIES`` -- empty-session creation retries;
                                               defaults to 5.
      ``TT_DAYTONA_RPC_RETRIES``            -- retries for idempotent RPCs; default 2.
      ``TT_DAYTONA_HEARTBEAT_SEC``          -- activity refresh interval; default
                                               180s, or 0 to disable.
    """

    api_key_env = ("DAYTONA_API_KEY",)
    api_url_env = ("DAYTONA_API_URL",)
    target_env = ("DAYTONA_TARGET",)

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        disk_gb: int | None = None,
        issue_tracker: SandboxIssueTracker | None = None,
        **_ignored,
    ) -> None:
        if disk_gb is not None and (
            isinstance(disk_gb, bool) or not isinstance(disk_gb, int) or disk_gb <= 0
        ):
            raise ValueError(
                f"daytona disk_gb must be a positive integer, got {disk_gb!r}"
            )
        self.image = image
        self.timeout = timeout
        self.disk_gb = disk_gb
        self.allocated_disk_gb: int | None = None
        self.issue_tracker = issue_tracker or SandboxIssueTracker()
        # Daytona is optional and imported lazily, so its SDK types are not
        # available for static annotations in this module.
        self._client: Any = None
        self._sb: Any = None
        self.sandbox_id = ""
        self._heartbeat_task: Any = None
        self._lost_error: BaseException | None = None

    def _record_issue(
        self,
        kind: str,
        *,
        phase: str,
        error: BaseException | str,
        recovered: bool = False,
        session_id: str = "",
        command_id: str = "",
        attempt: int | None = None,
        max_attempts: int | None = None,
        exit_code: int | None = None,
        emit_log: bool = True,
    ) -> None:
        message = " ".join(str(error).split())[:1000]
        error_type = type(error).__name__ if isinstance(error, BaseException) else ""
        issue = SandboxIssue(
            provider="daytona",
            kind=kind,
            phase=phase,
            recovered=recovered,
            error_type=error_type,
            message=message,
            sandbox_id=self.sandbox_id,
            session_id=session_id,
            command_id=command_id,
            attempt=attempt,
            max_attempts=max_attempts,
            exit_code=exit_code,
        )
        self.issue_tracker.record(issue)
        if not emit_log:
            return
        context = self.issue_tracker.context
        payload = {
            "event": "sandbox_issue",
            "provider": issue.provider,
            "kind": issue.kind,
            "phase": issue.phase,
            "recovered": issue.recovered,
            "instance_id": context.instance_id,
            "group_id": context.group_id,
            "rollout_id": context.rollout_id,
            "sandbox_id": issue.sandbox_id,
            "image": self.image,
            "disk_gb": self.allocated_disk_gb or self.disk_gb,
            "session_id": issue.session_id,
            "command_id": issue.command_id,
            "attempt": issue.attempt,
            "max_attempts": issue.max_attempts,
            "exit_code": issue.exit_code,
            "error_type": issue.error_type,
            "message": issue.message,
        }
        logger.warning("[sandbox_issue] %s", json.dumps(payload, sort_keys=True))

    def _mark_sandbox_lost(self, error: BaseException, *, phase: str) -> None:
        if self._lost_error is not None:
            return
        self._lost_error = error
        self._record_issue("sandbox_lost", phase=phase, error=error)

    def _raise_if_sandbox_lost(self) -> None:
        if self._lost_error is None:
            return
        raise RuntimeError(
            f"daytona sandbox {self.sandbox_id or '<unknown>'} is no longer available"
        ) from self._lost_error

    async def _retry_idempotent_rpc(
        self,
        call: Callable[[], Awaitable[Any]],
        *,
        phase: str,
        retry_kind: str,
        failed_kind: str,
        session_id: str = "",
        command_id: str = "",
    ) -> Any:
        import asyncio
        import random

        retries = int(_getenv("TT_DAYTONA_RPC_RETRIES", default="2"))
        if retries < 0:
            raise ValueError(
                f"TT_DAYTONA_RPC_RETRIES must be non-negative, got {retries}"
            )
        max_attempts = retries + 1
        backoff = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.wait_for(call(), timeout=_SESSION_RPC_TIMEOUT_SEC)
            except Exception as error:
                if _is_sandbox_gone_error(error):
                    self._mark_sandbox_lost(error, phase=phase)
                    raise
                terminal = attempt >= max_attempts or not _is_transient_rpc_error(error)
                self._record_issue(
                    failed_kind if terminal else retry_kind,
                    phase=phase,
                    error=error,
                    recovered=not terminal,
                    session_id=session_id,
                    command_id=command_id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    emit_log=terminal,
                )
                if terminal:
                    raise
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 5.0)
        raise AssertionError("idempotent RPC retry loop did not return or raise")

    async def _heartbeat_loop(self, interval_sec: float) -> None:
        import asyncio
        import random

        while True:
            await asyncio.sleep(interval_sec * (0.9 + 0.2 * random.random()))
            try:
                await asyncio.wait_for(
                    self._sb.refresh_activity(), timeout=_SESSION_RPC_TIMEOUT_SEC
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if _is_sandbox_gone_error(error):
                    self._mark_sandbox_lost(error, phase="heartbeat")
                    return
                self._record_issue(
                    "heartbeat_retry",
                    phase="heartbeat",
                    error=error,
                    recovered=True,
                    emit_log=False,
                )

    @property
    def daytona(self):
        """Underlying ``daytona.AsyncSandbox`` (for the fs-relay bridge)."""
        return self._sb

    async def __aenter__(self) -> DaytonaSandbox:
        import asyncio
        import random

        from daytona import CreateSandboxFromImageParams, Resources  # type: ignore

        self._client = await _get_shared_client(
            api_key=_getenv(*self.api_key_env),
            api_url=_getenv(*self.api_url_env) or None,
            target=_getenv(*self.target_env) or None,
        )
        cpu = int(_getenv("TT_DAYTONA_CPU", default="2"))
        mem = int(_getenv("TT_DAYTONA_MEM_GB", default="4"))
        disk = (
            self.disk_gb
            if self.disk_gb is not None
            else int(_getenv("TT_DAYTONA_DISK_GB", default="6"))
        )
        self.allocated_disk_gb = disk
        create_timeout = float(_getenv("TT_DAYTONA_CREATE_TIMEOUT", default="900"))
        # Cloud-side TTL so an orphan (left by a SIGKILL'd run that never reached
        # __aexit__, e.g. MAST preemption) self-reaps: once it goes idle it
        # auto-stops after auto_stop minutes, then auto-deletes immediately
        # (auto_delete=0). The heartbeat below refreshes cloud activity while a live
        # rollout is waiting on model generation and making no Daytona RPCs.
        auto_stop = int(_getenv("TT_DAYTONA_AUTO_STOP_MIN", default="10"))
        auto_delete = int(_getenv("TT_DAYTONA_AUTO_DELETE_MIN", default="0"))
        default_heartbeat_sec = min(180.0, auto_stop * 20.0) if auto_stop > 0 else 0.0
        heartbeat_sec = float(
            _getenv(
                "TT_DAYTONA_HEARTBEAT_SEC",
                default=str(default_heartbeat_sec),
            )
        )
        if heartbeat_sec < 0:
            raise ValueError(
                f"TT_DAYTONA_HEARTBEAT_SEC must be non-negative, got {heartbeat_sec}"
            )
        if int(_getenv("TT_DAYTONA_RPC_RETRIES", default="2")) < 0:
            raise ValueError("TT_DAYTONA_RPC_RETRIES must be non-negative")
        params = CreateSandboxFromImageParams(
            image=self.image,
            resources=Resources(cpu=cpu, memory=mem, disk=disk),
            labels=HARNESS_LABELS,
            auto_stop_interval=auto_stop,
            auto_delete_interval=auto_delete,
        )
        # Daytona create transiently 401s a valid key under a concurrent boot burst.
        # Retry with jittered backoff so a wide fanout does not retry in lockstep.
        # The create-concurrency semaphore (held only for the create+boot, released
        # before the rollout runs) keeps the create rate under Daytona's throttle.
        retries = int(_getenv("TT_DAYTONA_CREATE_RETRIES", default="5"))
        backoff = 5.0
        for attempt in range(retries + 1):
            # Hold the create-concurrency semaphore ONLY around the actual create
            # call, never during the backoff sleep. A create that hits a bad node
            # (e.g. sysbox-mgr unavailable) and backs off must release its slot so
            # other rollouts can boot; holding it across the exponential backoff
            # (up to ~135s over 5 retries) collapses the effective create
            # concurrency and starves long-tail groups (0 completed rollouts).
            try:
                async with _create_sem():
                    self._sb = await self._client.create(params, timeout=create_timeout)
                break
            except Exception as e:
                terminal = attempt >= retries
                self._record_issue(
                    "create_failed" if terminal else "create_retry",
                    phase="create",
                    error=e,
                    recovered=not terminal,
                    attempt=attempt + 1,
                    max_attempts=retries + 1,
                )
                if attempt >= retries:
                    raise
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 60.0)
        self.sandbox_id = self._sb.id
        if heartbeat_sec > 0:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(heartbeat_sec),
                name=f"daytona_heartbeat_{self.sandbox_id}",
            )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        import asyncio
        import random

        # Delete only this sandbox; never close the process-wide shared client --
        # other concurrent rollouts are still using its pooled connections.
        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._sb is None:
            return
        retries = int(_getenv("TT_DAYTONA_RPC_RETRIES", default="2"))
        if retries < 0:
            raise ValueError(
                f"TT_DAYTONA_RPC_RETRIES must be non-negative, got {retries}"
            )
        max_attempts = retries + 1
        backoff = 0.5
        for attempt in range(1, max_attempts + 1):
            try:
                await asyncio.wait_for(
                    self._client.delete(self._sb),
                    timeout=_SESSION_RPC_TIMEOUT_SEC,
                )
                break
            except Exception as error:
                # Delete is idempotent. A 404 means the desired final state already
                # holds, commonly because Daytona auto-deleted an idle sandbox.
                if _is_sandbox_gone_error(error):
                    break
                terminal = attempt >= max_attempts or not _is_transient_rpc_error(error)
                self._record_issue(
                    "delete_failed" if terminal else "delete_retry",
                    phase="delete",
                    error=error,
                    recovered=not terminal,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    emit_log=terminal,
                )
                if terminal:
                    break
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 5.0)
        self._raise_if_sandbox_lost()

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        import os

        self._raise_if_sandbox_lost()
        if timeout <= 0:
            raise ValueError(f"daytona exec timeout must be positive, got {timeout}")

        # The SDK timeout only controls the host->Daytona request. Keep its
        # high-latency floor separate from the in-sandbox command deadline.
        request_timeout = max(
            _DEFAULT_EXEC_REQUEST_TIMEOUT_SEC,
            int(os.environ.get("TT_DAYTONA_EXEC_TIMEOUT_MIN", "0")),
        )

        # The original command remains one opaque `bash -c` argument even when it
        # contains quotes or newlines.
        full = _build_exec_command(cmd, user=user, env=env, timeout=timeout)
        num_issues_before = self.issue_tracker.num_events
        try:
            rc, out = await self._session_exec(
                full,
                command_timeout=timeout,
                request_timeout=request_timeout,
            )
        except Exception as e:
            if self.issue_tracker.num_events == num_issues_before:
                self._record_issue("exec_failed", phase="exec", error=e)
            raise
        out_lower = out.lower()
        if rc != 0 and (
            "no space left on device" in out_lower or "errno 28" in out_lower
        ):
            self._record_issue(
                "command_disk_exhausted",
                phase="command",
                error=f"command exited {rc}: {out[:400]}",
                exit_code=rc,
            )
        # Match Open-Instruct's backend convention: GNU timeout reserves 124 for
        # the synthetic timeout diagnostic. Forced SIGKILL remains raw 137.
        err = f"Command timed out after {timeout}s." if rc == 124 else ""
        if rc == 124:
            self._record_issue(
                "command_timeout",
                phase="command",
                error=err,
                exit_code=rc,
            )
        if check and rc != 0:
            detail = out + (f"\n{err}" if out and err else err)
            raise RuntimeError(
                f"daytona exec failed (exit={rc}): {cmd[:120]}\n{detail[:400]}"
            )
        return rc, out, err

    async def _delete_exec_session(self, sid: str, *, reason: str) -> None:
        """Best-effort cleanup for a command whose outcome is not usable."""
        import asyncio

        try:
            await asyncio.wait_for(
                self._sb.process.delete_session(sid),
                timeout=_SESSION_RPC_TIMEOUT_SEC,
            )
        except Exception as e:
            self._record_issue(
                "session_cleanup_failed",
                phase="session_cleanup",
                error=f"{reason}: {e}",
                session_id=sid,
            )

    async def _recover_session_command_id(self, sid: str, full: str) -> str:
        """Find a submitted command after its execute response was lost."""
        import asyncio

        for delay in _COMMAND_RECOVERY_DELAYS_SEC:
            if delay:
                await asyncio.sleep(delay)
            try:
                session = await asyncio.wait_for(
                    self._sb.process.get_session(sid),
                    timeout=_SESSION_RPC_TIMEOUT_SEC,
                )
            except Exception as e:
                if _is_sandbox_gone_error(e):
                    self._mark_sandbox_lost(e, phase="execute_recovery")
                    raise
                self._record_issue(
                    "command_recovery_query_failed",
                    phase="execute_recovery",
                    error=e,
                    recovered=True,
                    session_id=sid,
                    emit_log=False,
                )
                continue
            commands = getattr(session, "commands", None) or []
            for command in reversed(commands):
                if getattr(command, "command", None) == full:
                    command_id = getattr(command, "id", None)
                    if command_id:
                        return str(command_id)
        return ""

    async def _session_exec(
        self,
        full: str,
        *,
        command_timeout: int,
        request_timeout: int,
    ) -> tuple[int, str]:
        """Submit one command, recover an ambiguous response, then read its result.

        Successful sessions are intentionally retained: deleting one kills
        detached children used by the bridge and Claude launcher.

        Daytona can keep a session command in RUNNING forever when the foreground
        shell exits after starting a background daemon. The wrapper writes the
        foreground exit code atomically, so completion does not depend on Daytona's
        descendant-aware command status.
        """
        import asyncio
        import random
        import uuid

        from daytona import SessionExecuteRequest

        # Pass every optional field explicitly (= the SDK's own defaults). In the
        # torchtitan/vllm import context the SDK rebuild drops these defaults.
        command_key = uuid.uuid4().hex
        observed_full, status_path, output_path = _build_observable_exec_command(
            full, command_key
        )
        request = SessionExecuteRequest(
            command=observed_full,
            run_async=True,
            var_async=None,
            suppress_input_echo=None,
            # Older Toolbox deployments use the deprecated JSON field while the
            # current SDK emits runAsync. Sending both true is unambiguous.
            additional_properties={"async": True},
        )

        # Retrying creation of an empty session is safe. Command submission below
        # is never replayed because the command may have started before a response
        # was lost. The old knob remains an alias for compatibility.
        retries = int(
            _getenv(
                "TT_DAYTONA_SESSION_CREATE_RETRIES",
                "TT_DAYTONA_EXEC_RETRIES",
                default="5",
            )
        )
        if retries < 0:
            raise ValueError(
                f"TT_DAYTONA_SESSION_CREATE_RETRIES must be non-negative, got {retries}"
            )
        backoff = 5.0
        sid = ""
        for attempt in range(retries + 1):
            sid = uuid.uuid4().hex
            try:
                await asyncio.wait_for(
                    self._sb.process.create_session(sid),
                    timeout=_SESSION_RPC_TIMEOUT_SEC,
                )
                break
            except Exception as e:
                if _is_sandbox_gone_error(e):
                    self._mark_sandbox_lost(e, phase="session_create")
                    raise
                try:
                    await asyncio.wait_for(
                        self._sb.process.delete_session(sid),
                        timeout=_SESSION_RPC_TIMEOUT_SEC,
                    )
                except Exception:
                    pass
                # Toolbox has already failed to create its session directory.
                # Retrying with a fresh UUID cannot free blocks or inodes, and the
                # Daytona daemon may retain the shell it started before mkdir failed.
                if "no space left on device" in str(e).lower():
                    self._record_issue(
                        "session_disk_exhausted",
                        phase="session_create",
                        error=e,
                        session_id=sid,
                        attempt=attempt + 1,
                        max_attempts=retries + 1,
                    )
                    raise
                terminal = attempt >= retries or not _is_transient_rpc_error(e)
                self._record_issue(
                    "session_create_failed" if terminal else "session_create_retry",
                    phase="session_create",
                    error=e,
                    recovered=not terminal,
                    session_id=sid,
                    attempt=attempt + 1,
                    max_attempts=retries + 1,
                    emit_log=terminal,
                )
                if terminal:
                    raise
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 60.0)

        loop = asyncio.get_running_loop()
        deadline = (
            loop.time()
            + command_timeout
            + _COMMAND_KILL_GRACE_SEC
            + _SESSION_POLL_GRACE_SEC
        )

        cid = ""
        try:
            resp = await self._sb.process.execute_session_command(
                sid,
                request,
                timeout=request_timeout,
            )
            cid = resp.cmd_id or ""
        except Exception as e:
            cid = await self._recover_session_command_id(sid, observed_full)
            if not cid:
                self._record_issue(
                    "execute_response_unconfirmed",
                    phase="execute_submit",
                    error=e,
                    session_id=sid,
                )
                await self._delete_exec_session(
                    sid, reason="command submission could not be confirmed"
                )
                raise
            self._record_issue(
                "execute_response_recovered",
                phase="execute_submit",
                error=e,
                recovered=True,
                session_id=sid,
                command_id=cid,
                emit_log=False,
            )
        if not cid:
            cid = await self._recover_session_command_id(sid, observed_full)
            if not cid:
                self._record_issue(
                    "execute_missing_command_id",
                    phase="execute_submit",
                    error="execute response contained no command id",
                    session_id=sid,
                )
                await self._delete_exec_session(
                    sid, reason="execute response contained no command id"
                )
                raise RuntimeError("daytona session exec returned no cmd_id")

        async def read_status() -> int | None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            self._raise_if_sandbox_lost()
            try:
                raw_status = await asyncio.wait_for(
                    self._sb.fs.download_file(
                        status_path,
                        min(_SESSION_RPC_TIMEOUT_SEC, max(1, int(remaining))),
                    ),
                    timeout=min(float(_SESSION_RPC_TIMEOUT_SEC), remaining),
                )
            except asyncio.TimeoutError as e:
                self._record_issue(
                    "poll_transient",
                    phase="command_poll",
                    error=e,
                    recovered=True,
                    session_id=sid,
                    command_id=cid,
                    emit_log=False,
                )
                return None
            except Exception as e:
                if _is_missing_file_error(e):
                    return None
                if _is_sandbox_gone_error(e):
                    self._mark_sandbox_lost(e, phase="command_poll")
                    raise
                if _is_transient_rpc_error(e):
                    self._record_issue(
                        "poll_transient",
                        phase="command_poll",
                        error=e,
                        recovered=True,
                        session_id=sid,
                        command_id=cid,
                        emit_log=False,
                    )
                    return None
                raise
            if isinstance(raw_status, bytes):
                status_text = raw_status.decode("utf-8", errors="replace").strip()
            else:
                status_text = str(raw_status).strip()
            try:
                return int(status_text)
            except ValueError as e:
                self._record_issue(
                    "command_status_invalid",
                    phase="command_poll",
                    error=f"invalid command status {status_text!r}",
                    session_id=sid,
                    command_id=cid,
                )
                raise RuntimeError(
                    f"daytona command wrote invalid status {status_text!r}"
                ) from e

        exit_code = await read_status()
        polls = 0
        while exit_code is None:
            if loop.time() >= deadline:
                self._record_issue(
                    "command_status_timeout",
                    phase="command_poll",
                    error="command status remained unavailable until the deadline",
                    session_id=sid,
                    command_id=cid,
                )
                raise TimeoutError(
                    "daytona exec status unavailable after "
                    f"{command_timeout + _COMMAND_KILL_GRACE_SEC + _SESSION_POLL_GRACE_SEC:.0f}s "
                    f"without replaying the command; cmd={full[:80]}"
                )
            await asyncio.sleep(0.1 if polls < 5 else 1.0)
            exit_code = await read_status()
            polls += 1

        output = await self._retry_idempotent_rpc(
            lambda: self._sb.fs.download_file(output_path, _SESSION_RPC_TIMEOUT_SEC),
            phase="command_output",
            retry_kind="command_output_retry",
            failed_kind="command_output_failed",
            session_id=sid,
            command_id=cid,
        )
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return exit_code, str(output)

    async def write_file(
        self, sandbox_path: str, content: FileContent, *, user: str = "root"
    ) -> None:
        import os

        parent = os.path.dirname(sandbox_path) or "/"
        await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root", check=False)
        if isinstance(content, Path):
            data = content.read_bytes()
        elif isinstance(content, bytes):
            data = content
        else:
            data = str(content).encode("utf-8")
        # fs.upload_file writes as root; chown afterwards for non-root owners.
        await self._retry_idempotent_rpc(
            lambda: self._sb.fs.upload_file(data, sandbox_path),
            phase="write_file",
            retry_kind="file_upload_retry",
            failed_kind="file_upload_failed",
        )
        if user and user != "root":
            await self.exec(
                f"chown {shlex.quote(user)}:{shlex.quote(user)} {shlex.quote(sandbox_path)}",
                user="root",
                check=False,
            )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        # root can read any file; the user arg is accepted for protocol parity.
        rc, out, _ = await self.exec(
            f"cat {shlex.quote(sandbox_path)}", user="root", timeout=60
        )
        return out if rc == 0 else ""

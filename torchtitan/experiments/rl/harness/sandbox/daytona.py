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

import logging
import shlex
from pathlib import Path
from typing import Any

from torchtitan.experiments.rl.harness.sandbox.base import (
    _getenv,
    ExecResult,
    FileContent,
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
      ``TT_DAYTONA_DISK_GB``               -- disk GiB per sandbox (default 10).
      ``TT_DAYTONA_CREATE_TIMEOUT``        -- snapshot-build/boot wait (default 900s).
      ``TT_DAYTONA_EXEC_TIMEOUT_MIN``       -- SDK request timeout floor; does not
                                               extend the command runtime limit.
      ``TT_DAYTONA_SESSION_CREATE_RETRIES`` -- empty-session creation retries;
                                               defaults to 5.
    """

    api_key_env = ("DAYTONA_API_KEY",)
    api_url_env = ("DAYTONA_API_URL",)
    target_env = ("DAYTONA_TARGET",)

    def __init__(self, image: str, *, timeout: int | None = None, **_ignored) -> None:
        self.image = image
        self.timeout = timeout
        # Daytona is optional and imported lazily, so its SDK types are not
        # available for static annotations in this module.
        self._client: Any = None
        self._sb: Any = None
        self.sandbox_id = ""

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
        disk = int(_getenv("TT_DAYTONA_DISK_GB", default="10"))
        create_timeout = float(_getenv("TT_DAYTONA_CREATE_TIMEOUT", default="900"))
        # Cloud-side TTL so an orphan (left by a SIGKILL'd run that never reached
        # __aexit__, e.g. MAST preemption) self-reaps: once it goes idle it
        # auto-stops after auto_stop minutes, then auto-deletes immediately
        # (auto_delete=0). A live rollout keeps the sandbox active (the host polls
        # its fs continuously via the bridge), so it is never stopped mid-run.
        auto_stop = int(_getenv("TT_DAYTONA_AUTO_STOP_MIN", default="10"))
        auto_delete = int(_getenv("TT_DAYTONA_AUTO_DELETE_MIN", default="0"))
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
                if attempt >= retries:
                    raise
                logger.warning(
                    "daytona create failed (attempt %d/%d): %s",
                    attempt + 1,
                    retries + 1,
                    e,
                )
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 60.0)
        self.sandbox_id = self._sb.id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Delete only this sandbox; never close the process-wide shared client --
        # other concurrent rollouts are still using its pooled connections.
        try:
            if self._sb is not None:
                await self._client.delete(self._sb)
        except Exception as e:
            logger.warning("daytona delete %s failed: %s", self.sandbox_id[:8], e)

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
        rc, out = await self._session_exec(
            full,
            command_timeout=timeout,
            request_timeout=request_timeout,
        )
        # Match Open-Instruct's backend convention: GNU timeout reserves 124 for
        # the synthetic timeout diagnostic. Forced SIGKILL remains raw 137.
        err = f"Command timed out after {timeout}s." if rc == 124 else ""
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
            logger.warning(
                "daytona session cleanup failed (session=%s, reason=%s): %s",
                sid[:8],
                reason,
                e,
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
            except Exception:
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
        """Submit one command, recover an ambiguous response, then poll it.

        Successful sessions are intentionally retained: deleting one kills
        detached children used by the bridge and Claude launcher.
        """
        import asyncio
        import random
        import uuid

        from daytona import SessionExecuteRequest

        # Pass every optional field explicitly (= the SDK's own defaults). In the
        # torchtitan/vllm import context the SDK rebuild drops these defaults.
        request = SessionExecuteRequest(
            command=full,
            run_async=True,
            var_async=None,
            suppress_input_echo=None,
            additional_properties={},
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
                try:
                    await asyncio.wait_for(
                        self._sb.process.delete_session(sid),
                        timeout=_SESSION_RPC_TIMEOUT_SEC,
                    )
                except Exception:
                    pass
                if attempt >= retries:
                    raise
                logger.warning(
                    "daytona session create failed (attempt %d/%d): %s",
                    attempt + 1,
                    retries + 1,
                    e,
                )
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 60.0)

        cid = ""
        try:
            resp = await self._sb.process.execute_session_command(
                sid,
                request,
                timeout=request_timeout,
            )
            cid = resp.cmd_id or ""
        except Exception:
            cid = await self._recover_session_command_id(sid, full)
            if not cid:
                await self._delete_exec_session(
                    sid, reason="command submission could not be confirmed"
                )
                raise
            logger.warning(
                "daytona recovered command after lost execute response "
                "(session=%s, command=%s)",
                sid[:8],
                cid,
            )
        if not cid:
            cid = await self._recover_session_command_id(sid, full)
            if not cid:
                await self._delete_exec_session(
                    sid, reason="execute response contained no command id"
                )
                raise RuntimeError("daytona session exec returned no cmd_id")

        loop = asyncio.get_running_loop()
        deadline = (
            loop.time()
            + command_timeout
            + _COMMAND_KILL_GRACE_SEC
            + _SESSION_POLL_GRACE_SEC
        )

        async def poll():
            # Daytona briefly returns an empty exit_code mid-command; the SDK then
            # raises "convert exit code to int". Treat that as "still running".
            # A single status query returns in seconds, so wrap it in a hard per-call
            # cap: a hung host->daytona HTTP call would otherwise block here forever
            # (the outer deadline is only checked between polls), stranding this
            # rollout and, under strict-FIFO batching, its whole group. On a hung
            # call, report "still running" so the outer deadline fires the clean
            # TimeoutError path instead.
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                return await asyncio.wait_for(
                    self._sb.process.get_session_command(sid, cid),
                    timeout=min(float(_SESSION_RPC_TIMEOUT_SEC), remaining),
                )
            except asyncio.TimeoutError:
                return None
            except Exception as e:
                msg = str(e)
                # "convert exit code": Daytona briefly returns an empty exit_code
                # mid-command -> treat as still running.
                # Transient host->daytona disconnects (e.g. DaytonaConnectionError
                # "Server disconnected") during a poll must NOT discard the whole
                # multi-turn rollout on one network hiccup: treat them as still
                # running so the poll loop retries (bounded by the outer deadline,
                # which fires a clean TimeoutError if the sandbox is truly gone).
                low = msg.lower()
                if (
                    "convert exit code" in low
                    or "disconnect" in low
                    or "connection" in low
                    or "timeout" in low
                ):
                    return None
                raise

        cmd = await poll()
        polls = 0
        while cmd is None or cmd.exit_code is None:
            if loop.time() >= deadline:
                raise TimeoutError(
                    "daytona exec status unavailable after "
                    f"{command_timeout + _COMMAND_KILL_GRACE_SEC + _SESSION_POLL_GRACE_SEC:.0f}s "
                    f"without replaying the command; cmd={full[:80]}"
                )
            await asyncio.sleep(0.1 if polls < 5 else 1.0)
            cmd = await poll()
            polls += 1

        logs = await asyncio.wait_for(
            self._sb.process.get_session_command_logs(sid, cid),
            timeout=_SESSION_RPC_TIMEOUT_SEC,
        )
        out = getattr(logs, "stdout", "") or ""
        err = getattr(logs, "stderr", "") or ""
        exit_code = int(cmd.exit_code)
        return exit_code, out + err

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
        await self._sb.fs.upload_file(data, sandbox_path)
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Terminal-Bench's Terminus-2 scaffold as a swappable harness.

Terminus-2 is a materially different agent from our other harnesses, which is the
point of having it: it drives a live tmux pane with *batches* of raw keystrokes
and observes the screen, rather than issuing one bash command and reading its
stdout. That buys interactive programs (answering a prompt, ``C-c`` on a hung
process, typing into gdb/vim, or waiting without acting) and packs several
commands into one model turn -- which is why published Terminus-2 turn counts run
far below ours on the same tasks.

Two seams make it work without vendoring the agent:

  - ``_SandboxEnvironment`` presents our ``Sandbox`` as the five members
    Terminus-2 touches (measured, not guessed: ``exec`` / ``upload_file`` /
    ``download_file`` / ``is_dir`` / ``default_user`` plus
    ``trial_paths.agent_dir``).
  - Terminus-2 reaches its model through LiteLLM, so it is pointed at the
    adapter's Anthropic endpoint with the session id as the API key. Turns land in
    the adapter's capture exactly as they do for the in-process harnesses.

CAVEAT: the model has to emit Terminus-2's XML (``<response><analysis><plan>
<commands><keystrokes>``). A policy trained under a tool-calling scaffold is off
distribution here, so check ``format_errors`` before reading anything into a
reward: sparse binary reward cannot teach a new output format.
"""

from __future__ import annotations

import logging
import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torchtitan.experiments.rl.harness.agents.spec import (
    AgentRun,
    AgentTask,
    register_agent,
)

logger = logging.getLogger(__name__)

# Terminus-2 asks LiteLLM for an Anthropic-provider call; the model name after the
# prefix is arbitrary (the adapter serves whatever session the key names).
_LITELLM_MODEL = os.environ.get("TMAX_TERMINUS_MODEL", "anthropic/titan-actor")
_PARSER = os.environ.get("TMAX_TERMINUS_PARSER", "xml")
# Terminus-2 batches commands per turn, so it needs far fewer than a one-command-
# per-turn scaffold; this only bounds a runaway loop.
_DEFAULT_MAX_TURNS = int(os.environ.get("TMAX_TERMINUS_MAX_TURNS", "64"))


class _SandboxEnvironment:
    """Our ``Sandbox`` in the shape Terminus-2 expects of a harbor environment."""

    def __init__(self, sandbox: Any, *, agent_dir: Path, user: str = "root") -> None:
        self._sandbox = sandbox
        self.default_user = user
        # Terminus-2 only reads ``trial_paths.agent_dir``, and only to place its
        # asciinema recording -- which we leave off.
        self.trial_paths = _TrialPaths(agent_dir=agent_dir)
        self.environment_dir = agent_dir
        self.environment_name = "titan-sandbox"
        self.session_id = ""

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ):
        from harbor.environments.base import ExecResult  # type: ignore

        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        exit_code, stdout, stderr = await self._sandbox.exec(
            command,
            user=str(user or self.default_user),
            env=env,
            check=False,
            **({"timeout": timeout_sec} if timeout_sec else {}),
        )
        return ExecResult(stdout=stdout, stderr=stderr, return_code=exit_code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        with open(source_path, "rb") as f:
            await self._sandbox.write_file(
                target_path, f.read(), user=self.default_user
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        content = await self._sandbox.read_file(source_path, user=self.default_user)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_text(content)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        exit_code, _out, _err = await self._sandbox.exec(
            f"test -d {shlex.quote(path)}",
            user=str(user or self.default_user),
            check=False,
            timeout=60,
        )
        return exit_code == 0


@dataclass(frozen=True, slots=True)
class _TrialPaths:
    """The one path Terminus-2 reads off the environment."""

    agent_dir: Path


async def terminus_agent(task: AgentTask) -> AgentRun:
    """Drive Terminus-2 against the task's sandbox and the adapter's policy."""
    from harbor.agents.terminus_2 import Terminus2  # type: ignore
    from harbor.models.agent.context import AgentContext  # type: ignore

    # LiteLLM reads the Anthropic key from the environment; the adapter uses it to
    # pick the session, so every rollout must scope it to its own id.
    previous_key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = task.session_id

    submitted = False
    turns = 0
    finish_reason = "unknown"
    with tempfile.TemporaryDirectory(prefix="tt-terminus-") as logs_dir:
        env = _SandboxEnvironment(task.sandbox, agent_dir=Path(logs_dir))
        try:
            max_episodes = task.max_turns or _DEFAULT_MAX_TURNS
            agent = Terminus2(
                logs_dir=Path(logs_dir),
                model_name=_LITELLM_MODEL,
                api_base=task.adapter.url,
                parser_name=_PARSER,
                record_terminal_session=False,
                max_turns=max_episodes,
                suppress_max_turns_warning=True,
            )
            context = AgentContext()
            await agent.setup(env)
            await agent.run(task.instruction, env, context)
            turns = int(getattr(agent, "_n_episodes", 0) or 0)
            # Terminus-2 returns early only once <task_complete>true</task_complete>
            # has been confirmed on a second consecutive turn; otherwise it runs the
            # loop out. So "ended before the cap" IS the submit signal.
            submitted = 0 < turns < max_episodes
            finish_reason = "submit" if submitted else "hit_max_turns"
        except Exception as e:
            logger.warning(
                "[terminus] session=%s failed: %s: %s",
                task.session_id,
                type(e).__name__,
                str(e)[:200],
            )
            finish_reason = "error"
        finally:
            if previous_key is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = previous_key

    return AgentRun(turns=turns, submitted=submitted, finish_reason=finish_reason)


register_agent("terminus", terminus_agent)

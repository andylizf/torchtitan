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
  - ``_AdapterLLM`` replaces Terminus-2's LiteLLM backend with a direct
    ``adapter.complete`` call. The adapter deliberately does NOT run an HTTP
    server on the tmax path (no loopback hop, no per-worker port), and rollout
    workers are separate processes, so ``adapter.url`` resolves to nothing there --
    pointing LiteLLM at it fails every rollout with "Cannot connect to host".
    Calling in-process also keeps turn capture on the same path the other
    harnesses use.

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

_PARSER = os.environ.get("TMAX_TERMINUS_PARSER", "xml")
# Terminus-2 batches commands per turn, so it needs far fewer than a one-command-
# per-turn scaffold; this only bounds a runaway loop.
_DEFAULT_MAX_TURNS = int(os.environ.get("TMAX_TERMINUS_MAX_TURNS", "64"))
# Terminus-2 asks the backend for a context limit to decide when to summarize.
_MAX_CONTEXT = int(os.environ.get("SWE_MAX_CONTEXT_LEN", "63488"))
# Per-turn generation cap; the adapter clamps it to the remaining context budget.
_TURN_MAX_TOKENS = int(os.environ.get("TMAX_TURN_MAX_TOKENS", "16384"))


class _AdapterExhausted(RuntimeError):
    """The adapter has no completion left for this session."""


class _AdapterLLM:
    """Terminus-2's LLM seam, backed by ``AnthropicAdapter.complete`` in-process.

    Terminus-2 only ever does ``await llm.call(prompt=..., message_history=...)``
    plus the two limit getters, so this is the whole surface. The adapter speaks
    Anthropic messages, which is also what it captures for training.
    """

    def __init__(
        self, adapter: Any, *, session_id: str, max_context: int, turn_max_tokens: int
    ) -> None:
        self._adapter = adapter
        self._session_id = session_id
        self._max_context = max_context
        self._turn_max_tokens = turn_max_tokens

    def get_model_context_limit(self) -> int:
        return self._max_context

    def get_model_output_limit(self) -> int | None:
        # Terminus-2 puts this number in the retry it sends after a truncated turn
        # ("you exceeded N tokens, break it into chunks"). None degrades that to
        # "the maximum output length", which the model cannot act on.
        return self._turn_max_tokens

    async def call(self, prompt: str, message_history=None, **_kwargs):
        from harbor.llms.base import (  # type: ignore
            LLMResponse,
            OutputLengthExceededError,
        )

        messages = list(message_history or [])
        messages.append({"role": "user", "content": prompt})
        reply = await self._adapter.complete(
            self._session_id,
            {
                "messages": messages,
                "max_tokens": self._turn_max_tokens,
                "stream": False,
            },
        )
        if reply is None:
            # The session is closed or the generator yielded nothing; ending the
            # trajectory is what the other harnesses do here too.
            raise _AdapterExhausted(
                f"adapter returned no completion for {self._session_id}"
            )
        text = "".join(
            block.get("text", "")
            for block in (reply.get("content") or [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        # A turn cut off at max_tokens has to be raised, not returned. Terminus-2
        # handles it inside its LLM call -- salvage a complete action out of the
        # truncated text, else re-ask for a shorter one -- and neither step costs an
        # episode. Returned as an ordinary reply it instead reaches the XML parser,
        # fails there, and burns an episode on the parser-warning retry. Both of
        # harbor's own backends raise here for the same reason.
        #
        # The adapter reports "max_tokens" for two different things, and only one of
        # them is a truncation: a generation that ran into the per-turn cap comes back
        # with output_tokens == cap, while a prompt that no longer fits the context
        # comes back with output_tokens == 0 and an empty completion. Re-asking the
        # second one is unbounded -- the retry appends to the history that is already
        # over budget, so it returns empty again -- so let it through as an empty
        # reply and end the trajectory, which is what it means.
        if (
            reply.get("stop_reason") in ("max_tokens", "length")
            and (reply.get("usage") or {}).get("output_tokens", 0) > 0
        ):
            raise OutputLengthExceededError(
                f"hit max_tokens={self._turn_max_tokens} for {self._session_id}",
                truncated_response=text,
            )
        return LLMResponse(content=text)


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
        if exit_code != 0:
            # Terminus-2 surfaces a tmux failure as "Failed to start tmux session.
            # Error: <stderr>", which says nothing when the provider returns
            # non-zero with an empty stderr. Log what actually ran so the failure
            # can be attributed instead of guessed at.
            logger.warning(
                "[terminus] exec exit=%d cmd=%r stdout=%r stderr=%r",
                exit_code,
                command[:400],
                (stdout or "")[-400:],
                (stderr or "")[-400:],
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


def _episodes(agent: Any) -> int:
    """Episodes Terminus-2 entered, 0 before its loop starts or if it never ran."""
    return int(getattr(agent, "_n_episodes", 0) or 0)


async def terminus_agent(task: AgentTask) -> AgentRun:
    """Drive Terminus-2 against the task's sandbox and the adapter's policy."""
    from harbor.agents.terminus_2 import Terminus2  # type: ignore
    from harbor.models.agent.context import AgentContext  # type: ignore

    submitted = False
    turns = 0
    finish_reason = "unknown"
    agent = None
    with tempfile.TemporaryDirectory(prefix="tt-terminus-") as logs_dir:
        env = _SandboxEnvironment(task.sandbox, agent_dir=Path(logs_dir))
        try:
            max_episodes = task.max_turns or _DEFAULT_MAX_TURNS
            agent = Terminus2(
                logs_dir=Path(logs_dir),
                model_name="titan-actor",
                parser_name=_PARSER,
                record_terminal_session=False,
                max_turns=max_episodes,
                suppress_max_turns_warning=True,
            )
            # Swap the LiteLLM backend for the in-process adapter before setup;
            # Terminus-2 only reads self._llm through ``call`` and the limit getters.
            agent._llm = _AdapterLLM(
                task.adapter,
                session_id=task.session_id,
                max_context=_MAX_CONTEXT,
                turn_max_tokens=_TURN_MAX_TOKENS,
            )
            context = AgentContext()
            # tmux bring-up rides on DaytonaSandbox's own session-create retry;
            # no retry loop here.
            await agent.setup(env)
            await agent.run(task.instruction, env, context)
            turns = _episodes(agent)
            # Terminus-2's loop has THREE exits, and only one of them is a submit:
            # it runs the episodes out; it returns early on a confirmed
            # <task_complete>true</task_complete> (the second consecutive one, at
            # which point ``_pending_completion`` is still set); or it returns early
            # because ``is_session_alive()`` went false, i.e. the tmux session died
            # under it. Reading "ended before the cap" as the submit signal folds that
            # third case into "submit" and scores a dead session as a real attempt.
            if turns >= max_episodes:
                finish_reason = "hit_max_turns"
            elif getattr(agent, "_pending_completion", False):
                finish_reason = "submit"
            else:
                finish_reason = "stopped_early"
            submitted = finish_reason == "submit"
        except Exception as e:
            logger.warning(
                "[terminus] session=%s failed: %s: %s",
                task.session_id,
                type(e).__name__,
                str(e)[:200],
            )
            # Keep the episodes the agent did get through; the turns it captured are
            # still trained on, so reporting 0 here misattributes them.
            turns = _episodes(agent)
            finish_reason = "error"

    return AgentRun(turns=turns, submitted=submitted, finish_reason=finish_reason)


register_agent("terminus", terminus_agent)

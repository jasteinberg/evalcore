"""Tools that run inside a container, one container per episode.

A shell is a stateful tool: what `ls` returns depends on the commands run
before it in the same episode.  Two consequences shape this module.

  * Each episode (one unit) gets its own container, started before the
    model's first turn and removed when the episode ends, however it ends.
    Episodes never see each other's files, and concurrent workers never
    share a container.
  * A sandbox tool's results are never cached on its arguments, since the
    same command can return different output at different points of an
    episode.  The episode as a whole is still cached (the agent's
    response, under its request key), so a resumed run does not replay it.

    sandbox = DockerSandbox("python:3.12-slim", network="none")
    agent = AgentBackend(inner, [shell_tool()], sandbox=sandbox)

`DockerSandbox` drives the `docker` command line, so it adds no Python
dependency.  The isolation is the container's: no network unless asked
for, memory and CPU limits, a command timeout.  It is not a hardened
sandbox (gVisor, network policy) for code that may try to escape; use one
of those for untrusted agents.

Failures follow tools.py.  A container that cannot be started or a docker
daemon that stops answering is the infrastructure's (ToolUnavailable, a
`tool_error` row).  A command that fails or times out is the model's to
handle: its exit status and output go back to it as the tool's result.
"""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .tools import Tool, ToolContext, ToolUnavailable

__all__ = ["DockerSandbox", "ExecResult", "SandboxSession", "shell_tool"]

Run = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class ExecResult:
    """One command's outcome.  `timed_out` is set when the command was
    killed at the sandbox's timeout: killed (status 124, 137) after at
    least the timeout had elapsed.  An earlier kill with status 137 is the
    memory limit's, and is reported as the exit status it is."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class SandboxSession(Protocol):
    """One episode's container."""

    def exec(self, command: str) -> ExecResult: ...

    def close(self) -> None: ...


class Sandbox(Protocol):
    def identity(self) -> dict[str, Any]: ...

    def start(self, ctx: ToolContext) -> SandboxSession: ...


def _docker(run: Run, args: Sequence[str], timeout: float,
            what: str) -> subprocess.CompletedProcess[str]:
    """`docker <args>`, or ToolUnavailable if docker itself fails (no
    daemon, no image, a hung daemon)."""
    try:
        p = run(["docker", *args], capture_output=True, text=True,
                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolUnavailable(f"docker {what}: {exc}") from exc
    return p


class DockerSandbox:
    """A fresh container per episode from `image`.

    `image` is resolved to its content digest once, when the identity is
    first asked for, so the request key changes when the tag is moved to
    other contents.  The image is not pulled: pull it before the run, so
    that a sweep does not depend on a registry mid-run.  It must provide
    `sh` and `timeout` (coreutils or busybox), which are checked at each
    container's start.

    `network="none"` (the default) gives the container no network.
    `memory` and `cpus` are docker's `--memory` and `--cpus`.  `timeout`
    bounds each command, in seconds, and `setup` lists shell commands run
    once in each new container before the model's first turn (copying a
    task's files, installing a package)."""

    def __init__(self, image: str, *, network: str = "none",
                 memory: str = "1g", cpus: float = 1.0, timeout: float = 60.0,
                 workdir: str = "/work", setup: Sequence[str] = (),
                 run: Run = subprocess.run) -> None:
        self.image, self.network, self.memory = image, network, memory
        self.cpus, self.timeout, self.workdir = cpus, timeout, workdir
        self.setup = tuple(setup)
        self._run = run
        self._digest: str | None = None

    def digest(self) -> str:
        """The image's content id (`docker image inspect`), read once."""
        if self._digest is None:
            p = _docker(self._run, ["image", "inspect", "--format", "{{.Id}}",
                                    self.image], 30, "image inspect")
            if p.returncode != 0 or not p.stdout.strip():
                raise ToolUnavailable(
                    f"image {self.image!r} is not present locally (pull it "
                    f"before the run): {p.stderr.strip()}")
            self._digest = p.stdout.strip()
        return self._digest

    def identity(self) -> dict[str, Any]:
        return {"sandbox": "docker", "image": self.image,
                "digest": self.digest(), "network": self.network,
                "memory": self.memory, "cpus": self.cpus,
                "timeout": self.timeout, "workdir": self.workdir,
                "setup": list(self.setup)}

    def start(self, ctx: ToolContext) -> _DockerSession:
        name = f"evalcore-{uuid.uuid4().hex[:12]}"
        p = _docker(self._run, [
            "run", "-d", "--rm", "--name", name, "--network", self.network,
            "--memory", self.memory, "--cpus", str(self.cpus),
            "--workdir", self.workdir, "--label",
            f"evalcore.unit={ctx.unit_id}", self.digest(),
            "sh", "-c", f"mkdir -p {shlex.quote(self.workdir)} && "
                        f"exec sleep infinity"], 120, "run")
        if p.returncode != 0:
            raise ToolUnavailable(f"container did not start: "
                                  f"{p.stderr.strip()}")
        s = _DockerSession(self, name)
        try:
            check = s.exec("command -v timeout >/dev/null")
            if check.exit_code != 0:
                raise ToolUnavailable(f"image {self.image!r} has no "
                                      f"`timeout` command")
            for cmd in self.setup:
                r = s.exec(cmd)
                if r.exit_code != 0:
                    raise ToolUnavailable(
                        f"setup command {cmd!r} failed ({r.exit_code}): "
                        f"{r.stderr.strip()[:500]}")
        except BaseException as exc:
            try:
                s.close()
            except ToolUnavailable as gone:
                raise ToolUnavailable(f"{exc}, and then {gone}") from exc
            raise
        return s


class _DockerSession:
    def __init__(self, box: DockerSandbox, name: str) -> None:
        self.box, self.name, self.closed = box, name, False

    def exec(self, command: str) -> ExecResult:
        t = self.box.timeout
        # `timeout` inside the container stops the command itself.  The
        # client-side timeout is only a backstop for a hung daemon.
        t0 = time.monotonic()
        p = _docker(self.box._run, [
            "exec", self.name, "timeout", "-s", "KILL", f"{t:g}",
            "sh", "-c", command], t + 30, "exec")
        elapsed = time.monotonic() - t0
        return ExecResult(p.returncode, p.stdout, p.stderr,
                          timed_out=p.returncode in (124, 137)
                          and elapsed >= t)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        p = _docker(self.box._run, ["rm", "-f", self.name], 60, "rm")
        if p.returncode != 0:
            raise ToolUnavailable(f"container {self.name} was not removed: "
                                  f"{p.stderr.strip()}")


def shell_tool(max_output: int = 10_000, name: str = "shell") -> Tool:
    """A tool that runs one shell command in the episode's container and
    returns its exit status, stdout and stderr, each cut to `max_output`
    characters (the model is told when).  A timed-out command is reported
    to the model as such."""

    def fn(arguments: dict[str, Any], session: SandboxSession) -> str:
        cmd = arguments["command"]
        if not isinstance(cmd, str):
            from .tools import ToolInputError
            raise ToolInputError(f"{name}: command must be a string")
        r = session.exec(cmd)

        def cut(s: str) -> str:
            if len(s) <= max_output:
                return s
            return (s[:max_output] + f"\n[... {len(s) - max_output} more "
                    f"characters not shown]")
        head = ("timed out (the command was stopped)" if r.timed_out
                else f"exit status {r.exit_code}")
        return f"{head}\n[stdout]\n{cut(r.stdout)}\n[stderr]\n{cut(r.stderr)}"

    return Tool(name, "Run a shell command (sh -c) in this task's container. "
                "Files persist between calls within the task.",
                {"type": "object",
                 "properties": {"command": {"type": "string",
                                            "description": "the command"}},
                 "required": ["command"]},
                fn, version=f"1:{max_output}", uses_sandbox=True)

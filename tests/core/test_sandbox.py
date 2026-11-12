"""Sandboxed tools: one container per episode, removed however the episode
ends; state persists within an episode and never across; failures of the
container are infrastructure (`tool_error`), failures of commands are the
model's.

A scripted `docker` command line stands in for the daemon.  It keeps each
container's files, so persistence is checked by what a later command
reads, and a clock the test controls, so a timeout and a memory kill
(both exit status 137) are told apart by elapsed time.  One test runs the
real docker CLI where it exists."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from evalcore import FunctionBackend, Item, Request, Response, execute, grid, units
from evalcore.core import sandbox as sandbox_mod
from evalcore.core.agent import AgentBackend
from evalcore.core.sandbox import DockerSandbox, shell_tool
from evalcore.core.tools import Tool, ToolContext, ToolUnavailable

TIMEOUT = 5.0


class FakeDocker:
    """The docker CLI as far as DockerSandbox uses it.  Inside a container,
    `echo TEXT > FILE`, `cat FILE`, `exit N`, `sleep forever` (runs past
    any timeout) and `oom` (killed by the memory limit at once) are
    understood."""

    def __init__(self, images: dict[str, str] | None = None,
                 start_fails: bool = False, rm_fails: bool = False,
                 has_timeout: bool = True) -> None:
        self.images = {"img:1": "sha256:aaa"} if images is None else images
        self.start_fails, self.rm_fails = start_fails, rm_fails
        self.has_timeout = has_timeout
        self.live: dict[str, dict[str, str]] = {}
        self.started: list[list[str]] = []
        self.removed: list[str] = []
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def __call__(self, argv: list[str], **kw: Any) -> Any:  # noqa: PLR0911 - one branch per command
        assert argv[0] == "docker"
        cmd, args = argv[1], argv[2:]
        if cmd == "image":
            image = args[-1]
            if image not in self.images:
                return self._done(argv, 1, err=f"No such image: {image}")
            return self._done(argv, 0, self.images[image] + "\n")
        if cmd == "run":
            if self.start_fails:
                return self._done(argv, 125, err="cannot connect to daemon")
            self.started.append(argv)
            self.live[args[args.index("--name") + 1]] = {}
            return self._done(argv, 0, "cid\n")
        if cmd == "rm":
            name = args[-1]
            if self.rm_fails:
                return self._done(argv, 1, err="daemon gone")
            self.live.pop(name)
            self.removed.append(name)
            return self._done(argv, 0)
        if cmd == "exec":
            name, line = args[0], args[-1]
            assert args[1:4] == ["timeout", "-s", "KILL"]
            assert args[-3:-1] == ["sh", "-c"]
            if name not in self.live:
                return self._done(argv, 1, err=f"No such container: {name}")
            return self._exec(argv, self.live[name], line)
        raise AssertionError(argv)

    def _exec(self, argv: list[str], files: dict[str, str], line: str) -> Any:  # noqa: PLR0911 - one branch per command
        if line.startswith("command -v timeout"):
            return self._done(argv, 0 if self.has_timeout else 1)
        if m := re.fullmatch(r"echo (.*) > (\S+)", line):
            files[m[2]] = m[1] + "\n"
            return self._done(argv, 0)
        if m := re.fullmatch(r"cat (\S+)", line):
            if m[1] in files:
                return self._done(argv, 0, files[m[1]])
            return self._done(argv, 1, err=f"cat: {m[1]}: No such file")
        if m := re.fullmatch(r"exit (\d+)", line):
            return self._done(argv, int(m[1]))
        if line == "sleep forever":
            self.now += TIMEOUT
            return self._done(argv, 137)
        if line == "oom":
            return self._done(argv, 137)
        if m := re.fullmatch(r"yes (\d+)", line):
            return self._done(argv, 0, "y" * int(m[1]))
        raise AssertionError(line)

    @staticmethod
    def _done(argv: list[str], code: int, out: str = "", err: str = "") -> Any:
        return subprocess.CompletedProcess(argv, code, out, err)


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    d = FakeDocker()
    monkeypatch.setattr(sandbox_mod.time, "monotonic", d.clock)
    return d


def box(d: FakeDocker, **kw: Any) -> DockerSandbox:
    return DockerSandbox("img:1", timeout=TIMEOUT, run=d, **kw)


def commands(*cmds: str, answer: str = "done") -> Callable[[Request], Response]:
    """A model that runs `cmds` in order, one per turn, then answers with
    every tool result it saw, joined."""
    def policy(req: Request) -> Response:
        seen = [m["content"] for m in req.messages if m.get("role") == "tool"]
        if len(seen) < len(cmds):
            return Response("x", meta={"tool_calls": [{
                "id": f"c{len(seen)}", "name": "shell",
                "arguments": {"command": cmds[len(seen)]}}]})
        return Response("x", text=answer + "\n" + "\n---\n".join(seen),
                        meta={"truncated": False})
    return policy


def agent(policy: Callable[[Request], Any], sb: DockerSandbox,
          **kw: Any) -> AgentBackend:
    inner = FunctionBackend(policy, identity={"policy": "p"},
                            supports_tools=True)
    return AgentBackend(inner, [shell_tool()], sandbox=sb, **kw)


def episode(be: AgentBackend, uid: str = "u") -> Response:
    return be.complete(Request(uid, [{"role": "user", "content": "q"}], {}))


# --- one container per episode ---------------------------------------------

def test_files_persist_within_an_episode_and_not_across(docker):
    be = agent(commands("echo hello > f", "cat f"), box(docker))
    first = episode(be, "u1")
    assert "exit status 0\n[stdout]\nhello" in first.text
    second = agent(commands("cat f"), box(docker))
    r = episode(second, "u2")
    assert "exit status 1" in r.text and "No such file" in r.text
    assert len(docker.started) == 2 and len(set(docker.removed)) == 2
    assert docker.live == {}


def test_the_container_is_started_isolated_from_the_digest(docker):
    episode(agent(commands("exit 0"), box(docker, memory="512m", cpus=0.5)),
            "unit-7")
    argv = docker.started[0]

    def opt(flag: str) -> str:
        return argv[argv.index(flag) + 1]
    assert (opt("--network"), opt("--memory"), opt("--cpus")) == (
        "none", "512m", "0.5")
    assert "sha256:aaa" in argv and "img:1" not in argv
    assert opt("--label") == "evalcore.unit=unit-7"


def test_the_container_is_removed_when_the_model_fails(docker):
    def fails(req: Request) -> Response:
        if any(m.get("role") == "tool" for m in req.messages):
            return Response("x", error="fatal: model refused")
        return commands("exit 0")(req)
    r = episode(agent(fails, box(docker)))
    assert r.error and docker.live == {} and len(docker.removed) == 1


def test_the_container_is_removed_when_a_tool_has_a_bug(docker):
    def broken(arguments: dict, session: Any) -> str:
        raise KeyError("bug in the tool")
    tool = Tool("shell", "d", {"type": "object", "properties": {
        "command": {"type": "string"}}}, broken, uses_sandbox=True)
    inner = FunctionBackend(commands("exit 0"), identity={"p": 1},
                            supports_tools=True)
    be = AgentBackend(inner, [tool], sandbox=box(docker))
    with pytest.raises(KeyError):
        episode(be)
    assert docker.live == {} and len(docker.removed) == 1


# --- whose failure ------------------------------------------------------------

def test_a_container_that_cannot_start_is_a_tool_error_row(tmp_path):
    d = FakeDocker(start_fails=True)
    df = execute(units(grid({"c": ["a"]}), [Item("q0", {})]),
                 agent(commands("exit 0"), box(d)),
                 lambda u: Request(u.id, [{"role": "user", "content": "q"}],
                                   {}),
                 lambda u, r: {}, tmp_path / "r.jsonl", workers=1,
                 stream=io.StringIO())
    assert df.iloc[0]["status"] == "tool_error"
    assert "daemon" in df.iloc[0]["error"]


def test_an_image_without_timeout_or_a_failed_setup_is_a_tool_error(docker):
    no_timeout = FakeDocker(has_timeout=False)
    r = episode(agent(commands("exit 0"), box(no_timeout)))
    assert r.meta["tool_error"] and "timeout" in r.error
    assert no_timeout.live == {}
    r = episode(agent(commands("exit 0"), box(docker, setup=["exit 3"])))
    assert r.meta["tool_error"] and "setup" in r.error and docker.live == {}


def test_an_image_not_present_is_refused_at_identity():
    be = agent(commands("exit 0"),
               DockerSandbox("absent:1", run=FakeDocker()))
    with pytest.raises(ToolUnavailable, match="pull it before the run"):
        be.identity()


def test_a_timeout_and_a_memory_kill_are_told_apart(docker):
    """Both exit 137.  Only the one that ran the full timeout timed out."""
    r = episode(agent(commands("sleep forever", "oom"), box(docker)))
    timed, killed = r.text.split("\n---\n")
    assert timed.splitlines()[1].startswith("timed out")
    assert killed.splitlines()[0] == "exit status 137"


def test_a_failed_removal_is_said_and_the_answer_stands():
    d = FakeDocker(rm_fails=True)
    with pytest.warns(RuntimeWarning, match="not removed"):
        r = episode(agent(commands("exit 0"), box(d)))
    assert r.ok and "not removed" in r.meta["sandbox_close_error"]


# --- caching and identity -----------------------------------------------------

def test_sandbox_results_are_never_cached(docker, tmp_path):
    """The same command, before and after a write, in one episode: two
    different results, even with a tool cache."""
    from evalcore import Cache
    be = agent(commands("cat f", "echo x > f", "cat f"), box(docker),
               tool_cache=Cache(tmp_path / "tc"))
    before, _, after = episode(be).text.split("\n---\n")
    assert "No such file" in before and "[stdout]\nx" in after


def test_identity_covers_the_image_digest_and_limits(docker):
    a = agent(commands("exit 0"), box(docker)).identity()
    other = FakeDocker(images={"img:1": "sha256:bbb"})
    b = agent(commands("exit 0"), box(other)).identity()
    c = agent(commands("exit 0"), box(docker, network="bridge")).identity()
    assert a["sandbox"]["digest"] == "sha256:aaa"
    assert len({str(a), str(b), str(c)}) == 3


def test_an_agent_without_a_sandbox_keeps_its_identity():
    inner = FunctionBackend(lambda r: Response("x", text="t"),
                            identity={"p": 1}, supports_tools=True)
    t = Tool("t", "d", {"type": "object"}, lambda a: "r")
    assert "sandbox" not in AgentBackend(inner, [t]).identity()
    assert "uses_sandbox" not in t.identity()


def test_tools_and_sandbox_must_come_together():
    inner = FunctionBackend(lambda r: Response("x", text="t"),
                            identity={"p": 1}, supports_tools=True)
    with pytest.raises(TypeError, match="pass sandbox="):
        AgentBackend(inner, [shell_tool()])
    plain = Tool("t", "d", {"type": "object"}, lambda a: "r")
    with pytest.raises(TypeError, match="no tool runs in it"):
        AgentBackend(inner, [plain], sandbox=box(FakeDocker()))
    with pytest.raises(ValueError, match="own container"):
        Tool("t", "d", {}, lambda a, s: "r", uses_context=True,
             uses_sandbox=True)


def test_long_output_is_cut_and_the_model_is_told(docker):
    be = AgentBackend(
        FunctionBackend(commands("yes 50"), identity={"p": 1},
                        supports_tools=True),
        [shell_tool(max_output=20)], sandbox=box(docker))
    out = episode(be).text
    assert "y" * 20 + "\n[... 30 more characters not shown]" in out
    assert "y" * 21 not in out


# --- the real docker CLI -------------------------------------------------------

def _docker_works() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True,
                          check=False).returncode == 0


@pytest.mark.skipif(not _docker_works(), reason="no docker daemon")
def test_a_real_container_keeps_files_has_no_network_and_times_out():
    image = "alpine:3.20"
    subprocess.run(["docker", "pull", "-q", image], check=True,
                   capture_output=True)
    sb = DockerSandbox(image, timeout=2)
    s = sb.start(ToolContext("u", "q", 0))
    try:
        assert s.exec("echo hello > f").exit_code == 0
        assert s.exec("cat f").stdout == "hello\n"
        assert s.exec("wget -q -T 2 -O- http://1.1.1.1").exit_code != 0
        assert s.exec("sleep 10").timed_out
    finally:
        s.close()
    gone = subprocess.run(["docker", "ps", "-aq", "--filter",
                           f"name={s.name}"], capture_output=True, text=True,
                          check=True)
    assert gone.stdout.strip() == ""

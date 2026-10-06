"""The contract a user-supplied backend must meet, and the check that says
which part of it a backend breaks."""

from __future__ import annotations

import io

import pytest

from evalcore import (
    Backend,
    EchoBackend,
    FunctionBackend,
    Item,
    Request,
    Response,
    check_backend,
    execute,
    grid,
    units,
)
from evalcore.core.backends import request_key

MSG = [{"role": "user", "content": "hi"}]


def test_function_backend_needs_a_serialisable_identity():
    with pytest.raises(ValueError, match="identity"):
        FunctionBackend(lambda r: "x", identity={})
    with pytest.raises(ValueError):
        FunctionBackend(lambda r: "x", identity={"t": float("nan")})


def test_function_backends_with_different_models_never_share_keys():
    a = FunctionBackend(lambda r: "a", identity={"checkpoint": "ckpt-A"})
    b = FunctionBackend(lambda r: "b", identity={"checkpoint": "ckpt-B"})
    req = Request("u", MSG, {})
    assert request_key(req, 0, a) != request_key(req, 0, b)


def test_function_backend_runs_and_passes_responses_through(tmp_path):
    rich = Response("ignored", text="t", meta={"truncated": True})

    def fn(req):
        return rich if "1" in req.messages[-1]["content"] else "plain"

    be = FunctionBackend(fn, identity={"model": "toy"})
    its = [Item(f"q{i}", {"q": f"item {i}"}) for i in range(2)]
    df = execute(units(grid({"m": ["x"]}), its), be,
             lambda u: Request(u.id, [{"role": "user",
                                       "content": u.item.payload["q"]}], {}),
             lambda u, r: {"t": r.text}, tmp_path / "r.jsonl", workers=1,
             stream=io.StringIO())
    got = df.set_index("item_id")
    assert got.loc["q0", "t"] == "plain"
    assert got.loc["q1", "status"] == "truncated"       # meta passed through


def test_a_conforming_backend_passes():
    rep = check_backend(EchoBackend())
    assert rep.ok, str(rep)
    assert any("truncated" in w for w in rep.warnings)  # echo reports none


class Broken(Backend):
    def __init__(self, fault):
        self.fault = fault

    def identity(self):
        return {"backend": "broken", "fault": self.fault}

    def complete(self, req):
        return Response(req.unit_id, text="x", meta={"truncated": False})

    def complete_batch(self, reqs):
        out = [self.complete(r) for r in reqs]
        if self.fault == "order":
            return out[::-1]
        if self.fault == "count":
            return out[:-1]
        if self.fault == "errors":
            return [Response(r.unit_id, error="fatal: nope") for r in reqs]
        return out


@pytest.mark.parametrize("fault,phrase", [
    ("order", "request order"), ("count", "3 responses for 4"),
    ("errors", "4 of 4 check requests failed")])
def test_each_broken_property_is_named(fault, phrase):
    rep = check_backend(Broken(fault))
    assert not rep.ok
    assert any(phrase in p for p in rep.problems), rep.problems


def test_a_backend_without_identity_is_named():
    class NoId(Backend):
        def complete(self, req):
            return Response(req.unit_id, text="x", meta={"truncated": False})

    rep = check_backend(NoId())
    assert any("NoId must define identity" in p for p in rep.problems)


def test_target_support_is_checked_when_claimed():
    class ClaimsTarget(Broken):
        supports_target = True

    rep = check_backend(ClaimsTarget("none"))
    assert any("no per-token logp" in p for p in rep.problems)

    class Scores(ClaimsTarget):
        def complete_batch(self, reqs):
            return [Response(r.unit_id, meta={"logp": [-0.1],
                                              "truncated": False})
                    if r.target else self.complete(r) for r in reqs]

    assert check_backend(Scores("none")).ok


class ShortOnly(Backend):
    """A backend with a requirement of its own: prompts under 10 characters."""

    name = "short"

    def identity(self):
        return self._base_identity()

    def refusal(self, req):
        if len(req.messages[-1]["content"]) >= 10:
            return "ShortOnly takes prompts under 10 characters"
        return super().refusal(req)

    def complete(self, req):
        self.calls = getattr(self, "calls", 0) + 1
        return Response(req.unit_id, text="ok")


def test_a_backends_own_refusal_is_an_error_row_and_never_a_call(tmp_path):
    its = [Item("a", {"p": "hi"}), Item("b", {"p": "far too long"})]
    be = ShortOnly()
    df = execute(units(grid({"m": [1]}), its), be,
             lambda u: Request(u.id, [{"role": "user",
                                       "content": u.item.payload["p"]}]),
             lambda u, r: {}, tmp_path / "r.jsonl", workers=1,
             stream=io.StringIO()).set_index("item_id")
    assert be.calls == 1
    assert df.loc["b", "error"] == "fatal: ShortOnly takes prompts under 10 characters"


def test_an_experiment_refuses_a_task_its_backend_refuses(tmp_path):
    from evalcore import Experiment, Task
    task = Task("long", [Item("a", {"messages": [{"role": "user",
                                                  "content": "far too long"}]})])
    with pytest.raises(ValueError, match="under 10 characters"):
        Experiment(task, ShortOnly(), grid({"m": [1]}), tmp_path / "r.jsonl")

"""The registry: a factory's signature is its config schema, and context
(keyword-only parameters) comes from the experiment, never the spec."""

from __future__ import annotations

import pytest

from evalcore.__main__ import main
from evalcore.registry import REGISTRIES, ConfigError, Registry, register


def reg():
    r = Registry("thing")

    @r("plain")
    def plain(size: int, colour: str = "red"):
        return ("plain", size, colour)

    @r("needs_docs")
    def needs_docs(k: int = 3, *, docs):
        return ("docs", k, docs)

    return r


def test_the_signature_is_the_schema():
    r = reg()
    assert r.build({"type": "plain", "size": 2}) == ("plain", 2, "red")
    assert r.build("needs_docs", docs=["d"]) == ("docs", 3, ["d"])
    assert r.schema("plain") == ({"size": "required", "colour": "red"}, [])
    assert r.schema("needs_docs") == ({"k": 3}, ["docs"])


def test_context_is_never_set_by_a_spec_and_must_be_available():
    r = reg()
    with pytest.raises(ConfigError, match=r"\['docs'\] cannot be set here"):
        r.build({"type": "needs_docs", "docs": []}, docs=["d"])
    with pytest.raises(ConfigError, match=r"needs \['docs'\], which is not "
                                          r"available here"):
        r.build({"type": "needs_docs"})


def test_context_a_factory_does_not_declare_is_not_passed():
    assert reg().build({"type": "plain", "size": 1}, docs=["d"]) == (
        "plain", 1, "red")


def test_list_prints_every_name_with_its_keys(capsys):
    assert main(["list", "retriever"]) == 0
    out = capsys.readouterr().out
    assert "bm25(k1=1.2, b=0.75)   [given: docs]" in out
    assert "rerank(first, pair_scorer, depth)" in out


def test_registering_a_taken_name_needs_replace():
    with pytest.raises(ValueError, match="already registered"):
        register("backend", "echo")(lambda: None)


def test_d17_no_built_in_factory_takes_kwargs():
    """D17.  Contract: a factory's signature is its schema, so no built-in
    (any factory defined in the evalcore package) may take **kwargs."""
    loose = sorted(f"{kind}:{name}"
                   for kind, reg in REGISTRIES.items()
                   for name, f in reg.factories.items()
                   if (getattr(f, "__module__", "") or "").startswith("evalcore")
                   and "**" in reg.schema(name)[0])
    assert loose == []
    assert len(REGISTRIES["backend"].factories) >= 4   # built-ins were seen

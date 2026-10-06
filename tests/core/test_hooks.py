"""Hooks: never silently absent, always removed, counted."""

from __future__ import annotations

import pytest

from evalcore.core.hooks import check_counts, hooked, match_modules

torch = pytest.importorskip("torch")


def net():
    torch.manual_seed(0)
    block = lambda: torch.nn.Sequential(torch.nn.Linear(4, 4),  # noqa: E731
                                        torch.nn.ReLU())
    return torch.nn.Sequential(block(), block(), block())


X = torch.randn(2, 4)


def test_a_pattern_matching_nothing_raises_before_anything_is_registered():
    m = net()
    with pytest.raises(KeyError, match=r"no module matches '0\.9'"), \
            hooked(m, {"*.0": lambda *a: None, "0.9": lambda *a: None}):
        pass
    assert all(not mod._forward_hooks for mod in m.modules())


def test_glob_patterns_and_call_counts():
    m = net()
    assert match_modules(m, "*.0") == ["0.0", "1.0", "2.0"]
    seen = []
    with hooked(m, {"*.0": lambda name, mod, args, out: seen.append(name)}) as c:
        m(X)
        m(X)
    assert seen == ["0.0", "1.0", "2.0"] * 2
    check_counts(c, 2)
    with pytest.raises(AssertionError, match="differ from 3"):
        check_counts(c, 3)


def test_hooks_are_removed_even_when_the_forward_raises():
    m = net()
    clean = m(X)

    def boom(name, mod, args, out):
        raise RuntimeError("bug in my hook")

    with pytest.raises(RuntimeError), hooked(m, {"1.0": boom}):
        m(X)
    assert all(not mod._forward_hooks for mod in m.modules())
    assert torch.equal(m(X), clean)                 # the model is untouched


def test_an_identity_hook_changes_nothing_and_an_edit_takes_effect():
    """alpha = 0 must be the identity, bitwise (an intervention at zero
    strength that changes the output is a bug in the hook, not an effect);
    a non-None return must replace the output."""
    m = net()
    clean = m(X)
    with hooked(m, {"1.0": lambda n, mod, a, out: out + 0.0 * out}):
        assert torch.equal(m(X), clean)
    with hooked(m, {"2.1": lambda n, mod, a, out: torch.zeros_like(out)}):
        assert torch.equal(m(X), torch.zeros_like(clean))
    assert torch.equal(m(X), clean)


def test_pre_hooks_with_kwargs_see_the_inputs():
    m = net()
    got = []
    with hooked(m, {"0.0": lambda n, mod, args, kwargs: got.append(
            args[0].shape)}, pre=True, with_kwargs=True):
        m(X)
    assert got == [torch.Size([2, 4])]

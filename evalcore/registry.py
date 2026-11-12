"""Named factories, so a JSON config can name any component, yours included.

Every configurable kind of component has a registry:

    task, backend, score, retriever, embedder, index, pair_scorer,
    analysis, stage

A factory is any callable (a function or a class) registered under a name:

    from evalcore.registry import register

    @register("backend", "my_server")
    def my_server(url: str, model: str, timeout: float = 60.0) -> Backend:
        return MyServerBackend(url, model, timeout)

and a config then says
{"backend": {"type": "my_server", "url": ..., "model": ...}}.
The factory's signature IS the schema: keys it does not take are refused
by name, parameters without a default are required, and no separate
schema needs writing.  (A factory taking **kwargs accepts any key and
then owns the checking.)

Some factories need more than their spec: a retriever needs the corpus, and a
retrieval backend needs the task.  Those are KEYWORD-ONLY parameters (after a
`*`): they are CONTEXT, supplied by the experiment through `build(spec, where,
docs=...)`, never settable from a spec, and a factory used where its context is
not available is refused by name.

A plugin module registers its factories on import.  Name it under
"plugins" in the config (a module name or a .py path) and `from_config`
imports it first.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

__all__ = ["KINDS", "REGISTRIES", "ConfigError", "Registry", "build", "register"]

F = TypeVar("F", bound=Callable[..., Any])

# While `recording()` is open: id(spec as given) -> (that spec, the spec
# with every default the factory filled in).  Holding the given spec keeps
# it alive, so its id cannot be reused by a later object while the record
# exists.  `resolve` swaps the resolved specs into the config.
Record = dict[int, tuple[Any, dict[str, Any]]]
_RECORD: ContextVar[Record | None] = ContextVar("_RECORD", default=None)


@contextmanager
def recording() -> Iterator[Record]:
    """Record every spec `build` resolves, nested ones included (a
    reranker's first-stage retriever is built inside the reranker's
    factory, from the dict the config gave it)."""
    rec: Record = {}
    token = _RECORD.set(rec)
    try:
        yield rec
    finally:
        _RECORD.reset(token)


def resolve(node: Any, rec: Record) -> Any:
    """`node` (a config, or part of one) with each spec `build` was given
    replaced by its resolved form, recursively.  Specs are matched by
    object identity, so the same text in a non-spec position is left
    alone.  A resolved spec's "type" is the name it was given, which for
    a spec given as a bare string is that same string object, so it is
    not resolved again."""
    hit = rec.get(id(node))
    if hit is not None and hit[0] is node:
        return {k: v if k == "type" else resolve(v, rec)
                for k, v in hit[1].items()}
    if isinstance(node, Mapping):
        return {k: resolve(v, rec) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [resolve(v, rec) for v in node]
    return node


class ConfigError(ValueError):
    """A config that cannot be read as written.  It names the offending key."""


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.factories: dict[str, Callable[..., Any]] = {}

    def __call__(self, name: str, *, replace: bool = False) -> Callable[[F], F]:
        """Decorator: register the decorated callable under `name`."""
        def deco(factory: F) -> F:
            self.add(name, factory, replace=replace)
            return factory
        return deco

    def add(self, name: str, factory: Callable[..., Any], *,
            replace: bool = False) -> None:
        if name in self.factories and not replace:
            raise ValueError(f"{self.kind} {name!r} is already registered "
                             f"(pass replace=True to override it)")
        self.factories[name] = factory

    def names(self) -> list[str]:
        return sorted(self.factories)

    def schema(self, name: str) -> tuple[dict[str, Any], list[str]]:
        """({key: default, or "required"}, [context names]): what a spec
        may set, and what the experiment supplies."""
        keys: dict[str, Any] = {}
        context = []
        for p in inspect.signature(self.factories[name]).parameters.values():
            if p.kind is p.VAR_KEYWORD:
                keys["**"] = "any key"
            elif p.kind is p.KEYWORD_ONLY:
                context.append(p.name)
            elif p.kind is not p.VAR_POSITIONAL:
                keys[p.name] = "required" if p.default is p.empty else p.default
        return keys, context

    def build(self, spec: Mapping[str, Any] | str, where: str | None = None,
              **context: Any) -> Any:
        """The component a spec describes.  `spec` is {"type": name, ...}
        or just the name.  Context values go to the factory's parameters of
        the same name, and only to those."""
        where = where or self.kind
        raw = spec
        if isinstance(spec, str):
            spec = {"type": spec}
        if not isinstance(spec, Mapping) or "type" not in spec:
            raise ConfigError(f"{where}: give a {{\"type\": ...}} naming a "
                              f"{self.kind}: one of {self.names()}")
        name = spec["type"]
        if name not in self.factories:
            raise ConfigError(f"{where}: unknown {self.kind} {name!r}. "
                              f"Registered: {self.names()}")
        factory = self.factories[name]
        params = inspect.signature(factory).parameters
        free = any(p.kind is p.VAR_KEYWORD for p in params.values())
        needs = {k: p for k, p in params.items() if p.kind is p.KEYWORD_ONLY}
        own = {k: p for k, p in params.items()
               if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)}
        given = {k: v for k, v in spec.items() if k != "type"}
        clash = sorted(set(given) & (set(needs) | set(context)))
        if clash:
            raise ConfigError(f"{where}: {clash} cannot be set here because they are "
                              f"supplied by the experiment")
        absent = sorted(k for k, p in needs.items()
                        if p.default is p.empty and k not in context)
        if absent:
            raise ConfigError(f"{where}: {self.kind} {name!r} needs {absent}, "
                              f"which is not available here")
        unknown = sorted(set(given) - set(own))
        if unknown and not free:
            raise ConfigError(f"{where}: unknown key(s) {unknown} for "
                              f"{self.kind} {name!r}. Allowed: {sorted(own)}")
        missing = sorted(k for k, p in own.items()
                         if p.default is p.empty and k not in given)
        if missing:
            raise ConfigError(f"{where}: {self.kind} {name!r} requires "
                              f"{missing}")
        rec = _RECORD.get()
        if rec is not None:
            # signature order, every default written out; a **kwargs
            # factory's extra keys as given
            rec[id(raw)] = (raw, {"type": name, **{
                k: given.get(k, p.default) for k, p in own.items()},
                **{k: v for k, v in given.items() if k not in own}})
        ctx = {k: v for k, v in context.items() if k in needs}
        return factory(**given, **ctx)


KINDS = ("task", "backend", "score", "retriever", "embedder", "index",
         "pair_scorer", "analysis", "stage")
REGISTRIES: dict[str, Registry] = {k: Registry(k) for k in KINDS}


def register(kind: str, name: str, *, replace: bool = False
             ) -> Callable[[F], F]:
    """Decorator: register a factory of `kind` under `name`."""
    if kind not in REGISTRIES:
        raise ValueError(f"unknown kind {kind!r}. Kinds: {list(KINDS)}")
    return REGISTRIES[kind](name, replace=replace)


def build(kind: str, spec: Mapping[str, Any] | str, where: str | None = None,
          **context: Any) -> Any:
    return REGISTRIES[kind].build(spec, where, **context)

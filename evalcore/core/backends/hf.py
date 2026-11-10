"""Local HuggingFace models: batched generation, forward passes, and
teacher-forced scoring of a target continuation."""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from ..hooks import check_counts, hooked
from ..spec import digest

if TYPE_CHECKING:
    import torch
    from transformers import DynamicCache, PreTrainedTokenizerBase

from .base import (
    DECODE_KEYS,
    Backend,
    Fatal,
    Request,
    Response,
    decode_signature,
    filled,
    merge_params,
    refuse_dropped,
    require,
    unread_params,
)
from .probe import FunctionProbe, Probe


class HFBackend(Backend):
    """Local HuggingFace forward passes, genuinely batched.

    `probe` (core/backends/probe.py) captures from inside each forward
    pass: its `collect(out, enc, reqs)` receives the raw model output, the
    tokenizer encoding and the requests, and returns one dict per request
    (`FunctionProbe(fn)` makes one from a plain function).  The layout of
    `enc` differs by path and under prefix reuse; `_probing` documents it
    key by key.  Exactly one dict per request, or the batch fails with an
    error naming the probe.

    Everything model-specific -- per-head attention shares over item spans,
    next-token distributions restricted to the in-context value set -- lives
    in the probe, so the core never grows a dependency on torch.

    `prefix_cache=True` keeps the key/value caches of up to `prefix_slots`
    recent prompts and reuses the one sharing the longest token-id prefix
    with the next prompt, so a call costs O(new tokens) rather than
    O(prompt).  A sequential run whose prompt k contains prompts 1..k-1
    then costs O(n) prompt tokens in total instead of O(n^2).  Matching is
    on token ids, never on strings, because a shared string prefix need not
    tokenize to a shared id prefix at a BPE boundary.

    What a slot holds.  For a prompt x_0..x_{T-1} a slot stores the cache
    of x_0..x_{T-2} only: everything after the call is cropped back to
    T - 1 positions.  The read position is the last prompt token T - 1,
    and a probe may patch activations there (and at generated tokens);
    under the causal mask a patch at position t changes the keys and
    values at t and later only, so positions < T - 1 are the unpatched
    model's whatever the probe did.  Hence a stored prefix never contains
    a patched position or a generated token, and every call computes at
    least its last prompt token fresh: a prompt identical to a stored one
    reuses T - 1 positions.

    Which slot.  The slot with the longest common prefix n (ties: the most
    recently used); n = 0 runs a full pass.  The new prompt's cache goes
    into the slot it extends when it extends one completely (that slot is
    superseded: any prompt matching it matches the new one at least as
    far), else into an empty slot, else over the least recently used.
    When the source slot is kept, its cache is copied before use, because
    a forward appends to a cache in place.  Each row's meta records
    `n_prefix_reused` (0 = full pass) and `prefix_slot` (the slot index
    0..K-1 it reused, None for a full pass).

    Memory.  A cache holds, per token, 2 (K and V) x n_layers x n_kv_heads
    x head_dim values: 2 x 36 x 2 x 128 x 2 bytes ~ 37 KB in bf16 for a
    36-layer model with 2 KV heads of dim 128.  The slots hold at most
    K x T_max tokens, plus one transient copy of up to T_max during a call
    that reuses a slot it keeps: (K + 1) x T_max x 37 KB, about 0.6 GB at
    K = 4 and T_max = 3000 for that model.

    Numbers under reuse depend on call history.  A reused and a full pass
    differ at rounding level (the attention and matmul sums run over a
    different split of the sequence), and which split point n a call gets
    depends on which prompts came before and on K.  That is why the flag
    and K enter `identity()`, and why `n_prefix_reused` in meta records the
    split of every row.  Refused rather than worked around: batch size > 1,
    and `target` requests (teacher forcing is not reused)."""

    name = "hf"
    supports_target = True
    # the decode params `_run_batch` passes to generate(); keep in step with it
    GENERATE_KEYS = frozenset({"max_tokens", "temperature",
                               "repetition_penalty"})

    # model and tokenizer are None only in tests that stub the forward path
    def __init__(self, model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 device: Any = None,
                 output_attentions: bool = False, generate: bool = False,
                 default_params: Mapping[str, Any] | None = None,
                 probe: Probe | None = None,
                 prefix_cache: bool = False,
                 prefix_slots: int | None = None) -> None:
        if prefix_cache and prefix_slots is None:
            raise ValueError(
                "prefix_cache=True needs prefix_slots (how many prompts' "
                "caches to keep): it decides which split point a call gets, "
                "so it has no default")
        if not prefix_cache and prefix_slots is not None:
            raise ValueError("prefix_slots is set but prefix_cache is off")
        if prefix_slots is not None and (
                not isinstance(prefix_slots, int) or prefix_slots < 1):
            raise ValueError(f"prefix_slots must be an int >= 1, "
                             f"got {prefix_slots!r}")
        self.model, self.tok = model, tokenizer
        self.probe, self.device = probe, device
        self.output_attentions, self.generate = output_attentions, generate
        self.default_params = dict(default_params or {})
        self.prefix_cache, self.prefix_slots = prefix_cache, prefix_slots
        # per slot: (the cached token ids, their KV cache), or None if empty;
        # and the tick at which it was last used, for LRU eviction
        self._slots: list[tuple[list[int], DynamicCache] | None] = \
            [None] * (prefix_slots or 0)
        self._last_used: list[int] = [0] * (prefix_slots or 0)
        self._tick = 0
        self._prefix_lock = threading.Lock()
        self._prepare_tokenizer()

    def identity(self) -> dict[str, Any]:
        # The checkpoint is named by its path AND the snapshot loaded: a hub
        # id keeps the same name_or_path at every revision (Pythia's step
        # axis), and only config._commit_hash tells them apart.  The probe
        # by its own identity, since what it captures is cached alongside
        # the text.
        return {**self._base_identity(),
                "model": getattr(self.model, "name_or_path", None),
                "commit": getattr(getattr(self.model, "config", None),
                                  "_commit_hash", None),
                # the same weights in bf16 and fp32 give different logits,
                # and a chat template decides the prompt the model reads;
                # neither shows in the request
                "dtype": self._dtypes(),
                "chat_template": (digest(self.tok.chat_template)
                                  if self._templated() else None),
                "generate": self.generate,
                "output_attentions": self.output_attentions,
                "probe": None if self.probe is None else self.probe.identity(),
                # only when on (the flag adds no key when off); K with it,
                # since K moves the split points
                **({"prefix_cache": True, "prefix_slots": self.prefix_slots}
                   if self.prefix_cache else {})}

    def refusal(self, req: Request) -> str | None:
        if self.prefix_cache and req.target is not None:
            return ("HFBackend(prefix_cache=True) does not teacher-force "
                    "targets; score them with a backend without prefix_cache")
        if self.probe is None:
            # without a probe nothing here reads a param outside DECODE_KEYS
            # (complete_batch checks those per path), so any other key -- a
            # label such as the model's name, a typo -- would move the cell
            # id while the forward pass stays the same.  A probe receives the
            # requests and may read such a key itself, so it is allowed then.
            unread = unread_params(merge_params(self.default_params,
                                                req.params), DECODE_KEYS)
            if unread:
                return (f"HFBackend without a probe reads no param {unread}: "
                        f"it uses only {sorted(DECODE_KEYS)}.  A label (the "
                        f"model's name, a condition) belongs in the cell's tags")
        return super().refusal(req)

    def _prepare_tokenizer(self) -> None:
        """Pin the two tokenizer settings this backend's index arithmetic assumes.

        Generated text is sliced off as `gen[:, prompt_len:]` and the forward path
        reads the final column, both of which are correct only when padding is on
        the LEFT -- under right padding they return pad columns and shifted spans
        with no error raised.  Many instruct tokenizers also ship `pad_token=None`
        (Llama, Mistral), which silently changes what the truncation check below
        measures; falling back to eos is safe because pad columns are masked out
        of attention.  A None tokenizer is left alone: the batching logic is
        tested with a stub that never reaches the encode path.
        """
        if self.tok is None:
            return
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if self.tok.pad_token_id is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")

    def complete(self, req: Request) -> Response:
        """A single request is a batch of one.  `execute` calls this whenever
        batch_size == 1; without it the base class raised
        NotImplementedError and every unit became an error row."""
        return self.complete_batch([req])[0]

    def _device(self) -> Any:
        """Where inputs go: `device` if given, else wherever the model
        already is.  Never a guess such as "cuda if available": the backend
        does not move the model, so a guessed device that differs from the
        model's fails, and with device_map="auto" (offloading) the model's
        own `device` is the one its first layer expects."""
        if self.device is not None:
            return self.device
        dev = getattr(self.model, "device", None)
        if dev is None:
            try:
                dev = next(self.model.parameters()).device
            except (AttributeError, StopIteration):
                dev = "cpu"
        return dev

    @contextmanager
    def _probing(self, enc: Any, reqs: Sequence[Request],
                 calls_known: bool = True) -> Iterator[SimpleNamespace]:
        """Around one forward pass: start the probe, install its hooks, and
        afterwards check the hook counts and collect one dict per request
        into `holder.extras`.  Set `holder.out` to the model output inside
        the block.  Without a probe this is a no-op.

        What `enc` holds (the probe's `start` and `collect` both get it).
        Positions in `enc` are always FULL-prompt positions, with or without
        prefix reuse:

            input_ids        the full sequence, cached positions included:
                             the prompt (generation, plain forward,
                             candidates) or prompt + target (targets)
            attention_mask   the same width, covering cached + fed positions,
                             so mask.sum(1) is the full prompt length
            padding_side     "left" for generation and plain forward,
                             "right" for candidates and targets
            last             (candidates) per row, the final real position
            n_prompt         (targets) per row, the number of prompt tokens
            n_prefix_reused  per row, how many leading positions came from
                             a cached prefix and were NOT fed (0 without
                             prefix_cache, and on the target path)

        The read position is the last real prompt token: `last[k]` when
        right-padded, the final column when left-padded.  Under reuse the
        batch is one unpadded row, so the final column is real.

        What the hooks see.  The first forward is fed only positions
        n_prefix_reused .. T-1, so every hooked module sees a sequence axis
        of T - n_prefix_reused there, and full position p sits at index
        p - n_prefix_reused (the read position at the last index).  Later
        forwards inside generate() see one new token each.  `out` likewise
        covers the fed positions only (logits, hidden states; attention
        weights have the fed positions as queries and all T as keys).  On
        the candidates path lm_head is applied only at the kept positions
        (`logits_to_keep`), one per row, with or without reuse.

        A patch is safe under reuse only at the read position or later
        (generated tokens): `_reusing` stores positions < T - 1, which a
        patch at T - 1 cannot reach under the causal mask.  A probe that
        writes to an earlier fed position would leave it in a stored prefix
        and reach later calls; that is not detectable here, so do not."""
        holder = SimpleNamespace(out=None, extras=[{} for _ in reqs])
        if self.probe is None:
            yield holder
            return
        self.probe.start(enc, reqs)
        hk = self.probe.hooks()
        with hooked(self.model, hk, pre=self.probe.pre,
                    with_kwargs=self.probe.with_kwargs) as counts:
            yield holder
        if hk and calls_known and self.probe.calls_per_forward is not None:
            check_counts(counts, self.probe.calls_per_forward)
        extras = list(self.probe.collect(holder.out, enc, reqs))
        if len(extras) != len(reqs):
            what = (f"FunctionProbe({getattr(self.probe.fn, '__name__', '?')})"
                    if isinstance(self.probe, FunctionProbe)
                    else type(self.probe).__name__)
            raise ValueError(
                f"{what} returned {len(extras)} dicts for {len(reqs)} "
                f"requests; it must return exactly one per request")
        holder.extras = extras

    @contextmanager
    def _reusing(self, model_in: Mapping[str, Any], feed_tail: bool
                 ) -> Iterator[tuple[dict[str, Any], int, int | None]]:
        """The model inputs for one forward (or generate) call, how many
        leading positions come from a cached prefix, and the slot they came
        from (None for a full pass).

        Without prefix_cache: `model_in` unchanged, 0 reused.  With it (one
        unpadded row of ids x_0..x_{T-1}): n = the longest common prefix
        with any slot's ids, capped at T - 1 so the read position T - 1 is
        always fed.  That slot's cache, cropped to n, is passed as
        `past_key_values`; the attention mask still covers all T positions,
        and positions continue from n (the model derives them from the
        cache length).  `feed_tail` feeds x_n..x_{T-1} only, for a plain
        forward; generate() takes the full ids and skips the cached part
        itself.  The call fills the cache in place, so afterwards it is
        cropped to T - 1 (dropping the read position, which a probe may
        have patched, and any generated tokens) and stored.  The class
        docstring says which slot it goes to.  If the call raises nothing
        is stored, and a slot consumed by the call stays empty: a
        half-written cache is never reused."""
        if not self.prefix_cache:
            yield dict(model_in), 0, None
            return
        import copy

        from transformers import DynamicCache
        unsliced = set(model_in) - {"input_ids", "attention_mask"}
        if unsliced:
            raise Fatal(f"prefix_cache cannot slice inputs {sorted(unsliced)}")
        ids = [int(t) for t in model_in["input_ids"][0].tolist()]
        with self._prefix_lock:
            src, n = self._best_slot(ids)
            stored = None if src is None else self._slots[src]
            dst: int
            if src is not None and stored is not None \
                    and n == len(stored[0]):
                dst = src                    # the new prompt supersedes it
            elif None in self._slots:
                dst = self._slots.index(None)
            else:
                dst = min(range(len(self._slots)),
                          key=self._last_used.__getitem__)
            if src is None or stored is None:
                cache = DynamicCache()
            else:
                if src == dst:
                    cache = stored[1]        # consumed: emptied until stored
                    self._slots[dst] = None
                else:
                    cache = copy.deepcopy(stored[1])     # the source slot stays
                _crop(cache, n)
                self._touch(src)
            fed = {**model_in, "past_key_values": cache}
            if feed_tail:
                fed["input_ids"] = model_in["input_ids"][:, n:]
            yield fed, n, src
            if len(ids) > 1:
                _crop(cache, len(ids) - 1)
                self._slots[dst] = (ids[:-1], cache)
                self._touch(dst)

    def _best_slot(self, ids: Sequence[int]) -> tuple[int | None, int]:
        """The slot sharing the longest token prefix with `ids` (ties: the
        most recently used) and that length capped at len(ids) - 1; (None,
        0) when no slot shares even the first token."""
        best: tuple[int, int, int] | None = None     # (n, last used, slot)
        for k, s in enumerate(self._slots):
            if s is None:
                continue
            n = min(_common_prefix(s[0], ids), len(ids) - 1)
            if n > 0 and (best is None or (n, self._last_used[k]) > best[:2]):
                best = (n, self._last_used[k], k)
        return (None, 0) if best is None else (best[2], best[0])

    def _touch(self, k: int) -> None:
        self._tick += 1
        self._last_used[k] = self._tick

    def _dtypes(self) -> str | None:
        """The parameter dtypes, sorted and comma-joined ("torch.bfloat16";
        a partly quantised model lists several).  None for the stub models
        some tests pass as `model=None`."""
        if self.model is None:
            return None
        return ",".join(sorted({str(p.dtype) for p in self.model.parameters()}))

    def _eos_ids(self) -> set[int]:
        """Token ids that end a generation: the tokenizer's eos and every id
        in the checkpoint's generation_config.eos_token_id (an int or a list;
        Llama 3 instruct stops on <|eot_id|> as well as <|end_of_text|>).
        generate() stops on the latter, so the truncation check reads it
        too.  A model without a generation_config (a test stub) has only
        the tokenizer's."""
        ids = {self.tok.eos_token_id} if self.tok.eos_token_id is not None \
            else set()
        cfg = getattr(self.model, "generation_config", None)
        extra = None if cfg is None else cfg.eos_token_id
        if extra is not None:
            ids |= set(extra) if isinstance(extra, (list, tuple)) else {extra}
        return {int(i) for i in ids}

    def _templated(self) -> bool:
        return bool(getattr(self.tok, "chat_template", None))

    def _encode(self, text: str) -> list[int]:
        """Token ids of rendered text.  A chat template already writes the
        special tokens it wants (Llama's emits BOS itself), so templated text
        is encoded WITHOUT adding them again; plain text gets the
        tokenizer's own (Pythia adds none)."""
        return list(self.tok(text, add_special_tokens=not self._templated())
                    ["input_ids"])

    def _render(self, req: Request) -> str:
        if self._templated():
            # tokenize=False returns the rendered string
            return cast(str, self.tok.apply_chat_template(
                [dict(m) for m in req.messages], tokenize=False,
                add_generation_prompt=True))
        return "\n\n".join(m["content"] for m in req.messages)

    def complete_batch(self, reqs: Sequence[Request]) -> list[Response]:
        """A batch shares one generate() call, so it must share one set of
        decode params.  Mixed batches are split and run as separate groups
        rather than silently taking the first request's settings -- otherwise
        a grid that varies temperature or max_tokens contaminates every cell
        whose requests happen to land behind another cell's in a batch.

        Requests with a `target` are teacher-force scored (`_score_batch`)
        and requests with `candidates` get the restricted next-token
        distribution (`_choice_batch`), whatever `generate` says; a batch
        mixing kinds is split by kind the same way."""
        kinds = ["target" if r.target is not None
                 else "choice" if r.candidates is not None else "plain"
                 for r in reqs]
        if self.prefix_cache:
            if len(reqs) > 1:
                raise Fatal(f"prefix_cache reuses one cached prompt per call: "
                            f"run with batch_size=1 (got {len(reqs)} requests)")
            if kinds[0] == "target":
                raise Fatal(cast(str, self.refusal(reqs[0])))
        if len(set(kinds)) > 1:
            return _by_group(reqs, kinds, self.complete_batch)
        if kinds[0] == "target":
            return self._score_batch(reqs)
        if kinds[0] == "choice":
            return self._choice_batch(reqs)
        def prepare(r: Request) -> str | None:
            """The request's decode signature, its batch group.  A plain
            forward reads logits once and samples nothing, so there any
            decode param would move the cell id and the cache key while
            changing nothing computed."""
            if not self.generate:
                refuse_dropped(r.params, frozenset(),
                               "HFBackend forward (generate=False)")
                return None
            p = merge_params(self.default_params, r.params)
            refuse_dropped(p, self.GENERATE_KEYS, "HFBackend.generate")
            return decode_signature(p)

        # refused per request, as on the scoring paths, so one bad request
        # never fails its batch
        out, ok = self._prepared(reqs, prepare)
        idx = [i for i, _ in ok]
        for j, resp in zip(idx, _by_group([reqs[i] for i in idx],
                                          [sig for _, sig in ok],
                                          self._run_batch), strict=True):
            out[j] = resp
        return filled(out)

    def _target_ids(self, req: Request) -> tuple[list[int], int]:
        """Token ids of prompt + target, and the number of prompt tokens.

        Tokenised JOINTLY, then checked: the joint ids must begin with the
        prompt's own ids.  Otherwise a merge straddles the join ("wor" +
        "ld" -> "world") and no token sequence scores the target alone, so
        the request is refused rather than silently scoring a merged token
        or a shifted span.  The usual convention -- prompt ending in "=",
        target starting with a space -- always passes, because byte-level
        BPE merges never cross a space-led boundary of this kind; the check
        is what makes that a guarantee instead of an assumption."""
        prompt = self._render(req)
        p_ids = self._encode(prompt)
        assert req.target is not None, "_target_ids needs a target"
        ids = self._encode(prompt + req.target)
        if not p_ids:
            raise Fatal("teacher forcing needs a non-empty prompt: the first "
                        "target token is predicted from the token before it")
        if ids[:len(p_ids)] != p_ids or len(ids) == len(p_ids):
            raise Fatal(
                f"target {req.target!r} does not begin on a token boundary "
                f"after the prompt: prompt + target tokenises across the "
                f"join.  Move the boundary (e.g. end the prompt before a "
                f"space and start the target with it).")
        return ids, len(p_ids)

    def _prepared(self, reqs: Sequence[Request],
                  prepare: Callable[[Request], Any]
                  ) -> tuple[list[Response | None], list[tuple[int, Any]]]:
        """Run `prepare` on each request; one that raises Fatal becomes an
        error response, the others are returned with their index, so one
        bad request never fails its batch."""
        out: list[Response | None] = [None] * len(reqs)
        rows = []
        for i, r in enumerate(reqs):
            try:
                rows.append((i, prepare(r)))
            except Fatal as exc:
                out[i] = Response(r.unit_id, error=f"fatal: {exc}")
        return out, rows

    def _right_padded(self, seqs: Sequence[Sequence[int]]) -> dict[str, Any]:
        """input_ids and attention_mask for sequences padded on the RIGHT, on
        the model's device.  Under a causal mask no real token attends to a
        later pad, so each row's logits are those of its unpadded run."""
        import torch
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok is not None else 0
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(list(s), dtype=torch.long)
            mask[k, :len(s)] = 1
        dev = self._device()
        return {"input_ids": ids.to(dev), "attention_mask": mask.to(dev)}

    def _choice_batch(self, reqs: Sequence[Request]) -> list[Response]:
        r"""Next-token log probabilities of each request's `candidates`.

        The prompt is `input_ids` if given (built in token space), else the
        rendered messages.  At its last position, with z the logits and
        p = softmax(z) over the FULL vocabulary:

            cand_logp_i = log p[c_i],      cand_mass = sum_i p[c_i],
            leakage = 1 - cand_mass,       top1_id = argmax_v z_v,
            top1_cand_index = i if top1_id = c_i, else -1.

        Probabilities are NOT renormalised over the set: leakage stays a
        separate number, and a scorer that wants the conditional
        distribution divides by cand_mass itself.  Right-padded batches;
        only each row's last position is materialised (`logits_to_keep`
        with the row positions), so a 1k-token prompt over a 150k vocabulary
        does not allocate full logits.  Whether the forward takes
        `logits_to_keep` is read from its signature, a named parameter and
        nothing less: a forward that swallows it in **kwargs would return
        full logits, read here at the wrong columns.  Without it, full
        logits.  Either way the logits' length is checked against what was
        asked for."""
        import torch

        def prepare(r: Request) -> tuple[list[int], list[int]]:
            refuse_dropped(r.params, frozenset(), "HFBackend candidate scoring")
            cands = [int(c) for c in (r.candidates or ())]
            if not cands:
                raise Fatal("candidates is empty")
            if len(set(cands)) != len(cands):
                raise Fatal("candidates contain duplicates")
            ids = (list(r.input_ids) if r.input_ids is not None
                   else self._encode(self._render(r)))
            if not ids:
                raise Fatal("empty prompt")
            return ids, cands

        out, prepared = self._prepared(reqs, prepare)
        if not prepared:
            return filled(out)
        rows = [(i, ids, cands) for i, (ids, cands) in prepared]
        model_in = self._right_padded([ids for _, ids, _ in rows])
        dev = model_in["input_ids"].device
        last = [len(ids) - 1 for _, ids, _ in rows]
        t0 = time.perf_counter()
        with torch.no_grad(), \
                self._reusing(model_in, feed_tail=True) as (fed, n_reused,
                                                            slot), \
                self._probing({**model_in, "padding_side": "right",
                               "last": last,
                               "n_prefix_reused": [n_reused] * len(rows)},
                              [reqs[i] for i, _, _ in rows]) as probed:
            # positions within what is fed: the uncached tail under reuse
            keep = sorted({pos - n_reused for pos in last})
            if "logits_to_keep" in inspect.signature(
                    self.model.forward).parameters:
                model_out = self.model(
                    **fed, output_attentions=self.output_attentions,
                    logits_to_keep=torch.tensor(keep, device=dev))
                col = {pos + n_reused: j for j, pos in enumerate(keep)}
                width = len(keep)
            else:
                model_out = self.model(**fed,
                                       output_attentions=self.output_attentions)
                col = {pos + n_reused: pos for pos in keep}
                width = fed["input_ids"].shape[1]
            if model_out.logits.shape[1] != width:
                raise RuntimeError(
                    f"the model returned logits at {model_out.logits.shape[1]} "
                    f"positions, expected {width}: its forward does not "
                    f"honour logits_to_keep as declared")
            probed.out = model_out
        dt = time.perf_counter() - t0
        extras = probed.extras
        logits = model_out.logits
        vocab = logits.shape[-1]
        for k, (i, _, cands) in enumerate(rows):
            if max(cands) >= vocab or min(cands) < 0:
                out[i] = Response(reqs[i].unit_id, error=(
                    f"fatal: candidate id outside the vocabulary (0..{vocab - 1})"))
                continue
            lsm = logits[k, col[last[k]]].float().log_softmax(-1).cpu()
            c = torch.tensor(cands, dtype=torch.long)
            clp = lsm[c]
            top = int(lsm.argmax())
            out[i] = Response(reqs[i].unit_id, text=None, meta={
                **extras[k],
                "cand_ids": cands, "cand_logp": clp.tolist(),
                "cand_mass": float(clp.exp().sum()),
                "leakage": float(1.0 - clp.exp().sum()),
                "top1_id": top,
                "top1_cand_index": cands.index(top) if top in cands else -1,
                "n_prompt": len(rows[k][1]), "truncated": False,
                "latency_s": dt / len(rows), "backend": "hf",
                **self._reuse_meta(n_reused, slot)})
        return filled(out)

    def _reuse_meta(self, n_reused: int, slot: int | None) -> dict[str, Any]:
        return ({"n_prefix_reused": n_reused, "prefix_slot": slot}
                if self.prefix_cache else {})

    def _score_batch(self, reqs: Sequence[Request]) -> list[Response]:
        r"""Teacher-forced log probabilities of each request's target.

        For a sequence x_0..x_{n-1} whose last n - P tokens are the target,
        the model's logits at position t predict x_{t+1}, so target token j
        (at index P + j) is scored from position P + j - 1:

            logp_j = log softmax(z_{P+j-1})[x_{P+j}],
            argmax_ok_j = [argmax z_{P+j-1} = x_{P+j}],
            em_tf = all_j argmax_ok_j.

        Padding is on the RIGHT here, unlike generation: under a causal mask
        no real token attends to a later pad, so each sequence's logits are
        those of its unpadded run and positions need no correction.  The
        log-softmax is taken in fp32 whatever the model's dtype.

        A request that fails its own checks (decode params, token boundary)
        becomes an error response; the rest of the batch is still scored."""
        import torch

        def prepare(r: Request) -> tuple[list[int], int]:
            if r.input_ids is not None:
                raise Fatal("a target with input_ids cannot be checked for a "
                            "token boundary; give the prompt as messages, or "
                            "score the continuation as candidates")
            # the request's own params only: a generate backend's decode
            # defaults are not part of a scoring request
            refuse_dropped(r.params, frozenset(),
                           "HFBackend teacher-forced scoring")
            return self._target_ids(r)

        out, prepared = self._prepared(reqs, prepare)
        if not prepared:
            return filled(out)
        seqs = [(i, ids, n_prompt) for i, (ids, n_prompt) in prepared]
        model_in = self._right_padded([ids for _, ids, _ in seqs])
        enc = {**model_in, "padding_side": "right",
               "n_prompt": [n_prompt for _, _, n_prompt in seqs],
               "n_prefix_reused": [0] * len(seqs)}
        t0 = time.perf_counter()
        batch_reqs = [reqs[i] for i, _, _ in seqs]
        with torch.no_grad(), self._probing(enc, batch_reqs) as probed:
            model_out = self.model(**model_in,
                                   output_attentions=self.output_attentions)
            probed.out = model_out
        dt = time.perf_counter() - t0
        logits = model_out.logits
        extras = probed.extras
        for k, (i, ids, n_prompt) in enumerate(seqs):
            n = len(ids)
            lsm = logits[k, n_prompt - 1:n - 1].float().log_softmax(-1).cpu()
            tgt = torch.tensor(ids[n_prompt:], dtype=torch.long)
            logp = lsm.gather(-1, tgt[:, None])[:, 0]
            ok = lsm.argmax(-1) == tgt
            out[i] = Response(reqs[i].unit_id, text=None, meta={
                **extras[k],
                "target_ids": tgt.tolist(),
                "target_tokens": [self.tok.decode([t]) for t in tgt.tolist()],
                "logp": logp.tolist(), "argmax_ok": ok.tolist(),
                "n_target": n - n_prompt, "sum_logp": float(logp.sum()),
                "em_tf": bool(ok.all()), "truncated": False,
                "latency_s": dt / len(seqs), "backend": "hf"})
        return filled(out)

    def _run_batch(self, reqs: Sequence[Request]) -> list[Response]:
        import torch
        if self.generate:
            # one generate() call decodes every row at reqs[0]'s params, so a
            # mixed batch must never get here; complete_batch splits it first
            sigs = {decode_signature(merge_params(self.default_params, r.params))
                    for r in reqs}
            if len(sigs) > 1:
                raise AssertionError(
                    f"internal: _run_batch got {len(sigs)} decode signatures; "
                    f"mixed batches must be split by complete_batch")
        prompts = [self._render(r) for r in reqs]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=not self._templated()
                       ).to(self._device())
        t0 = time.perf_counter()
        # generate() runs one forward per new token, so the per-forward call
        # count is not known in advance and is not checked there
        with torch.no_grad(), \
                self._reusing(enc, feed_tail=not self.generate) as (
                    fed, n_reused, slot), \
                self._probing({**enc, "padding_side": "left",
                               "n_prefix_reused": [n_reused] * len(reqs)},
                              reqs, calls_known=not self.generate) as probed:
            texts: list[str | None]
            if self.generate:
                p = merge_params(self.default_params, reqs[0].params)
                budget = int(require(p, "max_tokens", "HFBackend.generate"))
                # repetition_penalty is passed explicitly because checkpoints ship
                # their own (Qwen2.5-Instruct: 1.1) in generation_config.  Left to
                # the checkpoint, do_sample=False returns the argmax of PENALIZED
                # logits -- a greedy decode that disagrees with the model's own
                # logits and with any other harness scoring the same items.
                # generate() is a transformers method, unknown to torch's types
                gen = cast(Any, self.model).generate(
                    **fed, max_new_tokens=budget,
                    do_sample=p.get("temperature", 0.0) > 0,
                    temperature=p.get("temperature", 1.0) or 1.0,
                    repetition_penalty=p.get("repetition_penalty", 1.0))
                new = gen[:, enc["input_ids"].shape[1]:]
                texts = [t for t in self.tok.batch_decode(
                    new, skip_special_tokens=True)]
                # Cut, not finished: the row used the whole budget without
                # emitting a stop token.  generate() fills a finished row
                # with pad after its eos, so a row holding any eos or pad id
                # stopped on its own -- including one whose eos landed in
                # the last budget slot, which counting non-pad tokens would
                # call cut whenever pad != eos.
                stop = self._eos_ids() | (
                    {int(self.tok.pad_token_id)}
                    if self.tok.pad_token_id is not None else set())
                cut = [new.shape[1] >= budget
                       and not any(int(t) in stop for t in row) for row in new]
                out = None
            else:
                out = self.model(**fed, output_attentions=self.output_attentions)
                texts = [None] * len(reqs)
                cut = [False] * len(reqs)
            probed.out = out
        dt = time.perf_counter() - t0
        extras = probed.extras
        return [Response(r.unit_id, text=t,
                         meta={**e, "latency_s": dt / len(reqs), "backend": "hf",
                               "truncated": c,
                               **self._reuse_meta(n_reused, slot)})
                for r, t, e, c in zip(reqs, texts, extras, cut, strict=True)]


def _crop(cache: Any, n: int) -> None:
    """Cut `cache` to its first n positions.  The call is spelled the way
    every supported transformers reads alike: a negative count of positions
    to drop (5.x refuses a positive length, and 4.x reads crop(0) as "keep
    nothing"), and no call when there is nothing to drop."""
    drop = cache.get_seq_length() - n
    if drop < 0:
        raise ValueError(f"cannot crop a cache of {cache.get_seq_length()} "
                         f"positions to {n}")
    if drop:
        cache.crop(-drop)


def _by_group(reqs: Sequence[Request], keys: Sequence[Any],
              run: Callable[[Sequence[Request]], list[Response]]
              ) -> list[Response]:
    """`run` once per distinct key, on that key's requests in their original
    order, with the responses put back in place."""
    out: list[Response | None] = [None] * len(reqs)
    for key in dict.fromkeys(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        for j, resp in zip(idx, run([reqs[i] for i in idx]), strict=True):
            out[j] = resp
    return filled(out)


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n

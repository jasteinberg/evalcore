"""Local HuggingFace models: batched generation, forward passes, and
teacher-forced scoring of a target continuation."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from ..hooks import check_counts, hooked

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase

from .base import (
    Backend,
    Fatal,
    Request,
    Response,
    decode_signature,
    filled,
    merge_params,
    refuse_dropped,
    require,
)
from .probe import FunctionProbe, Probe


class HFBackend(Backend):
    """Local HuggingFace forward passes, genuinely batched.

    `probe` (core/backends/probe.py) captures from inside each forward
    pass: its `collect(out, enc, reqs)` receives the raw model output, the
    tokenizer encoding and the requests, and returns one dict per request
    (`FunctionProbe(fn)` makes one from a plain function).  The layout of
    `enc` differs by path, so read it from `enc` rather than assuming:
    generation and plain forward passes encode the prompt only,
    LEFT-padded; teacher-forced scoring encodes prompt + target,
    RIGHT-padded, and `enc` also carries `padding_side` and `n_prompt`
    (prompt tokens per request).  Exactly one dict per request, or the
    batch fails with an error naming the probe.

    Everything model-specific -- per-head attention shares over item spans,
    next-token distributions restricted to the in-context value set -- lives
    in the probe, so the core never grows a dependency on torch."""

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
                 probe: Probe | None = None) -> None:
        self.model, self.tok = model, tokenizer
        self.probe, self.device = probe, device
        self.output_attentions, self.generate = output_attentions, generate
        self.default_params = dict(default_params or {})
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
                "generate": self.generate,
                "output_attentions": self.output_attentions,
                "probe": None if self.probe is None else self.probe.identity()}

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
        the block.  Without a probe this is a no-op."""
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
        if getattr(self.tok, "chat_template", None):
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
        if len(set(kinds)) > 1:
            mixed: list[Response | None] = [None] * len(reqs)
            for kind in dict.fromkeys(kinds):
                idx = [i for i, k in enumerate(kinds) if k == kind]
                for j, resp in zip(idx, self.complete_batch(
                        [reqs[i] for i in idx]), strict=True):
                    mixed[j] = resp
            return filled(mixed)
        if kinds[0] == "target":
            return self._score_batch(reqs)
        if kinds[0] == "choice":
            return self._choice_batch(reqs)
        if self.generate:
            groups: dict[str, list[int]] = {}
            for i, r in enumerate(reqs):
                p = merge_params(self.default_params, r.params)
                refuse_dropped(p, self.GENERATE_KEYS, "HFBackend.generate")
                sig = decode_signature(p)
                groups.setdefault(sig, []).append(i)
            if len(groups) > 1:
                out: list[Response | None] = [None] * len(reqs)
                for idx in groups.values():
                    resps = self._run_batch([reqs[i] for i in idx])
                    for j, resp in zip(idx, resps, strict=True):
                        out[j] = resp
                return filled(out)
        return self._run_batch(reqs)

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
        does not allocate full logits.  A model whose forward does not take
        `logits_to_keep` falls back to full logits."""
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
        keep = sorted(set(last))
        enc = {**model_in, "padding_side": "right", "last": last}
        t0 = time.perf_counter()
        with torch.no_grad(), self._probing(
                enc, [reqs[i] for i, _, _ in rows]) as probed:
            try:
                model_out = self.model(
                    **model_in, output_attentions=self.output_attentions,
                    logits_to_keep=torch.tensor(keep, device=dev))
                col = {pos: j for j, pos in enumerate(keep)}
            except TypeError:                # forward has no logits_to_keep
                model_out = self.model(**model_in,
                                       output_attentions=self.output_attentions)
                col = {pos: pos for pos in keep}
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
                "latency_s": dt / len(rows), "backend": "hf"})
        return filled(out)

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
               "n_prompt": [n_prompt for _, _, n_prompt in seqs]}
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
        prompts = [self._render(r) for r in reqs]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=not self._templated()
                       ).to(self._device())
        t0 = time.perf_counter()
        # generate() runs one forward per new token, so the per-forward call
        # count is not known in advance and is not checked there
        with torch.no_grad(), self._probing(
                enc, reqs, calls_known=not self.generate) as probed:
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
                    **enc, max_new_tokens=budget,
                    do_sample=p.get("temperature", 0.0) > 0,
                    temperature=p.get("temperature", 1.0) or 1.0,
                    repetition_penalty=p.get("repetition_penalty", 1.0))
                new = gen[:, enc["input_ids"].shape[1]:]
                texts = [t for t in self.tok.batch_decode(
                    new, skip_special_tokens=True)]
                # hit the budget => the generation was cut, not finished
                cut = [bool(int((row != self.tok.pad_token_id).sum()) >= budget)
                       if self.tok.pad_token_id is not None
                       else bool(new.shape[1] >= budget) for row in new]
                out = None
            else:
                out = self.model(**enc, output_attentions=self.output_attentions)
                texts = [None] * len(reqs)
                cut = [False] * len(reqs)
            probed.out = out
        dt = time.perf_counter() - t0
        extras = probed.extras
        return [Response(r.unit_id, text=t,
                         meta={**e, "latency_s": dt / len(reqs), "backend": "hf",
                               "truncated": c})
                for r, t, e, c in zip(reqs, texts, extras, cut, strict=True)]

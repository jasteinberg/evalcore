"""Probes: capture from inside the forward pass, batch-safe, identified,
counted, and always removed."""

from __future__ import annotations

import io

import numpy as np
import pytest

from evalcore import HFBackend, Item, Probe, Request, execute, grid, load_arrays, units

torch = pytest.importorskip("torch")


class CausalMean(torch.nn.Module):
    def forward(self, e):
        return e.cumsum(1) / torch.arange(1, e.shape[1] + 1)[None, :, None]


class TinyLM(torch.nn.Module):
    """Named submodules, so there is something to hook: emb -> mix -> head."""

    def __init__(self, vocab=40, d=8):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, d)
        self.emb.weight.data = torch.randn(vocab, d, generator=g)
        self.mix = CausalMean()
        self.head = torch.nn.Linear(d, vocab, bias=False)
        self.head.weight.data = torch.randn(vocab, d, generator=g)

    def forward(self, input_ids, attention_mask=None, output_attentions=False,
                logits_to_keep=None):
        h = self.mix(self.emb(input_ids))
        if logits_to_keep is not None:
            h = h[:, logits_to_keep, :]

        class Out:
            logits = self.head(h)
        return Out()


class LastHidden(Probe):
    """The `mix` output at each row's final real position -- the batch-safe
    version of the 'read the last column' capture."""

    def __init__(self, tag="v1"):
        self.tag, self.buf = tag, {}

    def identity(self):
        return {"probe": "last_hidden", "module": "mix", "tag": self.tag}

    def hooks(self):
        return {"mix": lambda name, mod, args, out: self.buf.update(h=out)}

    def start(self, enc, reqs):
        self.buf.clear()

    def collect(self, out, enc, reqs):
        h = self.buf["h"]
        return [{"h_last": h[b, enc["last"][b]].detach().numpy()}
                for b in range(len(reqs))]


def be(probe=None, **kw):
    return HFBackend(TinyLM(), None, device="cpu", probe=probe, **kw)


PROMPTS = [[3, 4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14, 15, 16]]


def choice(k, ids):
    return Request(f"u{k}", [], {}, input_ids=ids, candidates=[20, 21])


def test_batched_capture_equals_one_request_at_a_time():
    b = be(LastHidden())
    batch = b.complete_batch([choice(k, p) for k, p in enumerate(PROMPTS)])
    for k, p in enumerate(PROMPTS):
        alone = b.complete_batch([choice(k, p)])[0]
        assert np.allclose(batch[k].meta["h_last"], alone.meta["h_last"])
    # and not trivially: rows differ from each other
    assert not np.allclose(batch[0].meta["h_last"], batch[1].meta["h_last"])


def test_through_the_runner_arrays_land_in_the_sidecar(tmp_path):
    out = tmp_path / "r.jsonl"
    its = [Item(f"q{k}", {"ids": p}) for k, p in enumerate(PROMPTS)]
    df = execute(units(grid({"m": ["toy"]}), its), be(LastHidden()),
             lambda u: Request(u.id, [], {}, input_ids=u.item.payload["ids"],
                               candidates=[20, 21]),
             lambda u, r: {}, out, workers=1, batch_size=3,
             stream=io.StringIO())
    assert (df["status"] == "ok").all()
    arrays = load_arrays(out, df.iloc[0]["meta_arrays"])
    assert arrays["h_last"].shape == (8,)


def test_the_probe_is_part_of_the_backend_identity():
    ids = {str(be(p).identity()) for p in (None, LastHidden("v1"),
                                           LastHidden("v2"))}
    assert len(ids) == 3


def test_a_probe_must_define_identity_and_collect():
    class Half(Probe):
        def collect(self, out, enc, reqs):
            return [{} for _ in reqs]

    with pytest.raises(TypeError, match="identity"):
        Half()


def test_a_module_running_more_often_than_declared_is_an_error():
    class Twice(TinyLM):
        def forward(self, input_ids, **kw):
            self.mix(self.emb(input_ids))            # an extra call
            return super().forward(input_ids, **kw)

    b = HFBackend(Twice(), None, device="cpu", probe=LastHidden())
    with pytest.raises(AssertionError, match="hook call counts differ from 1"):
        b.complete_batch([choice(0, PROMPTS[0])])
    assert not b.model.mix._forward_hooks          # removed anyway



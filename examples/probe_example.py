"""Probe example: capture and steer inside a local model, swept over a scale.

A random-init two-layer Qwen2 (built from a config, nothing downloaded)
reads prompts given as token ids, and each request asks for the
next-token distribution over two candidate ids, A and B.  A Probe adds
alpha * v to the residual stream after the last layer, at the read
position only, where v is the unembedding direction W_U[A] - W_U[B]
(normalised).  To first order this moves the margin

    m = log p(A) - log p(B)

linearly in alpha, as long as alpha is small against the residual's norm
|h|.  The final RMSNorm divides by the norm of h + alpha v, so once
alpha >> |h| the margin saturates at the value v alone would give.  The
grid spans both regimes.  The same probe captures the residual at the
read position before the patch.  That vector goes to an .npz sidecar, not
into the JSONL row, and its norm locates the crossover.

One backend per alpha: the probe's identity carries alpha, so each value
is its own cached model, and the request keys of different alphas never
collide.  `value_curves` then gives the mean margin at each alpha and its
slope d m / d alpha, with intervals from resampling items.

    python examples/probe_example.py OUT_DIR

Needs torch and transformers (`pip install -e ".[hf]"`).

This example calls `execute` (core/runner.py), the harness loop alone,
so that each step is visible.  `Experiment.run()` (pipeline.py) runs the
same loop as its execute stage, with calibration, a cost check, analysis
and a report around it.  `stream=None` turns off the progress lines.

Drafted with the assistance of Claude (Anthropic).
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import transformers

from evalcore import HFBackend, Item, Probe, Request, Unit, execute, grid, units
from evalcore.core.records import load_arrays, to_frame
from evalcore.evaluation.curves import value_curves

pd.set_option("display.width", 200)

VOCAB, WIDTH, LAYER = 64, 32, "model.layers.1"
A, B = 5, 6                                   # the two candidate token ids
ALPHAS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]


def tiny_qwen() -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=VOCAB, hidden_size=WIDTH, intermediate_size=2 * WIDTH,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128)
    return transformers.Qwen2ForCausalLM(cfg).eval().float()


# Prompts in token space need no tokenizer beyond a pad id: nothing is
# encoded, decoded or templated.
TOKENIZER = SimpleNamespace(chat_template=None, pad_token="<pad>",
                            eos_token="<pad>", pad_token_id=0, eos_token_id=0,
                            padding_side="right")


class Steer(Probe):
    """Adds alpha * v at the read position of the hooked layer's output,
    and records that position's residual before the patch."""

    def __init__(self, alpha: float, v: torch.Tensor) -> None:
        self.alpha, self.v = alpha, v
        self.seen: list[torch.Tensor] = []

    def identity(self) -> dict:
        # everything that changes what comes back: the scale, the layer, and
        # how v was made
        return {"probe": "steer", "alpha": self.alpha, "layer": LAYER,
                "v": f"unembed[{A}]-unembed[{B}], unit norm"}

    def hooks(self) -> dict:
        return {LAYER: self._patch}

    def start(self, enc, reqs) -> None:
        # The read position, as a hook index.  enc positions are full-prompt
        # positions.  Under prefix_cache the first n_prefix_reused of them
        # were not fed, so the hook sees index p - n_prefix_reused.  This probe
        # patches only the read position, which is safe under reuse.
        self.read = [p - n for p, n in zip(enc["last"], enc["n_prefix_reused"],
                                           strict=True)]
        self.seen = []

    def _patch(self, name, module, args, out):
        h = out[0] if isinstance(out, tuple) else out     # (batch, seq, width)
        rows = torch.arange(h.shape[0])
        self.seen.append(h[rows, self.read].detach().clone())
        h = h.clone()
        h[rows, self.read] += self.alpha * self.v
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def collect(self, out, enc, reqs) -> list[dict]:
        resid = self.seen[0]
        return [{"resid_read": resid[k].numpy()} for k in range(len(reqs))]


def items(n: int = 30) -> list[Item]:
    rng = np.random.default_rng(1)
    return [Item(f"p{i:02d}", {"ids": rng.integers(10, VOCAB, size=int(
        rng.integers(4, 12))).tolist()}) for i in range(n)]


def render(u: Unit) -> Request:
    return Request(u.id, [], {}, input_ids=u.item.payload["ids"],
                   candidates=[A, B])


def score(u: Unit, r) -> dict:
    la, lb = r.meta["cand_logp"]
    return {"margin": la - lb}


def main(out_dir: str) -> None:
    out = Path(out_dir) / "rows.jsonl"
    model = tiny_qwen()
    w = model.lm_head.weight.detach()
    v = (w[A] - w[B]) / (w[A] - w[B]).norm()
    its = items()
    for a in ALPHAS:
        be = HFBackend(model, TOKENIZER, device="cpu", probe=Steer(a, v))
        execute(units(grid({"alpha": [a]}), its), be, render, score, out,
                workers=1, batch_size=8, stream=None)

    df = to_frame(out)
    print(f"{len(df)} rows, statuses {df['status'].value_counts().to_dict()}")
    norms = [np.linalg.norm(load_arrays(out, rel)["resid_read"])
             for rel in df["meta_arrays"]]
    print(f"sidecars: resid_read of width {WIDTH} per row, median |h| at the "
          f"read position {np.median(norms):.3f}")

    with warnings.catch_warnings():               # printed below instead
        warnings.simplefilter("ignore", RuntimeWarning)
        curves = value_curves(df, "alpha", "margin", log_x=False, n_boot=500)
    print("\nmean margin and its slope d m / d alpha, per alpha:")
    print(curves.table[["alpha", "mean", "mean_lo", "mean_hi",
                        "chi", "chi_lo", "chi_hi"]].round(3).to_string(index=False))
    print("\ncaveats:", *curves.caveats, sep="\n  ")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/probe_example.py OUT_DIR")
    main(sys.argv[1])

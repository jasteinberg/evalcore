"""Stand-ins for a Hugging Face checkpoint: a character tokenizer, a
mean-pool causal LM, a recorder that writes down what the model was fed,
and a random-init Qwen2.  Nothing is downloaded.  Import after
`pytest.importorskip("torch")` (and "transformers" for `tiny_qwen` or a
batched tokenizer call)."""

from __future__ import annotations

import torch

from evalcore import Request

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,:+=\n^"


class CharTok:
    """One token per character, id 0 = "<pad>" = eos.  `merges` are
    multi-character tokens matched greedily ("ab" makes a join between "a"
    and "b" a merge the boundary check must catch).  `bos` adds that
    character unless told not to, and the chat template writes one too:
    the Llama arrangement, where encoding templated text WITH special
    tokens doubles the BOS.  A batched call is LEFT-padded, as HFBackend
    asks for when generating."""

    def __init__(self, merges: tuple[str, ...] = (), chat_template: str | None = None,
                 bos: str | None = None):
        self.vocab = ["<pad>", *merges, *CHARS]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.merges, self.bos = merges, bos
        self.pad_token = self.eos_token = "<pad>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self.chat_template = (chat_template if bos is None
                              else chat_template or "{{ bos_token }}...")

    @property
    def eos_token_id(self) -> int:
        # as in transformers: derived from eos_token, so setting the token
        # moves the id
        return self.index[self.eos_token]

    def encode_one(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.index[self.bos]] if self.bos and add_special_tokens else []
        i = 0
        while i < len(text):
            m = next((m for m in self.merges if text.startswith(m, i)), text[i])
            ids.append(self.index[m])
            i += len(m)
        return ids

    def __call__(self, text, add_special_tokens=True, return_tensors=None,
                 padding=False):
        if isinstance(text, str):
            return {"input_ids": self.encode_one(text, add_special_tokens)}
        import transformers
        rows = [self.encode_one(t, add_special_tokens) for t in text]
        width = max(len(r) for r in rows)
        ids = [[0] * (width - len(r)) + r for r in rows]           # left pad
        mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
        return transformers.BatchEncoding(
            {"input_ids": ids, "attention_mask": mask}, tensor_type="pt")

    def decode(self, ids) -> str:
        return "".join(self.vocab[i] for i in ids)

    def batch_decode(self, rows, skip_special_tokens=True) -> list[str]:
        return ["".join(self.vocab[int(i)] for i in r
                        if not (skip_special_tokens and int(i) == 0))
                for r in rows]

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True) -> str:
        return (self.bos or "") + "".join(m["content"] for m in msgs)


class ToyLM(torch.nn.Module):
    """h_t = mean of the embeddings of x_0..x_t, logits = W h_t.  Causal by
    construction, so right padding cannot reach an earlier position.  Its
    forward has no `logits_to_keep`, so HFBackend's fallback runs."""

    def __init__(self, vocab: int, d: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(vocab, d, generator=g))
        self.out = torch.nn.Parameter(torch.randn(d, vocab, generator=g))

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        e = self.emb[input_ids]
        h = e.cumsum(1) / torch.arange(1, e.shape[1] + 1)[None, :, None]

        class Out:
            logits = h @ self.out
        return Out()


class KeepLM(ToyLM):
    """ToyLM whose forward takes `logits_to_keep` (a tensor of positions),
    as current HF causal LMs do."""

    def forward(self, input_ids, attention_mask=None, output_attentions=False,
                logits_to_keep=None):
        full = super().forward(input_ids).logits
        keep = full if logits_to_keep is None else full[:, logits_to_keep, :]

        class Out:
            logits = keep
        return Out()


class SwallowLM(ToyLM):
    """Takes `logits_to_keep` only through **kwargs and ignores it: full
    logits come back."""

    def forward(self, input_ids, attention_mask=None, output_attentions=False,
                **kwargs):
        return super().forward(input_ids)


class IgnoreKeepLM(ToyLM):
    """Declares `logits_to_keep` and ignores it."""

    def forward(self, input_ids, attention_mask=None, output_attentions=False,
                logits_to_keep=None):
        return super().forward(input_ids)


class Recorder(torch.nn.Module):
    """Records the input_ids of every forward and generate call.  generate()
    returns its input followed by the hand-written rows in `new` (one row
    per prompt; default all id 1), so the generated ids are known without
    running a model."""

    def __init__(self, vocab: int, new: list[list[int]] | None = None):
        super().__init__()
        self.lm = KeepLM(vocab)
        self.new = new
        self.forward_ids: list[list[list[int]]] = []
        self.generate_calls: list[dict] = []

    def forward(self, input_ids, **kw):
        self.forward_ids.append(input_ids.tolist())
        return self.lm(input_ids, **{k: v for k, v in kw.items()
                                     if k == "logits_to_keep"})

    def generate(self, input_ids, **kw):
        self.generate_calls.append({"input_ids": input_ids.tolist(), **kw})
        new = (self.new if self.new is not None
               else [[1] * kw["max_new_tokens"]] * input_ids.shape[0])
        return torch.cat([input_ids, torch.tensor(new, dtype=torch.long)], 1)


def tiny_qwen(vocab: int, n_layers: int, width: int) -> torch.nn.Module:
    """A random-init Qwen2ForCausalLM in float32, eval mode, seed 0."""
    import transformers
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=vocab, hidden_size=width, intermediate_size=2 * width,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512)
    return transformers.Qwen2ForCausalLM(cfg).eval().float()


def user(uid: str, text: str, params: dict | None = None, **kw) -> Request:
    """A one-message chat request; `kw` are Request fields (target,
    candidates, ...)."""
    return Request(uid, [{"role": "user", "content": text}], params or {}, **kw)


class EmbedTok:
    """For HFEmbedder: characters as tokens; pads on `side`, returns torch
    tensors."""

    def __init__(self, side="right"):
        self.padding_side, self.pad_token, self.eos_token = side, "#", "#"

    def __call__(self, texts, padding=False, truncation=False, max_length=None,
                 return_tensors=None):
        if isinstance(texts, str):
            return {"input_ids": [ord(c) % 50 + 1 for c in texts]}
        ids = [[ord(c) % 50 + 1 for c in t] for t in texts]
        if truncation:
            ids = [x[:max_length] for x in ids]
        T = max(map(len, ids))
        pad = [[0] * (T - len(x)) for x in ids]
        rows = [x + p if self.padding_side == "right" else p + x
                for x, p in zip(ids, pad, strict=True)]
        mask = [[int(t != 0) for t in r] for r in rows]
        return {"input_ids": torch.tensor(rows),
                "attention_mask": torch.tensor(mask)}


class MeanEncoder(torch.nn.Module):
    """A causal toy: h_t is the masked running mean of the embeddings up to
    t, so a padded position cannot change a real one."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Parameter(torch.randn(64, 8, generator=g))

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        m = attention_mask.unsqueeze(-1).to(self.emb.dtype)
        e = self.emb[input_ids] * m
        h = e.cumsum(1) / m.cumsum(1).clamp(min=1)
        return type("Out", (), {"hidden_states": (e, h)})()

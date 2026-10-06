"""Worked example: what actually flows through the harness.

A two-arm multiple-choice eval with paraphrase augmentation, run against a
scripted stand-in model so the numbers are reproducible and cost nothing.
Prints one concrete object at each stage of the pipeline.

Drafted with the assistance of Claude (Anthropic).
"""
import json

import numpy as np
import pandas as pd

from evalcore import (
    Item,
    ParseFailure,
    Request,
    Response,
    Unit,
    execute,
    grid,
    paired_bootstrap,
    paired_pvalue,
    summarize,
    units,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
RNG = np.random.default_rng(11)

TOPICS = ["thermodynamics", "linear algebra", "electromagnetism", "optics"]
N_SOURCE, N_PARA = 40, 2
DISCRIM = 1.6                       # IRT 2PL slope, shared for simplicity
ABILITY = {"model-a": 0.85, "model-b": 0.35}


def build_items():
    """40 source questions, each with 2 paraphrases -> 120 rows per arm.
    The paraphrase carries parent_id, so all 3 share a cluster."""
    items = []
    for i in range(N_SOURCE):
        sid = f"q{i:03d}"
        topic = TOPICS[i % len(TOPICS)]
        b = float(RNG.normal(0, 1.0))                 # latent difficulty
        opts = ["A", "B", "C", "D"]
        base = dict(question=f"[{topic}] Which statement is correct? (item {i})",
                    options=opts, gold=opts[i % 4], difficulty=b)
        items.append(Item(sid, base, group=topic))
        for p in range(N_PARA):
            q = (f"[{topic}] Consider the following. Which is true? "
                 f"(item {i}, rewrite {p})")
            items.append(Item(f"{sid}-p{p}", {**base, "question": q},
                              group=topic, parent_id=sid))
    return items


TEMPLATES = {
    "plain": "{q}\n\n{opts}\n\nAnswer with a single letter.",
    "cot": "{q}\n\n{opts}\n\nThink step by step, then end with 'Answer: X'.",
}


def render(u: Unit) -> Request:
    """Unit -> Request.  This is the ONLY place the task format lives."""
    p = u.item.payload
    opts = "\n".join(f"({o}) option {o} for this item" for o in p["options"])
    body = TEMPLATES[u.cell.get("template")].format(q=p["question"], opts=opts)
    return Request(
        unit_id=u.id,
        messages=[{"role": "system", "content": "You are a physics examiner."},
                  {"role": "user", "content": body}],
        params={"model": u.cell.get("model"), "temperature": u.cell.get("temperature"),
                "max_tokens": 8 if u.cell.get("template") == "plain" else 256},
    )


class ScriptedModel:
    """Stand-in for an API. P(correct) = sigma(a(ability - difficulty)),
    plus a small CoT bonus, so arm A really is better and the paraphrases
    of one source item really are correlated."""

    def __call__(self, req: Request) -> str:
        seed = int(req.cache_key[:8], 16)
        rng = np.random.default_rng(seed)
        meta = REGISTRY[req.unit_id]
        z = DISCRIM * (ABILITY[meta["model"]] + meta["cot"] - meta["difficulty"])
        correct = rng.random() < 1 / (1 + np.exp(-z))
        pick = meta["gold"] if correct else rng.choice(
            [o for o in "ABCD" if o != meta["gold"]])
        if meta["template"] == "cot":
            return (f"The relevant relation fixes the sign of the response, "
                    f"which rules out two options.\nAnswer: {pick}")
        return pick


def score(u: Unit, r: Response) -> dict:
    """Response -> metrics.  Parsing lives here, and a response with no
    letter in it raises ParseFailure: the row is marked `unparsed` and
    counted in attrition, rather than scored as an incorrect answer."""
    txt = (r.text or "").strip()
    tail = txt.split("Answer:")[-1].strip() if "Answer:" in txt else txt
    pred = next((c for c in tail if c in "ABCD"), None)
    if pred is None:
        raise ParseFailure(f"no option letter in {txt[-60:]!r}")
    return {"pred": pred, "correct": float(pred == u.item.payload["gold"]),
            "resp_chars": len(txt)}


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


if __name__ == "__main__":
    from evalcore.core.backends import EchoBackend

    items = build_items()
    cells = grid({"model": ["model-a", "model-b"], "template": ["plain", "cot"],
                  "temperature": [0.0]})
    all_units = list(units(cells, items, repeats=1))

    REGISTRY = {u.id: {"model": u.cell.get("model"),
                       "template": u.cell.get("template"),
                       "cot": 0.35 * (u.cell.get("template") == "cot"),
                       "difficulty": u.item.payload["difficulty"],
                       "gold": u.item.payload["gold"]} for u in all_units}

    rule("1. ITEM  (one source + one of its paraphrases)")
    for it in (items[0], items[1]):
        print(json.dumps({"item_id": it.item_id, "group": it.group,
                          "parent_id": it.parent_id, "cluster": it.cluster,
                          "payload": it.payload}, indent=2, default=str))

    rule("2. CELL  (one point of the 2x2x1 grid)")
    c = cells[3]
    print(f"params = {dict(c.params)}\ncell_id = {c.id}")

    rule("3. REQUEST  (what leaves for the model)")
    u = next(x for x in all_units if x.cell.id == c.id and x.item.item_id == "q000")
    req = render(u)
    print(json.dumps({"unit_id": req.unit_id, "cache_key": req.cache_key,
                      "params": dict(req.params),
                      "messages": [dict(m) for m in req.messages]}, indent=2))

    rule("4. RESPONSE  (what comes back)")
    resp = EchoBackend(fn=ScriptedModel()).complete(req)
    print(json.dumps({"unit_id": resp.unit_id, "text": resp.text,
                      "meta": resp.meta, "error": resp.error,
                      "cached": resp.cached}, indent=2))

    rule("5. SCORE  (Response -> metrics)")
    print(json.dumps(score(u, resp), indent=2))

    rule("6. RUN  -> tidy frame (one row per unit)")
    df = execute(all_units, EchoBackend(fn=ScriptedModel()), render, score,
             out="/tmp/evalcore_demo/rows.jsonl", workers=4, log_every=10_000)
    cols = ["cell_id", "model", "template", "item_id", "cluster", "group",
            "repeat", "status", "pred", "correct", "resp_chars"]
    print(f"\nframe shape = {df.shape}")
    print("\none row, transposed:")
    print(df.loc[df.item_id == "q000"].iloc[0][cols].to_string())
    print("\nsame source item, 3 rows (original + 2 paraphrases), one arm:")
    sel = df[(df.cluster == "q000") & (df.model == "model-a")
             & (df.template == "cot")]
    print(sel[["item_id", "cluster", "pred", "correct", "resp_chars"]]
          .to_string(index=False))

    rule("7. SUMMARIZE  (per-cell point + CI + honest n)")
    s = summarize(df, "correct", by=["model", "template"], n_boot=4000)
    print(s[["model", "template", "point", "lo", "hi", "se", "n_clusters",
             "n_rows", "rho", "deff", "n_eff"]].round(3).to_string(index=False))

    rule("8. WHAT THE ROW BOOTSTRAP WOULD HAVE CLAIMED")
    sub = df[(df.model == "model-a") & (df.template == "cot")]
    v = sub["correct"].to_numpy(float)
    rg = np.random.default_rng(0)
    reps = v[rg.integers(0, v.size, (4000, v.size))].mean(1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    row_hw = (hi - lo) / 2
    cl = s[(s.model == "model-a") & (s.template == "cot")].iloc[0]
    print(f"row-resampled  half-width = {row_hw:.4f}   (N   = {len(sub)})")
    print(f"cluster        half-width = {(cl.hi-cl.lo)/2:.4f}   "
          f"(k   = {int(cl.n_clusters)})")
    print(f"ratio = {((cl.hi-cl.lo)/2)/row_hw:.2f}   "
          f"sqrt(DEFF) = {np.sqrt(cl.deff):.2f}")

    rule("9. PAIRED CONTRAST  (model-a - model-b, cot template)")
    d = df[df.template == "cot"]
    pb = paired_bootstrap(d, "correct", "model", "model-a", "model-b", n_boot=8000)
    print(f"delta = {pb.point:+.4f}   95% CI [{pb.lo:+.4f}, {pb.hi:+.4f}]"
          f"   se = {pb.se:.4f}   p = {paired_pvalue(pb):.4f}   k = {pb.n_clusters}")
    piv = (d.groupby(["cluster", "model"], observed=True)["correct"].mean()
             .unstack("model"))
    print(f"across-item correlation r = {piv.corr().iloc[0,1]:.3f}")
    print("\nunpaired, for contrast:")
    for m in ("model-a", "model-b"):
        r = s[(s.model == m) & (s.template == "cot")].iloc[0]
        print(f"  {m}: {r.point:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
    print("  -> the two intervals overlap while the paired difference does not "
          "contain zero")

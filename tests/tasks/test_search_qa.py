"""Generated search QA: gold answers re-derived from the documents, and
every question answerable by searching within its stated number of hops."""

from __future__ import annotations

import io
import re

from evalcore import AgentBackend, FunctionBackend, Response, grid
from evalcore.tasks.search_qa import make_world, search_qa


def test_world_documents_state_every_fact_and_names_are_unique():
    w = make_world(20, seed=3)
    text = "\n".join(w.docs.values())
    names = list(w.towns) + list(w.rivers) + list(w.people)
    assert len(names) == len(set(names))
    for t, f in w.towns.items():
        assert f"{t} is a town with a population of {f['population']:,}" in text
    for r, f in w.rivers.items():
        assert f"The {r} is a river that rises in {f['source']}." in text
    assert make_world(20, seed=3).docs == w.docs


def rederive(docs: dict, q: str):
    """The gold answer, from the document text alone."""
    body = "\n".join(docs.values())
    pop = lambda town: int(re.search(rf"\b{town} is a town with a population "   # noqa: E731
                                     rf"of ([\d,]+)", body).group(1).replace(",", ""))
    if m := re.fullmatch(r"How many people live in (\w+)\?", q):
        return pop(m.group(1))
    if m := re.fullmatch(r"How many people live in the town where the (\w+) "
                         r"rises\?", q):
        town = re.search(rf"The {m.group(1)} is a river that rises in (\w+)\.",
                         body).group(1)
        return pop(town)
    m = re.fullmatch(r"Which river flows through the town where (\w+ \w+) was "
                     r"born\?", q)
    town = re.search(rf"{m.group(1)} is a surveyor who was born in (\w+)\.",
                     body).group(1)
    return re.search(rf"\b{town} is a town [^.]*\. The (\w+) flows", body).group(1)


def test_every_gold_answer_is_rederived_from_the_documents():
    task, search = search_qa(24, seed=5)
    for it in task.items:
        assert rederive(search.docs, it.payload["messages"][1]["content"]) \
            == it.payload["gold"], it.item_id


def hop_follower(req):
    """Search for the entity the question names; if the hit names the next
    entity (a town), search for that; then answer from the last result.
    It never looks at the gold."""
    q = req.messages[1]["content"]
    results = [m["content"] for m in req.messages if m.get("role") == "tool"]
    if not results:
        ent = re.search(r"(?:live in (?:the town where the )?|where )"
                        r"(\w+(?: \w+)?)(?: rises| was born|\?)", q).group(1)
        return Response("x", meta={"tool_calls": [
            {"id": "1", "name": "search", "arguments": {"query": ent}}]})
    last = results[-1]
    if len(results) == 1 and ("rises" in q or "born" in q):
        town = re.search(r"(?:rises in|born in) (\w+)\.", last).group(1)
        return Response("x", meta={"tool_calls": [
            {"id": "2", "name": "search", "arguments": {"query": town}}]})
    if q.startswith("Which river"):
        town = re.search(r"born in (\w+)\.", results[0]).group(1)
        ans = re.search(rf"\b{town} is a town [^.]*\. The (\w+) flows", last).group(1)
    else:
        name = re.search(r"live in (\w+)\?", q)
        town = name.group(1) if name else re.search(r"rises in (\w+)\.",
                                                    results[0]).group(1)
        ans = re.search(rf"\b{town} is a town with a population of ([\d,]+)",
                        last).group(1)
    return Response("x", text=ans, meta={"truncated": False})


def test_every_question_is_answerable_within_its_hops(tmp_path):
    task, search = search_qa(16, seed=1)
    agent = AgentBackend(FunctionBackend(hop_follower, identity={"p": "hops"},
                                         supports_tools=True), [search.tool])
    df = task.run(grid({"model": ["toy"]}), agent, tmp_path / "r.jsonl",
                  workers=1, stream=io.StringIO())
    assert (df["status"] == "ok").all(), df[df["status"] != "ok"]["error"].tolist()
    assert (df["correct"] == 1.0).all()
    hops = {i.item_id: i.payload["meta"]["hops"] for i in task.items}
    assert (df["meta_n_tool_calls"] == df["item_id"].map(hops)).all()


def test_closed_book_cannot_answer(tmp_path):
    task, _ = search_qa(16, seed=1)
    guess = FunctionBackend(lambda r: "12000", identity={"p": "guess"})
    df = task.run(grid({"model": ["toy"]}), guess, tmp_path / "r.jsonl",
                  workers=1, stream=io.StringIO())
    numeric = df[df["group"] != "person_river"]
    assert (numeric["correct"] == 0.0).all()
    assert (df[df["group"] == "person_river"]["status"] == "unparsed").all()

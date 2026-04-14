from tests.ai.fakes import FakeGraph, FakeNode
from synapse.ai.context import graph_summary


def _chain(types: list[str]) -> FakeGraph:
    g = FakeGraph()
    nodes = [FakeNode(f"n{i+1}", t) for i, t in enumerate(types)]
    for n in nodes:
        g.add_node(n)
    for i in range(len(nodes) - 1):
        out = nodes[i].add_output("out_1")
        inn = nodes[i+1].add_input("in_1")
        out.connect_to(inn)
    return g


def test_graph_summary_empty():
    assert graph_summary(FakeGraph()) == "Canvas: empty."


def test_graph_summary_single_node():
    g = FakeGraph(); g.add_node(FakeNode("a", "CSVLoader"))
    s = graph_summary(g)
    assert s.startswith("Canvas: 1 node")
    assert "CSVLoader" in s


def test_graph_summary_linear_chain_shows_arrows():
    g = _chain(["CSVLoader", "ParticleProps", "SortTable", "TopN"])
    s = graph_summary(g)
    assert "4 nodes" in s
    assert "CSVLoader" in s and "TopN" in s
    assert "→" in s or "->" in s


def test_graph_summary_nonlinear_drops_arrow_notation():
    g = FakeGraph()
    a = FakeNode("a", "A"); b = FakeNode("b", "B")
    c = FakeNode("c", "C"); d = FakeNode("d", "D")
    for n in (a, b, c, d):
        g.add_node(n)
    a.add_output("o").connect_to(b.add_input("i"))
    c.add_output("o").connect_to(d.add_input("i"))
    s = graph_summary(g)
    assert "4 nodes" in s
    assert "disconnected" in s.lower() or "branches" in s.lower()


def test_graph_summary_under_200_chars():
    g = _chain(["Node"] * 50)
    assert len(graph_summary(g)) < 200


# --- estimate_tokens ------------------------------------------------------

def test_estimate_tokens_empty():
    from synapse.ai.context import estimate_tokens
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_short_text():
    from synapse.ai.context import estimate_tokens
    # 12-char string → 12//4 = 3
    assert estimate_tokens("hello world!") == 3


def test_estimate_tokens_monotonic():
    from synapse.ai.context import estimate_tokens
    short = estimate_tokens("hi")
    long_ = estimate_tokens("hi " * 1000)
    assert long_ > short


# --- HistoryRoller --------------------------------------------------------

def _msgs(pairs: list[tuple[str, str]]) -> list[dict]:
    return [{"role": r, "content": c} for r, c in pairs]


def test_history_roller_keeps_short_history_unchanged():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=8)
    msgs = _msgs([
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "what's a ROIMask?"),
        ("assistant", "a node that defines a region of interest."),
    ])
    assert roller.roll(msgs) == msgs


def test_history_roller_trims_beyond_window():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=2)
    msgs = []
    for i in range(4):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    out = roller.roll(msgs)
    assert out[0]["role"] == "system"
    assert "trimmed" in out[0]["content"].lower()
    assert {"role": "user", "content": "q2"} in out
    assert {"role": "user", "content": "q3"} in out


def test_history_roller_truncates_old_tool_results():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=8)
    msgs = [
        {"role": "user", "content": "turn 1"},
        {"role": "tool", "content": "BIG RESULT " * 100},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn 3"},  # current turn
    ]
    out = roller.roll(msgs)
    tool_msg = next(m for m in out if m["role"] == "tool")
    assert "truncated" in tool_msg["content"].lower()
    assert len(tool_msg["content"]) < 100


def test_history_roller_keeps_recent_tool_results():
    from synapse.ai.context import HistoryRoller
    roller = HistoryRoller(keep_turns=8)
    msgs = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "turn 2"},
        {"role": "tool", "content": "RECENT RESULT " * 100},
        {"role": "assistant", "content": "ok"},
    ]
    out = roller.roll(msgs)
    tool_msg = next(m for m in out if m["role"] == "tool")
    assert "RECENT RESULT" in tool_msg["content"]
    assert "truncated" not in tool_msg["content"].lower()

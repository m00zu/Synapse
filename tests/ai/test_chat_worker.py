import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore
from unittest.mock import MagicMock

from tests.ai.fakes import FakeGraph
from synapse.ai.clients.base import StreamEvent
from synapse.ai.chat_worker import ChatStreamWorker
from synapse.ai.tools import ToolDispatcher


class _Collector(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.tokens: list[str] = []
        self.tool_started: list[tuple[str, dict]] = []
        self.tool_finished: list[tuple[str, dict]] = []
        self.errors: list[str] = []
        self.turns_finished = 0

    def on_token(self, t: str):
        self.tokens.append(t)

    def on_tool_started(self, name: str, inp: dict):
        self.tool_started.append((name, inp))

    def on_tool_finished(self, name: str, result: dict):
        self.tool_finished.append((name, result))

    def on_error(self, msg: str):
        self.errors.append(msg)

    def on_turn_finished(self):
        self.turns_finished += 1


def _client(events):
    c = MagicMock()
    def stream(*a, **k):
        for e in events:
            yield e
    c.chat_with_tools_stream.side_effect = stream
    c.model = "m"
    return c


def test_worker_forwards_text_tokens_and_turn_finished():
    g = FakeGraph()
    client = _client([
        StreamEvent(kind="text", text="hi "),
        StreamEvent(kind="text", text="there"),
        StreamEvent(kind="done"),
    ])
    w = ChatStreamWorker(graph=g, client=client, dispatcher=ToolDispatcher(),
                         history=[], user_text="x")
    collector = _Collector()
    w.token_received.connect(collector.on_token)
    w.turn_finished.connect(collector.on_turn_finished)
    w._run_once()
    assert "".join(collector.tokens) == "hi there"
    assert collector.turns_finished == 1


def test_worker_emits_error_signal_on_client_error():
    g = FakeGraph()
    client = _client([StreamEvent(kind="error", error="boom")])
    w = ChatStreamWorker(graph=g, client=client, dispatcher=ToolDispatcher(),
                         history=[], user_text="x")
    collector = _Collector()
    w.error.connect(collector.on_error)
    w.turn_finished.connect(collector.on_turn_finished)
    w._run_once()
    assert collector.errors == ["boom"]


def test_worker_cancel_stops_mid_stream():
    g = FakeGraph()
    client = _client([
        StreamEvent(kind="text", text="a"),
        StreamEvent(kind="text", text="b"),
        StreamEvent(kind="text", text="c"),
        StreamEvent(kind="done"),
    ])
    w = ChatStreamWorker(graph=g, client=client, dispatcher=ToolDispatcher(),
                         history=[], user_text="x")
    collector = _Collector()
    w.token_received.connect(collector.on_token)
    w.turn_finished.connect(collector.on_turn_finished)
    def cancel_after_first(_):
        if len(collector.tokens) == 1:
            w.request_cancel()
    w.token_received.connect(cancel_after_first)
    w._run_once()
    assert collector.turns_finished >= 1


def test_worker_emits_workflow_preview_signal_after_generate_workflow():
    g = FakeGraph()
    # Simulate orchestrator producing a tool_call_finished for generate_workflow
    # followed by a text reply. Since ChatStreamWorker directly reads the
    # orchestrator's events, we stub the orchestrator's behaviour by feeding
    # the client a stream that produces the tool_call. We then register a fake
    # generate_workflow handler that returns a preview-like dict.
    from synapse.ai.tool_handlers import __init__  # just to ensure package load
    client = _client([
        StreamEvent(kind="tool_call",
                    tool_call={"id": "t1", "name": "generate_workflow",
                               "input": {"goal": "x"}}),
        StreamEvent(kind="done"),
    ])
    d = ToolDispatcher()
    d.register("generate_workflow", lambda inp: {
        "node_count": 1, "edge_count": 0,
        "preview_types": ["X"], "workflow": {"nodes": [], "edges": []},
        "canvas_was_empty": True,
    })
    w = ChatStreamWorker(graph=g, client=client, dispatcher=d,
                         history=[], user_text="build something")

    preview_payloads: list[dict] = []
    w.workflow_preview.connect(lambda d_: preview_payloads.append(d_))

    # Need a second streaming batch so the orchestrator doesn't hang waiting —
    # reset side_effect to a fresh iterator that gives done.
    batches = iter([
        [StreamEvent(kind="tool_call",
                     tool_call={"id": "t1", "name": "generate_workflow",
                                "input": {"goal": "x"}}),
         StreamEvent(kind="done")],
        [StreamEvent(kind="text", text="built!"),
         StreamEvent(kind="done")],
    ])
    def stream2(*a, **k):
        for e in next(batches):
            yield e
    client.chat_with_tools_stream.side_effect = stream2

    w._run_once()
    assert len(preview_payloads) == 1
    assert preview_payloads[0]["canvas_was_empty"] is True
    assert preview_payloads[0]["node_count"] == 1

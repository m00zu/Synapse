import pytest
from synapse.ai.tools import ToolDispatcher, TOOL_NAMES


def test_register_and_dispatch_roundtrip():
    d = ToolDispatcher()
    d.register("inspect_canvas", lambda inp: {"echo": inp})
    assert d.registered_names() == ("inspect_canvas",)
    assert d.dispatch("inspect_canvas", {"node_ids": ["x"]}) == {"echo": {"node_ids": ["x"]}}


def test_register_rejects_unknown_name():
    d = ToolDispatcher()
    with pytest.raises(ValueError):
        d.register("not_a_tool", lambda _: {})


def test_register_rejects_duplicates():
    d = ToolDispatcher()
    d.register("explain_node", lambda _: {})
    with pytest.raises(ValueError):
        d.register("explain_node", lambda _: {})


def test_dispatch_unknown_name_returns_error():
    d = ToolDispatcher()
    out = d.dispatch("explain_node", {"node_type": "X"})
    assert "error" in out and "No handler" in out["error"]


def test_dispatch_wraps_handler_exception_into_error():
    d = ToolDispatcher()
    def boom(_inp):
        raise RuntimeError("kaboom")
    d.register("explain_node", boom)
    out = d.dispatch("explain_node", {"node_type": "X"})
    assert out == {"error": "RuntimeError: kaboom"}


def test_dispatch_rejects_non_dict_handler_return():
    d = ToolDispatcher()
    d.register("explain_node", lambda _: "oops, not a dict")
    out = d.dispatch("explain_node", {"node_type": "X"})
    assert "error" in out and "non-dict" in out["error"]


def test_all_six_tool_names_are_registrable():
    d = ToolDispatcher()
    for name in TOOL_NAMES:
        d.register(name, lambda _inp, n=name: {"name": n})
    assert set(d.registered_names()) == set(TOOL_NAMES)

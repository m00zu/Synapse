import asyncio
import pytest

pytest.importorskip("PySide6")



@pytest.mark.asyncio
async def test_executor_emits_started_finished_per_node():
    from synapse.server.session import SessionState
    from synapse.server.executor import run_graph
    s = SessionState()
    src = s.graph.add_node("ImageReadNode")
    dst = s.graph.add_node("BinaryThresholdNode")
    s.graph.connect(src, dst)
    events = []
    async for ev in run_graph(s):
        events.append(ev)
    kinds = [e["kind"] for e in events]
    assert kinds.count("node_started") == 2
    assert kinds.count("node_finished") == 2


@pytest.mark.asyncio
async def test_executor_stop_flag_interrupts():
    from synapse.server.session import SessionState
    from synapse.server.executor import Executor
    s = SessionState()
    for _ in range(5):
        s.graph.add_node("ImageReadNode")
    exe = Executor(s)
    exe.request_stop()
    events = [e async for e in exe.run()]
    # Executor should observe the flag before starting any node.
    assert not any(e["kind"] == "node_started" for e in events)

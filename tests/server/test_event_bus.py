import asyncio
import pytest

from synapse.server.event_bus import EventBus


@pytest.mark.asyncio
async def test_subscriber_receives_published_event():
    bus = EventBus()
    q = bus.subscribe()
    await bus.publish({"kind": "node_started", "node_id": "n1"})
    got = await asyncio.wait_for(q.get(), timeout=1.0)
    assert got["node_id"] == "n1"


@pytest.mark.asyncio
async def test_multiple_subscribers_fan_out():
    bus = EventBus()
    q1 = bus.subscribe(); q2 = bus.subscribe()
    await bus.publish({"kind": "x"})
    assert (await q1.get())["kind"] == "x"
    assert (await q2.get())["kind"] == "x"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    await bus.publish({"kind": "x"})
    assert q.empty()

"""Tiny fan-out event bus for per-session WS messaging."""
from __future__ import annotations

import asyncio


class EventBus:
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subs.remove(q)
        except ValueError:
            pass

    async def publish(self, event: dict) -> None:
        for q in list(self._subs):
            await q.put(event)

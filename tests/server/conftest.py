"""Shared fixtures for the synapse.server test suite.

Design notes (hard-won):

- ``SessionState.__init__`` creates a ``NodeGraph()`` which instantiates a
  ``QGraphicsView``. Qt widgets are **main-thread-only**. A session-scoped
  *async* fixture can end up running on pytest-asyncio's worker thread and
  crash Qt with "Fatal Python error: Aborted".

  Fix: boot the app SYNCHRONOUSLY in a session-scoped fixture (``_booted_app``)
  so Qt construction happens on the main thread.

- ``NodeGraphHeadless.__init__`` walks ~160 node subclasses and calls
  ``register_node(cls)`` on each. That's cheap per call (no instantiation),
  but re-doing it for every test is wasteful. Session-scope amortizes it.

- Between tests, we reset the *graph* (via ``session.graph.clear()``) so each
  test sees an empty canvas. The expensive one-time class registration is
  preserved.
"""
import asyncio
import os

import pytest
import pytest_asyncio
import httpx


@pytest.fixture(scope="session")
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(scope="session")
def _booted_app(qapp):
    """Boot the FastAPI app once per test session. Synchronous so Qt widget
    construction happens on the main thread.

    We hand-wire app.state instead of running ``async with lifespan(app)``
    because the lifespan is async and would run on a worker thread under
    session-scoped pytest-asyncio fixtures."""
    from synapse.server.app import app
    from synapse.server.session import SessionState

    # Match lifespan setup without the async wrapper.
    app.state.catalog = None  # built lazily on first /api/nodes call
    app.state.session = SessionState(allow_path=os.environ.get("SYNAPSE_ALLOW_PATH"))

    yield app

    # Teardown — equivalent to the async with exit.
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        app.state.session.aclose()
    )


@pytest_asyncio.fixture
async def client(_booted_app):
    """Per-test httpx client. Resets the graph so each test sees empty state."""
    _booted_app.state.session.graph.clear()
    from httpx import ASGITransport
    async with httpx.AsyncClient(
        transport=ASGITransport(app=_booted_app), base_url="http://testserver"
    ) as c:
        yield c

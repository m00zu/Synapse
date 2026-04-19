"""FastAPI app factory + lifespan.

Lifespan:
  - On startup: bootstrap QApplication (headless), build widget catalog,
    build SessionState.
  - On shutdown: close session + drain preview-dir.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


_PLACEHOLDER_INDEX = """<!doctype html>
<html><head><meta charset="utf-8"><title>Synapse</title></head>
<body style="font-family:system-ui;padding:2rem;color:#c9d1d9;background:#0d1117">
<h1>Synapse serve</h1>
<p>Phase 1b placeholder. The React frontend ships in Phase 1c.</p>
<p>HTTP API is live at <a href="/docs" style="color:#58a6ff">/docs</a>.</p>
</body></html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure a QApplication exists before any node class is instantiated.
    from PySide6 import QtWidgets
    _qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Catalog is built lazily on first /api/nodes call — eager build here
    # double-instantiates every node class (once for spec capture, once for
    # NodeGraphQt registration in SessionState) which crashes Qt on macOS.
    app.state.catalog = None

    from synapse.server.session import SessionState
    app.state.session = SessionState(allow_path=os.environ.get("SYNAPSE_ALLOW_PATH"))

    yield

    await app.state.session.aclose()


def _get_catalog(app: FastAPI) -> dict:
    """Lazy catalog accessor — builds on first call, caches on app.state."""
    if app.state.catalog is None:
        from synapse.widgets.catalog import collect_widget_catalog
        app.state.catalog = collect_widget_catalog()
    return app.state.catalog


def create_app() -> FastAPI:
    app = FastAPI(title="Synapse serve", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return _PLACEHOLDER_INDEX

    from synapse.server.routes_nodes import router as nodes_router
    from synapse.server.routes_graph import router as graph_router
    from synapse.server.routes_exec import router as exec_router
    from synapse.server.ws import router as ws_router
    from synapse.server.routes_files import router as files_router
    app.include_router(nodes_router)
    app.include_router(graph_router)
    app.include_router(exec_router)
    app.include_router(ws_router)
    app.include_router(files_router)

    return app


app = create_app()

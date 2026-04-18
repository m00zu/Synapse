"""Shared fixtures for the synapse.server test suite."""
import pytest
import pytest_asyncio
import httpx


@pytest_asyncio.fixture
async def client():
    """ASGI-transported httpx client so we never bind a real port in tests."""
    from synapse.server.app import app, lifespan
    from httpx import ASGITransport
    # Run the lifespan context so app.state.session and .catalog are set up.
    async with lifespan(app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as c:
            yield c

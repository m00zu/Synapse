"""Smoke test: the synapse.mcp package can be imported."""


def test_package_imports():
    import synapse.mcp as mcp
    assert hasattr(mcp, '__name__')


def test_mcp_sdk_available():
    from mcp.server.fastmcp import FastMCP
    assert FastMCP is not None

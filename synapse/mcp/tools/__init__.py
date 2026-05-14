"""MCP tool factories -- each returns a plain callable suitable for FastMCP.

Tools accept a ``GraphController`` as first arg (closed over at registration
time) so they're trivially testable with FakeGraphController.
"""

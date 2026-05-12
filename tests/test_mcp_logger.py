"""Tests for MCPLogger thread-safety + ring buffer + snapshot."""
import threading

import pytest
from PySide6 import QtWidgets


@pytest.fixture(scope='module')
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _fresh_logger(qapp):
    """Get a fresh logger instance, resetting the singleton between tests."""
    from synapse.mcp import logger as logger_mod
    logger_mod.MCPLogger._instance = None
    return logger_mod.MCPLogger.instance()


def test_log_records_entry(qapp):
    from synapse.mcp.logger import MCPLogEntry
    log = _fresh_logger(qapp)
    e = MCPLogEntry(timestamp=1.0, tool='list_nodes',
                    args={}, success=True,
                    duration_ms=12.0, result_summary='3 nodes')
    log.log(e)
    snap = log.snapshot()
    assert len(snap) == 1
    assert snap[0].tool == 'list_nodes'
    assert snap[0].success is True


def test_ring_buffer_caps_at_max(qapp):
    from synapse.mcp.logger import MCPLogEntry
    log = _fresh_logger(qapp)
    # Override the cap for fast testing.
    log._entries.clear()
    cap = log.MAX_ENTRIES
    for i in range(cap + 50):
        log.log(MCPLogEntry(timestamp=float(i), tool='t', args={}))
    snap = log.snapshot()
    assert len(snap) == cap
    # Oldest 50 should have been dropped; newest preserved.
    assert snap[0].timestamp == float(50)
    assert snap[-1].timestamp == float(cap + 49)


def test_clear_empties_buffer(qapp):
    from synapse.mcp.logger import MCPLogEntry
    log = _fresh_logger(qapp)
    log.log(MCPLogEntry(timestamp=1.0, tool='t', args={}))
    log.log(MCPLogEntry(timestamp=2.0, tool='t', args={}))
    log.clear()
    assert log.snapshot() == []


def test_log_thread_safe(qapp):
    """Multiple producer threads should not corrupt the buffer."""
    from synapse.mcp.logger import MCPLogEntry
    log = _fresh_logger(qapp)
    n_threads, n_per = 5, 200

    def producer(tid: int):
        for i in range(n_per):
            log.log(MCPLogEntry(timestamp=float(tid * 1000 + i),
                                 tool=f'thr{tid}', args={'i': i}))

    threads = [threading.Thread(target=producer, args=(t,))
               for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = log.snapshot()
    expected = n_threads * n_per
    assert len(snap) == min(expected, log.MAX_ENTRIES)
    # All entries should be intact MCPLogEntry instances.
    for e in snap:
        assert isinstance(e, MCPLogEntry)
        assert e.tool.startswith('thr')

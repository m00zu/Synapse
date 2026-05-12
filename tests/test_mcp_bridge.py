"""Test ThreadHop: schedule a callable on the Qt main thread and block."""
import threading
import time

import pytest
from PySide6 import QtCore, QtWidgets


@pytest.fixture(scope='module')
def qapp():
    """Provide a single QApplication for all tests in the module."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_threadhop_same_thread_runs_inline(qapp):
    from synapse.mcp.bridge import ThreadHop
    hop = ThreadHop()
    result = hop.call(lambda: 42)
    assert result == 42


def test_threadhop_cross_thread_returns_value(qapp):
    from synapse.mcp.bridge import ThreadHop
    hop = ThreadHop()

    # Track which thread ran the body.
    main_thread = QtCore.QThread.currentThread()
    ran_on = []

    def body():
        ran_on.append(QtCore.QThread.currentThread())
        return 'ok'

    # Run hop.call from a worker thread; pump the Qt event loop on main.
    result_box = {}

    def worker():
        result_box['value'] = hop.call(body)

    t = threading.Thread(target=worker)
    t.start()
    # Pump events until worker finishes
    deadline = time.time() + 2.0
    while t.is_alive() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    t.join(timeout=1.0)
    assert not t.is_alive(), 'worker did not finish'
    assert result_box['value'] == 'ok'
    assert ran_on[0] is main_thread


def test_threadhop_propagates_exception(qapp):
    from synapse.mcp.bridge import ThreadHop
    hop = ThreadHop()

    def boom():
        raise ValueError('boom!')

    result_box = {}

    def worker():
        try:
            hop.call(boom)
        except Exception as e:
            result_box['err'] = e

    t = threading.Thread(target=worker)
    t.start()
    deadline = time.time() + 2.0
    while t.is_alive() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    t.join(timeout=1.0)
    assert isinstance(result_box.get('err'), ValueError)
    assert str(result_box['err']) == 'boom!'

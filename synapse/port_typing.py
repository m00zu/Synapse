"""Install Rust-style port-type checking on top of NodeGraphQt.

NodeGraphQt's stock ``Port.connect_to`` only validates via the
``accepted_port_types`` / ``rejected_port_types`` mechanism, which
keys on node class -- not port data-type.  This module monkey-patches
``Port.connect_to`` to additionally enforce **Liskov-substitutable**
data-type compatibility:

    * A MaskData output (`type=mask`) CAN connect to an ImageData
      input (`type=image`) because ``issubclass(MaskData, ImageData)``.
    * An ImageData output CANNOT connect to a MaskData input
      (narrowing is unsafe).
    * Sibling types reject (mask -> label).
    * ``'any'`` is the explicit wildcard.

Type-mismatched connections raise ``PortError`` with a clear message
naming both ports + types.  Plugins that introduce new data types
opt into subtype polymorphism by calling
``synapse.nodes.base.register_port_type(name, NodeData_subclass)``.

Call ``install_port_type_check()`` once at Synapse startup.  Calling
it again is a no-op (idempotent).
"""
from __future__ import annotations

from PySide6 import QtCore
from NodeGraphQt.base.port import Port, PortError

from .nodes.base import is_port_type_compatible


_INSTALLED = False
_ORIG_CONNECT_TO = None


class _PortTypeErrorSignaller(QtCore.QObject):
    """QObject emitter so port-type errors can reach the GUI.

    A single module-level instance is created on import.  Synapse's
    main window connects ``error_raised`` to a status-bar handler
    (see ``synapse/app.py``) so the user sees a visible message when
    a wire is rejected -- instead of just a traceback in the terminal
    (which doesn't exist in Nuitka builds).

    Qt automatically queues signal emissions across threads, so this
    works whether ``Port.connect_to`` is called from the GUI thread
    (drag-and-drop) or a worker thread (MCP).
    """
    error_raised = QtCore.Signal(str)


# Module-level singleton.  Imported by ``synapse/app.py`` to wire the
# signal to the status bar.
port_error_signaller = _PortTypeErrorSignaller()


def _port_type(port) -> str:
    """Return the data-type string registered for ``port`` (or '').

    Reads from ``_input_types`` or ``_output_types`` depending on the
    port's direction.  Direction is necessary because a single node
    can have an input and an output with the same name (e.g.
    CastType's 'data') that genuinely have different types.
    """
    node = port.node()
    # NodeGraphQt's Port.type_() returns 'in' or 'out'.
    direction = port.type_() if hasattr(port, 'type_') else None
    if direction == 'in':
        types = getattr(node, '_input_types', None)
    else:
        types = getattr(node, '_output_types', None)
    if not types:
        return ''
    return types.get(port.name(), '')


def _typed_connect_to(self, port=None, push_undo=True, emit_signal=True):
    """Type-checking wrapper around the original ``Port.connect_to``.

    On a type mismatch we (a) emit a Qt signal so the GUI can show a
    user-friendly message and (b) raise ``PortError`` so the MCP layer
    (and tests) still see the exception via the normal error path.
    """
    if port is not None:
        src_type = _port_type(self)
        dst_type = _port_type(port)
        if not is_port_type_compatible(src_type, dst_type):
            # Short, user-facing message (one line, no traceback noise).
            short_msg = (
                f"Cannot connect: "
                f"{self.node().name()}.{self.name()} ({src_type or '?'})"
                f" -> "
                f"{port.node().name()}.{port.name()} ({dst_type or '?'})"
                f" -- type mismatch."
            )
            # Longer technical message for the exception (caught by
            # MCP layer + tests).
            long_msg = (
                f"Port type mismatch: cannot connect "
                f"{self.node().name()}.{self.name()} ({src_type!r}) "
                f"to {port.node().name()}.{port.name()} ({dst_type!r}).  "
                f"Types must match exactly OR the source must be a "
                f"subclass of the destination."
            )
            try:
                port_error_signaller.error_raised.emit(short_msg)
            except Exception:
                # If Qt isn't initialised (e.g. in unit tests), don't
                # let the signal emission mask the underlying error.
                pass
            raise PortError(long_msg)
    return _ORIG_CONNECT_TO(self, port, push_undo, emit_signal)


def install_port_type_check() -> None:
    """Monkey-patch ``Port.connect_to`` to enforce type checks.

    Idempotent -- repeated calls are a no-op.  Call once at Synapse
    startup, before any nodes are registered.
    """
    global _INSTALLED, _ORIG_CONNECT_TO
    if _INSTALLED:
        return
    _ORIG_CONNECT_TO = Port.connect_to
    Port.connect_to = _typed_connect_to
    _INSTALLED = True


def uninstall_port_type_check() -> None:
    """Restore the original ``Port.connect_to`` (for tests)."""
    global _INSTALLED, _ORIG_CONNECT_TO
    if not _INSTALLED:
        return
    Port.connect_to = _ORIG_CONNECT_TO
    _ORIG_CONNECT_TO = None
    _INSTALLED = False

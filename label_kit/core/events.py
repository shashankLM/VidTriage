"""A tiny synchronous signal primitive for the Qt-free core layer.

Why not just use ``PySide6.Signal`` everywhere?  Because ``core`` is where the
data model lives, and a data model that can only be instantiated after a
``QApplication`` exists is a data model that never gets unit-tested.  Qt signals
are still used — but only at the boundaries where they earn their keep:
cross-thread delivery in ``media`` and ``plugins.runner``.

Handlers are held weakly, so connecting a bound method from a widget does not
keep that widget alive after Qt has deleted it.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Generic, TypeVar

from .logging import get_logger

__all__ = ["Event", "Subscription"]

T = TypeVar("T")

_log = get_logger(__name__)


class Subscription:
    """Handle returned by :meth:`Event.connect`; call :meth:`disconnect` to stop."""

    __slots__ = ("_alive", "_event", "_key")

    def __init__(self, event: Event, key: int) -> None:
        self._event = event
        self._key = key
        self._alive = True

    def disconnect(self) -> None:
        if self._alive:
            self._event._remove(self._key)
            self._alive = False

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()


class Event(Generic[T]):
    """A synchronous one-argument signal.

    Emission is exception-isolated: one misbehaving handler is logged and the
    remaining handlers still run.  A UI listener blowing up must not corrupt the
    data model that emitted the event.
    """

    __slots__ = ("_name", "_next_key", "_slots")

    def __init__(self, name: str = "event") -> None:
        self._name = name
        # key -> (weakref-to-self-or-None, function)
        self._slots: dict[int, tuple[weakref.ref | None, Callable]] = {}
        self._next_key = 0

    def connect(self, handler: Callable[[T], None]) -> Subscription:
        """Subscribe ``handler``. Bound methods are held weakly where possible.

        Three cases, and all three occur in practice:

        * A Python bound method (``widget.on_theme_changed``) — held weakly, so
          a destroyed widget unsubscribes itself.
        * A builtin method (``some_list.append``, common in tests and adapters)
          — has ``__self__`` but its owner often cannot be weak-referenced, and
          it has no ``__func__``. Held strongly.
        * A plain function or lambda — nothing to weakly reference. Held
          strongly, so a lambda passed inline does not vanish immediately.
        """
        key = self._next_key
        self._next_key += 1

        instance = getattr(handler, "__self__", None)
        function = getattr(handler, "__func__", None)
        if instance is not None and function is not None:
            try:
                reference = weakref.ref(
                    instance, lambda _ref, k=key: self._slots.pop(k, None),
                )
            except TypeError:
                reference = None  # owner does not support weak references
            if reference is not None:
                self._slots[key] = (reference, function)
                return Subscription(self, key)

        self._slots[key] = (None, handler)
        return Subscription(self, key)

    def _remove(self, key: int) -> None:
        self._slots.pop(key, None)

    def disconnect_all(self) -> None:
        self._slots.clear()

    def emit(self, payload: T) -> None:
        for key, (ref, func) in list(self._slots.items()):
            if ref is None:
                target = None
            else:
                target = ref()
                if target is None:
                    self._slots.pop(key, None)
                    continue
            try:
                func(payload) if target is None else func(target, payload)
            except Exception:  # noqa: BLE001 - one bad listener must not stop the rest
                _log.exception("Handler for event %r raised; continuing", self._name)

    @property
    def handler_count(self) -> int:
        return len(self._slots)

    def __repr__(self) -> str:
        return f"<Event {self._name!r} handlers={len(self._slots)}>"

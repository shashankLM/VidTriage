"""A generic, observable keyed registry.

Every extension point in the app — commands, tools, layers, panels, inference
models, exporters — is one of these.  Having a single implementation means a new
extension point costs one line rather than a new bespoke dict-plus-callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from .events import Event
from .logging import get_logger

__all__ = ["Registry", "RegistryChange"]

T = TypeVar("T")

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RegistryChange(Generic[T]):
    action: Literal["added", "removed", "replaced"]
    key: str
    value: T


class Registry(Generic[T]):
    """Insertion-ordered ``str -> T`` map that announces its own mutations.

    Args:
        name: Used in error messages and logs, e.g. ``"tool"``.
        owner_key: Optional callable extracting the owning plugin id from a
            value, enabling :meth:`unregister_owner` to clean up on deactivate.
    """

    def __init__(self, name: str, owner_key: Callable[[T], str | None] | None = None) -> None:
        self._name = name
        self._items: dict[str, T] = {}
        self._owner_key = owner_key
        self.changed: Event[RegistryChange[T]] = Event(f"{name}_registry.changed")

    # ── mutation ────────────────────────────────────────────────────────

    def register(self, key: str, value: T, *, replace: bool = False) -> T:
        """Add ``value`` under ``key``.

        Raises:
            KeyError: if ``key`` is taken and ``replace`` is False. Silent
                overwrite is refused because it turns a plugin id collision
                into a mystery at runtime rather than an error at load time.
        """
        if not key:
            raise ValueError(f"{self._name} key must be a non-empty string")
        existing = self._items.get(key)
        if existing is not None and not replace:
            raise KeyError(f"{self._name} {key!r} is already registered")

        self._items[key] = value
        action = "replaced" if existing is not None else "added"
        _log.debug("%s registry: %s %r", self._name, action, key)
        self.changed.emit(RegistryChange(action, key, value))
        return value

    def unregister(self, key: str) -> T | None:
        value = self._items.pop(key, None)
        if value is not None:
            _log.debug("%s registry: removed %r", self._name, key)
            self.changed.emit(RegistryChange("removed", key, value))
        return value

    def unregister_owner(self, owner: str) -> list[str]:
        """Remove everything contributed by plugin ``owner``. Returns the keys."""
        if self._owner_key is None:
            return []
        doomed = [k for k, v in self._items.items() if self._owner_key(v) == owner]
        for key in doomed:
            self.unregister(key)
        return doomed

    def clear(self) -> None:
        for key in list(self._items):
            self.unregister(key)

    # ── access ──────────────────────────────────────────────────────────

    def get(self, key: str, default: T | None = None) -> T | None:
        return self._items.get(key, default)

    def require(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none registered>"
            raise KeyError(f"unknown {self._name} {key!r}; known: {known}") from None

    def find(self, predicate: Callable[[T], bool]) -> list[T]:
        return [v for v in self._items.values() if predicate(v)]

    def keys(self) -> list[str]:
        return list(self._items)

    def values(self) -> list[T]:
        return list(self._items.values())

    def items(self) -> list[tuple[str, T]]:
        return list(self._items.items())

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __getitem__(self, key: str) -> T:
        return self.require(key)

    def __iter__(self) -> Iterator[T]:
        return iter(list(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<Registry {self._name!r} n={len(self._items)}>"

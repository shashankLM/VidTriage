"""Registry, events, commands, and the layering rule that keeps core Qt-free."""

from __future__ import annotations

import pytest

from vidtriage.core.commands import Command, CommandRegistry
from vidtriage.core.events import Event
from vidtriage.core.registry import Registry


class TestEvent:
    def test_delivers_payload(self):
        event: Event[int] = Event("t")
        seen: list[int] = []
        event.connect(seen.append)
        event.emit(7)
        assert seen == [7]

    def test_disconnect_stops_delivery(self):
        event: Event[int] = Event("t")
        seen: list[int] = []
        subscription = event.connect(seen.append)
        subscription.disconnect()
        event.emit(1)
        assert seen == []

    def test_one_bad_handler_does_not_block_the_others(self):
        event: Event[int] = Event("t")
        seen: list[int] = []
        event.connect(lambda _v: (_ for _ in ()).throw(ValueError("boom")))
        event.connect(seen.append)
        event.emit(3)
        assert seen == [3]

    def test_bound_methods_are_held_weakly(self):
        """A widget destroyed mid-session must not be called back into."""
        import gc

        event: Event[int] = Event("t")

        class Listener:
            def __init__(self) -> None:
                self.seen: list[int] = []

            def handle(self, value: int) -> None:
                self.seen.append(value)

        listener = Listener()
        event.connect(listener.handle)
        event.emit(1)
        assert listener.seen == [1]

        del listener
        gc.collect()
        event.emit(2)
        assert event.handler_count == 0


class TestRegistry:
    def test_duplicate_key_is_refused(self):
        """Silent overwrite would turn a plugin id collision into a mystery."""
        registry: Registry[str] = Registry("thing")
        registry.register("a", "first")
        with pytest.raises(KeyError):
            registry.register("a", "second")
        assert registry.get("a") == "first"

    def test_explicit_replace_is_allowed(self):
        registry: Registry[str] = Registry("thing")
        registry.register("a", "first")
        registry.register("a", "second", replace=True)
        assert registry.get("a") == "second"

    def test_require_names_the_known_keys(self):
        registry: Registry[str] = Registry("tool")
        registry.register("box", "x")
        with pytest.raises(KeyError, match="box"):
            registry.require("nope")

    def test_unregister_owner_removes_only_that_owner(self):
        registry: Registry[tuple[str, str]] = Registry("c", owner_key=lambda v: v[0])
        registry.register("a", ("p1", "a"))
        registry.register("b", ("p2", "b"))
        registry.register("c", ("p1", "c"))
        assert sorted(registry.unregister_owner("p1")) == ["a", "c"]
        assert registry.keys() == ["b"]

    def test_emits_change_events(self):
        registry: Registry[str] = Registry("thing")
        actions: list[str] = []
        registry.changed.connect(lambda change: actions.append(change.action))
        registry.register("a", "1")
        registry.register("a", "2", replace=True)
        registry.unregister("a")
        assert actions == ["added", "replaced", "removed"]

    def test_insertion_order_is_preserved(self):
        registry: Registry[int] = Registry("n")
        for i, key in enumerate("zebra"):
            registry.register(key, i)
        assert registry.keys() == list("zebra")


class TestCommands:
    def test_invoke_respects_is_enabled(self):
        calls: list[int] = []
        command = Command(
            id="x", title="X", handler=lambda: calls.append(1), is_enabled=lambda: False,
        )
        command.invoke()
        assert calls == []

    def test_handler_exceptions_are_contained(self):
        """A raising slot would otherwise unwind through the Qt event loop."""
        command = Command(
            id="x", title="X", handler=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        command.invoke()  # must not raise

    def test_checkable_handler_receives_the_new_state(self):
        seen: list[bool] = []
        command = Command(
            id="x", title="X", checkable=True, handler=seen.append,
            is_checked=lambda: False,
        )
        command.invoke(True)
        command.invoke()  # toggles from the current (False) state
        assert seen == [True, True]

    def test_menu_tree_nests_by_path(self):
        registry = CommandRegistry()
        registry.add(Command(id="a", title="A", handler=lambda: None, menu="View"))
        registry.add(Command(id="b", title="B", handler=lambda: None, menu="View/Overlays"))
        registry.add(Command(id="c", title="C", handler=lambda: None, menu="File"))
        root = registry.menu_tree()
        assert set(root.children) == {"View", "File"}
        assert set(root.children["View"].children) == {"Overlays"}
        assert [c.id for c in root.children["View"].sorted_commands()] == ["a"]

    def test_sections_are_grouped_for_separators(self):
        registry = CommandRegistry()
        registry.add(Command(id="a", title="A", handler=lambda: None, menu="F", section="0"))
        registry.add(Command(id="b", title="B", handler=lambda: None, menu="F", section="1"))
        registry.add(Command(id="c", title="C", handler=lambda: None, menu="F", section="0"))
        sections = registry.menu_tree().children["F"].sections()
        assert [[c.id for c in group] for group in sections] == [["a", "c"], ["b"]]

    def test_shortcut_map_last_registration_wins(self):
        registry = CommandRegistry()
        registry.add(Command(id="a", title="A", handler=lambda: None, shortcut="Ctrl+K"))
        registry.add(Command(id="b", title="B", handler=lambda: None, shortcut="Ctrl+K"))
        assert registry.shortcut_map()["Ctrl+K"].id == "b"

    def test_commands_without_a_menu_are_shortcut_only(self):
        registry = CommandRegistry()
        registry.add(Command(id="a", title="A", handler=lambda: None, shortcut="Q"))
        assert registry.menu_tree().children == {}
        assert "Q" in registry.shortcut_map()


def test_core_never_imports_qt():
    """``core`` must stay constructible without a QApplication.

    This is the constraint that keeps the data model unit-testable. If it ever
    fails, some convenience import has quietly made the model depend on a
    running GUI.
    """
    import importlib
    import pkgutil
    import sys

    import vidtriage.core

    for module_info in pkgutil.iter_modules(vidtriage.core.__path__):
        name = f"vidtriage.core.{module_info.name}"
        importlib.import_module(name)
        source = sys.modules[name]
        offenders = [
            attr for attr, value in vars(source).items()
            if getattr(value, "__module__", "").startswith("PySide6")
        ]
        assert not offenders, f"{name} imports Qt objects: {offenders}"

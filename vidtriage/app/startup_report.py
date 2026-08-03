"""What loaded and what did not, printed once on the way up.

Reports every plugin and every model, not only the broken ones: "sam is missing
its checkpoint" is the same kind of fact as "sam loaded", and a reader scanning
startup wants the whole list in one place rather than having to infer an
absence.

Plugin availability and model availability are different questions, which is why
there are two tables. The SAM plugin loads fine with no checkpoint on disk — it
is the *model* that cannot run, and its remedy is the one thing on this screen a
user may need to act on.

``rich`` is imported once, at the top, and every table helper then uses it
unguarded: :func:`report_startup` checks
:func:`~vidtriage.core.console.rich_available` before calling any of them, so
past that single gate the import is known to have worked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.console import console, rich_available
from ..core.logging import get_logger
from ..plugins.models import Availability, InferenceModel

try:
    from rich.table import Table
    from rich.text import Text
except ImportError:
    # Only the plain-text branch of report_startup runs in this case, and it
    # touches neither name.
    Table = Text = None

if TYPE_CHECKING:
    from ..plugins.manager import PluginState
    from .context import AppContext

__all__ = ["report_startup"]

_log = get_logger(__name__)

_PLUGIN_STATUS_STYLES = {
    "active": "green",
    "unavailable": "yellow",
    "error": "bold red",
    "disabled": "dim",
    "inactive": "dim",
}


def report_startup(context: AppContext) -> None:
    """Print the report, and raise anything actionable to the status bar."""
    states = sorted(context.plugins.states.values(), key=lambda state: state.id)
    if not states:
        return

    if rich_available():
        _print_tables(context, states)
    else:
        _log_report(context, states)

    broken = [s for s in states if s.error]
    unavailable = [s for s in states if not s.error and s.enabled and not s.availability.ok]

    if broken:
        context.status(
            f"{len(broken)} plugin(s) failed to load — see View ▸ Plugins", 10000,
        )
    elif unavailable:
        names = ", ".join(s.id for s in unavailable)
        context.status(f"Unavailable plugin(s): {names} — see View ▸ Plugins", 8000)


def _print_tables(context: AppContext, states: list[PluginState]) -> None:
    terminal = console()

    # Blank lines around the block: log records carry a timestamp column and
    # tables do not, so without the separation a header row reads as another log
    # line that has lost its prefix.
    terminal.line()
    terminal.print(_plugin_table(states))
    models = _model_table(context)
    if models is not None:
        terminal.line()
        terminal.print(models)
    terminal.line()


def _log_report(context: AppContext, states: list[PluginState]) -> None:
    """The same report as plain log lines, for an install without ``rich``."""
    _log.info("Plugins: %s", ", ".join(f"{s.id}={s.status}" for s in states))
    for key, model in sorted(context.models.items()):
        availability = _model_availability(model)
        if not availability.ok:
            _log.info("Model %r unavailable: %s", key, availability.message)


def _plugin_table(states: list[PluginState]) -> Any:
    """Cells are :class:`~rich.text.Text`, never markup strings.

    The values here are filesystem paths and exception text; a path containing
    square brackets would otherwise be parsed as a style tag and vanish.
    """
    table = _table("plugins", "plugin")
    for state in states:
        status = Text(state.status, style=_PLUGIN_STATUS_STYLES.get(state.status, ""))
        table.add_row(state.id, status, _plugin_detail(state))
    return table


def _model_table(context: AppContext) -> Any:
    """Registered inference models and whether each can actually run."""
    models = sorted(context.models.items())
    if not models:
        return None

    table = _table("models", "model")
    for key, model in models:
        availability = _model_availability(model)
        if availability.ok:
            table.add_row(key, Text("ready", style="green"), _capabilities(model))
        else:
            table.add_row(key, Text("unavailable", style="yellow"), _remedy(availability))
    return table


def _table(title: str, first_column: str) -> Any:
    table = Table(box=None, pad_edge=False, title=title,
                  title_justify="left", title_style="bold")
    table.add_column(first_column, style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail", overflow="fold")
    return table


def _plugin_detail(state: PluginState) -> Text:
    """The one thing worth reading about this plugin.

    Deliberately narrower than what **View ▸ Plugins** shows for the same state:
    a dialog has a scrollable pane for the whole traceback, one startup row does
    not.
    """
    if state.error:
        # ``error`` holds a formatted traceback, already logged whole by the
        # manager. The last line is the exception itself — the part that says
        # what actually went wrong.
        return Text(state.error.strip().rsplit("\n", 1)[-1].strip())

    if not state.availability.ok:
        return _remedy(state.availability)

    return Text(state.origin, style="dim")


def _remedy(availability: Availability) -> Text:
    """Why it cannot run, and the command that fixes it.

    The styled equivalent of ``Availability.message``, which the plain-text path
    uses; the two must stay in step.
    """
    detail = Text(availability.reason)
    if availability.remedy:
        detail.append("\n" + availability.remedy, style="cyan")
    return detail


def _model_availability(model: InferenceModel) -> Availability:
    """A model that cannot answer the question is reported, never fatal."""
    try:
        return model.availability()
    except Exception as exc:  # noqa: BLE001 - a broken probe must not stop startup
        return Availability(False, reason=f"availability check failed: {exc}")


def _capabilities(model: InferenceModel) -> Text:
    """Which gestures this model responds to — the thing a user acts on."""
    names = [
        flag.name.removesuffix("_PROMPT").lower().replace("_", "-")
        for flag in model.capabilities
        if flag.name
    ]
    return Text(" · ".join(names) or "no prompts", style="dim")

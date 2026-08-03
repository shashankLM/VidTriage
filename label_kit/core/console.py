"""Terminal output — one shared console, or nothing at all.

Two rules shape this module.

**One Console, not two.** The log handler and the startup report both write to
stderr. Two ``rich`` Consoles over one stream each track their own cursor state
and will overwrite each other's output, so both callers take the instance from
:func:`console` rather than building their own.

**``rich`` is optional.** ``label_kit.core.logging`` already holds the line that
logging is diagnostics and never a hard dependency; the same applies to anything
printed on the way up. Every function here has an answer for the case where the
import fails, and the app starts either way — just plainly.

Nothing in this module formats a message. Callers pass plain text and let the
handler style it, because log messages interpolate arbitrary paths and plugin
ids: a filename containing ``[bold]`` would otherwise be swallowed as markup.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

__all__ = ["console", "rich_available"]


@functools.cache
def console() -> Console | None:
    """The shared stderr console, or ``None`` when ``rich`` is not installed.

    Probed once and cached — including a ``None`` result, so a missing ``rich``
    costs one failed import for the life of the process. Built lazily so
    importing :mod:`label_kit.core` does not pay for terminal detection in
    processes that never log. Tests re-probe with ``console.cache_clear()``.
    """
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console(stderr=True)


def rich_available() -> bool:
    """Whether styled output is possible at all."""
    return console() is not None

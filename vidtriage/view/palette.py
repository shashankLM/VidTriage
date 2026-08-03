"""Deterministic colours for annotation labels.

The same label must get the same colour in this session, the next session, and
on a colleague's machine — otherwise screenshots and side-by-side comparisons
become misleading. So the colour is derived from a stable hash of the label
text, not from an incrementing counter.

Hues are drawn from a fixed set chosen to stay distinguishable on both dark and
light backgrounds and to remain separable under the common forms of colour
vision deficiency (no red/green-only pairings).
"""

from __future__ import annotations

import hashlib

from PySide6.QtGui import QColor

__all__ = ["LabelPalette", "label_color"]

# Hue, saturation, value triples. Ordered so that adjacent indices differ
# strongly, which matters because early labels tend to land on early indices.
_SWATCHES: tuple[tuple[int, int, int], ...] = (
    (205, 190, 245),  # blue
    (25, 205, 250),   # amber
    (145, 175, 215),  # teal
    (330, 175, 240),  # magenta
    (95, 165, 205),   # green
    (270, 160, 240),  # violet
    (15, 200, 245),   # orange
    (190, 200, 225),  # cyan
    (55, 185, 220),   # olive
    (300, 150, 235),  # purple
    (170, 195, 210),  # sea green
    (350, 180, 250),  # rose
)

_UNLABELLED = QColor(160, 160, 160)


def label_color(label: str, *, alpha: int = 255) -> QColor:
    """Stable colour for ``label``. Empty labels get neutral grey."""
    if not label:
        color = QColor(_UNLABELLED)
        color.setAlpha(alpha)
        return color

    digest = hashlib.blake2s(label.encode("utf-8"), digest_size=4).digest()
    hue, sat, val = _SWATCHES[int.from_bytes(digest, "big") % len(_SWATCHES)]
    color = QColor.fromHsv(hue, sat, val)
    color.setAlpha(alpha)
    return color


class LabelPalette:
    """Cache in front of :func:`label_color`, with per-label overrides.

    A plugin that knows its own class colours (a YOLO model shipping the COCO
    palette, say) can pin them with :meth:`override` and everything downstream
    picks them up.
    """

    def __init__(self) -> None:
        self._overrides: dict[str, QColor] = {}
        self._cache: dict[tuple[str, int], QColor] = {}

    def override(self, label: str, color: QColor) -> None:
        self._overrides[label] = QColor(color)
        self._cache = {k: v for k, v in self._cache.items() if k[0] != label}

    def clear_overrides(self) -> None:
        self._overrides.clear()
        self._cache.clear()

    def color(self, label: str, alpha: int = 255) -> QColor:
        key = (label, alpha)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        base = self._overrides.get(label)
        color = QColor(base) if base is not None else label_color(label, alpha=alpha)
        if base is not None:
            color.setAlpha(alpha)
        self._cache[key] = color
        return color

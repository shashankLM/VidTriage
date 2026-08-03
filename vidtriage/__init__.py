"""VidTriage — video and image triage, frame annotation, model-assisted labelling.

Layering, innermost first. Each layer may import the ones above it, never below:

``core``
    Data model and extension primitives. **No Qt.** Geometry, frames,
    annotations, events, registries, commands.
``media``
    Decoding and playback. OpenCV on a worker thread, behind a Qt facade.
``view``
    The canvas, its overlay layers, its interaction tools, theming.
``persistence``
    Sidecars, exporters, settings.
``plugins``
    The extension contract, the inference model API, the threaded runner, and
    the first-party plugins.
``app``
    The shell: context, window, menus generated from the registries.

Everything a user can do arrives as a plugin contribution — including the
built-in triage workflow. See :mod:`vidtriage.plugins.api`.
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]

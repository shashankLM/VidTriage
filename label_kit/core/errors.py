"""Exception hierarchy.

Everything label-kit raises deliberately derives from :class:`LabelKitError`, so
UI-level handlers can catch that one type and show a message instead of swallowing
genuine programming bugs behind a bare ``except Exception``.
"""

from __future__ import annotations

__all__ = [
    "DecodeError",
    "FileOperationError",
    "InferenceError",
    "LabelKitError",
    "MediaError",
    "PersistenceError",
    "PluginError",
    "PluginLoadError",
    "PluginNotAvailableError",
    "SourceOpenError",
]


class LabelKitError(Exception):
    """Base class for every error the application raises on purpose."""


# ── media ───────────────────────────────────────────────────────────────


class MediaError(LabelKitError):
    """A media source could not be read."""


class SourceOpenError(MediaError):
    """A video or image file could not be opened."""


class DecodeError(MediaError):
    """A frame could not be decoded from an otherwise-open source."""


# ── plugins ─────────────────────────────────────────────────────────────


class PluginError(LabelKitError):
    """Base for plugin subsystem failures."""


class PluginLoadError(PluginError):
    """A plugin module raised while being imported or activated."""


class PluginNotAvailableError(PluginError):
    """A plugin was requested but its prerequisites are unmet.

    Attributes:
        remedy: Actionable fix shown to the user, e.g. ``"pip install ultralytics"``.
    """

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.remedy = remedy


class InferenceError(PluginError):
    """A model failed while running inference."""


# ── persistence ─────────────────────────────────────────────────────────


class PersistenceError(LabelKitError):
    """Reading or writing persisted state failed."""


class FileOperationError(LabelKitError):
    """A move / copy / delete on a media file failed."""

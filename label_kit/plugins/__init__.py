"""Plugin subsystem: the contract, the registry, the runner, the built-ins."""

from __future__ import annotations

from .api import PanelArea, PanelSpec, Plugin, PluginContext
from .manager import ENTRY_POINT_GROUP, PLUGIN_ATTR, PluginManager, PluginState
from .models import (
    Availability,
    BoxPrompt,
    Capability,
    InferenceModel,
    InferenceRequest,
    InferenceResult,
    ParamSpec,
    PointPrompt,
    Prompt,
    TextPrompt,
    WholeFramePrompt,
)
from .runner import InferenceRunner

__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGIN_ATTR",
    "Availability",
    "BoxPrompt",
    "Capability",
    "InferenceModel",
    "InferenceRequest",
    "InferenceResult",
    "InferenceRunner",
    "PanelArea",
    "PanelSpec",
    "ParamSpec",
    "Plugin",
    "PluginContext",
    "PluginManager",
    "PluginState",
    "PointPrompt",
    "Prompt",
    "TextPrompt",
    "WholeFramePrompt",
]

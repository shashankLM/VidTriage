"""The inference model contract.

A model turns a frame plus an optional prompt into annotations. That is the
entire interface — adding SAM 3, RT-DETR, a Grounding-DINO variant, or an
in-house ONNX detector means writing one subclass and registering it. Nothing
in the canvas, the tools, the store or the window changes.

Three parts make an adapter drop-in rather than a fork:

:class:`Capability`
    Declares which prompts the model understands. The UI offers a model only
    for gestures it can actually service, so a point click never reaches a
    detector that has no idea what to do with one.

:meth:`InferenceModel.availability`
    Answered *without* importing heavy dependencies. A model whose package or
    weights are missing shows up disabled with an actionable remedy
    ("pip install ultralytics") instead of crashing on first use.

:attr:`InferenceModel.parameters`
    Declares tunables as data. The settings UI is generated from it, so a model
    with a confidence threshold gets a working slider without shipping a widget.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Literal

from ..core.annotations import Annotation
from ..core.frames import Frame
from ..core.geometry import Geometry, Point, Rect
from ..core.logging import get_logger

__all__ = [
    "Availability",
    "BoxPrompt",
    "Capability",
    "InferenceModel",
    "InferenceRequest",
    "InferenceResult",
    "ParamSpec",
    "PointPrompt",
    "Prompt",
    "TextPrompt",
    "WholeFramePrompt",
]

_log = get_logger(__name__)


class Capability(Flag):
    """What kinds of input a model accepts."""

    NONE = 0
    #: Runs on the entire frame with no user input (detectors).
    WHOLE_FRAME = auto()
    #: Accepts a rectangle: either a region to search, or a box prompt.
    BOX_PROMPT = auto()
    #: Accepts include/exclude clicks (SAM-family).
    POINT_PROMPT = auto()
    #: Accepts a free-text class description (open-vocabulary models).
    TEXT_PROMPT = auto()
    #: Can propagate a result to following frames.
    TRACKING = auto()

    def describes(self, prompt: Prompt) -> bool:
        return bool(self & prompt.required_capability)


@dataclass(frozen=True)
class Availability:
    """Whether a model can run right now, and what to do if it cannot."""

    ok: bool
    reason: str = ""
    remedy: str = ""

    @classmethod
    def available(cls) -> Availability:
        return cls(True)

    @classmethod
    def missing_package(cls, package: str, pip_name: str | None = None) -> Availability:
        return cls(
            False,
            reason=f"Python package {package!r} is not installed",
            remedy=f"pip install {pip_name or package}",
        )

    @classmethod
    def missing_weights(cls, path: str, hint: str = "") -> Availability:
        return cls(False, reason=f"Model weights not found at {path}", remedy=hint)

    def __bool__(self) -> bool:
        return self.ok

    @property
    def message(self) -> str:
        return f"{self.reason}\n{self.remedy}".strip() if not self.ok else "Available"


@dataclass(frozen=True)
class ParamSpec:
    """A tunable the settings UI can render generically."""

    name: str
    label: str
    kind: Literal["float", "int", "bool", "choice", "text"]
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""

    def coerce(self, value: Any) -> Any:
        """Force ``value`` into range and type. Falls back to the default."""
        try:
            if self.kind == "bool":
                return bool(value)
            if self.kind == "int":
                result = int(value)
            elif self.kind == "float":
                result = float(value)
            elif self.kind == "choice":
                text = str(value)
                return text if text in self.choices else self.default
            else:
                return str(value)
        except (TypeError, ValueError):
            return self.default

        if self.minimum is not None:
            result = max(result, type(result)(self.minimum))
        if self.maximum is not None:
            result = min(result, type(result)(self.maximum))
        return result


# ── prompts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Prompt:
    """Base for the user input accompanying a request."""

    @property
    def required_capability(self) -> Capability:
        return Capability.NONE

    @property
    def region(self) -> Rect | None:
        """The area of interest, if the prompt implies one."""
        return None


@dataclass(frozen=True)
class WholeFramePrompt(Prompt):
    """Run on everything."""

    @property
    def required_capability(self) -> Capability:
        return Capability.WHOLE_FRAME


@dataclass(frozen=True)
class BoxPrompt(Prompt):
    """A rectangle: the patch to search, or the box to segment."""

    box: Rect

    @property
    def required_capability(self) -> Capability:
        return Capability.BOX_PROMPT

    @property
    def region(self) -> Rect | None:
        return self.box


@dataclass(frozen=True)
class PointPrompt(Prompt):
    """Include / exclude clicks, optionally narrowed by a box."""

    points: tuple[tuple[Point, bool], ...]
    box: Rect | None = None

    @property
    def required_capability(self) -> Capability:
        return Capability.POINT_PROMPT

    @property
    def region(self) -> Rect | None:
        return self.box

    @property
    def positive_points(self) -> tuple[Point, ...]:
        return tuple(p for p, positive in self.points if positive)

    @property
    def negative_points(self) -> tuple[Point, ...]:
        return tuple(p for p, positive in self.points if not positive)

    def with_point(self, point: Point, positive: bool = True) -> PointPrompt:
        return PointPrompt((*self.points, (point, positive)), self.box)


@dataclass(frozen=True)
class TextPrompt(Prompt):
    """A free-text class description for open-vocabulary models."""

    text: str
    box: Rect | None = None

    @property
    def required_capability(self) -> Capability:
        return Capability.TEXT_PROMPT

    @property
    def region(self) -> Rect | None:
        return self.box


# ── request / result ────────────────────────────────────────────────────


@dataclass(frozen=True)
class InferenceRequest:
    """One unit of work handed to a model on a worker thread.

    The frame is passed whole, not pre-cropped: a model that benefits from
    surrounding context (most detectors do) can use it, while one that wants
    only the patch calls :meth:`patch`. Cropping in the caller would throw away
    information the model might need and force every adapter to undo it.
    """

    frame: Frame
    prompt: Prompt = field(default_factory=WholeFramePrompt)
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def image(self):
        """Full RGB frame, ``(H, W, 3)`` uint8."""
        return self.frame.image

    def patch(self) -> tuple[Any, Rect]:
        """The prompt region's pixels and its rect, or the whole frame."""
        region = self.prompt.region
        if region is None:
            return self.frame.image, self.frame.rect
        clipped = region.clamped_to(self.frame.size)
        return self.frame.crop(clipped), clipped

    def annotation(
        self,
        geometry: Geometry,
        *,
        label: str = "",
        score: float | None = None,
        source: str = "",
        **attributes: Any,
    ) -> Annotation:
        """Build an annotation already pinned to this request's frame."""
        return Annotation(
            frame=self.frame.ref,
            geometry=geometry,
            label=label,
            score=score,
            source=source or "model",
            attributes=attributes,
        )


@dataclass(frozen=True)
class InferenceResult:
    """What a model produced, plus timing."""

    request_id: str
    model_id: str
    annotations: tuple[Annotation, ...]
    elapsed_ms: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.annotations)


# ── the model ───────────────────────────────────────────────────────────


class InferenceModel(ABC):
    """Base class for every model adapter.

    Threading contract: :meth:`load`, :meth:`infer` and :meth:`unload` are
    called on a worker thread and must never touch Qt widgets. The runner
    guarantees only one of them runs at a time per model instance, so an
    adapter does not need internal locking.
    """

    #: Unique id. Namespace it with your plugin, e.g. ``"yolo.detect"``.
    id: str = ""
    #: Name shown in menus.
    display_name: str = ""
    #: One-line description for tooltips.
    description: str = ""
    #: Which prompts this model understands.
    capabilities: Capability = Capability.NONE
    #: Declarative tunables; the settings UI is generated from these.
    parameters: tuple[ParamSpec, ...] = ()
    #: Plugin that contributed this model, filled in by ``PluginContext``.
    owner: str | None = None

    def __init__(self) -> None:
        self._loaded = False

    # ── availability & lifecycle ────────────────────────────────────────

    def availability(self) -> Availability:
        """Can this model run? Must be cheap — no heavy imports, no weights.

        Called every time a menu opens, so importing torch here would stall the
        UI for seconds.
        """
        return Availability.available()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load on first use. Called by the runner on a worker thread."""
        if not self._loaded:
            self.load()
            self._loaded = True

    def load(self) -> None:
        """Import dependencies and read weights. Slow; runs off the GUI thread."""

    def unload(self) -> None:
        """Release weights and GPU memory."""
        self._loaded = False

    # ── inference ───────────────────────────────────────────────────────

    @abstractmethod
    def infer(self, request: InferenceRequest) -> Sequence[Annotation]:
        """Run the model. Called on a worker thread.

        Return annotations in **image pixel coordinates** of the full frame —
        if you cropped a patch, add the patch origin back before returning.
        """

    # ── helpers for adapters ────────────────────────────────────────────

    def supports(self, prompt: Prompt) -> bool:
        return self.capabilities.describes(prompt)

    def resolved_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Defaults from :attr:`parameters`, with validated overrides applied."""
        values = {spec.name: spec.default for spec in self.parameters}
        if overrides:
            by_name = {spec.name: spec for spec in self.parameters}
            for key, value in overrides.items():
                spec = by_name.get(key)
                values[key] = spec.coerce(value) if spec else value
        return values

    def run(self, request: InferenceRequest) -> InferenceResult:
        """Load if needed, time the call, and wrap the output. Used by the runner."""
        self.ensure_loaded()
        started = time.perf_counter()
        annotations = tuple(self.infer(request))
        elapsed = (time.perf_counter() - started) * 1000
        _log.debug(
            "%s: %d annotation(s) in %.0fms", self.id, len(annotations), elapsed,
        )
        return InferenceResult(
            request_id=request.id,
            model_id=self.id,
            annotations=annotations,
            elapsed_ms=elapsed,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id!r}>"

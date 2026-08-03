"""Segment Anything — click or box a thing, get its mask.

Backed by Meta's ``segment_anything`` package. The interesting engineering here
is not the model call, it is the two things around it:

**Embedding cache.** ``SamPredictor.set_image`` runs the image encoder, which is
where essentially all the cost lives — seconds on CPU. Prompting is
milliseconds afterwards. Caching the encoder state per frame is what turns
"click, wait, click, wait" into a usable refine-by-clicking loop, and it is why
:class:`~label_kit.plugins.runner.InferenceRunner` serialises calls per model
rather than running them concurrently.

**Honest availability.** SAM checkpoints are hundreds of megabytes and are not
shipped with anything. If no checkpoint is found the model reports exactly where
it looked and what to download, instead of appearing in the menu and then
failing.

*Adding SAM 2 or SAM 3:* subclass :class:`InferenceModel`, declare
``POINT_PROMPT | BOX_PROMPT``, and register it from a plugin. Nothing in the
canvas, the tools or the annotate workflow needs to change — they are written
against :class:`~label_kit.plugins.models.Capability`, not against this class.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ....core.annotations import Annotation
from ....core.frames import FrameRef
from ....core.geometry import Mask
from ....core.logging import get_logger
from ....persistence.settings import CONFIG_DIR
from ...api import Plugin, PluginContext
from ...models import (
    Availability,
    BoxPrompt,
    Capability,
    InferenceModel,
    InferenceRequest,
    ParamSpec,
    PointPrompt,
)

__all__ = ["PLUGIN", "WEIGHTS_DIR", "SamModel", "SamPlugin", "find_checkpoint"]

_log = get_logger(__name__)

WEIGHTS_DIR = CONFIG_DIR / "weights"

#: Checkpoint filename → ``segment_anything`` model type, cheapest first.
#: vit_b leads because this is routinely run on a CPU-only machine, where vit_h
#: takes the better part of a minute per frame and vit_b a few seconds.
_KNOWN_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("sam_vit_b_01ec64.pth", "vit_b"),
    ("sam_vit_l_0b3195.pth", "vit_l"),
    ("sam_vit_h_4b8939.pth", "vit_h"),
)

_DOWNLOAD_HINT = (
    "Download a SAM checkpoint into ~/.labelkit/weights/ — for example:\n"
    "  mkdir -p ~/.labelkit/weights && cd ~/.labelkit/weights\n"
    "  curl -LO https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)


def _search_dirs() -> list[Path]:
    dirs = [WEIGHTS_DIR, Path.cwd() / "weights", Path.cwd()]
    override = os.environ.get("VIDTRIAGE_SAM_CHECKPOINT")
    if override:
        candidate = Path(override)
        dirs.insert(0, candidate if candidate.is_dir() else candidate.parent)
    return dirs


def find_checkpoint(explicit: str | None = None) -> tuple[Path, str] | None:
    """Locate a SAM checkpoint. Returns ``(path, model_type)`` or ``None``."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path, _model_type_for(path.name)

    for directory in _search_dirs():
        if not directory.is_dir():
            continue
        for filename, model_type in _KNOWN_CHECKPOINTS:
            candidate = directory / filename
            if candidate.is_file():
                return candidate, model_type
    return None


def _model_type_for(filename: str) -> str:
    for known, model_type in _KNOWN_CHECKPOINTS:
        if known == filename:
            return model_type
    lowered = filename.lower()
    for suffix in ("vit_h", "vit_l", "vit_b"):
        if suffix in lowered:
            return suffix
    return "vit_b"


class SamModel(InferenceModel):
    """Point- and box-promptable segmentation."""

    id = "sam.predict"
    display_name = "SAM — segment from prompt"
    description = "Click a point or drag a box; SAM returns the object's mask"
    capabilities = Capability.POINT_PROMPT | Capability.BOX_PROMPT

    parameters = (
        ParamSpec(
            "checkpoint", "Checkpoint path", "text", "",
            help="Leave blank to auto-detect in ~/.labelkit/weights/",
        ),
        ParamSpec(
            "device", "Device", "choice", "cpu", choices=("cpu", "cuda"),
            help="cuda requires a GPU-enabled torch build",
        ),
        ParamSpec(
            "return_all", "Return all candidates", "bool", False,
            help="SAM proposes three masks per prompt; keep them all instead of the best",
        ),
        ParamSpec(
            "min_area", "Minimum area (px)", "int", 16, minimum=0, maximum=1_000_000,
            help="Discard masks smaller than this — usually stray speckle",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._predictor: Any = None
        self._checkpoint: Path | None = None
        self._embedded_frame: FrameRef | None = None

    # ── availability ────────────────────────────────────────────────────

    def availability(self) -> Availability:
        if importlib.util.find_spec("segment_anything") is None:
            return Availability.missing_package("segment_anything", "segment-anything")
        if importlib.util.find_spec("torch") is None:
            return Availability.missing_package("torch")
        if find_checkpoint() is None:
            looked = ", ".join(str(d) for d in _search_dirs() if d.is_dir())
            return Availability(
                False,
                reason=f"No SAM checkpoint found (looked in: {looked or 'nowhere readable'})",
                remedy=_DOWNLOAD_HINT,
            )
        return Availability.available()

    # ── lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        found = find_checkpoint()
        if found is None:
            from ....core.errors import PluginNotAvailableError

            raise PluginNotAvailableError("No SAM checkpoint available", _DOWNLOAD_HINT)

        checkpoint, model_type = found
        from segment_anything import SamPredictor, sam_model_registry

        _log.info("SAM: loading %s (%s)", checkpoint.name, model_type)
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=self._device())
        self._predictor = SamPredictor(sam)
        self._checkpoint = checkpoint
        self._embedded_frame = None

    def _device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - fall back rather than fail to load
            return "cpu"

    def unload(self) -> None:
        self._predictor = None
        self._embedded_frame = None
        super().unload()

    # ── inference ───────────────────────────────────────────────────────

    def infer(self, request: InferenceRequest) -> Sequence[Annotation]:
        if self._predictor is None:
            return []

        self._embed(request)
        point_coords, point_labels, box = self._build_prompt(request)
        if point_coords is None and box is None:
            return []

        masks, scores, _logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        order = np.argsort(scores)[::-1]
        if not bool(request.params.get("return_all", False)):
            order = order[:1]

        min_area = int(request.params.get("min_area", 0))
        annotations: list[Annotation] = []
        for rank, index in enumerate(order):
            mask = Mask.from_full_frame(masks[index])
            if mask.is_empty or mask.pixel_count < min_area:
                continue
            annotations.append(request.annotation(
                mask,
                score=float(scores[index]),
                source=self.id,
                checkpoint=self._checkpoint.name if self._checkpoint else "",
                candidate=rank,
            ))
        return annotations

    def _embed(self, request: InferenceRequest) -> None:
        """Run the image encoder, but only when the frame actually changed.

        This is the whole performance story: the encoder dominates, and a
        refine-by-clicking session hits it once instead of once per click.
        """
        if self._embedded_frame == request.frame.ref:
            return
        self._predictor.set_image(request.image)
        self._embedded_frame = request.frame.ref
        _log.debug("SAM: embedded frame %s", request.frame.ref)

    @staticmethod
    def _build_prompt(
        request: InferenceRequest,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        prompt = request.prompt
        point_coords = point_labels = box = None

        if isinstance(prompt, PointPrompt) and prompt.points:
            point_coords = np.array(
                [[p.x, p.y] for p, _positive in prompt.points], dtype=np.float32,
            )
            point_labels = np.array(
                [1 if positive else 0 for _p, positive in prompt.points], dtype=np.int32,
            )
            if prompt.box is not None:
                box = np.array(prompt.box.as_xyxy(), dtype=np.float32)
        elif isinstance(prompt, BoxPrompt):
            box = np.array(prompt.box.as_xyxy(), dtype=np.float32)

        return point_coords, point_labels, box


class SamPlugin(Plugin):
    id = "sam"
    name = "Segment Anything"
    description = "Point- and box-prompted segmentation (Meta SAM)"
    default_enabled = True

    def availability(self) -> Availability:
        if importlib.util.find_spec("segment_anything") is None:
            return Availability.missing_package("segment_anything", "segment-anything")
        return Availability.available()

    def activate(self, ctx: PluginContext) -> None:
        model = ctx.add_model(SamModel())
        ctx.add_command(
            id="sam.arm", title="Prompt With SAM", shortcut="Ctrl+Shift+S",
            menu="Models", section="1",
            handler=lambda: self._arm(ctx),
            is_enabled=lambda m=model: m.availability().ok,
            description="Switch to the point tool with SAM armed",
        )
        ctx.add_command(
            id="sam.weights_info", title="Where To Get SAM Weights…",
            menu="Models", section="9",
            handler=lambda: ctx.app.status(_DOWNLOAD_HINT.replace("\n", "  "), 20000),
        )

    @staticmethod
    def _arm(ctx: PluginContext) -> None:
        """Hand the user a working SAM setup in one keystroke."""
        annotate = ctx.app.plugins.plugins.get("annotate")
        if annotate is not None and annotate.is_active:
            annotate.set_prompt_model(SamModel.id)
        else:
            ctx.app.status("Enable the Annotate plugin to prompt models", 5000)


PLUGIN = SamPlugin

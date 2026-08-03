"""Ultralytics YOLO detection and segmentation.

Two models, both real:

``yolo.detect``
    Boxes. Runs on the whole frame, or — given a box prompt — only inside that
    region. Restricting to a region is not just a crop for speed: on a 4K
    dashcam frame a traffic light is a handful of pixels after the model's
    letterbox resize, and detecting inside a hand-drawn region recovers it.

``yolo.segment``
    The same, for ``-seg`` weights, returning masks instead of boxes.

Weights download on first use via ultralytics' own resolver, so the default
``yolo11n.pt`` needs no manual setup — but the first run needs network access,
which the description says out loud rather than mysteriously hanging.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ....core.annotations import Annotation
from ....core.geometry import Mask, Rect
from ....core.logging import get_logger
from ...api import Plugin, PluginContext
from ...models import (
    Availability,
    Capability,
    InferenceModel,
    InferenceRequest,
    ParamSpec,
)

__all__ = ["PLUGIN", "YoloDetectModel", "YoloPlugin", "YoloSegmentModel"]

_log = get_logger(__name__)

_DETECT_WEIGHTS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt")
_SEGMENT_WEIGHTS = ("yolo11n-seg.pt", "yolo11s-seg.pt", "yolov8n-seg.pt")

#: Weights live here rather than wherever the app happened to be launched from.
WEIGHTS_DIR = Path.home() / ".vidtriage" / "weights"

#: Minimum patch size. Below this the letterbox upscale is mostly interpolation.
_MIN_PATCH_PX = 96
#: Extra context around a box prompt, as a fraction of its size. A crop cut
#: exactly to the drawn box strips the surroundings a detector relies on, and
#: detection rates fall off sharply; a margin recovers them.
_CONTEXT_RATIO = 0.35


def _resolve_weights(name: str) -> Path | str:
    """Return a path inside :data:`WEIGHTS_DIR`, downloading there if needed.

    Given a bare filename, ultralytics downloads into the current working
    directory — which for a GUI app is wherever the user launched it, and in
    development is the repository. Keeping weights in one known place also means
    the several-megabyte download happens once per machine, not once per folder.
    """
    if os.path.sep in name or name.startswith("~"):
        return Path(name).expanduser()

    target = WEIGHTS_DIR / name
    if target.is_file():
        return target

    try:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(target))
    except Exception as exc:  # noqa: BLE001 - private-ish API; degrade gracefully
        _log.warning(
            "Could not pre-download %s into %s (%s); "
            "letting ultralytics resolve it instead", name, WEIGHTS_DIR, exc,
        )
        return name

    return target if target.is_file() else name


class _UltralyticsModel(InferenceModel):
    """Shared plumbing for the ultralytics adapters."""

    weights_choices: tuple[str, ...] = _DETECT_WEIGHTS

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._loaded_weights: str | None = None

    @property
    def parameters(self) -> tuple[ParamSpec, ...]:
        return (
            ParamSpec(
                "weights", "Weights", "choice", self.weights_choices[0],
                choices=self.weights_choices,
                help="Downloaded automatically on first use",
            ),
            ParamSpec(
                "confidence", "Confidence", "float", 0.25,
                minimum=0.01, maximum=0.99, step=0.05,
                help="Minimum score for a detection to be kept",
            ),
            ParamSpec(
                "iou", "NMS IoU", "float", 0.45, minimum=0.1, maximum=0.95, step=0.05,
            ),
            ParamSpec(
                "max_detections", "Max detections", "int", 100, minimum=1, maximum=1000,
            ),
        )

    def availability(self) -> Availability:
        if importlib.util.find_spec("ultralytics") is None:
            return Availability.missing_package("ultralytics")
        if importlib.util.find_spec("torch") is None:
            return Availability.missing_package("torch")
        return Availability.available()

    def _ensure_weights(self, weights: str) -> Any:
        """Load, reusing the existing model when the weights have not changed."""
        if self._model is not None and self._loaded_weights == weights:
            return self._model

        from ultralytics import YOLO

        _log.info("%s: loading weights %s", self.id, weights)
        self._model = YOLO(str(_resolve_weights(weights)))
        self._loaded_weights = weights
        return self._model

    def unload(self) -> None:
        self._model = None
        self._loaded_weights = None
        super().unload()

    def _predict(self, image: np.ndarray, params: dict[str, Any]) -> Any:
        model = self._ensure_weights(str(params["weights"]))
        results = model.predict(
            # Ultralytics interprets a numpy source as BGR, while a Frame is
            # RGB throughout this application. Without the swap every colour
            # is inverted for the model — a red traffic light reads as blue,
            # which quietly degrades accuracy on exactly the footage this is
            # for. ascontiguousarray materialises the reversed view.
            source=np.ascontiguousarray(image[:, :, ::-1]),
            conf=float(params["confidence"]),
            iou=float(params["iou"]),
            max_det=int(params["max_detections"]),
            verbose=False,
        )
        return results[0] if results else None

    @staticmethod
    def _padded_region(request: InferenceRequest) -> Rect:
        """The prompt region, grown for context and clipped to the frame.

        Two separate reasons to grow it, and both matter in practice:

        * A crop cut exactly to the drawn box removes the surroundings a
          detector uses to recognise the object, and detection rates drop.
        * A very small drag becomes almost pure interpolation once letterboxed
          up to the network's input size.
        """
        region = request.prompt.region
        if region is None:
            return request.frame.rect

        pad_x = max(
            (_MIN_PATCH_PX - region.width) / 2, region.width * _CONTEXT_RATIO, 0.0,
        )
        pad_y = max(
            (_MIN_PATCH_PX - region.height) / 2, region.height * _CONTEXT_RATIO, 0.0,
        )
        grown = Rect(
            region.x1 - pad_x, region.y1 - pad_y, region.x2 + pad_x, region.y2 + pad_y,
        )
        return grown.clamped_to(request.frame.size)

    def _class_name(self, result: Any, class_index: int) -> str:
        names = getattr(result, "names", None) or {}
        return str(names.get(class_index, class_index))


class YoloDetectModel(_UltralyticsModel):
    id = "yolo.detect"
    display_name = "YOLO — detect"
    description = "Object detection. Whole frame, or inside a box you draw."
    capabilities = Capability.WHOLE_FRAME | Capability.BOX_PROMPT
    weights_choices = _DETECT_WEIGHTS

    def infer(self, request: InferenceRequest) -> Sequence[Annotation]:
        region = self._padded_region(request)
        rows, cols = region.to_pixel_slice(request.frame.size)
        patch = request.frame.image[rows, cols]
        origin_x, origin_y = float(cols.start), float(rows.start)

        result = self._predict(patch, request.params)
        boxes = getattr(result, "boxes", None) if result is not None else None
        if boxes is None or len(boxes) == 0:
            return []

        annotations: list[Annotation] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), score, class_index in zip(xyxy, confidences, classes, strict=False):
            annotations.append(request.annotation(
                # Patch coordinates back into full-frame space.
                Rect(
                    float(x1) + origin_x, float(y1) + origin_y,
                    float(x2) + origin_x, float(y2) + origin_y,
                ),
                label=self._class_name(result, int(class_index)),
                score=float(score),
                source=self.id,
                weights=request.params["weights"],
            ))
        return annotations


class YoloSegmentModel(_UltralyticsModel):
    id = "yolo.segment"
    display_name = "YOLO — segment"
    description = "Instance segmentation with -seg weights. Returns masks."
    capabilities = Capability.WHOLE_FRAME | Capability.BOX_PROMPT
    weights_choices = _SEGMENT_WEIGHTS

    def infer(self, request: InferenceRequest) -> Sequence[Annotation]:
        region = self._padded_region(request)
        rows, cols = region.to_pixel_slice(request.frame.size)
        patch = request.frame.image[rows, cols]
        origin_x, origin_y = int(cols.start), int(rows.start)

        result = self._predict(patch, request.params)
        masks = getattr(result, "masks", None) if result is not None else None
        boxes = getattr(result, "boxes", None) if result is not None else None
        if masks is None or boxes is None or len(boxes) == 0:
            return []

        frame_width, frame_height = request.frame.size.as_int()
        patch_height, patch_width = patch.shape[:2]

        annotations: list[Annotation] = []
        mask_data = masks.data.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for patch_mask, score, class_index in zip(mask_data, confidences, classes, strict=False):
            resized = _resize_mask(patch_mask, patch_width, patch_height)
            # Paste the patch-sized mask back into a full-frame canvas.
            full = np.zeros((frame_height, frame_width), dtype=bool)
            y2 = min(origin_y + patch_height, frame_height)
            x2 = min(origin_x + patch_width, frame_width)
            full[origin_y:y2, origin_x:x2] = resized[: y2 - origin_y, : x2 - origin_x]

            mask = Mask.from_full_frame(full)
            if mask.is_empty:
                continue
            annotations.append(request.annotation(
                mask,
                label=self._class_name(result, int(class_index)),
                score=float(score),
                source=self.id,
                weights=request.params["weights"],
            ))
        return annotations


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize of a float mask to ``(height, width)`` booleans.

    Ultralytics returns masks at the network's stride, not the input size.
    """
    import cv2

    binary = (mask > 0.5).astype(np.uint8)
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return binary.astype(bool)


class YoloPlugin(Plugin):
    id = "yolo"
    name = "YOLO (Ultralytics)"
    description = "Detection and instance segmentation via the ultralytics package"
    default_enabled = True

    def availability(self) -> Availability:
        if importlib.util.find_spec("ultralytics") is None:
            return Availability.missing_package("ultralytics")
        return Availability.available()

    def activate(self, ctx: PluginContext) -> None:
        ctx.add_model(YoloDetectModel())
        ctx.add_model(YoloSegmentModel())


PLUGIN = YoloPlugin

"""Frame annotation — manual drawing and model-assisted labelling.

This is the plugin that connects a gesture to a model. The canvas publishes a
:class:`~vidtriage.view.tools.ToolResult`; this plugin decides what it means:

* **Manual** (the default) — the shape is saved as an annotation with the
  current label.
* **A model is armed** — the shape becomes a *prompt*. Drag a box and a detector
  runs inside it, or a segmenter is asked for the object it contains. Click a
  point and SAM-family models return the object under the cursor; right-click
  adds an exclude point, and Shift accumulates points into one prompt so you can
  refine a mask click by click.

Nothing here is specific to any model. Arming works off
:class:`~vidtriage.plugins.models.Capability`, so a newly registered adapter
appears in the "Prompt With" menu and works immediately.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import QInputDialog

from ....core.annotations import MANUAL_SOURCE, Annotation
from ....core.geometry import Point
from ....core.logging import get_logger
from ....view.tools import ToolResult
from ...api import Plugin, PluginContext
from ...models import BoxPrompt, Capability, InferenceModel, PointPrompt
from .panel import AnnotationPanel

__all__ = ["PLUGIN", "AnnotatePlugin"]

_log = get_logger(__name__)

_DEFAULT_LABEL = "object"
_MAX_RECENT_LABELS = 24
#: Prompt-capable models can be armed; whole-frame-only detectors cannot.
_PROMPTABLE = Capability.BOX_PROMPT | Capability.POINT_PROMPT


class AnnotatePlugin(Plugin):
    id = "annotate"
    name = "Annotate"
    description = "Draw boxes, points and polygons; prompt models to do it for you"
    default_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.current_label = _DEFAULT_LABEL
        self.recent_labels: list[str] = []
        self._prompt_model_id: str | None = None
        self._pending_points: list[tuple[Point, bool]] = []
        self._ctx: PluginContext | None = None

    # ── lifecycle ───────────────────────────────────────────────────────

    def activate(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        self.current_label = str(ctx.settings.get("label", _DEFAULT_LABEL))
        self.recent_labels = list(ctx.settings.get("recent_labels", []))
        self._prompt_model_id = ctx.settings.get("prompt_model")

        ctx.connect(ctx.app.canvas.tool_result, self._on_tool_result)
        ctx.connect(ctx.app.canvas.annotation_edited, self._on_annotation_edited)
        ctx.connect(ctx.app.canvas.delete_requested, self._on_delete_requested)
        ctx.connect(ctx.app.inference_finished, self._on_inference_finished)
        ctx.connect(ctx.app.frame_changed, self._on_frame_changed)

        ctx.add_panel(
            id="annotate.panel", title="Annotations", area="right",
            visible_by_default=True, shortcut="Ctrl+Shift+A",
            factory=lambda: AnnotationPanel(ctx.app, self),
        )

        ctx.add_command(
            id="annotate.set_label", title="Set Label…", shortcut="Ctrl+L",
            menu="Annotate", section="0", order=10, handler=self._prompt_for_label,
            description="Label applied to new annotations and model results",
        )
        ctx.add_command(
            id="annotate.promote", title="Accept Model Predictions",
            shortcut="Ctrl+Shift+P", menu="Annotate", section="0", order=20,
            handler=self.promote_predictions,
            description="Mark predictions on this frame as confirmed",
        )
        ctx.add_command(
            id="annotate.clear_predictions", title="Discard Model Predictions",
            menu="Annotate", section="0", order=30, handler=self._clear_predictions,
        )
        ctx.add_command(
            id="annotate.rerun", title="Re-run Last Prompt", shortcut="Ctrl+R",
            menu="Annotate", section="1", order=10, handler=self._rerun_prompt,
            is_enabled=lambda: bool(self._pending_points and self.prompt_model),
        )
        ctx.add_command(
            id="annotate.prompt.manual", title="Manual (no model)",
            menu="Annotate/Prompt With", order=0, checkable=True,
            is_checked=lambda: self._prompt_model_id is None,
            handler=lambda checked: self.set_prompt_model(None),
        )

        # One entry per prompt-capable model, kept in sync as plugins load.
        from ....app.commands import RegistryCommands
        from ....core.commands import Command

        self._model_commands = RegistryCommands(
            ctx.app.commands, ctx.app.models,
            lambda key, model: None if not (model.capabilities & _PROMPTABLE) else Command(
                id=f"annotate.prompt.{key}",
                title=model.display_name or key,
                description=model.description,
                menu="Annotate/Prompt With",
                owner=self.id,
                checkable=True,
                is_checked=lambda m=key: self._prompt_model_id == m,
                is_enabled=lambda m=model: m.availability().ok,
                handler=lambda checked, m=key: self.set_prompt_model(m),
            ),
        )

    def deactivate(self) -> None:
        commands = getattr(self, "_model_commands", None)
        if commands is not None:
            commands.dispose()
        self._pending_points.clear()
        self._ctx = None

    # ── state ───────────────────────────────────────────────────────────

    @property
    def prompt_model(self) -> InferenceModel | None:
        if self._ctx is None or self._prompt_model_id is None:
            return None
        return self._ctx.app.models.get(self._prompt_model_id)

    def set_prompt_model(self, model_id: str | None) -> None:
        self._prompt_model_id = model_id
        self._pending_points.clear()
        if self._ctx is None:
            return
        self._ctx.settings["prompt_model"] = model_id

        model = self.prompt_model
        if model is None:
            self._ctx.app.status("Prompting off — shapes are saved as drawn", 4000)
            return

        wants = []
        if model.capabilities & Capability.BOX_PROMPT:
            wants.append("box")
        if model.capabilities & Capability.POINT_PROMPT:
            wants.append("point")
        self._ctx.app.status(
            f"Prompting {model.display_name} — draw a {' or '.join(wants)}", 5000,
        )
        # Put the user straight into a tool the model can actually consume.
        preferred = "box" if "box" in wants else "point"
        if self._ctx.app.canvas.active_tool.id not in wants:
            self._ctx.app.canvas.set_tool(preferred)

    def set_current_label(self, label: str) -> None:
        self.current_label = label or _DEFAULT_LABEL
        if label and label not in self.recent_labels:
            self.recent_labels.insert(0, label)
            del self.recent_labels[_MAX_RECENT_LABELS:]
        if self._ctx is not None:
            self._ctx.settings["label"] = self.current_label
            self._ctx.settings["recent_labels"] = self.recent_labels

    # ── tool routing ────────────────────────────────────────────────────

    def _on_tool_result(self, result: ToolResult) -> None:
        ctx = self._ctx
        if ctx is None or ctx.app.current_frame is None:
            return

        model = self.prompt_model
        if result.tool_id == "box":
            if model is not None and model.capabilities & Capability.BOX_PROMPT:
                self._pending_points.clear()
                ctx.app.run_model(model, BoxPrompt(result.geometry))
            else:
                self._add_manual(result.geometry)
        elif result.tool_id == "point":
            if model is not None and model.capabilities & Capability.POINT_PROMPT:
                self._prompt_with_point(result)
            else:
                self._add_manual(result.geometry)
        elif result.tool_id == "polygon":
            self._add_manual(result.geometry)

    def _prompt_with_point(self, result: ToolResult) -> None:
        """Accumulate points when Shift is held; otherwise start a fresh prompt."""
        if not result.additive:
            self._pending_points.clear()
        self._pending_points.append((result.geometry, result.positive))
        self._rerun_prompt()

    def _rerun_prompt(self) -> None:
        ctx, model = self._ctx, self.prompt_model
        if ctx is None or model is None or not self._pending_points:
            return
        ctx.app.run_model(model, PointPrompt(tuple(self._pending_points)))

    def _add_manual(self, geometry) -> None:
        ctx = self._ctx
        frame = ctx.app.current_frame if ctx else None
        if ctx is None or frame is None:
            return
        annotation = Annotation(
            frame=frame.ref,
            geometry=geometry,
            label=self.current_label,
            source=MANUAL_SOURCE,
        )
        ctx.app.annotations.add(annotation)
        self.set_current_label(self.current_label)
        ctx.app.canvas.select_annotations([annotation.id])

    # ── canvas edits ────────────────────────────────────────────────────

    def _on_annotation_edited(self, annotation: Annotation) -> None:
        if self._ctx is not None:
            self._ctx.app.annotations.update(annotation)

    def _on_delete_requested(self, ids: tuple[str, ...]) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        store = ctx.app.annotations
        removed = store.remove([
            a for a in (store.by_id(i) for i in ids) if a is not None
        ])
        if removed:
            ctx.app.status(f"Deleted {len(removed)} annotation(s)", 2500)

    def _on_frame_changed(self, _frame) -> None:
        # A prompt is about one frame; carrying points across would silently
        # segment the wrong image.
        self._pending_points.clear()

    # ── inference results ───────────────────────────────────────────────

    def _on_inference_finished(self, result) -> None:
        ctx = self._ctx
        if ctx is None or not result.annotations:
            return
        # A prompt-driven model has no idea what the thing is called; give its
        # output the label the user is working with. Detectors name their own
        # classes, so those are left alone.
        labelled = [
            a if a.label else a.with_label(self.current_label) for a in result.annotations
        ]
        ctx.app.annotations.add(labelled)

    # ── commands ────────────────────────────────────────────────────────

    def _prompt_for_label(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        known = sorted({*ctx.app.annotations.labels(), *self.recent_labels})
        if known:
            label, accepted = QInputDialog.getItem(
                ctx.app.window, "VidTriage — Label",
                "Label for new annotations:", known,
                known.index(self.current_label) if self.current_label in known else 0,
                True,
            )
        else:
            label, accepted = QInputDialog.getText(
                ctx.app.window, "VidTriage — Label",
                "Label for new annotations:", text=self.current_label,
            )
        if accepted and label.strip():
            self.set_current_label(label.strip())
            ctx.app.status(f"Label: {self.current_label}", 3000)

    def promote_predictions(self) -> None:
        """Mark this frame's predictions as human-confirmed."""
        ctx = self._ctx
        frame = ctx.app.current_frame if ctx else None
        if ctx is None or frame is None:
            return
        store = ctx.app.annotations
        # The original model id is kept in attributes: provenance still matters
        # after a human has signed off on the result.
        promoted = [
            replace(
                a,
                label=a.label or self.current_label,
                source=MANUAL_SOURCE,
                attributes={**a.attributes, "predicted_by": a.source},
            )
            for a in store.for_frame(frame.ref) if a.is_prediction
        ]
        if promoted:
            store.update(promoted)
            ctx.app.status(f"Accepted {len(promoted)} prediction(s)", 3000)

    def _clear_predictions(self) -> None:
        ctx = self._ctx
        frame = ctx.app.current_frame if ctx else None
        if ctx is None or frame is None:
            return
        store = ctx.app.annotations
        removed = store.remove([
            a for a in store.for_frame(frame.ref) if a.is_prediction
        ])
        ctx.app.status(f"Discarded {len(removed)} prediction(s)", 3000)


PLUGIN = AnnotatePlugin

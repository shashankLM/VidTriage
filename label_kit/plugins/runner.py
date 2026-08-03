"""Runs inference off the GUI thread and routes results back safely.

On a CPU-only machine a SAM forward pass takes seconds. Running that on the GUI
thread would freeze the window solid — which is exactly why the model layer was
never bolted onto the old player.

What this gives every adapter for free:

**Serialisation per model.** A model object holds mutable state (a set image
embedding, a KV cache). Two concurrent ``infer`` calls on one instance would
interleave and corrupt it, so each model gets its own lane and at most one call
runs at a time.

**Supersede semantics.** Clicking three points in a second should produce the
result for the third, not three results arriving out of order. A queued request
replaces the one waiting behind it, and a completed result is dropped if a newer
request for that model has since been submitted.

**Failure isolation.** An adapter that raises reports an error and leaves the
lane usable; it does not take down the pool or the app.

Cancellation is *result-discarding*, not interruption: a torch forward pass
already in flight cannot be stopped, so the work finishes and the answer is
thrown away. Callers are told via :attr:`InferenceRunner.superseded`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from ..core.logging import get_logger
from .models import InferenceModel, InferenceRequest, InferenceResult

__all__ = ["InferenceRunner"]

_log = get_logger(__name__)


@dataclass
class _Lane:
    """Per-model serialisation and supersede bookkeeping."""

    lock: threading.Lock
    latest_request_id: str = ""


class _TaskSignals(QObject):
    """QRunnable cannot carry signals, so each task borrows this object.

    Because the runner lives on the GUI thread, these connections are queued and
    the payload lands back on the GUI thread — the only place it may touch the
    annotation store.
    """

    started = Signal(str, str)            # model_id, request_id
    finished = Signal(object)             # InferenceResult
    failed = Signal(str, str, str)        # model_id, request_id, message
    superseded = Signal(str, str)         # model_id, request_id


class _InferenceTask(QRunnable):
    def __init__(
        self,
        model: InferenceModel,
        request: InferenceRequest,
        lane: _Lane,
        signals: _TaskSignals,
    ) -> None:
        super().__init__()
        self._model = model
        self._request = request
        self._lane = lane
        self._signals = signals
        self.setAutoDelete(True)

    def _is_current(self) -> bool:
        return self._lane.latest_request_id == self._request.id

    def run(self) -> None:
        model_id, request_id = self._model.id, self._request.id

        # Cheap pre-check: a newer request arrived while we sat in the queue.
        if not self._is_current():
            self._signals.superseded.emit(model_id, request_id)
            return

        with self._lane.lock:
            # Re-check after acquiring: we may have waited behind a long call.
            if not self._is_current():
                self._signals.superseded.emit(model_id, request_id)
                return

            self._signals.started.emit(model_id, request_id)
            try:
                result = self._model.run(self._request)
            except Exception as exc:  # noqa: BLE001 - adapter faults stay contained
                _log.exception("Model %s failed on request %s", model_id, request_id)
                self._signals.failed.emit(model_id, request_id, f"{type(exc).__name__}: {exc}")
                return

        if self._is_current():
            self._signals.finished.emit(result)
        else:
            self._signals.superseded.emit(model_id, request_id)


class InferenceRunner(QObject):
    """Submits inference work and delivers results on the GUI thread."""

    started = Signal(str, str)       # model_id, request_id
    finished = Signal(object)        # InferenceResult
    failed = Signal(str, str, str)   # model_id, request_id, message
    superseded = Signal(str, str)    # model_id, request_id
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None, max_threads: int = 2) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        # Models are internally multi-threaded; oversubscribing the pool just
        # makes every request slower. Two lets a fast detector overlap a slow
        # segmenter without thrashing.
        self._pool.setMaxThreadCount(max(1, max_threads))
        self._lanes: dict[str, _Lane] = {}
        self._running: set[str] = set()

        self._signals = _TaskSignals()
        self._signals.started.connect(self._on_started)
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._signals.superseded.connect(self._on_superseded)

    # ── submission ──────────────────────────────────────────────────────

    def submit(self, model: InferenceModel, request: InferenceRequest) -> str:
        """Queue ``request``, superseding anything pending for the same model.

        Returns the request id, which every signal echoes back.
        """
        lane = self._lanes.setdefault(model.id, _Lane(lock=threading.Lock()))
        previous = lane.latest_request_id
        lane.latest_request_id = request.id
        if previous:
            _log.debug("%s: request %s supersedes %s", model.id, request.id, previous)

        self._pool.start(_InferenceTask(model, request, lane, self._signals))
        return request.id

    def cancel(self, model_id: str) -> None:
        """Discard whatever is in flight for ``model_id``."""
        lane = self._lanes.get(model_id)
        if lane is not None:
            lane.latest_request_id = ""

    def cancel_all(self) -> None:
        for lane in self._lanes.values():
            lane.latest_request_id = ""

    # ── state ───────────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        return bool(self._running)

    @property
    def running_models(self) -> frozenset[str]:
        return frozenset(self._running)

    def wait_for_done(self, timeout_ms: int = 30_000) -> bool:
        """Block until the pool drains. For shutdown and tests only."""
        return self._pool.waitForDone(timeout_ms)

    def shutdown(self) -> None:
        self.cancel_all()
        if not self._pool.waitForDone(5_000):
            _log.warning("Inference pool still busy at shutdown; abandoning tasks")

    # ── signal relays (GUI thread) ──────────────────────────────────────

    def _on_started(self, model_id: str, request_id: str) -> None:
        was_busy = self.is_busy
        self._running.add(model_id)
        self.started.emit(model_id, request_id)
        if not was_busy:
            self.busy_changed.emit(True)

    def _on_finished(self, result: InferenceResult) -> None:
        self._clear_running(result.model_id)
        self.finished.emit(result)

    def _on_failed(self, model_id: str, request_id: str, message: str) -> None:
        self._clear_running(model_id)
        self.failed.emit(model_id, request_id, message)

    def _on_superseded(self, model_id: str, request_id: str) -> None:
        self._clear_running(model_id)
        self.superseded.emit(model_id, request_id)

    def _clear_running(self, model_id: str) -> None:
        was_busy = self.is_busy
        self._running.discard(model_id)
        if was_busy and not self.is_busy:
            self.busy_changed.emit(False)

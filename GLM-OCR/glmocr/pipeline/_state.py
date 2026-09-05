"""Shared mutable state for the three-stage async pipeline.

This object is created once per ``Pipeline.process()`` call and passed to
all three worker threads.  It holds the inter-thread queues, accumulated
results, and the UnitTracker reference.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, List, Optional

from glmocr.pipeline._unit_tracker import UnitTracker


class PipelineState:
    """Thread-safe container shared by loader / layout / recognition workers.

    Queues (dict messages flow through these):
        page_queue   — Stage 1 → Stage 2
        region_queue — Stage 2 → Stage 3

    Accumulated results (list, not a queue — main thread needs random access):
        recognition_results — Stage 3 appends, main thread snapshots
    """

    def __init__(
        self,
        page_maxsize: int = 100,
        region_maxsize: int = 2000,
    ):
        # ── Inter-thread queues ──────────────────────────────────────
        self.page_queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=page_maxsize)
        self.region_queue: queue.Queue[Dict[str, Any]] = queue.Queue(
            maxsize=region_maxsize
        )

        # ── Per-page data (stage 1 & 2 write, main thread reads) ─────
        self.images_dict: Dict[int, Any] = {}
        self.layout_results_dict: Dict[int, List] = {}

        # ── Counters (stage 1 writes, main thread reads after join) ──
        self.num_images_loaded: List[int] = [0]
        self.unit_indices_holder: List[Optional[List[int]]] = [None]

        # ── Recognition results (stage 3 appends, main thread reads) ─
        self._results_by_page: Dict[int, List[Dict]] = {}
        self._results_lock = threading.Lock()

        # ── Pre-cropped images for image-type regions ─────────────────
        self._image_region_store: Dict[int, Dict[tuple, Any]] = {}
        self._image_store_lock = threading.Lock()

        # ── Layout visualization images (page_idx → PIL Image) ────────
        self.layout_vis_images: Dict[int, Any] = {}

        # ── UnitTracker (set before threads start) ───────────────────
        self._tracker: Optional[UnitTracker] = None

        # ── Exception collection ─────────────────────────────────────
        self._exceptions: List[Dict[str, Any]] = []
        self._exception_lock = threading.Lock()

        # ── Shutdown coordination ─────────────────────────────────────
        self._shutdown_event = threading.Event()

    # ------------------------------------------------------------------
    # Shutdown helpers
    # ------------------------------------------------------------------

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_event.is_set()

    def request_shutdown(self) -> None:
        """Signal all workers to stop processing."""
        self._shutdown_event.set()
        tracker = self._tracker
        if tracker is not None:
            tracker.signal_shutdown()

    def safe_put(
        self, q: queue.Queue, msg: Dict[str, Any], timeout: float = 0.5
    ) -> bool:
        """Put *msg* on *q*, returning ``False`` if shutdown was requested."""
        while not self._shutdown_event.is_set():
            try:
                q.put(msg, timeout=timeout)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def drain_queue(q: queue.Queue) -> None:
        """Drain all items from *q* to unblock any blocked producers."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Page registration (delegated to tracker)
    # ------------------------------------------------------------------

    def register_page(self, page_idx: int, unit_idx: int) -> None:
        """Register a ``page_idx → unit_idx`` mapping in the tracker.

        Called by the data-loading worker (t1) for every loaded page.
        """
        tracker = self._tracker
        if tracker is not None:
            tracker.register_page(page_idx, unit_idx)

    # ------------------------------------------------------------------
    # Recognition results
    # ------------------------------------------------------------------

    def add_recognition_result(self, page_idx: int, region: Dict) -> None:
        """Append a completed region result and notify the tracker."""
        with self._results_lock:
            self._results_by_page.setdefault(page_idx, []).append(region)
        tracker = self._tracker
        if tracker is not None:
            tracker.on_region_done(page_idx)

    def get_grouped_results(self, page_indices: List[int]) -> List[List[Dict]]:
        """Return recognition results grouped by page for the given indices."""
        with self._results_lock:
            return [list(self._results_by_page.get(pi, [])) for pi in page_indices]

    def release_unit_data(self, page_indices: List[int]) -> None:
        """Release per-page data for a unit after it has been emitted.

        Frees recognition results and layout results so that memory is not
        held for the lifetime of the entire process() call.
        """
        with self._results_lock:
            for pi in page_indices:
                self._results_by_page.pop(pi, None)
        for pi in page_indices:
            self.layout_results_dict.pop(pi, None)

    # ------------------------------------------------------------------
    # Pre-cropped image store (for image-type regions)
    # ------------------------------------------------------------------

    def store_cropped_image(self, page_idx: int, bbox: list, image: Any) -> None:
        """Store a pre-cropped image for an image-type (skip) region.

        Called by the recognition worker for regions with ``task_type == "skip"``.
        """
        key = tuple(bbox)
        with self._image_store_lock:
            self._image_region_store.setdefault(page_idx, {})[key] = image

    def collect_cropped_images_for_unit(
        self, page_indices: List[int]
    ) -> Dict[tuple, Any]:
        """Collect pre-cropped images for one unit, re-keyed by local page index.

        Returns a dict mapping ``(local_page_idx, *bbox)`` → PIL Image.
        Consumed entries are removed from the store to free memory.
        """
        result: Dict[tuple, Any] = {}
        with self._image_store_lock:
            for local_idx, global_idx in enumerate(page_indices):
                page_store = self._image_region_store.pop(global_idx, {})
                for bbox_key, img in page_store.items():
                    result[(local_idx, *bbox_key)] = img
        return result

    # ------------------------------------------------------------------
    # UnitTracker lifecycle
    # ------------------------------------------------------------------

    def set_tracker(self, tracker: UnitTracker) -> None:
        """Attach *tracker* to the shared state.

        Must be called **before** any worker thread is started so that
        ``register_page``, ``finalize_unit``, and ``on_region_done`` are
        never no-ops.
        """
        self._tracker = tracker

    def finalize_unit(self, unit_idx: int, region_count: int) -> None:
        """Delegate to the tracker's ``finalize_unit`` if a tracker is attached.

        Called by the layout worker (t2) after it has processed all pages of
        *unit_idx*.
        """
        tracker = self._tracker
        if tracker is not None:
            tracker.finalize_unit(unit_idx, region_count)

    # ------------------------------------------------------------------
    # Exception handling
    # ------------------------------------------------------------------

    def record_exception(self, source: str, exc: Exception) -> None:
        with self._exception_lock:
            self._exceptions.append({"source": source, "exception": exc})
        self._shutdown_event.set()
        tracker = self._tracker
        if tracker is not None:
            tracker.signal_shutdown()

    def raise_if_exceptions(self) -> None:
        with self._exception_lock:
            if self._exceptions:
                raise RuntimeError(
                    "; ".join(
                        f"{e['source']}: {e['exception']}" for e in self._exceptions
                    )
                )

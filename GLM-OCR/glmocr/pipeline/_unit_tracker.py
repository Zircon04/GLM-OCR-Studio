"""UnitTracker — tracks per-unit (per input URL) region completion.

One "unit" corresponds to a single input URL (one image file or one PDF).
A PDF unit may span multiple pages, each page having multiple regions.

Fully dynamic registration protocol
------------------------------------
The tracker is created *before* any worker thread starts, knowing only
``num_units``.  All other metadata is registered incrementally:

1. **``register_page(page_idx, unit_idx)``** — called by Stage 1 / t1 for
   every page that is successfully loaded.  Builds the ``page → unit``
   mapping on the fly.

2. **``finalize_unit(u, region_count)``** — called by Stage 2 / t2 as soon as
   all pages of unit *u* have been layout-detected.  At that point the total
   region count for the unit is known and the tracker can check whether Stage 3
   has already finished all recognitions, immediately notifying the main thread
   if so.

3. **``on_region_done(page_idx)``** — called by Stage 3 / t3 for every
   completed recognition.  If the unit has already been finalised (its
   ``region_count`` is known) and the counter reaches the target, the unit is
   enqueued for the main thread.

4. **``signal_shutdown()``** — called when an error is recorded; puts a
   ``None`` sentinel on the ready queue so the main thread unblocks.

Thread safety:
  - ``register_page()``  is called from Thread 1 (data-loading worker).
  - ``finalize_unit()``  is called from Thread 2 (layout worker).
  - ``on_region_done()`` is called from Thread 3 (recognition worker).
  - ``wait_next_ready_unit()`` is called from the main thread.
  All mutations are serialised by a single lock.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, List, Optional


class UnitTracker:
    """Tracks region-level completion for each input unit.

    Args:
        num_units: Total number of input URLs being processed.
    """

    def __init__(self, num_units: int):
        self._num_units = num_units
        self._unit_image_indices: List[List[int]] = [[] for _ in range(num_units)]
        self._unit_region_count: List[Optional[int]] = [None] * num_units
        self._unit_for_image: Dict[int, int] = {}
        self._done_count: List[int] = [0] * num_units
        self._ready_queue: queue.Queue[Optional[int]] = queue.Queue()
        self._notified: set = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Phase 1: called from t1 for each loaded page
    # ------------------------------------------------------------------

    def register_page(self, page_idx: int, unit_idx: int) -> None:
        """Register a ``page_idx → unit_idx`` mapping.

        Called by the data-loading worker (t1) for every successfully loaded
        page, *before* the page is placed on the page queue.  This guarantees
        that by the time Stage 3 calls ``on_region_done(page_idx)``, the
        mapping is already present.
        """
        with self._lock:
            self._unit_image_indices[unit_idx].append(page_idx)
            self._unit_for_image[page_idx] = unit_idx

    # ------------------------------------------------------------------
    # Phase 2: called from t2 when all pages of a unit are layout-done
    # ------------------------------------------------------------------

    def finalize_unit(self, u: int, region_count: int) -> None:
        """Record the total region count for unit *u* and check completion.

        Called by the layout worker (t2) immediately after it has processed
        all pages belonging to unit *u*.  If Stage 3 has already finished all
        recognitions for that unit, the unit is enqueued for the main thread.
        """
        with self._lock:
            self._unit_region_count[u] = region_count
            if self._done_count[u] >= region_count and u not in self._notified:
                self._ready_queue.put(u)
                self._notified.add(u)

    # ------------------------------------------------------------------
    # Runtime: called from t3 after each region completes
    # ------------------------------------------------------------------

    def on_region_done(self, page_idx: int) -> None:
        """Increment the counter for the unit owning *page_idx*.

        O(1).  If the unit's region_count is known and the counter reaches
        the target, the unit is enqueued for the main thread.
        """
        with self._lock:
            u = self._unit_for_image.get(page_idx)
            if u is None:
                return
            self._done_count[u] += 1
            rc = self._unit_region_count[u]
            if rc is not None and self._done_count[u] >= rc and u not in self._notified:
                self._ready_queue.put(u)
                self._notified.add(u)

    # ------------------------------------------------------------------
    # Shutdown: wake up blocked main thread on error
    # ------------------------------------------------------------------

    def signal_shutdown(self) -> None:
        """Put a ``None`` sentinel on the ready queue to unblock the main thread."""
        self._ready_queue.put(None)

    # ------------------------------------------------------------------
    # Consumption: called from the main thread
    # ------------------------------------------------------------------

    def wait_next_ready_unit(self) -> Optional[int]:
        """Block until the next unit is ready and return its index.

        Returns ``None`` when ``signal_shutdown()`` has been called (error
        path).
        """
        return self._ready_queue.get()

    @property
    def num_units(self) -> int:
        return self._num_units

    @property
    def unit_image_indices(self) -> List[List[int]]:
        return self._unit_image_indices

    @property
    def unit_region_count(self) -> List[Optional[int]]:
        return self._unit_region_count

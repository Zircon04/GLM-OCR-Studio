#!/usr/bin/env python3
"""Offline memory-gate regressions; no OCR models or server required."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "manager"))
import job_runner as runner

GIB = 1024 ** 3


class MemoryProfileTests(unittest.TestCase):
    def setUp(self):
        runner.cancelled = False

    def tearDown(self):
        runner.cancelled = False

    def test_zero_total_memory_waits_then_recovers(self):
        with patch.object(runner.psutil, "virtual_memory", side_effect=[
            SimpleNamespace(total=0, available=0),
            SimpleNamespace(total=16 * GIB, available=4 * GIB),
        ]), patch.object(runner.time, "sleep"), patch.object(runner, "emit") as emit:
            runner.wait_for_memory()
        emit.assert_called_once()
        self.assertIsNone(emit.call_args.kwargs["free_percent"])

    def test_ten_percent_floor_cannot_be_disabled_by_small_reserve(self):
        with patch.object(runner, "read_profile", return_value={"reserved": 0}), \
             patch.object(runner.psutil, "virtual_memory", return_value=SimpleNamespace(total=64 * GIB, available=4 * GIB)), \
             patch.object(runner, "emit", side_effect=lambda *a, **k: runner.handle_signal(None, None)) as emit:
            runner.wait_for_memory()
        emit.assert_called_once()
        self.assertEqual(emit.call_args.kwargs["minimum_percent"], 10)

    def test_exact_explicit_reserve_in_bytes_allows_progress(self):
        with patch.object(runner, "read_profile", return_value={"reserved": 7, "memory": 32}), \
             patch.object(runner.psutil, "virtual_memory", return_value=SimpleNamespace(total=16 * GIB, available=7 * GIB)), \
             patch.object(runner.time, "sleep", side_effect=AssertionError("exact reserve should start")), \
             patch.object(runner, "emit") as emit:
            runner.wait_for_memory()
        emit.assert_not_called()

    def test_missing_memory_telemetry_waits_then_recovers(self):
        healthy = SimpleNamespace(total=16 * GIB, available=4 * GIB)
        with patch.object(runner.psutil, "virtual_memory", side_effect=[OSError("telemetry unavailable"), healthy]), \
             patch.object(runner.time, "sleep"), \
             patch.object(runner, "emit") as emit:
            runner.wait_for_memory()
        emit.assert_called_once()
        self.assertEqual(emit.call_args.args[0], "memory_wait")
        self.assertIsNone(emit.call_args.kwargs["free_percent"])

    def test_resume_records_current_profile_without_reprocessing_done_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "page.png"
            source.write_bytes(b"fixture")
            output = Path(temp) / "result"
            output.mkdir()
            old_profile = {"name": "auto", "workers": 4, "pages": 6, "regions": 112, "memory": 16, "reserved": 7}
            profile = {**old_profile, "workers": 2, "pages": 2, "regions": 32, "reserved": 3}
            state = {"source": str(source), "output": str(output), "result_name": "page", "status": "paused", "profile": old_profile,
                     "started_at": time.time(), "processing_seconds": 0, "total_pages": 2,
                     "chunks": [{"id": "done", "start": 0, "end": 1, "status": "done", "merged": True, "purged": True},
                                {"id": "todo", "start": 1, "end": 2, "status": "paused", "input": str(source)}]}
            runner.atomic_json(output / "job-state.json", state)
            args = argparse.Namespace(input=str(source), output=str(output), enhance="off", exports=False, export_formats="", chunk_size=None)
            def stop_ocr(*_args):
                runner.handle_signal(None, None)
                return 130
            with patch.object(runner, "read_profile", return_value=profile), \
                 patch.object(runner.psutil, "virtual_memory", return_value=SimpleNamespace(total=16 * GIB, available=4 * GIB)), \
                 patch.object(runner, "emit"), \
                 patch.object(runner, "run_streaming", side_effect=stop_ocr) as launch:
                self.assertEqual(runner.run_job(args), 130)
            launch.assert_called_once()
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("pipeline.max_workers") + 1], "2")
            saved = json.loads((output / "job-state.json").read_text())
            self.assertEqual(saved["chunks"][0]["status"], "done")
            self.assertEqual(saved["chunks"][1]["status"], "paused")
            self.assertEqual(saved["profile"], profile)

    def test_cancel_during_memory_wait_never_starts_chunk(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "page.png"
            source.write_bytes(b"fixture; OCR must not be launched")
            output = Path(temp) / "result"
            args = argparse.Namespace(input=str(source), output=str(output), enhance="off", exports=False, export_formats="", chunk_size=None)
            def event(kind, **payload):
                if kind == "memory_wait":
                    runner.handle_signal(None, None)
            with patch.object(runner.psutil, "virtual_memory", return_value=SimpleNamespace(total=16 * GIB, available=GIB)), \
                 patch.object(runner, "emit", side_effect=event) as emit, \
                 patch.object(runner, "run_streaming", side_effect=AssertionError("cancelled memory wait must not launch OCR")) as launch:
                self.assertEqual(runner.run_job(args), 130)
            launch.assert_not_called()
            state = json.loads((output / "job-state.json").read_text())
            self.assertEqual(state["status"], "paused")
            self.assertEqual(state["chunks"][0]["status"], "pending")
            self.assertNotIn("chunk_started", [call.args[0] for call in emit.call_args_list])

    def test_explicit_large_reserve_is_not_relaxed_or_rounded_down(self):
        for declared, available in [(16, int(6.9 * GIB)), (32, 4 * GIB)]:
            with self.subTest(declared_total=declared, available=available), \
                 patch.object(runner, "read_profile", return_value={"reserved": 7, "memory": declared}), \
                 patch.object(runner, "memory_free_percent", return_value=int(available / (16 * GIB) * 100)), \
                 patch.object(runner.psutil, "virtual_memory", return_value=SimpleNamespace(total=16 * GIB, available=available, percent=100 - available / (16 * GIB) * 100)), \
                 patch.object(runner, "emit", side_effect=lambda *a, **k: runner.handle_signal(None, None)) as emit:
                runner.cancelled = False
                runner.wait_for_memory()
                emit.assert_called_once()
                self.assertEqual(emit.call_args.args[0], "memory_wait")

    def test_memory_wait_cancels_promptly(self):
        timer = threading.Timer(0.05, runner.handle_signal, args=(None, None))
        with patch.object(runner, "memory_free_percent", return_value=1), \
             patch.object(runner.psutil, "virtual_memory", return_value=SimpleNamespace(total=16 * GIB, available=GIB, percent=99)), \
             patch.object(runner, "emit"):
            started = time.monotonic()
            timer.start()
            try:
                runner.wait_for_memory()
            finally:
                timer.join()
        self.assertLess(time.monotonic() - started, 1.0, "cancel must interrupt the 5-second memory poll")

    def test_gui_auto_generation_keeps_stock_16gb_limits(self):
        # Swift/AppKit is not available in the offline Linux test environment.
        # Inspect only the explicit <=16 GiB branch used to generate this file.
        swift = (ROOT / "gui/main.swift").read_text(encoding="utf-8")
        auto = next(line for line in swift.splitlines() if 'case "auto": return PerformanceProfile' in line)
        for key, expected in {"serverSequences": 4, "sdkWorkers": 2, "pageQueue": 2, "regionQueue": 32, "reservedGB": 3}.items():
            with self.subTest(field=key):
                match = re.search(rf"{key}: m <= 16 \? (\d+)", auto)
                self.assertIsNotNone(match)
                self.assertEqual(int(match.group(1)), expected)

    def test_stock_16gb_profile_starts_with_4gb_available(self):
        with patch.dict("os.environ", {"GLMOCR_PROFILE_FILE": str(ROOT / "config/performance-profile.env")}), \
             patch.object(runner, "memory_free_percent", return_value=25), \
             patch.object(runner.psutil, "virtual_memory", return_value=SimpleNamespace(total=16 * GIB, available=4 * GIB, percent=75)), \
             patch.object(runner.time, "sleep", side_effect=AssertionError("stock profile waits indefinitely at 4 GiB available")):
            runner.wait_for_memory()
            profile = runner.read_profile()
        self.assertGreaterEqual(profile["reserved"], 3)
        self.assertLessEqual(profile["workers"], 2)
        self.assertLessEqual(profile["pages"], 2)
        self.assertLessEqual(profile["regions"], 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)

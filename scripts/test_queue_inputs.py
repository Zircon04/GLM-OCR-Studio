#!/usr/bin/env python3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import night_queue
import queue_inputs


class QueueInputTests(unittest.TestCase):
    def test_zip_roundtrip_and_reuse_without_archive(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pdf = root / "book.pdf"
            pdf.write_bytes(b"%PDF-1.4 test")
            archive = root / "pack.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.write(pdf, "book.pdf")
            unpacked = queue_inputs.prepare_input(archive, root / "work")
            self.assertTrue(queue_inputs.manifest_matches(unpacked))
            archive.unlink()
            again = queue_inputs.prepare_input(root / "work" / "pack", root / "work")
            self.assertEqual(again, unpacked)

    def test_second_supervisor_is_refused_and_low_disk_pauses(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "one.pdf"
            source.write_bytes(b"%PDF")
            calls = []

            def runner(item, output):
                calls.append(item)
                output.mkdir(parents=True)
                (output / "job-state.json").write_text('{"status":"complete"}', encoding="utf-8")
                return 0

            code = night_queue.run_queue([source], root / "out", root / "work", runner, plugged=lambda: True, free=lambda _path: 10**12, stop_below=1)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 1)
            again = night_queue.run_queue([source], root / "out", root / "work", runner, plugged=lambda: True, free=lambda _path: 10**12, stop_below=1)
            self.assertEqual(again, 0)
            self.assertEqual(len(calls), 1)
            paused = night_queue.run_queue([root / "two.pdf"], root / "out", root / "other", runner, plugged=lambda: False, free=lambda _path: 10**12, stop_below=1)
            self.assertEqual(paused, 75)
            lock = root / "locked" / night_queue.LOCK_NAME
            fd = night_queue.acquire_lock(lock)
            try:
                busy = night_queue.run_queue([source], root / "out", root / "locked", runner, plugged=lambda: True, free=lambda _path: 10**12, stop_below=1)
                self.assertEqual(busy, 2)
            finally:
                night_queue.release_lock(fd, lock)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""The reduced outputs and optimization must precede destructive cleanup."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import job_runner as runner


class JobOptimizationTests(unittest.TestCase):
    def tearDown(self):
        runner.cancelled = False

    def test_cli_defaults_are_only_requested_three_formats(self):
        with patch.object(sys, "argv", ["job_runner", "--input", "source.pdf", "--output", "result"]), patch.object(runner, "run_job", return_value=0) as run:
            self.assertEqual(runner.main(), 0)
            self.assertEqual(run.call_args.args[0].export_formats, "md,split_md,searchable_pdf")

    def test_optimization_failure_and_pause_keep_metadata_for_retry(self):
        for stop in (False, True):
            with self.subTest(cancelled=stop), tempfile.TemporaryDirectory() as temp:
                runner.cancelled = False
                root = Path(temp); source = root / "scan.png"; source.touch()
                output = root / "out"; book = output / "scan"; book.mkdir(parents=True)
                metadata = book / "scan.json"; metadata.write_text("[[]]")
                (book / "scan.md").write_text("# Book")
                state = {"source": str(source), "output": str(output), "result_name": "scan", "status": "paused",
                         "started_at": time.time(), "total_pages": 1,
                         "chunks": [{"id": "done", "start": 0, "end": 1, "status": "done", "merged": True, "purged": True}]}
                runner.atomic_json(output / "job-state.json", state)
                args = argparse.Namespace(input=str(source), output=str(output), enhance="off", exports=True,
                                          export_formats="md,split_md,searchable_pdf", chunk_size=None)
                def streaming(command, stage, *extra):
                    if stage == "optimize":
                        runner.cancelled = stop
                        return 7
                    return 0
                with patch.object(runner, "emit"), patch.object(runner, "run_streaming", side_effect=streaming), patch.object(runner.subprocess, "run"):
                    self.assertEqual(runner.run_job(args), 130 if stop else 7)
                self.assertTrue(metadata.exists())
                self.assertEqual(json.loads((output / "job-state.json").read_text())["status"], "paused" if stop else "error")
                runner.cancelled = False
                with patch.object(runner, "emit"), patch.object(runner, "run_streaming", return_value=0) as stages, patch.object(runner.subprocess, "run"):
                    self.assertEqual(runner.run_job(args), 0)
                    self.assertEqual([call.args[1] for call in stages.call_args_list], ["export", "optimize"])

    def test_optimization_runs_after_exports_and_before_cleanup(self):
        runner.cancelled = False
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "scan.png"; source.touch()
            output = root / "output"
            book = output / "scan"; book.mkdir(parents=True)
            metadata = book / "scan.json"; metadata.write_text("[[]]")
            (book / "scan.md").write_text("# Book")
            state = {"source": str(source), "output": str(output), "result_name": "scan", "status": "paused",
                     "started_at": time.time(), "total_pages": 1,
                     "chunks": [{"id": "done", "start": 0, "end": 1, "status": "done", "merged": True, "purged": True}]}
            runner.atomic_json(output / "job-state.json", state)
            args = argparse.Namespace(input=str(source), output=str(output), enhance="off", exports=True,
                                      export_formats="md,split_md,searchable_pdf", chunk_size=None)
            order = []
            def streaming(command, stage, *extra, **kwargs):
                order.append(stage)
                self.assertTrue(metadata.exists())
                return 0
            finalize = runner.finalize_book_dir
            def cleanup(*args):
                order.append("cleanup")
                finalize(*args)
            with patch.object(runner, "emit"), patch.object(runner, "run_streaming", side_effect=streaming), patch.object(runner, "finalize_book_dir", side_effect=cleanup), patch.object(runner.subprocess, "run"):
                self.assertEqual(runner.run_job(args), 0)
            self.assertEqual(order, ["export", "optimize", "cleanup"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

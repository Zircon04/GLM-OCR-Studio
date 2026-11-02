#!/usr/bin/env python3
"""Failed exports must remain resumable without another OCR pass."""
import argparse
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manager"))
import job_runner as runner


class ExportFailureTests(unittest.TestCase):
    def test_export_failure_preserves_metadata_and_resumes_without_ocr(self):
        runner.cancelled = False
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "scan.png"
            source.touch()
            output = root / "output"
            book = output / "scan"
            book.mkdir(parents=True)
            (book / "scan.md").write_text("# Test\n\n$$x=1$$\n")
            metadata = book / "scan.json"
            metadata.write_text('[[]]')
            state = {"source": str(source), "output": str(output), "result_name": "scan", "status": "paused",
                     "started_at": time.time(), "total_pages": 1,
                     "chunks": [{"id": "done", "start": 0, "end": 1, "status": "done", "merged": True, "purged": True}]}
            state_path = output / "job-state.json"
            runner.atomic_json(state_path, state)
            args = argparse.Namespace(input=str(source), output=str(output), enhance="off", exports=True,
                                      export_formats="searchable_pdf", chunk_size=None)
            with patch.object(runner, "emit") as emit, patch.object(runner, "run_streaming", return_value=3) as export, patch.object(runner.subprocess, "run"):
                self.assertEqual(runner.run_job(args), 3)
                export.assert_called_once()
                self.assertEqual(export.call_args.args[1], "export")
                self.assertNotIn("complete", [call.args[0] for call in emit.call_args_list])
            self.assertTrue(metadata.is_file(), "OCR metadata needed to retry PDF export was deleted")
            self.assertTrue((output / ".job").exists())
            self.assertEqual(json.loads(state_path.read_text())["status"], "error")
            with patch.object(runner, "emit"), patch.object(runner, "run_streaming", return_value=0) as export, patch.object(runner.subprocess, "run"):
                self.assertEqual(runner.run_job(args), 0)
                self.assertEqual([call.args[1] for call in export.call_args_list], ["export", "optimize"])
            self.assertEqual(json.loads(state_path.read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)

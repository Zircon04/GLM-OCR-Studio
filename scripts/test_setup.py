#!/usr/bin/env python3
"""Offline setup/enhancement regressions; never install into real environments."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
from unittest import mock
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="glm-setup-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / "setup.command", self.root)
        shutil.copytree(ROOT / "requirements", self.root / "requirements")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/build_app.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.root / "scripts/build_app.sh").chmod(0o755)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.log"
        fake_python = self.bin / "python3.12"
        fake_python.write_text('''#!/bin/sh
case "$1" in
-c) echo 3.12;;
--version) echo 'Python 3.12.0';;
-m)
  if [ "$2" = venv ]; then
    mkdir -p "$3/bin"
    cp "$0" "$3/bin/python"
    cp "$0" "$3/bin/pip"
    echo "create $3" >> "$SETUP_TEST_LOG"
  elif [ "$2" = ensurepip ]; then
    rm -f "$(dirname "$0")/no-pip"
    echo "seed-pip $0" >> "$SETUP_TEST_LOG"
  elif [ "$2" = pip ]; then
    if [ -f "$(dirname "$0")/no-pip" ]; then exit 19; fi
    shift 2
    echo "$0 $*" >> "$SETUP_TEST_LOG"
    if [ "${SETUP_TEST_FAIL:-}" = 1 ]; then exit 17; fi
  fi;;
*)
  echo "$0 $*" >> "$SETUP_TEST_LOG"
  if [ "${SETUP_TEST_FAIL:-}" = 1 ]; then exit 17; fi;;
esac
''')
        fake_python.chmod(0o755)
        (self.bin / "swiftc").write_text("#!/bin/sh\nexit 0\n")
        (self.bin / "swiftc").chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin",
                        SETUP_TEST_LOG=str(self.log))

    def run_setup(self):
        return subprocess.run([shutil.which("zsh") or "bash", "setup.command"],
                              cwd=self.root, env=self.env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)

    def calls(self):
        return self.log.read_text() if self.log.exists() else ""

    def test_incomplete_venv_creation_resumes_without_deleting_files(self):
        env = self.root / ".venv-mlx"
        env.mkdir()
        marker = env / "owner-file"
        marker.write_text("preserve")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertIn("-r requirements/mlx.txt", self.calls())

    def test_fresh_setup_installs_all_requirements(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout)
        for name in ("mlx", "enhance", "sdk"):
            self.assertIn(f"create .venv-{name}", self.calls())
            self.assertIn(f"-r requirements/{name}.txt", self.calls())
        self.assertIn("./GLM-OCR[layout]", (self.root / "requirements/sdk.txt").read_text())

    def test_failed_install_is_not_reported_successful_and_can_resume(self):
        self.env["SETUP_TEST_FAIL"] = "1"
        failed = self.run_setup()
        self.assertNotEqual(failed.returncode, 0)
        self.assertNotIn("Установка успешно завершена", failed.stdout)
        del self.env["SETUP_TEST_FAIL"]
        resumed = self.run_setup()
        self.assertEqual(resumed.returncode, 0, resumed.stdout)
        self.assertIn("-r requirements/mlx.txt", self.calls())

    def test_uv_environment_without_pip_is_repaired_in_place(self):
        for name in ("mlx", "enhance", "sdk"):
            env = self.root / f".venv-{name}" / "bin"
            env.mkdir(parents=True)
            shutil.copy2(self.bin / "python3.12", env / "python")
            (env / "no-pip").touch()
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self.calls().count("seed-pip"), 3)
        self.assertNotIn("create ", self.calls())

    def test_existing_environments_resume_requirements_installation(self):
        for name in ("mlx", "enhance", "sdk"):
            env = self.root / f".venv-{name}" / "bin"
            env.mkdir(parents=True)
            for command in ("python", "pip"):
                shutil.copy2(self.bin / "python3.12", env / command)
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout)
        for name in ("mlx", "enhance", "sdk"):
            self.assertIn(f"-r requirements/{name}.txt", self.calls())
        self.assertNotIn("create ", self.calls(), "Never recreate existing owner environments")


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy
            from PIL import Image
        except ImportError:
            self.skipTest("Image regression fixtures require numpy and Pillow")
        self.np, self.Image = numpy, Image
        self.temp = tempfile.TemporaryDirectory(prefix="glm-enhance-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        imgs = self.root / "book/imgs"
        imgs.mkdir(parents=True)
        self.src = imgs / "sample.png"
        Image.new("RGB", (12, 10), "white").save(self.src)
        (imgs.parent / "book.md").write_text("![scan](imgs/sample.png)")
        self.original = self.src.read_bytes()
        self.pipeline = types.ModuleType("realesrgan_mlx.pipeline_mlx")
        self.pipeline.make_upsampler = mock.Mock(side_effect=FileNotFoundError("weights not cached"))
        mlx_core = types.ModuleType("mlx.core")
        mlx_core.default_device = lambda: "test-device"
        self.stubs = {"cv2": types.ModuleType("cv2"), "mlx": types.ModuleType("mlx"),
                      "mlx.core": mlx_core, "realesrgan_mlx": types.ModuleType("realesrgan_mlx"),
                      "realesrgan_mlx.pipeline_mlx": self.pipeline}

    def run_enhance(self, mode="hybrid"):
        spec = importlib.util.spec_from_file_location("enhance_results_test", ROOT / "enhancement/enhance_results.py")
        module = importlib.util.module_from_spec(spec)
        output = io.StringIO()
        # Metal/OpenCV are platform dependencies. Exercise the real orchestration,
        # file output and manifest, replacing only restoration/inference at the seam.
        with mock.patch.dict(os.environ), mock.patch.dict(sys.modules, self.stubs), mock.patch.object(sys, "argv", ["enhance_results.py", str(self.root), "--mode", mode]), contextlib.redirect_stdout(output):
            spec.loader.exec_module(module)
            module.classical_restore = lambda image: image.copy()
            result = module.main()
        return result, output.getvalue()

    def test_cached_hybrid_upsampler_still_upscales(self):
        upsampler = mock.Mock()
        upsampler.enhance.side_effect = lambda image, outscale: (self.np.repeat(self.np.repeat(image, 2, axis=0), 2, axis=1), "RGB")
        self.pipeline.make_upsampler.side_effect = None
        self.pipeline.make_upsampler.return_value = upsampler
        code, output = self.run_enhance()
        self.assertEqual(code, 0)
        self.assertIn("ENHANCE_DONE 1 1", output)
        manifest = json.loads((self.root / "image-enhancement-manifest.json").read_text())
        self.assertEqual(manifest[0]["enhanced_size"], [24, 20])
        self.assertIn("Real-ESRGAN", manifest[0]["method"])
        enhanced_md = self.src.parent.parent / "book_enhanced.md"
        self.assertIn("imgs_enhanced/sample.png", enhanced_md.read_text())
        self.assertEqual(self.src.read_bytes(), self.original)

    def test_missing_optional_package_falls_back_in_hybrid_mode(self):
        self.stubs["realesrgan_mlx.pipeline_mlx"] = None
        code, output = self.run_enhance()
        self.assertEqual(code, 0)
        self.assertIn("ENHANCE_WARNING", output)
        self.assertIn("ENHANCE_DONE 1 0", output)

    def test_no_images_does_not_initialize_upscaler(self):
        self.src.unlink()
        code, output = self.run_enhance()
        self.assertEqual(code, 0)
        self.assertIn("ENHANCE_DONE 0 0", output)
        self.pipeline.make_upsampler.assert_not_called()

    def test_inference_failure_still_saves_classical_result(self):
        upsampler = mock.Mock()
        upsampler.enhance.side_effect = RuntimeError("Metal allocation failed")
        self.pipeline.make_upsampler.side_effect = None
        self.pipeline.make_upsampler.return_value = upsampler
        code, output = self.run_enhance()
        self.assertEqual(code, 0)
        self.assertIn("ENHANCE_WARNING", output)
        self.assertIn("ENHANCE_DONE 1 0", output)
        self.assertTrue((self.src.parent.parent / "imgs_enhanced/sample.png").is_file())
        self.assertEqual(list(self.root.rglob("*.pre_esrgan.png")), [])
        self.assertEqual(self.src.read_bytes(), self.original)

    def test_hybrid_cannot_enable_implicit_model_downloads(self):
        seen = []
        def load(*args, **kwargs):
            seen.append(os.environ.get("HF_HUB_OFFLINE"))
            raise FileNotFoundError("not cached")
        self.pipeline.make_upsampler.side_effect = load
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            self.run_enhance()
        self.assertEqual(seen, ["1"])

    def test_safe_mode_does_not_require_metal_or_realesrgan(self):
        self.stubs["mlx.core"] = None
        self.stubs["realesrgan_mlx.pipeline_mlx"] = None
        code, output = self.run_enhance("safe")
        self.assertEqual(code, 0)
        self.assertIn("ENHANCE_DONE 1 0", output)

    def test_missing_hybrid_weights_preserve_classical_enhancement(self):
        code, output = self.run_enhance()
        self.assertEqual(code, 0)
        self.assertIn("ENHANCE_WARNING", output)
        self.assertIn("ENHANCE_DONE 1 0", output)
        self.assertEqual(self.src.read_bytes(), self.original)
        self.assertTrue((self.src.parent.parent / "imgs_enhanced/sample.png").is_file())
        manifest = json.loads((self.root / "image-enhancement-manifest.json").read_text())
        self.assertEqual(len(manifest), 1)
        self.assertNotIn("Real-ESRGAN", manifest[0]["method"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Coordinator — orchestrates multi-GPU engine startup, file sharding,
worker launching, progress monitoring, and graceful shutdown."""

import io
import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
import concurrent.futures
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from engine import read_progress, start_engine, wait_for_service
from gpu_utils import (
    _print_err,
    get_gpu_info,
    shard_files,
    collect_files,
    find_available_ports,
    filter_available_gpus,
)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class _TeeWriter:
    """Write to both a terminal stream and a log file simultaneously."""

    def __init__(self, terminal: io.TextIOBase, log_file: io.TextIOBase):
        self._terminal = terminal
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._terminal.write(data)
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()

    def fileno(self) -> int:
        return self._terminal.fileno()

    def isatty(self) -> bool:
        return self._terminal.isatty()


class Coordinator:
    """Orchestrates multi-GPU OCR processing."""

    def __init__(self, args):
        self.args = args
        self.engine_procs: Dict[int, subprocess.Popen] = {}
        self.worker_procs: Dict[int, subprocess.Popen] = {}
        self.progress_files: Dict[int, str] = {}
        self.file_handles: List[Any] = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path("logs") / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._shutdown = False
        self._start_time = time.time()
        self._orig_stdout: Any = None
        self._orig_stderr: Any = None
        self._main_log_fh: Any = None

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._install_tee()

        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        try:
            self._run_impl()
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
            self._cleanup()
            self._uninstall_tee()

    def _install_tee(self) -> None:
        """Redirect stdout and stderr so all output also goes to main.log.

        The original stderr is saved as ``_orig_stderr`` and passed to tqdm
        directly, so progress-bar control characters never reach the log.
        """
        log_path = self.log_dir / "main.log"
        self._main_log_fh = open(log_path, "w")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._orig_stdout, self._main_log_fh)
        sys.stderr = _TeeWriter(self._orig_stderr, self._main_log_fh)

    def _uninstall_tee(self) -> None:
        """Restore original stdout/stderr."""
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        if self._main_log_fh is not None:
            try:
                self._main_log_fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _on_signal(self, signum, frame):
        if self._shutdown:
            _print_err("\n[FORCE] Second signal received, force killing...")
            self._force_kill_all()
            os._exit(1)
        _print_err(
            f"\n[INFO] Signal {signum} received, shutting down gracefully..."
        )
        self._shutdown = True

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run_impl(self) -> None:
        self._step1_detect_gpus()
        if self._shutdown:
            return

        available = self._available_gpus
        n_gpus = len(available)

        total_files, files, shards, input_root = self._step2_collect_files(
            n_gpus
        )
        if self._shutdown:
            return

        gpu_port_map = self._step3_start_engines(available)
        if self._shutdown:
            return

        ready_pairs, shards = self._step4_wait_services(
            available, gpu_port_map, shards, total_files
        )
        if self._shutdown or not ready_pairs:
            return

        self._step5_start_workers(ready_pairs, shards, input_root)
        if self._shutdown:
            return

        self._monitor_progress(total_files)
        self._print_summary(total_files)

    # ------------------------------------------------------------------
    # Step 1 — Detect GPUs
    # ------------------------------------------------------------------

    def _step1_detect_gpus(self) -> None:
        print("=" * 60)
        print("  GLM-OCR Multi-GPU Launcher")
        print("=" * 60)

        gpus = get_gpu_info()
        if not gpus:
            _print_err("[ERROR] No GPUs found.")
            sys.exit(1)

        print(f"\n[1/5] Detected {len(gpus)} GPU(s):")
        for g in gpus:
            pct = g["used_mb"] / max(g["total_mb"], 1)
            bar_len = int(pct * 20)
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            print(
                f"  GPU {g['id']:>1}: {g['name']:<26} "
                f"[{bar}] {g['used_mb']:>5}/{g['total_mb']}MB "
                f"(free: {g['free_mb']}MB)"
            )

        gpu_ids = None
        if self.args.gpus != "auto":
            gpu_ids = [int(x.strip()) for x in self.args.gpus.split(",")]

        available = filter_available_gpus(gpus, self.args.min_free_mb, gpu_ids)
        if not available:
            _print_err(
                f"\n[ERROR] No GPUs have >= {self.args.min_free_mb}MB "
                "free memory."
            )
            sys.exit(1)

        print(
            f"\n  Using {len(available)} GPU(s): "
            f"{[g['id'] for g in available]}"
        )
        self._available_gpus = available

    # ------------------------------------------------------------------
    # Step 2 — Collect and shard files
    # ------------------------------------------------------------------

    def _step2_collect_files(
        self, n_gpus: int
    ) -> Tuple[int, List[str], List[List[str]], Optional[str]]:
        print(f"\n[2/5] Scanning: {self.args.input}")
        files = collect_files(self.args.input)
        total_files = len(files)
        print(f"  Found {total_files} file(s)")

        shards = shard_files(files, n_gpus)
        for gpu, shard in zip(self._available_gpus, shards):
            print(f"  GPU {gpu['id']}: {len(shard)} files")

        input_path = Path(self.args.input)
        input_root = (
            str(input_path.absolute()) if input_path.is_dir() else None
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        return total_files, files, shards, input_root

    # ------------------------------------------------------------------
    # Step 3 — Start engine services
    # ------------------------------------------------------------------

    def _step3_start_engines(self, available: List[Dict]) -> Dict[int, int]:
        print(f"\n[3/5] Starting {self.args.engine} services...")

        gpu_port_map: Dict[int, int] = {}

        ports = find_available_ports(self.args.base_port, len(available))
        if len(ports) < len(available):
            _print_err(
                f"  [WARN] Only found {len(ports)} available ports "
                f"(need {len(available)}). Some GPUs will be skipped."
            )

        for i, gpu in enumerate(available):
            if self._shutdown:
                break
            if i >= len(ports):
                _print_err(f"  [SKIP] GPU {gpu['id']}: no available port")
                continue
            port = ports[i]
            gpu_id = gpu["id"]

            proc, log_path, log_fh = start_engine(
                engine=self.args.engine,
                model=self.args.model,
                gpu_id=gpu_id,
                port=port,
                log_dir=str(self.log_dir),
                extra_args=self.args.engine_args or "",
            )
            self.engine_procs[gpu_id] = proc
            self.file_handles.append(log_fh)
            gpu_port_map[gpu_id] = port
            print(
                f"  GPU {gpu_id} -> port {port}  "
                f"(pid {proc.pid}, log: {log_path.name})"
            )

        return gpu_port_map

    # ------------------------------------------------------------------
    # Step 4 — Wait for services to be ready
    # ------------------------------------------------------------------

    def _step4_wait_services(
        self,
        available: List[Dict],
        gpu_port_map: Dict[int, int],
        shards: List[List[str]],
        total_files: int,
    ) -> Tuple[List[Tuple[int, int]], List[List[str]]]:
        print(
            f"\n[4/5] Waiting for services to be ready "
            f"(timeout: {self.args.timeout}s)..."
        )

        ready_pairs: List[Tuple[int, int]] = []
        ready_shard_indices: List[int] = []

        future_map: Dict[
            concurrent.futures.Future, Tuple[int, int, int]
        ] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(available)
        ) as executor:
            for i, gpu in enumerate(available):
                gpu_id = gpu["id"]
                if gpu_id not in gpu_port_map:
                    continue
                port = gpu_port_map[gpu_id]
                proc = self.engine_procs[gpu_id]
                future = executor.submit(
                    wait_for_service, port, proc, self.args.timeout
                )
                future_map[future] = (i, gpu_id, port)

            for future in concurrent.futures.as_completed(future_map):
                if self._shutdown:
                    break
                i, gpu_id, port = future_map[future]
                success, elapsed = future.result()
                if success:
                    print(
                        f"  GPU {gpu_id} (port {port}): "
                        f"Ready  ({elapsed}s)"
                    )
                    ready_pairs.append((gpu_id, port))
                    ready_shard_indices.append(i)
                else:
                    proc = self.engine_procs[gpu_id]
                    if proc.poll() is not None:
                        print(
                            f"  GPU {gpu_id} (port {port}): CRASHED  "
                            f"(exit={proc.returncode}, {elapsed}s)"
                        )
                    else:
                        print(
                            f"  GPU {gpu_id} (port {port}): "
                            f"TIMEOUT  ({elapsed}s)"
                        )
                        self._kill_proc(proc)

        if not ready_pairs:
            _print_err(
                "\n[ERROR] No engine services started successfully!\n"
                f"  Check logs: {self.log_dir}"
            )
            return [], []

        n_ready = len(ready_pairs)
        n_total = len(available)
        if n_ready < n_total:
            all_files: List[str] = []
            for shard in shards:
                all_files.extend(shard)
            shards = shard_files(all_files, n_ready)
            print(
                f"\n  {n_ready}/{n_total} GPUs ready. "
                f"Redistributed {total_files} files across "
                f"{n_ready} GPU(s)."
            )
        else:
            shards = [shards[i] for i in ready_shard_indices]

        return ready_pairs, shards

    # ------------------------------------------------------------------
    # Step 5 — Start workers
    # ------------------------------------------------------------------

    def _step5_start_workers(
        self,
        ready_pairs: List[Tuple[int, int]],
        shards: List[List[str]],
        input_root: Optional[str],
    ) -> None:
        print("\n[5/5] Starting workers...")

        entry_point = os.path.join(_PACKAGE_DIR, "launch.py")

        for (gpu_id, port), shard in zip(ready_pairs, shards):
            if self._shutdown:
                break

            filelist_path = str(self.log_dir / f"shard_gpu{gpu_id}.json")
            with open(filelist_path, "w") as f:
                json.dump(shard, f)

            progress_path = str(
                self.log_dir / f"progress_gpu{gpu_id}.json"
            )
            self.progress_files[gpu_id] = progress_path

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            worker_cmd = [
                sys.executable,
                entry_point,
                "--worker",
                "--gpu-id",
                str(gpu_id),
                "--port",
                str(port),
                "--filelist",
                filelist_path,
                "--output",
                self.args.output,
                "--progress-file",
                progress_path,
                "--log-level",
                self.args.log_level or "WARNING",
            ]
            if input_root:
                worker_cmd.extend(["--input-root", input_root])
            if self.args.config:
                worker_cmd.extend(["--config", self.args.config])
            if getattr(self.args, "no_save", False):
                worker_cmd.append("--no-save")

            worker_log = self.log_dir / f"worker_gpu{gpu_id}.log"
            wfh = open(worker_log, "w")
            self.file_handles.append(wfh)

            proc = subprocess.Popen(
                worker_cmd,
                env=env,
                stdout=wfh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.worker_procs[gpu_id] = proc
            print(
                f"  GPU {gpu_id}: {len(shard)} files "
                f"-> worker pid {proc.pid}"
            )

    # ------------------------------------------------------------------
    # Progress monitoring
    # ------------------------------------------------------------------

    def _monitor_progress(self, total_files: int) -> None:
        print(f"\n{'=' * 60}")

        try:
            from tqdm import tqdm

            pbar: Any = tqdm(
                total=total_files,
                desc="Total",
                unit="file",
                file=self._orig_stderr or sys.stderr,
                dynamic_ncols=True,
            )
        except ImportError:
            pbar = None

        last_total = 0
        dead_engines: set = set()

        while not self._shutdown:
            self._check_engines(dead_engines)

            all_done = True
            total_completed = 0
            total_failed = 0
            gpu_display: Dict[int, str] = {}

            for gpu_id, proc in self.worker_procs.items():
                prog = read_progress(
                    self.progress_files.get(gpu_id, "")
                )
                if prog:
                    total_completed += prog["completed"]
                    total_failed += prog["failed"]
                    gpu_display[gpu_id] = (
                        f"{prog['completed']}/{prog['total']}"
                    )
                    status = prog["status"]
                else:
                    gpu_display[gpu_id] = "init"
                    status = "init"

                alive = proc.poll() is None
                done_status = status in ("done", "done_with_errors")
                errored = status.startswith("error")

                if alive and not done_status and not errored:
                    all_done = False
                elif not alive and not done_status:
                    gpu_display[gpu_id] += f"(exit:{proc.returncode})"

            delta = total_completed - last_total
            if pbar and delta > 0:
                pbar.update(delta)
                last_total = total_completed

            if pbar:
                parts = [
                    f"G{gid}:{s}"
                    for gid, s in sorted(gpu_display.items())
                ]
                pbar.set_postfix_str(" ".join(parts), refresh=True)

            if all_done:
                total_completed, total_failed = self._aggregate_progress()
                delta = total_completed - last_total
                if pbar and delta > 0:
                    pbar.update(delta)
                break

            time.sleep(1)

        if pbar:
            pbar.close()

    def _check_engines(self, dead_engines: set) -> None:
        """Check engine processes and kill workers whose engine has died."""
        for gpu_id, proc in self.engine_procs.items():
            if gpu_id in dead_engines:
                continue
            if proc.poll() is not None:
                dead_engines.add(gpu_id)
                print(
                    f"\n  [ERROR] Engine on GPU {gpu_id} crashed "
                    f"(exit code: {proc.returncode}). "
                    f"Killing worker for GPU {gpu_id}..."
                )
                worker = self.worker_procs.get(gpu_id)
                if worker and worker.poll() is None:
                    self._kill_proc(worker)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self, total_files: int) -> None:
        total_completed, total_failed = self._aggregate_progress()
        elapsed = int(time.time() - self._start_time)
        mins, secs = divmod(elapsed, 60)

        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")

        for gpu_id in sorted(self.progress_files.keys()):
            prog = read_progress(self.progress_files[gpu_id])
            if prog:
                print(
                    f"  GPU {gpu_id}: "
                    f"{prog['completed']}/{prog['total']} done, "
                    f"{prog['failed']} failed  [{prog['status']}]"
                )

        print(
            f"\n  Total:   {total_completed}/{total_files} completed, "
            f"{total_failed} failed"
        )
        print(f"  Time:    {mins}m {secs}s")
        print(f"  Output:  {self.args.output}")
        print(f"  Logs:    {self.log_dir}")

        if total_failed > 0:
            self._report_failures()

    def _report_failures(self) -> None:
        all_failed: List[Dict] = []
        for gpu_id in self.progress_files:
            fp = self.progress_files[gpu_id].replace(
                ".json", "_failed.json"
            )
            if os.path.exists(fp):
                try:
                    with open(fp) as f:
                        all_failed.extend(json.load(f))
                except Exception:
                    pass
        if all_failed:
            summary = self.log_dir / "failed_files.json"
            with open(summary, "w") as f:
                json.dump(all_failed, f, ensure_ascii=False, indent=2)
            print(f"\n  Failed files: {summary}")
            launch = os.path.join(_PACKAGE_DIR, "launch.py")
            print(
                "  Re-run with just the failed files:\n"
                f"    python {launch} "
                f"-i <failed_dir> -o {self.args.output}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate_progress(self) -> Tuple[int, int]:
        total_completed = 0
        total_failed = 0
        for gpu_id in self.progress_files:
            prog = read_progress(self.progress_files[gpu_id])
            if prog:
                total_completed += prog["completed"]
                total_failed += prog["failed"]
        return total_completed, total_failed

    def _kill_proc(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
        except (ProcessLookupError, OSError, PermissionError):
            pass

    def _cleanup(self) -> None:
        _print_err("\n[INFO] Cleaning up subprocesses...")

        for gpu_id, proc in self.worker_procs.items():
            self._kill_proc(proc)

        for gpu_id, proc in self.engine_procs.items():
            self._kill_proc(proc)

        for fh in self.file_handles:
            try:
                fh.close()
            except Exception:
                pass

        _print_err("[INFO] All processes stopped.")

    def _force_kill_all(self) -> None:
        for proc in list(self.worker_procs.values()) + list(
            self.engine_procs.values()
        ):
            try:
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

#!/usr/bin/env python3
"""Mac integration: pause after first committed chunk, resume without repeating it.

Runs only when explicitly invoked; uses an isolated hermes-test directory.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "hermes-test" / "pause-integration"


def profile(reserve):
    (WORK / "profile.env").write_text(f"PROFILE=auto\nSERVER_SEQS=4\nSDK_WORKERS=2\nPAGE_QUEUE=2\nREGION_QUEUE=32\nTOTAL_MEMORY_GB=16\nRESERVED_MEMORY_GB={reserve}\n")


def main():
    WORK.mkdir(parents=True, exist_ok=False)
    profile(3)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", GLMOCR_PROFILE_FILE=str(WORK / "profile.env"))
    command = [sys.executable, str(ROOT / "manager/job_runner.py"), "--input", str(ROOT / "hermes-test/resonators.pdf"),
               "--output", str(WORK / "result"), "--enhance", "off", "--no-exports", "--chunk-size", "1"]
    first_log = WORK / "first.log"
    with first_log.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        try:
            deadline = time.monotonic() + 120
            gated = False
            while time.monotonic() < deadline:
                text = first_log.read_text()
                if '"type": "chunk_started"' in text and not gated:
                    # First chunk is already admitted; hold the next one.
                    profile(100)
                    gated = True
                if '"type": "memory_wait"' in text:
                    break
                if process.poll() is not None:
                    raise AssertionError(f"Exited before pause: {process.returncode}\n{text}")
                time.sleep(0.05)
            else:
                raise TimeoutError("No inter-chunk memory wait")
            started = time.monotonic()
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=5) == 130
            latency = time.monotonic() - started
            assert latency < 1.0, latency
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    state_path = WORK / "result/job-state.json"
    paused = json.loads(state_path.read_text())
    assert paused["status"] == "paused", paused
    assert [chunk["status"] for chunk in paused["chunks"]] == ["done", "pending"], paused
    completed_at = paused["chunks"][0]["completed_at"]
    print(f"PAUSED_AFTER_FIRST_CHUNK latency={latency:.3f}s", flush=True)
    profile(3)
    with (WORK / "resume.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env, check=True, timeout=120)
    final = json.loads(state_path.read_text())
    assert final["status"] == "complete", final
    assert final["chunks"][0]["completed_at"] == completed_at
    events = [json.loads(line[len("GLMOCR_EVENT "):]) for line in (WORK / "resume.log").read_text().splitlines() if line.startswith("GLMOCR_EVENT ")]
    chunks = [event["index"] for event in events if event["type"] == "chunk_started"]
    assert chunks == [2], chunks
    assert final["profile"]["reserved"] == 3
    print("RESUMED_ONLY_SECOND_CHUNK OK", flush=True)


if __name__ == "__main__":
    main()

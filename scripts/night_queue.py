#!/usr/bin/env python3
"""Owner-authorized, sequential, disk-guarded overnight OCR with pilot gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from queue_inputs import queue_source

GIB = 1024 ** 3
MIN_FREE = 15 * GIB
STOP = False
STALL_SECONDS = 20 * 60      # watchdog: no page progress for this long ...
IDLE_CPU = 5.0               # ... while the model server stays below this CPU % => stalled


def read_group_pages(root, default=8):
    """Pages per glmocr process (owner-approved 2026-10-23). Read live from <root>/group-pages; write 1 to revert."""
    try:
        return min(32, max(1, int((root / 'group-pages').read_text().strip())))
    except (OSError, ValueError):
        return default


def last_pages_done(log_path):
    """(progress marker, pages_done, stage) from the job log. The marker is the absolute offset of the
    newest non-heartbeat GLMOCR event, so any real event (page done, enhancement/optimize/export
    progress) counts as progress; heartbeats do not."""
    try:
        size = log_path.stat().st_size
        with log_path.open('rb') as f:
            start = max(0, size - 262144); f.seek(start); tail = f.read()
    except OSError:
        return None, None, None
    marker = pages = stage = None; pos = len(tail)
    for raw in reversed(tail.split(b'\n')):
        pos -= len(raw) + 1
        line = raw.decode(errors='replace')
        if not line.startswith('GLMOCR_EVENT '):
            continue
        try: ev = json.loads(line[13:])
        except ValueError: continue
        if stage is None and ev.get('type') == 'heartbeat':
            stage = ev.get('stage'); pages = ev.get('pages_done')
        if marker is None and ev.get('type') != 'heartbeat':
            marker = start + pos
        if marker is not None and stage is not None:
            break
    return marker, pages, stage


def model_server_cpu():
    """CPU % of the mlx_vlm.server process(es); None if not found."""
    try:
        out = subprocess.run(['/bin/ps', '-Ao', 'pcpu=,command='], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    vals = [float(l.split(None, 1)[0]) for l in out.splitlines() if 'mlx_vlm.server' in l]
    return sum(vals) if vals else None


def kill_job_tree(child, output):
    """Stop the job runner and any glmocr parse working on this output dir."""
    try: child.send_signal(signal.SIGTERM); child.wait(timeout=30)
    except subprocess.TimeoutExpired: child.kill(); child.wait()
    out = subprocess.run(['/bin/ps', '-Ao', 'pid=,command='], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(' ')
        if 'glmocr' in cmd and str(output) + '/' in cmd and int(pid) != os.getpid():
            try: os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError: pass


def atomic(path, value):
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(part, path)


def ac_power():
    return 'AC Power' in subprocess.check_output(['pmset', '-g', 'batt'], text=True)


def request_stop(*args):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--install', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(); install = Path(args.install).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'supervisor.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('Supervisor lock exists; inspect its PID before resuming')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-dimsu', '-w', str(os.getpid())])
    manifest = json.loads((root / 'manifest.json').read_text())
    archive = Path.home() / 'Downloads/Local Send/glm_ocr_batch.zip'
    state_path = root / 'night-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'started_at': time.time(), 'pilot_deadline': time.time() + 600,
        'phase': 'pilot', 'completed': [], 'errors': [], 'total': len(manifest)}
    cache = root / 'input-cache'; cache.mkdir(exist_ok=True)
    results = root / 'results'; results.mkdir(exist_ok=True)
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    child = None
    try:
        with queue_source(archive, cache, root / 'inputs-ready.json') as z:
            entries = {x.filename: x for x in z.infolist() if not x.is_dir()}
            assert len(entries) == len(manifest), 'Archive/manifest count mismatch'
            assert set(entries) == {x['ocr_pdf_filename'] for x in manifest}, 'Archive/manifest names mismatch'
            # Pilot starts with an already representative two-page Russian
            # equations/figures document, then other short engineering files.
            pilot = [x for x in manifest if x['sha256'].startswith('f6d95dcf')]
            pilot += [x for x in manifest if x not in pilot and 2 <= (x.get('page_count') or 0) <= 6
                      and x.get('discipline') == 'РЭС (Радиоэлектронные системы)'][:8]
            order = pilot + [x for x in manifest if x not in pilot]
            prio_path = root / 'priority.json'  # optional, owner-approved reprioritization (2026-10-23)
            if prio_path.exists():
                prio = json.loads(prio_path.read_text())
                skip = set(prio.get('skip', []))
                rank = {h: i for i, h in enumerate(prio.get('order', []))}
                order = sorted((x for x in order if x['sha256'] not in skip),
                               key=lambda x: rank.get(x['sha256'], len(rank)))
            stalls = state.setdefault('stalls', {})  # watchdog: id -> number of stalls
            order = [x for x in order if not stalls.get(x['sha256'])] + [x for x in order if stalls.get(x['sha256']) == 1]
            for item in order:
                name = item['ocr_pdf_filename']; ident = item['sha256']
                if ident in state['completed']:
                    continue
                # Cloud/Mac split (2026-10-23, Grok Bot): the first side to create claims/<sha256> (O_EXCL) owns the doc.
                claim = root / 'claims' / ident
                claim.parent.mkdir(exist_ok=True)
                try:
                    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, b'mac'); os.close(fd)
                except FileExistsError:
                    if claim.read_text().strip() != 'mac':
                        continue  # claimed by the cloud worker on the Hermes server
                if Path(name).name != name or not name.lower().endswith('.pdf'):
                    raise ValueError('Unsafe archive member name: ' + name)
                while state['phase'] == 'await_review' and not STOP:
                    state['updated_at'] = time.time(); atomic(state_path, state)
                    if (root / 'continue-approved').exists():
                        state['phase'] = 'full'; atomic(state_path, state); break
                    time.sleep(5)
                if STOP: break
                if state['phase'] == 'pilot' and time.time() >= state['pilot_deadline']:
                    state['phase'] = 'await_review'; atomic(state_path, state)
                    while not STOP and not (root / 'continue-approved').exists():time.sleep(5)
                    if STOP:break
                    state['phase'] = 'full'
                free = shutil.disk_usage(root).free
                if free < MIN_FREE + entries[name].file_size * 2 or not ac_power():
                    state['phase'] = 'safety_paused'; state['reason'] = 'Insufficient free space or AC power disconnected'; break
                source = cache / name
                if not source.exists():
                    part = source.with_suffix('.extracting')
                    with z.open(name) as src, part.open('wb') as dst:
                        while block := src.read(1024 * 1024):
                            if shutil.disk_usage(root).free < MIN_FREE:
                                raise OSError('Disk reserve reached during extraction')
                            dst.write(block)
                    os.replace(part, source)
                output = results / ident[:12]
                state.update(current=ident, current_name=name, current_output=str(output), free_bytes=free, updated_at=time.time())
                atomic(state_path, state)
                cmd = [str(install / '.venv-sdk/bin/python'), str(install / 'manager/job_runner.py'),
                       '--input', str(source), '--output', str(output), '--enhance', 'hybrid', '--chunk-size', '1',
                       '--export-formats', 'md,split_md,searchable_pdf']
                env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           ORT_DISABLE_TELEMETRY='1',  # onnxruntime 1.31 1DS telemetry spins at process exit on macOS
                           GLMOCR_GROUP_PAGES=str(read_group_pages(root)))  # chunks stay 1 page; N pages share one glmocr process
                with (logs / (ident[:12] + '.log')).open('a') as log:
                    child = subprocess.Popen(cmd, cwd=install, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid; atomic(state_path, state)
                    stalled = False; log_path = logs / (ident[:12] + '.log')
                    seen_marker = last_pages_done(log_path)[0]; seen_pages = None; progress_at = time.time(); busy_at = time.time(); next_check = 0
                    while child.poll() is None:
                        if time.time() >= next_check:
                            next_check = time.time() + 60
                            marker, seen_pages, stage = last_pages_done(log_path)
                            if marker != seen_marker or stage != 'ocr':  # only the OCR stage is watched
                                seen_marker = marker; progress_at = time.time()
                            cpu = model_server_cpu()
                            if cpu is None or cpu >= IDLE_CPU:
                                busy_at = time.time()
                            if time.time() - progress_at >= STALL_SECONDS and time.time() - busy_at >= STALL_SECONDS:
                                kill_job_tree(child, output)
                                n = stalls.get(ident, 0) + 1; stalls[ident] = n
                                state.setdefault('stall_log', []).append({'id': ident, 'name': name, 'pages_done': seen_pages, 'n': n, 'time': time.time()})
                                if n == 1:
                                    order.append(item)  # retry once at the end of the queue (resumes from its partial chunks)
                                state['reason'] = 'Watchdog: %s stalled at page %s (attempt %d)' % (name, seen_pages, n)
                                atomic(state_path, state); stalled = True
                                break
                        free = shutil.disk_usage(root).free
                        state.update(updated_at=time.time(), free_bytes=free)
                        pilot_done = state['phase'] == 'pilot' and time.time() >= state['pilot_deadline']
                        safety = free < MIN_FREE or not ac_power()
                        if STOP or pilot_done or safety:
                            child.send_signal(signal.SIGTERM)
                            try:child.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                child.kill(); child.wait()
                            state['phase'] = 'stopped' if STOP else ('safety_paused' if safety else 'await_review')
                            state['reason'] = 'Requested stop' if STOP else ('Low disk or AC disconnected' if safety else 'Ten-minute pilot awaiting review')
                            break
                        atomic(state_path, state)
                        time.sleep(5)
                    code = child.wait(); child = None
                state.pop('child_pid', None)
                if stalled:
                    state['updated_at'] = time.time(); atomic(state_path, state)
                    continue
                job_path = output / 'job-state.json'
                job = json.loads(job_path.read_text()) if job_path.exists() else {}
                if code == 0 and job.get('status') == 'complete':
                    state['completed'].append(ident)
                    state['last_completed_output'] = str(output)
                    state['last_completed_name'] = name
                    source.unlink(missing_ok=True)  # Only our extracted copy; original ZIP is retained.
                    state['last_completion_at'] = time.time()
                elif state['phase'] not in {'await_review', 'safety_paused', 'stopped'}:
                    state['errors'].append({'id': ident, 'code': code, 'error': job.get('error'), 'time': time.time()})
                    state['phase'] = 'error'; state['reason'] = 'OCR failed; paused rather than silently skipping'; atomic(state_path, state); break
                state['updated_at'] = time.time(); atomic(state_path, state)
                if state['phase'] in {'safety_paused', 'stopped'}: break
                if state['phase'] == 'await_review':
                    # Resume the SAME partially committed job, not the next one.
                    state['resume_required'] = ident
                    atomic(state_path, state)
                    break
            else:
                state['phase'] = 'complete'
    except Exception as exc:
        state['phase'] = 'error'; state['reason'] = str(exc)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
        state['updated_at'] = time.time(); atomic(state_path, state)
        caffeinate.terminate()
        lock.unlink(missing_ok=True)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == '__main__':
    main()

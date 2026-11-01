"""Agent M8b/M8c hook: replace placeholder block with real worker dispatch.

PATCH for control_service/agent.py — replace the placeholder worker section:

    # M8a: placeholder worker — report state, do NOT load models.
    ...
with:
    dispatch_review(args.sock, m, review_dir)

Keep the file below as control_service/worker_dispatch.py and import it.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import os

log = logging.getLogger("agent.dispatch")

WORKERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers")


def dispatch_review(sock: str, meta: dict, review_dir: str) -> None:
    """Run real worker subprocess for one review dir (blocking, one at a time)."""
    from scheduler import report_worker
    import json

    stamp = meta.get("_stamp", "?")
    rdir = meta.get("_dir", os.path.join(review_dir, stamp))
    model = meta.get("active_model", "")
    worker = "nanoowl" if "nanoowl" in model else "qwen"
    script = os.path.join(WORKERS_DIR,
                           "nanoowl_worker.py" if worker == "nanoowl" else "qwen_worker.py")
    try:
        report_worker(sock, f"{worker}-working")
        log.info("WORKER %s start %s", worker, stamp)
        r = subprocess.run([sys.executable, script, "--review", rdir],
                           capture_output=True, text=True, timeout=600)
        log.info("WORKER %s rc=%d %s", worker, r.returncode,
                 (r.stdout + r.stderr)[-300:])
        # summarize for web UI (<=120 chars)
        try:
            res = json.load(open(os.path.join(rdir, "result.json")))
            if worker == "qwen":
                summary = f"qwen done:{res.get('qwen_text', '')[:60]}"
            else:
                n = len(res.get("owl_boxes", []))
                summary = f"owl done:{n} boxes {res.get('lat_ms', 0):.0f}ms"
        except Exception:
            summary = f"{worker} done"
        report_worker(sock, f"done:{summary}"[:120])
        # back to idle after a beat so the UI shows the result
        import time
        time.sleep(1)
        report_worker(sock, "idle")
    except subprocess.TimeoutExpired:
        log.warning("WORKER %s TIMEOUT %s", worker, stamp)
        try:
            report_worker(sock, "idle")
        except Exception:
            pass
    except Exception as exc:
        log.error("WORKER dispatch failed: %s", exc)

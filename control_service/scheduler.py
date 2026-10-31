"""M8a scheduler: YOLO everyday, review queue + worker escalation.

Iron rule: scheduler NEVER touches the inference pipeline. It only:
  1. polls edgevisiond Unix socket (STATUS / RULE / WORKER),
  2. scans pending_review/ for new meta.json (daemon-written, atomic),
  3. reports worker state back via WORKER <state> (shows on web UI),
  4. appends escalation notes (worker writes result.json itself).

Real model workers live in control_service/workers/ (per-worker subprocess).
"""
from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass


@dataclass
class RuleState:
    active_model: str = "yolo"
    reason: str = "default: yolo everyday"
    low_conf_streak: int = 0
    low_conf_frames: int = 0
    last_max_conf: float = 0.0
    reviews_written: int = 0
    review_dir: str = "./pending_review"


def decide(low_conf_streak: int, dets_empty: bool) -> RuleState:
    """Pure rule function — unit-testable without hardware.

    Mirrors the daemon-side hint logic so off-board review matches on-board.
    """
    if low_conf_streak >= 3 and dets_empty:
        return RuleState("yolo+vlm?", "empty streak>=3: needs VLM review",
                         low_conf_streak, 0, 0.0)
    if low_conf_streak >= 3:
        return RuleState("yolo+nanoowl?",
                         "low-conf streak>=3: candidate for NanoOWL 2nd-stage",
                         low_conf_streak, 0, 0.0)
    return RuleState()


def query_daemon(sock_path: str = "/tmp/edgevisiond.sock", cmd: str = "RULE",
                 timeout: float = 5.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall((cmd + "\n").encode())
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in chunk:
                break
        return json.loads(buf.decode().strip())
    finally:
        s.close()


def report_worker(sock_path: str, state: str, timeout: float = 5.0) -> dict:
    """Report worker progress to daemon (shows on web UI 2nd row)."""
    return query_daemon(sock_path, f"WORKER {state}"[:160], timeout)


def poll_once(sock_path: str = "/tmp/edgevisiond.sock") -> RuleState:
    """Ask the daemon for its rule view; fall back to local decide() on STATUS."""
    try:
        r = query_daemon(sock_path, "RULE")
        return RuleState(
            active_model=r.get("active_model", "yolo"),
            reason=r.get("reason", "?"),
            low_conf_streak=int(r.get("low_conf_streak", 0)),
            low_conf_frames=int(r.get("low_conf_frames", 0)),
            last_max_conf=float(r.get("last_max_conf", 0.0)),
            reviews_written=int(r.get("reviews_written", 0)),
            review_dir=r.get("review_dir", "./pending_review"),
        )
    except Exception as exc:  # daemon down -> report, never crash caller
        return RuleState("unknown", f"scheduler: daemon unreachable ({exc})")


def scan_reviews(review_dir: str) -> list[dict]:
    """List review dirs that have final meta.json but no result.json yet.

    Daemon writes meta.json atomically (.tmp->rename), so presence of
    meta.json (not .tmp) means the crop is complete and safe to read.
    """
    out: list[dict] = []
    try:
        names = sorted(os.listdir(review_dir))
    except OSError:
        return out
    for name in names:
        d = os.path.join(review_dir, name)
        meta = os.path.join(d, "meta.json")
        result = os.path.join(d, "result.json")
        if os.path.isfile(meta) and not os.path.isfile(result):
            try:
                with open(meta) as f:
                    m = json.load(f)
                m["_stamp"] = name
                m["_dir"] = d
                out.append(m)
            except (OSError, ValueError):
                continue
    return out


def scan_results(review_dir: str) -> list[dict]:
    """List review dirs that have both meta.json and result.json."""
    out: list[dict] = []
    try:
        names = sorted(os.listdir(review_dir))
    except OSError:
        return out
    for name in names:
        d = os.path.join(review_dir, name)
        meta = os.path.join(d, "meta.json")
        result = os.path.join(d, "result.json")
        if os.path.isfile(meta) and os.path.isfile(result):
            try:
                with open(result) as f:
                    r = json.load(f)
                r["_stamp"] = name
                r["_dir"] = d
                out.append(r)
            except (OSError, ValueError):
                continue
    return out

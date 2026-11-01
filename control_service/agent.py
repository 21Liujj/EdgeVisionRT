"""M8a agent loop: poll daemon RULE, scan pending_review/, drive worker placeholder.

Usage (on board):
  python3 control_service/agent.py --sock /tmp/edgevisiond.sock --interval 2
  python3 control_service/agent.py --once   # single pass (smoke test)
"""
from __future__ import annotations

import argparse
import logging
import time

from scheduler import decide, poll_once, report_worker, scan_results, scan_reviews

# M12 logging: timestamps + level on every line so journald/docker logs are
# greppable without SSH. Single format across agent + workers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("agent")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/edgevisiond.sock")
    ap.add_argument("--review-dir", default=None,
                    help="override review dir (default: from daemon RULE)")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    last_model = ""
    seen_reviews: set[str] = set()
    while True:
        st = poll_once(args.sock)
        review_dir = args.review_dir or st.review_dir
        if st.active_model != last_model:
            log.info("model=%s reason=%s streak=%d max_conf=%.2f reviews=%d",
                     st.active_model, st.reason, st.low_conf_streak,
                     st.last_max_conf, st.reviews_written)
            last_model = st.active_model

        # cross-check pure rule fn stays in sync with daemon hint
        _ = decide(st.low_conf_streak, dets_empty=("vlm" in st.active_model))

        # scan for new review dirs (daemon edge-triggered dumps)
        try:
            pending = scan_reviews(review_dir)
            done = scan_results(review_dir)
        except Exception as exc:
            log.warning("scan error: %s", exc)
            pending, done = [], []

        for m in pending:
            stamp = m.get("_stamp", "?")
            if stamp not in seen_reviews:
                seen_reviews.add(stamp)
                log.info("NEW REVIEW %s trigger=%s model=%s boxes=%d crop=%s",
                         stamp, m.get("trigger"), m.get("active_model"),
                         len(m.get("yolo_boxes", [])), m.get("crop"))
                # M8b/M8c: real worker subprocess (one at a time, blocking).
                try:
                    from worker_dispatch import dispatch_review
                    dispatch_review(args.sock, m, review_dir)
                except Exception as exc:
                    log.error("worker dispatch failed: %s", exc)

        if done and args.once:
            log.info("done results: %d", len(done))
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

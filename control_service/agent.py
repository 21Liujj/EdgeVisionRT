"""M8a agent loop: poll daemon RULE, scan pending_review/, drive worker placeholder.

Usage (on board):
  python3 control_service/agent.py --sock /tmp/edgevisiond.sock --interval 2
  python3 control_service/agent.py --once   # single pass (smoke test)
"""
from __future__ import annotations

import argparse
import time

from scheduler import decide, poll_once, report_worker, scan_results, scan_reviews


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
            print(f"[agent] model={st.active_model} reason={st.reason} "
                  f"streak={st.low_conf_streak} max_conf={st.last_max_conf:.2f} "
                  f"reviews={st.reviews_written}", flush=True)
            last_model = st.active_model

        # cross-check pure rule fn stays in sync with daemon hint
        _ = decide(st.low_conf_streak, dets_empty=("vlm" in st.active_model))

        # scan for new review dirs (daemon edge-triggered dumps)
        try:
            pending = scan_reviews(review_dir)
            done = scan_results(review_dir)
        except Exception as exc:
            print(f"[agent] scan error: {exc}", flush=True)
            pending, done = [], []

        for m in pending:
            stamp = m.get("_stamp", "?")
            if stamp not in seen_reviews:
                seen_reviews.add(stamp)
                print(f"[agent] NEW REVIEW {stamp} trigger={m.get('trigger')} "
                      f"model={m.get('active_model')} "
                      f"boxes={len(m.get('yolo_boxes', []))} "
                      f"crop={m.get('crop')}", flush=True)
                # M8b/M8c: real worker subprocess (one at a time, blocking).
                try:
                    from worker_dispatch import dispatch_review
                    dispatch_review(args.sock, m, review_dir)
                except Exception as exc:
                    print(f"[agent] worker dispatch failed: {exc}", flush=True)

        if done and args.once:
            print(f"[agent] done results: {len(done)}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

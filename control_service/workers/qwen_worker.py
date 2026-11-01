"""M8c Qwen worker: pending_review/crop.jpg -> Qwen2/3-VL single question -> result.json.

Calls the prebuilt demo binary (no model code copied here):
  edgeqwen_runtime_demo --engines-dir ... --visual-engines-dir ... --image ... --prompt ...

Requires EDGELLM_PLUGIN_PATH=.../libNvInfer_edgellm_plugin.so (see m8 plan §probe).
Single-review CLI. Exits 0 on attempt (result.json always written).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("worker.qwen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import QWEN_ASK as DEMO
from paths import QWEN_PLUGIN as PLUGIN
# engines baked into qwen_ask (llm + visual dirs); no --engines-dir flags.

FIXED_QUESTIONS = [
    "Describe the cropped object in one sentence. Is it abnormal?",
    "What is in this cropped image? Answer in one short sentence.",
]


def run(cmd: list[str], env: dict, timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--question", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    review = args.review
    try:
        meta = json.load(open(os.path.join(review, "meta.json")))
    except Exception as exc:
        log.error("bad meta: %s", exc)
        return 2
    crop = os.path.join(review, meta.get("crop", "crop.jpg"))
    question = args.question or meta.get("prompt_qwen") or FIXED_QUESTIONS[0]

    env = dict(os.environ)
    env["EDGELLM_PLUGIN_PATH"] = PLUGIN
    t0 = time.perf_counter()
    rc, out = run([DEMO, "--image", crop, "--prompt", question,
                   "--max-tokens", "64"], env, args.timeout)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    # demo prints answer text + a trailing {...profile json...}; keep last profile line
    text, profile = "", {}
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and "vision_mean_ms" in line:
            try:
                profile = json.loads(line)
            except ValueError:
                pass
            continue
        if line and not text and not line.startswith("["):
            text = line[:500]
            break
    if not text:
        # fallback: last non-empty non-log line
        for line in reversed(out.strip().splitlines()):
            s = line.strip()
            if s and not s.startswith("[") and not s.startswith("{"):
                text = s[:500]
                break

    result = {
        "worker": "qwen",
        "ok": bool(text) and rc == 0,
        "lat_ms": round(lat_ms, 2),
        "vision_ms": profile.get("vision_mean_ms", 0),
        "prefill_ms": profile.get("prefill_mean_ms", 0),
        "decode_ms": profile.get("decode_mean_ms", 0),
        "tok_per_s": profile.get("gen_tokens_per_second", 0),
        "peak_mem_mb": profile.get("peak_unified_memory_mb", 0),
        "qwen_text": text,
        "question": question,
        "error": "" if text else f"rc={rc} no text",
    }
    tmp = os.path.join(review, "result.json.tmp")
    fin = os.path.join(review, "result.json")
    json.dump(result, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.rename(tmp, fin)
    log.info("done ok=%s lat_ms=%.1f text=%s", result["ok"], result["lat_ms"],
             text[:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

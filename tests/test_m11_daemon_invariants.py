"""M11 host-side invariant test (no hardware, no CUDA).

Validates the daemon refactor in apps/edgevisiond/main.cc by static inspection:
  1. No std::system() shell calls remain (use std::filesystem instead).
  2. update_snapshot uses sx/sy scaling (640px model coords -> 480px snap).
  3. Telemetry bounded: latencies cap 600, events cap 200, trend cap 120.
  4. /proc,/sys reads never run under mu (sampled OUTSIDE mu comment present).
  5. trend_add_locked takes pre-sampled cpu/gpu/mem (no file I/O under lock).
"""
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MAIN = os.path.join(ROOT, "apps", "edgevisiond", "main.cc")

fails = 0

def check(name, cond, detail=""):
    global fails
    print(("PASS" if cond else "FAIL"), name, detail)
    fails += (not cond)

try:
    with open(MAIN, encoding="utf-8", errors="replace") as f:
        src = f.read()
except OSError as e:
    print("FAIL open main.cc", e)
    sys.exit(1)

# 1) no shell
check("no std::system(", "std::system(" not in src)
check("uses std::filesystem", "std::filesystem" in src)
check("no mkdir -p shell string", "mkdir -p" not in src)

# 2) coordinate mapping: sx/sy scaling present, old bug pattern gone
check("update_snapshot has sx/sy scaling",
      "static_cast<float>(dst_w) / target_w" in src and
      "static_cast<float>(dst_h) / target_h" in src)
check("old bug d.x*dst_w gone", "d.x*dst_w" not in src)
check("snapshot scales by sx/sy", "float cx = d.x * sx" in src and "float bw = d.w * sx" in src)

# 3) bounded telemetry
check("latencies capped at 600", re.search(r"v\.size\(\) > 600", src) is not None)
check("events capped at 200", "events.size() > 200" in src)
check("trend capped at 120", "trend.size() > 120" in src)
check("trend_add_locked signature has cpu/gpu",
      re.search(r"trend_add_locked\(.*cpu.*gpu", src) is not None)

# 4) lock discipline: sys sampling outside mu
check("STATUS samples OUTSIDE mu",
      "Sample sys files OUTSIDE mu" in src)
check("Mu guards counters only (do_start comment)",
      "mu guards state/counters only" in src or "mu guards" in src.lower() or
      "Blocking I/O (mkdir" in src)
# ensure read_mem/read_gpu_load/cpu sample appear BEFORE first lock in STATUS block
m = re.search(r'if \(line == "STATUS"\) \{(.*?)\n    \}\n    if \(line == "META"\)', src, re.S)
if m:
    block = m.group(1)
    i_read = block.find("read_mem()")
    i_lock = block.find("lock_guard<std::mutex> lk(mu)")
    check("STATUS: read_mem before first mu lock", 0 <= i_read < i_lock,
          f"(read@{i_read} lock@{i_lock})")
else:
    check("STATUS block found", False)

# 5) pipeline sys cache exists (~1Hz, no lock)
check("pipeline ~1Hz sys cache present", "last_sys_sample_mono" in src)

# 6) snapshot 503 path carries CORS (file:// demo uses fetch+blob)
m503 = re.search(r"503 Busy.*?Access-Control-Allow-Origin", src, re.S)
check("snapshot 503 has CORS", m503 is not None)

print("INVARIANTS_TEST", "PASS" if fails == 0 else f"FAIL({fails})")
sys.exit(1 if fails else 0)

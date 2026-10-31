"""Host-side unit test for the M7 rule function (no hardware needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "control_service"))
from scheduler import decide

cases = [
    # (streak, empty, expect_model)
    (0, False, "yolo"),
    (2, False, "yolo"),
    (3, False, "yolo+nanoowl?"),
    (5, False, "yolo+nanoowl?"),
    (3, True, "yolo+vlm?"),
    (10, True, "yolo+vlm?"),
]
fails = 0
for streak, empty, want in cases:
    got = decide(streak, empty).active_model
    ok = got == want
    fails += (not ok)
    print(("PASS" if ok else "FAIL"), f"streak={streak} empty={empty} -> {got} (want {want})")
print("RULE_TEST", "PASS" if fails == 0 else f"FAIL({fails})")
sys.exit(1 if fails else 0)

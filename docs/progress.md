# EdgeVisionRT — progress (single source of truth)

> Rule: code and reports can lie stale; this file is the only place that says
> "what stage are we at, what evidence exists, what is next".

## Current: M12 logging hardening — retention + crash-safety + disk self-inspection (done, host-verified 2026-11-01)

| Item | State | Evidence |
|---|---|---|
| `sessions/` retention (`sessions_keep=20`, pruned on START outside `mu`) | done (host invariant) | `tests/test_m11_daemon_invariants.py` 28/28 pass; `prune_oldest(sessions_dir, sessions_keep)` |
| `events.jsonl` crash-safety (`fflush` every ~30 frames + on review) | done (host invariant) | `events_unflushed` counter; `fflush(events_fp)` outside `mu` |
| `summary.json` durability (`fflush`+`fsync` file + `fsync` dir) | done (host invariant) | `fsync(sfd)` before rename, `fsync(dfd)` after renames, best-effort |
| `DISK` self-inspection (`/api/disk`: writable/probe/fs/sessions/reviews) | done (host invariant) | `handle_cmd("DISK")` all-I/O-outside-`mu`; console `r_disk/r_sessn/r_revn` panel |
| Python logging with timestamps (`agent`+`dispatch`+`owl`+`qwen`) | done (host py_compile) | `logging.basicConfig` single format `%(asctime)s [%(levelname)s]` |
| Host regression (m7 rule + m8 contract + m11/m12 invariants) | done (Windows host) | `test_m7_rule.py` 6/6, `test_m8_contract.py` 23/23, `test_m11_daemon_invariants.py` 28/28 |
| Board verification (build M12 + START/STOP + /api/disk + STOP durability) | TODO (board) | needs `cmake --preset board-native`, START/STOP smoke, `/api/disk` check |

## Previous: M11 daemon hardening — lock discipline + bounded telemetry (done, host-verified 2026-11-01)

| Item | State | Evidence |
|---|---|---|
| `update_snapshot` 640px→480px sx/sy fix | done (host invariant) | `tests/test_m11_daemon_invariants.py` 14/14 pass; `apps/edgevisiond/main.cc:964` |
| `do_start` blocking I/O out of `mu` (mkdir/fopen/sess.start/join) | done (host invariant) | `do_start` lock only guards state/counters; `STATUS`/`METRICS` sample `/proc`,`/sys` OUTSIDE `mu` then snapshot |
| Bounded telemetry (lat 600 / events 200 / trend 120) | done (host invariant) | `push_capped`/`trend_add_locked`/`push_event_locked` |
| `std::system()` shell removal → `std::filesystem` | done (host invariant) | no `std::system` in `main.cc`; `create_directories/directory_iterator/remove_all` |
| Host regression (m7 rule + m8 contract + m11 invariants) | done (Windows host) | `test_m7_rule.py` 6/6, `test_m8_contract.py` 23/23, `test_m11_daemon_invariants.py` 14/14 |
| Board verification (build + START/STOP + box overlay + long-run) | TODO (board) | needs `cmake --preset board-native`, START/STOP smoke, snapshot box check |

## Previous: M8 fusion + COCO full eval (done, board-verified 2026-10-31)

| Item | State | Evidence |
|---|---|---|
| Single-repo fusion (vendors/ + config + paths.py + eval/) | done (board-verified) | `docs/m8_fusion_report.md`; `RUN_OWL` resolves to `vendors/nanoowl/run_nanoowl.py` |
| YOLO COCO val2017 full (5000 imgs) | done (board) | AP 0.216 / AP50 0.301, 4.34ms avg (`eval/out_yolo/`) |
| NanoOWL COCO val2017 full (5000 imgs) | done (board) | AP 0.259 / AP50 0.408 (`eval/out_owl/coco_eval_summary.json`) |
| Cup live 3-stage cascade (YOLO→OWL→Qwen) | done (board) | `pending_review/20261031_111708_f12/`; Qwen "Yes, there is a cup..." 15.6s |
| M8d docs | done | `docs/m8_fusion_report.md` |

## Previous: M7 scheduler placeholder (done, board-verified 2026-10-30)

| Item | State | Evidence |
|---|---|---|
| Repo skeleton (CMake/presets/core/queue/camera/probe/tests) | done (host-side) | files under `edgevisionrt/` |
| Host offline build (`host-offline` preset + ctest) | done (Windows MSYS2 ucrt64 g++15.2 + pip cmake4.4.4/ninja1.13.2, 2/2 pass) | `build/host`, `ctest --test-dir build/host` |
| M0 env+camera freeze | done (board) | `reports/environment.json`, `reports/camera_caps.json` |
| M1 V4L2 streaming (`capture_probe --stream`) | done (board) | MJPEG 720p session artifacts |
| M2 TRT deploy (YOLOv8-n FP16) | done (board) | `models/yolov8n_fp16.engine`, `infer_bench` p50 ~4.4ms |
| M3 self CUDA preprocess | done (board) | `src/backends/cuda_preprocess.cu`, M2 latency includes it |
| M4 BSP/power (15W vs MAXN_SUPER + jetson_clocks) | done (board) | `docs/m4_tech_analysis.md`; MAXN locked 4.36ms avg |
| M5 uvcvideo diagnostics (ftrace vb2) | done (board) | `docs/m5_tech_analysis.md`; MJPEG 720p 30fps clean |
| M6 service (systemd + udev + atomic sessions + web UI) | done (board) | `docs/m6_tech_analysis.md`; `edgevision.service` active |
| M7 scheduler placeholder + rule UI | done (board-verified) | `docs/m7_tech_analysis.md`; `RULE`/`api/rule`/yellow bar/agent smoke all pass; stream smoke 416 frames avg 4.36ms |
| M7 rule unit test (host, no hw) | done | `tests/test_m7_rule.py` 6/6 pass |
| 8h long-run stability | DEFERRED | per plan, not started |
| docker image | DEFERRED | per plan, not started |
| NanoOWL/VLM real workers | NEXT | M7 placeholder hooks ready (`pending_review/` convention) |

## Milestone map

- M0 env+camera freeze → M1 V4L2 data plane → M2 TRT deploy (YOLOv8-n FP16) →
  M3 self CUDA preprocess → M4 BSP/power → M5 uvcvideo diagnostics →
  M6 service + 8h stability → M7 selective fusion (NanoOWL/NanoSAM/VLM workers)

M0–M6 done on board; M7 placeholder done (rule plumbing + visualization).
Next: real 2nd-stage worker (NanoOWL VRAM check first) or 8h run.

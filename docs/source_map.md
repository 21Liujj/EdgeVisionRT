# source_map — what lives where, and what is OFF-LIMITS

## This repo (personal, written by me)

| Path | Owns |
|---|---|
| `include/`, `src/` | C++17 core: frame pool, queue, camera backend |
| `cuda/` | self-written preprocess kernels (M3) |
| `backends/` | TRT runtime wrapper (M2), fusion workers (M7) |
| `apps/` | `edgevisiond`, `capture_probe`, `infer_bench`, `edgevisionctl` |
| `control_service/` | Python control plane (no pixel copying) |
| `config/` | schema + board profiles (single config source) |
| `kernel/`, `bsp/`, `deploy/` | uvc patch, DT overlay/rootfs manifest, systemd+udev |
| `models/` | manifests only (hashes, no binaries) |
| `scripts/probe_*` | M0 board/camera evidence collectors |

## Reference repo (read-only): `../realtime_capture_source_full_pack/`

May READ for engineering patterns; must NOT copy code/modules into this repo.
GMSL / PICO / OSS / docker-systemd topology are out of scope and stay there.

Inherited principles only:
1. single config source (`config/device_config.json` → our `config/*.json` + schema)
2. command/status channel split (`/hw_daemon_cmd_mq` + `/hw_daemon_status_mq` → our protocol v3)
3. bounded queues, atomic session write, mqueue depth semantics

## Old personal repos (reference for migration_matrix.md)

- `edge-perception` (813 files): `cpp_runtime/` (trt_engine, preprocessing, .cu kernel),
  `edge_perception/runtime/`, `scripts/bench*`, `data/eval_manifests/`
- `edgeqwen-jetson` (own files only): `scripts/*.sh`, `benchmark/*.sh`, `docs/final/*`

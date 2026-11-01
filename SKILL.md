# SKILL.md — EdgeVisionRT (single-file agent playbook)

> 给 AI / 新人进场用。读完这个 + `AGENTS.md` 就能安全改代码，不用全仓扫描。

## 0. 一句话定位

Jetson Orin Nano Super 上的常驻视觉 daemon：V4L2 采 MJPG → 软解 → CUDA stretch 到 640×640 → TensorRT YOLOv8n-fp16 → NMS → 快照/RTSP/落盘。Python 只做调度和 worker，不碰像素热路径。

## 1. 铁律（违反必出线上事故）

1. `sent != accepted != completed`：`START` 回 `accepted` 必须是驱动真正在流；`STOP` 回 `completed` 必须是已停 **且** summary/events 已 rename 落盘。
2. 预处理是**全帧 stretch、无 letterbox**。`postprocess` 输出是 **640 像素坐标**（不是归一化）。画框/crop/写 COCO 必须先除以 `target_w/target_h`。
3. 热路径（`capture_loop`/`run_one`）永不阻塞：不等待 HTTP、浏览器、网络、日志落盘。队列满就丢并计数（`pool_drop/queue_drop`）。
4. 持锁不做 IO：`mu` 只保护数字和状态。`sess.start`、`write_bgr_jpeg`、`join()`、`/proc`、`/sys` 采样必须放锁外（快照后再锁内拷贝）。`std::system()` 已删，目录/清理走 `std::filesystem`。
5. 耗时用单调钟（`CLOCK_MONOTONIC`/`steady_clock`），墙上时间只给人看。GPU 计时是 `enqueue+sync` 包住的真实完成时间。
6. `.engine` 永不进 git（只走 `models/MANIFEST.md`）。`file_infer` 产物进 `replay_*/`，永不进 `sessions/`。
7. 状态机单写者：`state` 只由 control 线程的 `do_start/do_stop` 改；pipeline 只读 `stop_session`。
8. 跨进程只走换行 JSON 文本（Unix socket `/tmp/edgevisiond.sock`），不传指针、不依赖结构体布局。

## 2. 改代码的标准动作

| 任务 | 必须同时看 |
|---|---|
| 改 start/stop | Unix socket + HTTP `/api/start\|stop` + `do_start/do_stop` + pipeline 退出 + summary 落盘 |
| 改框/坐标 | `postprocess`(640px) → `update_snapshot`(sx/sy 换算到 480px) → `maybe_dump_review`(归一化写 meta) → worker 反算 |
| 改相机 | `v4l2_capture.cc` + `stream_session.cc` + `cuda_preprocess.cu` |
| 加指标 | 说清口径（采集/推理/预览/浏览器 FPS 是四个数）、窗口样本量；缺数填 `--` 不填 0 |
| 加线程 | 先写清：停止信号 → 唤醒阻塞 → join → 释放；禁 detached + 强制 cancel |

## 3. 构建 / 运行 / 验证（板上主路径）

```bash
cmake --preset board-native && cmake --build build/board
./build/board/edgevisiond --engine models/yolov8n_fp16.engine --device /dev/video-edgevision --http 8080 --rtsp 8554
python3 control_service/agent.py --sock /tmp/edgevisiond.sock
# 离线复现（无相机）
./build/board/file_infer --input eval/images --out replay_test/
# 主机单测（无 CUDA）
python3 tests/test_m8_contract.py && python3 tests/test_m7_rule.py && python3 tests/test_m11_daemon_invariants.py
```

## 4. 排障口诀

- 框贴边/挤到角 → 查 `update_snapshot` 的 sx/sy（`d.x*dst_w` 是错的，应为 `d.x*sx`）。
- `STATUS` 卡住 → 查是否持 `mu` 做 `sess.start` / JPEG / `system()`。
- 内存涨 → 查 `latencies` / `events` / `trend` 是否有 cap（600/200/120）。
- 框和图对不上 → 查快照源帧 ID（`snap_src_id` / `frame_no`）是否一起落盘，跨 resize 是否按比例换算。
- 崩溃只留 `.tmp` 是正常的；终态必须是 rename 后的完整文件。

## 5. 进度真相

只信 `docs/progress.md`。`replay_*` 不算板上验收。没实测就标待验证，不编造稳定性和性能数字。

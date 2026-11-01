# AGENTS.md — EdgeVisionRT

> 给后续 AI / 人看的工程契约。目标：不做全量扫描就能安全改代码。
> 优先级：用户当前指令 > 本文件 > 泛化假设。文档过期时以代码为准，但改代码必须同步改这里。

## 1. 先看什么（按需读，不要全扫）

| 任务 | 先看 |
|---|---|
| 运行时问题 / 状态不对 | `apps/edgevisiond/main.cc`（`do_start/do_stop/run_one/update_snapshot/handle_http`），`src/pipeline/stream_session.cc`，`src/camera/v4l2_capture.cc` |
| 推理变慢 / 精度不对 | `src/backends/trt_engine.cc`，`src/backends/cuda_preprocess.cu`，`apps/file_infer/main.cc`（离线复现用它） |
| 调度 / worker 没跑 | `control_service/agent.py`，`control_service/scheduler.py`，`control_service/worker_dispatch.py`，`control_service/paths.py` |
| 部署 / 开机不跑 | `deploy/entrypoint.sh`，`deploy/run_board.sh`，`deploy/Dockerfile`，`deploy/systemd/edgevision.service`，`deploy/udev/99-edgevision-camera.rules` |
| 进度真相 | `docs/progress.md`（唯一真相源），`docs/m8_fusion_report.md` |

## 2. 运行时拓扑（一句话）

```
systemd edgevision.service
 └─ edgevisiond (C++): control线程(Unix socket) + pipeline线程(采集→推理) + http线程(:8080) + rtsp线程(:8554)
 └─ agent.py (Python): 轮询 RULE + 扫 pending_review/ → 起 worker 子进程（串行，用完即退）
```

- 单容器单 CUDA 上下文。8GB 统一内存，一次只活一个模型。
- 摄像头唯一主人：`StreamSession` → `V4L2Capture`。谁都不许另开 `/dev/video*`。
- `edgevisiond` 三种状态外露必须一致：Unix socket `STATUS` / HTTP `/api/status` / journal 心跳。

## 3. 铁律（违反必出线上事故）

1. `sent != accepted != completed`：`START` 回 `accepted` 必须是驱动真正在流；`STOP` 回 `completed` 必须是 session 已停 **且** summary.json + events.jsonl 都 rename 落盘。只看返回值不看状态回流就是 bug。
2. 预处理是 **全帧 stretch 到 640×640，无 letterbox**。`postprocess` 输出是 **640 像素坐标**，不是归一化。任何画框 / crop / 写 COCO 都必须先除以 `target_w/target_h`。反例：`update_snapshot` 曾写成 `d.x*dst_w` 把框挤到边缘。
3. 热路径永不阻塞：pipeline 线程不等待 HTTP / 浏览器 / 日志落盘 / 网络。队列满就丢并计数（`pool_drop/queue_drop`），不许无界增长。
4. 持锁不做 IO：`mu` 只保护数字和状态，不许在持锁时做 `sess.start`、`write_bgr_jpeg`、`system()`、`join()`。`do_start` 的 V4L2 建链必须放锁外。
5. 延迟只用单调钟：`CLOCK_MONOTONIC` 测耗时，`CLOCK_REALTIME` 只给人类看。GPU 计时是 `enqueueV3 + streamSync` 包住的真实完成时间，不是提交时间。
6. 引擎不进 git：`.engine` 只走 `models/MANIFEST.md` + 板上挂载。仓库只放源码和清单。
7. replay 不算验收：`file_infer` 产物必须进 `replay_*/`，永不进 `sessions/`。
8. 状态机单写者：`state` 只由 control 线程的 `do_start/do_stop` 改，pipeline 线程只读 `stop_session`。`STARTING` 期间的 `STOP` 必须明确拒绝或排队，不许两个线程同时改状态。
9. 跨进程不用裸结构体：Unix socket 走换行 JSON 文本，不许传指针和编译器布局。
10. 未测量不许写：`--` 表示未知，不许填 0 冒充。Orin Nano 无 NVENC，文档不许写硬件编码已验证（除非实测过）。

## 4. 关键调用链

### 启动
`main()` → `init_cuda` + `load_engine` → control/http 线程 → `START` → `do_start`（建 `sessions/<stamp>` + `sess.start(MJPG 1280x720)`）→ `pipeline_thread=pipeline_loop`

### 每帧
`StreamSession::capture_loop`（DQBUF→拷pool→try_push，满则 `queue_drop++`）
→ `pipeline_loop consume(lease, 2000ms)` → `run_one`：mjpeg软解 → H2D → `cuda_preprocess` → `engine.infer` → D2H → `postprocess+NMS` → `update_snapshot`（480px+画框+rtsp.push）→ 记 `frames_ok/latencies` → rule streak → 边缘触发 review 落盘

### 停止
`STOP` → `state=STOPPING` → `stop_session=true` → join pipeline → `sess.stop()` → 落盘 summary+events（双 rename）→ `state=IDLE` 回 `completed`

### 升级链
streak==3 → `pending_review/<stamp>/{crop.jpg,meta.json}`（`.tmp→rename` 原子）→ agent 扫到 → `worker_dispatch` 起子进程 → `result.json` → `WORKER done:xxx` 回 daemon（web 可见）

## 5. 资源所有权

| 资源 | 创建/拥有/释放 | 借用者 | 禁忌 |
|---|---|---|---|
| `/dev/video-edgevision` fd + MMAP buf | `V4L2Capture::start/stop/close` | `StreamSession::capture_loop`（仅 read） | 不许第二处 open；拔插后必须 `stop+close` 再重建 |
| FramePool slot | `StreamSession`（pool_） | `FrameLease`（RAII，析构 release） | lease 不许出 pipeline 线程；epoch 对不上必须判无效 |
| `bgr_gpu` / `stream` | `Daemon::init_cuda` / 析构 `cudaFree/Destroy` | `run_one`（当帧借用） | GPU 任务未 sync 前不许复用输入 |
| TRT engine/context/IO | `TrtEngine::load/prepare_io/cleanup` | pipeline 线程 `infer()` | 不许跨线程并发 infer（单 context 单 stream） |
| snapshot `snap_bgr` | `update_snapshot`（`snap_mu` 下 swap） | http/rtsp 线程（拷贝后 encode/push） | http 编 jpeg 不许持 `mu`，只持 `snap_mu` 短拷贝 |
| session 目录/events_fp | `do_start` 建，`do_stop` 关+rename | pipeline 写 events 行 | 崩溃只留 `.tmp`，终态永远是 rename 后的完整文件 |

## 6. 改动 checklist

- 改 start/stop：同时看 Unix socket + HTTP `/api/start|stop` + `do_start/do_stop` + pipeline 退出 + summary 落盘。
- 改相机链路：同时看 `v4l2_capture.cc` + `stream_session.cc` + `cuda_preprocess.cu`。
- 改框/坐标：同时看 `postprocess`（640px）→ `update_snapshot`（sx/sy 换算）→ `maybe_dump_review`（归一化写 meta）→ worker 的反算。改一处必须对齐三处。
- 改预览：同时看 `device_config.json` + `entrypoint.sh` + rtsp 端口映射。
- 加指标：标口径（采集FPS/推理FPS/预览FPS/浏览器FPS 是四个数），标窗口样本量，缺数显 `--` 不许填 0。
- 加线程：先说停信号→唤醒→join→释放顺序，不许 detached + 强制 cancel。

## 7. 构建/运行/调试

```bash
# 板上原生（主路径）
cmake --preset board-native && cmake --build build/board
./build/board/edgevisiond --engine models/yolov8n_fp16.engine --device /dev/video-edgevision --http 8080 --rtsp 8554
python3 control_service/agent.py --sock /tmp/edgevisiond.sock
# Docker（thin runtime，二进制宿主编好灌进去）
docker build -f deploy/Dockerfile -t edgevisionrt:board . && bash deploy/run_board.sh
# 离线复现（无相机）
./build/board/file_infer --input eval/images --out replay_test/
# 主机单测（无 CUDA）
python3 tests/test_m8_contract.py && python3 tests/test_m7_rule.py && python3 tests/test_m11_daemon_invariants.py
```

## 8. 已知锐边（已复验，2026-11-01 修完 M11 首批）

- `update_snapshot` 坐标系（已修：640px→480px 用 sx/sy；主机 `test_m11_daemon_invariants.py` 覆盖，仍需板上看框是否贴合）。
- `do_start` 持 `mu` 做 `sess.start`（已修：建链/建目录/fopen 全在锁外；`STATUS`/`METRICS` 的 `/proc`、`/sys` 采样也在锁外快照）。
- `latencies` 无界增长（已修：`latencies/lat_decode/lat_infer/lat_post` cap 600 + `trend` 120 桶 + `events` 200 条）。
- `std::system("mkdir/ls/rm")` shell 调用（已修：全换成 `std::filesystem`，无 fork、无注入面）。
- `handle_http` 每次全量 JSON + `snapshot.jpg` 轮询，多客户端会放大 CPU → 待限频/ETag（未修，M12）。
- Orin Nano 无 NVENC：`--rtsp-encoder nvv4l2h264` 只是预留，默认 x264，不许在文档写硬件编码已验证。

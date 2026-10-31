# M6: 服务化 (systemd 常驻) — 技术解析文档

## 目标
把 `infer_bench`(跑完即退)升级为常驻 `edgevisiond`,由 systemd 托管:
开机自起、崩溃重启、USB 相机热插拔不丢设备、会话产物原子落盘。
长稳 8h 测试暂缓,本阶段交付服务化本身。

## 架构

```
systemd edgevision.service
  └─ edgevisiond --device /dev/video-edgevision --sock /tmp/edgevisiond.sock
       ├─ control thread: Unix socket (STATUS|START|STOP|QUIT) + 5s heartbeat→journal
       └─ pipeline thread: StreamSession → MJPEG软解 → H2D → CUDA预处理 → TRT → D2H → NMS
                           → sessions/<stamp>/{events.jsonl, summary.json}
```

## 铁律: sent != accepted != completed
- START 回 `accepted:true` 的前提是 `StreamSession::start()` 已返回 true
  (V4L2 真正在流),不是"收到命令"。
- STOP 回 `completed:true` 的前提是 session 已 stop **且**
  `summary.json.tmp → summary.json` + `events.jsonl.tmp → events.jsonl`
  两次 rename 都成功。rename 是原子操作,崩溃只会留下 `.tmp`,不会污染终态。
- STATUS 的 `frames` 是唯一真实进度证明,不是控制面自说自话。

## 新增文件

| 文件 | 作用 |
|------|------|
| `apps/edgevisiond/main.cc` | 常驻 daemon:控制 socket + pipeline 双线程 |
| `deploy/systemd/edgevision.service` | systemd 单元:开机自起,失败 5s 重启,限流 120s/5次 |
| `deploy/udev/99-edgevision-camera.rules` | 按 VID:PID (4a54:5232) 建 `/dev/video-edgevision` 符号链接 |

## 关键设计
- **udev 符号链接**:daemon 永远打开 `/dev/video-edgevision`,USB 重枚举导致
  video0→video1 时不受影响。按 VID:PID 匹配,不按内核序号。
- **WorkingDirectory=/home/nvidia/edgevisionrt**:engine 与 sessions 相对路径
  解析基准,避免 systemd 默认 `/` 下找不到模型。
- **Warmup 帧仍跑推理但不记录**:传感器收敛期(M5:首 3 帧 bytesused 爬坡)
  不污染延迟统计,与 infer_bench 语义一致。
- **退出时收尾**:SIGTERM/SIGINT → 若 session 在 RUNNING,先 do_stop
  (落盘 summary) 再退出,不丢尾部数据。
- **NoNewPrivileges=true**:daemon 只需相机/models/sessions/socket,最小权限。

## 验证结果 (板上实测)
- `systemctl status`:active (running), Main PID 常驻, memory 67.5M
- `/dev/video-edgevision -> video0` 符号链接生效
- 完整一轮: STATUS(IDLE) → START(accepted) → RUNNING(frames 444/15s)
  → STOP(completed) → summary.json 落盘:
  `frames_ok=505, lat p50=4.77ms, p99=4.79ms, dets=491, 0 fail`
- journal 心跳每 5s 一条 state+frames,可直接对接 M7 调度审计
- 服务已 `enable`,重启后自动拉起

## 遗留
- 长稳 8h + 内存泄漏检测:延后,按计划待办
- docker 镜像:下一步(B),板子需先装 docker + nvidia runtime
- control_service Python 调度 + edgevisionctl CLI: M7 范畴,本次只验证了
  socket 协议手工调用

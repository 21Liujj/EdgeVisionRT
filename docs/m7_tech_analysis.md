# M7: 调度占位 + 网页规则可视化 — 技术解析文档

## 目标
在 M6 服务化之上加"看得见的调度"：YOLO 日常常开，难例才升级模型（NanoOWL 二阶段 / VLM 复核）。
本阶段只做**占位**：规则算得出、看得见、进日志，但不真加载第二个模型
（Orin Nano 8GB 装不下 YOLO + NanoOWL + VLM 三引擎常驻）。

## 架构

```
edgevisiond (C++, 单引擎 YOLO)
 ├─ pipeline thread: 推理照旧, 每帧多算 max_conf + low_conf streak (纯元数据)
 ├─ control socket:  STATUS(+rule字段) | RULE(新) | START | STOP | QUIT
 └─ http :8080:      / (黄条rule) | /api/rule(新) | /snapshot.jpg | /api/dets | /api/status | /api/start|stop

control_service/ (Python, 板上常驻候选)
 ├─ scheduler.py:  query_daemon(RULE/STATUS) + decide() 纯函数
 └─ agent.py:      轮询, 模型切换才打印 + placeholder 入队点
```

## 铁律（未动）
- 调度永远不碰 pipeline：只读 STATUS/RULE，只调 START/STOP。
  START 仍要求 `StreamSession::start()==true` 才回 accepted；
  STOP 仍要求 session停 + 双 rename 落盘才回 completed。
- daemon 挂了 agent 只报警（`model=unknown`），不拉崩。

## 规则（daemon 侧 + scheduler.py 同构）
- 默认 `active_model=yolo`，`reason=default: yolo everyday`。
- 连续 ≥3 帧低置信（无框 或 max_conf < 0.50，可 `--low-conf` 调）且有框
  → `yolo+nanoowl?`：候选 NanoOWL 二阶段。
- 连续 ≥3 帧全空 → `yolo+vlm?`：需 VLM 复核。
- streak 断一次即清零回 yolo；warmup 帧不计入统计（沿用 M6 语义）。

## 新增/修改文件

| 文件 | 改动 |
|------|------|
| `apps/edgevisiond/main.cc` | Daemon 加 `active_model/rule_reason/low_conf_*`（mu 保护）；run_one 算 max_conf/streak；`handle_cmd` 加 `RULE`，`STATUS` 追加规则字段；`handle_http` 加 `/api/rule`；首页加 `#rule` 黄条（绿=yolo，橙=升级候选）；main 加 `--low-conf` |
| `control_service/scheduler.py` | `decide()` 纯函数 + `query_daemon()` + `poll_once()` |
| `control_service/agent.py` | 轮询 loop，切换打印 + placeholder 钩子 |
| `tests/test_m7_rule.py` | 6 用例 host 侧单测（无需硬件） |

## 验证结果（板上实测，2026-10-30，MAXN_SUPER + jetson_clocks 已锁）
- 锁频恢复：`infer_bench` 100 帧 `avg 4.36ms p50 4.36 p99 4.38`（重启掉频后曾 12.8ms）。
- 编译：`cmake --build build/board --target edgevisiond` 一次过（仅 `??'` trigraph warning，无害）。
- `RULE` 命令：`{"active_model":"yolo","reason":"default...","low_conf_thresh":0.50,...}`。
- `/api/rule` 200；首页含 `id=rule`；`STATUS` 带规则字段。
- STREAM SMOKE：START(accepted) → 12s 空场景 416 帧 → 规则翻为
  `yolo+vlm? / empty streak>=3`（streak=353，符合预期：镜头前无目标）→
  JPG 200（24471B）→ STOP(completed)，`summary lat_avg 4.36ms`。
- `agent.py --once` 正确打印切换 + placeholder。
- `events.jsonl` 每行新增 `max_conf/low_conf/rule`；`summary.json` 延迟口径不变。
- 网页：`http://100.117.193.104:8080/` 顶黄条实时显示 model/reason/streak/max_conf。

## 遗留（真升级时再做）
- NanoOWL worker：先本机验证显存占用，再定常驻/按需起停；可疑帧 crop 存 `pending_review/`。
- 远端 VLM：抽查可疑帧，结果合并回 session（不进实时链路）。
- `jetson_clocks` 重启掉频：加开机自锁（systemd oneshot 或 service 前置）。
- 长稳 8h + docker 镜像：仍按原计划待办。

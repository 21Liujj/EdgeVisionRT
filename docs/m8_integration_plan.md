# M8：三仓合一集成计划 — edgevisionrt 为底座，NanoOWL/Qwen 为二级 worker

> 状态：设计文档（未开工）。M7 占位已闭环（见 `m7_tech_analysis.md`）。
> 目标：把三个碎片仓整合成一个可上电运行、可演示、可验收的端侧级联系统。

---

## 1. 背景与现状：手里到底有哪三套东西

### 1.1 edgevisionrt（底座：腿）— 本地仓库 `C:\Users\31542\Desktop\gpt\edgevisionrt`

- **链路**：USB UVC 相机 720p MJPG → V4L2 MMAP（`src/camera/v4l2_capture.cc`）
  → MJPEG 软解 → H2D → 自写 CUDA 预处理（`src/backends/cuda_preprocess.cu`）
  → TensorRT YOLOv8-n FP16（640×640）→ D2H → NMS。
- **服务**：常驻 `edgevisiond`（`apps/edgevisiond/main.cc`），双线程
  （control socket + pipeline），systemd 托管（`deploy/systemd/edgevision.service`），
  udev 按 VID:PID 绑 `/dev/video-edgevision`，会话产物原子落盘
  （`summary.json.tmp → summary.json`，`events.jsonl.tmp → events.jsonl`）。
- **调度占位（M7）**：每帧算 `max_conf + low_conf streak`（纯元数据，不碰推理），
  `STATUS`/`RULE` 命令 + `/api/rule` + 网页黄条 + `control_service/scheduler.py::decide()` 纯函数。
  规则：默认 yolo；连续≥3 帧低置信有框 → `yolo+nanoowl?`；连续≥3 帧全空 → `yolo+vlm?`。
- **网页**：`:8080/`，START/STOP、黄条规则、`/snapshot.jpg` + canvas 框、
  `/api/dets`、`/api/status`。铁律 `sent != accepted != completed` 全程保持。
- **实测基线（2026-10-30，MAXN_SUPER + jetson_clocks 已锁）**：
  `infer_bench` 100 帧 avg **4.36ms** p50 4.36 p99 4.38；stream smoke 416 帧 avg 4.36ms。
- **缺什么**：只有 YOLO 80 类闭集；难例（类别外、低置信、空帧有东西）无能力；
  M7 的 `yolo+nanoowl?` / `yolo+vlm?` 是空 placeholder，无 `pending_review/`，无 worker。

### 1.2 edge-perception（眼）— https://github.com/21Liujj/edge-perception

- **任务**：开放词汇分割。输入一张图 + 一句话（如 `"a cup"`），输出像素级 mask。
  不固定类别，不用为新物体重训。
- **链路**：
  ```
  图 + 文本 → NanoOWL（OWL-ViT base-patch32，TensorRT FP16，文本条件检测）
            → 框 + 置信度（~100–150ms，~800MB）
            → NanoSAM（ResNet18 蒸馏 encoder + MobileSAM decoder，FP16/INT8）
            → mask（~80–120ms，~600MB）
  端到端 ~200–300ms，~1.4GB（AGX Orin，JetPack 6.2，TRT 10.3 实测）
  ```
- **做到哪**：Stage A–D 引擎打通；Stage E INT8 对比（mIoU 0.741→0.728，-1.3%，
  快 19%，结论生产用 FP16）；Stage F C++ runtime（`cpp_runtime/`）；Stage G 负结果
  （训练-free 后处理救不了 24×24 patch grid 的定位粗，诚实归档）。
- **短板（自认）**：定位粗（每 patch 32px，先天）；单目标（只取最高分）；
  无跟踪；定图 768×768 最优；纯单图 demo，无视频流、无服务化。
- **轻量化**：选小模型（base-patch32 + ResNet18）+ FP16/INT8 实测 + 复用官方蒸馏权重；
  **无剪枝**，无自训蒸馏。
- **在集成里的角色**：YOLO 摇头时叫来的复核员——按文本查一遍（"有没有、是什么"），
  有框再 SAM 抠 mask。不指望它精修定位（Stage G 已证伪此路）。

### 1.3 edgeqwen-jetson（嘴）— https://github.com/21Liujj/edgeqwen-jetson

- **任务**：Qwen2-VL-2B 板上看图说话。不联网，单轮问答。
- **结构**：双引擎——Visual Encoder（视觉塔 FP16，~229ms）+ LLM Engine
  （Qwen2-2B + KV cache，prefill ~173ms + decode ~42ms/token），`~24 tok/s`，
  峰值统一内存 **~2.2GB**（Orin Nano Super 8G，25W，JetPack 6.2，TRT 10.3）。
- **接口**：`libedgeqwen_runtime.a` + 单头 `edgeqwen/runtime.h`，
  `Runtime(config).generate(messages)` 一把梭，`getLastProfile().toJson()` 看耗时。
  engines 约 4.6GB，不在仓内，需 `build_llm.sh` 现 build。
- **做到哪**：Stage 1–13（转换→benchmark TextVQA/GQA/VQAv2→kernel profiling→
  内存分析→500 次/28min 稳定→SDK 打包）。
- **短板（自认）**：单轮（无多轮 KV reset）；多线程串行（mutex）；只有 `.a` 无 `.so`/C ABI。
- **轻量化**：2B 选型 + FP16 直转，无剪枝无蒸馏。
- **在集成里的角色**：最后一级复核。YOLO、NanoOWL 都拿不准 → 抽一帧问一句
  （如"图中红色线缆旁边的物体是什么，是否异常"），20 token ≈ 1s，结论回写 session。
  单轮、串行正好够用（事件级调用，几秒一次）。

### 1.4 为什么必须合（三个理由）

1. **装不下**：YOLO 常驻 + OWL/SAM 1.4G + Qwen 峰值 2.2G，全常驻在 8G 统一内存上必爆。
   必须分时复用——这正是 M7 占位逻辑存在的意义。
2. **互补**：底座缺二阶段能力（placeholder 空着），眼和嘴缺视频流+服务+调度。
   一边是腿，一边是脑和嘴。
3. **Stage G 直接定调用方式**：NanoOWL 定位救不了 → 别当每帧精修，只当抽帧复核。
   低频调用 + 文本查询，这就是集成形态。

---

## 2. 整合目标：要实现一个什么

> 一句话：一块 Orin Nano Super，上电就跑——YOLO 常开做实时闭集检测（4.36ms），
> 遇到它看不懂的帧自动抽出来调 NanoOWL 做开放词汇定位+抠 mask，再拿不准就问 Qwen 一句，
> 最后网页上全程可见、session 里全程可审计。

```
USB 720p 实时流
  → [YOLO 常驻 4.36ms] 人/车/杯子等 80 类，框出来就完事
  → [可疑？M7 streak 规则] 连续≥3 帧低置信/空帧 → 抽 1 帧 → pending_review/
  → [NanoOWL 按需起 200–300ms] "红色线缆旁边是什么？" → 框（开放词汇）
  → [NanoSAM 接着跑 80–120ms] 框 → 像素 mask（YOLO 框可直接转 box prompt，免二次检测）
  → [Qwen 按需起 约1s] "描述这团东西，是否异常？" → 文本结论
  → 回写 events.jsonl + 网页黄条第二行，全程可看
```

### 2.1 是什么、不是什么（面试口径）

- **是**端侧部署优化（三层全占）：
  （1）模型层：FP16 全量 TRT、INT8 对比实验、选小模型/蒸馏权重复用；
  （2）系统层：8G 内存下四引擎分时复用、CUDA 预处理自写、V4L2 零拷贝、MAXN+锁频保基线；
  （3）服务层：systemd 常驻、udev、原子落盘、socket 铁律、网页可视化。
- **不是**算法创新：不发新结构、不刷 mIoU。讲"在一块 8G 板上把四引擎级联跑通，
  满足实时+功耗+稳定"，这就是端侧岗位的本职。

### 2.2 非目标（本期不做）

- 不重训任何模型（NanoOWL/Qwen 权重原样用，只加抽帧+crop+调用层）。
- 不统一输入尺寸（YOLO 640、OWL 768 各走各 resize，只保证"同一帧、同一目标"对上）。
- 不做多目标跟踪、不做多轮对话（Qwen 单轮正好够用）。
- 不上 docker（沿 M6/M7 结论，延后）。

---

## 3. 总体架构：底座不动，按需升级

```
┌─ edgevisiond（C++，常驻，唯一碰相机/推理实时链路的进程） ────────┐
│ pipeline: V4L2 → 解码 → CUDA预处理 → YOLO TRT → NMS → snapshot    │
│   └─ 新增：streak 触发时写 pending_review/<stamp>/{crop.jpg, meta.json}│
│       （原子写 .tmp→rename；推理延迟零增加；只做元数据+存图）        │
│ control+http: STATUS|RULE|START|STOP + :8080（黄条+worker行）      │
└──────────────────────────────────────────────────────────────────┘
          │ meta.json（帧号/时间戳/YOLO框/conf/触发理由）
          ▼
┌─ pending_review/（文件队列，铁律外，不进 socket） ───────────────┐
│ <stamp>/crop.jpg（原分辨率 crop，不是 480px 预览小图）              │
│ <stamp>/meta.json（见 §5 接口契约）                                │
│ <stamp>/result.json（worker 回写，统一格式）                        │
└──────────────────────────────────────────────────────────────────┘
          │ worker 池：同一时间只活一个（进程互斥+文件锁），跑完即退
          ▼
┌─ nanoowl worker（按需起停子进程） ──┐  ┌─ qwen worker（按需起停子进程） ─┐
│ 复用 edge-perception 的 engines +   │  │ 复用 edgeqwen 的 .a + runtime.h  │
│ C++ runtime（或 python pipeline，    │  │ 单轮 generate()，问一句复核       │
│ 先在 Nano Super 上重验延迟）         │  │ 峰值 2.2G，与 YOLO 同跑需实测挤占 │
└────────────────────────────────────┘  └──────────────────────────────────┘
          │ result.json（框/mask路径/文本结论/耗时/显存）
          ▼
┌─ control_service/agent.py（调度，铁律外） ────────────────────────┐
│ 扫 pending_review/ → 按优先级起 worker → 回写 events.jsonl → 网页  │
│ 调度只读 STATUS/RULE、只调 START/STOP；daemon 挂了只报警不拉崩     │
└──────────────────────────────────────────────────────────────────┘
```

**铁律重申**：worker 永远不碰相机、不碰 pipeline、不碰 control socket 的
accepted/completed 语义。`sent != accepted != completed` 只属于 daemon 的 START/STOP。

---

## 4. 数据流：视频帧怎么喂给图片模型（两块胶水）

### 4.1 胶水一：输入格式统一（好办，纯工程）

| 事项 | YOLO 侧（已有） | NanoOWL/NanoSAM 侧（复用） | Qwen 侧（复用） |
|---|---|---|---|
| 输入来源 | V4L2 解码 BGR（1280×720） | 同一帧的 **原分辨率 crop**（非 480px 预览图） | 同一帧 crop 或整帧（按问题定） |
| resize | CUDA 预处理 → 640×640 | 各自预处理 → 768×768（OWL 原生） | visual encoder 原生（build 时定） |
| prompt | 无（闭集） | 文本（来自规则：如低置信类名/通用"unknown object"）+ YOLO 框转 box prompt 给 SAM（免二次检测） | 文本（如"描述 crop 中物体，是否异常，只回一句"） |
| 输出 | 框+cls+conf（4.36ms） | 框+mask（200–300ms） | 文本（~1s/20token） |

- crop 规则：取触发帧 YOLO 最高分框外扩 20%（空帧则取中心 50% 整图），存原分辨率，
  不做超分，不压缩画质（Qwen/OWL 对压缩伪影敏感）。
- 同一目标对齐：`meta.json` 带 `frame_no + ts_wall_ns + yolo_boxes`，
  worker 结果回写时原样带回，`events.jsonl` 按帧号 join。

### 4.2 胶水二：调用节奏统一（核心，靠 M7 规则）

- 量级差：YOLO 4.36ms vs OWL 200–300ms vs Qwen ~1s——差两个数量级，不可能每帧跟。
- 节奏：YOLO 先跑；只有 streak 触发才抽 **1 帧**送二阶段；二阶段内部 OWL→SAM→Qwen
  串行降级（OWL 有高分框且 mask 置信够就不问 Qwen；拿不准才问）。
- 频率上限：worker 串行 + 互斥锁，天然限流；Qwen 按"疑似事件"级调用（几秒一次），不进帧循环。
- 显存互斥：同一时间只活一个 worker 进程，跑完即退；Qwen 2.2G 峰值与 YOLO 常驻同跑的
  挤占以上板实测为准（先测 `tegrastats` VDD_IN + 解码延迟抖动）。

---

## 5. 接口契约（先定死，再写代码）

### 5.1 `pending_review/<stamp>/meta.json`（daemon 写，worker 读）

```json
{
  "frame_no": 1234,
  "ts_wall_ns": 1793365545000000000,
  "trigger": "low_conf_streak>=3 | empty_streak>=3",
  "active_model": "yolo+nanoowl?",
  "yolo_boxes": [{"x":100,"y":80,"w":200,"h":150,"conf":0.32,"cls":0}],
  "last_max_conf": 0.32,
  "crop": "crop.jpg",
  "prompt_owl": "unknown object near red cable",
  "prompt_qwen": "描述图中裁剪区域的物体，判断是否异常，只回一句话。"
}
```

### 5.2 `pending_review/<stamp>/result.json`（worker 写，agent 读）

```json
{
  "worker": "nanoowl | qwen",
  "ok": true,
  "lat_ms": 245.0,
  "peak_mem_mb": 810,
  "owl_boxes": [{"x":0,"y":0,"w":0,"h":0,"conf":0.0,"label":"cup"}],
  "mask_path": "mask.png",
  "qwen_text": "图中为散落的黑色线缆，无异常。",
  "error": ""
}
```

### 5.3 回写（agent 做）

- `events.jsonl` 追加一行：`{"frame":N,"escalated":true,"worker":"nanoowl",...}`，
  与原 YOLO 行按 `frame_no` join，不改原行格式（向后兼容）。
- 网页黄条第二行：`worker=idle | owl-working | qwen-working | done:<摘要>`，
  摘要截断 60 字，文本只走 HTTP，不走 socket（socket 只传状态，不传大文本）。

---

## 6. 显存/功耗预算（8G 统一内存，先算账再上板）

| 常驻/按需 | 组件 | 显存（历史实测） | 备注 |
|---|---|---|---|
| 常驻 | edgevisiond + YOLO FP16 + snapshot | 约 300–500MB（`systemd status` 曾报 67.5M RSS，+CUDA/TRT 显存另计，需重测 `tegrastats`） | 基线，不能动 |
| 按需 | NanoOWL FP16 | ~800MB | AGX 数，Nano 待重验 |
| 按需 | NanoSAM FP16 | ~600MB | 同上；INT8 省但掉点，生产用 FP16 |
| 按需 | Qwen2-VL-2B 峰值 | ~2.2GB | Nano Super 实测；engines 4.6GB 落盘 |
| 互斥后峰值 | YOLO + 最多一个 worker | 约 ≤3GB（含 unified memory 抖动余量） | 8G 可用约 7.6G（含系统），理论 OK，实测为准 |

- 功耗：MAXN_SUPER + `jetson_clocks` 锁频（M7 已验证 4.36ms）；worker 起停时看
  `VDD_IN` 尖峰；`jetson_clocks` 重启掉频问题本期加开机自锁（oneshot service）。
- 第一实测数：NanoOWL 在 **Nano Super** 上的真实延迟（AGX 的 100–150ms 不能抄）。

---

## 7. 复用清单：不动那两个仓，只调接口

- **NanoOWL**：优先复用 `edge-perception` 的 `cpp_runtime/` + 已 build 的 engines；
  若 Nano 上 engines 需重 build，用它的 `scripts/build_engines.py`，不抄代码。
  文本 prompt 由 agent 按触发理由组装（先固定两句模板，不做 prompt 工程）。
- **NanoSAM**：YOLO/OWL 框直接转 box prompt，免二次检测；mask 存 `mask.png`，
  网页可选叠加（先只给下载链接，不做前端融合）。
- **Qwen**：调 `edgeqwen_runtime.a` + `runtime.h`，`generate()` 单轮；
  多线程串行、只有 `.a` 都接受（事件级调用碰不到这两个短板）。
  engines 4.6GB  board 上现 build，不进 git（`.gitignore` 已有模型忽略则沿用，否则加）。
- **版本冻结**：JetPack 6.2 / TRT 10.3 / CUDA 12.x / 模型 commit hash 写进
  `reports/environment.json` 旁的 `reports/workers.json`（新增）。

---

## 8. 评测体系：从图片指标换成视频事件级

| 层 | 旧口径（保留作回归） | 新口径（验收用） |
|---|---|---|
| YOLO | infer_bench 延迟 | 不变：avg/p50/p99（基线 4.36ms） |
| OWL/SAM | 图片集 mIoU（0.741 FP16） | 视频事件：YOLO 漏检中被 OWL 找回率；端到端触发→mask 延迟 |
| Qwen | TextVQA/GQA/VQAv2 | 事件级：可疑帧→文本结论延迟；结论可用率（人工抽 50 条判对错） |
| 系统 | 无 | worker 峰值显存；`VDD_IN` 尖峰；daemon 延迟抖动（worker 同跑时 YOLO p99 涨多少） |
| 服务 | session 落盘 | events 可 join 率（result 能对回 frame_no 的比例，要求 100%） |

---

## 9. 轻量化现状与剪枝坑位（诚实版，简历直接用）

- 已做：选小模型（2B / base-patch32 / ResNet18）+ FP16 全量 TRT + INT8 对比实验
  （-1.3% mIoU / +19% 速度，结论 FP16）+ 复用官方蒸馏权重（NanoSAM ResNet18）。
- 没做：剪枝（attention head / 层剪）、自训蒸馏、Qwen 量化到 INT8/4bit。
- 坑位：OWL-ViT head 剪枝、Qwen2B 层剪——现有 harness（`evaluate_pipeline.py` /
  `reproduce_benchmarks.sh` / `infer_bench`）可直接验剪完效果，不用搭新架子。
  本期不做，单独立项。

---

## 10. 实施步骤（分四步，每步可独立验收）

- **M8a 抽帧落盘（先做）**：daemon 加 `pending_review/` 原子写（crop+meta）；
  agent 加扫目录（只扫不调 worker，先打日志）；网页黄条第二行 `worker=idle`。
  验收：STREAK 触发 → 目录出现 crop+meta → events 有 trigger 行 → YOLO 延迟不变。
- **M8b NanoOWL worker**：板上重验 OWL 延迟（Nano Super 实测数替换 AGX 数）；
  worker 子进程调 `edge-perception` runtime；`result.json` 回写 + 网页 `owl-working/done`。
  验收：事件级找回率 + 端到端延迟 + 显存峰值三数。
- **M8c Qwen worker**：`libedgeqwen_runtime.a` 接入；固定 prompt 模板两句；
  结果文本回写 + 网页 `qwen-working/done`。验收：可疑帧→结论延迟 + 50 条人工可用率。
- **M8d 收尾**：`jetson_clocks` 开机自锁；`docs/progress.md` 更新；三个仓 README
  互链（主项目 edgevisionrt + 两个 worker 仓）；简历口径定稿（一主二从）。

---

## 11. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| Nano 在 Nano Super 上远慢于 AGX（如 >400ms） | 中 | 降输入到 640/512 重验；只当"有无"用，不当"精修"用；不行就只留 Qwen |
| Qwen 与 YOLO 同跑挤占（p99 抖动） | 中 | worker 互斥 + 降 Qwen 并发为 1；实测 `tegrastats`，超标则 worker 运行时 YOLO 降帧（抽帧模式） |
| engines 4.6G 板上 build 太久/失败 | 低 | 沿用 edgeqwen 的 `build_llm.sh`，先在空闲时段跑；失败则 engines 放 SSD，不进 eMMC |
| 文本 prompt 玄学 | 低 | 先固定两句模板，不做 prompt 调优；结论只当"参考"，不进控制 |
| 三仓版本漂移 | 中 | `reports/workers.json` 冻 commit hash + engine md5；CI 只验接口契约（`test_m8_contract.py`） |

---

## 12. 文件清单（本期要动/要加的）

| 文件 | 动 |
|---|---|
| `apps/edgevisiond/main.cc` | 加 pending_review 原子写 + `/api/worker` + 黄条第二行 |
| `control_service/scheduler.py` | 加扫目录 + 起 worker + 回写（placeholder 换真入队） |
| `control_service/agent.py` | 加 worker 心跳 + 摘要打印 |
| `control_service/workers/*`（新） | `nanoowl_worker.py`、`qwen_worker.py`（薄封装，只调那两个仓的接口） |
| `pending_review/`（运行时目录，不进 git） | crop+meta+result 约定（§5） |
| `tests/test_m8_contract.py`（新） | meta/result json schema 单测（host 可跑） |
| `reports/workers.json`（新） | worker 版本+engine md5+实测延迟 |
| `docs/m8_integration_plan.md` | 本文 |
| `docs/progress.md` | M8a–d 逐项打勾 |

---

## 13. 验收标准（一句话版）

> 空场景 12s 不误触发；手进镜头 YOLO 出框黄条回 yolo；遮挡/怪东西触发 OWL→mask→网页可见；
> 极难例触发 Qwen→一句话结论回写；全程 YOLO p99 不涨超 0.5ms；events 100% 可 join；
> `jetson_clocks` 重启不掉。

---

*下一步（等拍板）：M8a 开工，还是先上板验 NanoOWL 在 Nano Super 上的真实延迟（半小时出数）。*

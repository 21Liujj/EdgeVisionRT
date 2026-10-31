# M4: BSP/功耗 — 技术解析文档

## 目标
用 `nvpmodel` 切换功耗档 + `jetson_clocks` 锁频，以 `tegrastats` 采集
`infer_bench` (YOLOv8-n FP16, 100帧) 在 15W 与 MAXN_SUPER 下的
延迟/功耗基线，回答"满血模式值多少电、换多少延迟"。

## 方法
- 档位: `nvpmodel -m 0` (15W) vs `-m 2` (MAXN_SUPER) + `jetson_clocks`
- 采样: idle 10s (`--interval 500`), load 与 infer_bench 并跑 (`--interval 200`)
- 指标: `VDD_IN` 平均功耗, `GR3D_FREQ` 平均利用率, infer_bench SUMMARY
- 注意: 必须验证 `nvpmodel -q` 真正切过去——直接 `echo nvidia | sudo -S
  nvpmodel -m 0`，把 `$SUDO` 变量存管道写法会静默失效(第一次对比翻车原因)

## 结果

| 指标 | 15W (id 0) | MAXN_SUPER + clocks |
|------|-----------|---------------------|
| idle VDD_IN | 4360 mW | 6351 mW |
| load VDD_IN | 5457 mW | 7601 mW |
| infer avg / p50 | 14.08 / 14.08 ms | 4.74 / 4.74 ms |
| infer p99 / max | 14.24 / 14.24 ms | 4.75 / 4.75 ms |
| GPU 利用率 | 36.5% | 12.4% |
| CPU 频率 | ~1.19 GHz (ondemand) | 1.728 GHz (locked) |
| 帧成功率 | 100/100 | 100/100 |

## 关键结论
- **延迟 2.97x**: 14.08ms → 4.74ms，GPU 从限频(15W档)到满频是主因
- **load 功耗 +2.1W**: 5.46W → 7.60W，换 3 倍延迟，划算
- **idle 功耗 +2.0W**: `jetson_clocks` 锁频后空载也烧电，常驻服务要权衡
  (按需锁频 vs 常锁)
- **GPU 利用率反降** (36.5% → 12.4%): 同一份工作更快做完，利用率自然低，
  不能拿利用率直接对比两档负载
- **单帧能耗**: 15W 76.8mJ vs MAXN 36.0mJ —— race-to-idle 成立，
  干得快反而每帧更省电

## 遗留
- `jetson_clocks` 常锁的 idle 代价 → M6 长稳服务用按需调频策略验证
- GPU 真实频率节点未找到(debugfs/devfreq 路径均无)，频率值由 nvpmodel
  档位 + 延迟反推，未直接读到 Hz 数


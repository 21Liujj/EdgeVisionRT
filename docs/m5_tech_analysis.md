# M5: UVC 诊断 — 技术解析文档

## 目标
回答"MJPEG 瓶颈到底卡在哪":USB 线 / uvcvideo 驱动 / 应用层解码三选一,
用数据而非猜测定位。

## 基线(事前已知)
- 相机 Generic Web Camera (4a54:5232),USB 2.0 480M,`uvcvideo` 5.15.148
- 在用 MJPEG 1280x720@30;YUYV 720p 仅 15fps(带宽原罪,MJPEG 压缩比约 4:1)
- 内核无 usbmon 模块(ftrace 可用:vb2/v4l2 events),5.15-tegra 未编 usbmon

## 方法(全部用户态+ftrace,零内核改动)
1. `v4l2-ctl --stream-mmap --stream-count=300` 纯采集,摘掉解码/推理
2. `ftrace vb2_dqbuf` 逐帧时间戳,算帧间隔分布 + bytesused 分布
3. MJPEG vs YUYV 对照,分离"驱动层"与"USB 带宽层"瓶颈

## 结果

### A. V4L2 层纯采集帧率
| 配置 | 名义 | 实测 | 结论 |
|------|------|------|------|
| MJPG 1280x720 | 30fps | 30.00fps (300帧/10.83s) | V4L2 层满血 |
| MJPG 640x360 | 30fps | 30.00fps | 同上 |
| YUYV 960x544 | 30fps | 20.83fps (300帧/14.93s) | **驱动/带宽层丢帧** |

### B. MJPG 720p 逐帧间隔(ftrace,150帧)
- min 31.94 / p50 32.05 / p99 36.01 ms —— 148 帧均匀如钟
- 仅 1 个 gap:第 148 帧处 432ms(单次,尾部,疑似 stream-teardown 竞争,非稳态)
- bytesused:首 3 帧爬坡(113K→160K→189K,曝光/白平衡收敛),之后稳定 ~290KB
- 稳态结论:USB 线 + 驱动 DQBUF 路径**无瓶颈**,MJPEG 720p@30 可用

### C. YUYV 对照(150帧,无 gap 却少帧)
- 间隔极其均匀:p50 48.00ms,抖动仅 0.08ms,零 gap
- 但 48ms = 20.8fps ≠ 30fps 名义值 —— 相机端就没发足
- 根因:USB 2.0 高速 bulk 对 YUYV 原始流带宽不够(720p YUYV 理论 27.6MB/s,
  叠加协议开销超 USB2 实际吞吐),相机固件降帧保画质
- 这是物理层天花板,驱动调优救不了 → 选 MJPEG 是正确架构决策(M0 已定,本次闭环验证)

## 关键结论
1. **瓶颈不在驱动**:MJPG 路径 V4L2 层 30.00fps,p50 抖动 0.1ms,无需 uvc patch
2. **YUYV 天花板是 USB2 带宽**:48ms 均匀间隔 = 相机端主动降帧,非驱动丢包
3. **首帧爬坡**:前 3 帧 bytesused 递增 + MJPG 首帧偏小,应用层 warmup 至少 5 帧
   (infer_bench 已用 --warmup 10,合理;--warmup 20 更稳)
4. **尾部单次 432ms gap**:只在 stream 停止边缘出现,不影响连续采集,
   4 帧 vb2 缓冲下稳态无欠载(queued=3 恒定,无欠流)

## 遗留 / 不做事项
- 不写 kernel patch:诊断证明驱动层健康,动内核零收益
- usbmon 缺失:5.15-tegra 未编译该模块;本次用 ftrace vb2 时间戳达到同等定位效果
- NVJPEG 硬解(MJPEG CPU 软解仍在端到端路径上) → 留给 M6/M7 按需优化,本阶段不展开
- 建议:M6 长稳加一条"连续 8h DQBUF 间隔 p99"监控,432ms 级 gap 若在稳态复现再追


# M1 技术解析：V4L2 数据面

## 1. M1 要解决什么问题

M0 阶段我们有了一个能打开 /dev/video0、逐帧读取、写入文件的 `capture_probe`。
但它缺了三样东西，任何实时推理管线都不能没有：

| 缺失 | 后果 | M1 怎么补 |
|---|---|---|
| 帧不在池子里，每次 `read()` copy 到调用栈上的临时变量 | 无法让多个消费者（推理+预览+录制）共享同一帧；内存碎片 | FramePool 预分配 + acquire_index/slot_ref 零拷贝 |
| 没有所有权语义 | 谁在用？用完了吗？能不能复用？全是口头约定 | FrameLease：RAII 租约，析构自动归还，epoch 防 use-after-free |
| 没有生产者-消费者解耦 | capture_probe 在 for 循环里同步 read → 相机抖动会导致推理掉帧 | StreamSession：后台捕获线程 → 有界队列 → 消费者拉取 |

一句话：**M0 是"能读到帧"，M1 是"帧能被安全地、高效率地在多条执行路径间流转"**。

## 2. 架构：三层管道

```
┌──────────────────────────────────────────────────────────┐
│                    StreamSession                          │
│                                                          │
│  ┌─────────────┐     ┌──────────┐     ┌───────────────┐ │
│  │ V4L2Capture │────▶│ FramePool │────▶│ BoundedQueue  │ │
│  │  (内核驱动)  │DQBUF│ (6槽预分配)│push │ (容量4,slot int)│ │
│  │             │copy │           │     │               │ │
│  └─────────────┘     └──────────┘     └───────┬───────┘ │
│                          ▲                     │         │
│                          │ release()           │ pop()   │
│                          │ (析构归还)           ▼         │
│                     ┌──────────┐         ┌────────────┐  │
│                     │FrameLease│◀────────│ 消费者线程   │  │
│                     │(RAII租约) │consume()│(推理/预览)  │  │
│                     └──────────┘         └────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 2.1 FramePool：预分配，零拷贝

```cpp
class FramePool {
  int acquire_index();   // 返回空闲槽位号，-1=池耗尽
  Frame& slot_ref(int);  // 按槽位引用帧（不拷贝数据）
  void release(int);     // 归还槽位
  uint64_t epoch_of(int); // 该槽位的代数（防 use-after-free）
};
```

关键设计决策：
- **acquire_index + slot_ref 分离**。旧版 `acquire(Frame& out)` 会把整个 `vector<uint8_t>` 拷贝出来——在 MJPEG 帧 200KB+ 时每次拷贝 200KB。新版 acquire_index 只返回整数，slot_ref 返回池内引用，消费者通过 FrameLease 直接操作池内内存。
- **epoch 计数器**：每次 acquire 一个槽位时 `epoch++`。FrameLease 持有创建时的 epoch 值，`valid()` 检查 `pool->epoch_of(slot) == my_epoch`。如果槽位被归还后又重新分配，epoch 不匹配，旧租约自动失效。
- **pool 耗尽 = 丢帧，不阻塞**。`acquire_index()` 返回 -1 时生产者计数 drop（`frames_dropped_pool`），sleep 500us 后重试。不做阻塞等待——因为阻塞生产者等于阻塞 V4L2 驱动，会导致内核缓冲区溢出。

### 2.2 FrameLease：RAII 帧租约

```cpp
class FrameLease {
  // 构造: 接管 slot，记录当时 epoch
  // 析构: 自动 pool->release(slot)
  // 移动: 所有权转移，源变空
  // 拷贝: 禁止
  // valid(): pool 还在 + slot 合法 + epoch 匹配 + frame.valid==true
};
```

这是整个数据面安全性的基石。三条规则：
1. **不能拷贝**（=delete）。同一帧不能同时被两个消费者"拥有"。
2. **可以移动**。`lease = sess.consume(lease, timeout)` 移动语义，旧租约自动归还，新租约接管。
3. **自动归还**。租约离开作用域时，帧自动回到池中。不需要显式 `release()`。

**epoch 如何在测试中验证**：测试特意模拟了"归还→重新分配同槽"场景，确认旧租约的 `valid()` 返回 false。

### 2.3 BoundedQueue<int>：解耦生产者和消费者

队列元素是 `int`（FramePool 槽位号），不是帧数据本身。
- **生产者**：`pool->acquire_index()` → 填充帧 → `queue->try_push(slot)`。失败则 release slot + 计数 drop（`frames_dropped_queue`）。
- **消费者**：`queue->wait_pop(slot, timeout_ms)` → 构造 `FrameLease(&pool, slot)` → 使用完毕后析构自动归还。
- **shutdown**：`stop()` 时 `running_=false` + `queue->shutdown()` → `wait_pop` 立即返回 false。

**为什么用有界队列而不是无界队列**：无界队列在消费者慢时会无限堆积帧——每个帧都占着一块 FramePool 槽和一个内核 mmap buffer。堆积超过 pool_slots 时生产者直接拿不到槽位（pool 耗尽 drop）。有界队列的"满则丢"语义让系统有确定的背压行为，不会有内存暴涨。

### 2.4 StreamSession：整体编排

```cpp
bool StreamSession::start(device, w, h, fourcc,
                          pool_slots=6, queue_capacity=4, buffer_count=4);
bool StreamSession::consume(FrameLease& lease, int timeout_ms);
void StreamSession::stop();
StreamStats stats();
```

**start() 的执行顺序**：`cap.open() → cap.start() → 读协商格式/驱动FPS → 按协商尺寸建FramePool → 建BoundedQueue → 启动capture_loop线程`。FramePool 在 camera start 之后才建，因为此时才知道驱动实际给的宽高（可能不准）。

**capture_loop 核心逻辑**：
```
while running:
  slot = pool.acquire_index()
  if slot < 0: drop_pool++; sleep 500us; continue
  if cap.read(pool[slot]) fails: release; error++; continue
  pool[slot].valid = true
  captured++
  if queue.try_push(slot) fails: release; dropped_queue++
  update queue_peak_depth (CAS loop)
```

**stop() 的关闭顺序**：`running_=false → queue.shutdown() → worker.join()`。这里没有先关 camera——capture_loop 在 running_=false 后会执行 `cap.stop(); cap.close()` 再退出。这样保证所有已入队的帧可以被消费者取完。

## 3. 丢帧统计：三重来源

| Counter | 含义 | 何时递增 |
|---|---|---|
| `frames_dropped_pool` | FramePool 6个槽位全忙 | `acquire_index()` 返回 -1 |
| `frames_dropped_queue` | BoundedQueue 4个槽位全满 | `try_push(slot)` 返回 false |
| `frames_error` | V4L2 read 失败 | `cap.read()` 返回 false |

三个计数器都是 `std::atomic<uint64_t>`，生产者线程写、消费者线程读（`stats()`），无锁安全。

**实测结果**：60 帧 1280x720 MJPG，三种 drop 全部为 0，队列峰值=1。说明当前 consumer 速度完全跟上 producer。

## 4. 和 capture_probe 的关系

M0 的 `capture_probe` 保持不动——它是诊断工具，直接操作 V4L2Capture，用于快速验证相机是否在位、驱动是否正常。M1 的 `StreamSession` 是生产管线，后端推理和录制都走它。两者互不替代：

| | capture_probe | StreamSession |
|---|---|---|
| 用途 | 诊断、benchmark | 生产管线 |
| 帧生命周期 | 临时变量 | FramePool + FrameLease |
| 是否解耦 | 同步 for 循环 | 后台线程 + 有界队列 |
| 丢帧可见性 | 只有 ok/fail | 三重来源详细计数 |

## 5. 测试覆盖

| 测试 | 内容 | 平台 |
|---|---|---|
| `test_bounded_queue` | push/pop/shutdown/dropped | host + board |
| `test_frame` | FramePool acquire/release/epoch | host + board |
| `test_frame_lease` | RAII析构归还、移动语义、epoch防stale | host + board |
| `test_stream_session` | 60帧完整管道、stats验证 | board only (需V4L2) |

全部 4/4 通过，stream_session 3.07s 内完成 60 帧采集，0 丢帧。

## 6. 下一步：M2 TensorRT YOLOv8-n

M1 交付了一个可靠的帧流管道。M2 要做的事情：
1. 在板端验证 TensorRT 可用（`import tensorrt`）
2. 导出/构建 YOLOv8-n FP16 TensorRT engine（ONNX → trtexec → .engine）
3. 实现 `TrtEngine` RAII wrapper（`backends/trt/trt_engine.h`）
4. 对接 StreamSession：`consume() → 解码MJPEG → 预处理 → 推理 → 输出bbox`
5. parity 测试：同输入、同模型权重、ONNX Runtime vs TensorRT 精度差 < 1e-3
# mm-sidecar 与 phase 融合方案正式设计文档

## 1. 文档概述

### 1.1 文档目的

本文档用于对本次完整 Session 中围绕 `vLLM + Qwen3/3.5 VL` 多模态链路优化、`mm-sidecar` 融合方案、基准验证、调度与 fallback 设计等全部讨论内容进行正式复盘、梳理、整合与定稿，形成一份可直接交付、可继续演进、可落库维护的专业方案设计文档。

本文档覆盖如下内容：

- 需求背景与原始目标
- 已确认的 vLLM 原生行为与基线结论
- 历次方案迭代与被废弃方案
- 最终收敛的 `mm-sidecar v1` 设计
- `TP worker fallback` 的最终保留策略
- `PD 多 P` 场景下的部署与缓存边界
- 风险、限制、待测量确认项与后续演进路线

### 1.2 文档范围

本文档聚焦于以下主题：

- `vLLM` 多图多模态请求的 API server / engine / TP worker 协作链路
- `mm-sidecar` 在多图下载、读取、解码、CPU preprocess 方向的架构设计
- `local file / http / base64` 三类输入的统一处理策略
- `TP worker fallback` 的状态机、租约、决策与多 rank 一致性设计
- 单机与 `PD 多 P` 场景下的 sidecar 部署边界

本文档不直接覆盖以下内容：

- `vllm-ascend` 相关方案与适配
- GPU 侧 `image_embeds` sidecar 化
- 视频侧 sidecar 化
- 最终实现代码细节与接口定义的逐行代码级规格
- 完整的端到端 profiling 实现细节

### 1.3 文档结论层级

本文档中所有内容分为三类：

- **最终定稿方案**：本轮讨论已明确确认，可作为当前落地基线。
- **保留但待测量确认**：方向已提出，但是否纳入 `V1` 仍需用真实耗时与正确性结果确认。
- **已废弃或暂不采用方案**：本轮讨论已明确放弃，或推迟到后续版本再讨论。

---

## 2. 背景与目标

### 2.1 原始背景

本次工作的起点是部署 `vLLM + Qwen3/3.5-0.8B` 服务，并验证其能够正常通过 `curl` 请求完成响应。在服务部署可用后，重点迅速转向多模态链路性能、调度正确性与架构重构问题。

用户使用场景聚焦于如下请求特征：

- 模型：`Qwen3/3.5 VL` 多模态模型
- 模型长度：`max_model_len = 16k`
- 文本输入规模：约 `10k token`
- 图像输入规模：`13` 张 `288 x 512`
- 请求方式：`HTTP`
- 重点关注：多图场景下 API server、engine core、TP worker 各阶段耗时与可优化空间

### 2.2 原始性能与验证要求

本次 Session 中明确提出过如下性能与验证要求：

- 对服务进行预热
- 关闭 `mm cache` 缓存影响
- 单次 profile 不足，要求预热 `3` 次、采样 `5` 次
- 输出 `avg` 与 `max` 等统计信息
- 若各项波动处于可接受范围，可仅展示 `avg` 表
- 需要确认进入 LLM 的 token 规模是否等于：
  - 文本约 `10k token`
  - 图像经 processor 压缩/展开后的视觉 token
  - 最终总 token
- 需要拆解 API server / engine core / TP worker 的重点阶段耗时
- API server 打点必须通过 monkey patch 方式植入，不允许修改原始源码
- 重点 API server 函数包括：
  - `AsyncMultiModalContentParser._image_with_async`
  - `HfRenderer._tokenize_prompt_async`
  - `Qwen3VLMultiModalProcessor.apply`
- `HfRenderer._tokenize_prompt_async` 需统计到最后一个数据片段返回为止
- 需求后续明确排除 `vllm-ascend`

### 2.3 架构升级目标

在 profiling 需求基础上，后续讨论的核心目标演进为：

- 将多图下载、读取、解码、CPU preprocess 从 API server 热路径中解耦
- 引入 `mm-sidecar`，利用多核多进程并发能力处理多图输入
- 保持 `engine` 正确按 token 大小调度
- 不破坏 `prefix caching`
- 不破坏 `ViT DP` 与 `TP worker` 既有分工
- 避免 sidecar 成为新的强阻塞点
- 在 sidecar 未及时完成时，保留 `TP worker fallback` 自救能力

---

## 3. 已确认的基线事实

### 3.1 已确认的 vLLM 原生多图 HTTP 下载行为

通过阅读本地源码 `vllm 0.18`，已确认如下原生行为：

- API server 侧多图 HTTP 下载基线为**单进程 asyncio 并发模型**
- `AsyncMultiModalItemTracker.resolve_items()` 使用 `asyncio.gather` 并发等待多模态 coroutine
- `AsyncMultiModalContentParser._image_with_uuid_async()` 调用 `MediaConnector.fetch_image_async()`
- `MediaConnector.load_from_url_async()` 对 HTTP URL 走 `connection.async_get_bytes()`
- `global_http_connection` 底层复用 `aiohttp.ClientSession(reuse_client=True)`
- 图片下载完成后，再将 `media_io.load_bytes` 放入线程池执行后续解码

结论如下：

- 对“仅统计下载耗时”的 benchmark 而言，原生 vLLM 的基线等价于：
  - 单进程
  - `asyncio` 并发多图下载
  - 共享 `aiohttp.ClientSession`

### 3.2 download-only benchmark 最终可信结论

为验证“单进程异步下载”与“多核多进程下载器”之间的差异，远端服务器上完成了一次 download-only benchmark。

测试条件：

- 图片：`13` 张 `288 x 512 JPEG`
- 预热：`3` 次
- 采样：`5` 次
- 基线：单进程 `asyncio + aiohttp`，复用 `ClientSession`
- 对比方案：`13` 个绑核常驻 worker，多进程下载
- 统计口径：多进程方案每轮 latency 取 `13` 个 worker 内部下载耗时的最大值

最终可信结果：

| 指标  | async 基线  | 13 进程绑核下载 |
| --- | ---------:| ---------:|
| avg | 20.616 ms | 11.220 ms |
| min | 19.201 ms | 9.627 ms  |
| max | 23.029 ms | 13.002 ms |
| p50 | 20.538 ms | 11.036 ms |
| p95 | 22.642 ms | 12.694 ms |

多核并发观测结果：

- `13/13` worker 绑核成功
- `independent_cpu_binding = true`
- 实际运行 CPU 覆盖 `0..12`
- 每个 worker 的 `observed_processors` 与 `assigned_cpu` 一致
- 在 `3` 秒观察窗口内完成 `4286` 次下载

结论如下：

- 将多图下载从原生单进程异步下载器拆出，采用多核多进程独立下载器，在该场景下有明确收益
- 此结论仅证明“下载阶段”收益成立，不等价于端到端多模态全链路收益已成立

### 3.3 方案设计约束

在后续正式方案中，以下约束已被明确确认：

- `PD 多 P` 场景下，优先采用“每个 prefill 节点本地一个 sidecar”的设计
- `V1` 暂不做全局共享 sidecar
- `V1` 暂不使用 `shm`
- `V1` 暂不落盘，优先使用 CPU 内存缓存池
- `V1` 保留 `TP worker fallback`
- `TP worker fallback` 完成后，不将结果 publish 回 sidecar
- `V1` 仅覆盖 image 的 CPU preprocess，不做 `image_embeds`
- `V1` 不考虑视频

---

## 4. 历次迭代与决策链路

### 4.1 第一阶段：从服务验证转向多模态链路拆解

最初目标是确保 `vLLM + Qwen3/3.5-0.8B` 服务能够正常启动、可通过 `curl` 请求访问，并支持较长文本与多图输入。随着服务可用性逐步建立，关注重点转向如下问题：

- API server 多模态处理是否会成为瓶颈
- 多图下载、解码、processor 是否与 tokenizer 串行，导致吞吐下降
- 如何在不修改源码的前提下完成精细 profiling

### 4.2 第二阶段：提出并讨论并行化方向

先后提出过如下并行化方向：

- API server 图像处理与 tokenizer 并行
- API server 反序列化与图像处理并行
- 引入独立 `mm-sidecar` 服务，将多图下载、读取、预处理完全从 API server 脱离
- 将图片任务分发给多进程、绑核的独立 worker，以获得多核加速

此阶段形成的核心判断：

- 仅在 API server 内做轻量异步优化，收益有限
- 将多图处理从 API server 中剥离为 sidecar，是更有潜力的方向

### 4.3 第三阶段：围绕 sidecar 的职责划分展开分歧

围绕 `mm-sidecar` 的职责与形态，曾出现多种设计分支。

#### 4.3.1 临时思路：以 `h,w probe` 为中心的 metadata 先行

曾提出：为了让 engine 提前感知 token 规模，可由某一侧尽早获得图片 `h,w`，再推导 `image_grid_thw` 或调度 token 数。

此思路的出发点是：

- API server 不应等待完整 preprocess 完成后再让 engine 感知请求规模
- `h,w` 看似是最轻量的提取信息

后续问题逐步暴露：

- `h,w` 仅是原图尺寸，不等于 processor 后的调度合同
- 对 Qwen3/3.5 VL，这可能无法保证视觉 token 数、`mrope` 输入与实际 processor 结果完全一致
- 对 `base64` 场景，`h,w` 也不一定能以极低成本尽早获取

当前结论：

- “仅依赖 `h,w` 提前调度”不作为最终定稿方案
- 是否需要扩展为更完整 metadata/manifest，保留为待测量确认项

#### 4.3.2 临时思路：sidecar 采用 ingest lane / probe lane / processor lane 多段式流水

曾提出将 sidecar 拆分为：

- source ingestion lane
- metadata probe lane
- processor lane

此设计意图在于：

- 下载/解析头信息尽快返回
- 重 CPU preprocess 在后续阶段完成

后续讨论中暴露的问题：

- `base64` 未必能在极轻量阶段就稳定拿到必要信息
- source lane 与 processor lane 之间存在额外跨进程搬运
- manager 或 source executor 容易成为单核瓶颈

当前结论：

- 多 lane 设计不作为 `V1` 主设计
- `V1` 更偏向 manager 控制面 + worker 数据面闭环

#### 4.3.3 临时思路：source executor 与 processor executor 分离

曾讨论将下载与 processor 分别放入不同执行器或不同核。

反对理由包括：

- source executor 仍可能成为下载瓶颈
- 下载后还要跨进程传给 processor，增加额外成本
- 对小图场景，多一次搬运可能抵消收益

当前结论：

- `V1` 不做 source / process 分离执行器
- 下载、读取、解码、processor 在同一 worker 内连续完成

### 4.4 第四阶段：fallback、owner、缓存与 publish 路径收敛

随着 sidecar 方案逐渐成形，讨论重点转向：

- sidecar 慢、排队、卡死时如何处理
- `TP worker fallback` 是否必须保留
- sidecar 与 worker 如何避免重复下载、重复 preprocess
- worker 做完后是否回传 sidecar

形成的关键结论：

- `TP worker fallback` 必须保留
- 但 fallback 不能是“任何时刻各 rank 自己重做一遍”的松散逻辑
- 需要用 `lease / CAS / owner / epoch` 约束 fallback
- `TP worker fallback` 只服务当前请求，不 publish 回 sidecar

### 4.5 第五阶段：manifest 思路被提出，但暂不直接定稿

在更高层次复盘中提出：sidecar 不应只返回 `h,w`，而应返回更完整的调度合同，例如：

- `grid_thw`
- placeholder token count
- processor signature
- payload shape / dtype
- item identity

该方向的理论优势非常明显，但用户明确要求：

- 该建议暂不直接定稿
- 必须通过真实耗时对比测试，确认其额外成本是否合理，再决定是否纳入 `V1`

当前结论：

- 完整 manifest 方向保留
- 作为“待测量确认”的增强项，不直接写死为 `V1` 必选

### 4.6 已废弃或暂不采用方案清单

| 方案                                      | 处理结论    | 原因                    |
| --------------------------------------- | ------- | --------------------- |
| 仅靠 `h,w` 作为最终调度依据                       | 暂不定稿    | 可能与真实 processor 结果不一致 |
| 统一先落盘再由 processor worker 读取             | `V1` 放弃 | 额外 I/O 成本高，抵消收益风险大    |
| `shm` 作为 `V1` 主数据通道                     | `V1` 放弃 | 生命周期难控，易导致内存泄漏、踩踏与爆内存 |
| manager 参与大对象搬运                         | 放弃      | 易成为控制面与数据面的双重瓶颈       |
| sidecar 完成 raw data 后暴露 `raw_ready` 给外部 | 放弃      | 对调度无直接价值，状态机复杂化       |
| worker 完成 fallback 后 publish 回 sidecar  | 放弃      | 抢占 TP worker CPU，收益不足 |
| `V1` 直接做全局跨机 sidecar 共享                 | 放弃      | 正确性、缓存、一致性与网络复杂度过高    |
| 取消 `TP worker fallback`                 | 放弃      | sidecar 会退化为隐式阻塞依赖    |

---

## 5. 当前定稿方案总览

## 5.1 方案定位

当前最终定稿的 `V1` 方案定位如下：

> `mm-sidecar v1` 是部署在每个 prefill 节点本地的、面向 image CPU 预处理的辅助服务。其目标是在不破坏 `engine` 调度、`ViT DP`、`TP worker` 既有职责划分的前提下，将多图下载、读取、解码、CPU preprocess 从 API server 热路径中部分解耦，并通过受控的 `TP worker fallback` 保证请求在 sidecar 未及时完成时仍能继续执行。

### 5.2 关键原则

- 正确性优先于激进并行化
- sidecar 是增强链路，不得成为单点阻塞依赖
- manager 只做控制面
- worker 负责完整数据面
- `V1` 不落盘
- `V1` 不用 `shm`
- `V1` 保留 `TP worker fallback`
- `V1` 仅覆盖 image CPU preprocess
- `PD 多 P` 场景下，每个 prefill 节点本地一个 sidecar

### 5.3 V1 与 V1.5 的边界

#### V1

- 本地 sidecar
- CPU memory cache pool
- image CPU preprocess
- `TP worker fallback`
- 无 `shm`
- 无全局跨机共享缓存
- 无 `image_embeds`
- 无视频支持

#### V1.5

- 在 `V1` 正确性、收益、时延画像稳定后，再讨论：
  - 更完整 manifest
  - scheduler-aware readiness
  - 更细粒度 admission 策略
  - 更优零拷贝数据通道

---

## 6. 模块拆解

### 6.1 API Server

API server 在 `V1` 中承担以下职责：

- 接收多模态请求
- 解析文本与多模态输入描述
- 为每个 image 构建 `fallback_descriptor`
- 异步向 sidecar manager 发起 `PREPARE`
- 将 sidecar 句柄、descriptor 与 processor fingerprint 附加到请求上下文
- 不参与大对象搬运
- 不参与实际图像下载与 preprocess

API server 在 `V1` 中不承担以下职责：

- 不负责完整图片下载
- 不负责图像 decode
- 不负责 HF processor
- 不决定最终 fallback

### 6.2 Sidecar Manager

manager 是控制面组件，职责如下：

- 接收 `PREPARE`
- 为每个 image 创建状态记录
- 维护 `cache_key / lease / epoch / owner / queue / TTL / metrics`
- 选择 sidecar worker
- 为 `TP worker` 提供批量状态查询
- 提供 `TRY_FALLBACK_CLAIM`
- 执行 owner 切换与 CAS

manager 明确不承担以下职责：

- 不下载图片
- 不 decode
- 不执行 processor
- 不搬运 `pixel_values`

### 6.3 Sidecar Worker

worker 是数据面执行单元，职责如下：

- 读取本地文件
- 下载 HTTP 图像
- 解码 base64 图像
- 图片 decode
- 执行 HF processor CPU preprocess
- 产出 `pixel_values + image_grid_thw`
- 写入 CPU 内存缓存池
- 向 manager 汇报状态与结果元信息

worker 设计原则：

- 单张图完整数据链在同一 worker 进程内闭环
- 避免 source / processor 分离造成额外 IPC
- 通过多进程 + 绑核方式获得并发能力

### 6.4 Engine Core

`V1` 方案中，engine core 不承担 sidecar 内部逻辑，不直接参与 owner 决策，不负责 fallback。其职责仅限于：

- 透传轻量 metadata
- 驱动请求继续进入既有调度流程

### 6.5 TP Worker

TP worker 在 `V1` 中承担关键职责：

- 到达视觉消费点时，批量查询 sidecar 状态
- 作为最终 fallback 决策触发点
- 在满足条件时执行 `TRY_FALLBACK_CLAIM`
- 对 claim 成功的图执行本地读图 / 下载 / decode / preprocess

特别说明：

- `TP worker fallback` 是 `V1` 的必要兜底能力
- 但它必须是**受限、协作式、批量化、有 lease/CAS 约束**的 fallback

### 6.6 Request Coordinator Rank

在 TP / ViT DP 多 rank 场景下，必须有一个 `request coordinator rank` 统一决定：

- 哪些图使用 sidecar 结果
- 哪些图 fallback
- 哪个 rank 负责生成 fallback 结果

此组件是 `V1` 中保证多 rank 一致性的关键，不可省略。

---

## 7. 三类输入的统一处理设计

## 7.1 local file

### 输入形式

- 本地路径

### sidecar 主路径

- worker 直接读取本地文件
- decode 图像
- 执行 preprocess

### fallback 路径

- `TP worker` 本地读取同一路径
- 再执行 decode 与 preprocess

### 身份信息

当前建议：

- `path`
- `inode`
- `size`
- `mtime_ns`

若文件系统信息不可用，可退化为：

- `resolved_path`
- `size`
- `mtime_ns`

### 说明

`local file` 不应被排除在 fallback 之外。否则 sidecar 排队时，最容易本地完成的输入反而会被 sidecar 阻塞。

## 7.2 HTTP

### 输入形式

- `url`
- headers
- timeout policy
- redirect policy

### sidecar 主路径

- sidecar worker 独立下载
- decode
- preprocess

### fallback 路径

- `TP worker` 在 claim 成功后自行下载
- decode
- preprocess

### 身份信息

当前建议分层如下：

- correctness 层：
  - `url`
  - 请求 headers
  - timeout / redirect policy
- best-effort dedupe 层：
  - `ETag`
  - `Last-Modified`
  - `Content-Length`
  - 可选头部摘要

### 说明

`HTTP` 是 sidecar 受益最明显的路径，但也是最容易引入排队与远端不确定性的路径，因此必须保留 fallback。

## 7.3 base64

### 输入形式

- 请求内 base64 编码图片

### sidecar 主路径

- manager 接收请求描述后，由 worker 执行 base64 decode
- decode 成图像对象
- preprocess

### fallback 路径

- worker 必须持有可恢复原始图像的 bytes 或等价 descriptor
- 在 claim 成功后本地完成 decode 与 preprocess

### 身份信息

当前不采用全量内容 hash 作为 `V1` 主策略。

当前建议：

- correctness 层：请求级 descriptor 强绑定，不做跨请求强缓存
- best-effort dedupe 层：
  - `mime`
  - `encoded_length`
  - prefix / suffix 片段摘要
  - 轻量 rolling hash

### 说明

对 `base64`，在不做全量 hash 前，不应将弱摘要当作强 correctness cache key 使用。`V1` 中更稳妥的做法是：

- 请求内可靠恢复
- 跨请求仅做 best-effort dedupe

---

## 8. 数据与状态设计

### 8.1 fallback_descriptor

每个 image 在 API server 侧必须生成一份完整 `fallback_descriptor`，供 sidecar 与 `TP worker fallback` 共享。

最小字段集合如下：

- `request_id`
- `request_media_index`
- `source_type`: `local_file | http | base64`
- `processor_fingerprint`
- `source_payload`
  - `local_file`: `path`, `size`, `mtime_ns`, `inode`
  - `http`: `url`, `headers`, `timeout_ms`, `allow_redirects`
  - `base64`: 原始压缩 bytes 或可恢复 bytes 的 request-scoped 描述
- `safety_limits`
  - 最大下载字节数
  - 最大 decode 尺寸
  - 最大处理耗时

### 8.2 processor_fingerprint

`processor_fingerprint` 是 `V1` 必须落地的最小一致性约束，用于保证 sidecar 与 fallback 使用相同 preprocess 配置。

当前必须纳入 fingerprint 的内容：

- processor 类名
- model revision
- image resize / crop / pad 策略
- interpolation
- normalization 配置
- dtype / layout 关键配置

### 8.3 manifest 范围的当前结论

是否在 `V1` 中直接返回完整 manifest，当前保持如下处理：

- **已确认必须返回**：
  - `image_grid_thw`
  - `pixel_values`
  - `processor_fingerprint`
  - `request_media_index`
- **保留为待测量确认项**：
  - placeholder token count
  - payload shape / dtype 扩展字段
  - 完整 item identity
  - 更完整 processor signature

原因如下：

- 完整 manifest 理论上更稳
- 但是否纳入 `V1`，需先对额外耗时进行真实测量，再决定是否写死为必需项

### 8.4 sidecar 状态机

`V1` 最终采用的外部状态机如下：

- `ABSENT`
- `QUEUED`
- `SIDECAR_RUNNING`
- `READY`
- `FAILED`
- `EXPIRED`
- `FALLBACK_CLAIMED`
- `FALLBACK_LOCAL_DONE`
- `BYPASS`

说明：

- 不对外暴露 `raw_ready`
- manager 内部可保留更细状态，但外部协作以以上状态为准

### 8.5 lease / CAS / epoch

每个 image 对应一个 `cache_key` 与一份 `lease`。

规则如下：

- 同一时刻只允许一个合法 producer owner
- sidecar worker 开始处理前必须持有 lease
- worker fallback 前必须申请 `TRY_FALLBACK_CLAIM`
- 成功 claim 后，epoch 增加
- sidecar worker 在后续 heartbeat 或阶段边界发现 epoch 被更新后，必须停止 publish 或丢弃结果
- manager 不可用时，允许 worker fail-open fallback，但结果仅在本请求内使用

---

## 9. CPU 内存缓存池设计

### 9.1 当前决策

`V1` 不落盘，不使用 `shm`，采用 CPU 内存缓存池。

### 9.2 缓存层次

推荐采用两层缓存：

#### inflight pool

- 用途：保存正在下载、decode、preprocess 中的对象
- 生命周期：请求相关，短 TTL
- 特点：强引用，禁止回收直到状态结束

#### reusable pool

- 用途：保存已完成的 `pixel_values + image_grid_thw`
- 生命周期：受总字节预算与 LRU 规则控制
- 特点：best-effort 复用

### 9.3 淘汰策略

当前采用：

- 以字节数为主的 LRU 或近似 LRU
- 优先回收 reusable pool
- inflight pool 仅在超时 / 失败 / 取消后释放

### 9.4 不采用落盘的原因

`V1` 不落盘的原因已明确如下：

- 小图场景下多一次磁盘或文件系统 I/O 可能抵消 sidecar 收益
- 引入落盘会增加路径管理、清理、错误恢复复杂度
- 现阶段优先验证 sidecar 架构正确性与 CPU preprocess 收益

### 9.5 不采用 shm 的原因

`V1` 不采用 `shm` 的原因已明确如下：

- 反复申请与释放难管理
- 异常路径可能导致释放不及时
- 高并发下存在爆内存与踩踏风险
- 在未完成对象生命周期画像前，不适合引入复杂共享内存池

---

## 10. TP Worker Fallback 设计

### 10.1 最终结论

`TP worker fallback` 在 `V1` 中保留，且为必要能力。

其保留原因如下：

- sidecar 与 tokenizer / engine admission / scheduling 是异步并行推进的
- 请求完全可能在 sidecar 尚未完成时已进入视觉消费点
- 若 worker 没有自救能力，sidecar 将退化为隐式阻塞依赖

### 10.2 fallback 的定位

`TP worker fallback` 不是常态主路径，而是**受限、协作式、有预算控制的兜底路径**。

### 10.3 允许 fallback 的条件

允许 fallback 的条件如下：

- 当前 media 是 `V1` 支持的 image
- `processor_fingerprint` 一致
- `fallback_descriptor` 完整可恢复
- worker 已到达视觉消费点
- sidecar 状态属于以下之一：
  - `ABSENT`
  - `QUEUED`
  - `SIDECAR_RUNNING`
  - `FAILED`
  - `EXPIRED`
  - manager `unavailable`
- 近 ready 等待预算已耗尽
- 当前请求尚未为该图固定最终来源

### 10.4 禁止 fallback 的条件

禁止 fallback 的条件如下：

- descriptor 不完整
- processor 配置不一致
- 某些 rank 已消费 sidecar 结果，而其他 rank 企图再切换为 fallback
- 当前图已确定来源，后续不可中途切换
- 当前 media 不是 `V1` 明确支持范围

### 10.5 fallback 决策点

最终 fallback 决策点固定为：

> **TP worker 的 request coordinator rank**

不是 API server，不是 manager，也不是每个 rank 各自独立做决定。

### 10.6 多图批处理规则

多图场景必须批量处理，不允许逐图串行等待。

流程如下：

1. `BATCH_GET_STATUS(media_handles)`
2. 对 `READY` 图并发拉取
3. 对 `QUEUED / SIDECAR_RUNNING` 图进行一次短等待
4. 再次批量检查
5. 对仍未 ready 的图批量 `TRY_FALLBACK_CLAIM`
6. claim 成功的图，由指定 rank 执行本地 fallback

### 10.7 near-ready 等待预算

`V1` 当前固定如下建议：

- `GET_STATUS_BATCH` 必须使用极短 RPC timeout
- near-ready 等待窗口不超过 `2 ms`
- 只允许一次极短等待，不允许无限轮询

### 10.8 request-level source plan

同一请求中，允许一部分图使用 sidecar，另一部分图使用 fallback，但必须先生成统一的 `source plan`。

示例：

```text
media_0 -> USE_SIDECAR
media_1 -> FALLBACK(producer_rank=2)
media_2 -> USE_SIDECAR
```

该 `source plan` 一经生成，不允许中途切换。

### 10.9 多 rank 一致性规则

多 rank 场景下必须遵守以下规则：

- 不允许每个 rank 独立决定 fallback
- 必须由 coordinator rank 统一生成 `source plan`
- 每张图只允许一个 fallback producer rank
- 其余 rank 只消费结果，不重复 preprocess

### 10.10 fallback 结果的使用边界

fallback 结果只允许：

- 服务于当前请求

fallback 结果不允许：

- publish 回 sidecar
- 写入跨请求共享缓存

---

## 11. PD 多 P 场景设计

### 11.1 已定稿原则

`PD 多 P` 场景下，sidecar 采用如下原则：

- 每个 prefill 节点本地部署一个 sidecar
- 不做跨机全局 sidecar 共享
- 不做全局统一缓存

### 11.2 设计原因

原因如下：

- 跨机共享 sidecar 会显著增加 payload 传输复杂度
- 缓存一致性与 lease 语义更难保证
- 首版更应优先验证“单节点本地 sidecar”的端到端收益

### 11.3 缓存边界

在 `PD 多 P` 场景下：

- sidecar 缓存是**节点本地缓存**
- 不要求全局命中
- 跨节点重复处理在 `V1` 中是可接受的

### 11.4 传输边界

当前不采用：

- 一个中心 sidecar 服务多个 prefill 节点
- 跨机共享 `pixel_values`

当前采用：

- 请求路由到哪个 prefill 节点，就由该节点本地 sidecar 负责处理与缓存

---

## 12. 流程逻辑

### 12.0 全局时序字符图总览

以下时序图使用统一参与方命名：

- `Client`：请求发起方
- `API`：API server
- `Mgr`：sidecar manager
- `SW[i]`：第 `i` 个 sidecar worker
- `EC`：engine core / scheduler
- `Coord`：TP worker request coordinator rank
- `TP-Rk`：普通 TP / ViT DP rank

说明：

- 时序图中的 `Img[k]` 表示请求中的第 `k` 张图
- `FP` 表示 `processor_fingerprint`
- `FD` 表示 `fallback_descriptor`
- `SG` 表示 sidecar handle / state handle
- `Plan` 表示 request-level source plan

### 12.0.1 主路径时序图：sidecar 提前完成，TP worker 直接消费

```text
Client                      API                         Mgr                         SW[0..N]                    EC                     Coord / TP-Ranks
  |                          |                           |                             |                          |                           |
  |---- HTTP Request ------->|                           |                             |                          |                           |
  |                          |-- parse text ----------->|                             |                          |                           |
  |                          |-- parse media refs ------|                             |                          |                           |
  |                          |-- build FD[Img0..12] ---|                             |                          |                           |
  |                          |-- calc/request FP ------|                             |                          |                           |
  |                          |-- async PREPARE(req, FD, FP) ------------------------->|                          |                           |
  |                          |                           |-- create state per image -->|                          |                           |
  |                          |                           |   state=QUEUED             |                          |                           |
  |                          |                           |-- assign worker ---------->|                          |                           |
  |                          |-- continue tokenizer ----|                             |                          |                           |
  |                          |                           |                             |-- acquire lease -------->|                          |
  |                          |                           |                             |   state=SIDECAR_RUNNING |                          |
  |                          |                           |                             |-- local/http/base64 read|                          |
  |                          |                           |                             |-- image decode ---------|                          |
  |                          |                           |                             |-- HF preprocess --------|                          |
  |                          |                           |                             |-- produce pixel_values  |                          |
  |                          |                           |<--------- READY(meta) -----|                          |
  |                          |<------ SG handles -------|                             |                          |                           |
  |                          |------------------------------ request enters engine ----------------------------->|                           |
  |                          |                                                                                  |-- schedule request ------>|
  |                          |                                                                                  |                           |
  |                          |                                                                                  |<-- at vision consume point|
  |                          |                                                                                  |                           |
  |                          |                                                                                  |-- BATCH_GET_STATUS(SG) -->|
  |                          |                                                                                  |<-- all READY ------------|
  |                          |                                                                                  |-- build Plan ----------->|
  |                          |                                                                                  |   Img0..12=USE_SIDECAR   |
  |                          |                                                                                  |-- broadcast Plan ------->|
  |                          |                                                                                  |-- FETCH_RESULT batch --->|
  |                          |                           |---------------- route by owner ---------------------->|                           |
  |                          |                           |<--------------- result meta/ptr ---------------------|                           |
  |                          |                                                                                  |-- consume tensors ------>|
  |<----------------------------------------------- normal model response --------------------------------------|                           |
```

关键约束：

- API 不执行大对象图像处理，仅构建 `FD + FP + SG`
- manager 只做状态、lease、路由
- worker 在单进程内完成读取 / 下载 / decode / preprocess
- 若全部图在视觉消费点前变为 `READY`，本请求完全不进入 fallback

### 12.0.2 near-ready 短等待分支时序图

```text
Coord                         Mgr                         SW[x]                         TP-Ranks
  |                            |                            |                              |
  |-- BATCH_GET_STATUS ------->|                            |                              |
  |<-- Img0 READY             -|                            |                              |
  |    Img1 SIDECAR_RUNNING   -|                            |                              |
  |    Img2 QUEUED            -|                            |                              |
  |                            |                            |                              |
  |-- start near-ready timer (<= 2 ms) -------------------->|                              |
  |                            |                            |-- continue decode/preprocess |
  |                            |<--------- Img1 READY ------|                              |
  |-- BATCH_GET_STATUS(recheck)->|                          |                              |
  |<-- Img0 READY, Img1 READY, Img2 still QUEUED ----------|                              |
  |-- build Plan ------------->|                            |                              |
  |   Img0 USE_SIDECAR         |                            |                              |
  |   Img1 USE_SIDECAR         |                            |                              |
  |   Img2 unresolved          |                            |                              |
  |-- broadcast partial Plan ------------------------------------------------------------->|
```

关键约束：

- `QUEUED / SIDECAR_RUNNING` 只允许一次极短 near-ready 窗口
- 不允许长轮询、无限等待或排队等待 sidecar
- near-ready 结束后，缺失子集必须立即进入 claim/fallback 分支

### 12.0.3 TP worker fallback 分支时序图

```text
Coord                         Mgr                         SW[y]                        Fallback Producer Rank          Other TP-Ranks
  |                            |                            |                                  |                             |
  |-- BATCH_GET_STATUS ------->|                            |                                  |                             |
  |<-- Img5 RUNNING           -|                            |                                  |                             |
  |    Img6 QUEUED            -|                            |                                  |                             |
  |                            |                            |                                  |                             |
  |-- near-ready wait <=2ms ->|                            |                                  |                             |
  |-- recheck status --------->|                            |                                  |                             |
  |<-- Img5 RUNNING           -|                            |                                  |                             |
  |    Img6 QUEUED            -|                            |                                  |                             |
  |                            |                            |                                  |                             |
  |-- TRY_FALLBACK_CLAIM([Img5,Img6], deadline, req_id) -->|                                  |                             |
  |                            |-- CAS/lease/epoch check -->|                                  |                             |
  |                            |   if claim granted:         |                                  |                             |
  |<-- claim ok:              -|   Img5 -> producer rank 2  |                                  |                             |
  |    Img5 FALLBACK_CLAIMED   |   Img6 -> producer rank 2  |                                  |                             |
  |    Img6 FALLBACK_CLAIMED   |   epoch=N+1                |                                  |                             |
  |-- build Plan ------------->|                            |                                  |                             |
  |   Img5 FALLBACK(rank=2)    |                            |                                  |                             |
  |   Img6 FALLBACK(rank=2)    |                            |                                  |                             |
  |-- broadcast Plan ------------------------------------------------------------------------------------------->|
  |                                                                 |                                  |                             |
  |                            |                            |-- heartbeat/stage boundary ------>|                             |
  |                            |                            |   sees epoch changed              |                             |
  |                            |                            |   drop/abort publish             |                             |
  |                                                                 |                                  |                             |
  |----------------------------------------------------------------------------------------------> read/download |
  |                                                                                                  decode       |
  |                                                                                                  preprocess   |
  |                                                                                                  build local  |
  |                                                                                                  artifact     |
  |<---------------------------------------------------------------------------------------------- local ready  |
  |-- distribute/route current-request result --------------------------------------------------------------->|
  |----------------------------------------------------------------------------------------------------------> consume
```

关键约束：

- fallback 前必须先 `TRY_FALLBACK_CLAIM`
- claim 成功后，sidecar 原 owner 不得再合法 publish
- fallback 结果只服务当前请求，不回写 sidecar
- 同一图只允许一个 fallback producer rank

### 12.0.4 manager 异常时的 fail-open fallback 时序图

```text
Coord                         Mgr                           Fallback Producer Rank          Other TP-Ranks
  |                            |                                      |                             |
  |-- BATCH_GET_STATUS ------->|                                      |                             |
  |<-- timeout / unavailable --|                                      |                             |
  |                            |                                      |                             |
  |-- mark manager_unavailable |                                      |                             |
  |-- build emergency Plan --->|                                      |                             |
  |   unresolved imgs ->       |                                      |                             |
  |   FAIL-OPEN FALLBACK       |                                      |                             |
  |-- broadcast Plan ------------------------------------------------------------------------------->|
  |------------------------------------------------------------------------------------------------> read/download
  |                                                                                                  decode
  |                                                                                                  preprocess
  |<------------------------------------------------------------------------------------------------ local ready
  |--------------------------------------------------------------------------------------------------------------> consume
```

关键约束：

- manager 不可用时允许 fail-open fallback
- 该路径下不做共享 lease 一致性保证
- 结果严格限定为本请求内使用
- 不写 sidecar cache，不尝试回填 manager

### 12.0.5 PD 多 P 本地 sidecar 部署拓扑时序图

```text
Client
  |
  |---- Request A ----> Prefill Node P0: API/EC/TP + local sidecar(Mgr0, SW0..n)
  |                        |-- sidecar prepare on P0 ----------------------------------------------|
  |                        |-- tensors consumed on P0 ---------------------------------------------|
  |
  |---- Request B ----> Prefill Node P1: API/EC/TP + local sidecar(Mgr1, SW0..n)
                           |-- sidecar prepare on P1 ----------------------------------------------|
                           |-- tensors consumed on P1 ---------------------------------------------|

No V1 path:
  P0 sidecar  ----X---- share pixel_values/image_grid_thw/cache with P1 sidecar
  P1 sidecar  ----X---- share global cache/lease with P0 sidecar
```

关键约束：

- `V1` 不做全局 sidecar，不做跨机共享 `pixel_values`
- 请求路由到哪个 prefill 节点，就由该节点本地 sidecar 处理与缓存
- 跨节点重复处理在 `V1` 中可接受

### 12.1 sidecar 主流程

1. API server 收到请求
2. 解析文本输入与多模态输入
3. 为每张图构建 `fallback_descriptor`
4. 异步向 sidecar manager 发 `PREPARE`
5. manager 为每张图分配 worker、创建状态与 lease
6. worker 执行读取 / 下载 / decode / preprocess
7. worker 产出 `pixel_values + image_grid_thw`
8. worker 将结果写入 CPU 内存缓存池
9. manager 将状态更新为 `READY`
10. TP worker 到视觉消费点时查询并消费结果

### 12.2 fallback 流程

1. TP worker 到达视觉消费点
2. coordinator rank 批量查询所有图状态
3. 对 `READY` 图并发拉取
4. 对 `QUEUED / SIDECAR_RUNNING` 图短等待后复查
5. 对仍未 ready 的图发 `TRY_FALLBACK_CLAIM`
6. claim 成功的图由指定 producer rank 执行本地读图 / 下载 / decode / preprocess
7. producer rank 将 fallback 结果供当前请求内相关 rank 消费
8. fallback 结果不回写 sidecar

### 12.3 manager 异常流程

若 manager 不可用：

- 允许 `TP worker` fail-open fallback
- 该 fallback 结果严格限制为本请求内使用
- 不进行共享缓存写入

---

## 13. 参数与规则建议

### 13.1 时间预算建议

当前 `V1` 建议参数如下：

- sidecar 状态查询超时：极短 RPC timeout
- near-ready 等待窗口：`<= 2 ms`
- fallback claim 超时：极短
- fallback 本地处理：受请求剩余 deadline 约束

### 13.2 队列规则建议

- manager 只维护任务队列，不处理数据
- worker 负载均衡依据：
  - 当前运行任务数
  - 已占用内存估计
  - 近期平均处理耗时
- `V1` 优先使用简单、可观测的调度规则，避免引入复杂优先级系统

### 13.3 缓存规则建议

- correctness cache 与 best-effort dedupe 分离
- 无法保证强身份一致性的输入，不进入跨请求强缓存
- reusable cache 以总字节预算为第一淘汰条件

---

## 14. 风险说明

### 14.1 调度合同不精确风险

若仅使用过于简化的 metadata，可能导致：

- 视觉 token 数与真实 processor 结果不一致
- 调度错误
- prefix cache key 不一致

该风险当前尚未完全消除，因此“完整 manifest 是否纳入 `V1`”保留为待测量确认项。

### 14.2 sidecar 排队导致 fallback 频繁触发

若 sidecar 容量不足：

- fallback 频率会显著升高
- TP worker 可能承受额外 CPU 压力

### 14.3 worker fallback 带来的 CPU 竞争

即使 fallback 是必要能力，也会带来：

- TP worker 侧额外 CPU 占用
- 推理期与 preprocess 竞争 CPU 资源

因此 fallback 必须受预算与状态机约束，不能泛化为常态路径。

### 14.4 无 shm / 无落盘带来的内存压力风险

`V1` 选择 CPU 内存缓存池可以降低实现复杂度，但会带来：

- 内存占用增长更快
- 高峰期可能需要更严格的 LRU 与 inflight 限制

### 14.5 manager 异常时的一致性风险

manager 是控制面中心。一旦异常：

- worker 可能需要 fail-open fallback
- 共享 lease 一致性会下降

因此 manager 需要良好的指标与健康检查机制。

---

## 15. 待优化与待确认项

### 15.1 必须后续实测确认的项

以下内容已明确列为“待测量确认”，不在本轮直接写死：

- `h,w`、`grid_thw`、完整 manifest 三种返回粒度的耗时对比
- placeholder token count 是否应纳入 `V1` 必选 metadata
- payload shape / dtype 扩展字段的必要性

### 15.2 后续可优化项

- 更精确的 worker 负载感知调度
- 更细粒度的 queue priority
- sidecar 与 TP worker 的更轻量 IPC
- 更可靠的 base64 身份策略
- `V1.5` 中更完整 manifest

### 15.3 当前未展开的项

- 端到端 monkey patch profiling 方案的具体实现
- `engine core` 与 `TP worker` 函数级热点画像
- `encoder cache` 在特定部署下的最终位置、复用与命中边界

---

## 16. 最终落地结论

### 16.1 当前正式定稿方案

当前正式定稿的 `mm-sidecar v1` 方案如下：

- 每个 prefill 节点本地部署一个 sidecar
- sidecar 只负责 image CPU preprocess
- manager 仅做控制面
- worker 在单进程内完成下载 / 读取 / decode / preprocess
- `V1` 不落盘
- `V1` 不使用 `shm`
- `V1` 使用 CPU 内存缓存池
- API server 生成 `fallback_descriptor`
- `TP worker fallback` 保留
- fallback 决策由 `request coordinator rank` 统一拍板
- fallback 结果不回写 sidecar

### 16.2 V1 不做的事项

当前明确不纳入 `V1` 的内容如下：

- 视频 sidecar 化
- `image_embeds`
- 全局跨机 sidecar 共享
- worker fallback 后回写 sidecar
- `shm`
- 统一先落盘
- 无约束的每 rank 独立 fallback

### 16.3 V1 的实现门槛

`V1` 实现前必须具备以下基础：

- 明确的 `fallback_descriptor`
- 明确的 `processor_fingerprint`
- 可观测的状态机与 lease/CAS
- CPU 内存缓存池的字节预算与回收策略
- 多 rank `source plan` 协调逻辑

### 16.4 V1.5 升级条件

仅当以下条件满足时，再进入 `V1.5`：

- `V1` 正确性稳定
- benchmark 与线上画像证明 sidecar 有明显收益
- fallback 频率与额外 CPU 成本处于可接受范围
- sidecar 的队列与内存行为已被充分观测

---

## 17. 附录：本轮最终决策摘要

### 17.1 已确认

- 原生 vLLM 多图 HTTP 下载基线是单进程 `asyncio + shared aiohttp session`
- 多核多进程下载器对 download-only 有明确收益
- `PD 多 P` 下 sidecar 先按“每个 prefill 节点本地一个”设计
- `V1` 不落盘
- `V1` 不用 `shm`
- `V1` 保留 `TP worker fallback`
- `TP worker fallback` 不回写 sidecar

### 17.2 暂不定稿，待测量确认

- `V1` 是否直接返回完整 manifest
- metadata 的最小必要集合究竟应是：
  - `h,w`
  - `grid_thw`
  - 还是更完整合同

### 17.3 已废弃

- 仅靠 `h,w` 即作为最终调度依据
- 统一先落盘再由 worker 读取
- `V1` 直接使用 `shm`
- 取消 worker fallback
- worker fallback 后 publish 回 sidecar

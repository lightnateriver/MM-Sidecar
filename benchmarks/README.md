# Benchmarks

本目录用于归档与 `mm-sidecar-with-phase` 设计相关的 benchmark 与观测结论。

当前目录内脚本均已放入本工程维护，不依赖外部 skill 目录或 `api server opt` 工程路径。

## 当前已确认结果

### download-only 基准

- 场景：`13 x 288x512 JPEG`
- 基线：单进程 `asyncio + aiohttp + shared ClientSession`
- 对比：`13` 个绑核常驻 worker，多进程下载
- 预热：`3` 次
- 采样：`5` 次

说明：

- 该 benchmark 是本次讨论中的历史验证样例
- `13` 图 / `13` worker 不代表正式方案默认部署参数
- 正式方案当前默认建议为：`32` 个绑核 processor worker，目标覆盖 `1~40` 图场景

最终可信结果：

| 模式 | avg | min | max | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| async | 20.616 ms | 19.201 ms | 23.029 ms | 20.538 ms | 22.642 ms |
| bind-cpu multiprocess | 11.220 ms | 9.627 ms | 13.002 ms | 11.036 ms | 12.694 ms |

观测结论：

- `13/13` worker 绑核成功
- 实际运行 CPU 覆盖 `0..12`
- `3` 秒内总下载次数 `4286`
- 说明“多核多进程下载器”相对原生单进程异步下载基线具有明确收益

## 说明

- 上述 benchmark 仅验证下载阶段收益，不直接等价于端到端多模态收益。
- 端到端收益仍需在后续将 sidecar 真正接入请求链路后继续验证。
- `base64` 明显快于 `http` 的批量结果，不能直接解释为 processor 纯计算更快；它更可能反映了 `http` 场景里 transport、server queueing 与 connection setup 的额外成本。
- 当前补充脚本：
  - `benchmark_chat_service.py`
    - 串行回放数据集并统计 `TTFT / TPOT / E2E`
  - `generate_real_multimodal_dataset.py`
    - 构造 `10k` 文本 token + 多图的真实图片数据集

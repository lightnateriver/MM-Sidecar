# mm-sidecar-with-phase

本仓库用于沉淀 `mm-sidecar + phase` 方向的正式设计文档、决策记录、运行期 monkey patch 原型与 benchmark 工具。

## 新 Agent 接手指南

如果你是新接手本仓库的 agent，建议按这个顺序阅读：

1. [docs/strict_benchmark_summary_2026-06-24.md](docs/strict_benchmark_summary_2026-06-24.md)
   - 当前最重要的阶段性总结。
   - 记录 strict benchmark 协议、baseline/patched/affinity 结果、绑核方式、当前瓶颈判断和下一步优先级。
2. [docs/mm_sidecar_with_phase_formal_design.md](docs/mm_sidecar_with_phase_formal_design.md)
   - 正式方案设计文档。
   - 用于理解整体架构、阶段边界、废弃方案和长期设计目标。
3. [src/mm_sidecar/sidecar/README.md](src/mm_sidecar/sidecar/README.md)
   - sidecar manager / processor worker / service runtime 的目录说明。
4. [src/mm_sidecar/integrations/vllm_patch/README.md](src/mm_sidecar/integrations/vllm_patch/README.md)
   - vLLM monkey patch 集成层说明。
5. [benchmarks/README.md](benchmarks/README.md)
   - benchmark 脚本背景和历史结果说明。

当前工程目录：

```text
/mnt/d/project/mm-sidecar/mm-sidecar-with-phase
```

远端运行目录：

```text
/root/mm-sidecar-with-phase
```

远端 benchmark 产物目录：

```text
/root/mm-sidecar-e2e
```

本地 `tmp/` 只用于临时试验记录、复制的结果表和一次性分析脚本，不应视为稳定生产代码。

## 当前内容

- [docs/mm_sidecar_with_phase_formal_design.md](docs/mm_sidecar_with_phase_formal_design.md)
  - 基于本次完整 Session 形成的正式方案设计文档。
  - 覆盖背景目标、迭代过程、废弃方案、最终方案、规则定义、风险与待确认项。
- [docs/strict_benchmark_summary_2026-06-24.md](docs/strict_benchmark_summary_2026-06-24.md)
  - 当前 strict benchmark、NUMA affinity 测试和性能瓶颈分析的阶段性总结。
- `src/mm_sidecar/contracts/`
  - sidecar 与 patch 共用的合同层对象。
- `src/mm_sidecar/integrations/vllm_patch/`
  - 放在本工程内的 `vllm` API server monkey patch 与 launcher。
  - 运行时不依赖外部 `api server opt` 工程路径。
- `src/mm_sidecar/sidecar/`
  - 阶段 C 的 sidecar manager、processor worker、状态机与 CPU 内存缓存池。
- `benchmarks/`
  - 与当前方案配套的数据集生成、服务压测与合同层验证脚本。

## 关键代码入口

- `src/mm_sidecar/sidecar/service.py`
  - sidecar 独立服务、client、worker 配置、默认 worker CPU affinity map。
- `src/mm_sidecar/sidecar/manager.py`
  - sidecar 状态机、`prepare()`、`wait_for_metadata()`、`fetch_ready()`。
- `src/mm_sidecar/sidecar/processor.py`
  - processor worker pool、worker 绑核、图片读取/解码/预处理。
- `src/mm_sidecar/integrations/vllm_patch/patches.py`
  - vLLM API server monkey patch 安装入口和 request capture middleware。
- `src/mm_sidecar/integrations/vllm_patch/sidecar_bridge.py`
  - API server 侧 descriptor capture、sidecar prepare、metadata wait、debug payload。
- `src/mm_sidecar/integrations/vllm_patch/worker_sidecar.py`
  - vLLM model worker 侧从 sidecar 拉取 artifact 并替换 multimodal feature data。
- `src/mm_sidecar/integrations/vllm_patch/api_fast_path.py`
  - descriptor-only / synthetic Qwen image fast path。
- `benchmarks/e2e_smoke_matrix.py`
  - strict end-to-end benchmark driver。

## 当前运行方式

远端 patched API server 通常通过本仓库 launcher 启动：

```bash
cd /root/mm-sidecar-with-phase
MM_SIDECAR_TRANSPORT=unix \
MM_SIDECAR_SOCKET_PATH=/tmp/mm-sidecar-e2e.sock \
MM_SIDECAR_WORKER_COUNT=32 \
MM_SIDECAR_WORKER_POOL_MODE=process \
MM_SIDECAR_ENABLE_DEBUG_ROUTE=1 \
MM_SIDECAR_ENABLE_API_FAST_PATH=1 \
MM_SIDECAR_WORKER_DEBUG=1 \
MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE=1 \
PYTHONPATH=/root/mm-sidecar-with-phase/src \
/root/miniconda3/bin/python -m mm_sidecar.integrations.vllm_patch.launcher \
  --model /autodl-fs/data/qwen3.5-0.8b \
  --served-model-name qwen3vl-0.8b \
  --host 127.0.0.1 \
  --port 18001 \
  --max-model-len 16384 \
  --trust-remote-code \
  --allowed-local-media-path /root/mm-sidecar-e2e \
  --max-num-seqs 1 \
  --no-enable-log-requests
```

NUMA-aware affinity 测试使用了 disjoint CPU ranges：

```text
mm processor workers: CPU 0-31, one worker per CPU
sidecar launcher/manager/listener: CPU 32-47
API server: CPU 96-143
All CPUs are on NUMA node0.
```

具体实现方式见 [strict_benchmark_summary_2026-06-24.md](docs/strict_benchmark_summary_2026-06-24.md) 的 `NUMA-Affinity Run` 小节。

## Strict Benchmark

当前可信 strict 协议：

```text
3 warmup runs + 5 measured runs per case
transports: local_path, http, base64
image counts: 1, 13, 20
每个 run 使用不同随机图片
不同 transport/case 之间也不复用图片
全矩阵生成 816 张随机图
```

示例命令：

```bash
cd /root/mm-sidecar-with-phase
/root/miniconda3/bin/python benchmarks/e2e_smoke_matrix.py \
  --host http://127.0.0.1:18001 \
  --model auto \
  --work-dir /root/mm-sidecar-e2e/strict_matrix_new \
  --out /root/mm-sidecar-e2e/strict_matrix_new.json \
  --transports local_path,http,base64 \
  --image-counts 1,13,20 \
  --warmup 3 \
  --runs 5 \
  --fetch-debug \
  --image-seed 20260625
```

已完成结果：

```text
/root/mm-sidecar-e2e/strict_matrix_baseline_r1.json
/root/mm-sidecar-e2e/strict_matrix_baseline_r2.json
/root/mm-sidecar-e2e/strict_matrix_r1.json
/root/mm-sidecar-e2e/strict_matrix_r2.json
/root/mm-sidecar-e2e/strict_matrix_affinity_r1.json
```

小的 markdown 表已复制到：

```text
tmp/strict_benchmarks/
```

## 当前结论

- Patched sidecar 明确改善 HTTP 多图场景，尤其 `http 13` 和 `http 20`。
- 单图场景仍明显慢，因为 sidecar 控制面固定开销盖过收益。
- `local_path` / `base64` 原生路径已经较快，当前统一走 sidecar 不一定划算。
- NUMA-aware affinity 对 `20` 图场景有效，但没有解决根因。
- worker 单图处理大约 `27-30 ms/image`，但 API server metadata wait 仍可到 `100ms+`。
- 当前证据表明额外 wait 主要在 sidecar 控制面，不是图像 preprocess 本身，也不是 metadata 阶段向 API server 返回大 tensor payload。

## 下一步优先级

1. 给 `wait_for_metadata()`、sidecar service RPC、worker result queue 增加 request-scoped timestamp。
2. 拆分 `probed` 和完整 preprocess，让 worker 在 `schedule_item` 生成后立即返回 metadata。
3. 对 `local_path/base64` 的单图或低收益场景增加 bypass / threshold policy。
4. 单独分析 model-worker `fetch_ready()` 的大 tensor payload 移动成本。
5. 保持 benchmark 严格协议，避免图片缓存或重复输入污染性能结论。

## 维护约定

- 本仓库当前以设计文档为主，后续实现、实验脚本、原型代码、benchmark 结果建议按专题增量提交。
- 设计文档中的 `V1`、`V1.5`、`待测量确认` 项为后续演进边界，新增实现应显式标注所对应阶段。
- 若后续补充 benchmark 或 profiling 结果，应优先在现有正式文档中追加“版本化附录”，避免产生多份相互冲突的方案说明。
- 与历史 `vllm-api-server-opt` 相关的可复用能力，应迁入本仓库后再继续维护，不再通过跨工程引用方式拼装运行链路。

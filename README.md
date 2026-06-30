# mm-sidecar-with-phase

`mm-sidecar-with-phase` 是当前主代码目录，承载 sidecar 服务、vLLM monkey patch、正式设计文档和 benchmark 工具。

根目录 `README.md` 只说明主代码如何使用、当前方案是什么、以及新接手的人应先看什么。实验过程、对照数据和阶段性结论统一放在 `docs/`。

## 当前状态

- 主代码已合入 early metadata return。
- 主代码已合入 metadata / ready split queue。
- API prepare 热路径里的同步统计与 preview 计算已移出主路径。
- sidecar 主代码已支持正式 CPU affinity 配置：
  `MM_SIDECAR_WORKER_CPU_SET`、`MM_SIDECAR_CONTROL_CPU_SET`。
- 当前推荐数据面是 `MM_SIDECAR_PAYLOAD_STORAGE=local_file` +
  `MM_SIDECAR_PAYLOAD_FILE_FORMAT=npy` + `MM_SIDECAR_PAYLOAD_DTYPE=fp32`。
  `raw` 仅保留为 opt-in 实验格式；bf16 payload 后端已按 strict 结果移除。
- TP2 多卡路径使用 vLLM `--mm-encoder-tp-mode data`，并由 worker 侧
  ViT-DP shard-fetch 从 sidecar 拉取当前 rank 需要的图像 payload。
- 当前仓库可直接用于 sidecar + patched vLLM 联调、strict benchmark 和远端复测。

## 仓库结构

- [docs/mm_sidecar_with_phase_formal_design.md](docs/mm_sidecar_with_phase_formal_design.md)
  正式方案设计与阶段边界。
- [docs/strict_benchmark_summary_2026-06-24.md](docs/strict_benchmark_summary_2026-06-24.md)
  strict benchmark 方法、实验记录、性能对比与结论。
- [src/mm_sidecar/sidecar/README.md](src/mm_sidecar/sidecar/README.md)
  sidecar 进程模型、模块职责和运行接口。
- [src/mm_sidecar/integrations/vllm_patch/README.md](src/mm_sidecar/integrations/vllm_patch/README.md)
  vLLM patch 入口、请求接管点和 worker 侧替换流程。
- `benchmarks/`
  benchmark 驱动、数据准备和结果输出工具。

## 关键入口

- `src/mm_sidecar/sidecar/processor.py`
  图像加载、probe、payload 生成。
- `src/mm_sidecar/sidecar/manager.py`
  prepare、metadata wait、ready drain、artifact 获取。
- `src/mm_sidecar/sidecar/service.py`
  sidecar 服务装配、worker 池配置、CPU affinity。
- `src/mm_sidecar/integrations/vllm_patch/patches.py`
  API server patch 安装入口。
- `src/mm_sidecar/integrations/vllm_patch/sidecar_bridge.py`
  API 侧 descriptor capture、prepare、metadata wait。
- `src/mm_sidecar/integrations/vllm_patch/worker_sidecar.py`
  model worker 侧 artifact 获取与 feature replace。

## 运行主代码

以下命令针对远端标准运行目录 `/root/mm-sidecar-with-phase`，本地调试时把路径替换为本地工程目录即可。

### 1. 启动 sidecar

```bash
cd /root/mm-sidecar-with-phase

MM_SIDECAR_TRANSPORT=unix \
MM_SIDECAR_SOCKET_PATH=/tmp/mm-sidecar-e2e.sock \
MM_SIDECAR_WORKER_COUNT=32 \
MM_SIDECAR_WORKER_POOL_MODE=process \
MM_SIDECAR_WORKER_CPU_SET=0-31 \
MM_SIDECAR_CONTROL_CPU_SET=32-47 \
MM_SIDECAR_PAYLOAD_STORAGE=local_file \
MM_SIDECAR_PAYLOAD_FILE_FORMAT=npy \
MM_SIDECAR_PAYLOAD_DTYPE=fp32 \
PYTHONPATH=/root/mm-sidecar-with-phase/src \
taskset -c 0-47 /root/miniconda3/bin/python -m mm_sidecar.sidecar.launcher
```

### 2. 启动 patched vLLM API server

```bash
cd /root/mm-sidecar-with-phase

PYTHONPATH=/root/mm-sidecar-with-phase/src \
MM_SIDECAR_TRANSPORT=unix \
MM_SIDECAR_SOCKET_PATH=/tmp/mm-sidecar-e2e.sock \
MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE=1 \
MM_SIDECAR_METADATA_WAIT_MS=2 \
MM_SIDECAR_ENABLE_DEBUG_ROUTE=1 \
MM_SIDECAR_WORKER_FETCH_PROFILE=1 \
MM_SIDECAR_ENABLE_VIT_DP_SHARD_FETCH=1 \
MM_SIDECAR_ENABLE_VIT_DP_DIRECT_ENCODE=0 \
taskset -c 96-143 /root/miniconda3/bin/python \
  -m mm_sidecar.integrations.vllm_patch.launcher \
  --model /autodl-fs/data/qwen3.5-0.8b \
  --served-model-name qwen3vl-0.8b \
  --host 127.0.0.1 \
  --port 18001 \
  --max-model-len 16384 \
  --trust-remote-code \
  --allowed-local-media-path /root/mm-sidecar-e2e \
  --max-num-seqs 1 \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data \
  --enforce-eager \
  --no-enable-log-requests
```

## CPU affinity

主代码支持两类正式 affinity 配置：

- `MM_SIDECAR_WORKER_CPU_SET`
  指定 mm processor workers 的 CPU 集合。
- `MM_SIDECAR_CONTROL_CPU_SET`
  指定 sidecar manager、listener、service control-plane 的 CPU 集合。

推荐把 worker 和 control-plane 分开绑核，避免 manager/listener 与 payload 处理争抢同一批 CPU。

远端已验证的一组布局如下：

```text
workers:        CPU 0-31
control-plane:  CPU 32-47
API server:     CPU 96-143
```

如果不显式设置这两个变量，sidecar 会基于当前进程可见 CPU 集合做默认分配。

## Strict benchmark

标准入口：

```bash
cd /root/mm-sidecar-with-phase

/root/miniconda3/bin/python benchmarks/e2e_smoke_matrix.py \
  --host http://127.0.0.1:18001 \
  --model auto \
  --work-dir /root/mm-sidecar-e2e/strict_matrix \
  --out /root/mm-sidecar-e2e/strict_matrix.json \
  --transports local_path,http,base64 \
  --image-counts 1,13,20 \
  --warmup 3 \
  --runs 5 \
  --fetch-debug \
  --image-seed 20260625
```

当前 strict 协议与对比口径见：
[docs/strict_benchmark_summary_2026-06-24.md](docs/strict_benchmark_summary_2026-06-24.md)

## 新接手顺序

建议按下面顺序进入代码：

1. [docs/mm_sidecar_with_phase_formal_design.md](docs/mm_sidecar_with_phase_formal_design.md)
2. [docs/strict_benchmark_summary_2026-06-24.md](docs/strict_benchmark_summary_2026-06-24.md)
3. [src/mm_sidecar/sidecar/README.md](src/mm_sidecar/sidecar/README.md)
4. [src/mm_sidecar/integrations/vllm_patch/README.md](src/mm_sidecar/integrations/vllm_patch/README.md)
5. `processor.py` -> `manager.py` -> `sidecar_bridge.py` -> `worker_sidecar.py`

如果接手的是性能问题，先看 `docs/strict_benchmark_summary_2026-06-24.md` 的最新 strict 结论，再决定是继续看 API metadata 路径还是 model worker fetch/replace 路径。

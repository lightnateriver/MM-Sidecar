# Strict Benchmark Summary 2026-06-24

## Project Directory

Current project root:

```text
/mnt/d/project/mm-sidecar/mm-sidecar-with-phase
```

Remote benchmark host:

```text
58.144.141.28:16446
```

Remote benchmark artifact root:

```text
/root/mm-sidecar-e2e
```

## Current Code Layout

- `src/mm_sidecar/contracts/`: shared contract objects for media source, artifact, limits, signature, and manifests.
- `src/mm_sidecar/sidecar/`: sidecar manager, processor worker pool, runtime service, protocol, cache, and launcher.
- `src/mm_sidecar/integrations/vllm_patch/`: external vLLM monkey patch, request capture, descriptor-only fast path, Qwen adapter, and worker-side replacement.
- `benchmarks/e2e_smoke_matrix.py`: strict end-to-end multimodal matrix benchmark. It now supports deterministic random image generation and strict no-image-reuse allocation.
- `tmp/strict_benchmarks/`: temporary copied benchmark tables and analysis notes for this round.

## Strict Protocol

Each case uses:

```text
3 warmup runs + 5 measured runs
```

Matrix:

```text
transports: local_path, http, base64
image counts: 1, 13, 20
```

Strict image policy:

```text
Every run uses distinct generated random images.
No image is reused across transports or image-count cases.
Full matrix requires 816 generated images.
```

## Compared Runs

- Baseline strict round 1: `/root/mm-sidecar-e2e/strict_matrix_baseline_r1.json`
- Baseline strict round 2: `/root/mm-sidecar-e2e/strict_matrix_baseline_r2.json`
- Patched strict round 1: `/root/mm-sidecar-e2e/strict_matrix_r1.json`
- Patched strict round 2: `/root/mm-sidecar-e2e/strict_matrix_r2.json`
- Patched affinity strict round 1: `/root/mm-sidecar-e2e/strict_matrix_affinity_r1.json`

The corresponding markdown tables are copied into:

```text
tmp/strict_benchmarks/
```

## Two-Round Baseline vs Patched

| case | baseline TTFT avg ms | patched TTFT avg ms | delta ms | verdict |
|---|---:|---:|---:|---|
| local_path 1 | 88.97 | 160.13 | +71.16 | slower |
| local_path 13 | 362.36 | 387.74 | +25.38 | slower |
| local_path 20 | 589.00 | 646.31 | +57.30 | slower |
| http 1 | 88.96 | 171.22 | +82.26 | slower |
| http 13 | 1391.83 | 385.21 | -1006.62 | faster |
| http 20 | 1811.86 | 617.24 | -1194.62 | faster |
| base64 1 | 92.04 | 177.62 | +85.58 | slower |
| base64 13 | 389.65 | 400.60 | +10.95 | slower |
| base64 20 | 614.85 | 598.48 | -16.37 | slightly faster |

Main result:

```text
The current sidecar path clearly helps HTTP multi-image cases.
It hurts single-image cases and most local_path/base64 cases because the original path is already cheap and sidecar control-plane overhead dominates.
```

## NUMA-Affinity Run

Affinity layout used:

```text
mm processor workers: CPU 0-31, one worker per CPU
sidecar launcher/manager/listener: CPU 32-47
API server: CPU 96-143
All CPUs are on NUMA node0.
```

How affinity was implemented:

```text
Worker affinity is implemented in code:
src/mm_sidecar/sidecar/service.py
  _available_cpu_ids()
  _default_cpu_affinity_map()
  _manager_config_from_env()

src/mm_sidecar/sidecar/processor.py
  _bind_worker_cpu()
  MultiProcessProcessorWorkerPool.__init__()
```

The worker mapping is derived from the sidecar process affinity. For the affinity run,
the sidecar service was started under:

```bash
taskset -c 0-47 /root/miniconda3/bin/python -m mm_sidecar.sidecar.launcher
```

Because the service process saw only CPUs `0-47`, `_default_cpu_affinity_map(32)`
assigned workers to single-CPU affinities `0, 1, ..., 31`. Runtime verification:

```text
worker pid 19867 Cpus_allowed_list: 0
...
worker pid 19898 Cpus_allowed_list: 31
```

The sidecar launcher and manager/listener were then moved away from worker CPUs:

```bash
taskset -pc 32-47 <sidecar-launcher-pid>
taskset -pc 32-47 <sidecar-manager-pid>
```

The API server was launched on the same NUMA node but a disjoint CPU range:

```bash
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
  --no-enable-log-requests
```

Runtime verification:

```text
sidecar launcher Cpus_allowed_list: 32-47
sidecar manager  Cpus_allowed_list: 32-47
API server       Cpus_allowed_list: 96-143
workers          Cpus_allowed_list: 0..31, one CPU per worker
```

| case | baseline TTFT ms | patched TTFT ms | affinity TTFT ms | affinity vs patched |
|---|---:|---:|---:|---:|
| local_path 1 | 88.97 | 160.13 | 154.20 | -5.93 |
| local_path 13 | 362.36 | 387.74 | 447.28 | +59.54 |
| local_path 20 | 589.00 | 646.31 | 556.87 | -89.44 |
| http 1 | 88.96 | 171.22 | 172.79 | +1.57 |
| http 13 | 1391.83 | 385.21 | 368.00 | -17.21 |
| http 20 | 1811.86 | 617.24 | 506.22 | -111.02 |
| base64 1 | 92.04 | 177.62 | 171.51 | -6.11 |
| base64 13 | 389.65 | 400.60 | 403.52 | +2.92 |
| base64 20 | 614.85 | 598.48 | 508.78 | -89.70 |

Affinity helps `20` image cases, especially `http 20` and `base64 20`, but does not solve single-image overhead.

## NUMA-Affinity Rerun After Sync

After syncing the local project to `/root/mm-sidecar-with-phase`, the same affinity layout was restarted and strict benchmark was rerun with seed `20260626`.

Remote artifacts:

```text
/root/mm-sidecar-e2e/strict_matrix_affinity_rerun_20260626.json
/root/mm-sidecar-e2e/strict_matrix_affinity_rerun_20260626.md
```

The markdown table is also copied to:

```text
tmp/strict_benchmarks/strict_matrix_affinity_rerun_20260626.md
```

| case | baseline TTFT | patched TTFT | affinity r1 | affinity rerun | rerun vs r1 | rerun vs baseline | rerun vs patched |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_path 1 | 88.97 | 160.13 | 154.20 | 157.56 | +3.36 | +68.59 | -2.57 |
| local_path 13 | 362.36 | 387.74 | 447.28 | 426.01 | -21.28 | +63.65 | +38.27 |
| local_path 20 | 589.00 | 646.31 | 556.87 | 567.59 | +10.73 | -21.41 | -78.71 |
| http 1 | 88.96 | 171.22 | 172.79 | 170.73 | -2.06 | +81.77 | -0.48 |
| http 13 | 1391.83 | 385.21 | 368.00 | 352.69 | -15.30 | -1039.13 | -32.52 |
| http 20 | 1811.86 | 617.24 | 506.22 | 569.44 | +63.22 | -1242.42 | -47.81 |
| base64 1 | 92.04 | 177.62 | 171.51 | 175.75 | +4.24 | +83.71 | -1.87 |
| base64 13 | 389.65 | 400.60 | 403.52 | 367.12 | -36.40 | -22.53 | -33.48 |
| base64 20 | 614.85 | 598.48 | 508.78 | 538.62 | +29.83 | -76.23 | -59.86 |

Rerun conclusion:

```text
The rerun is consistent with the earlier observation:
- HTTP multi-image remains much faster than baseline.
- Single-image cases remain slower than baseline.
- Affinity remains helpful for 20-image cases versus non-affinity patched runs.
- http 20 has run-to-run variance, but still keeps the same directional result.
```

## Current Bottleneck Interpretation

Measured worker-side image processing is around:

```text
27-30 ms/image
```

However, API server-side metadata wait remains much larger:

| case | metadata wait avg ms | worker total max avg ms | estimated control-plane gap |
|---|---:|---:|---:|
| http 13 | 126.25 | 28.54 | 97.71 |
| http 20 | 181.89 | 31.11 | 150.78 |
| base64 13 | 123.55 | 38.76 | 84.79 |
| base64 20 | 158.12 | 39.45 | 118.67 |
| local_path 13 | 143.25 | 28.40 | 114.86 |

Current evidence says the extra wait is not image computation. It is likely in the sidecar control plane:

- worker process to manager process result visibility
- multiprocessing queue drain
- manager `wait_for_metadata()` polling
- Unix socket RPC response path
- descriptor-only retry behavior

Preprocessed tensor payload is not returned to the API server in the metadata wait path. Metadata wait returns handles, status snapshots, and schedule items. The larger tensor payload is fetched later on the model-worker side through `fetch_ready()`.

## Next Engineering Priorities

1. Add request-scoped timestamps around `wait_for_metadata()` and sidecar service RPC boundaries.
2. Emit worker result `put` timestamps and manager `_drain_results()` receive timestamps.
3. Split `probed` from full preprocess so metadata can return immediately after `schedule_item` is built.
4. Add a transport/count bypass policy for local_path/base64 single-image or cheap cases.
5. Investigate large tensor payload movement separately in the model-worker `fetch_ready()` path.

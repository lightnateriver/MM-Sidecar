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

## Early Metadata Return Monkey Patch

This experiment was implemented as an external monkey patch only:

```text
tmp/early_metadata_sidecar_launcher.py
```

No core source file was changed for this run. The launcher monkey patches the sidecar processor path at runtime so a worker emits `probed` immediately after `schedule_item` is built, then continues full image tensor preprocessing and emits `ready` later. This tests whether API server metadata wait is blocked by full preprocess completion.

Remote artifacts:

```text
/root/mm-sidecar-e2e/strict_matrix_fetch_profile_20260627.json
/root/mm-sidecar-e2e/strict_matrix_early_metadata_20260628.json
```

Comparison below uses the same strict protocol. `baseline` is the two-round baseline average, `before early metadata` is the affinity/fetch-profile control run, and `after early metadata` is the monkey-patched run.

| case | baseline TTFT ms | before early metadata TTFT ms | after early metadata TTFT ms | after vs before | after vs baseline | before metadata wait ms | after metadata wait ms | metadata wait delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_path 1 | 88.97 | 159.53 | 123.04 | -36.49 | +34.07 | 23.19 | 11.44 | -11.75 |
| local_path 13 | 362.36 | 433.64 | 397.21 | -36.43 | +34.85 | 136.92 | 120.82 | -16.10 |
| local_path 20 | 589.00 | 552.93 | 466.59 | -86.34 | -122.41 | 51.97 | 109.57 | +57.60 |
| http 1 | 88.96 | 174.78 | 124.10 | -50.68 | +35.14 | 30.66 | 11.93 | -18.73 |
| http 13 | 1391.83 | 383.68 | 306.83 | -76.85 | -1085.00 | 126.92 | 57.38 | -69.54 |
| http 20 | 1811.86 | 537.89 | 514.06 | -23.83 | -1297.80 | 176.67 | 83.91 | -92.76 |
| base64 1 | 92.04 | 175.42 | 158.35 | -17.07 | +66.31 | 29.16 | 15.23 | -13.93 |
| base64 13 | 389.65 | 383.52 | 335.72 | -47.80 | -53.93 | 120.71 | 17.85 | -102.86 |
| base64 20 | 614.85 | 511.41 | 515.22 | +3.81 | -99.63 | 150.96 | 94.91 | -56.05 |

Result:

```text
Early metadata return lowers metadata wait for most cases and usually lowers TTFT.
The largest metadata wait reductions are base64 13, http 20, http 13, and base64 20.
Single-image cases are still slower than baseline because fixed sidecar overhead remains.
Some TTFT deltas do not track metadata wait deltas because model-worker-side fetch_ready payload transfer remains a separate downstream bottleneck.
```

## Metadata Wait Trace Breakdown

To split the remaining `metadata wait`, a second external monkey patch was added:

```text
tmp/metadata_trace_sidecar_launcher.py
```

This patch keeps the early metadata behavior and adds request-scoped timestamps for:

- worker `probed` result `put`
- manager `probed` result `apply`
- manager `wait_for_metadata()` loop start / return

The rerun below uses the same strict protocol, but focuses on multi-image cases only:

```text
transports: local_path, http, base64
image counts: 13, 20
3 warmup + 5 measured
distinct random images for every run
```

Remote artifact:

```text
/root/mm-sidecar-e2e/metadata_trace_matrix_multionly_20260624.json
```

Derived timing breakdown:

| case | TTFT ms | API metadata wait ms | manager wait loop ms | wait -> last worker probed put ms | last probed put -> manager apply ms | manager apply -> API return ms | max worker start -> probe ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_path 13 | 335.03 | 74.99 | 70.64 | 1.29 | 70.10 | 0.47 | 4.96 |
| local_path 20 | 454.37 | 65.59 | 57.95 | 1.13 | 58.37 | 0.99 | 4.85 |
| http 13 | 318.81 | 63.76 | 56.40 | 2.33 | 53.32 | 0.75 | 9.37 |
| http 20 | 465.17 | 155.16 | 143.60 | 13.28 | 95.80 | 35.44 | 24.27 |
| base64 13 | 323.45 | 53.14 | 44.73 | 0.02 | 46.88 | 0.72 | 4.99 |
| base64 20 | 491.71 | 154.89 | 89.97 | 0.46 | 119.35 | 0.97 | 5.77 |

Interpretation:

```text
For local_path 13/20, http 13, and base64 13/20, the dominant cost is not worker probe generation.
The dominant cost is that the final worker "probed" result becomes visible to the manager much later than it is put into the multiprocessing result queue.
In those cases, manager apply -> API return is usually sub-1 ms, so the API server is not the main residual bottleneck.
http 20 is the main outlier: it still shows a large queue/apply gap, and also an extra ~35 ms between manager apply and API return.
```

## Split-Queue Monkey Patch

Based on the trace result above, a third external monkey patch was added:

```text
tmp/metadata_split_queue_sidecar_launcher.py
tmp/light_api_prepare_launcher.py
```

What this experiment changes:

- keep early metadata return
- split `started`/`probed`/`failed` and `ready payload` onto different multiprocessing queues
- let a background thread drain `ready` payloads asynchronously
- keep API request-side prepare lighter by skipping synchronous `source_plan_preview()` and `manager.stats()` on the hot path

No core source file was changed for this run.

Remote artifact:

```text
/root/mm-sidecar-e2e/metadata_split_full_20260624.json
```

Comparison below uses the same strict protocol. `before` is the early-metadata monkey patch run and `after` is the split-queue plus light-prepare monkey patch run.

| case | baseline TTFT ms | before TTFT ms | after TTFT ms | after vs before | after vs baseline | before metadata wait ms | after metadata wait ms | metadata wait delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_path 1 | 88.97 | 123.04 | 83.82 | -39.22 | -5.15 | 11.44 | 3.25 | -8.19 |
| local_path 13 | 362.36 | 397.21 | 353.96 | -43.24 | -8.39 | 120.82 | 12.06 | -108.76 |
| local_path 20 | 589.00 | 466.59 | 456.57 | -10.03 | -132.44 | 109.57 | 7.39 | -102.19 |
| http 1 | 88.96 | 124.10 | 111.68 | -12.41 | +22.73 | 11.93 | 5.92 | -6.01 |
| http 13 | 1391.83 | 306.83 | 369.04 | +62.21 | -1022.78 | 57.38 | 76.50 | +19.12 |
| http 20 | 1811.86 | 514.06 | 486.24 | -27.82 | -1325.62 | 83.91 | 13.27 | -70.64 |
| base64 1 | 92.04 | 158.35 | 112.50 | -45.86 | +20.45 | 15.23 | 4.91 | -10.32 |
| base64 13 | 389.65 | 335.72 | 273.20 | -62.53 | -116.45 | 17.85 | 7.90 | -9.95 |
| base64 20 | 614.85 | 515.22 | 527.91 | +12.69 | -86.94 | 94.91 | 7.49 | -87.41 |

Initial result:

```text
The split-queue patch clearly lowers metadata wait in most cases.
It strongly improves local_path and base64 TTFT in most cases.
However, HTTP end-to-end TTFT does not show a stable improvement signal from this run alone.
```

## HTTP Rerun After Split-Queue Patch

Because the first split-queue full-matrix result for HTTP looked suspicious, HTTP-only strict rerun was executed under the same patched service:

```text
transport: http
image counts: 1, 13, 20
3 warmup + 5 measured
distinct random images for every run
```

Remote artifact:

```text
/root/mm-sidecar-e2e/metadata_split_http_rerun1_20260624.json
```

Comparison against the early-metadata run and the first split-queue run:

| http case | before TTFT ms | split-queue r1 TTFT ms | split-queue rerun TTFT ms | rerun vs before | rerun vs r1 | before metadata wait ms | rerun metadata wait ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| http 1 | 124.10 | 111.68 | 131.25 | +7.15 | +19.56 | 11.93 | 5.82 |
| http 13 | 306.83 | 369.04 | 373.66 | +66.82 | +4.61 | 57.38 | 12.46 |
| http 20 | 514.06 | 486.24 | 528.68 | +14.62 | +42.44 | 83.91 | 15.09 |

Revised conclusion:

```text
The split-queue patch does reduce HTTP metadata wait, but HTTP TTFT improvement is not stable.
http 13 is a stable regression versus the early-metadata run.
http 20 looked faster in the first run, but the rerun is slower than the early-metadata run, so the earlier "http 20 improved" conclusion is not reliable.
This means the patch likely moved cost out of metadata wait and into a later stage, instead of reducing total HTTP request critical-path time.
The most likely next bottleneck is model-worker-side fetch_ready()/replace, not metadata wait itself.
```

## Main-Code Merge Validation On Remote

After merging the validated changes into `src/`, main-code-only remote reruns were executed from:

```text
/root/mm-sidecar-with-phase/src
```

without using the `tmp/` launchers.

Two extra points had to be made explicit in main code so the deployment could represent the validated runtime shape:

- `MM_SIDECAR_WORKER_CPU_SET`
- `MM_SIDECAR_CONTROL_CPU_SET`

These let main code express the earlier validated affinity layout directly:

```text
workers: 0-31
sidecar manager/listener/control plane: 32-47
API server: 96-143
```

Main-code remote artifact:

```text
/root/mm-sidecar-e2e/strict_maincode_final_20260624.json
/root/mm-sidecar-e2e/strict_maincode_final_20260624.md
```

Measured comparison against the earlier `tmp` split-queue + light-prepare run:

| case | tmp patched TTFT ms | main-code final TTFT ms | delta ms | tmp semantic | main semantic |
|---|---:|---:|---:|---:|---:|
| local_path 1 | 83.82 | 106.32 | +22.49 | 5/5 | 4/5 |
| local_path 13 | 353.96 | 462.83 | +108.87 | 4/5 | 5/5 |
| local_path 20 | 456.57 | 641.06 | +184.50 | 5/5 | 5/5 |
| http 1 | 111.68 | 116.95 | +5.26 | 5/5 | 5/5 |
| http 13 | 369.04 | 392.02 | +22.98 | 4/5 | 5/5 |
| http 20 | 486.24 | 604.82 | +118.58 | 5/5 | 5/5 |
| base64 1 | 112.50 | 132.35 | +19.85 | 5/5 | 5/5 |
| base64 13 | 273.20 | 353.44 | +80.25 | 4/5 | 4/5 |
| base64 20 | 527.91 | 618.94 | +91.02 | 5/5 | 4/5 |

At the same time, the main-code run still shows that API-side metadata wait remains low:

| case | main-code metadata wait avg ms | main-code API prepare total avg ms |
|---|---:|---:|
| local_path 1 | 5.15 | 5.23 |
| local_path 13 | 7.00 | 7.18 |
| local_path 20 | 11.89 | 12.07 |
| http 1 | 7.18 | 7.26 |
| http 13 | 7.71 | 7.97 |
| http 20 | 13.42 | 13.69 |
| base64 1 | 5.10 | 5.16 |
| base64 13 | 7.47 | 7.58 |
| base64 20 | 11.35 | 11.66 |

Interpretation:

```text
The merged main code preserves the intended metadata-wait improvement and remains functionally correct enough for the current smoke gate.
However, the end-to-end full-matrix TTFT still does not fully match the earlier tmp-based experimental launcher results.
This means "optimization merged" and "performance fully reproduced" are not yet the same thing.
The remaining gap is not on the API-side metadata wait path anymore; it is elsewhere in the request critical path.
```

## TP2 ViT-DP Direct-Cache Shard Fetch Validation On CUDA

After the TP2 ViT-DP shard-fetch path showed possible multi-image color drift, the
worker patch was changed so `MM_SIDECAR_ENABLE_VIT_DP_SHARD_FETCH=1` writes encoder
outputs directly into `encoder_cache` by `feature.identifier`. This mirrors the
safe alignment pattern used by the earlier direct-encode experiment and avoids
returning shard-fetched outputs to the stock `_execute_mm_encoder()` zip path.

Code commits:

```text
c25479d fix: direct-cache vit-dp shard fetch outputs
d87f4b5 test: guard shard-fetch direct cache wrapper
```

Remote CUDA validation used 2x RTX 4090D with:

```text
model: /autodl-fs/data/qwen3.5-0.8b
TP: 2
ViT-DP: --mm-encoder-tp-mode data
serving mode: --enforce-eager
sidecar workers: 32
strict protocol: warmup 3 + measured 5, no repeated image paths inside a run
seed: 2026062701
```

Artifacts:

```text
/root/mm-sidecar-e2e/tp2_sidecar_directcache_seed2701_20260627.json
/root/mm-sidecar-e2e/tp2_sidecar_directcache_seed2701_20260627.md
/root/mm-sidecar-e2e/tp2_baseline_vitdp_seed2701_20260627.json
/root/mm-sidecar-e2e/tp2_baseline_vitdp_seed2701_20260627.md
/root/mm-sidecar-e2e/tp2_sidecar_directcache_vs_baseline_seed2701_20260627.md
```

Image uniqueness was checked for both runs:

```text
baseline: 816/816 unique, duplicates=0
sidecar: 816/816 unique, duplicates=0
```

Focused replay note:

```text
The previously suspicious img_0099 13-image case returned "Orange" even as a
single-image request on the CUDA service, so it is not a valid sidecar ordering
failure signal on this backend. Multi-image index probes matched single-image
answers for the first three images.
```

Same-seed strict comparison:

| transport | images | base TTFT ms | sidecar TTFT ms | delta TTFT ms | speedup | base E2E ms | sidecar E2E ms | delta E2E ms | base semantic | sidecar semantic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local_path | 1 | 80.15 | 94.21 | +14.06 | 0.85x | 105.07 | 119.47 | +14.39 | 5/5 | 5/5 |
| local_path | 13 | 225.38 | 227.27 | +1.89 | 0.99x | 249.05 | 251.14 | +2.09 | 5/5 | 5/5 |
| local_path | 20 | 411.80 | 335.49 | -76.31 | 1.23x | 436.35 | 452.88 | +16.53 | 5/5 | 3/5 |
| http | 1 | 78.51 | 94.73 | +16.22 | 0.83x | 103.17 | 119.97 | +16.81 | 5/5 | 5/5 |
| http | 13 | 1272.30 | 1185.72 | -86.57 | 1.07x | 1294.82 | 1208.95 | -85.87 | 5/5 | 5/5 |
| http | 20 | 1486.50 | 1359.13 | -127.37 | 1.09x | 1510.61 | 1383.75 | -126.86 | 5/5 | 5/5 |
| base64 | 1 | 78.50 | 90.21 | +11.71 | 0.87x | 103.26 | 115.21 | +11.94 | 5/5 | 5/5 |
| base64 | 13 | 232.68 | 194.59 | -38.09 | 1.20x | 255.51 | 217.69 | -37.82 | 5/5 | 5/5 |
| base64 | 20 | 422.57 | 283.59 | -138.98 | 1.49x | 446.87 | 353.17 | -93.70 | 5/5 | 4/5 |

Loose semantic interpretation:

```text
Baseline measured semantic: strict 45/45, loose 45/45.
Sidecar measured semantic: strict 42/45, loose 45/45.
The three sidecar strict failures all included the expected color word, but the
model answered in a full sentence instead of a single word. These are output
contract failures, not observed visual-order failures.
```

Interpretation:

```text
The direct-cache shard-fetch path removes the previous suspected multi-image
ordering risk: 13-image cases are strict clean on all transports, and 20-image
loose semantics are clean.

The performance profile is mixed by case:
- 1-image cases are slower because sidecar/direct-cache overhead dominates.
- 13-image base64 and 20-image base64/http show clear TTFT wins.
- local_path 20 shows TTFT win but E2E is polluted by longer full-sentence
  outputs in the sidecar run.

The remaining precision issue is output-contract stability under 20-image
local_path/base64 sidecar runs, not image payload alignment.
```

## TP2 Single-Image Bypass And Timing Breakdown

The TP2 direct-cache run showed that 1-image requests were consistently slower
than baseline because fixed sidecar/control-plane overhead dominated. The main
code now defaults to `MM_SIDECAR_MIN_IMAGE_COUNT=2`, so single-image requests
with an available sidecar manager bypass the sidecar payload path and keep the
native vLLM image path. Multi-image requests still use sidecar.

Code commits:

```text
ffc0f34 feat: bypass single-image sidecar and add timings
82152ca test: relax api fast path payload assertion
808f2e1 fix: keep descriptor-only dummy off for single-image bypass
```

Important correctness fix:

```text
The first implementation disabled the final sidecar payload for single-image
requests, but descriptor-only parsing still replaced the actual image with a
1x1 dummy placeholder. That made local_path/base64 single-image runs return
"black". The fix keeps descriptor-only dummy disabled until the parsed image
index reaches MM_SIDECAR_MIN_IMAGE_COUNT. With the default threshold of 2,
single-image requests keep the real native image; multi-image requests can use
dummy placeholders from the second image onward.
```

Remote CUDA validation used:

```text
model: /autodl-fs/data/qwen3.5-0.8b
TP: 2
ViT-DP: --mm-encoder-tp-mode data
serving mode: --enforce-eager
sidecar workers: 32
strict protocol: warmup 3 + measured 5
seed: 2026062703
```

Artifacts:

```text
/root/mm-sidecar-e2e/tp2_sidecar_min2_dummyfix_seed2703_20260627.json
/root/mm-sidecar-e2e/tp2_sidecar_min2_dummyfix_seed2703_20260627.md
/root/mm-sidecar-e2e/tp2_sidecar_min2_timing_seed2703_20260627.json
/root/mm-sidecar-e2e/tp2_sidecar_min2_timing_seed2703_20260627.md
/root/mm-sidecar-e2e/run_logs_20260627_105852_direct_cache/api.log
```

Comparison note: baseline and old sidecar columns use the prior strict seed
`2026062701`; the new min2 columns use seed `2026062703`.

| transport | images | baseline TTFT | old sidecar TTFT | min2 bug TTFT | min2 fixed TTFT | fixed semantic |
|---|---:|---:|---:|---:|---:|---:|
| local_path | 1 | 80.15 | 94.21 | 68.47 | 83.86 | 5/5 |
| local_path | 13 | 225.38 | 227.27 | 222.68 | 214.11 | 5/5 |
| local_path | 20 | 411.80 | 335.49 | 300.69 | 296.21 | 5/5 |
| http | 1 | 78.51 | 94.73 | 86.96 | 84.14 | 5/5 |
| http | 13 | 1272.30 | 1185.72 | 1183.80 | 1187.50 | 5/5 |
| http | 20 | 1486.50 | 1359.13 | 1390.77 | 1381.13 | 5/5 |
| base64 | 1 | 78.50 | 90.21 | 68.55 | 80.89 | 5/5 |
| base64 | 13 | 232.68 | 194.59 | 200.21 | 200.97 | 5/5 |
| base64 | 20 | 422.57 | 283.59 | 306.92 | 305.19 | 4/5 |

Fixed-run E2E summary:

| transport | images | TTFT avg/max ms | E2E avg/max ms | semantic |
|---|---:|---:|---:|---:|
| local_path | 1 | 83.86/85.44 | 109.23/110.80 | 5/5 |
| local_path | 13 | 214.11/237.57 | 238.37/262.08 | 5/5 |
| local_path | 20 | 296.21/326.97 | 321.27/351.93 | 5/5 |
| http | 1 | 84.14/85.48 | 109.64/111.01 | 5/5 |
| http | 13 | 1187.50/1203.28 | 1211.15/1225.40 | 5/5 |
| http | 20 | 1381.13/1527.82 | 1405.97/1553.02 | 5/5 |
| base64 | 1 | 80.89/81.26 | 106.19/106.50 | 5/5 |
| base64 | 13 | 200.97/209.22 | 224.96/233.73 | 5/5 |
| base64 | 20 | 305.19/322.86 | 360.66/486.41 | 4/5 |

API-side timing from the fixed run:

```text
single-image bypass:
  sidecar_prepare.enabled=false
  reason=image_count_below_min
  api_prepare_total ~= 0.02 ms
  descriptor_build/metadata_wait ~= 0 ms

multi-image sidecar:
  api_prepare_total ~= 3.6-7.4 ms
  descriptor_build ~= 0.3-0.7 ms
  metadata wait ~= 2.7-5.6 ms
```

Worker-side timing from direct-cache debug logs:

| transport | images | worker fetch total ms | client RPC ms | local fallback ms | encode+gather ms | ready barrier ms | payload bytes/rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_path | 13 | 59.95 | 28.21 | 20.26 | 11.97 | 9.07 | 23.0 MB |
| local_path | 20 | 30.02 | 22.26 | 16.81 | 10.33 | 3.43 | 17.7 MB |
| http | 13 | 29.16 | 28.56 | n/a | 12.07 | 2.24 | 23.0 MB |
| http | 20 | 21.34 | 20.82 | n/a | 11.14 | 2.62 | 17.7 MB |
| base64 | 13 | 52.90 | 21.05 | 19.06 | 11.26 | 9.02 | 23.0 MB |
| base64 | 20 | 34.66 | 26.97 | 11.54 | 10.33 | 4.30 | 17.7 MB |

Interpretation:

```text
The metadata-wait/API-prepare path is no longer the dominant bottleneck.
For multi-image cases, remaining time is mostly in model-worker side payload
movement and fallback/sidecar artifact materialization:

- client_rpc_total/fetch_ms is often 20-28 ms per rank.
- local fallback, when present, adds roughly 10-20 ms.
- direct encode + all-gather is roughly 10-12 ms.
- encoder-cache writes are negligible, around 0.1 ms.

The next high-value optimization is therefore not metadata wait. It is reducing
or avoiding the 10-25 MB/rank tensor payload transfer/materialization path,
for example via shared-memory tensor payloads or a direct worker-side source
plan that avoids Python object serialization for pixel_values.
```

## Next Engineering Priorities

1. Reduce model-worker tensor payload transfer/materialization overhead for `pixel_values`.
2. Add request-body pre-counting so descriptor-only dummy can also skip the first image in multi-image requests while still preserving single-image native bypass.
3. Investigate why local_path/base64 still trigger local fallback in some measured worker paths.
4. Keep the single-image bypass policy enabled by default and expand it only if real traffic shows 2-image cheap cases are also slower.
5. Continue output-contract hardening for 20-image strict runs; loose visual semantics are already clean.

## TP2 Batch `fetch_ready` Experiment

A batch artifact fetch path was implemented and tested as a low-intrusion
alternative to shared-memory tensor payload transport.

Implementation commits:

```text
377144b perf: batch sidecar ready artifact fetch
6b81077 fix: keep batch ready fetch opt-in
```

The manager and client now expose `fetch_ready_batch()`, and the coordinator can
use it when explicitly enabled. The default remains the prior per-artifact
`fetch_ready()` path because the strict benchmark showed batch payload transfer
is not a safe default on this backend.

Runtime switch:

```bash
MM_SIDECAR_ENABLE_BATCH_FETCH_READY=1
```

Remote artifacts:

```text
/root/mm-sidecar-e2e/tp2_sidecar_batch_fetch_probe_seed2709_20260627.json
/root/mm-sidecar-e2e/tp2_sidecar_batch_fetch_strict_seed2710_20260627.json
/root/mm-sidecar-e2e/run_logs_20260627_150841_direct_cache/api.log
```

Strict protocol:

```text
TP2 + --mm-encoder-tp-mode data + shard-fetch/direct-cache
MM processor workers: 32
warmup 3 + measured 5
transports: local_path, http, base64
image counts: 1, 13, 20
unique generated images for every case/run
```

Performance comparison:

| transport | imgs | base TTFT | old sidecar TTFT | batch-fetch TTFT | vs old | vs base | base E2E | old E2E | batch E2E | semantic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base64 | 1 | 78.50 | 80.89 | 82.03 | +1.14 | +3.53 | 103.26 | 106.19 | 107.88 | 5/5 |
| base64 | 13 | 232.68 | 200.97 | 216.22 | +15.26 | -16.46 | 255.51 | 224.96 | 240.74 | 5/5 |
| base64 | 20 | 422.57 | 305.19 | 299.36 | -5.82 | -123.20 | 446.87 | 360.66 | 420.12 | 3/5 |
| http | 1 | 78.51 | 84.14 | 85.04 | +0.91 | +6.54 | 103.17 | 109.64 | 110.76 | 5/5 |
| http | 13 | 1272.30 | 1187.50 | 1232.42 | +44.92 | -39.88 | 1294.82 | 1211.15 | 1256.25 | 5/5 |
| http | 20 | 1486.50 | 1381.13 | 1491.41 | +110.28 | +4.92 | 1510.61 | 1405.97 | 1517.04 | 5/5 |
| local_path | 1 | 80.15 | 83.86 | 85.53 | +1.68 | +5.38 | 105.07 | 109.23 | 111.16 | 5/5 |
| local_path | 13 | 225.38 | 214.11 | 236.95 | +22.84 | +11.57 | 249.05 | 238.37 | 261.32 | 5/5 |
| local_path | 20 | 411.80 | 296.21 | 335.85 | +39.64 | -75.94 | 436.35 | 321.27 | 361.08 | 5/5 |

Worker-side fetch profile from the batch run:

| transport | imgs | worker fetch avg ms | fetch_ms avg | client RPC avg | local fallback avg | payload MB avg | batch RPC calls/run-rank | batch items/run-rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base64 | 13 | 71.37 | 27.42 | 27.39 | 28.73 | 21.9 | 1.00 | 2.80 |
| base64 | 20 | 32.20 | 20.38 | 20.34 | 15.76 | 16.9 | 1.00 | 4.00 |
| http | 13 | 64.34 | 63.54 | 63.50 | 0.00 | 21.9 | 1.00 | 6.50 |
| http | 20 | 43.47 | 42.72 | 42.68 | 0.00 | 16.9 | 1.00 | 5.00 |
| local_path | 13 | 87.23 | 29.48 | 29.45 | 32.71 | 21.9 | 1.00 | 2.70 |
| local_path | 20 | 49.00 | 24.16 | 24.12 | 25.60 | 16.9 | 1.00 | 3.50 |

Conclusion:

```text
Batch fetch successfully reduces RPC call count to one call per run/rank, but it
does not reduce the dominant payload-transfer cost. In HTTP 13/20 it is clearly
slower, likely because one large multiprocessing response increases pickle/send
critical-path blocking and removes useful per-artifact scheduling granularity.

Keep the code as an opt-in diagnostic/experimental path, but keep it disabled by
default. The next optimization should target payload representation or reducing
local fallback frequency, not merely reducing the number of fetch_ready RPCs.
```

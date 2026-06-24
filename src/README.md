# src

当前 `src/` 已包含：

- 阶段 A：合同层代码
- 阶段 B：已迁入本工程的 `vllm` monkey patch 集成层原型
- 阶段 C：sidecar manager / processor worker 运行时原型

阶段 A 目标：

- 冻结 `MediaSourceRef`
- 冻结 `NormalizedImage`
- 冻结 `ScheduleManifest`
- 冻结 `ArtifactDescriptor`
- 冻结 `ProcessorSignature`
- 冻结输入限制与错误码

这些对象后续将被：

- 本工程内的 API server transport normalization
- sidecar manager / worker
- TP worker fallback
- precision / manifest parity tests

共同复用。

阶段 B 当前落地内容：

- `integrations/vllm_patch/launcher.py`
  - 以外部 launcher 方式启动 pip 安装版 `vllm`
  - 运行链路保持在本工程内，不再依赖跨工程 import
- `integrations/vllm_patch/patches.py`
  - 对 `AsyncMultiModalContentParser` / `MultiModalContentParser`
    / `HfRenderer` / `build_app` 做 monkey patch
- `integrations/vllm_patch/normalization.py`
  - 统一归一化 `local_path / http / base64` 三类图像输入
- `integrations/vllm_patch/context.py`
  - request-scoped capture 上下文与可选 debug 观测面
- `integrations/vllm_patch/README.md`
  - 约束 patch 目录的职责边界与运行方式

阶段 C 当前落地内容：

- `sidecar/protocol.py`
  - `fallback_descriptor`、外部状态机、handle 与状态快照
- `sidecar/cache.py`
  - CPU memory reusable pool 与 inflight 标记
- `sidecar/manager.py`
  - `PREPARE / BATCH_GET_STATUS / TRY_FALLBACK_CLAIM / FETCH_READY`
- `sidecar/processor.py`
  - inline 与 multiprocess processor worker pool
- `sidecar/__init__.py`
  - 阶段 C 运行时统一导出入口

# Tests

当前阶段先提供 `contracts` 层单测。

后续阶段约定：

- `unit/`：纯数据结构与 helper 测试
- `precision/`：`local_path/http/base64` 三输入精度一致性
- `manifest_parity/`：sidecar manifest 与原始进程内路径逐项对比
- `integration/`：manager / worker / artifact / API bridge 集成
- `fallback/`：TP worker fallback 注入测试
- `latency/`：关键阶段耗时与回归阈值


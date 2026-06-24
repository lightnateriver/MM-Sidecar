# vllm_patch

本目录承载当前 `mm-sidecar-with-phase` 的 API server 侧 monkey patch 原型。

目标：

- 保持对 pip 安装版 `vllm` 的非侵入式接入
- 通过外部 launcher + 运行期 monkey patch 注入能力
- 在本工程内收敛 `local_path / http / base64` 三种输入的 transport 归一化与 request capture

约束：

- 不修改远端安装环境中的 `vllm` 源码
- 不依赖外部 `api server opt` 工程目录进行运行时 import
- 仅允许通过本目录下的 launcher、patches 与 contracts 形成自包含运行链路

当前落地点：

- `launcher.py`
  - 启动 pip 安装版 `vllm`，并在进入 `api_server.run_server` 前应用 patch
- `patches.py`
  - 对 `AsyncMultiModalContentParser`、`MultiModalContentParser`、`HfRenderer` 与 `build_app` 做 request-scoped monkey patch
- `normalization.py`
  - 统一构建 `local_path / http / base64` 的 `NormalizedImage`
- `context.py`
  - request capture、debug route 观测与序列化输出

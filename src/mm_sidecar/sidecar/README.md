# sidecar

本目录承载 `mm-sidecar-with-phase` 的阶段 C 运行时原型。

当前范围：

- manager 控制面
- processor worker 数据面
- 独立 sidecar RPC service / client
- `local_path / http / base64` 三种 image transport
- CPU memory cache pool
- `TRY_FALLBACK_CLAIM` 与 fallback 状态机

当前不包含：

- 与 `TP worker` 的真实线上集成
- `shm`
- GPU / NPU 图预处理
- 跨机共享缓存

实现说明：

- `protocol.py`
  - 定义阶段 C 运行时协议对象
- `cache.py`
  - 管理 reusable pool 与 inflight 标记
- `manager.py`
  - 管理 `PREPARE`、状态查询、claim 与缓存一致性
- `processor.py`
  - 提供 inline 与 multiprocess worker pool，并在 worker 内闭环完成 image I/O、decode 与基础 CPU preprocess
- `service.py`
  - 提供独立 sidecar 进程的本地 RPC 封装，以及 API server 使用的 client 工厂
- `launcher.py`
  - 启动独立 sidecar 服务进程，供 monkey-patched API server 通过本地 client 连接

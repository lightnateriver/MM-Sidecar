from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryCacheConfig:
    max_reusable_bytes: int = 512 * 1024 * 1024
    reusable_entry_ttl_s: float = 300.0


@dataclass(frozen=True, slots=True)
class WorkerPoolConfig:
    worker_count: int = 32
    cpu_affinity_map: tuple[tuple[int, ...], ...] | None = None
    start_method: str = "fork"


@dataclass(frozen=True, slots=True)
class SidecarManagerConfig:
    cache: MemoryCacheConfig = MemoryCacheConfig()
    workers: WorkerPoolConfig = WorkerPoolConfig()

"""Stage C sidecar runtime primitives."""

from .cache import CpuMemoryCachePool
from .config import MemoryCacheConfig, SidecarManagerConfig, WorkerPoolConfig
from .coordinator import (
    SidecarFallbackCoordinator,
    SidecarFetchBatch,
    SourcePlan,
    SourcePlanDecision,
    SourcePlanEntry,
)
from .manager import SidecarManager
from .processor import (
    InlineProcessorWorkerPool,
    MultiProcessProcessorWorkerPool,
    ProcessorWorkerPool,
)
from .protocol import (
    FallbackClaimResult,
    FallbackDescriptor,
    PreparedArtifact,
    SidecarHandle,
    SidecarState,
    SidecarStatusSnapshot,
    build_cache_key,
)

__all__ = [
    "CpuMemoryCachePool",
    "FallbackClaimResult",
    "FallbackDescriptor",
    "InlineProcessorWorkerPool",
    "MemoryCacheConfig",
    "MultiProcessProcessorWorkerPool",
    "PreparedArtifact",
    "ProcessorWorkerPool",
    "SidecarFallbackCoordinator",
    "SidecarFetchBatch",
    "SidecarHandle",
    "SidecarManager",
    "SidecarManagerConfig",
    "SidecarState",
    "SidecarStatusSnapshot",
    "SourcePlan",
    "SourcePlanDecision",
    "SourcePlanEntry",
    "WorkerPoolConfig",
    "build_cache_key",
]

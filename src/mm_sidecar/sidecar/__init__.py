"""Stage C sidecar runtime primitives."""

from .cache import CpuMemoryCachePool
from .config import MemoryCacheConfig, SidecarManagerConfig, WorkerPoolConfig
from .coordinator import (
    SidecarFallbackCoordinator,
    SidecarFetchBatch,
    SourcePlan,
    SourcePlanDecision,
    SourcePlanEntry,
    build_ranked_claimer_id,
    parse_ranked_claimer_id,
)
from .manager import SidecarManager
from .processor import (
    InlineProcessorWorkerPool,
    MultiProcessProcessorWorkerPool,
    ProcessorWorkerPool,
)
from .service import (
    SidecarClient,
    SidecarServiceConfig,
    SidecarServiceProcess,
    connect_sidecar_client_from_env,
    create_sidecar_client,
    describe_sidecar_service_config,
    sidecar_service_config_from_env,
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
    "SidecarClient",
    "SidecarHandle",
    "SidecarManager",
    "SidecarManagerConfig",
    "SidecarServiceConfig",
    "SidecarServiceProcess",
    "connect_sidecar_client_from_env",
    "describe_sidecar_service_config",
    "SidecarState",
    "SidecarStatusSnapshot",
    "SourcePlan",
    "SourcePlanDecision",
    "SourcePlanEntry",
    "WorkerPoolConfig",
    "build_cache_key",
    "build_ranked_claimer_id",
    "create_sidecar_client",
    "parse_ranked_claimer_id",
    "sidecar_service_config_from_env",
]

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from typing import Any

from mm_sidecar.contracts import (
    ArtifactDescriptor,
    CapturedImageRef,
    ImageScheduleItem,
    IngressLimits,
)


def _now_ms() -> float:
    return time.time() * 1000.0


def build_cache_key(item_identity: str, processor_signature_value: str) -> str:
    return f"{item_identity}|{processor_signature_value}"


class SidecarState(str, Enum):
    ABSENT = "ABSENT"
    QUEUED = "QUEUED"
    SIDECAR_RUNNING = "SIDECAR_RUNNING"
    READY = "READY"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    FALLBACK_CLAIMED = "FALLBACK_CLAIMED"
    FALLBACK_LOCAL_DONE = "FALLBACK_LOCAL_DONE"
    BYPASS = "BYPASS"


@dataclass(frozen=True, slots=True)
class FallbackDescriptor:
    request_id: str
    request_media_index: int
    captured_image: CapturedImageRef
    ingress_limits: IngressLimits
    processor_signature_value: str
    item_identity: str
    orig_size_hw: tuple[int, int] | None = None
    http_headers: tuple[tuple[str, str], ...] = ()
    http_timeout_ms: int = 30_000
    allow_redirects: bool = True
    payload_hint: object | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.request_media_index < 0:
            raise ValueError("request_media_index must be non-negative")
        if not self.processor_signature_value:
            raise ValueError("processor_signature_value must not be empty")
        if not self.item_identity:
            raise ValueError("item_identity must not be empty")
        if self.http_timeout_ms <= 0:
            raise ValueError("http_timeout_ms must be positive")
        if self.orig_size_hw is not None:
            height, width = self.orig_size_hw
            if height <= 0 or width <= 0:
                raise ValueError("orig_size_hw must be positive when provided")

    @property
    def cache_key(self) -> str:
        return build_cache_key(self.item_identity, self.processor_signature_value)


@dataclass(frozen=True, slots=True)
class SidecarHandle:
    request_id: str
    request_media_index: int
    cache_key: str
    epoch: int


@dataclass(frozen=True, slots=True)
class SidecarLookupResult:
    cache_key: str
    handle: SidecarHandle | None
    descriptor: FallbackDescriptor | None
    state: SidecarState
    updated_at_ms: float
    claimed_by: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class SidecarStatusSnapshot:
    handle: SidecarHandle
    state: SidecarState
    epoch: int
    updated_at_ms: float
    owner_worker_id: int | None = None
    claimed_by: str | None = None
    artifact_descriptor: ArtifactDescriptor | None = None
    schedule_item: ImageScheduleItem | None = None
    timings_ms: dict[str, float] | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedArtifact:
    handle: SidecarHandle
    descriptor: ArtifactDescriptor
    payload: Any
    timings_ms: dict[str, float] | None = None
    fetch_diagnostics_ms: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class FallbackClaimResult:
    handle: SidecarHandle
    granted: bool
    state: SidecarState
    epoch: int
    claimed_by: str | None
    updated_at_ms: float
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class SidecarManagerStats:
    queued_items: int
    running_items: int
    ready_items: int
    failed_items: int
    fallback_claimed_items: int
    reusable_cache_items: int
    reusable_cache_bytes: int
    active_inflight_items: int
    observed_at_ms: float = field(default_factory=_now_ms)

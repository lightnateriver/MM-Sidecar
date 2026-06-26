from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .manager import SidecarManager
from .protocol import (
    FallbackDescriptor,
    PreparedArtifact,
    SidecarHandle,
    SidecarState,
    SidecarStatusSnapshot,
)


class SourcePlanDecision(str, Enum):
    USE_SIDECAR = "USE_SIDECAR"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True, slots=True)
class SourcePlanEntry:
    request_media_index: int
    decision: SourcePlanDecision
    producer_rank: int | None = None
    handle: SidecarHandle | None = None
    state: SidecarState | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SourcePlan:
    request_id: str
    entries: tuple[SourcePlanEntry, ...]
    near_ready_wait_ms: float
    used_fail_open: bool


@dataclass(frozen=True, slots=True)
class SidecarFetchBatch:
    source_plan: SourcePlan
    sidecar_artifacts: tuple[PreparedArtifact, ...]
    fallback_descriptors: tuple[FallbackDescriptor, ...]


_CLAIMER_RANK_MARKER = "::producer_rank="


def build_ranked_claimer_id(*, request_id: str, producer_rank: int) -> str:
    return f"{request_id}{_CLAIMER_RANK_MARKER}{int(producer_rank)}"


def parse_ranked_claimer_id(claimer_id: str | None) -> int | None:
    if not claimer_id:
        return None
    marker_index = claimer_id.rfind(_CLAIMER_RANK_MARKER)
    if marker_index < 0:
        return None
    raw_rank = claimer_id[marker_index + len(_CLAIMER_RANK_MARKER):].strip()
    if not raw_rank:
        return None
    try:
        parsed = int(raw_rank)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


class SidecarFallbackCoordinator:
    def __init__(
        self,
        *,
        manager: SidecarManager | None,
        claimer_id: str,
        producer_rank: int,
        near_ready_wait_ms: float = 2.0,
        poll_interval_ms: float = 1.0,
        fallback_wait_ms: float = 1_000.0,
        observe_plan_wait_ms: float = 50.0,
    ) -> None:
        self._manager = manager
        self._claimer_id = claimer_id
        self._producer_rank = producer_rank
        self._near_ready_wait_ms = near_ready_wait_ms
        self._poll_interval_ms = poll_interval_ms
        self._fallback_wait_ms = fallback_wait_ms
        self._observe_plan_wait_ms = observe_plan_wait_ms

    def build_source_plan(
        self,
        *,
        descriptors: list[FallbackDescriptor] | tuple[FallbackDescriptor, ...],
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...] | None = None,
        claim: bool = True,
        wait_for_ready: bool = True,
    ) -> SourcePlan:
        if not descriptors:
            raise ValueError("descriptors must not be empty")

        if self._manager is None:
            return self._build_fail_open_plan(descriptors)
        if handles is None:
            raise ValueError("handles are required when manager is available")
        if len(descriptors) != len(handles):
            raise ValueError("descriptors and handles must have the same length")

        initial = self._manager.batch_get_status(handles)
        unresolved_states = {
            SidecarState.QUEUED,
            SidecarState.SIDECAR_RUNNING,
        }
        unresolved_indexes = {
            snapshot.handle.request_media_index
            for snapshot in initial
            if snapshot.state in unresolved_states
        }

        waited_ms = 0.0
        final = initial
        if wait_for_ready and unresolved_indexes and self._near_ready_wait_ms > 0.0:
            wait_start = _now_ms()
            deadline = wait_start + self._near_ready_wait_ms
            while True:
                final = self._manager.batch_get_status(handles)
                unresolved_indexes = {
                    snapshot.handle.request_media_index
                    for snapshot in final
                    if snapshot.state in unresolved_states
                }
                if not unresolved_indexes:
                    break
                if _now_ms() >= deadline:
                    break
                time.sleep(self._poll_interval_ms / 1000.0)
            waited_ms = max(0.0, _now_ms() - wait_start)

        claim_targets = [
            snapshot.handle
            for snapshot in final
            if claim and snapshot.state is not SidecarState.READY
        ]
        claim_results_by_index: dict[int, Any] = {}
        if claim and claim_targets:
            claims = self._manager.try_fallback_claim(claim_targets, self._claimer_id)
            claim_results_by_index = {
                claim.handle.request_media_index: claim for claim in claims
            }

        snapshot_by_index = {
            snapshot.handle.request_media_index: snapshot for snapshot in final
        }
        entries: list[SourcePlanEntry] = []
        for descriptor, handle in zip(descriptors, handles):
            snapshot = snapshot_by_index.get(handle.request_media_index)
            if snapshot is not None and snapshot.state is SidecarState.READY:
                entries.append(
                    SourcePlanEntry(
                        request_media_index=handle.request_media_index,
                        decision=SourcePlanDecision.USE_SIDECAR,
                        handle=handle,
                        state=snapshot.state,
                        reason="ready_before_fallback",
                    )
                )
                continue

            claim_result = claim_results_by_index.get(handle.request_media_index)
            if claim_result is not None and claim_result.granted:
                entries.append(
                    SourcePlanEntry(
                        request_media_index=handle.request_media_index,
                        decision=SourcePlanDecision.FALLBACK,
                        producer_rank=self._producer_rank,
                        handle=claim_result.handle,
                        state=snapshot.state if snapshot is not None else SidecarState.ABSENT,
                        reason="fallback_claim_granted",
                    )
                )
                continue

            if claim_result is not None and claim_result.state is SidecarState.READY:
                entries.append(
                    SourcePlanEntry(
                        request_media_index=handle.request_media_index,
                        decision=SourcePlanDecision.USE_SIDECAR,
                        handle=handle,
                        state=claim_result.state,
                        reason="ready_after_claim_race",
                    )
                )
                continue

            claimed_rank = parse_ranked_claimer_id(
                claim_result.claimed_by if claim_result is not None else None
            )
            if (
                claim_result is not None
                and claim_result.state in {
                    SidecarState.FALLBACK_CLAIMED,
                    SidecarState.FALLBACK_LOCAL_DONE,
                }
                and claimed_rank is not None
            ):
                entries.append(
                    SourcePlanEntry(
                        request_media_index=handle.request_media_index,
                        decision=SourcePlanDecision.FALLBACK,
                        producer_rank=claimed_rank,
                        handle=claim_result.handle,
                        state=claim_result.state,
                        reason="fallback_claim_already_owned",
                    )
                )
                continue

            if claim and claim_result is not None and not claim_result.granted:
                owner = claim_result.claimed_by or "unknown"
                state = claim_result.state.value
                raise RuntimeError(
                    "fallback claim denied for "
                    f"media index {handle.request_media_index}: "
                    f"state={state}, claimed_by={owner}, "
                    f"error={claim_result.error_message or 'none'}"
                )

            entries.append(
                SourcePlanEntry(
                    request_media_index=descriptor.request_media_index,
                    decision=SourcePlanDecision.FALLBACK,
                    producer_rank=self._producer_rank,
                    handle=handle,
                    state=snapshot.state if snapshot is not None else SidecarState.ABSENT,
                    reason=(
                        "preview_requires_fallback"
                        if not claim
                        else "claim_denied_fail_open"
                    ),
                )
            )

        return SourcePlan(
            request_id=descriptors[0].request_id,
            entries=tuple(sorted(entries, key=lambda item: int(item.request_media_index))),
            near_ready_wait_ms=waited_ms,
            used_fail_open=False,
        )

    def fetch_according_to_plan(
        self,
        *,
        descriptors: list[FallbackDescriptor] | tuple[FallbackDescriptor, ...],
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...] | None = None,
        source_plan: SourcePlan | None = None,
    ) -> SidecarFetchBatch:
        plan = source_plan or self.build_source_plan(
            descriptors=descriptors,
            handles=handles,
        )
        descriptor_by_index = {
            descriptor.request_media_index: descriptor for descriptor in descriptors
        }

        sidecar_artifacts: list[PreparedArtifact] = []
        fallback_descriptors: list[FallbackDescriptor] = []
        for entry in plan.entries:
            if entry.decision is SourcePlanDecision.USE_SIDECAR:
                if self._manager is None or entry.handle is None:
                    raise RuntimeError("sidecar fetch requested without manager/handle")
                artifact = self._manager.fetch_ready(entry.handle)
                if artifact is None:
                    raise RuntimeError(
                        f"sidecar artifact missing for media index {entry.request_media_index}"
                    )
                sidecar_artifacts.append(artifact)
                continue

            descriptor = descriptor_by_index.get(entry.request_media_index)
            if descriptor is None:
                raise RuntimeError(
                    f"fallback descriptor missing for media index {entry.request_media_index}"
                )
            if (
                entry.producer_rank is not None
                and entry.producer_rank != self._producer_rank
            ):
                sidecar_artifacts.append(self._wait_and_fetch_remote_fallback(entry))
                continue
            fallback_descriptors.append(descriptor)

        return SidecarFetchBatch(
            source_plan=plan,
            sidecar_artifacts=tuple(sidecar_artifacts),
            fallback_descriptors=tuple(
                sorted(
                    fallback_descriptors,
                    key=lambda item: int(item.request_media_index),
                )
            ),
        )

    def _build_fail_open_plan(
        self,
        descriptors: list[FallbackDescriptor] | tuple[FallbackDescriptor, ...],
    ) -> SourcePlan:
        return SourcePlan(
            request_id=descriptors[0].request_id,
            entries=tuple(
                SourcePlanEntry(
                    request_media_index=descriptor.request_media_index,
                    decision=SourcePlanDecision.FALLBACK,
                    producer_rank=self._producer_rank,
                    handle=None,
                    state=SidecarState.ABSENT,
                    reason="manager_unavailable_fail_open",
                )
                for descriptor in sorted(
                    descriptors,
                    key=lambda item: int(item.request_media_index),
                )
            ),
            near_ready_wait_ms=0.0,
            used_fail_open=True,
        )

    def preview_source_plan(
        self,
        *,
        descriptors: list[FallbackDescriptor] | tuple[FallbackDescriptor, ...],
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...] | None = None,
    ) -> SourcePlan:
        return self.build_source_plan(
            descriptors=descriptors,
            handles=handles,
            claim=False,
            wait_for_ready=False,
        )

    def observe_source_plan(
        self,
        *,
        descriptors: list[FallbackDescriptor] | tuple[FallbackDescriptor, ...],
        wait_timeout_ms: float | None = None,
    ) -> SourcePlan:
        if not descriptors:
            raise ValueError("descriptors must not be empty")
        if self._manager is None:
            return self._build_fail_open_plan(descriptors)

        started_ms = _now_ms()
        deadline_ms = started_ms + max(
            0.0,
            self._observe_plan_wait_ms if wait_timeout_ms is None else wait_timeout_ms,
        )
        cache_keys = [descriptor.cache_key for descriptor in descriptors]
        while True:
            lookups = self._manager.lookup_by_cache_keys(cache_keys)
            entries = self._build_observed_entries(
                descriptors=descriptors,
                lookups=lookups,
            )
            if entries is not None:
                return SourcePlan(
                    request_id=descriptors[0].request_id,
                    entries=entries,
                    near_ready_wait_ms=max(0.0, _now_ms() - started_ms),
                    used_fail_open=False,
                )
            if _now_ms() >= deadline_ms:
                break
            time.sleep(self._poll_interval_ms / 1000.0)

        raise RuntimeError(
            "coordinator source plan unavailable for "
            f"request {descriptors[0].request_id}"
        )

    def _build_observed_entries(
        self,
        *,
        descriptors: list[FallbackDescriptor] | tuple[FallbackDescriptor, ...],
        lookups: tuple[Any, ...],
    ) -> tuple[SourcePlanEntry, ...] | None:
        lookup_by_key = {
            lookup.cache_key: lookup
            for lookup in lookups
        }
        entries: list[SourcePlanEntry] = []
        for descriptor in descriptors:
            lookup = lookup_by_key.get(descriptor.cache_key)
            if lookup is None or lookup.handle is None:
                return None
            if lookup.state is SidecarState.READY:
                entries.append(
                    SourcePlanEntry(
                        request_media_index=descriptor.request_media_index,
                        decision=SourcePlanDecision.USE_SIDECAR,
                        handle=lookup.handle,
                        state=lookup.state,
                        reason="ready_observed_from_manager",
                    )
                )
                continue
            if lookup.state in {
                SidecarState.FALLBACK_CLAIMED,
                SidecarState.FALLBACK_LOCAL_DONE,
            }:
                claimed_rank = parse_ranked_claimer_id(lookup.claimed_by)
                if claimed_rank is None:
                    return None
                entries.append(
                    SourcePlanEntry(
                        request_media_index=descriptor.request_media_index,
                        decision=SourcePlanDecision.FALLBACK,
                        producer_rank=claimed_rank,
                        handle=lookup.handle,
                        state=lookup.state,
                        reason="fallback_observed_from_manager",
                    )
                )
                continue
            return None
        return tuple(
            sorted(entries, key=lambda item: int(item.request_media_index))
        )

    def _wait_and_fetch_remote_fallback(
        self,
        entry: SourcePlanEntry,
    ) -> PreparedArtifact:
        if self._manager is None or entry.handle is None:
            raise RuntimeError("remote fallback fetch requires manager/handle")
        snapshots = self._manager.wait_for_states(
            [entry.handle],
            {SidecarState.READY, SidecarState.FALLBACK_LOCAL_DONE},
            timeout_ms=self._fallback_wait_ms,
            poll_interval_ms=self._poll_interval_ms,
        )
        snapshot = snapshots[0]
        if snapshot.state not in {SidecarState.READY, SidecarState.FALLBACK_LOCAL_DONE}:
            raise RuntimeError(
                "remote fallback artifact unavailable for media index "
                f"{entry.request_media_index}: state={snapshot.state.value}"
            )
        artifact = self._manager.fetch_ready(snapshot.handle)
        if artifact is None:
            raise RuntimeError(
                f"remote fallback artifact missing for media index {entry.request_media_index}"
            )
        return artifact


__all__ = [
    "build_ranked_claimer_id",
    "parse_ranked_claimer_id",
    "SidecarFallbackCoordinator",
    "SidecarFetchBatch",
    "SourcePlan",
    "SourcePlanDecision",
    "SourcePlanEntry",
]

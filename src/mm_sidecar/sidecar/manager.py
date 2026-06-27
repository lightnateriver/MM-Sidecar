from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, cast

from mm_sidecar.contracts import ArtifactDescriptor, ImageScheduleItem

from .cache import CpuMemoryCachePool
from .config import SidecarManagerConfig
from .processor import (
    InlineProcessorWorkerPool,
    ProcessorWorkerPool,
    WorkerResult,
    WorkerTask,
)
from .protocol import (
    FallbackClaimResult,
    FallbackDescriptor,
    PreparedArtifact,
    SidecarHandle,
    SidecarLookupResult,
    SidecarManagerStats,
    SidecarState,
    SidecarStatusSnapshot,
)


def _now_ms() -> float:
    return time.time() * 1000.0


@dataclass(slots=True)
class _ManagedEntry:
    descriptor: FallbackDescriptor
    epoch: int
    state: SidecarState
    updated_at_ms: float
    owner_worker_id: int | None = None
    claimed_by: str | None = None
    artifact_descriptor: ArtifactDescriptor | None = None
    schedule_item: ImageScheduleItem | None = None
    timings_ms: dict[str, float] | None = None
    error_message: str | None = None
    fallback_local_payload: Any | None = None
    fallback_local_timings_ms: dict[str, float] | None = None

    def build_handle(self) -> SidecarHandle:
        return SidecarHandle(
            request_id=self.descriptor.request_id,
            request_media_index=self.descriptor.request_media_index,
            cache_key=self.descriptor.cache_key,
            epoch=self.epoch,
        )

    def to_snapshot(self) -> SidecarStatusSnapshot:
        return SidecarStatusSnapshot(
            handle=self.build_handle(),
            state=self.state,
            epoch=self.epoch,
            updated_at_ms=self.updated_at_ms,
            owner_worker_id=self.owner_worker_id,
            claimed_by=self.claimed_by,
            artifact_descriptor=self.artifact_descriptor,
            schedule_item=self.schedule_item,
            timings_ms=dict(self.timings_ms) if self.timings_ms is not None else None,
            error_message=self.error_message,
        )


class SidecarManager:
    def __init__(
        self,
        config: SidecarManagerConfig | None = None,
        worker_pool: ProcessorWorkerPool | None = None,
        cache_pool: CpuMemoryCachePool | None = None,
    ) -> None:
        self._config = config or SidecarManagerConfig()
        self._cache_pool = cache_pool or CpuMemoryCachePool(self._config.cache)
        self._worker_pool = worker_pool or InlineProcessorWorkerPool(
            worker_count=self._config.workers.worker_count
        )
        self._entries: dict[str, _ManagedEntry] = {}
        self._worker_loads = {
            worker_id: 0 for worker_id in range(self._worker_pool.worker_count)
        }
        self._lock = threading.RLock()
        self._ready_drain_stop = threading.Event()
        self._ready_drain_thread: threading.Thread | None = None
        if hasattr(self._worker_pool, "poll_ready"):
            self._ready_drain_thread = threading.Thread(
                target=self._ready_drain_loop,
                daemon=True,
            )
            self._ready_drain_thread.start()

    def close(self) -> None:
        self._ready_drain_stop.set()
        if self._ready_drain_thread is not None:
            self._ready_drain_thread.join(timeout=1.0)
        self._worker_pool.close()

    def prepare(
        self,
        descriptors: list[FallbackDescriptor] | tuple[FallbackDescriptor, ...],
    ) -> tuple[SidecarHandle, ...]:
        self._drain_results()
        handles: list[SidecarHandle] = []
        with self._lock:
            for descriptor in descriptors:
                entry = self._ensure_entry_for_descriptor(descriptor)
                handles.append(entry.build_handle())
        return tuple(handles)

    def batch_get_status(
        self,
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...],
    ) -> tuple[SidecarStatusSnapshot, ...]:
        self._drain_results()
        with self._lock:
            snapshots: list[SidecarStatusSnapshot] = []
            for handle in handles:
                entry = self._entries.get(handle.cache_key)
                if entry is None:
                    snapshots.append(
                        SidecarStatusSnapshot(
                            handle=handle,
                            state=SidecarState.ABSENT,
                            epoch=handle.epoch,
                            updated_at_ms=_now_ms(),
                        )
                    )
                    continue
                self._refresh_ready_entry(entry)
                if not self._handle_matches_entry(handle, entry):
                    snapshots.append(
                        SidecarStatusSnapshot(
                            handle=handle,
                            state=SidecarState.ABSENT,
                            epoch=handle.epoch,
                            updated_at_ms=_now_ms(),
                            error_message="stale_handle",
                        )
                    )
                    continue
                snapshots.append(entry.to_snapshot())
            return tuple(snapshots)

    def lookup_by_cache_keys(
        self,
        cache_keys: list[str] | tuple[str, ...],
    ) -> tuple[SidecarLookupResult, ...]:
        self._drain_results()
        with self._lock:
            results: list[SidecarLookupResult] = []
            for cache_key in cache_keys:
                entry = self._entries.get(cache_key)
                if entry is None:
                    results.append(
                        SidecarLookupResult(
                            cache_key=cache_key,
                            handle=None,
                            descriptor=None,
                            state=SidecarState.ABSENT,
                            updated_at_ms=_now_ms(),
                            error_message="cache_key_not_found",
                        )
                    )
                    continue
                self._refresh_ready_entry(entry)
                results.append(
                    SidecarLookupResult(
                        cache_key=cache_key,
                        handle=entry.build_handle(),
                        descriptor=entry.descriptor,
                        state=entry.state,
                        updated_at_ms=entry.updated_at_ms,
                        claimed_by=entry.claimed_by,
                        error_message=entry.error_message,
                    )
                )
            return tuple(results)

    def wait_for_states(
        self,
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...],
        target_states: set[SidecarState],
        timeout_ms: float,
        poll_interval_ms: float = 1.0,
    ) -> tuple[SidecarStatusSnapshot, ...]:
        deadline = _now_ms() + timeout_ms
        while True:
            snapshots = self.batch_get_status(handles)
            if all(snapshot.state in target_states for snapshot in snapshots):
                return snapshots
            if _now_ms() >= deadline:
                return snapshots
            time.sleep(poll_interval_ms / 1000.0)

    def wait_for_metadata(
        self,
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...],
        timeout_ms: float,
        poll_interval_ms: float = 1.0,
    ) -> tuple[SidecarStatusSnapshot, ...]:
        deadline = _now_ms() + timeout_ms
        terminal_states = {
            SidecarState.READY,
            SidecarState.FAILED,
            SidecarState.EXPIRED,
            SidecarState.FALLBACK_CLAIMED,
            SidecarState.FALLBACK_LOCAL_DONE,
            SidecarState.BYPASS,
        }
        while True:
            snapshots = self.batch_get_status(handles)
            if all(
                snapshot.schedule_item is not None or snapshot.state in terminal_states
                for snapshot in snapshots
            ):
                return snapshots
            if _now_ms() >= deadline:
                return snapshots
            time.sleep(poll_interval_ms / 1000.0)

    def fetch_ready(self, handle: SidecarHandle) -> PreparedArtifact | None:
        artifacts = self.fetch_ready_batch([handle])
        return artifacts[0] if artifacts else None

    def fetch_ready_batch(
        self,
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...],
    ) -> tuple[PreparedArtifact | None, ...]:
        fetch_started_ms = _now_ms()
        self._drain_ready_results()
        after_ready_drain_ms = _now_ms()
        self._drain_results()
        after_result_drain_ms = _now_ms()
        shared_count = max(1, len(handles))
        with self._lock:
            return tuple(
                self._fetch_ready_locked(
                    handle,
                    fetch_started_ms=fetch_started_ms,
                    after_ready_drain_ms=after_ready_drain_ms,
                    after_result_drain_ms=after_result_drain_ms,
                    shared_count=shared_count,
                )
                for handle in handles
            )

    def _fetch_ready_locked(
        self,
        handle: SidecarHandle,
        *,
        fetch_started_ms: float,
        after_ready_drain_ms: float,
        after_result_drain_ms: float,
        shared_count: int,
    ) -> PreparedArtifact | None:
        entry = self._entries.get(handle.cache_key)
        if entry is None:
            return None
        self._refresh_ready_entry(entry)
        if not self._handle_matches_entry(handle, entry):
            return None
        if entry.state not in {SidecarState.READY, SidecarState.FALLBACK_LOCAL_DONE}:
            return None
        if entry.epoch != handle.epoch:
            return None
        if entry.state is SidecarState.FALLBACK_LOCAL_DONE:
            if (
                entry.artifact_descriptor is None
                or entry.fallback_local_payload is None
            ):
                return None
            cache_get_finished_ms = _now_ms()
            entry.updated_at_ms = _now_ms()
            return PreparedArtifact(
                handle=handle,
                descriptor=entry.artifact_descriptor,
                payload=entry.fallback_local_payload,
                timings_ms=(
                    dict(entry.fallback_local_timings_ms)
                    if entry.fallback_local_timings_ms is not None
                    else (
                        dict(entry.timings_ms)
                        if entry.timings_ms is not None
                        else None
                    )
                ),
                fetch_diagnostics_ms={
                    "manager_fetch_total": max(
                        0.0,
                        _now_ms() - fetch_started_ms,
                    )
                    / shared_count,
                    "manager_ready_drain": max(
                        0.0,
                        after_ready_drain_ms - fetch_started_ms,
                    )
                    / shared_count,
                    "manager_result_drain": max(
                        0.0,
                        after_result_drain_ms - after_ready_drain_ms,
                    )
                    / shared_count,
                    "manager_cache_get": 0.0,
                    "manager_post_cache": max(
                        0.0,
                        _now_ms() - cache_get_finished_ms,
                    ),
                    "manager_fetch_batch_count": 1.0 / shared_count,
                    "manager_fetch_batch_items": 1.0,
                    "manager_local_payload": 1.0,
                },
            )
        cache_get_started_ms = _now_ms()
        cached = self._cache_pool.get(handle.cache_key)
        cache_get_finished_ms = _now_ms()
        if cached is None:
            entry.state = SidecarState.EXPIRED
            entry.updated_at_ms = _now_ms()
            return None
        descriptor, payload = cached
        entry.artifact_descriptor = descriptor
        entry.updated_at_ms = _now_ms()
        return PreparedArtifact(
            handle=handle,
            descriptor=descriptor,
            payload=payload,
            timings_ms=dict(entry.timings_ms) if entry.timings_ms is not None else None,
            fetch_diagnostics_ms={
                "manager_fetch_total": max(
                    0.0,
                    _now_ms() - fetch_started_ms,
                )
                / shared_count,
                "manager_ready_drain": max(
                    0.0,
                    after_ready_drain_ms - fetch_started_ms,
                )
                / shared_count,
                "manager_result_drain": max(
                    0.0,
                    after_result_drain_ms - after_ready_drain_ms,
                )
                / shared_count,
                "manager_cache_get": max(
                    0.0,
                    cache_get_finished_ms - cache_get_started_ms,
                ),
                "manager_post_cache": max(
                    0.0,
                    _now_ms() - cache_get_finished_ms,
                ),
                "manager_fetch_batch_count": 1.0 / shared_count,
                "manager_fetch_batch_items": 1.0,
            },
        )

    def try_fallback_claim(
        self,
        handles: list[SidecarHandle] | tuple[SidecarHandle, ...],
        claimer_id: str,
    ) -> tuple[FallbackClaimResult, ...]:
        self._drain_results()
        results: list[FallbackClaimResult] = []
        with self._lock:
            for handle in handles:
                entry = self._entries.get(handle.cache_key)
                if entry is None:
                    results.append(
                        FallbackClaimResult(
                            handle=handle,
                            granted=True,
                            state=SidecarState.ABSENT,
                            epoch=handle.epoch,
                            claimed_by=claimer_id,
                            updated_at_ms=_now_ms(),
                        )
                    )
                    continue
                self._refresh_ready_entry(entry)
                if not self._handle_matches_entry(handle, entry):
                    current_handle = entry.build_handle()
                    results.append(
                        FallbackClaimResult(
                            handle=current_handle,
                            granted=False,
                            state=entry.state,
                            epoch=entry.epoch,
                            claimed_by=entry.claimed_by,
                            updated_at_ms=_now_ms(),
                            error_message="stale_handle",
                        )
                    )
                    continue
                granted = False
                if entry.state in {
                    SidecarState.QUEUED,
                    SidecarState.SIDECAR_RUNNING,
                    SidecarState.FAILED,
                    SidecarState.EXPIRED,
                    SidecarState.ABSENT,
                }:
                    entry.epoch += 1
                    entry.state = SidecarState.FALLBACK_CLAIMED
                    entry.claimed_by = claimer_id
                    entry.artifact_descriptor = None
                    entry.fallback_local_payload = None
                    entry.fallback_local_timings_ms = None
                    entry.updated_at_ms = _now_ms()
                    entry.error_message = None
                    granted = True
                elif entry.state is SidecarState.FALLBACK_CLAIMED and entry.claimed_by == claimer_id:
                    granted = True
                result_handle = entry.build_handle()
                results.append(
                    FallbackClaimResult(
                        handle=result_handle,
                        granted=granted,
                        state=entry.state,
                        epoch=entry.epoch,
                        claimed_by=entry.claimed_by,
                        updated_at_ms=entry.updated_at_ms,
                        error_message=entry.error_message,
                    )
                )
        return tuple(results)

    def publish_fallback_local_result(
        self,
        handle: SidecarHandle,
        claimer_id: str,
        descriptor: ArtifactDescriptor,
        payload: Any,
        timings_ms: dict[str, float] | None = None,
    ) -> SidecarStatusSnapshot:
        with self._lock:
            entry = self._entries.get(handle.cache_key)
            if entry is None:
                return SidecarStatusSnapshot(
                    handle=handle,
                    state=SidecarState.ABSENT,
                    epoch=handle.epoch,
                    updated_at_ms=_now_ms(),
                    error_message="entry_not_found",
                )
            if not self._handle_matches_entry(handle, entry):
                return SidecarStatusSnapshot(
                    handle=entry.build_handle(),
                    state=entry.state,
                    epoch=entry.epoch,
                    updated_at_ms=entry.updated_at_ms,
                    owner_worker_id=entry.owner_worker_id,
                    claimed_by=entry.claimed_by,
                    artifact_descriptor=entry.artifact_descriptor,
                    schedule_item=entry.schedule_item,
                    timings_ms=(
                        dict(entry.timings_ms) if entry.timings_ms is not None else None
                    ),
                    error_message="stale_handle",
                )
            if entry.claimed_by != claimer_id:
                return SidecarStatusSnapshot(
                    handle=entry.build_handle(),
                    state=entry.state,
                    epoch=entry.epoch,
                    updated_at_ms=entry.updated_at_ms,
                    owner_worker_id=entry.owner_worker_id,
                    claimed_by=entry.claimed_by,
                    artifact_descriptor=entry.artifact_descriptor,
                    schedule_item=entry.schedule_item,
                    timings_ms=(
                        dict(entry.timings_ms) if entry.timings_ms is not None else None
                    ),
                    error_message="claim_mismatch",
                )
            entry.state = SidecarState.FALLBACK_LOCAL_DONE
            entry.artifact_descriptor = descriptor
            entry.fallback_local_payload = payload
            entry.fallback_local_timings_ms = (
                dict(timings_ms) if timings_ms is not None else None
            )
            entry.updated_at_ms = _now_ms()
            entry.error_message = None
            return entry.to_snapshot()

    def mark_fallback_local_done(
        self,
        handle: SidecarHandle,
        claimer_id: str,
    ) -> SidecarStatusSnapshot:
        with self._lock:
            entry = self._entries.get(handle.cache_key)
            if entry is None:
                return SidecarStatusSnapshot(
                    handle=handle,
                    state=SidecarState.ABSENT,
                    epoch=handle.epoch,
                    updated_at_ms=_now_ms(),
                )
            if entry.state is SidecarState.FALLBACK_CLAIMED and entry.claimed_by == claimer_id:
                entry.state = SidecarState.FALLBACK_LOCAL_DONE
                entry.updated_at_ms = _now_ms()
            return entry.to_snapshot()

    def stats(self) -> SidecarManagerStats:
        self._drain_ready_results()
        self._drain_results()
        with self._lock:
            counts = {
                SidecarState.QUEUED: 0,
                SidecarState.SIDECAR_RUNNING: 0,
                SidecarState.READY: 0,
                SidecarState.FAILED: 0,
                SidecarState.FALLBACK_CLAIMED: 0,
            }
            for entry in self._entries.values():
                self._refresh_ready_entry(entry)
                if entry.state in counts:
                    counts[entry.state] += 1
            cache_stats = self._cache_pool.stats()
            return SidecarManagerStats(
                queued_items=counts[SidecarState.QUEUED],
                running_items=counts[SidecarState.SIDECAR_RUNNING],
                ready_items=counts[SidecarState.READY],
                failed_items=counts[SidecarState.FAILED],
                fallback_claimed_items=counts[SidecarState.FALLBACK_CLAIMED],
                reusable_cache_items=cache_stats["reusable_items"],
                reusable_cache_bytes=cache_stats["reusable_bytes"],
                active_inflight_items=cache_stats["inflight_items"],
            )

    def _ensure_entry_for_descriptor(self, descriptor: FallbackDescriptor) -> _ManagedEntry:
        cache_key = descriptor.cache_key
        entry = self._entries.get(cache_key)
        if entry is not None:
            self._refresh_ready_entry(entry)
            if entry.state in {SidecarState.QUEUED, SidecarState.SIDECAR_RUNNING, SidecarState.READY}:
                entry.descriptor = descriptor
                return entry
        cached = self._cache_pool.get(cache_key)
        if cached is not None:
            artifact_descriptor, _ = cached
            entry = _ManagedEntry(
                descriptor=descriptor,
                epoch=entry.epoch + 1 if entry is not None else 1,
                state=SidecarState.READY,
                updated_at_ms=_now_ms(),
                artifact_descriptor=artifact_descriptor,
                timings_ms=entry.timings_ms if entry is not None else None,
            )
            self._entries[cache_key] = entry
            return entry
        worker_id = self._choose_worker_id()
        epoch = entry.epoch + 1 if entry is not None else 1
        entry = _ManagedEntry(
            descriptor=descriptor,
            epoch=epoch,
            state=SidecarState.QUEUED,
            updated_at_ms=_now_ms(),
            owner_worker_id=worker_id,
        )
        self._entries[cache_key] = entry
        self._cache_pool.mark_inflight(cache_key)
        self._worker_loads[worker_id] += 1
        self._worker_pool.submit(
            WorkerTask(
                cache_key=cache_key,
                epoch=epoch,
                assigned_worker_id=worker_id,
                descriptor=descriptor,
            )
        )
        return entry

    def _choose_worker_id(self) -> int:
        return min(self._worker_loads, key=lambda worker_id: self._worker_loads[worker_id])

    def _drain_results(self) -> None:
        results = self._worker_pool.poll()
        if not results:
            return
        with self._lock:
            for result in results:
                self._apply_worker_result(result)

    def _drain_ready_results(self, max_items: int | None = None) -> int:
        poll_ready = getattr(self._worker_pool, "poll_ready", None)
        if poll_ready is None:
            return 0
        results = cast(list[WorkerResult], poll_ready(max_items=max_items))
        if not results:
            return 0
        with self._lock:
            for result in results:
                self._apply_worker_result(result)
        return len(results)

    def _ready_drain_loop(self) -> None:
        while not self._ready_drain_stop.is_set():
            drained = self._drain_ready_results(max_items=1)
            if drained == 0:
                time.sleep(0.0005)

    def _apply_worker_result(self, result: WorkerResult) -> None:
        entry = self._entries.get(result.cache_key)
        if entry is None:
            return
        if result.event_type == "started":
            if result.epoch == entry.epoch and entry.state is SidecarState.QUEUED:
                entry.state = SidecarState.SIDECAR_RUNNING
                entry.owner_worker_id = result.worker_id
                entry.updated_at_ms = result.at_ms
            return

        if result.event_type == "probed":
            if result.epoch != entry.epoch:
                return
            if result.schedule_item is not None:
                entry.schedule_item = result.schedule_item
            if (
                entry.descriptor.orig_size_hw is None
                and result.schedule_item is not None
            ):
                entry.descriptor = FallbackDescriptor(
                    request_id=entry.descriptor.request_id,
                    request_media_index=entry.descriptor.request_media_index,
                    captured_image=entry.descriptor.captured_image,
                    ingress_limits=entry.descriptor.ingress_limits,
                    processor_signature_value=entry.descriptor.processor_signature_value,
                    item_identity=entry.descriptor.item_identity,
                    orig_size_hw=result.schedule_item.orig_size_hw,
                    http_headers=entry.descriptor.http_headers,
                    http_timeout_ms=entry.descriptor.http_timeout_ms,
                    allow_redirects=entry.descriptor.allow_redirects,
                    payload_hint=entry.descriptor.payload_hint,
                )
            if result.timings_ms is not None:
                entry.timings_ms = dict(result.timings_ms)
            entry.updated_at_ms = result.at_ms
            return

        if self._worker_loads.get(result.worker_id, 0) > 0:
            self._worker_loads[result.worker_id] -= 1
        self._cache_pool.clear_inflight(result.cache_key)

        if result.epoch != entry.epoch:
            return
        if entry.state is SidecarState.FALLBACK_CLAIMED:
            return

        if result.event_type == "ready" and result.descriptor is not None and result.payload is not None:
            self._cache_pool.put(result.cache_key, result.descriptor, result.payload)
            entry.state = SidecarState.READY
            entry.artifact_descriptor = result.descriptor
            entry.fallback_local_payload = None
            entry.fallback_local_timings_ms = None
            if result.schedule_item is not None:
                entry.schedule_item = result.schedule_item
            entry.timings_ms = dict(result.timings_ms) if result.timings_ms is not None else None
            entry.error_message = None
            entry.updated_at_ms = result.at_ms
            return

        if result.event_type == "failed":
            entry.state = SidecarState.FAILED
            entry.timings_ms = dict(result.timings_ms) if result.timings_ms is not None else None
            entry.error_message = result.error_message
            entry.updated_at_ms = result.at_ms

    def _refresh_ready_entry(self, entry: _ManagedEntry) -> None:
        if entry.state is not SidecarState.READY:
            return
        cached = self._cache_pool.get(entry.descriptor.cache_key)
        if cached is None:
            entry.state = SidecarState.EXPIRED
            entry.updated_at_ms = _now_ms()
            entry.artifact_descriptor = None
        else:
            entry.artifact_descriptor = cached[0]

    def _handle_matches_entry(
        self,
        handle: SidecarHandle,
        entry: _ManagedEntry,
    ) -> bool:
        current = entry.build_handle()
        return (
            handle.cache_key == current.cache_key
            and handle.epoch == current.epoch
            and handle.request_id == current.request_id
            and handle.request_media_index == current.request_media_index
        )

from __future__ import annotations

import math
import os
import time
from typing import Any

from mm_sidecar.contracts import (
    CapturedImageRef,
    ImageScheduleItem,
    IngressLimits,
    NormalizedImage,
    ProcessorConfig,
    ProcessorSignature,
)
from mm_sidecar.integrations.vllm_patch.context import RequestCapture
from mm_sidecar.sidecar import (
    InlineProcessorWorkerPool,
    MultiProcessProcessorWorkerPool,
    SidecarFallbackCoordinator,
    SidecarManager,
    SidecarManagerConfig,
    SidecarState,
    WorkerPoolConfig,
)
from mm_sidecar.sidecar.coordinator import SourcePlan, SourcePlanEntry, SourcePlanDecision
from mm_sidecar.sidecar.protocol import FallbackDescriptor, SidecarHandle


def _vllm_env(name: str, default: Any) -> Any:
    try:
        import vllm.envs as vllm_envs
    except Exception:
        return default
    return getattr(vllm_envs, name, default)


def _safe_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _available_cpu_ids() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
        if cpu_ids:
            return cpu_ids
    cpu_count = os.cpu_count() or 1
    return tuple(range(cpu_count))


def _default_worker_count() -> int:
    return max(1, min(32, len(_available_cpu_ids())))


def _default_cpu_affinity_map(worker_count: int) -> tuple[tuple[int, ...], ...]:
    cpu_ids = _available_cpu_ids()
    return tuple((cpu_ids[index % len(cpu_ids)],) for index in range(worker_count))


def _worker_pool_mode() -> str:
    return os.getenv("MM_SIDECAR_WORKER_POOL_MODE", "process").strip().lower() or "process"


def _build_manager_config_from_env() -> SidecarManagerConfig:
    worker_count = _safe_int(
        os.getenv("MM_SIDECAR_WORKER_COUNT"),
        _default_worker_count(),
    )
    reusable_bytes = _safe_int(
        os.getenv("MM_SIDECAR_REUSABLE_CACHE_BYTES"),
        512 * 1024 * 1024,
    )
    return SidecarManagerConfig(
        cache=SidecarManagerConfig().cache.__class__(
            max_reusable_bytes=reusable_bytes,
            reusable_entry_ttl_s=float(
                os.getenv("MM_SIDECAR_REUSABLE_TTL_S", "300.0")
            ),
        ),
        workers=WorkerPoolConfig(
            worker_count=worker_count,
            cpu_affinity_map=_default_cpu_affinity_map(worker_count),
            start_method=os.getenv("MM_SIDECAR_WORKER_START_METHOD", "fork"),
        ),
    )


def create_sidecar_manager_from_env() -> SidecarManager:
    config = _build_manager_config_from_env()
    mode = _worker_pool_mode()
    if mode == "inline":
        worker_pool = InlineProcessorWorkerPool(worker_count=config.workers.worker_count)
    else:
        worker_pool = MultiProcessProcessorWorkerPool(config.workers)
    return SidecarManager(config=config, worker_pool=worker_pool)


def describe_sidecar_runtime_config() -> dict[str, Any]:
    config = _build_manager_config_from_env()
    return {
        "worker_pool_mode": _worker_pool_mode(),
        "worker_count": config.workers.worker_count,
        "cpu_affinity_map": [list(item) for item in config.workers.cpu_affinity_map or ()],
        "reusable_cache_bytes": config.cache.max_reusable_bytes,
        "reusable_ttl_s": config.cache.reusable_entry_ttl_s,
    }


def _round_to_factor(value: int, factor: int) -> int:
    return max(factor, int(round(value / factor)) * factor)


def _fallback_qwen_smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    height = _round_to_factor(height, factor)
    width = _round_to_factor(width, factor)
    area = height * width
    if area < min_pixels:
        scale = math.sqrt(min_pixels / max(area, 1))
        height = _round_to_factor(int(height * scale), factor)
        width = _round_to_factor(int(width * scale), factor)
    elif area > max_pixels:
        scale = math.sqrt(max_pixels / max(area, 1))
        height = _round_to_factor(max(factor, int(height * scale)), factor)
        width = _round_to_factor(max(factor, int(width * scale)), factor)
    return height, width


def _qwen_smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            smart_resize as qwen_smart_resize,
        )
    except Exception:
        return _fallback_qwen_smart_resize(
            height=height,
            width=width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    return qwen_smart_resize(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _build_ingress_limits() -> IngressLimits:
    return IngressLimits(
        max_image_count=_safe_int(os.getenv("MM_SIDECAR_MAX_IMAGE_COUNT"), 40),
        max_encoded_bytes=_safe_int(
            os.getenv("MM_SIDECAR_MAX_ENCODED_BYTES"),
            64 * 1024 * 1024,
        ),
        max_decoded_bytes=_safe_int(
            os.getenv("MM_SIDECAR_MAX_DECODED_BYTES"),
            512 * 1024 * 1024,
        ),
        max_pixels_per_image=_safe_int(
            os.getenv("MM_SIDECAR_MAX_PIXELS_PER_IMAGE"),
            1280 * 28 * 28,
        ),
    )


def _resolve_processor_signature(renderer: Any, params: Any) -> ProcessorSignature:
    hf_config = renderer.model_config.hf_config
    vision_config = getattr(hf_config, "vision_config", None)
    patch_size = int(getattr(vision_config, "patch_size", 14))
    merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
    temporal_patch_size = int(getattr(vision_config, "temporal_patch_size", 1))
    mm_processor_kwargs = getattr(params, "mm_processor_kwargs", {}) or {}
    min_pixels = int(mm_processor_kwargs.get("min_pixels", 28 * 28))
    max_pixels = int(mm_processor_kwargs.get("max_pixels", 1280 * 28 * 28))
    do_resize = bool(mm_processor_kwargs.get("do_resize", True))
    model_name = (
        getattr(renderer.model_config, "model", None)
        or getattr(hf_config, "_name_or_path", None)
        or hf_config.__class__.__name__
    )
    revision = (
        getattr(renderer.model_config, "revision", None)
        or getattr(hf_config, "_commit_hash", None)
        or "unknown"
    )
    config = ProcessorConfig(
        model_name=str(model_name),
        revision=str(revision),
        processor_name=hf_config.__class__.__name__,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        do_resize=do_resize,
    )
    return ProcessorSignature.from_config(config)


def _build_schedule_item(
    item_index: int,
    normalized_image: NormalizedImage,
    processor_signature: ProcessorSignature,
    params: Any,
) -> ImageScheduleItem:
    mm_processor_kwargs = getattr(params, "mm_processor_kwargs", {}) or {}
    patch_size = 14
    merge_size = 2
    temporal_patch_size = 1

    signature_parts = {
        piece.split("=", 1)[0]: piece.split("=", 1)[1]
        for piece in processor_signature.value.split("|")
        if "=" in piece
    }
    patch_size = int(signature_parts.get("patch", patch_size))
    merge_size = int(signature_parts.get("merge", merge_size))
    temporal_patch_size = int(signature_parts.get("temporal", temporal_patch_size))
    do_resize = bool(int(signature_parts.get("do_resize", "1")))
    min_pixels = int(signature_parts.get("min_pixels", str(28 * 28)))
    max_pixels = int(signature_parts.get("max_pixels", str(1280 * 28 * 28)))

    orig_h, orig_w = normalized_image.orig_size_hw
    if do_resize:
        resized_h, resized_w = _qwen_smart_resize(
            height=orig_h,
            width=orig_w,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    else:
        resized_h = max(patch_size * merge_size, orig_h)
        resized_w = max(patch_size * merge_size, orig_w)

    grid_t = max(1, 1 // temporal_patch_size)
    grid_h = max(1, resized_h // patch_size)
    grid_w = max(1, resized_w // patch_size)
    placeholder_token_count = (grid_t * grid_h * grid_w) // (merge_size**2)

    return ImageScheduleItem(
        item_index=item_index,
        item_identity=normalized_image.source_ref.source_key,
        processor_signature=processor_signature,
        orig_size_hw=normalized_image.orig_size_hw,
        preprocessed_size_hw=(resized_h, resized_w),
        image_grid_thw=(grid_t, grid_h, grid_w),
        placeholder_token_count=max(1, placeholder_token_count),
    )


def _build_captured_image_from_normalized(
    normalized_image: NormalizedImage,
) -> CapturedImageRef:
    return CapturedImageRef(
        source_ref=normalized_image.source_ref,
        mime_type=normalized_image.mime_type,
        byte_size=normalized_image.byte_size,
        local_materialized_path=normalized_image.local_materialized_path,
    )


def _build_fallback_descriptor_from_capture_ref(
    *,
    request_id: str,
    item_index: int,
    captured_ref: CapturedImageRef,
    processor_signature: ProcessorSignature,
    ingress_limits: IngressLimits,
    image_timeout_ms: int,
    allow_redirects: bool,
    http_headers: tuple[tuple[str, str], ...] = (),
) -> FallbackDescriptor:
    return FallbackDescriptor(
        request_id=request_id,
        request_media_index=item_index,
        captured_image=captured_ref,
        ingress_limits=ingress_limits,
        processor_signature_value=processor_signature.value,
        item_identity=captured_ref.source_ref.source_key,
        orig_size_hw=None,
        http_headers=http_headers,
        http_timeout_ms=image_timeout_ms,
        allow_redirects=allow_redirects,
    )


def build_fallback_descriptors(
    capture: RequestCapture,
    renderer: Any,
    params: Any,
) -> list[FallbackDescriptor]:
    processor_signature = _resolve_processor_signature(renderer, params)
    ingress_limits = _build_ingress_limits()
    image_timeout_ms = int(_vllm_env("VLLM_IMAGE_FETCH_TIMEOUT", 5)) * 1000
    allow_redirects = bool(_vllm_env("VLLM_MEDIA_URL_ALLOW_REDIRECTS", True))
    media_io_kwargs = getattr(params, "media_io_kwargs", {}) or {}
    image_media_io_kwargs = media_io_kwargs.get("image") or {}
    raw_http_headers = image_media_io_kwargs.get("headers") or {}
    http_headers = tuple(
        (str(key), str(value))
        for key, value in raw_http_headers.items()
    )
    descriptors: list[FallbackDescriptor] = []
    descriptor_by_index: dict[int, FallbackDescriptor] = {}
    for item_index, _media_uuid, captured_ref in capture.iter_captured_image_refs():
        descriptor_by_index[int(item_index)] = _build_fallback_descriptor_from_capture_ref(
            request_id=capture.request_id,
            item_index=item_index,
            captured_ref=captured_ref,
            processor_signature=processor_signature,
            ingress_limits=ingress_limits,
            image_timeout_ms=image_timeout_ms,
            allow_redirects=allow_redirects,
            http_headers=http_headers,
        )

    for item_index, _media_uuid, normalized_image in capture.iter_normalized_images():
        existing = descriptor_by_index.get(int(item_index))
        if existing is not None:
            descriptor_by_index[int(item_index)] = FallbackDescriptor(
                request_id=existing.request_id,
                request_media_index=existing.request_media_index,
                captured_image=existing.captured_image,
                ingress_limits=existing.ingress_limits,
                processor_signature_value=existing.processor_signature_value,
                item_identity=existing.item_identity,
                orig_size_hw=normalized_image.orig_size_hw,
                http_headers=existing.http_headers,
                http_timeout_ms=existing.http_timeout_ms,
                allow_redirects=existing.allow_redirects,
                payload_hint=existing.payload_hint,
            )
            continue
        descriptor_by_index[int(item_index)] = FallbackDescriptor(
            request_id=capture.request_id,
            request_media_index=item_index,
            captured_image=_build_captured_image_from_normalized(normalized_image),
            ingress_limits=ingress_limits,
            processor_signature_value=processor_signature.value,
            item_identity=normalized_image.source_ref.source_key,
            orig_size_hw=normalized_image.orig_size_hw,
            http_headers=http_headers,
            http_timeout_ms=image_timeout_ms,
            allow_redirects=allow_redirects,
        )

    for item_index in sorted(descriptor_by_index):
        descriptors.append(descriptor_by_index[item_index])
    return descriptors


def prepare_single_capture_item_for_sidecar(
    capture: RequestCapture,
    renderer: Any,
    params: Any,
    *,
    item_index: int,
    captured_ref: CapturedImageRef,
) -> dict[str, Any] | None:
    manager = capture.sidecar_manager
    if manager is None:
        return None

    existing_descriptor = capture.get_prepared_descriptor(item_index)
    existing_handle = capture.get_prepared_handle(item_index)
    if existing_descriptor is not None and existing_handle is not None:
        return {
            "descriptor": existing_descriptor,
            "handle": existing_handle,
            "prepared": False,
        }

    processor_signature = _resolve_processor_signature(renderer, params)
    ingress_limits = _build_ingress_limits()
    media_io_kwargs = getattr(params, "media_io_kwargs", {}) or {}
    image_media_io_kwargs = media_io_kwargs.get("image") or {}
    raw_http_headers = image_media_io_kwargs.get("headers") or {}
    http_headers = tuple(
        (str(key), str(value))
        for key, value in raw_http_headers.items()
    )
    descriptor = _build_fallback_descriptor_from_capture_ref(
        request_id=capture.request_id,
        item_index=item_index,
        captured_ref=captured_ref,
        processor_signature=processor_signature,
            ingress_limits=ingress_limits,
            image_timeout_ms=int(_vllm_env("VLLM_IMAGE_FETCH_TIMEOUT", 5)) * 1000,
            allow_redirects=bool(_vllm_env("VLLM_MEDIA_URL_ALLOW_REDIRECTS", True)),
            http_headers=http_headers,
        )
    handles = manager.prepare([descriptor])
    handle = handles[0]
    capture.add_prepared_sidecar_item(item_index, descriptor, handle)
    return {
        "descriptor": descriptor,
        "handle": handle,
        "prepared": True,
    }


def _serialize_handle(handle: Any) -> dict[str, Any]:
    return {
        "request_id": handle.request_id,
        "request_media_index": handle.request_media_index,
        "cache_key": handle.cache_key,
        "epoch": handle.epoch,
    }


def _serialize_schedule_item(
    item: ImageScheduleItem,
    descriptor: FallbackDescriptor,
) -> dict[str, Any]:
    return {
        "request_media_index": item.item_index,
        "item_identity": item.item_identity,
        "transport": descriptor.captured_image.source_ref.transport.value,
        "source_key": descriptor.captured_image.source_ref.source_key,
        "orig_size_hw": list(item.orig_size_hw),
        "preprocessed_size_hw": list(item.preprocessed_size_hw),
        "image_grid_thw": list(item.image_grid_thw),
        "placeholder_token_count": item.placeholder_token_count,
        "processor_signature": item.processor_signature.value,
    }


def _serialize_planned_items(
    planned_items: list[ImageScheduleItem],
    descriptors: list[FallbackDescriptor],
) -> list[dict[str, Any]]:
    descriptor_by_index = {
        int(descriptor.request_media_index): descriptor
        for descriptor in descriptors
    }
    serialized: list[dict[str, Any]] = []
    for item in planned_items:
        descriptor = descriptor_by_index.get(int(item.item_index))
        if descriptor is None:
            continue
        serialized.append(_serialize_schedule_item(item, descriptor))
    return serialized


def _schedule_item_from_snapshot(
    snapshot: Any,
) -> ImageScheduleItem | None:
    item = getattr(snapshot, "schedule_item", None)
    return item if isinstance(item, ImageScheduleItem) else None


def _schedule_items_from_snapshots(
    descriptors: list[FallbackDescriptor],
    snapshots: tuple[Any, ...],
) -> list[ImageScheduleItem]:
    snapshot_by_index = {
        int(snapshot.handle.request_media_index): snapshot
        for snapshot in snapshots
        if getattr(snapshot, "handle", None) is not None
    }
    planned_items: list[ImageScheduleItem] = []
    for descriptor in descriptors:
        snapshot = snapshot_by_index.get(int(descriptor.request_media_index))
        if snapshot is None:
            continue
        item = _schedule_item_from_snapshot(snapshot)
        if item is not None:
            planned_items.append(item)
    return planned_items


def _normalized_image_from_descriptor(
    descriptor: FallbackDescriptor,
) -> NormalizedImage | None:
    if descriptor.orig_size_hw is None:
        return None
    return NormalizedImage(
        source_ref=descriptor.captured_image.source_ref,
        orig_size_hw=descriptor.orig_size_hw,
        mime_type=(
            descriptor.captured_image.mime_type
            or descriptor.captured_image.source_ref.mime_type
            or "image/unknown"
        ),
        byte_size=descriptor.captured_image.byte_size,
        decoded_size_hw=descriptor.orig_size_hw,
        local_materialized_path=descriptor.captured_image.local_materialized_path,
    )


def _fallback_schedule_items_from_descriptors(
    descriptors: list[FallbackDescriptor],
    capture: RequestCapture,
    params: Any,
) -> list[ImageScheduleItem]:
    normalized_by_index = {
        int(item_index): normalized_image
        for item_index, _media_uuid, normalized_image in capture.iter_normalized_images()
    }
    planned_items: list[ImageScheduleItem] = []
    for descriptor in descriptors:
        item_index = int(descriptor.request_media_index)
        normalized_image = normalized_by_index.get(item_index)
        if normalized_image is None:
            normalized_image = _normalized_image_from_descriptor(descriptor)
        if normalized_image is None:
            continue
        planned_items.append(
            _build_schedule_item(
                item_index=item_index,
                normalized_image=normalized_image,
                processor_signature=ProcessorSignature(
                    value=descriptor.processor_signature_value
                ),
                params=params,
            )
        )
    return planned_items


def _descriptor_only_capture_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _descriptor_only_metadata_retry_ms() -> float:
    return float(
        os.getenv("MM_SIDECAR_DESCRIPTOR_ONLY_METADATA_WAIT_MS", "250.0")
    )


def _merge_schedule_items(
    descriptors: list[FallbackDescriptor],
    snapshots: tuple[Any, ...],
    capture: RequestCapture,
    params: Any,
) -> list[ImageScheduleItem]:
    planned_by_index = {
        int(item.item_index): item
        for item in _schedule_items_from_snapshots(descriptors, snapshots)
    }
    fallback_items = _fallback_schedule_items_from_descriptors(
        descriptors,
        capture,
        params,
    )
    for item in fallback_items:
        planned_by_index.setdefault(int(item.item_index), item)

    return [
        planned_by_index[int(descriptor.request_media_index)]
        for descriptor in descriptors
        if int(descriptor.request_media_index) in planned_by_index
    ]


def _serialize_snapshot(snapshot: Any) -> dict[str, Any]:
    return {
        "handle": _serialize_handle(snapshot.handle),
        "state": snapshot.state.value,
        "epoch": snapshot.epoch,
        "updated_at_ms": snapshot.updated_at_ms,
        "owner_worker_id": snapshot.owner_worker_id,
        "claimed_by": snapshot.claimed_by,
        "artifact_id": (
            snapshot.artifact_descriptor.artifact_id
            if snapshot.artifact_descriptor is not None
            else None
        ),
        "timings_ms": dict(snapshot.timings_ms) if snapshot.timings_ms is not None else None,
        "error_message": snapshot.error_message,
    }


def _serialize_source_plan_entry(entry: SourcePlanEntry) -> dict[str, Any]:
    return {
        "request_media_index": entry.request_media_index,
        "decision": entry.decision.value,
        "producer_rank": entry.producer_rank,
        "handle": _serialize_handle(entry.handle) if entry.handle is not None else None,
        "state": entry.state.value if entry.state is not None else None,
        "reason": entry.reason,
    }


def _serialize_source_plan(plan: SourcePlan) -> dict[str, Any]:
    return {
        "request_id": plan.request_id,
        "near_ready_wait_ms": plan.near_ready_wait_ms,
        "used_fail_open": plan.used_fail_open,
        "entries": [_serialize_source_plan_entry(entry) for entry in plan.entries],
    }


def _deserialize_handle(payload: dict[str, Any]) -> SidecarHandle:
    return SidecarHandle(
        request_id=str(payload["request_id"]),
        request_media_index=int(payload["request_media_index"]),
        cache_key=str(payload["cache_key"]),
        epoch=int(payload["epoch"]),
    )


def refresh_capture_for_debug(capture: RequestCapture) -> None:
    payload = capture.sidecar_prepare
    manager = capture.sidecar_manager
    if payload is None or manager is None:
        return
    raw_handles = payload.get("handles")
    if not isinstance(raw_handles, list) or not raw_handles:
        return

    handles = [
        _deserialize_handle(item)
        for item in raw_handles
        if isinstance(item, dict)
    ]
    if not handles:
        return

    snapshots = manager.batch_get_status(handles)
    payload["final_statuses"] = [_serialize_snapshot(snapshot) for snapshot in snapshots]
    payload["ready_item_count"] = sum(
        1 for snapshot in snapshots if snapshot.state is SidecarState.READY
    )
    payload["final_failed_item_count"] = sum(
        1 for snapshot in snapshots if snapshot.state is SidecarState.FAILED
    )

    timing_rows = [
        snapshot.timings_ms
        for snapshot in snapshots
        if snapshot.timings_ms is not None
    ]
    if timing_rows:
        stage_names = sorted({key for row in timing_rows for key in row})
        payload["worker_timings_ms"] = {
            "avg": {
                stage: sum(row.get(stage, 0.0) for row in timing_rows) / len(timing_rows)
                for stage in stage_names
            },
            "max": {
                stage: max(row.get(stage, 0.0) for row in timing_rows)
                for stage in stage_names
            },
            "sum": {
                stage: sum(row.get(stage, 0.0) for row in timing_rows)
                for stage in stage_names
            },
        }


def prepare_capture_for_sidecar(
    capture: RequestCapture,
    renderer: Any,
    params: Any,
) -> dict[str, Any] | None:
    if capture.sidecar_prepare is not None:
        return capture.sidecar_prepare

    if not capture.iter_captured_image_refs() and not capture.iter_normalized_images():
        payload = {
            "enabled": True,
            "prepared_image_count": 0,
            "reason": "no_images_captured",
        }
        capture.set_sidecar_prepare(payload)
        return payload

    descriptors = build_fallback_descriptors(capture, renderer, params)
    for descriptor in descriptors:
        if capture.get_prepared_descriptor(int(descriptor.request_media_index)) is None:
            capture.add_prepared_sidecar_item(
                int(descriptor.request_media_index),
                descriptor,
                capture.get_prepared_handle(int(descriptor.request_media_index)),
            )

    manager = capture.sidecar_manager
    if manager is None:
        planned_items = [
            _build_schedule_item(
                item_index=item_index,
                normalized_image=normalized_image,
                processor_signature=_resolve_processor_signature(renderer, params),
                params=params,
            )
            for item_index, _media_uuid, normalized_image in capture.iter_normalized_images()
        ]
        coordinator = SidecarFallbackCoordinator(
            manager=None,
            claimer_id=capture.request_id,
            producer_rank=0,
            near_ready_wait_ms=0.0,
        )
        source_plan_preview = coordinator.preview_source_plan(descriptors=descriptors)
        payload = {
            "enabled": False,
            "prepared_image_count": len(descriptors),
            "total_placeholder_token_count": sum(
                item.placeholder_token_count for item in planned_items
            ),
            "processor_signature": (
                descriptors[0].processor_signature_value if descriptors else None
            ),
            "planned_items": _serialize_planned_items(planned_items, descriptors),
            "source_plan_preview": _serialize_source_plan(source_plan_preview),
            "handles": [],
            "initial_statuses": [],
            "reason": "sidecar_manager_unavailable",
            "timings_ms": {
                "manager_prepare": 0.0,
                "batch_get_status": 0.0,
                "descriptor_only_metadata_retry": 0.0,
                "source_plan_preview": 0.0,
                "manager_stats": 0.0,
                "total": 0.0,
            },
            "manager_stats": None,
        }
        capture.set_sidecar_prepare(payload)
        return payload

    prepared_by_index = {
        item_index: (descriptor, handle)
        for item_index, descriptor, handle in capture.iter_prepared_sidecar_items()
        if handle is not None
    }
    handles: list[SidecarHandle] = []
    descriptors_for_prepare: list[FallbackDescriptor] = []
    for descriptor in descriptors:
        prepared = prepared_by_index.get(int(descriptor.request_media_index))
        if prepared is not None:
            handles.append(prepared[1])
        else:
            descriptors_for_prepare.append(descriptor)

    prepare_start = time.perf_counter()
    if descriptors_for_prepare:
        new_handles = manager.prepare(descriptors_for_prepare)
        handle_by_index = {
            int(descriptor.request_media_index): handle
            for descriptor, handle in zip(descriptors_for_prepare, new_handles)
        }
        for descriptor in descriptors_for_prepare:
            handle = handle_by_index[int(descriptor.request_media_index)]
            capture.add_prepared_sidecar_item(
                int(descriptor.request_media_index),
                descriptor,
                handle,
            )
    else:
        handle_by_index = {}
    after_prepare = time.perf_counter()
    final_handles = [
        capture.get_prepared_handle(int(descriptor.request_media_index))
        or handle_by_index[int(descriptor.request_media_index)]
        for descriptor in descriptors
    ]
    handles = tuple(final_handles)
    metadata_wait_total_ms = 0.0
    descriptor_only_metadata_retry_ms = 0.0
    metadata_wait_ms = float(os.getenv("MM_SIDECAR_METADATA_WAIT_MS", "2.0"))
    wait_start = time.perf_counter()
    if metadata_wait_ms > 0.0:
        snapshots = manager.wait_for_metadata(handles, timeout_ms=metadata_wait_ms)
    else:
        snapshots = manager.batch_get_status(handles)
    metadata_wait_total_ms += (time.perf_counter() - wait_start) * 1000.0
    planned_items = _merge_schedule_items(
        descriptors,
        snapshots,
        capture,
        params,
    )
    if (
        _descriptor_only_capture_enabled()
        and len(planned_items) < len(descriptors)
        and not capture.iter_normalized_images()
    ):
        retry_timeout_ms = _descriptor_only_metadata_retry_ms()
        if retry_timeout_ms > 0.0:
            retry_start = time.perf_counter()
            snapshots = manager.wait_for_metadata(
                handles,
                timeout_ms=retry_timeout_ms,
            )
            descriptor_only_metadata_retry_ms = (
                time.perf_counter() - retry_start
            ) * 1000.0
            metadata_wait_total_ms += descriptor_only_metadata_retry_ms
            planned_items = _merge_schedule_items(
                descriptors,
                snapshots,
                capture,
                params,
            )
        if len(planned_items) < len(descriptors):
            capture.add_error(
                "descriptor-only metadata incomplete after retry: "
                f"planned={len(planned_items)} expected={len(descriptors)}"
            )
    after_status = time.perf_counter()

    payload = {
        "enabled": True,
        "prepared_image_count": len(descriptors),
        "total_placeholder_token_count": sum(
            item.placeholder_token_count for item in planned_items
        ),
        "processor_signature": (
            descriptors[0].processor_signature_value if descriptors else None
        ),
        "planned_items": _serialize_planned_items(planned_items, descriptors),
        "source_plan_preview": None,
        "handles": [_serialize_handle(handle) for handle in handles],
        "initial_statuses": [_serialize_snapshot(snapshot) for snapshot in snapshots],
        "timings_ms": {
            "manager_prepare": (after_prepare - prepare_start) * 1000.0,
            "batch_get_status": metadata_wait_total_ms,
            "descriptor_only_metadata_retry": descriptor_only_metadata_retry_ms,
            "source_plan_preview": 0.0,
            "manager_stats": 0.0,
            "total": (after_status - prepare_start) * 1000.0,
        },
        "manager_stats": None,
    }
    capture.set_sidecar_prepare(payload)
    return payload

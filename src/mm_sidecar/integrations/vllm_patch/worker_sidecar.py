from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import importlib
import math
import os
import sys
import time
from threading import Lock
from typing import Any

from mm_sidecar.integrations.vllm_patch.carrier import (
    SidecarRequestPlan,
    decode_sidecar_request_plan,
    get_sidecar_payload_from_params,
)
from mm_sidecar.sidecar import (
    SidecarFallbackCoordinator,
    SidecarFetchBatch,
    SourcePlanDecision,
    build_ranked_claimer_id,
    connect_sidecar_client_from_env,
)
from mm_sidecar.sidecar.processor import run_descriptor_locally
from mm_sidecar.integrations.vllm_patch.qwen_adapter import (
    sidecar_artifact_to_qwen_mm_kwargs_item,
    get_request_payload_from_qwen_mm_kwargs_item,
    is_synthetic_qwen_mm_kwargs_item,
    planned_item_to_vit_dp_placeholder_qwen_mm_kwargs_item,
    replace_feature_data_from_sidecar_artifacts,
)


_PATCH_MARKER_ATTR = "_mm_sidecar_worker_patch_installed"
_VISION_PATCH_MARKER_ATTR = "_mm_sidecar_vit_dp_shard_fetch_patch_installed"
_VISION_ORIGINAL_FN_ATTR = "_mm_sidecar_original_run_dp_sharded_mrope_vision_model"
_VIT_DP_SHARD_CONTEXT_ATTR = "_mm_sidecar_vit_dp_shard_fetch_context"
_PREPARED_SCHEDULER_OUTPUT_ID_ATTR = (
    "_mm_sidecar_last_prepared_scheduler_output_id"
)
_MISSING_ATTR = object()


@dataclass(frozen=True, slots=True)
class WorkerSidecarBinding:
    request_id: str
    enabled: bool
    processor_signature: str | None
    prepared_image_count: int
    total_placeholder_token_count: int
    plan_payload: dict[str, Any]
    decoded_plan: Any


@dataclass(frozen=True, slots=True)
class TpWorkerRole:
    local_rank: int
    world_size: int
    coordinator_rank: int
    is_coordinator: bool


@dataclass(frozen=True, slots=True)
class WorkerShardSelection:
    scheduled_image_indexes: tuple[int, ...]
    local_image_indexes: tuple[int, ...]
    local_descriptors: tuple[Any, ...]
    local_handles: tuple[Any, ...]
    local_planned_items: tuple[dict[str, Any], ...]
    remote_image_indexes: tuple[int, ...]
    use_vit_data_parallel: bool


@dataclass(frozen=True, slots=True)
class LocalShardExecutionPlan:
    req_id: str
    req_state: Any
    binding: WorkerSidecarBinding
    descriptors: tuple[Any, ...]
    handles: tuple[Any, ...]
    source_plan: Any
    image_features: tuple[Any, ...]
    image_input_ids: tuple[int, ...]
    local_indices: tuple[int, ...]
    order: tuple[int, ...]
    counts: tuple[int, ...]
    loads: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VitDpDirectEncodeResult:
    handled_request_ids: tuple[str, ...]
    fallback_scheduled: dict[str, list[int]]


@dataclass(frozen=True, slots=True)
class VitDpShardFetchItem:
    req_id: str
    req_state: Any
    binding: WorkerSidecarBinding
    feature: Any
    request_media_index: int
    descriptor: Any
    handle: Any
    planned_item: dict[str, Any] | None
    grid_thw: tuple[int, int, int]


@dataclass(slots=True)
class VitDpShardFetchContext:
    model_runner: Any
    role: TpWorkerRole
    items: tuple[VitDpShardFetchItem, ...]
    cursor: int = 0


class _DeferVitDpShardFetchToVisionPatch(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, float]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


_CLIENT_UNSET = object()
_CLIENT_LOCK = Lock()
_CLIENT_CACHE: Any | None | object = _CLIENT_UNSET


def _worker_debug_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_WORKER_DEBUG", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _emit_worker_debug(message: str) -> None:
    if not _worker_debug_enabled():
        return
    sys.stderr.write(f"mm-sidecar worker debug: {message}\n")


def _worker_fetch_profile_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_WORKER_FETCH_PROFILE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_mm_merge_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_ENABLE_SAFE_MM_MERGE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _vit_dp_direct_encode_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_ENABLE_VIT_DP_DIRECT_ENCODE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _vit_dp_shard_fetch_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_ENABLE_VIT_DP_SHARD_FETCH", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _batch_fetch_ready_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_ENABLE_BATCH_FETCH_READY", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _native_vit_dp_full_replacement_mode(
    model_runner: Any,
    role: TpWorkerRole,
) -> bool:
    return (
        role.world_size > 1
        and _uses_vit_data_parallel(model_runner)
        and not _vit_dp_direct_encode_enabled()
    )


def _read_non_negative_int_env(*names: str) -> int | None:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = raw.strip()
        if not value:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed >= 0:
            return parsed
    return None


def _call_rank_getter(module_name: str, attr_name: str) -> int | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    getter = getattr(module, attr_name, None)
    if getter is None:
        return None
    try:
        value = getter()
    except Exception:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _resolve_tp_worker_role() -> TpWorkerRole:
    local_rank = _read_non_negative_int_env("MM_SIDECAR_TP_LOCAL_RANK")
    if local_rank is None:
        for module_name, attr_name in (
            ("vllm.distributed", "get_tensor_model_parallel_rank"),
            ("vllm.distributed.parallel_state", "get_tensor_model_parallel_rank"),
            (
                "vllm.model_executor.parallel_utils.parallel_state",
                "get_tensor_model_parallel_rank",
            ),
        ):
            local_rank = _call_rank_getter(module_name, attr_name)
            if local_rank is not None:
                break
    if local_rank is None:
        local_rank = _read_non_negative_int_env("TP_RANK", "LOCAL_RANK", "RANK")
    if local_rank is None:
        local_rank = 0

    world_size = _read_non_negative_int_env("MM_SIDECAR_TP_WORLD_SIZE")
    if world_size is None:
        for module_name, attr_name in (
            ("vllm.distributed", "get_tensor_model_parallel_world_size"),
            ("vllm.distributed.parallel_state", "get_tensor_model_parallel_world_size"),
            (
                "vllm.model_executor.parallel_utils.parallel_state",
                "get_tensor_model_parallel_world_size",
            ),
        ):
            world_size = _call_rank_getter(module_name, attr_name)
            if world_size is not None:
                break
    if world_size is None:
        world_size = _read_non_negative_int_env("TP_WORLD_SIZE", "WORLD_SIZE")
    if world_size is None or world_size <= 0:
        world_size = 1

    coordinator_rank = _read_non_negative_int_env("MM_SIDECAR_TP_COORDINATOR_RANK")
    if coordinator_rank is None:
        coordinator_rank = 0
    coordinator_rank = max(0, min(coordinator_rank, world_size - 1))

    return TpWorkerRole(
        local_rank=local_rank,
        world_size=world_size,
        coordinator_rank=coordinator_rank,
        is_coordinator=(local_rank == coordinator_rank),
    )


def _peer_plan_wait_ms() -> float:
    raw = os.getenv("MM_SIDECAR_TP_COORDINATOR_WAIT_MS")
    if raw is None or not raw.strip():
        return 50.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 50.0


def _remote_fallback_wait_ms() -> float:
    raw = os.getenv("MM_SIDECAR_TP_FALLBACK_WAIT_MS")
    if raw is None or not raw.strip():
        return 1_000.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1_000.0


def _native_vit_dp_ready_wait_ms() -> float:
    raw = os.getenv("MM_SIDECAR_NATIVE_VIT_DP_READY_WAIT_MS")
    if raw is None or not raw.strip():
        return 50.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 50.0


def _vit_dp_direct_cache_ready_wait_ms() -> float:
    raw = os.getenv("MM_SIDECAR_VIT_DP_DIRECT_CACHE_READY_WAIT_MS")
    if raw is None or not raw.strip():
        return 2.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 2.0


def _running_ready_wait_ms() -> float:
    raw = os.getenv("MM_SIDECAR_RUNNING_READY_WAIT_MS")
    if raw is None or not raw.strip():
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _defer_vit_dp_direct_cache_on_fallback_enabled() -> bool:
    value = os.getenv(
        "MM_SIDECAR_DEFER_VIT_DP_DIRECT_CACHE_ON_FALLBACK",
        "0",
    ).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _mode_is_data(value: Any) -> bool:
    if value is None:
        return False
    raw = getattr(value, "value", value)
    return str(raw).strip().lower() == "data"


def _uses_vit_data_parallel(model_runner: Any) -> bool:
    for candidate in (
        getattr(model_runner, "model", None),
        getattr(model_runner, "model_config", None),
        getattr(model_runner, "vllm_config", None),
    ):
        if bool(getattr(candidate, "use_data_parallel", False)):
            return True
        multimodal_config = getattr(candidate, "multimodal_config", None)
        if _mode_is_data(getattr(multimodal_config, "mm_encoder_tp_mode", None)):
            return True
    get_model = getattr(model_runner, "get_model", None)
    if callable(get_model):
        try:
            resolved_model = get_model()
        except Exception:
            resolved_model = None
        for candidate in (
            resolved_model,
            getattr(resolved_model, "model", None),
            getattr(resolved_model, "visual", None),
            getattr(resolved_model, "multimodal_config", None),
        ):
            if bool(getattr(candidate, "use_data_parallel", False)):
                return True
            multimodal_config = getattr(candidate, "multimodal_config", None)
            if _mode_is_data(getattr(multimodal_config, "mm_encoder_tp_mode", None)):
                return True
            if _mode_is_data(getattr(candidate, "mm_encoder_tp_mode", None)):
                return True
    model_config = getattr(model_runner, "model_config", None)
    multimodal_config = (
        getattr(model_config, "multimodal_config", None)
        if model_config is not None
        else None
    )
    if _mode_is_data(getattr(multimodal_config, "mm_encoder_tp_mode", None)):
        return True
    vllm_config = getattr(model_runner, "vllm_config", None)
    if vllm_config is not None:
        model_config = getattr(vllm_config, "model_config", None)
        multimodal_config = (
            getattr(model_config, "multimodal_config", None)
            if model_config is not None
            else None
        )
        if _mode_is_data(getattr(multimodal_config, "mm_encoder_tp_mode", None)):
            return True
    return False


def _load_balance_assignment(
    sizes: list[int],
    num_workers: int,
) -> tuple[list[int], list[int], list[int]]:
    sample_count = len(sizes)
    if sample_count == 0:
        return [], [0] * num_workers, [0] * num_workers

    worker_assignments = [list[int]() for _ in range(num_workers)]
    worker_loads = [0] * num_workers
    large_to_small = sorted(
        range(sample_count),
        key=lambda idx: sizes[idx],
        reverse=True,
    )
    for idx in large_to_small:
        target = min(range(num_workers), key=lambda worker_id: worker_loads[worker_id])
        worker_assignments[target].append(idx)
        worker_loads[target] += sizes[idx]

    shuffle_indices = list[int]()
    worker_sample_counts = list[int]()
    for worker_id in range(num_workers):
        shuffle_indices.extend(worker_assignments[worker_id])
        worker_sample_counts.append(len(worker_assignments[worker_id]))
    return shuffle_indices, worker_sample_counts, worker_loads


def _extract_feature_grid_thw(
    req_state: Any,
    request_media_index: int,
) -> tuple[int, int, int] | None:
    mm_features = getattr(req_state, "mm_features", None)
    if not isinstance(mm_features, list):
        return None
    if request_media_index < 0 or request_media_index >= len(mm_features):
        return None
    feature_data = getattr(mm_features[request_media_index], "data", None)
    if feature_data is None:
        return None

    values = []
    if isinstance(feature_data, dict):
        values.append(feature_data.get("image_grid_thw"))
    try:
        values.append(feature_data["image_grid_thw"])
    except Exception:
        pass
    get_data = getattr(feature_data, "get_data", None)
    if callable(get_data):
        try:
            mapping = get_data()
        except Exception:
            mapping = None
        if isinstance(mapping, dict):
            values.append(mapping.get("image_grid_thw"))

    for candidate in values:
        if candidate is None:
            continue
        raw = getattr(candidate, "data", candidate)
        try:
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
        except Exception:
            pass
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            try:
                return (int(raw[0]), int(raw[1]), int(raw[2]))
            except (TypeError, ValueError):
                continue
        if (
            isinstance(raw, (list, tuple))
            and len(raw) == 1
            and isinstance(raw[0], (list, tuple))
            and len(raw[0]) == 3
        ):
            try:
                return (int(raw[0][0]), int(raw[0][1]), int(raw[0][2]))
            except (TypeError, ValueError):
                continue
    return None


def _resolve_grid_thw_for_index(
    req_state: Any,
    request_media_index: int,
    planned_item: dict[str, Any] | None,
) -> tuple[int, int, int] | None:
    if isinstance(planned_item, dict):
        raw_grid_thw = planned_item.get("image_grid_thw")
        if isinstance(raw_grid_thw, (list, tuple)) and len(raw_grid_thw) == 3:
            try:
                return (
                    int(raw_grid_thw[0]),
                    int(raw_grid_thw[1]),
                    int(raw_grid_thw[2]),
                )
            except (TypeError, ValueError):
                pass
    return _extract_feature_grid_thw(req_state, request_media_index)


def _binding_planned_items_by_index(
    binding: WorkerSidecarBinding,
) -> dict[int, dict[str, Any]]:
    planned_item_by_index: dict[int, dict[str, Any]] = {}
    for fallback_position, planned_item in enumerate(
        binding.decoded_plan.planned_items or ()
    ):
        if not isinstance(planned_item, dict):
            continue
        raw_index = planned_item.get("request_media_index", fallback_position)
        try:
            planned_index = int(raw_index)
        except (TypeError, ValueError):
            planned_index = fallback_position
        planned_item_by_index[planned_index] = planned_item
    return planned_item_by_index


def _collect_vit_dp_shard_fetch_items(
    model_runner: Any,
    scheduler_output: Any,
    *,
    role: TpWorkerRole,
) -> tuple[VitDpShardFetchItem, ...] | None:
    if (
        role.world_size <= 1
        or not _vit_dp_shard_fetch_enabled()
        or _vit_dp_direct_encode_enabled()
        or not _uses_vit_data_parallel(model_runner)
    ):
        return None
    if _get_model_visual(model_runner) is None:
        return None

    requests = getattr(model_runner, "requests", None)
    scheduled_encoder_inputs = getattr(
        scheduler_output,
        "scheduled_encoder_inputs",
        None,
    )
    if not isinstance(requests, dict) or not scheduled_encoder_inputs:
        return None

    items: list[VitDpShardFetchItem] = []
    for req_id, image_input_ids in scheduled_encoder_inputs.items():
        req_state = requests.get(req_id)
        if req_state is None:
            return None
        binding = get_request_mm_sidecar_binding(req_state)
        if binding is None:
            binding = bind_request_mm_sidecar(req_state)
        if binding is None or not binding.enabled:
            return None

        descriptor_by_index = {
            int(descriptor.request_media_index): descriptor
            for descriptor in binding.decoded_plan.fallback_descriptors
        }
        handle_by_index = {
            int(handle.request_media_index): handle
            for handle in binding.decoded_plan.handles
        }
        planned_item_by_index = _binding_planned_items_by_index(binding)
        mm_features = getattr(req_state, "mm_features", None)
        if not isinstance(mm_features, list):
            return None

        for raw_input_id in image_input_ids:
            try:
                request_media_index = int(raw_input_id)
                feature = mm_features[request_media_index]
            except Exception:
                return None
            if getattr(feature, "modality", None) != "image":
                return None

            descriptor = descriptor_by_index.get(request_media_index)
            handle = handle_by_index.get(request_media_index)
            if descriptor is None or handle is None:
                return None

            planned_item = planned_item_by_index.get(request_media_index)
            grid_thw = _resolve_grid_thw_for_index(
                req_state,
                request_media_index,
                planned_item,
            )
            if grid_thw is None:
                return None

            items.append(
                VitDpShardFetchItem(
                    req_id=str(req_id),
                    req_state=req_state,
                    binding=binding,
                    feature=feature,
                    request_media_index=request_media_index,
                    descriptor=descriptor,
                    handle=handle,
                    planned_item=planned_item,
                    grid_thw=grid_thw,
                )
            )

    return tuple(items) if items else None


def _materialize_vit_dp_shard_fetch_placeholders(
    items: tuple[VitDpShardFetchItem, ...],
) -> None:
    for item in items:
        if item.planned_item is None:
            continue
        item.feature.data = planned_item_to_vit_dp_placeholder_qwen_mm_kwargs_item(
            item.planned_item,
            processor_signature=item.binding.processor_signature,
        )
        setattr(item.req_state, "mm_sidecar_vit_dp_shard_fetch_prepared", True)


def _context_slice_for_grid_thw(
    context: VitDpShardFetchContext,
    grid_thw_list: list[list[int]],
) -> tuple[VitDpShardFetchItem, ...] | None:
    count = len(grid_thw_list)
    if count <= 0:
        return None
    start = context.cursor
    end = start + count
    if end > len(context.items):
        return None
    items = context.items[start:end]
    for item, grid_thw in zip(items, grid_thw_list):
        if tuple(int(value) for value in grid_thw) != item.grid_thw:
            return None
    context.cursor = end
    return items


def _get_model_visual(model_runner: Any) -> Any | None:
    model = getattr(model_runner, "model", None)
    if model is None:
        get_model = getattr(model_runner, "get_model", None)
        if callable(get_model):
            try:
                model = get_model()
            except Exception:
                model = None
    return getattr(model, "visual", None)


def _attach_vit_dp_shard_fetch_context(
    model_runner: Any,
    scheduler_output: Any,
) -> list[tuple[Any, str, Any]]:
    role = _resolve_tp_worker_role()
    items = _collect_vit_dp_shard_fetch_items(
        model_runner,
        scheduler_output,
        role=role,
    )
    if not items:
        return []

    visual = _get_model_visual(model_runner)
    if visual is None:
        return []

    previous = getattr(visual, _VIT_DP_SHARD_CONTEXT_ATTR, _MISSING_ATTR)
    context = VitDpShardFetchContext(
        model_runner=model_runner,
        role=role,
        items=items,
    )
    setattr(visual, _VIT_DP_SHARD_CONTEXT_ATTR, context)
    _emit_worker_debug(
        "attached vit_dp_shard_fetch_context "
        f"rank={role.local_rank}/{role.world_size} items={len(items)}"
    )
    return [(visual, _VIT_DP_SHARD_CONTEXT_ATTR, previous)]


def _restore_patched_attrs(restored: list[tuple[Any, str, Any]]) -> None:
    for target, attr_name, original_value in reversed(restored):
        try:
            if original_value is _MISSING_ATTR:
                delattr(target, attr_name)
            else:
                setattr(target, attr_name, original_value)
        except Exception:
            pass


def _select_worker_mm_shard(
    model_runner: Any,
    binding: WorkerSidecarBinding,
    *,
    req_state: Any,
    scheduled_encoder_input_ids: list[int] | tuple[int, ...] | None,
    role: TpWorkerRole,
) -> WorkerShardSelection:
    descriptor_by_index = {
        int(descriptor.request_media_index): descriptor
        for descriptor in binding.decoded_plan.fallback_descriptors
    }
    handle_by_index = {
        int(handle.request_media_index): handle
        for handle in binding.decoded_plan.handles
    }
    planned_item_by_index = _binding_planned_items_by_index(binding)

    if scheduled_encoder_input_ids:
        scheduled_image_indexes = tuple(
            int(item_index)
            for item_index in scheduled_encoder_input_ids
            if int(item_index) in descriptor_by_index
        )
    else:
        scheduled_image_indexes = tuple(sorted(descriptor_by_index))

    use_vit_data_parallel = (
        role.world_size > 1
        and _uses_vit_data_parallel(model_runner)
        and _vit_dp_direct_encode_enabled()
    )
    if not use_vit_data_parallel:
        local_image_indexes = scheduled_image_indexes
    else:
        shard_sizes: list[int] = []
        for request_media_index in scheduled_image_indexes:
            planned_item = planned_item_by_index.get(int(request_media_index))
            grid_thw = _resolve_grid_thw_for_index(
                req_state,
                int(request_media_index),
                planned_item,
            )
            if grid_thw is None:
                local_image_indexes = scheduled_image_indexes
                break
            try:
                size = int(grid_thw[0]) * int(grid_thw[1]) * int(grid_thw[2])
            except (TypeError, ValueError):
                local_image_indexes = scheduled_image_indexes
                break
            shard_sizes.append(size)
        else:
            image_to_worker, worker_counts, _worker_loads = _load_balance_assignment(
                shard_sizes,
                role.world_size,
            )
            start = sum(worker_counts[:role.local_rank])
            end = start + worker_counts[role.local_rank]
            local_positions = image_to_worker[start:end]
            local_image_indexes = tuple(
                int(scheduled_image_indexes[position]) for position in local_positions
            )

    local_image_index_set = set(local_image_indexes)
    local_descriptors = tuple(
        descriptor_by_index[item_index]
        for item_index in local_image_indexes
        if item_index in descriptor_by_index
    )
    local_handles = tuple(
        handle_by_index[item_index]
        for item_index in local_image_indexes
        if item_index in handle_by_index
    )
    local_planned_items = tuple(
        planned_item_by_index[item_index]
        for item_index in local_image_indexes
        if item_index in planned_item_by_index
    )
    remote_image_indexes = tuple(
        item_index
        for item_index in scheduled_image_indexes
        if item_index not in local_image_index_set
    )
    return WorkerShardSelection(
        scheduled_image_indexes=scheduled_image_indexes,
        local_image_indexes=local_image_indexes,
        local_descriptors=local_descriptors,
        local_handles=local_handles,
        local_planned_items=local_planned_items,
        remote_image_indexes=remote_image_indexes,
        use_vit_data_parallel=use_vit_data_parallel,
    )


def _materialize_remote_vit_dp_placeholders(
    req_state: Any,
    binding: WorkerSidecarBinding,
    selection: WorkerShardSelection,
) -> None:
    if not selection.use_vit_data_parallel or not selection.remote_image_indexes:
        return
    mm_features = getattr(req_state, "mm_features", None)
    if not isinstance(mm_features, list):
        return
    planned_item_by_index: dict[int, dict[str, Any]] = {}
    for fallback_position, planned_item in enumerate(
        binding.decoded_plan.planned_items or ()
    ):
        if not isinstance(planned_item, dict):
            continue
        raw_index = planned_item.get("request_media_index", fallback_position)
        try:
            planned_index = int(raw_index)
        except (TypeError, ValueError):
            planned_index = fallback_position
        planned_item_by_index[planned_index] = planned_item

    for request_media_index in selection.remote_image_indexes:
        if request_media_index < 0 or request_media_index >= len(mm_features):
            continue
        planned_item = planned_item_by_index.get(int(request_media_index))
        if planned_item is None:
            continue
        feature = mm_features[request_media_index]
        if getattr(feature, "modality", None) != "image":
            continue
        feature_data = getattr(feature, "data", None)
        if not _feature_data_looks_placeholder(feature_data):
            continue
        feature.data = planned_item_to_vit_dp_placeholder_qwen_mm_kwargs_item(
            planned_item,
            processor_signature=binding.processor_signature,
        )


def _emit_shard_debug(
    binding: WorkerSidecarBinding,
    *,
    role: TpWorkerRole,
    selection: WorkerShardSelection,
) -> None:
    _emit_worker_debug(
        f"req={binding.request_id} shard_select "
        f"vit_dp={int(selection.use_vit_data_parallel)} "
        f"rank={role.local_rank}/{role.world_size} "
        f"scheduled={len(selection.scheduled_image_indexes)} "
        f"local={len(selection.local_image_indexes)} "
        f"remote={len(selection.remote_image_indexes)} "
        f"local_idx={list(selection.local_image_indexes)}"
    )


def _artifact_payload_bytes(artifacts: list[Any] | tuple[Any, ...]) -> int:
    total = 0
    for artifact in artifacts:
        descriptor = getattr(artifact, "descriptor", None)
        payload_nbytes = getattr(descriptor, "payload_nbytes", None)
        if isinstance(payload_nbytes, int):
            total += payload_nbytes
    return total


def _merge_fetch_diagnostics(
    artifacts: list[Any] | tuple[Any, ...],
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for artifact in artifacts:
        diagnostics = getattr(artifact, "fetch_diagnostics_ms", None)
        if not isinstance(diagnostics, dict):
            continue
        for key, value in diagnostics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            merged[key] = merged.get(key, 0.0) + numeric
    return merged


def _debug_value_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    if raw is None:
        return "none"
    return str(raw)


def _diagnostic_suffix(value: Any) -> str:
    raw = _debug_value_name(value).lower()
    chars = [
        char if ("a" <= char <= "z" or "0" <= char <= "9") else "_"
        for char in raw
    ]
    suffix = "".join(chars).strip("_")
    return suffix or "unknown"


def _source_plan_numeric_diagnostics(
    source_plan: Any,
    *,
    role: TpWorkerRole,
) -> dict[str, float]:
    entries = tuple(getattr(source_plan, "entries", ()) or ())
    diagnostics: dict[str, float] = {
        "source_plan_entry_count": float(len(entries)),
        "source_plan_use_sidecar_count": 0.0,
        "source_plan_fallback_count": 0.0,
        "source_plan_local_fallback_count": 0.0,
        "source_plan_remote_fallback_count": 0.0,
        "source_plan_near_ready_wait_ms": float(
            getattr(source_plan, "near_ready_wait_ms", 0.0) or 0.0
        ),
        "source_plan_running_ready_wait_ms": float(
            getattr(source_plan, "running_ready_wait_ms", 0.0) or 0.0
        ),
        "source_plan_used_fail_open": (
            1.0 if bool(getattr(source_plan, "used_fail_open", False)) else 0.0
        ),
    }
    diagnostics["source_plan_reported_wait_ms"] = (
        diagnostics["source_plan_near_ready_wait_ms"]
        + diagnostics["source_plan_running_ready_wait_ms"]
    )
    for entry in entries:
        decision = getattr(entry, "decision", None)
        if decision == SourcePlanDecision.USE_SIDECAR:
            diagnostics["source_plan_use_sidecar_count"] += 1.0
        elif decision == SourcePlanDecision.FALLBACK:
            diagnostics["source_plan_fallback_count"] += 1.0
            producer_rank = getattr(entry, "producer_rank", None)
            if producer_rank is not None and int(producer_rank) != role.local_rank:
                diagnostics["source_plan_remote_fallback_count"] += 1.0
            else:
                diagnostics["source_plan_local_fallback_count"] += 1.0

        state_key = "source_plan_state_" + _diagnostic_suffix(
            getattr(entry, "state", None)
        )
        diagnostics[state_key] = diagnostics.get(state_key, 0.0) + 1.0
        reason_key = "source_plan_reason_" + _diagnostic_suffix(
            getattr(entry, "reason", None)
        )
        diagnostics[reason_key] = diagnostics.get(reason_key, 0.0) + 1.0
    return diagnostics


def _source_plan_entries_debug(
    source_plan: Any,
    *,
    only_indexes: set[int] | None = None,
) -> str:
    entries = []
    for entry in getattr(source_plan, "entries", ()) or ():
        try:
            request_media_index = int(getattr(entry, "request_media_index"))
        except (TypeError, ValueError):
            continue
        if only_indexes is not None and request_media_index not in only_indexes:
            continue
        producer_rank = getattr(entry, "producer_rank", None)
        rank_text = "-" if producer_rank is None else str(producer_rank)
        entries.append(
            f"{request_media_index}:"
            f"{_debug_value_name(getattr(entry, 'decision', None))}:"
            f"{_debug_value_name(getattr(entry, 'state', None))}:"
            f"{getattr(entry, 'reason', None) or 'none'}:"
            f"rank={rank_text}"
        )
    return ",".join(entries)


def _descriptor_indexes(
    descriptors: list[Any] | tuple[Any, ...],
) -> tuple[int, ...]:
    indexes: list[int] = []
    for descriptor in descriptors:
        try:
            indexes.append(int(descriptor.request_media_index))
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(indexes)


def _artifact_indexes(artifacts: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    indexes: list[int] = []
    for artifact in artifacts:
        try:
            indexes.append(int(artifact.handle.request_media_index))
        except (AttributeError, TypeError, ValueError):
            continue
    return tuple(indexes)


def _all_tp_ranks_ready_for_direct_encode(
    ready: bool,
    *,
    role: TpWorkerRole,
) -> bool:
    if role.world_size <= 1:
        return ready
    try:
        import torch
        from vllm.distributed import tensor_model_parallel_all_reduce
    except Exception:
        return ready

    device = None
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
    except Exception:
        device = None
    if device is None:
        device = torch.device("cpu")

    value = torch.tensor([1 if ready else 0], device=device, dtype=torch.int32)
    reduced = tensor_model_parallel_all_reduce(value)
    try:
        return int(reduced.item()) == role.world_size
    except Exception:
        return ready


def _flatten_multimodal_embeddings_for_safe_merge(
    multimodal_embeddings: Any,
) -> Any:
    import torch

    flattened: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                raise ValueError("multimodal embedding tensor must not be scalar")
            if value.ndim == 1:
                flattened.append(value.reshape(1, value.shape[0]))
            else:
                flattened.append(value.reshape(-1, value.shape[-1]))
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
            return
        raise TypeError(
            "unsupported multimodal embedding item: "
            f"{value.__class__.__name__}"
        )

    collect(multimodal_embeddings)
    if not flattened:
        return None
    if len(flattened) == 1:
        return flattened[0]
    return torch.cat(flattened, dim=0)


def _embed_input_ids_with_safe_mm_merge(
    *,
    model_runner: Any,
    model: Any,
    original_embed_input_ids: Any,
    embed_args: tuple[Any, ...],
    embed_kwargs: dict[str, Any],
) -> Any:
    import torch

    if embed_args:
        input_ids = embed_args[0]
        positional_mm_embeddings = embed_args[1] if len(embed_args) > 1 else None
    else:
        input_ids = embed_kwargs.get("input_ids")
        positional_mm_embeddings = None
    multimodal_embeddings = embed_kwargs.get(
        "multimodal_embeddings",
        positional_mm_embeddings,
    )
    is_multimodal = embed_kwargs.get("is_multimodal")

    direct_req_ids = getattr(
        model_runner,
        "mm_sidecar_last_direct_encode_req_ids",
        (),
    )
    if (
        not _safe_mm_merge_enabled()
        or not direct_req_ids
        or multimodal_embeddings is None
        or len(multimodal_embeddings) == 0
        or is_multimodal is None
        or input_ids is None
    ):
        return original_embed_input_ids(*embed_args, **embed_kwargs)

    language_model = getattr(model, "language_model", None)
    text_embedder = getattr(language_model, "embed_input_ids", None)
    embed_text_input_ids = getattr(model, "_embed_text_input_ids", None)
    if not callable(text_embedder) or not callable(embed_text_input_ids):
        return original_embed_input_ids(*embed_args, **embed_kwargs)

    # In the sidecar direct path these positions are overwritten by vision
    # embeddings immediately below. Mask them before text embedding so TP vocab
    # shards never see model-specific placeholder ids that can be invalid for a
    # rank-local embedding table.
    masked_input_ids = input_ids.masked_fill(is_multimodal, 0)
    inputs_embeds = text_embedder(masked_input_ids)
    mm_embeds_flat = _flatten_multimodal_embeddings_for_safe_merge(
        multimodal_embeddings,
    )
    if mm_embeds_flat is None:
        return inputs_embeds

    if not isinstance(is_multimodal, torch.Tensor):
        return original_embed_input_ids(*embed_args, **embed_kwargs)
    positions = is_multimodal.nonzero(as_tuple=False).reshape(-1)
    if int(positions.shape[0]) != int(mm_embeds_flat.shape[0]):
        raise ValueError(
            "sidecar safe multimodal merge count mismatch: "
            f"embeddings={int(mm_embeds_flat.shape[0])} "
            f"placeholders={int(positions.shape[0])}"
        )

    if mm_embeds_flat.device != inputs_embeds.device:
        mm_embeds_flat = mm_embeds_flat.to(device=inputs_embeds.device)
    if positions.device != inputs_embeds.device:
        positions = positions.to(device=inputs_embeds.device)
    inputs_embeds.index_copy_(
        0,
        positions,
        mm_embeds_flat.to(dtype=inputs_embeds.dtype),
    )
    _emit_worker_debug(
        "safe_mm_merge applied "
        f"tokens={int(positions.shape[0])} "
        f"hidden={int(mm_embeds_flat.shape[-1])}"
    )
    return inputs_embeds


def get_worker_sidecar_client(required: bool = False) -> Any | None:
    global _CLIENT_CACHE
    with _CLIENT_LOCK:
        if _CLIENT_CACHE is _CLIENT_UNSET:
            client = connect_sidecar_client_from_env(required=required)
            _CLIENT_CACHE = client
            return client
        return _CLIENT_CACHE


def reset_worker_sidecar_client_cache() -> None:
    global _CLIENT_CACHE
    with _CLIENT_LOCK:
        _CLIENT_CACHE = _CLIENT_UNSET


def _get_sidecar_payload_from_mm_features(req_state: Any) -> dict[str, Any] | None:
    mm_features = getattr(req_state, "mm_features", None)
    if not isinstance(mm_features, list):
        return None
    for feature in mm_features:
        payload = _find_sidecar_payload_in_feature_data(getattr(feature, "data", None))
        if payload is not None:
            return payload
    return None


def _find_sidecar_payload_in_feature_data(value: Any) -> dict[str, Any] | None:
    payload = _payload_attr_or_none(value)
    if payload is not None:
        return payload

    data_attr = getattr(value, "data", None)
    if data_attr is not None and data_attr is not value:
        payload = _payload_attr_or_none(data_attr)
        if payload is not None:
            return payload

    for child in _iter_first_level_children(value):
        payload = _payload_attr_or_none(child)
        if payload is not None:
            return payload

        child_data = getattr(child, "data", None)
        if child_data is not None and child_data is not child:
            payload = _payload_attr_or_none(child_data)
            if payload is not None:
                return payload

    return None


def _payload_attr_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        payload = get_request_payload_from_qwen_mm_kwargs_item(value)
    except RecursionError:
        return None
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _iter_first_level_children(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return tuple(value.values())
    if isinstance(value, (list, tuple)):
        return tuple(value)

    values = getattr(value, "values", None)
    if callable(values):
        try:
            return tuple(values())
        except RecursionError:
            return ()
        except Exception:
            return ()
    return ()


def bind_request_mm_sidecar(req_state: Any) -> WorkerSidecarBinding | None:
    sampling_params = getattr(req_state, "sampling_params", None)
    payload = (
        get_sidecar_payload_from_params(sampling_params)
        if sampling_params is not None
        else None
    )
    if payload is None:
        payload = _get_sidecar_payload_from_mm_features(req_state)
    if payload is not None:
        decoded_plan = decode_sidecar_request_plan(payload)
    else:
        decoded_plan = _reconstruct_request_plan_from_manager(req_state)
        if decoded_plan is None:
            return None
        payload = {}

    binding = WorkerSidecarBinding(
        request_id=str(decoded_plan.request_id),
        enabled=bool(decoded_plan.enabled),
        processor_signature=decoded_plan.processor_signature,
        prepared_image_count=int(decoded_plan.prepared_image_count),
        total_placeholder_token_count=int(decoded_plan.total_placeholder_token_count),
        plan_payload=payload,
        decoded_plan=decoded_plan,
    )
    setattr(req_state, "mm_sidecar_binding", binding)
    return binding


def _reconstruct_request_plan_from_manager(
    req_state: Any,
) -> SidecarRequestPlan | None:
    mm_features = getattr(req_state, "mm_features", None)
    if not isinstance(mm_features, list) or not mm_features:
        return None
    cache_keys: list[str] = []
    for feature in mm_features:
        identifier = getattr(feature, "identifier", None)
        modality = getattr(feature, "modality", None)
        if modality != "image" or not isinstance(identifier, str) or not identifier:
            continue
        cache_keys.append(identifier)
    if not cache_keys:
        return None

    client = get_worker_sidecar_client(required=False)
    if client is None:
        return None
    try:
        lookups = client.lookup_by_cache_keys(cache_keys)
    except Exception:
        return None
    if not lookups:
        return None

    descriptors = []
    handles = []
    request_id: str | None = None
    for lookup in lookups:
        descriptor = getattr(lookup, "descriptor", None)
        handle = getattr(lookup, "handle", None)
        if descriptor is None or handle is None:
            return None
        descriptors.append(descriptor)
        handles.append(handle)
        if request_id is None:
            request_id = str(descriptor.request_id)

    if request_id is None:
        return None

    return SidecarRequestPlan(
        version=1,
        request_id=request_id,
        enabled=True,
        reason="reconstructed_from_manager_lookup",
        processor_signature=descriptors[0].processor_signature_value,
        prepared_image_count=len(descriptors),
        total_placeholder_token_count=0,
        planned_items=tuple(),
        fallback_descriptors=tuple(descriptors),
        handles=tuple(handles),
    )


def get_request_mm_sidecar_binding(req_state: Any) -> WorkerSidecarBinding | None:
    binding = getattr(req_state, "mm_sidecar_binding", None)
    return binding if isinstance(binding, WorkerSidecarBinding) else None


def build_worker_source_plan(
    req_state: Any,
    *,
    producer_rank: int | None = None,
    near_ready_wait_ms: float = 2.0,
    poll_interval_ms: float = 1.0,
) -> Any | None:
    binding = get_request_mm_sidecar_binding(req_state)
    if binding is None:
        binding = bind_request_mm_sidecar(req_state)
    if binding is None:
        return None

    descriptors = list(binding.decoded_plan.fallback_descriptors)
    if not descriptors:
        return None

    role = _resolve_tp_worker_role()
    effective_rank = role.local_rank if producer_rank is None else producer_rank
    client = get_worker_sidecar_client(required=False)
    coordinator = SidecarFallbackCoordinator(
        manager=client,
        claimer_id=build_ranked_claimer_id(
            request_id=binding.request_id,
            producer_rank=effective_rank,
        ),
        producer_rank=effective_rank,
        near_ready_wait_ms=near_ready_wait_ms,
        poll_interval_ms=poll_interval_ms,
        fallback_wait_ms=_remote_fallback_wait_ms(),
        observe_plan_wait_ms=_peer_plan_wait_ms(),
        batch_fetch_ready=_batch_fetch_ready_enabled(),
        running_ready_wait_ms=_running_ready_wait_ms(),
    )

    if client is None or not binding.enabled:
        plan = coordinator.preview_source_plan(descriptors=descriptors, handles=None)
    else:
        plan = coordinator.preview_source_plan(
            descriptors=descriptors,
            handles=list(binding.decoded_plan.handles),
        )
    setattr(req_state, "mm_sidecar_source_plan_preview", plan)
    return plan


def _append_runner_error(model_runner: Any, message: str) -> None:
    errors = getattr(model_runner, "mm_sidecar_worker_errors", None)
    if not isinstance(errors, list):
        errors = []
        setattr(model_runner, "mm_sidecar_worker_errors", errors)
    errors.append(message)
    _emit_worker_debug(f"error: {message}")


def _increment_runner_counter(model_runner: Any, attr_name: str) -> None:
    current = getattr(model_runner, attr_name, 0)
    try:
        current_value = int(current)
    except (TypeError, ValueError):
        current_value = 0
    setattr(model_runner, attr_name, current_value + 1)


def prepare_scheduled_mm_inputs_before_encoder(
    model_runner: Any,
    scheduler_output: Any,
) -> int:
    current_scheduler_output_id = id(scheduler_output)
    if getattr(
        model_runner,
        _PREPARED_SCHEDULER_OUTPUT_ID_ATTR,
        None,
    ) == current_scheduler_output_id:
        return 0

    bind_count = 0
    preview_count = 0
    replaced_count = 0
    try:
        bind_count = bind_scheduled_requests(model_runner, scheduler_output)
    except Exception as exc:
        _append_runner_error(
            model_runner,
            "bind_scheduled_requests failed: "
            f"{exc.__class__.__name__}: {exc}",
        )
    try:
        preview_count = build_scheduled_source_plan_previews(
            model_runner,
            scheduler_output,
        )
    except Exception as exc:
        _append_runner_error(
            model_runner,
            "build_scheduled_source_plan_previews failed: "
            f"{exc.__class__.__name__}: {exc}",
        )
    try:
        replaced_count = try_replace_scheduled_mm_inputs_from_sidecar(
            model_runner,
            scheduler_output,
        )
    except Exception as exc:
        _append_runner_error(
            model_runner,
            "try_replace_scheduled_mm_inputs_from_sidecar failed: "
            f"{exc.__class__.__name__}: {exc}",
        )

    setattr(
        model_runner,
        _PREPARED_SCHEDULER_OUTPUT_ID_ATTR,
        current_scheduler_output_id,
    )
    _increment_runner_counter(model_runner, "mm_sidecar_last_prepare_count")
    setattr(model_runner, "mm_sidecar_last_bind_count", bind_count)
    setattr(model_runner, "mm_sidecar_last_source_plan_count", preview_count)
    setattr(model_runner, "mm_sidecar_last_prepare_replaced_count", replaced_count)
    _emit_worker_debug(
        "prepare scheduler_output="
        f"{current_scheduler_output_id} bind={bind_count} "
        f"preview={preview_count} replaced={replaced_count}"
    )
    return replaced_count


def bind_scheduled_requests(model_runner: Any, scheduler_output: Any) -> int:
    requests = getattr(model_runner, "requests", None)
    if not isinstance(requests, dict):
        return 0

    count = 0
    for new_req_data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        req_id = getattr(new_req_data, "req_id", None)
        if req_id is None:
            continue
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        try:
            if bind_request_mm_sidecar(req_state) is not None:
                count += 1
        except Exception as exc:
            _append_runner_error(
                model_runner,
                "bind_request_mm_sidecar failed for "
                f"{req_id}: {exc.__class__.__name__}: {exc}",
            )
    if count:
        setattr(model_runner, "mm_sidecar_last_bound_count", count)
    return count


def build_scheduled_source_plan_previews(
    model_runner: Any,
    scheduler_output: Any,
) -> int:
    requests = getattr(model_runner, "requests", None)
    if not isinstance(requests, dict):
        return 0

    scheduled_encoder_inputs = getattr(
        scheduler_output,
        "scheduled_encoder_inputs",
        None,
    )
    if not scheduled_encoder_inputs:
        return 0

    count = 0
    for req_id in scheduled_encoder_inputs:
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        try:
            if build_worker_source_plan(req_state) is not None:
                count += 1
        except Exception as exc:
            _append_runner_error(
                model_runner,
                "build_worker_source_plan failed for "
                f"{req_id}: {exc.__class__.__name__}: {exc}",
            )
    if count:
        setattr(model_runner, "mm_sidecar_last_source_plan_count", count)
    return count


def _feature_data_missing_for_descriptors(
    req_state: Any,
    descriptors: list[Any] | tuple[Any, ...],
) -> bool:
    mm_features = getattr(req_state, "mm_features", None)
    if not isinstance(mm_features, list):
        return False
    for descriptor in descriptors:
        index = int(descriptor.request_media_index)
        if index < 0 or index >= len(mm_features):
            continue
        feature_data = getattr(mm_features[index], "data", None)
        if _feature_data_looks_placeholder(feature_data):
            return True
    return False


def _feature_data_looks_placeholder(value: Any) -> bool:
    try:
        import torch
    except Exception:
        torch = None

    seen: set[int] = set()

    def _walk(item: Any) -> bool:
        if item is None:
            return True
        item_id = id(item)
        if item_id in seen:
            return False
        seen.add(item_id)

        if is_synthetic_qwen_mm_kwargs_item(item):
            return True
        if getattr(item, "_mm_sidecar_synthetic_placeholder", False):
            return True
        if torch is not None and isinstance(item, torch.Tensor):
            return item.numel() == 0
        if isinstance(item, dict):
            return any(_walk(child) for child in item.values())
        if isinstance(item, (list, tuple)):
            return any(_walk(child) for child in item)

        data_attr = getattr(item, "data", None)
        if data_attr is not None and data_attr is not item:
            if _walk(data_attr):
                return True

        get_data = getattr(item, "get_data", None)
        if callable(get_data):
            try:
                data_mapping = get_data()
            except Exception:
                data_mapping = None
            if isinstance(data_mapping, dict) and any(
                _walk(child) for child in data_mapping.values()
            ):
                return True
        return False

    return _walk(value)


def _run_local_fallback_artifacts(
    descriptors: list[Any] | tuple[Any, ...],
) -> tuple[Any, ...]:
    return tuple(
        run_descriptor_locally(
            descriptor,
            epoch=0,
            worker_id=-1,
        )
        for descriptor in descriptors
    )


def _publish_local_fallback_artifacts(
    client: Any,
    source_plan: Any,
    local_artifacts: tuple[Any, ...] | list[Any],
    *,
    claimer_id: str,
    producer_rank: int,
) -> None:
    handle_by_index = {
        int(entry.request_media_index): entry.handle
        for entry in getattr(source_plan, "entries", ())
        if (
            getattr(entry, "decision", None) is not None
            and getattr(entry, "producer_rank", None) == producer_rank
            and getattr(entry, "handle", None) is not None
        )
    }
    for artifact in local_artifacts:
        handle = handle_by_index.get(int(artifact.handle.request_media_index))
        if handle is None:
            continue
        client.publish_fallback_local_result(
            handle,
            claimer_id,
            artifact.descriptor,
            artifact.payload,
            artifact.timings_ms,
        )


def _can_degrade_remote_fallback_to_local(
    *,
    role: TpWorkerRole,
    exc: Exception,
) -> bool:
    if role.world_size <= 1 or role.is_coordinator:
        return False
    message = str(exc)
    return (
        "remote fallback artifact unavailable" in message
        or "remote fallback artifact missing" in message
    )


def _build_remote_fallback_degraded_fetch_batch(
    *,
    client: Any,
    source_plan: Any,
    descriptors: list[Any] | tuple[Any, ...],
) -> SidecarFetchBatch:
    descriptor_by_index = {
        int(descriptor.request_media_index): descriptor
        for descriptor in descriptors
    }
    sidecar_artifacts: list[Any] = []
    fallback_descriptors: list[Any] = []

    for entry in getattr(source_plan, "entries", ()):
        descriptor = descriptor_by_index.get(int(entry.request_media_index))
        if descriptor is None:
            continue
        if getattr(entry, "decision", None) is SourcePlanDecision.USE_SIDECAR:
            handle = getattr(entry, "handle", None)
            if handle is None:
                raise RuntimeError(
                    "sidecar ready entry missing handle for "
                    f"media index {entry.request_media_index}"
                )
            artifact = client.fetch_ready(handle)
            if artifact is None:
                raise RuntimeError(
                    "sidecar artifact missing during degraded fetch for "
                    f"media index {entry.request_media_index}"
                )
            sidecar_artifacts.append(artifact)
            continue
        fallback_descriptors.append(descriptor)

    return SidecarFetchBatch(
        source_plan=source_plan,
        sidecar_artifacts=tuple(sidecar_artifacts),
        fallback_descriptors=tuple(
            sorted(
                fallback_descriptors,
                key=lambda item: int(item.request_media_index),
            )
        ),
    )


def _scheduled_image_features_for_request(
    req_state: Any,
    image_input_ids: list[int] | tuple[int, ...],
) -> tuple[list[Any], list[int]]:
    mm_features = getattr(req_state, "mm_features", None)
    if not isinstance(mm_features, list):
        return [], []

    image_features: list[Any] = []
    image_feature_ids: list[int] = []
    for mm_input_id in image_input_ids:
        try:
            feature = mm_features[int(mm_input_id)]
        except Exception:
            continue
        if getattr(feature, "modality", None) != "image":
            return [], []
        image_features.append(feature)
        image_feature_ids.append(int(mm_input_id))
    return image_features, image_feature_ids


def _resolve_vit_dp_local_indices(
    binding: WorkerSidecarBinding,
    req_state: Any,
    image_features: list[Any] | tuple[Any, ...],
    image_feature_ids: list[int] | tuple[int, ...],
    *,
    role: TpWorkerRole,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    planned_item_by_index = _binding_planned_items_by_index(binding)
    patch_sizes: list[int] = []
    for feature, feature_id in zip(image_features, image_feature_ids):
        feature_data = getattr(feature, "data", None)
        grid_thw = _extract_feature_grid_thw(
            type("ReqStateProxy", (), {"mm_features": [feature]})(),
            0,
        )
        if grid_thw is None:
            planned_item = planned_item_by_index.get(int(feature_id))
            grid_thw = _resolve_grid_thw_for_index(
                req_state,
                int(feature_id),
                planned_item,
            )
        if grid_thw is None:
            raise RuntimeError(
                f"missing image_grid_thw for request_media_index={int(feature_id)}"
            )
        patch_sizes.append(int(math.prod(grid_thw)))

    order, counts, loads = _load_balance_assignment(patch_sizes, role.world_size)
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)
    local_indices = tuple(order[prefix[role.local_rank] : prefix[role.local_rank + 1]])
    return local_indices, tuple(order), tuple(counts), tuple(loads)


def _build_vit_dp_execution_plan_for_request(
    model_runner: Any,
    req_id: str,
    req_state: Any,
    image_input_ids: list[int] | tuple[int, ...],
    *,
    role: TpWorkerRole,
) -> LocalShardExecutionPlan | None:
    binding = get_request_mm_sidecar_binding(req_state)
    if binding is None:
        binding = bind_request_mm_sidecar(req_state)
    if binding is None or not binding.enabled:
        return None

    image_features, image_feature_ids = _scheduled_image_features_for_request(
        req_state,
        image_input_ids,
    )
    if not image_features or len(image_features) != len(image_input_ids):
        return None

    descriptor_by_index = {
        int(descriptor.request_media_index): descriptor
        for descriptor in binding.decoded_plan.fallback_descriptors
    }
    handle_by_index = {
        int(handle.request_media_index): handle
        for handle in binding.decoded_plan.handles
    }

    try:
        local_indices, order, counts, loads = _resolve_vit_dp_local_indices(
            binding,
            req_state,
            image_features,
            image_feature_ids,
            role=role,
        )
    except Exception:
        return None

    local_request_media_ids = {
        int(image_feature_ids[local_idx]) for local_idx in local_indices
    }

    descriptors: list[Any] = []
    handles: list[Any] = []
    for feature_id in image_feature_ids:
        if int(feature_id) not in local_request_media_ids:
            continue
        descriptor = descriptor_by_index.get(int(feature_id))
        handle = handle_by_index.get(int(feature_id))
        if descriptor is None or handle is None:
            return None
        descriptors.append(descriptor)
        handles.append(handle)

    source_plan = None
    if descriptors:
        client = get_worker_sidecar_client(required=False)
        coordinator = SidecarFallbackCoordinator(
            manager=client,
            claimer_id=build_ranked_claimer_id(
                request_id=binding.request_id,
                producer_rank=role.local_rank,
            ),
            producer_rank=role.local_rank,
            near_ready_wait_ms=_vit_dp_direct_cache_ready_wait_ms(),
            poll_interval_ms=1.0,
            fallback_wait_ms=_remote_fallback_wait_ms(),
            observe_plan_wait_ms=_peer_plan_wait_ms(),
            batch_fetch_ready=_batch_fetch_ready_enabled(),
            running_ready_wait_ms=_running_ready_wait_ms(),
        )
        if client is None or not binding.enabled:
            source_plan = coordinator.preview_source_plan(
                descriptors=descriptors,
                handles=None,
            )
        else:
            source_plan = coordinator.build_source_plan(
                descriptors=descriptors,
                handles=handles,
            )

    plan = LocalShardExecutionPlan(
        req_id=req_id,
        req_state=req_state,
        binding=binding,
        descriptors=tuple(descriptors),
        handles=tuple(handles),
        source_plan=source_plan,
        image_features=tuple(image_features),
        image_input_ids=tuple(image_feature_ids),
        local_indices=local_indices,
        order=order,
        counts=counts,
        loads=loads,
    )
    setattr(req_state, "mm_sidecar_vit_dp_execution_plan", plan)
    return plan


def _sidecar_or_fallback_items_for_plan(
    plan: LocalShardExecutionPlan,
    *,
    role: TpWorkerRole,
) -> tuple[list[Any], dict[str, float]]:
    total_start = time.perf_counter()
    local_request_media_ids = [
        int(plan.image_input_ids[local_idx]) for local_idx in plan.local_indices
    ]
    descriptor_by_index = {
        int(descriptor.request_media_index): descriptor for descriptor in plan.descriptors
    }
    handle_by_index = {
        int(handle.request_media_index): handle for handle in plan.handles
    }
    local_descriptors = [
        descriptor_by_index[item_id]
        for item_id in local_request_media_ids
        if item_id in descriptor_by_index
    ]
    local_handles = [
        handle_by_index[item_id]
        for item_id in local_request_media_ids
        if item_id in handle_by_index
    ]
    if not local_descriptors:
        return [], {}

    client = get_worker_sidecar_client(required=False)
    diagnostics: dict[str, float] = {}
    artifacts: list[Any] = []
    if client is None or not plan.binding.enabled:
        diagnostics.update(
            {
                "source_plan_entry_count": float(len(local_descriptors)),
                "source_plan_fallback_count": float(len(local_descriptors)),
                "source_plan_local_fallback_count": float(len(local_descriptors)),
                "source_plan_manager_unavailable_count": float(
                    len(local_descriptors)
                ),
            }
        )
        fallback_start = time.perf_counter()
        artifacts.extend(_run_local_fallback_artifacts(local_descriptors))
        diagnostics["local_fallback_ms"] = (
            time.perf_counter() - fallback_start
        ) * 1000.0
        diagnostics.update(_merge_fetch_diagnostics(artifacts))
        diagnostics["payload_bytes"] = float(_artifact_payload_bytes(artifacts))
        diagnostics["artifact_count"] = float(len(artifacts))
        convert_start = time.perf_counter()
        items = [
            sidecar_artifact_to_qwen_mm_kwargs_item(artifact) for artifact in artifacts
        ]
        diagnostics["artifact_to_mm_kwargs_ms"] = (
            time.perf_counter() - convert_start
        ) * 1000.0
        diagnostics["worker_fetch_total_ms"] = (
            time.perf_counter() - total_start
        ) * 1000.0
        return (
            items,
            diagnostics,
        )

    coordinator = SidecarFallbackCoordinator(
        manager=client,
        claimer_id=build_ranked_claimer_id(
            request_id=plan.binding.request_id,
            producer_rank=role.local_rank,
        ),
        producer_rank=role.local_rank,
        near_ready_wait_ms=_vit_dp_direct_cache_ready_wait_ms(),
        poll_interval_ms=1.0,
        fallback_wait_ms=_remote_fallback_wait_ms(),
        observe_plan_wait_ms=_peer_plan_wait_ms(),
        batch_fetch_ready=_batch_fetch_ready_enabled(),
        running_ready_wait_ms=_running_ready_wait_ms(),
    )
    fetch_start = time.perf_counter()
    fetch_batch = coordinator.fetch_according_to_plan(
        descriptors=local_descriptors,
        handles=local_handles,
        source_plan=plan.source_plan,
    )
    diagnostics["fetch_ms"] = (time.perf_counter() - fetch_start) * 1000.0
    diagnostics.update(
        _source_plan_numeric_diagnostics(fetch_batch.source_plan, role=role)
    )
    artifacts.extend(fetch_batch.sidecar_artifacts)
    sidecar_indexes = _artifact_indexes(fetch_batch.sidecar_artifacts)
    fallback_indexes = _descriptor_indexes(fetch_batch.fallback_descriptors)
    diagnostics["fetch_sidecar_count"] = float(len(sidecar_indexes))
    diagnostics["fetch_local_fallback_count"] = float(len(fallback_indexes))
    if _worker_fetch_profile_enabled():
        _emit_worker_debug(
            "direct_cache_source_plan "
            f"req={plan.binding.request_id} "
            f"rank={role.local_rank}/{role.world_size} "
            f"local_media={list(local_request_media_ids)} "
            f"entries={_source_plan_entries_debug(fetch_batch.source_plan)}"
        )
    if fetch_batch.fallback_descriptors:
        if (
            _defer_vit_dp_direct_cache_on_fallback_enabled()
            and _vit_dp_shard_fetch_enabled()
            and not _vit_dp_direct_encode_enabled()
        ):
            raise _DeferVitDpShardFetchToVisionPatch(
                "direct-cache source plan needs local fallback for "
                f"media indexes {list(fallback_indexes)}",
                diagnostics,
            )
        if _worker_fetch_profile_enabled():
            _emit_worker_debug(
                "direct_cache_fallback "
                f"req={plan.binding.request_id} "
                f"rank={role.local_rank}/{role.world_size} "
                f"fallback_media={list(fallback_indexes)} "
                "entries="
                f"{_source_plan_entries_debug(fetch_batch.source_plan, only_indexes=set(fallback_indexes))}"
            )
        fallback_start = time.perf_counter()
        local_fallback_artifacts = _run_local_fallback_artifacts(
            fetch_batch.fallback_descriptors,
        )
        diagnostics["local_fallback_ms"] = (
            time.perf_counter() - fallback_start
        ) * 1000.0
        if role.world_size > 1:
            _publish_local_fallback_artifacts(
                client,
                fetch_batch.source_plan,
                local_fallback_artifacts,
                claimer_id=build_ranked_claimer_id(
                    request_id=plan.binding.request_id,
                    producer_rank=role.local_rank,
                ),
                producer_rank=role.local_rank,
            )
        artifacts.extend(local_fallback_artifacts)
    diagnostics.update(_merge_fetch_diagnostics(artifacts))
    diagnostics["payload_bytes"] = float(_artifact_payload_bytes(artifacts))
    diagnostics["artifact_count"] = float(len(artifacts))
    artifact_by_index = {
        int(artifact.handle.request_media_index): artifact for artifact in artifacts
    }
    convert_start = time.perf_counter()
    items = [
        sidecar_artifact_to_qwen_mm_kwargs_item(artifact_by_index[item_id])
        for item_id in local_request_media_ids
        if item_id in artifact_by_index
    ]
    diagnostics["artifact_to_mm_kwargs_ms"] = (
        time.perf_counter() - convert_start
    ) * 1000.0
    diagnostics["worker_fetch_total_ms"] = (
        time.perf_counter() - total_start
    ) * 1000.0
    return (
        items,
        diagnostics,
    )


def _artifact_to_pixel_values_tensor(
    artifact: Any,
    *,
    reference_pixel_values: Any,
) -> Any:
    import torch

    payload = getattr(artifact, "payload", None)
    raw_pixel_values = getattr(payload, "pixel_values", None)
    if raw_pixel_values is None:
        raise RuntimeError("sidecar artifact missing pixel_values payload")
    if isinstance(raw_pixel_values, torch.Tensor):
        tensor = raw_pixel_values
    else:
        tensor = torch.as_tensor(raw_pixel_values)
    return tensor.to(
        device=reference_pixel_values.device,
        dtype=reference_pixel_values.dtype,
    ).contiguous()


def _fetch_vit_dp_shard_pixel_values(
    items: tuple[VitDpShardFetchItem, ...],
    image_idxs_local: list[int],
    *,
    role: TpWorkerRole,
    reference_pixel_values: Any,
) -> tuple[list[Any], dict[str, float]]:
    if not image_idxs_local:
        return [], {}

    total_start = time.perf_counter()
    local_items = tuple(items[index] for index in image_idxs_local)
    descriptors = [item.descriptor for item in local_items]
    handles = [item.handle for item in local_items]
    local_media_indexes = tuple(int(item.request_media_index) for item in local_items)
    client = get_worker_sidecar_client(required=False)
    diagnostics: dict[str, float] = {}
    artifacts: list[Any] = []

    if client is None:
        diagnostics.update(
            {
                "source_plan_entry_count": float(len(local_items)),
                "source_plan_fallback_count": float(len(local_items)),
                "source_plan_local_fallback_count": float(len(local_items)),
                "source_plan_manager_unavailable_count": float(len(local_items)),
            }
        )
        if _worker_fetch_profile_enabled():
            _emit_worker_debug(
                "vit_dp_shard_fetch_plan "
                f"rank={role.local_rank}/{role.world_size} "
                "manager=0 "
                f"local_media={list(local_media_indexes)} "
                "reason=manager_unavailable_fail_open"
            )
        fallback_start = time.perf_counter()
        artifacts.extend(_run_local_fallback_artifacts(descriptors))
        diagnostics["local_fallback_ms"] = (
            time.perf_counter() - fallback_start
        ) * 1000.0
    else:
        coordinator = SidecarFallbackCoordinator(
            manager=client,
            claimer_id=build_ranked_claimer_id(
                request_id=local_items[0].binding.request_id,
                producer_rank=role.local_rank,
            ),
            producer_rank=role.local_rank,
            near_ready_wait_ms=_native_vit_dp_ready_wait_ms(),
            poll_interval_ms=1.0,
            fallback_wait_ms=_remote_fallback_wait_ms(),
            observe_plan_wait_ms=_peer_plan_wait_ms(),
            batch_fetch_ready=_batch_fetch_ready_enabled(),
            running_ready_wait_ms=_running_ready_wait_ms(),
        )
        source_plan_start = time.perf_counter()
        source_plan = coordinator.build_source_plan(
            descriptors=descriptors,
            handles=handles,
            claim=False,
            wait_for_ready=True,
        )
        diagnostics["source_plan_ms"] = (
            time.perf_counter() - source_plan_start
        ) * 1000.0
        diagnostics.update(_source_plan_numeric_diagnostics(source_plan, role=role))
        if _worker_fetch_profile_enabled():
            _emit_worker_debug(
                "vit_dp_shard_fetch_plan "
                f"rank={role.local_rank}/{role.world_size} "
                f"manager=1 local_media={list(local_media_indexes)} "
                f"entries={_source_plan_entries_debug(source_plan)}"
            )
        fetch_start = time.perf_counter()
        fetch_batch = coordinator.fetch_according_to_plan(
            descriptors=descriptors,
            handles=handles,
            source_plan=source_plan,
        )
        diagnostics["fetch_ms"] = (time.perf_counter() - fetch_start) * 1000.0
        artifacts.extend(fetch_batch.sidecar_artifacts)
        sidecar_indexes = _artifact_indexes(fetch_batch.sidecar_artifacts)
        fallback_indexes = _descriptor_indexes(fetch_batch.fallback_descriptors)
        diagnostics["fetch_sidecar_count"] = float(len(sidecar_indexes))
        diagnostics["fetch_local_fallback_count"] = float(len(fallback_indexes))
        if fetch_batch.fallback_descriptors:
            if _worker_fetch_profile_enabled():
                _emit_worker_debug(
                    "vit_dp_shard_fetch_fallback "
                    f"rank={role.local_rank}/{role.world_size} "
                    f"fallback_media={list(fallback_indexes)} "
                    "entries="
                    f"{_source_plan_entries_debug(source_plan, only_indexes=set(fallback_indexes))}"
                )
            fallback_start = time.perf_counter()
            artifacts.extend(_run_local_fallback_artifacts(fetch_batch.fallback_descriptors))
            diagnostics["local_fallback_ms"] = (
                time.perf_counter() - fallback_start
            ) * 1000.0

    artifact_by_index = {
        int(artifact.handle.request_media_index): artifact for artifact in artifacts
    }
    pixel_tensors: list[Any] = []
    tensor_start = time.perf_counter()
    for item in local_items:
        artifact = artifact_by_index.get(item.request_media_index)
        if artifact is None:
            raise RuntimeError(
                "vit dp shard fetch missing artifact for "
                f"media index {item.request_media_index}"
            )
        pixel_tensors.append(
            _artifact_to_pixel_values_tensor(
                artifact,
                reference_pixel_values=reference_pixel_values,
            )
        )
    diagnostics["artifact_to_tensor_ms"] = (
        time.perf_counter() - tensor_start
    ) * 1000.0

    diagnostics.update(_merge_fetch_diagnostics(artifacts))
    diagnostics["payload_bytes"] = float(_artifact_payload_bytes(artifacts))
    diagnostics["artifact_count"] = float(len(artifacts))
    diagnostics["local_image_count"] = float(len(local_items))
    diagnostics["worker_fetch_total_ms"] = (
        time.perf_counter() - total_start
    ) * 1000.0
    for item in local_items:
        setattr(item.req_state, "mm_sidecar_last_fetch_profile_ms", diagnostics)
    return pixel_tensors, diagnostics


def _run_dp_sharded_mrope_vision_model_with_sidecar(
    vision_model: Any,
    pixel_values: Any,
    grid_thw_list: list[list[int]],
    *,
    rope_type: str,
    context_items: tuple[VitDpShardFetchItem, ...],
    context: VitDpShardFetchContext,
    vision_module: Any,
) -> tuple[Any, ...]:
    import itertools
    import torch

    from vllm.distributed import tensor_model_parallel_all_gather

    total_start = time.perf_counter()
    role = context.role
    tp_size = role.world_size
    tp_rank_local = role.local_rank

    assignment_start = time.perf_counter()
    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len = (
        vision_module.get_load_balance_assignment(patches_per_image, tp_size)
    )
    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]
    image_idxs_local = image_to_tp_rank[
        cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]
    ]
    assignment_ms = (time.perf_counter() - assignment_start) * 1000.0

    fetch_call_start = time.perf_counter()
    pixel_tensors, diagnostics = _fetch_vit_dp_shard_pixel_values(
        context_items,
        image_idxs_local,
        role=role,
        reference_pixel_values=pixel_values,
    )
    diagnostics["vit_dp_fetch_call_ms"] = (
        time.perf_counter() - fetch_call_start
    ) * 1000.0
    diagnostics["vit_dp_assignment_ms"] = assignment_ms

    concat_start = time.perf_counter()
    if pixel_tensors:
        pixel_values_local = torch.cat(pixel_tensors, dim=0)
    else:
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
    diagnostics["vit_dp_pixel_concat_ms"] = (
        time.perf_counter() - concat_start
    ) * 1000.0

    if rope_type == "rope_2d":
        embed_dim_reduction_factor = (
            vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
        )
    else:
        embed_dim_reduction_factor = (
            vision_model.spatial_merge_size * vision_model.spatial_merge_size
        )

    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    vision_start = time.perf_counter()
    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(
                pixel_values_local,
                torch.tensor(local_grid_thw_list),
            )
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, local_grid_thw_list)
        else:
            image_embeds_local = torch.empty(
                (0, vision_model.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    diagnostics["vit_dp_vision_ms"] = (time.perf_counter() - vision_start) * 1000.0

    gather_start = time.perf_counter()
    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty(
                (
                    padding_size,
                    image_embeds_local.shape[1],
                    image_embeds_local.shape[2],
                ),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    gathered_embeds = tensor_model_parallel_all_gather(
        image_embeds_local_padded,
        dim=0,
    )
    diagnostics["vit_dp_all_gather_ms"] = (
        time.perf_counter() - gather_start
    ) * 1000.0

    reconstruct_start = time.perf_counter()
    rank_embeddings = list[Any]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (
            grouped_pixel_values_len[rank] // embed_dim_reduction_factor
        )
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [
        patch_size // embed_dim_reduction_factor
        for patch_size in patches_per_image
    ]
    original_order_embeddings: list[Any | None] = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            rank_images = image_to_tp_rank[current_idx : current_idx + count]
            rank_embed = rank_embeddings[rank]
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[
                    embed_start : embed_start + img_patches
                ]
                embed_start += img_patches
            current_idx += count

    out_embeddings = tuple(
        embed for embed in original_order_embeddings if embed is not None
    )
    if len(out_embeddings) != len(original_order_embeddings):
        raise RuntimeError("vit dp shard fetch found unassigned embeddings")

    diagnostics["vit_dp_reconstruct_ms"] = (
        time.perf_counter() - reconstruct_start
    ) * 1000.0
    diagnostics["vit_dp_total_ms"] = (time.perf_counter() - total_start) * 1000.0
    setattr(context.model_runner, "mm_sidecar_last_fetch_profile_ms", diagnostics)
    setattr(
        context.model_runner,
        "mm_sidecar_last_vit_dp_shard_fetch",
        {
            "rank": role.local_rank,
            "world_size": role.world_size,
            "local_indices": tuple(int(index) for index in image_idxs_local),
            "local_request_media_indexes": tuple(
                int(context_items[index].request_media_index)
                for index in image_idxs_local
            ),
            "total_images": len(grid_thw_list),
            "payload_bytes": diagnostics.get("payload_bytes", 0.0),
            "timings_ms": dict(diagnostics),
        },
    )
    debug_extra = ""
    if _worker_fetch_profile_enabled():
        debug_extra = " " + " ".join(
            f"{key}={value:.3f}"
            for key, value in sorted(diagnostics.items())
        )
    _emit_worker_debug(
        "vit_dp_shard_fetch "
        f"rank={role.local_rank}/{role.world_size} "
        f"local_indices={list(image_idxs_local)} "
        f"total_images={len(grid_thw_list)} "
        f"payload_bytes={diagnostics.get('payload_bytes', 0.0):.0f}"
        f"{debug_extra}"
    )
    return out_embeddings


def install_vit_dp_shard_fetch_patch() -> bool:
    try:
        vision_module = importlib.import_module("vllm.model_executor.models.vision")
    except Exception:
        return False
    original = getattr(vision_module, "run_dp_sharded_mrope_vision_model", None)
    if original is None:
        return False
    if getattr(original, _VISION_PATCH_MARKER_ATTR, False):
        return False

    @wraps(original)
    def wrapped_run_dp_sharded_mrope_vision_model(
        vision_model: Any,
        pixel_values: Any,
        grid_thw_list: list[list[int]],
        *,
        rope_type: str,
    ) -> tuple[Any, ...]:
        if not _vit_dp_shard_fetch_enabled() or _vit_dp_direct_encode_enabled():
            return original(
                vision_model,
                pixel_values,
                grid_thw_list,
                rope_type=rope_type,
            )
        context = getattr(vision_model, _VIT_DP_SHARD_CONTEXT_ATTR, None)
        if not isinstance(context, VitDpShardFetchContext):
            return original(
                vision_model,
                pixel_values,
                grid_thw_list,
                rope_type=rope_type,
            )
        context_items = _context_slice_for_grid_thw(context, grid_thw_list)
        if context_items is None:
            return original(
                vision_model,
                pixel_values,
                grid_thw_list,
                rope_type=rope_type,
            )
        return _run_dp_sharded_mrope_vision_model_with_sidecar(
            vision_model,
            pixel_values,
            grid_thw_list,
            rope_type=rope_type,
            context_items=context_items,
            context=context,
            vision_module=vision_module,
        )

    setattr(wrapped_run_dp_sharded_mrope_vision_model, _VISION_PATCH_MARKER_ATTR, True)
    setattr(wrapped_run_dp_sharded_mrope_vision_model, _VISION_ORIGINAL_FN_ATTR, original)
    vision_module.run_dp_sharded_mrope_vision_model = (
        wrapped_run_dp_sharded_mrope_vision_model
    )

    for module_name in (
        "vllm.model_executor.models.qwen2_vl",
        "vllm.model_executor.models.qwen2_5_vl",
        "vllm.model_executor.models.qwen3_vl",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if hasattr(module, "run_dp_sharded_mrope_vision_model"):
            setattr(
                module,
                "run_dp_sharded_mrope_vision_model",
                wrapped_run_dp_sharded_mrope_vision_model,
            )

    _emit_worker_debug("installed ViT-DP shard fetch vision patch")
    return True


def _manual_encode_and_gather_local_items(
    model_runner: Any,
    *,
    image_features: tuple[Any, ...],
    local_indices: tuple[int, ...],
    local_items: list[Any],
    order: tuple[int, ...],
    counts: tuple[int, ...],
) -> list[Any]:
    import torch

    try:
        from vllm.distributed import tensor_model_parallel_all_gather
        from vllm.multimodal.utils import group_and_batch_mm_kwargs
        from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs
    except Exception as exc:
        raise RuntimeError(f"vllm direct encode dependencies unavailable: {exc}") from exc

    model = getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("model_runner.model is required for local direct encode")

    local_mm_kwargs = [("image", item) for item in local_items]
    local_outputs: list[torch.Tensor] = []
    original_use_data_parallel = getattr(model, "use_data_parallel", None)
    if original_use_data_parallel is not None:
        model.use_data_parallel = False
    try:
        for _, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            local_mm_kwargs,
            device=getattr(model_runner, "device", None),
            pin_memory=bool(getattr(model_runner, "pin_memory", False)),
        ):
            batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
            sanity_check_mm_encoder_outputs(
                batch_outputs,
                expected_num_items=num_items,
            )
            local_outputs.extend(batch_outputs)
    finally:
        if original_use_data_parallel is not None:
            model.use_data_parallel = original_use_data_parallel

    tp_size = len(counts)
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)

    output_sizes = [int(feature.mm_position.get_num_embeds()) for feature in image_features]
    grouped_output_lens = []
    for rank in range(tp_size):
        rank_indices = order[prefix[rank] : prefix[rank + 1]]
        grouped_output_lens.append(sum(output_sizes[idx] for idx in rank_indices))

    hidden_size = int(getattr(getattr(model, "visual", None), "out_hidden_size", 0))
    if bool(getattr(model, "is_multimodal_pruning_enabled", False)):
        hidden_size += 5

    if local_outputs:
        local_cat = torch.cat(local_outputs, dim=0)
        hidden_size = int(local_cat.shape[1])
    else:
        local_cat = torch.empty(
            (0, hidden_size),
            device=getattr(model_runner, "device", None),
            dtype=getattr(getattr(model, "visual", None), "dtype", torch.float32),
        )

    max_len_per_rank = max(grouped_output_lens) if grouped_output_lens else 0
    if local_cat.shape[0] < max_len_per_rank:
        pad = torch.empty(
            (max_len_per_rank - local_cat.shape[0], hidden_size),
            device=local_cat.device,
            dtype=local_cat.dtype,
        )
        local_cat = torch.cat([local_cat, pad], dim=0)

    gathered = tensor_model_parallel_all_gather(local_cat.contiguous(), dim=0)
    rank_embeddings: list[torch.Tensor] = []
    for rank in range(tp_size):
        start = rank * max_len_per_rank
        end = start + grouped_output_lens[rank]
        rank_embeddings.append(gathered[start:end])

    original_order_embeddings: list[Any] = [None] * len(image_features)
    current_idx = 0
    for rank in range(tp_size):
        count = counts[rank]
        if count <= 0:
            continue
        rank_indices = order[current_idx : current_idx + count]
        rank_embed = rank_embeddings[rank]
        embed_start = 0
        for image_idx in rank_indices:
            image_len = output_sizes[image_idx]
            original_order_embeddings[image_idx] = rank_embed[
                embed_start : embed_start + image_len
            ]
            embed_start += image_len
        current_idx += count

    if any(embed is None for embed in original_order_embeddings):
        raise RuntimeError("local direct encode failed to reconstruct all image embeddings")
    return original_order_embeddings


def _try_execute_vit_dp_sidecar_direct_encode(
    model_runner: Any,
    scheduler_output: Any,
) -> VitDpDirectEncodeResult | None:
    scheduled_encoder_inputs = getattr(
        scheduler_output,
        "scheduled_encoder_inputs",
        None,
    )
    if not scheduled_encoder_inputs:
        return None
    direct_encode_requested = _vit_dp_direct_encode_enabled()
    shard_fetch_direct_requested = (
        _vit_dp_shard_fetch_enabled()
        and not direct_encode_requested
    )
    if not direct_encode_requested and not shard_fetch_direct_requested:
        return None
    if not _uses_vit_data_parallel(model_runner):
        return None

    role = _resolve_tp_worker_role()
    handled_reqs: list[str] = []
    fallback_scheduled: dict[str, list[int]] = {}
    ready_requests: list[
        tuple[
            str,
            Any,
            LocalShardExecutionPlan,
            list[Any],
            dict[str, float],
        ]
    ] = []

    for req_id, image_input_ids in scheduled_encoder_inputs.items():
        request_start = time.perf_counter()
        image_input_ids_list = list(image_input_ids)
        req_state = getattr(model_runner, "requests", {}).get(req_id)
        plan = None
        local_items: list[Any] = []
        diagnostics: dict[str, float] = {}
        ready = False
        error: Exception | None = None
        if req_state is not None:
            try:
                plan_start = time.perf_counter()
                plan = _build_vit_dp_execution_plan_for_request(
                    model_runner,
                    req_id,
                    req_state,
                    image_input_ids_list,
                    role=role,
                )
                direct_plan_ms = (time.perf_counter() - plan_start) * 1000.0
                diagnostics["direct_plan_ms"] = direct_plan_ms
                if plan is not None:
                    items_start = time.perf_counter()
                    local_items, diagnostics = _sidecar_or_fallback_items_for_plan(
                        plan,
                        role=role,
                    )
                    diagnostics = dict(diagnostics)
                    diagnostics["direct_plan_ms"] = direct_plan_ms
                    diagnostics["direct_sidecar_items_ms"] = (
                        time.perf_counter() - items_start
                    ) * 1000.0
                    if len(local_items) != len(plan.local_indices):
                        raise RuntimeError(
                            "local direct encode item count mismatch: "
                            f"expected={len(plan.local_indices)} "
                            f"actual={len(local_items)}"
                        )
                    ready = True
            except Exception as exc:
                error = exc
                if isinstance(exc, _DeferVitDpShardFetchToVisionPatch):
                    diagnostics = dict(exc.diagnostics)

        barrier_start = time.perf_counter()
        all_ready = _all_tp_ranks_ready_for_direct_encode(ready, role=role)
        diagnostics["direct_ready_barrier_ms"] = (
            time.perf_counter() - barrier_start
        ) * 1000.0
        diagnostics["direct_prepare_total_ms"] = (
            time.perf_counter() - request_start
        ) * 1000.0
        if req_state is not None:
            setattr(req_state, "mm_sidecar_last_fetch_profile_ms", diagnostics)
        if not all_ready or plan is None:
            if shard_fetch_direct_requested:
                if error is not None:
                    if isinstance(error, _DeferVitDpShardFetchToVisionPatch):
                        _emit_worker_debug(
                            "vit dp shard-fetch direct cache write deferred to "
                            "vision patch for "
                            f"{req_id}: {error}"
                        )
                    else:
                        _append_runner_error(
                            model_runner,
                            "vit dp shard-fetch direct cache write deferred to "
                            "vision patch for "
                            f"{req_id}: {error.__class__.__name__}: {error}",
                        )
                _emit_worker_debug(
                    f"req={req_id} shard_fetch_direct_defer "
                    f"rank={role.local_rank}/{role.world_size} "
                    f"local_ready={int(ready)} all_ready={int(all_ready)}"
                )
                return None
            fallback_scheduled[req_id] = image_input_ids_list
            if error is not None:
                _append_runner_error(
                    model_runner,
                    "vit dp direct encode disabled for "
                    f"{req_id}: {error.__class__.__name__}: {error}",
                )
            _emit_worker_debug(
                f"req={req_id} direct_encode_fallback "
                f"rank={role.local_rank}/{role.world_size} "
                f"local_ready={int(ready)} all_ready={int(all_ready)}"
            )
            continue

        ready_requests.append((req_id, req_state, plan, local_items, diagnostics))

    for req_id, req_state, plan, local_items, diagnostics in ready_requests:
        execute_start = time.perf_counter()
        try:
            encode_start = time.perf_counter()
            gathered_outputs = _manual_encode_and_gather_local_items(
                model_runner,
                image_features=plan.image_features,
                local_indices=plan.local_indices,
                local_items=local_items,
                order=plan.order,
                counts=plan.counts,
            )
            diagnostics["direct_manual_encode_gather_ms"] = (
                time.perf_counter() - encode_start
            ) * 1000.0
        except Exception as exc:
            _append_runner_error(
                model_runner,
                "vit dp direct encode failed after ready barrier for "
                f"{req_id}: {exc.__class__.__name__}: {exc}",
            )
            raise
        cache_start = time.perf_counter()
        for feature, encoder_output in zip(plan.image_features, gathered_outputs):
            clone = getattr(encoder_output, "clone", None)
            if callable(clone):
                encoder_output = clone()
            contiguous = getattr(encoder_output, "contiguous", None)
            if callable(contiguous):
                encoder_output = contiguous()
            model_runner.encoder_cache[feature.identifier] = encoder_output
            maybe_save = getattr(model_runner, "maybe_save_ec_to_connector", None)
            if callable(maybe_save):
                maybe_save(model_runner.encoder_cache, feature.identifier)
        diagnostics["direct_cache_write_ms"] = (
            time.perf_counter() - cache_start
        ) * 1000.0
        diagnostics["direct_execute_total_ms"] = (
            time.perf_counter() - execute_start
        ) * 1000.0
        setattr(req_state, "mm_sidecar_last_fetch_profile_ms", diagnostics)
        setattr(model_runner, "mm_sidecar_last_fetch_profile_ms", diagnostics)
        handled_reqs.append(req_id)
        debug_extra = ""
        if _worker_fetch_profile_enabled():
            debug_extra = " " + " ".join(
                f"{key}={value:.3f}"
                for key, value in sorted(diagnostics.items())
            )
        _emit_worker_debug(
            f"req={plan.binding.request_id} "
            f"{'shard_fetch_direct_encode' if shard_fetch_direct_requested else 'direct_encode'} "
            f"rank={role.local_rank}/{role.world_size} "
            f"local_indices={list(plan.local_indices)} "
            f"total_images={len(plan.image_features)}"
            f"{debug_extra}"
        )

    if handled_reqs or fallback_scheduled:
        setattr(model_runner, "mm_sidecar_last_direct_encode_req_ids", tuple(handled_reqs))
        setattr(
            model_runner,
            "mm_sidecar_last_direct_encode_fallback_req_ids",
            tuple(fallback_scheduled),
        )
        return VitDpDirectEncodeResult(
            handled_request_ids=tuple(handled_reqs),
            fallback_scheduled=fallback_scheduled,
        )
    return None


def try_replace_scheduled_mm_inputs_from_sidecar(
    model_runner: Any,
    scheduler_output: Any,
) -> int:
    requests = getattr(model_runner, "requests", None)
    if not isinstance(requests, dict):
        return 0

    scheduled_encoder_inputs = getattr(
        scheduler_output,
        "scheduled_encoder_inputs",
        None,
    )
    if not scheduled_encoder_inputs:
        return 0

    role = _resolve_tp_worker_role()
    setattr(
        model_runner,
        "mm_sidecar_last_tp_role",
        {
            "local_rank": role.local_rank,
            "world_size": role.world_size,
            "coordinator_rank": role.coordinator_rank,
            "is_coordinator": role.is_coordinator,
        },
    )

    replaced = 0
    native_vit_dp_full_replacement = _native_vit_dp_full_replacement_mode(
        model_runner,
        role,
    )
    vit_dp_shard_fetch_items = _collect_vit_dp_shard_fetch_items(
        model_runner,
        scheduler_output,
        role=role,
    )
    vit_dp_shard_fetch_req_ids: set[str] = set()
    if vit_dp_shard_fetch_items:
        _materialize_vit_dp_shard_fetch_placeholders(vit_dp_shard_fetch_items)
        vit_dp_shard_fetch_req_ids = {
            item.req_id for item in vit_dp_shard_fetch_items
        }
        setattr(
            model_runner,
            "mm_sidecar_last_vit_dp_shard_fetch_prepared_count",
            len(vit_dp_shard_fetch_items),
        )
        _emit_worker_debug(
            "prepare vit_dp_shard_fetch "
            f"items={len(vit_dp_shard_fetch_items)} "
            f"reqs={sorted(vit_dp_shard_fetch_req_ids)}"
        )
    for req_id in scheduled_encoder_inputs:
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        binding = get_request_mm_sidecar_binding(req_state)
        if binding is None:
            binding = bind_request_mm_sidecar(req_state)
        if binding is None:
            continue
        if (
            role.world_size > 1
            and _uses_vit_data_parallel(model_runner)
            and _vit_dp_direct_encode_enabled()
        ):
            setattr(req_state, "mm_sidecar_vit_dp_prepared", True)
            continue
        if str(req_id) in vit_dp_shard_fetch_req_ids:
            setattr(req_state, "mm_sidecar_vit_dp_shard_fetch_prepared", True)
            continue
        selection = _select_worker_mm_shard(
            model_runner,
            binding,
            req_state=req_state,
            scheduled_encoder_input_ids=scheduled_encoder_inputs.get(req_id),
            role=role,
        )
        _emit_shard_debug(binding, role=role, selection=selection)
        descriptors = list(selection.local_descriptors)
        handles = list(selection.local_handles)
        if not descriptors:
            _materialize_remote_vit_dp_placeholders(req_state, binding, selection)
            continue
        if not _feature_data_missing_for_descriptors(req_state, descriptors):
            _materialize_remote_vit_dp_placeholders(req_state, binding, selection)
            _emit_worker_debug(
                f"req={binding.request_id} media={len(descriptors)} "
                "skip_replace feature_data_ready"
            )
            continue
        client = get_worker_sidecar_client(required=False)
        if client is None or not binding.enabled:
            local_artifacts = _run_local_fallback_artifacts(descriptors)
            replaced += replace_feature_data_from_sidecar_artifacts(
                req_state,
                local_artifacts,
            )
            _emit_worker_debug(
                f"req={binding.request_id} mode=fail_open_fallback "
                f"replaced={len(local_artifacts)}"
            )
            coordinator = SidecarFallbackCoordinator(
                manager=None,
                claimer_id=build_ranked_claimer_id(
                    request_id=binding.request_id,
                    producer_rank=role.local_rank,
                ),
                producer_rank=role.local_rank,
                near_ready_wait_ms=0.0,
                poll_interval_ms=1.0,
            )
            setattr(
                req_state,
                "mm_sidecar_source_plan",
                coordinator.preview_source_plan(descriptors=descriptors),
            )
            setattr(req_state, "mm_sidecar_fallback_descriptors", tuple(descriptors))
            _materialize_remote_vit_dp_placeholders(req_state, binding, selection)
            continue
        try:
            claimer_id = build_ranked_claimer_id(
                request_id=binding.request_id,
                producer_rank=role.local_rank,
            )
            coordinator = SidecarFallbackCoordinator(
                manager=client,
                claimer_id=claimer_id,
                producer_rank=role.local_rank,
                near_ready_wait_ms=(
                    _native_vit_dp_ready_wait_ms()
                    if native_vit_dp_full_replacement
                    else 2.0
                ),
                poll_interval_ms=1.0,
                fallback_wait_ms=_remote_fallback_wait_ms(),
                observe_plan_wait_ms=_peer_plan_wait_ms(),
                batch_fetch_ready=_batch_fetch_ready_enabled(),
                running_ready_wait_ms=_running_ready_wait_ms(),
            )
            plan_start = time.perf_counter()
            if native_vit_dp_full_replacement:
                source_plan = coordinator.build_source_plan(
                    descriptors=descriptors,
                    handles=handles,
                    claim=False,
                    wait_for_ready=True,
                )
            elif (
                role.world_size > 1
                and not role.is_coordinator
                and not selection.use_vit_data_parallel
            ):
                source_plan = coordinator.observe_source_plan(
                    descriptors=descriptors,
                )
            else:
                source_plan = coordinator.build_source_plan(
                    descriptors=descriptors,
                    handles=handles,
                )
            source_plan_ms = (time.perf_counter() - plan_start) * 1000.0
            fetch_start = time.perf_counter()
            try:
                fetch_batch = coordinator.fetch_according_to_plan(
                    descriptors=descriptors,
                    handles=handles,
                    source_plan=source_plan,
                )
            except Exception as exc:
                if not _can_degrade_remote_fallback_to_local(
                    role=role,
                    exc=exc,
                ):
                    raise
                _emit_worker_debug(
                    f"req={binding.request_id} mode=remote_fallback_timeout_degrade "
                    f"error={exc.__class__.__name__}: {exc}"
                )
                fetch_batch = _build_remote_fallback_degraded_fetch_batch(
                    client=client,
                    source_plan=source_plan,
                    descriptors=descriptors,
                )
            fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
            artifacts = list(fetch_batch.sidecar_artifacts)
            fetch_diagnostics_ms = _merge_fetch_diagnostics(artifacts)
            if fetch_batch.fallback_descriptors:
                if _feature_data_missing_for_descriptors(
                    req_state,
                    fetch_batch.fallback_descriptors,
                ):
                    local_artifacts = _run_local_fallback_artifacts(
                        fetch_batch.fallback_descriptors,
                    )
                    if role.world_size > 1 and not native_vit_dp_full_replacement:
                        _publish_local_fallback_artifacts(
                            client,
                            fetch_batch.source_plan,
                            local_artifacts,
                            claimer_id=claimer_id,
                            producer_rank=role.local_rank,
                        )
                    artifacts.extend(local_artifacts)
                    _emit_worker_debug(
                        f"req={binding.request_id} mode=mixed "
                        f"sidecar={len(fetch_batch.sidecar_artifacts)} "
                        f"fallback={len(fetch_batch.fallback_descriptors)}"
                    )
                else:
                    setattr(req_state, "mm_sidecar_source_plan", fetch_batch.source_plan)
                    setattr(
                        req_state,
                        "mm_sidecar_fallback_descriptors",
                        fetch_batch.fallback_descriptors,
                    )
                    continue
            if not artifacts:
                setattr(req_state, "mm_sidecar_source_plan", fetch_batch.source_plan)
                setattr(
                    req_state,
                    "mm_sidecar_fallback_descriptors",
                    fetch_batch.fallback_descriptors,
                )
                continue
            replace_start = time.perf_counter()
            replaced += replace_feature_data_from_sidecar_artifacts(
                req_state,
                tuple(artifacts),
            )
            replace_ms = (time.perf_counter() - replace_start) * 1000.0
            fetch_profile = {
                "source_plan_ms": source_plan_ms,
                "fetch_ms": fetch_ms,
                "replace_ms": replace_ms,
                "payload_bytes": float(_artifact_payload_bytes(artifacts)),
                **fetch_diagnostics_ms,
            }
            setattr(req_state, "mm_sidecar_last_fetch_profile_ms", fetch_profile)
            setattr(model_runner, "mm_sidecar_last_fetch_profile_ms", fetch_profile)
            if _worker_fetch_profile_enabled():
                _emit_worker_debug(
                    "fetch_profile "
                    f"req={binding.request_id} "
                    f"media={len(descriptors)} "
                    f"sidecar_artifacts={len(fetch_batch.sidecar_artifacts)} "
                    f"fallback={len(fetch_batch.fallback_descriptors)} "
                    f"payload_bytes={_artifact_payload_bytes(artifacts)} "
                    f"source_plan_ms={source_plan_ms:.3f} "
                    f"fetch_ms={fetch_ms:.3f} "
                    f"replace_ms={replace_ms:.3f} "
                    + " ".join(
                        f"{key}={value:.3f}"
                        for key, value in sorted(fetch_diagnostics_ms.items())
                    )
                )
            if fetch_batch.sidecar_artifacts and not fetch_batch.fallback_descriptors:
                _emit_worker_debug(
                    f"req={binding.request_id} mode=sidecar_ready "
                    f"replaced={len(fetch_batch.sidecar_artifacts)}"
                )
            setattr(req_state, "mm_sidecar_source_plan", fetch_batch.source_plan)
            setattr(
                req_state,
                "mm_sidecar_fallback_descriptors",
                fetch_batch.fallback_descriptors,
            )
            _materialize_remote_vit_dp_placeholders(req_state, binding, selection)
        except Exception as exc:
            _append_runner_error(
                model_runner,
                "try_replace_scheduled_mm_inputs_from_sidecar failed for "
                f"{req_id}: {exc.__class__.__name__}: {exc}",
            )
    if replaced:
        setattr(model_runner, "mm_sidecar_last_replaced_feature_count", replaced)
    return replaced



def install_gpu_model_runner_patch(gpu_model_runner_cls: Any) -> bool:
    if getattr(gpu_model_runner_cls, _PATCH_MARKER_ATTR, False):
        return False

    original_update_states = gpu_model_runner_cls._update_states
    original_batch_mm_inputs = gpu_model_runner_cls._batch_mm_inputs_from_scheduler
    original_preprocess = getattr(gpu_model_runner_cls, "_preprocess", None)
    original_gather_mm_embeddings = getattr(
        gpu_model_runner_cls,
        "_gather_mm_embeddings",
        None,
    )
    original_execute_mm_encoder = getattr(
        gpu_model_runner_cls,
        "_execute_mm_encoder",
        None,
    )

    @wraps(original_update_states)
    def wrapped_update_states(self: Any, scheduler_output: Any):
        result = original_update_states(self, scheduler_output)
        bind_scheduled_requests(self, scheduler_output)
        return result

    @wraps(original_batch_mm_inputs)
    def wrapped_batch_mm_inputs_from_scheduler(self: Any, scheduler_output: Any):
        prepare_scheduled_mm_inputs_before_encoder(self, scheduler_output)
        return original_batch_mm_inputs(self, scheduler_output)

    if original_execute_mm_encoder is not None:
        @wraps(original_execute_mm_encoder)
        def wrapped_execute_mm_encoder(self: Any, scheduler_output: Any):
            prepare_scheduled_mm_inputs_before_encoder(self, scheduler_output)
            direct_result = _try_execute_vit_dp_sidecar_direct_encode(
                self,
                scheduler_output,
            )
            if direct_result is not None:
                _emit_worker_debug(
                    "execute_mm_encoder direct_result "
                    f"handled={list(direct_result.handled_request_ids)} "
                    f"fallback={list(direct_result.fallback_scheduled)}"
                )
                if not direct_result.fallback_scheduled:
                    _emit_worker_debug("execute_mm_encoder return_direct_empty")
                    return []
                original_scheduled = scheduler_output.scheduled_encoder_inputs
                scheduler_output.scheduled_encoder_inputs = (
                    direct_result.fallback_scheduled
                )
                try:
                    return original_execute_mm_encoder(self, scheduler_output)
                finally:
                    scheduler_output.scheduled_encoder_inputs = original_scheduled
            restored = _attach_vit_dp_shard_fetch_context(self, scheduler_output)
            try:
                return original_execute_mm_encoder(self, scheduler_output)
            finally:
                _restore_patched_attrs(restored)

    if original_gather_mm_embeddings is not None:
        @wraps(original_gather_mm_embeddings)
        def wrapped_gather_mm_embeddings(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
            if _worker_debug_enabled():
                cache = getattr(self, "encoder_cache", {})
                request_ids = tuple(getattr(getattr(self, "input_batch", None), "req_ids", ()))
                feature_keys: list[str] = []
                for req_id in request_ids:
                    req_state = getattr(self, "requests", {}).get(req_id)
                    for feature in getattr(req_state, "mm_features", ()) or ():
                        identifier = getattr(feature, "identifier", None)
                        if isinstance(identifier, str):
                            feature_keys.append(identifier)
                present = sum(1 for key in feature_keys if key in cache)
                _emit_worker_debug(
                    "gather_mm_embeddings enter "
                    f"reqs={list(request_ids)} "
                    f"features={len(feature_keys)} cache_present={present}"
                )
            try:
                result = original_gather_mm_embeddings(
                    self,
                    scheduler_output,
                    *args,
                    **kwargs,
                )
            except Exception as exc:
                _emit_worker_debug(
                    "gather_mm_embeddings error "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise
            if _worker_debug_enabled():
                try:
                    mm_embeds, is_mm_embed = result
                    embed_count = len(mm_embeds)
                    mask_shape = getattr(is_mm_embed, "shape", None)
                    embed_meta = [
                        {
                            "shape": str(getattr(item, "shape", None)),
                            "stride": str(getattr(item, "stride", lambda: None)()),
                            "contiguous": bool(
                                getattr(item, "is_contiguous", lambda: False)()
                            ),
                        }
                        for item in mm_embeds[:3]
                    ]
                except Exception:
                    embed_count = -1
                    mask_shape = None
                    embed_meta = []
                _emit_worker_debug(
                    "gather_mm_embeddings exit "
                    f"embeds={embed_count} mask_shape={mask_shape} "
                    f"embed_meta={embed_meta}"
                )
            return result

    if original_preprocess is not None:
        @wraps(original_preprocess)
        def wrapped_preprocess(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any):
            _emit_worker_debug("preprocess enter")
            restored: list[tuple[Any, str, Any]] = []
            model = getattr(self, "model", None)
            embed_input_ids = getattr(model, "embed_input_ids", None)
            should_patch_embed = _worker_debug_enabled() or _safe_mm_merge_enabled()
            if should_patch_embed:
                if callable(embed_input_ids):
                    @wraps(embed_input_ids)
                    def patched_embed_input_ids(*embed_args: Any, **embed_kwargs: Any):
                        mm_embeds = embed_kwargs.get("multimodal_embeddings")
                        is_multimodal = embed_kwargs.get("is_multimodal")
                        mm_embed_meta = []
                        if isinstance(mm_embeds, list):
                            for item in mm_embeds:
                                mm_embed_meta.append(
                                    {
                                        "shape": str(getattr(item, "shape", None)),
                                        "dtype": str(getattr(item, "dtype", None)),
                                        "device": str(getattr(item, "device", None)),
                                    }
                                )
                        if _worker_debug_enabled():
                            _emit_worker_debug(
                                "embed_input_ids enter "
                                f"mm_embeds={len(mm_embeds) if isinstance(mm_embeds, list) else 'n/a'} "
                                f"is_multimodal_shape={getattr(is_multimodal, 'shape', None)} "
                                f"mm_embed_meta={mm_embed_meta}"
                            )
                        try:
                            result = _embed_input_ids_with_safe_mm_merge(
                                model_runner=self,
                                model=model,
                                original_embed_input_ids=embed_input_ids,
                                embed_args=embed_args,
                                embed_kwargs=embed_kwargs,
                            )
                        except Exception as exc:
                            _emit_worker_debug(
                                "embed_input_ids error "
                                f"{exc.__class__.__name__}: {exc}"
                            )
                            raise
                        if _worker_debug_enabled():
                            _emit_worker_debug(
                                "embed_input_ids exit "
                                f"shape={getattr(result, 'shape', None)}"
                            )
                        return result

                    try:
                        setattr(model, "embed_input_ids", patched_embed_input_ids)
                        restored.append((model, "embed_input_ids", embed_input_ids))
                    except Exception:
                        pass

            if _worker_debug_enabled():
                embed_text_input_ids = getattr(model, "_embed_text_input_ids", None)
                if callable(embed_text_input_ids):
                    @wraps(embed_text_input_ids)
                    def debug_embed_text_input_ids(
                        input_ids: Any,
                        inner_embed_input_ids: Any,
                        *text_args: Any,
                        **text_kwargs: Any,
                    ):
                        is_multimodal = text_kwargs.get("is_multimodal")
                        _emit_worker_debug(
                            "embed_text_input_ids enter "
                            f"input_shape={getattr(input_ids, 'shape', None)} "
                            f"input_dtype={getattr(input_ids, 'dtype', None)} "
                            f"input_device={getattr(input_ids, 'device', None)} "
                            f"is_multimodal_shape={getattr(is_multimodal, 'shape', None)} "
                            f"has_oov={getattr(model, '_has_oov_mm_tokens', None)}"
                        )

                        @wraps(inner_embed_input_ids)
                        def debug_inner_embed_input_ids(inner_input_ids: Any):
                            _emit_worker_debug(
                                "language_model_embed_input_ids enter "
                                f"shape={getattr(inner_input_ids, 'shape', None)} "
                                f"dtype={getattr(inner_input_ids, 'dtype', None)} "
                                f"device={getattr(inner_input_ids, 'device', None)}"
                            )
                            result = inner_embed_input_ids(inner_input_ids)
                            _emit_worker_debug(
                                "language_model_embed_input_ids exit "
                                f"shape={getattr(result, 'shape', None)} "
                                f"dtype={getattr(result, 'dtype', None)} "
                                f"device={getattr(result, 'device', None)}"
                            )
                            return result

                        try:
                            result = embed_text_input_ids(
                                input_ids,
                                debug_inner_embed_input_ids,
                                *text_args,
                                **text_kwargs,
                            )
                        except Exception as exc:
                            _emit_worker_debug(
                                "embed_text_input_ids error "
                                f"{exc.__class__.__name__}: {exc}"
                            )
                            raise
                        _emit_worker_debug(
                            "embed_text_input_ids exit "
                            f"shape={getattr(result, 'shape', None)} "
                            f"dtype={getattr(result, 'dtype', None)} "
                            f"device={getattr(result, 'device', None)}"
                        )
                        return result

                    try:
                        setattr(model, "_embed_text_input_ids", debug_embed_text_input_ids)
                        restored.append(
                            (model, "_embed_text_input_ids", embed_text_input_ids)
                        )
                    except Exception:
                        pass

                prepare_mm_inputs = getattr(self, "_prepare_mm_inputs", None)
                if callable(prepare_mm_inputs):
                    @wraps(prepare_mm_inputs)
                    def debug_prepare_mm_inputs(*prepare_args: Any, **prepare_kwargs: Any):
                        _emit_worker_debug("prepare_mm_inputs enter")
                        try:
                            result = prepare_mm_inputs(*prepare_args, **prepare_kwargs)
                        except Exception as exc:
                            _emit_worker_debug(
                                "prepare_mm_inputs error "
                                f"{exc.__class__.__name__}: {exc}"
                            )
                            raise
                        try:
                            input_ids, inputs_embeds = result
                            input_shape = getattr(input_ids, "shape", None)
                            embeds_shape = getattr(inputs_embeds, "shape", None)
                        except Exception:
                            input_shape = None
                            embeds_shape = None
                        _emit_worker_debug(
                            "prepare_mm_inputs exit "
                            f"input_ids_shape={input_shape} "
                            f"inputs_embeds_shape={embeds_shape}"
                        )
                        return result

                    try:
                        setattr(self, "_prepare_mm_inputs", debug_prepare_mm_inputs)
                        restored.append((self, "_prepare_mm_inputs", prepare_mm_inputs))
                    except Exception:
                        pass

            try:
                result = original_preprocess(self, scheduler_output, *args, **kwargs)
            except Exception as exc:
                _emit_worker_debug(
                    f"preprocess error {exc.__class__.__name__}: {exc}"
                )
                raise
            finally:
                for target, attr_name, original_value in reversed(restored):
                    try:
                        setattr(target, attr_name, original_value)
                    except Exception:
                        pass
            _emit_worker_debug("preprocess exit")
            return result

    gpu_model_runner_cls._update_states = wrapped_update_states
    gpu_model_runner_cls._batch_mm_inputs_from_scheduler = (
        wrapped_batch_mm_inputs_from_scheduler
    )
    if original_execute_mm_encoder is not None:
        gpu_model_runner_cls._execute_mm_encoder = wrapped_execute_mm_encoder
    if original_gather_mm_embeddings is not None:
        gpu_model_runner_cls._gather_mm_embeddings = wrapped_gather_mm_embeddings
    if original_preprocess is not None:
        gpu_model_runner_cls._preprocess = wrapped_preprocess
    setattr(gpu_model_runner_cls, _PATCH_MARKER_ATTR, True)
    _emit_worker_debug(
        "installed GPUModelRunner patch "
        f"execute_mm_encoder={'yes' if original_execute_mm_encoder is not None else 'no'} "
        f"gather_mm_embeddings={'yes' if original_gather_mm_embeddings is not None else 'no'} "
        f"preprocess={'yes' if original_preprocess is not None else 'no'}"
    )
    return True

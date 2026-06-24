from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
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
    connect_sidecar_client_from_env,
)
from mm_sidecar.sidecar.processor import run_descriptor_locally
from mm_sidecar.integrations.vllm_patch.qwen_adapter import (
    get_request_payload_from_qwen_mm_kwargs_item,
    is_synthetic_qwen_mm_kwargs_item,
    replace_feature_data_from_sidecar_artifacts,
)


_PATCH_MARKER_ATTR = "_mm_sidecar_worker_patch_installed"
_PREPARED_SCHEDULER_OUTPUT_ID_ATTR = (
    "_mm_sidecar_last_prepared_scheduler_output_id"
)


@dataclass(frozen=True, slots=True)
class WorkerSidecarBinding:
    request_id: str
    enabled: bool
    processor_signature: str | None
    prepared_image_count: int
    total_placeholder_token_count: int
    plan_payload: dict[str, Any]
    decoded_plan: Any


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


def _artifact_payload_bytes(artifacts: list[Any] | tuple[Any, ...]) -> int:
    total = 0
    for artifact in artifacts:
        descriptor = getattr(artifact, "descriptor", None)
        payload_nbytes = getattr(descriptor, "payload_nbytes", None)
        if isinstance(payload_nbytes, int):
            total += payload_nbytes
    return total


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
    producer_rank: int = 0,
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

    client = get_worker_sidecar_client(required=False)
    coordinator = SidecarFallbackCoordinator(
        manager=client,
        claimer_id=binding.request_id,
        producer_rank=producer_rank,
        near_ready_wait_ms=near_ready_wait_ms,
        poll_interval_ms=poll_interval_ms,
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

    replaced = 0
    for req_id in scheduled_encoder_inputs:
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        binding = get_request_mm_sidecar_binding(req_state)
        if binding is None:
            binding = bind_request_mm_sidecar(req_state)
        if binding is None:
            continue
        descriptors = list(binding.decoded_plan.fallback_descriptors)
        if not descriptors:
            continue
        if not _feature_data_missing_for_descriptors(req_state, descriptors):
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
                claimer_id=binding.request_id,
                producer_rank=0,
                near_ready_wait_ms=0.0,
                poll_interval_ms=1.0,
            )
            setattr(
                req_state,
                "mm_sidecar_source_plan",
                coordinator.preview_source_plan(descriptors=descriptors),
            )
            setattr(req_state, "mm_sidecar_fallback_descriptors", tuple(descriptors))
            continue
        try:
            coordinator = SidecarFallbackCoordinator(
                manager=client,
                claimer_id=binding.request_id,
                producer_rank=0,
                near_ready_wait_ms=2.0,
                poll_interval_ms=1.0,
            )
            fetch_start = time.perf_counter()
            fetch_batch = coordinator.fetch_according_to_plan(
                descriptors=descriptors,
                handles=list(binding.decoded_plan.handles),
            )
            fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
            artifacts = list(fetch_batch.sidecar_artifacts)
            if fetch_batch.fallback_descriptors:
                if _feature_data_missing_for_descriptors(
                    req_state,
                    fetch_batch.fallback_descriptors,
                ):
                    artifacts.extend(
                        _run_local_fallback_artifacts(
                            fetch_batch.fallback_descriptors,
                        )
                    )
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
            if _worker_fetch_profile_enabled():
                _emit_worker_debug(
                    "fetch_profile "
                    f"req={binding.request_id} "
                    f"media={len(descriptors)} "
                    f"sidecar_artifacts={len(fetch_batch.sidecar_artifacts)} "
                    f"fallback={len(fetch_batch.fallback_descriptors)} "
                    f"payload_bytes={_artifact_payload_bytes(artifacts)} "
                    f"fetch_ms={fetch_ms:.3f} "
                    f"replace_ms={replace_ms:.3f}"
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
            return original_execute_mm_encoder(self, scheduler_output)

    gpu_model_runner_cls._update_states = wrapped_update_states
    gpu_model_runner_cls._batch_mm_inputs_from_scheduler = (
        wrapped_batch_mm_inputs_from_scheduler
    )
    if original_execute_mm_encoder is not None:
        gpu_model_runner_cls._execute_mm_encoder = wrapped_execute_mm_encoder
    setattr(gpu_model_runner_cls, _PATCH_MARKER_ATTR, True)
    _emit_worker_debug(
        "installed GPUModelRunner patch "
        f"execute_mm_encoder={'yes' if original_execute_mm_encoder is not None else 'no'}"
    )
    return True

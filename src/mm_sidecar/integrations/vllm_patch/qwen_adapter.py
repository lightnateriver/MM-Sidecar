from __future__ import annotations

import json
from typing import Any

import numpy as np

from mm_sidecar.contracts import ProcessorSignature
from mm_sidecar.sidecar.protocol import PreparedArtifact

SYNTHETIC_PLACEHOLDER_ATTR = "_mm_sidecar_synthetic_placeholder"
REQUEST_PAYLOAD_ATTR = "_mm_sidecar_request_payload"


def _spatial_merge_size_from_signature(processor_signature: str | None) -> int:
    if not processor_signature:
        return 2
    for piece in processor_signature.split("|"):
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        if key == "merge":
            try:
                return max(1, int(value))
            except ValueError:
                return 2
    return 2


def _signature_parts(processor_signature: str | None) -> dict[str, str]:
    if not processor_signature:
        return {}
    return ProcessorSignature.parse(processor_signature)


def _int_signature_value(
    processor_signature: str | None,
    key: str,
    default: int,
) -> int:
    raw = _signature_parts(processor_signature).get(key)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _to_torch_tensor(value: Any, *, dtype: Any | None = None):
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.contiguous()


def _build_qwen_mm_kwargs_item(
    *,
    pixel_values: Any,
    image_grid_thw: Any,
    spatial_merge_size: int,
):
    from vllm.multimodal.inputs import (
        MultiModalFieldConfig,
        MultiModalKwargsItem,
    )

    image_pixel_grid_sizes = image_grid_thw.prod(-1)
    fields = {
        "pixel_values": MultiModalFieldConfig.flat_from_sizes(
            "image",
            image_pixel_grid_sizes,
        ),
        "image_grid_thw": MultiModalFieldConfig.batched(
            "image",
            keep_on_cpu=True,
        ),
    }
    # Keep the computed embed size available for local assertions/debugging; the
    # model's own field factory derives the same value from image_grid_thw.
    image_embed_grid_sizes = (
        image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
    )
    del image_embed_grid_sizes

    return MultiModalKwargsItem(
        {
            "pixel_values": fields["pixel_values"].build_elems(
                "pixel_values",
                pixel_values,
            )[0],
            "image_grid_thw": fields["image_grid_thw"].build_elems(
                "image_grid_thw",
                image_grid_thw,
            )[0],
        }
    )


def _planned_item_qwen_shape_info(
    planned_item: dict[str, Any],
    *,
    processor_signature: str | None,
) -> tuple[tuple[int, int, int], int, int]:
    raw_grid_thw = planned_item.get("image_grid_thw")
    if not isinstance(raw_grid_thw, (list, tuple)) or len(raw_grid_thw) != 3:
        raise ValueError("planned_item requires image_grid_thw 3-tuple")

    grid_thw = tuple(int(value) for value in raw_grid_thw)
    item_signature = (
        None
        if planned_item.get("processor_signature") is None
        else str(planned_item.get("processor_signature"))
    )
    effective_signature = item_signature or processor_signature
    patch_size = _int_signature_value(effective_signature, "patch", 14)
    temporal_patch_size = _int_signature_value(effective_signature, "temporal", 1)
    spatial_merge_size = _spatial_merge_size_from_signature(effective_signature)
    flattened_patch_size = 3 * temporal_patch_size * patch_size * patch_size
    return grid_thw, spatial_merge_size, flattened_patch_size


def sidecar_artifact_to_qwen_mm_kwargs_item(
    artifact: PreparedArtifact,
):
    import torch

    payload = artifact.payload
    pixel_values = _to_torch_tensor(payload.pixel_values, dtype=torch.float32)
    image_grid_thw = _to_torch_tensor(
        [payload.image_grid_thw],
        dtype=torch.long,
    )
    spatial_merge_size = _spatial_merge_size_from_signature(
        artifact.descriptor.processor_signature.value
    )
    return _build_qwen_mm_kwargs_item(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
    )


def planned_item_to_synthetic_qwen_mm_kwargs_item(
    planned_item: dict[str, Any],
    *,
    processor_signature: str | None,
):
    import torch

    grid_thw, spatial_merge_size, flattened_patch_size = (
        _planned_item_qwen_shape_info(
            planned_item,
            processor_signature=processor_signature,
        )
    )

    # The API server only needs shape-bearing placeholders to drive native
    # prompt update logic. Real pixel_values are supplied later by sidecar or
    # TP-worker fallback before encoder execution.
    pixel_values = torch.empty(
        (0, flattened_patch_size),
        dtype=torch.float32,
    )
    image_grid_thw = torch.tensor([grid_thw], dtype=torch.long)
    item = _build_qwen_mm_kwargs_item(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
    )
    setattr(item, SYNTHETIC_PLACEHOLDER_ATTR, True)
    return item


def planned_item_to_vit_dp_placeholder_qwen_mm_kwargs_item(
    planned_item: dict[str, Any],
    *,
    processor_signature: str | None,
):
    import torch

    grid_thw, spatial_merge_size, flattened_patch_size = (
        _planned_item_qwen_shape_info(
            planned_item,
            processor_signature=processor_signature,
        )
    )

    pixel_values = torch.zeros(
        (int(grid_thw[0]) * int(grid_thw[1]) * int(grid_thw[2]), flattened_patch_size),
        dtype=torch.float32,
    )
    image_grid_thw = torch.tensor([grid_thw], dtype=torch.long)
    item = _build_qwen_mm_kwargs_item(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
    )
    setattr(item, SYNTHETIC_PLACEHOLDER_ATTR, True)
    return item


def is_synthetic_qwen_mm_kwargs_item(item: Any) -> bool:
    return bool(getattr(item, SYNTHETIC_PLACEHOLDER_ATTR, False))


def attach_request_payload_to_qwen_mm_kwargs_item(
    item: Any,
    payload: dict[str, Any],
) -> None:
    _attach_request_payload(item, payload)
    values = getattr(item, "values", None)
    if callable(values):
        for value in values():
            _attach_request_payload(value, payload)


def _attach_request_payload(value: Any, payload: dict[str, Any]) -> None:
    try:
        setattr(
            value,
            REQUEST_PAYLOAD_ATTR,
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True),
        )
    except Exception:
        pass


def get_request_payload_from_qwen_mm_kwargs_item(item: Any) -> dict[str, Any] | None:
    payload = getattr(item, REQUEST_PAYLOAD_ATTR, None)
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return None
    try:
        decoded = json.loads(payload)
    except Exception:
        return None
    return decoded if isinstance(decoded, dict) else None


def replace_feature_data_from_sidecar_artifacts(
    req_state: Any,
    artifacts: tuple[PreparedArtifact, ...] | list[PreparedArtifact],
) -> int:
    mm_features = getattr(req_state, "mm_features", None)
    if not isinstance(mm_features, list):
        return 0

    artifact_by_index = {
        int(artifact.handle.request_media_index): artifact for artifact in artifacts
    }
    replaced = 0
    for index, artifact in artifact_by_index.items():
        if index < 0 or index >= len(mm_features):
            continue
        feature = mm_features[index]
        if getattr(feature, "modality", None) != "image":
            continue
        feature.data = sidecar_artifact_to_qwen_mm_kwargs_item(artifact)
        replaced += 1
    return replaced

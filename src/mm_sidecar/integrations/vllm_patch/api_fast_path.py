from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

from mm_sidecar.integrations.vllm_patch.context import get_current_capture
from mm_sidecar.integrations.vllm_patch.qwen_adapter import (
    attach_request_payload_to_qwen_mm_kwargs_item,
    planned_item_to_synthetic_qwen_mm_kwargs_item,
)


def api_fast_path_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_ENABLE_API_FAST_PATH", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def descriptor_only_capture_enabled() -> bool:
    value = os.getenv("MM_SIDECAR_DESCRIPTOR_ONLY_CAPTURE", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _record(timing_ctx: Any, name: str):
    recorder = getattr(timing_ctx, "record", None)
    if callable(recorder):
        return recorder(name)
    return nullcontext()


def _positive_modality_counts(mm_data_items: Any) -> dict[str, int]:
    get_all_counts = getattr(mm_data_items, "get_all_counts", None)
    if not callable(get_all_counts):
        return {}
    return {
        str(modality): int(count)
        for modality, count in dict(get_all_counts()).items()
        if int(count) > 0
    }


def _is_supported_qwen_processor(processor: Any) -> bool:
    cls = processor.__class__
    name = cls.__name__.lower()
    module = getattr(cls, "__module__", "").lower()
    return "qwen" in name and "vl" in name and "qwen" in module


def _planned_items_by_index(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = payload.get("planned_items")
    if not isinstance(raw_items, list):
        return []
    planned = [dict(item) for item in raw_items if isinstance(item, dict)]
    planned.sort(key=lambda item: int(item.get("request_media_index", 0)))
    return planned


def _hashes_from_payload(
    payload: dict[str, Any],
    planned_items: list[dict[str, Any]],
) -> dict[str, list[str]]:
    raw_handles = payload.get("handles")
    handle_by_index: dict[int, str] = {}
    if isinstance(raw_handles, list):
        for handle in raw_handles:
            if not isinstance(handle, dict):
                continue
            try:
                handle_by_index[int(handle["request_media_index"])] = str(
                    handle["cache_key"]
                )
            except (KeyError, TypeError, ValueError):
                continue

    image_hashes: list[str] = []
    for item in planned_items:
        index = int(item.get("request_media_index", len(image_hashes)))
        cache_key = handle_by_index.get(index)
        if cache_key is None:
            item_identity = str(item.get("item_identity") or item.get("source_key"))
            signature = str(
                item.get("processor_signature")
                or payload.get("processor_signature")
                or "processor=unknown"
            )
            cache_key = f"{item_identity}|{signature}"
        image_hashes.append(cache_key)
    return {"image": image_hashes}


def _mark_fast_path(
    payload: dict[str, Any],
    *,
    used: bool,
    reason: str,
    image_count: int = 0,
) -> None:
    payload["api_fast_path"] = {
        "used": used,
        "reason": reason,
        "image_count": image_count,
    }


def _build_mm_input(
    *,
    prompt_token_ids: list[int],
    mm_kwargs: Any,
    mm_hashes: dict[str, list[str]],
    mm_placeholders: Any,
) -> Any:
    try:
        from vllm.inputs import mm_input

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )
    except ImportError:
        pass

    try:
        from vllm.inputs.engine import mm_input

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )
    except ImportError:
        pass

    from vllm.multimodal.inputs import mm_inputs

    return mm_inputs(
        prompt_token_ids=prompt_token_ids,
        mm_kwargs=mm_kwargs,
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
    )


def try_apply_api_fast_path(
    processor: Any,
    inputs: Any,
    timing_ctx: Any,
) -> Any | None:
    if not api_fast_path_enabled():
        return None
    if not _is_supported_qwen_processor(processor):
        return None

    counts = _positive_modality_counts(getattr(inputs, "mm_data_items", None))
    if set(counts) != {"image"}:
        return None
    image_count = int(counts["image"])

    capture = get_current_capture()
    if capture is None or capture.sidecar_prepare is None:
        return None
    payload = capture.sidecar_prepare
    if not isinstance(payload, dict):
        return None

    planned_items = _planned_items_by_index(payload)
    if len(planned_items) != image_count:
        _mark_fast_path(
            payload,
            used=False,
            reason="planned_item_count_mismatch",
            image_count=image_count,
        )
        return None
    if not all("image_grid_thw" in item for item in planned_items):
        _mark_fast_path(
            payload,
            used=False,
            reason="missing_image_grid_thw",
            image_count=image_count,
        )
        return None

    from vllm.multimodal.inputs import MultiModalKwargsItems

    processor_signature = (
        None
        if payload.get("processor_signature") is None
        else str(payload.get("processor_signature"))
    )
    with _record(timing_ctx, "mm_sidecar_build_synthetic_mm_kwargs"):
        synthetic_items = [
            planned_item_to_synthetic_qwen_mm_kwargs_item(
                item,
                processor_signature=processor_signature,
            )
            for item in planned_items
        ]
        if synthetic_items:
            attach_request_payload_to_qwen_mm_kwargs_item(synthetic_items[0], payload)
        mm_kwargs = MultiModalKwargsItems(
            {
                "image": synthetic_items,
            }
        )
        mm_hashes = _hashes_from_payload(payload, planned_items)

    prompt = inputs.prompt
    with _record(timing_ctx, "mm_sidecar_tokenize_text_only"):
        if isinstance(prompt, str):
            prompt_ids = processor._apply_hf_processor_text_only(
                prompt,
                inputs.tokenization_kwargs,
            )
        else:
            prompt_ids = processor._apply_hf_processor_tokens_only(prompt)

    with _record(timing_ctx, "mm_sidecar_prompt_updates"):
        mm_prompt_updates = processor._get_mm_prompt_updates(
            inputs.mm_data_items,
            inputs.hf_processor_mm_kwargs,
            mm_kwargs,
        )
        prompt_ids, mm_placeholders = processor._maybe_apply_prompt_updates(
            mm_items=inputs.mm_data_items,
            prompt_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_prompt_updates=mm_prompt_updates,
            is_update_applied=False,
        )

    mm_placeholder_ranges = {
        modality: [item.to_range() for item in placeholders]
        for modality, placeholders in mm_placeholders.items()
    }
    _mark_fast_path(
        payload,
        used=True,
        reason="synthetic_qwen_image_path",
        image_count=image_count,
    )
    return _build_mm_input(
        prompt_token_ids=prompt_ids,
        mm_kwargs=mm_kwargs,
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholder_ranges,
    )

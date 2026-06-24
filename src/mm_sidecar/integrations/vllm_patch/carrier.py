from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm_sidecar.contracts import CapturedImageRef, IngressLimits, MediaTransport
from mm_sidecar.contracts.media_source import MediaSourceRef
from mm_sidecar.integrations.vllm_patch.context import RequestCapture
from mm_sidecar.sidecar.protocol import FallbackDescriptor, SidecarHandle

REQUEST_PAYLOAD_KEY = "mm_sidecar"
REQUEST_PAYLOAD_VERSION = 1


@dataclass(frozen=True, slots=True)
class SidecarRequestPlan:
    version: int
    request_id: str
    enabled: bool
    reason: str | None
    processor_signature: str | None
    prepared_image_count: int
    total_placeholder_token_count: int
    planned_items: tuple[dict[str, Any], ...]
    fallback_descriptors: tuple[FallbackDescriptor, ...]
    handles: tuple[SidecarHandle, ...]


def _serialize_media_source_ref(source_ref: MediaSourceRef) -> dict[str, Any]:
    return {
        "transport": source_ref.transport.value,
        "source_key": source_ref.source_key,
        "media_uuid": source_ref.media_uuid,
        "request_scope_key": source_ref.request_scope_key,
        "image_url": source_ref.image_url,
        "local_path": source_ref.local_path,
        "mime_type": source_ref.mime_type,
    }


def _deserialize_media_source_ref(payload: dict[str, Any]) -> MediaSourceRef:
    return MediaSourceRef(
        transport=MediaTransport(str(payload["transport"])),
        source_key=str(payload["source_key"]),
        media_uuid=str(payload["media_uuid"]),
        request_scope_key=(
            None
            if payload.get("request_scope_key") is None
            else str(payload["request_scope_key"])
        ),
        image_url=(
            None if payload.get("image_url") is None else str(payload["image_url"])
        ),
        local_path=(
            None if payload.get("local_path") is None else str(payload["local_path"])
        ),
        mime_type=(
            None if payload.get("mime_type") is None else str(payload["mime_type"])
        ),
    )


def _serialize_captured_image_ref(image_ref: CapturedImageRef) -> dict[str, Any]:
    return {
        "source_ref": _serialize_media_source_ref(image_ref.source_ref),
        "mime_type": image_ref.mime_type,
        "byte_size": image_ref.byte_size,
        "local_materialized_path": image_ref.local_materialized_path,
    }


def _deserialize_captured_image_ref(payload: dict[str, Any]) -> CapturedImageRef:
    return CapturedImageRef(
        source_ref=_deserialize_media_source_ref(dict(payload["source_ref"])),
        mime_type=(
            None if payload.get("mime_type") is None else str(payload["mime_type"])
        ),
        byte_size=(
            None if payload.get("byte_size") is None else int(payload["byte_size"])
        ),
        local_materialized_path=(
            None
            if payload.get("local_materialized_path") is None
            else str(payload["local_materialized_path"])
        ),
    )


def _serialize_ingress_limits(limits: IngressLimits) -> dict[str, int]:
    return {
        "max_image_count": int(limits.max_image_count),
        "max_encoded_bytes": int(limits.max_encoded_bytes),
        "max_decoded_bytes": int(limits.max_decoded_bytes),
        "max_pixels_per_image": int(limits.max_pixels_per_image),
    }


def _deserialize_ingress_limits(payload: dict[str, Any]) -> IngressLimits:
    return IngressLimits(
        max_image_count=int(payload["max_image_count"]),
        max_encoded_bytes=int(payload["max_encoded_bytes"]),
        max_decoded_bytes=int(payload["max_decoded_bytes"]),
        max_pixels_per_image=int(payload["max_pixels_per_image"]),
    )


def serialize_sidecar_handle(handle: SidecarHandle) -> dict[str, Any]:
    return {
        "request_id": handle.request_id,
        "request_media_index": int(handle.request_media_index),
        "cache_key": handle.cache_key,
        "epoch": int(handle.epoch),
    }


def deserialize_sidecar_handle(payload: dict[str, Any]) -> SidecarHandle:
    return SidecarHandle(
        request_id=str(payload["request_id"]),
        request_media_index=int(payload["request_media_index"]),
        cache_key=str(payload["cache_key"]),
        epoch=int(payload["epoch"]),
    )


def serialize_fallback_descriptor(descriptor: FallbackDescriptor) -> dict[str, Any]:
    return {
        "request_id": descriptor.request_id,
        "request_media_index": int(descriptor.request_media_index),
        "captured_image": _serialize_captured_image_ref(descriptor.captured_image),
        "ingress_limits": _serialize_ingress_limits(descriptor.ingress_limits),
        "processor_signature_value": descriptor.processor_signature_value,
        "item_identity": descriptor.item_identity,
        "orig_size_hw": (
            None
            if descriptor.orig_size_hw is None
            else [int(descriptor.orig_size_hw[0]), int(descriptor.orig_size_hw[1])]
        ),
        "http_headers": [
            [str(key), str(value)] for key, value in descriptor.http_headers
        ],
        "http_timeout_ms": int(descriptor.http_timeout_ms),
        "allow_redirects": bool(descriptor.allow_redirects),
    }


def deserialize_fallback_descriptor(payload: dict[str, Any]) -> FallbackDescriptor:
    orig_size_raw = payload.get("orig_size_hw")
    orig_size_hw = None
    if isinstance(orig_size_raw, (list, tuple)) and len(orig_size_raw) == 2:
        orig_size_hw = (int(orig_size_raw[0]), int(orig_size_raw[1]))

    return FallbackDescriptor(
        request_id=str(payload["request_id"]),
        request_media_index=int(payload["request_media_index"]),
        captured_image=_deserialize_captured_image_ref(dict(payload["captured_image"])),
        ingress_limits=_deserialize_ingress_limits(dict(payload["ingress_limits"])),
        processor_signature_value=str(payload["processor_signature_value"]),
        item_identity=str(payload["item_identity"]),
        orig_size_hw=orig_size_hw,
        http_headers=tuple(
            (str(item[0]), str(item[1]))
            for item in payload.get("http_headers", [])
            if isinstance(item, (list, tuple)) and len(item) == 2
        ),
        http_timeout_ms=int(payload.get("http_timeout_ms", 30_000)),
        allow_redirects=bool(payload.get("allow_redirects", True)),
    )


def build_request_sidecar_payload(capture: RequestCapture) -> dict[str, Any] | None:
    prepared = capture.sidecar_prepare
    if prepared is None:
        return None

    descriptors: list[dict[str, Any]] = []
    handles: list[dict[str, Any]] = []
    for _item_index, descriptor, handle in capture.iter_prepared_sidecar_items():
        if descriptor is not None:
            descriptors.append(serialize_fallback_descriptor(descriptor))
        if handle is not None:
            handles.append(serialize_sidecar_handle(handle))

    return {
        "version": REQUEST_PAYLOAD_VERSION,
        "request_id": capture.request_id,
        "enabled": bool(prepared.get("enabled", False)),
        "reason": (
            None if prepared.get("reason") is None else str(prepared.get("reason"))
        ),
        "processor_signature": (
            None
            if prepared.get("processor_signature") is None
            else str(prepared.get("processor_signature"))
        ),
        "prepared_image_count": int(prepared.get("prepared_image_count", 0)),
        "total_placeholder_token_count": int(
            prepared.get("total_placeholder_token_count", 0)
        ),
        "planned_items": list(prepared.get("planned_items") or []),
        "fallback_descriptors": descriptors,
        "handles": handles,
    }


def attach_sidecar_payload_to_params(
    params: Any,
    capture: RequestCapture,
) -> dict[str, Any] | None:
    payload = build_request_sidecar_payload(capture)
    if payload is None:
        return None

    raw_extra_args = getattr(params, "extra_args", None)
    if isinstance(raw_extra_args, dict):
        raw_extra_args[REQUEST_PAYLOAD_KEY] = payload
        return payload
    else:
        extra_args = {}
    extra_args[REQUEST_PAYLOAD_KEY] = payload
    try:
        setattr(params, "extra_args", extra_args)
    except Exception:
        object.__setattr__(params, "extra_args", extra_args)
    return payload


def get_sidecar_payload_from_params(params: Any) -> dict[str, Any] | None:
    raw_extra_args = getattr(params, "extra_args", None)
    if not isinstance(raw_extra_args, dict):
        return None
    payload = raw_extra_args.get(REQUEST_PAYLOAD_KEY)
    return payload if isinstance(payload, dict) else None


def decode_sidecar_request_plan(payload: dict[str, Any]) -> SidecarRequestPlan:
    return SidecarRequestPlan(
        version=int(payload.get("version", REQUEST_PAYLOAD_VERSION)),
        request_id=str(payload["request_id"]),
        enabled=bool(payload.get("enabled", False)),
        reason=None if payload.get("reason") is None else str(payload.get("reason")),
        processor_signature=(
            None
            if payload.get("processor_signature") is None
            else str(payload["processor_signature"])
        ),
        prepared_image_count=int(payload.get("prepared_image_count", 0)),
        total_placeholder_token_count=int(
            payload.get("total_placeholder_token_count", 0)
        ),
        planned_items=tuple(
            dict(item)
            for item in payload.get("planned_items", [])
            if isinstance(item, dict)
        ),
        fallback_descriptors=tuple(
            deserialize_fallback_descriptor(dict(item))
            for item in payload.get("fallback_descriptors", [])
            if isinstance(item, dict)
        ),
        handles=tuple(
            deserialize_sidecar_handle(dict(item))
            for item in payload.get("handles", [])
            if isinstance(item, dict)
        ),
    )

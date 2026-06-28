from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any

from mm_sidecar.contracts import CapturedImageRef, NormalizedImage

_REQUEST_CAPTURE: contextvars.ContextVar["RequestCapture | None"] = (
    contextvars.ContextVar("mm_sidecar_request_capture", default=None)
)


def _now_ms() -> float:
    return time.time() * 1000.0


def _serialize_hw(value: tuple[int, int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(value[0]), int(value[1])]


def _serialize_image(item_index: int, media_uuid: str, image: NormalizedImage) -> dict[str, Any]:
    source_ref = image.source_ref
    return {
        "item_index": item_index,
        "media_uuid": media_uuid,
        "transport": source_ref.transport.value,
        "source_key": source_ref.source_key,
        "request_scope_key": source_ref.request_scope_key,
        "image_url": source_ref.image_url,
        "local_path": source_ref.local_path,
        "mime_type": image.mime_type,
        "orig_size_hw": _serialize_hw(image.orig_size_hw),
        "decoded_size_hw": _serialize_hw(image.decoded_size_hw),
        "byte_size": image.byte_size,
        "local_materialized_path": image.local_materialized_path,
    }


def _serialize_capture_ref(
    item_index: int,
    media_uuid: str,
    image_ref: CapturedImageRef,
) -> dict[str, Any]:
    source_ref = image_ref.source_ref
    return {
        "item_index": item_index,
        "media_uuid": media_uuid,
        "transport": source_ref.transport.value,
        "source_key": source_ref.source_key,
        "request_scope_key": source_ref.request_scope_key,
        "image_url": source_ref.image_url,
        "local_path": source_ref.local_path,
        "mime_type": image_ref.mime_type,
        "orig_size_hw": None,
        "decoded_size_hw": None,
        "byte_size": image_ref.byte_size,
        "local_materialized_path": image_ref.local_materialized_path,
        "capture_kind": "descriptor_only",
    }


@dataclass(slots=True)
class RequestCapture:
    request_id: str
    method: str
    path: str
    sidecar_manager: Any | None = None
    started_at_ms: float = field(default_factory=_now_ms)
    finished_at_ms: float | None = None
    status_code: int | None = None
    reserved_image_count: int = 0
    prompt_has_multimodal: bool = False
    prompt_text_length: int | None = None
    prompt_mm_uuid_counts: dict[str, int] = field(default_factory=dict)
    prompt_mm_uuids: dict[str, list[str | None]] = field(default_factory=dict)
    images: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sidecar_prepare: dict[str, Any] | None = None
    worker_fetch_profile: dict[str, Any] | None = None
    _captured_refs: list[tuple[int, str, CapturedImageRef]] = field(default_factory=list)
    _normalized_images: list[tuple[int, str, NormalizedImage]] = field(default_factory=list)
    _prepared_descriptors: dict[int, Any] = field(default_factory=dict)
    _prepared_handles: dict[int, Any] = field(default_factory=dict)

    def reserve_image_slot(self) -> int:
        item_index = self.reserved_image_count
        self.reserved_image_count += 1
        return item_index

    def add_captured_image_ref(
        self,
        item_index: int,
        media_uuid: str,
        image_ref: CapturedImageRef,
    ) -> None:
        self._captured_refs.append((item_index, media_uuid, image_ref))
        self.images.append(
            _serialize_capture_ref(
                item_index=item_index,
                media_uuid=media_uuid,
                image_ref=image_ref,
            )
        )

    def add_normalized_image(
        self,
        item_index: int,
        media_uuid: str,
        normalized_image: NormalizedImage,
    ) -> None:
        self._normalized_images.append((item_index, media_uuid, normalized_image))
        serialized = _serialize_image(
            item_index=item_index,
            media_uuid=media_uuid,
            image=normalized_image,
        )
        serialized["capture_kind"] = "normalized"
        for idx, existing in enumerate(self.images):
            if (
                int(existing.get("item_index", -1)) == int(item_index)
                and str(existing.get("media_uuid")) == media_uuid
            ):
                self.images[idx] = serialized
                break
        else:
            self.images.append(serialized)

    def iter_captured_image_refs(self) -> list[tuple[int, str, CapturedImageRef]]:
        return sorted(self._captured_refs, key=lambda item: int(item[0]))

    def iter_normalized_images(self) -> list[tuple[int, str, NormalizedImage]]:
        return sorted(self._normalized_images, key=lambda item: int(item[0]))

    def add_prepared_sidecar_item(
        self,
        item_index: int,
        descriptor: Any,
        handle: Any,
    ) -> None:
        self._prepared_descriptors[int(item_index)] = descriptor
        self._prepared_handles[int(item_index)] = handle

    def get_prepared_descriptor(self, item_index: int) -> Any | None:
        return self._prepared_descriptors.get(int(item_index))

    def get_prepared_handle(self, item_index: int) -> Any | None:
        return self._prepared_handles.get(int(item_index))

    def iter_prepared_sidecar_items(self) -> list[tuple[int, Any, Any]]:
        indexes = sorted(self._prepared_descriptors)
        return [
            (
                item_index,
                self._prepared_descriptors[item_index],
                self._prepared_handles.get(item_index),
            )
            for item_index in indexes
        ]

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def set_sidecar_prepare(self, payload: dict[str, Any]) -> None:
        self.sidecar_prepare = payload

    def add_render_metadata(self, prompt: dict[str, Any]) -> None:
        if isinstance(prompt.get("prompt"), str):
            self.prompt_text_length = len(prompt["prompt"])

        mm_uuids = prompt.get("multi_modal_uuids")
        if isinstance(mm_uuids, dict):
            self.prompt_has_multimodal = True
            normalized_uuids: dict[str, list[str | None]] = {}
            for modality, values in mm_uuids.items():
                if not isinstance(values, list):
                    continue
                normalized_values = [
                    None if value is None else str(value) for value in values
                ]
                normalized_uuids[str(modality)] = normalized_values
            self.prompt_mm_uuids = normalized_uuids
            self.prompt_mm_uuid_counts = {
                modality: len(values) for modality, values in normalized_uuids.items()
            }

        if prompt.get("multi_modal_data") is not None:
            self.prompt_has_multimodal = True

    def finalize(self, status_code: int | None) -> None:
        self.finished_at_ms = _now_ms()
        self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        images = sorted(self.images, key=lambda item: int(item["item_index"]))
        duration_ms = None
        if self.finished_at_ms is not None:
            duration_ms = self.finished_at_ms - self.started_at_ms

        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "duration_ms": duration_ms,
            "reserved_image_count": self.reserved_image_count,
            "captured_image_count": len(images),
            "prompt_has_multimodal": self.prompt_has_multimodal,
            "prompt_text_length": self.prompt_text_length,
            "prompt_mm_uuid_counts": dict(self.prompt_mm_uuid_counts),
            "prompt_mm_uuids": dict(self.prompt_mm_uuids),
            "images": images,
            "sidecar_prepare": self.sidecar_prepare,
            "worker_fetch_profile": self.worker_fetch_profile,
            "errors": list(self.errors),
        }


def get_current_capture() -> RequestCapture | None:
    return _REQUEST_CAPTURE.get()


def set_current_capture(capture: RequestCapture) -> contextvars.Token[RequestCapture | None]:
    return _REQUEST_CAPTURE.set(capture)


def reset_current_capture(token: contextvars.Token[RequestCapture | None]) -> None:
    _REQUEST_CAPTURE.reset(token)

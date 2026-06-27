from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import MediaTransport, StorageKind
from .errors import SidecarContractError
from .enums import SidecarErrorCode


@dataclass(frozen=True, slots=True)
class MediaSourceRef:
    transport: MediaTransport
    source_key: str
    media_uuid: str
    request_scope_key: str | None
    image_url: str | None = None
    local_path: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if not self.source_key:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_SOURCE,
                "source_key must not be empty",
            )
        if not self.media_uuid:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_SOURCE,
                "media_uuid must not be empty",
            )
        if self.transport is MediaTransport.LOCAL_PATH and not self.local_path:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_SOURCE,
                "local_path transport requires local_path",
            )
        if self.transport in (MediaTransport.HTTP, MediaTransport.BASE64) and not self.image_url:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_SOURCE,
                f"{self.transport.value} transport requires image_url",
            )


@dataclass(frozen=True, slots=True)
class CapturedImageRef:
    source_ref: MediaSourceRef
    mime_type: str | None = None
    byte_size: int | None = None
    local_materialized_path: str | None = None


@dataclass(frozen=True, slots=True)
class LocalFileTensorPayloadRef:
    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    format: str = "npy"


@dataclass(frozen=True, slots=True)
class ImageTensorPayload:
    pixel_values: Any
    image_grid_thw: tuple[int, int, int]
    payload_shape: tuple[int, int]
    payload_dtype: str
    storage_kind: StorageKind
    resized_size_hw: tuple[int, int] | None = None
    orig_size_hw: tuple[int, int] | None = None
    pixel_mean: float | None = None
    pixel_std: float | None = None

    @property
    def nbytes(self) -> int:
        pixel_values = self.pixel_values
        if isinstance(pixel_values, LocalFileTensorPayloadRef):
            return int(pixel_values.nbytes)
        if hasattr(pixel_values, "nbytes"):
            return int(pixel_values.nbytes)
        if isinstance(pixel_values, (bytes, bytearray, memoryview)):
            return len(pixel_values)
        return 0


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    source_ref: MediaSourceRef
    orig_size_hw: tuple[int, int]
    mime_type: str
    byte_size: int | None = None
    decoded_size_hw: tuple[int, int] | None = None
    local_materialized_path: str | None = None

    def __post_init__(self) -> None:
        height, width = self.orig_size_hw
        if height <= 0 or width <= 0:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_SOURCE,
                "orig_size_hw must contain positive integers",
            )

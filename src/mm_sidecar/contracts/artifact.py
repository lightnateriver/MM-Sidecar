from __future__ import annotations

from dataclasses import dataclass

from .enums import SidecarErrorCode, StorageKind
from .errors import SidecarContractError
from .signature import ProcessorSignature


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    artifact_id: str
    item_identity: str
    processor_signature: ProcessorSignature
    image_grid_thw: tuple[int, int, int]
    payload_shape: tuple[int, int]
    payload_dtype: str
    storage_kind: StorageKind
    payload_nbytes: int | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_ARTIFACT,
                "artifact_id must not be empty",
            )
        if not self.item_identity:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_ARTIFACT,
                "item_identity must not be empty",
            )
        if len(self.image_grid_thw) != 3 or any(v <= 0 for v in self.image_grid_thw):
            raise SidecarContractError(
                SidecarErrorCode.INVALID_ARTIFACT,
                "image_grid_thw must be a positive 3-tuple",
            )
        if len(self.payload_shape) != 2 or any(v <= 0 for v in self.payload_shape):
            raise SidecarContractError(
                SidecarErrorCode.INVALID_ARTIFACT,
                "payload_shape must be a positive 2-tuple",
            )
        if not self.payload_dtype:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_ARTIFACT,
                "payload_dtype must not be empty",
            )


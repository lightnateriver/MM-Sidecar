from __future__ import annotations

from dataclasses import dataclass

from .errors import SidecarContractError
from .enums import SidecarErrorCode
from .signature import ProcessorSignature


@dataclass(frozen=True, slots=True)
class ImageScheduleItem:
    item_index: int
    item_identity: str
    processor_signature: ProcessorSignature
    orig_size_hw: tuple[int, int]
    preprocessed_size_hw: tuple[int, int]
    image_grid_thw: tuple[int, int, int]
    placeholder_token_count: int

    def __post_init__(self) -> None:
        if self.item_index < 0:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_MANIFEST,
                "item_index must be non-negative",
            )
        if not self.item_identity:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_MANIFEST,
                "item_identity must not be empty",
            )
        if self.placeholder_token_count <= 0:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_MANIFEST,
                "placeholder_token_count must be positive",
            )
        if len(self.image_grid_thw) != 3 or any(v <= 0 for v in self.image_grid_thw):
            raise SidecarContractError(
                SidecarErrorCode.INVALID_MANIFEST,
                "image_grid_thw must be a positive 3-tuple",
            )


@dataclass(frozen=True, slots=True)
class ScheduleManifest:
    image_count: int
    total_placeholder_token_count: int
    items: tuple[ImageScheduleItem, ...]

    def __post_init__(self) -> None:
        if self.image_count <= 0:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_MANIFEST,
                "image_count must be positive",
            )
        if len(self.items) != self.image_count:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_MANIFEST,
                "image_count must equal len(items)",
            )
        expected_total = sum(item.placeholder_token_count for item in self.items)
        if self.total_placeholder_token_count != expected_total:
            raise SidecarContractError(
                SidecarErrorCode.INVALID_MANIFEST,
                "total_placeholder_token_count must equal sum(item.placeholder_token_count)",
            )


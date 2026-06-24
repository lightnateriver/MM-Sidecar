from __future__ import annotations

from dataclasses import dataclass

from .enums import SidecarErrorCode
from .errors import SidecarContractError


@dataclass(frozen=True, slots=True)
class IngressLimits:
    max_image_count: int
    max_encoded_bytes: int
    max_decoded_bytes: int
    max_pixels_per_image: int

    def validate_image_count(self, image_count: int) -> None:
        if image_count > self.max_image_count:
            raise SidecarContractError(
                SidecarErrorCode.IMAGE_COUNT_LIMIT_EXCEEDED,
                f"image_count={image_count} exceeds max_image_count={self.max_image_count}",
            )

    def validate_encoded_bytes(self, encoded_bytes: int) -> None:
        if encoded_bytes > self.max_encoded_bytes:
            raise SidecarContractError(
                SidecarErrorCode.PAYLOAD_LIMIT_EXCEEDED,
                f"encoded_bytes={encoded_bytes} exceeds max_encoded_bytes={self.max_encoded_bytes}",
            )

    def validate_decoded_bytes(self, decoded_bytes: int) -> None:
        if decoded_bytes > self.max_decoded_bytes:
            raise SidecarContractError(
                SidecarErrorCode.PAYLOAD_LIMIT_EXCEEDED,
                f"decoded_bytes={decoded_bytes} exceeds max_decoded_bytes={self.max_decoded_bytes}",
            )

    def validate_pixel_count(self, pixel_count: int) -> None:
        if pixel_count > self.max_pixels_per_image:
            raise SidecarContractError(
                SidecarErrorCode.PAYLOAD_LIMIT_EXCEEDED,
                f"pixel_count={pixel_count} exceeds max_pixels_per_image={self.max_pixels_per_image}",
            )


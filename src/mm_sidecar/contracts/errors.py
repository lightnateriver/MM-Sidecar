from __future__ import annotations

from dataclasses import dataclass

from .enums import SidecarErrorCode


@dataclass(slots=True)
class SidecarContractError(Exception):
    code: SidecarErrorCode
    message: str

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"


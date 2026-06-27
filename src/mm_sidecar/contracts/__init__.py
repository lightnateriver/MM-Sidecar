"""Public contract objects for staged mm-sidecar development."""

from .artifact import ArtifactDescriptor
from .enums import MediaTransport, SidecarErrorCode, StorageKind
from .errors import SidecarContractError
from .manifest import ImageScheduleItem, ScheduleManifest
from .limits import IngressLimits
from .media_source import (
    CapturedImageRef,
    ImageTensorPayload,
    LocalFileTensorPayloadRef,
    MediaSourceRef,
    NormalizedImage,
)
from .signature import ProcessorConfig, ProcessorSignature

__all__ = [
    "ArtifactDescriptor",
    "CapturedImageRef",
    "ImageScheduleItem",
    "IngressLimits",
    "ImageTensorPayload",
    "LocalFileTensorPayloadRef",
    "MediaSourceRef",
    "MediaTransport",
    "NormalizedImage",
    "ProcessorConfig",
    "ProcessorSignature",
    "ScheduleManifest",
    "SidecarContractError",
    "SidecarErrorCode",
    "StorageKind",
]

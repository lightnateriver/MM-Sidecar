from __future__ import annotations

from enum import Enum


class MediaTransport(str, Enum):
    LOCAL_PATH = "local_path"
    HTTP = "http"
    BASE64 = "base64"


class StorageKind(str, Enum):
    CPU_MEMORY = "cpu_memory"
    LOCAL_FILE = "local_file"
    REMOTE_REF = "remote_ref"


class SidecarErrorCode(str, Enum):
    INVALID_TRANSPORT = "invalid_transport"
    INVALID_SOURCE = "invalid_source"
    INVALID_MANIFEST = "invalid_manifest"
    INVALID_ARTIFACT = "invalid_artifact"
    INVALID_PROCESSOR_SIGNATURE = "invalid_processor_signature"
    PAYLOAD_LIMIT_EXCEEDED = "payload_limit_exceeded"
    IMAGE_COUNT_LIMIT_EXCEEDED = "image_count_limit_exceeded"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    SOURCE_NOT_READY = "source_not_ready"
    FALLBACK_REQUIRED = "fallback_required"


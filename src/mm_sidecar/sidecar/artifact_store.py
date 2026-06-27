from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from mm_sidecar.contracts import (
    ArtifactDescriptor,
    ImageTensorPayload,
    LocalFileTensorPayloadRef,
    StorageKind,
)


def local_file_payload_enabled() -> bool:
    storage = os.environ.get("MM_SIDECAR_PAYLOAD_STORAGE", "").strip().lower()
    if storage in {"local_file", "file", "npy"}:
        return True
    raw = os.environ.get("MM_SIDECAR_ENABLE_LOCAL_FILE_PAYLOAD", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def local_file_payload_mmap_enabled() -> bool:
    raw = os.environ.get("MM_SIDECAR_PAYLOAD_MMAP", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def materialize_payload_to_local_file(
    *,
    cache_key: str,
    epoch: int,
    descriptor: ArtifactDescriptor,
    payload: ImageTensorPayload,
) -> tuple[ArtifactDescriptor, ImageTensorPayload, float]:
    started = time.perf_counter()
    payload_dir = Path(
        os.environ.get("MM_SIDECAR_PAYLOAD_DIR", "")
        or (Path(tempfile.gettempdir()) / "mm_sidecar_payloads")
    )
    payload_dir.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(
        f"{cache_key}|{epoch}|{os.getpid()}|{time.time_ns()}".encode("utf-8")
    ).hexdigest()
    final_path = payload_dir / f"{token}.npy"
    tmp_path = payload_dir / f".{token}.tmp"

    pixel_values = np.ascontiguousarray(payload.pixel_values)
    with open(tmp_path, "wb") as handle:
        np.save(handle, pixel_values, allow_pickle=False)
    os.replace(tmp_path, final_path)

    ref = LocalFileTensorPayloadRef(
        path=str(final_path),
        shape=tuple(int(dim) for dim in pixel_values.shape),
        dtype=str(pixel_values.dtype),
        nbytes=int(pixel_values.nbytes),
    )
    stored_descriptor = replace(
        descriptor,
        storage_kind=StorageKind.LOCAL_FILE,
        payload_nbytes=int(pixel_values.nbytes),
    )
    stored_payload = replace(
        payload,
        pixel_values=ref,
        storage_kind=StorageKind.LOCAL_FILE,
    )
    return stored_descriptor, stored_payload, (time.perf_counter() - started) * 1000.0


def load_local_file_tensor_ref(ref: LocalFileTensorPayloadRef) -> Any:
    if ref.format != "npy":
        raise ValueError(f"unsupported local tensor payload format: {ref.format}")
    array = np.load(
        ref.path,
        mmap_mode="c" if local_file_payload_mmap_enabled() else None,
        allow_pickle=False,
    )
    if tuple(array.shape) != tuple(ref.shape):
        raise ValueError(
            "local tensor payload shape mismatch: "
            f"expected={ref.shape} actual={tuple(array.shape)}"
        )
    if str(array.dtype) != str(ref.dtype):
        raise ValueError(
            "local tensor payload dtype mismatch: "
            f"expected={ref.dtype} actual={array.dtype}"
        )
    return array


def cleanup_local_file_payload(payload: Any) -> None:
    ref = getattr(payload, "pixel_values", None)
    if not isinstance(ref, LocalFileTensorPayloadRef):
        return
    try:
        os.unlink(ref.path)
    except FileNotFoundError:
        return
    except OSError:
        return

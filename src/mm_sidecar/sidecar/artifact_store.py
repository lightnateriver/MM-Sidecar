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
    if storage in {"local_file", "file", "npy", "raw"}:
        return True
    raw = os.environ.get("MM_SIDECAR_ENABLE_LOCAL_FILE_PAYLOAD", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def local_file_payload_mmap_enabled() -> bool:
    raw = os.environ.get("MM_SIDECAR_PAYLOAD_MMAP", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def local_file_payload_format() -> str:
    raw = os.environ.get("MM_SIDECAR_PAYLOAD_FILE_FORMAT", "npy").strip().lower()
    if not raw:
        return "npy"
    if raw not in {"npy", "raw"}:
        raise ValueError(f"unsupported local tensor payload file format: {raw}")
    return raw


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
    payload_format = local_file_payload_format()
    final_path = payload_dir / f"{token}.{payload_format}"
    tmp_path = payload_dir / f".{token}.tmp"

    pixel_values = np.ascontiguousarray(payload.pixel_values)
    with open(tmp_path, "wb") as handle:
        if payload_format == "raw":
            handle.write(pixel_values.tobytes(order="C"))
        else:
            np.save(handle, pixel_values, allow_pickle=False)
    os.replace(tmp_path, final_path)

    ref = LocalFileTensorPayloadRef(
        path=str(final_path),
        shape=tuple(int(dim) for dim in pixel_values.shape),
        dtype=str(pixel_values.dtype),
        nbytes=int(pixel_values.nbytes),
        format=payload_format,
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


def _expected_raw_nbytes(ref: LocalFileTensorPayloadRef) -> int:
    dtype = np.dtype(ref.dtype)
    item_count = 1
    for dim in ref.shape:
        item_count *= int(dim)
    return item_count * int(dtype.itemsize)


def _validate_local_file_tensor_array(
    ref: LocalFileTensorPayloadRef,
    array: Any,
) -> None:
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
    if int(getattr(array, "nbytes", -1)) != int(ref.nbytes):
        raise ValueError(
            "local tensor payload nbytes mismatch: "
            f"expected={ref.nbytes} actual={getattr(array, 'nbytes', None)}"
        )


def load_local_file_tensor_ref(ref: LocalFileTensorPayloadRef) -> Any:
    if ref.format == "raw":
        expected_nbytes = _expected_raw_nbytes(ref)
        if int(ref.nbytes) != expected_nbytes:
            raise ValueError(
                "raw local tensor payload nbytes mismatch: "
                f"expected={expected_nbytes} actual={ref.nbytes}"
            )
        actual_size = Path(ref.path).stat().st_size
        if actual_size != expected_nbytes:
            raise ValueError(
                "raw local tensor payload file size mismatch: "
                f"expected={expected_nbytes} actual={actual_size}"
            )
        dtype = np.dtype(ref.dtype)
        if local_file_payload_mmap_enabled():
            array = np.memmap(
                ref.path,
                dtype=dtype,
                mode="c",
                shape=tuple(int(dim) for dim in ref.shape),
                order="C",
            )
        else:
            array = np.fromfile(ref.path, dtype=dtype).reshape(
                tuple(int(dim) for dim in ref.shape)
            )
        _validate_local_file_tensor_array(ref, array)
        return array
    if ref.format != "npy":
        raise ValueError(f"unsupported local tensor payload format: {ref.format}")
    array = np.load(
        ref.path,
        mmap_mode="c" if local_file_payload_mmap_enabled() else None,
        allow_pickle=False,
    )
    _validate_local_file_tensor_array(ref, array)
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

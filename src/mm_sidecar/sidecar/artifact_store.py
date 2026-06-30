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
    if storage in {
        "local_file",
        "file",
        "npy",
        "raw",
        "torch",
        "numpy_bf16",
    }:
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
    if raw not in {"npy", "raw", "torch", "numpy_bf16"}:
        raise ValueError(f"unsupported local tensor payload file format: {raw}")
    return raw


def local_file_payload_dtype() -> str:
    raw = os.environ.get("MM_SIDECAR_PAYLOAD_DTYPE", "fp32").strip().lower()
    if not raw or raw in {"fp32", "float32"}:
        return "fp32"
    if raw in {"bf16", "bfloat16"}:
        return "bf16"
    raise ValueError(f"unsupported local tensor payload dtype: {raw}")


def local_file_ref_storage_dtype(ref: LocalFileTensorPayloadRef) -> str:
    if ref.format == "numpy_bf16":
        return "uint16"
    return ref.dtype


def _torch_dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def _torch_tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _item_count(shape: tuple[int, ...]) -> int:
    item_count = 1
    for dim in shape:
        item_count *= int(dim)
    return item_count


def _expected_numpy_bf16_nbytes(ref: LocalFileTensorPayloadRef) -> int:
    return _item_count(ref.shape) * np.dtype(np.uint16).itemsize


def _validate_local_file_payload_config(
    *,
    payload_format: str,
    payload_dtype: str,
) -> None:
    supported = {
        ("npy", "fp32"),
        ("raw", "fp32"),
        ("torch", "bf16"),
        ("numpy_bf16", "bf16"),
    }
    if (payload_format, payload_dtype) not in supported:
        raise ValueError(
            "unsupported local tensor payload format/dtype combination: "
            f"format={payload_format} dtype={payload_dtype}"
        )


def materialize_payload_to_local_file(
    *,
    cache_key: str,
    epoch: int,
    descriptor: ArtifactDescriptor,
    payload: ImageTensorPayload,
) -> tuple[ArtifactDescriptor, ImageTensorPayload, dict[str, Any]]:
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
    payload_dtype = local_file_payload_dtype()
    _validate_local_file_payload_config(
        payload_format=payload_format,
        payload_dtype=payload_dtype,
    )
    final_path = payload_dir / f"{token}.{payload_format}"
    tmp_path = payload_dir / f".{token}.tmp"

    pixel_values = np.ascontiguousarray(payload.pixel_values)
    logical_dtype = str(pixel_values.dtype)
    stored_dtype = str(pixel_values.dtype)
    stored_nbytes = int(pixel_values.nbytes)
    payload_tensor_cast_ms = 0.0
    payload_torch_save_ms = 0.0
    payload_numpy_bf16_save_ms = 0.0

    with open(tmp_path, "wb") as handle:
        if payload_format == "raw":
            handle.write(pixel_values.tobytes(order="C"))
        elif payload_format == "npy":
            np.save(handle, pixel_values, allow_pickle=False)
        elif payload_format == "torch":
            import torch

            cast_started = time.perf_counter()
            torch_tensor = torch.from_numpy(pixel_values).to(dtype=torch.bfloat16)
            payload_tensor_cast_ms = (
                time.perf_counter() - cast_started
            ) * 1000.0
            logical_dtype = _torch_dtype_name(torch_tensor.dtype)
            stored_dtype = logical_dtype
            stored_nbytes = _torch_tensor_nbytes(torch_tensor)
            save_started = time.perf_counter()
            torch.save(torch_tensor, handle)
            payload_torch_save_ms = (time.perf_counter() - save_started) * 1000.0
        else:
            import torch

            cast_started = time.perf_counter()
            torch_tensor = torch.from_numpy(pixel_values).to(dtype=torch.bfloat16)
            bf16_bits = torch_tensor.view(torch.uint16).contiguous().numpy()
            payload_tensor_cast_ms = (
                time.perf_counter() - cast_started
            ) * 1000.0
            logical_dtype = _torch_dtype_name(torch_tensor.dtype)
            stored_dtype = "uint16"
            stored_nbytes = int(bf16_bits.nbytes)
            save_started = time.perf_counter()
            handle.write(bf16_bits.tobytes(order="C"))
            payload_numpy_bf16_save_ms = (
                time.perf_counter() - save_started
            ) * 1000.0
    os.replace(tmp_path, final_path)
    payload_local_file_write_ms = (time.perf_counter() - started) * 1000.0

    ref = LocalFileTensorPayloadRef(
        path=str(final_path),
        shape=tuple(int(dim) for dim in pixel_values.shape),
        dtype=logical_dtype,
        nbytes=stored_nbytes,
        format=payload_format,
    )
    stored_descriptor = replace(
        descriptor,
        storage_kind=StorageKind.LOCAL_FILE,
        payload_dtype=logical_dtype,
        payload_nbytes=stored_nbytes,
    )
    stored_payload = replace(
        payload,
        pixel_values=ref,
        payload_dtype=logical_dtype,
        storage_kind=StorageKind.LOCAL_FILE,
    )
    return (
        stored_descriptor,
        stored_payload,
        {
            "payload_tensor_cast_ms": payload_tensor_cast_ms,
            "payload_torch_save_ms": payload_torch_save_ms,
            "payload_numpy_bf16_save_ms": payload_numpy_bf16_save_ms,
            "payload_local_file_write_ms": payload_local_file_write_ms,
            "payload_stored_nbytes": float(stored_nbytes),
            "payload_stored_dtype": stored_dtype,
            "payload_file_format": payload_format,
        },
    )


def _expected_raw_nbytes(ref: LocalFileTensorPayloadRef) -> int:
    dtype = np.dtype(ref.dtype)
    return _item_count(ref.shape) * int(dtype.itemsize)


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
    if ref.format == "npy":
        array = np.load(
            ref.path,
            mmap_mode="c" if local_file_payload_mmap_enabled() else None,
            allow_pickle=False,
        )
        _validate_local_file_tensor_array(ref, array)
        return array
    if ref.format == "torch":
        import torch

        tensor = torch.load(ref.path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                "torch local tensor payload must deserialize to torch.Tensor"
            )
        if tuple(int(dim) for dim in tensor.shape) != tuple(ref.shape):
            raise ValueError(
                "local tensor payload shape mismatch: "
                f"expected={ref.shape} actual={tuple(int(dim) for dim in tensor.shape)}"
            )
        actual_dtype = _torch_dtype_name(tensor.dtype)
        if actual_dtype != str(ref.dtype):
            raise ValueError(
                "local tensor payload dtype mismatch: "
                f"expected={ref.dtype} actual={actual_dtype}"
            )
        actual_nbytes = _torch_tensor_nbytes(tensor)
        if actual_nbytes != int(ref.nbytes):
            raise ValueError(
                "local tensor payload nbytes mismatch: "
                f"expected={ref.nbytes} actual={actual_nbytes}"
            )
        return tensor
    if ref.format == "numpy_bf16":
        expected_nbytes = _expected_numpy_bf16_nbytes(ref)
        if int(ref.nbytes) != expected_nbytes:
            raise ValueError(
                "numpy_bf16 local tensor payload nbytes mismatch: "
                f"expected={expected_nbytes} actual={ref.nbytes}"
            )
        actual_size = Path(ref.path).stat().st_size
        if actual_size != expected_nbytes:
            raise ValueError(
                "numpy_bf16 local tensor payload file size mismatch: "
                f"expected={expected_nbytes} actual={actual_size}"
            )
        dtype = np.dtype(np.uint16)
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
        if tuple(array.shape) != tuple(ref.shape):
            raise ValueError(
                "local tensor payload shape mismatch: "
                f"expected={ref.shape} actual={tuple(array.shape)}"
            )
        if str(array.dtype) != "uint16":
            raise ValueError(
                "local tensor payload storage dtype mismatch: "
                f"expected=uint16 actual={array.dtype}"
            )
        if int(getattr(array, "nbytes", -1)) != expected_nbytes:
            raise ValueError(
                "local tensor payload nbytes mismatch: "
                f"expected={expected_nbytes} actual={getattr(array, 'nbytes', None)}"
            )
        return array
    raise ValueError(f"unsupported local tensor payload format: {ref.format}")


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
